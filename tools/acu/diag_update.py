#!/usr/bin/env python3
"""ACU D1 (docs/EXPERIMENT_RECORDS.md#acu-diag-20261104): per-step update decomposition from existing PFO-engine
readings. Zero training, CPU, read-only on the readings.

    python -m tools.acu.diag_update --out outputs/acu/20261104/diag/d1

Spaces (one per DOF group):
  T  translation, mm (51-D [0:3])
  R  root rotation increments, deg: log(R_a R_b^T) as a rotation vector
  F  hand-frame joints, mm: FK with translation and root rotation zeroed (articulation only), 21 OpenPose joints
For every network reading and each zgz recording:
  e3    update D = f(prev) - prev against the true motion m = target - prev (prev = GT previous state). LS gain
        G = <D, m> / |m|^2 (response to real motion), injection n = D - G m, and the E3 split |(G - 1) m|^2 + |n|^2.
        Low-motion quartile (smallest |m| in the group's space): RMS |D|, i.e. change with no motion to follow.
  pert  retention in the same spaces, rho_J = <g(f(x+d)) - g(f(x)), g(x+d) - g(x)> / |g(x+d) - g(x)|^2, and the
        cross-talk: RMS change of the other groups caused by the perturbation.
  e1    own closed loop: D = -a e_prev + G m + n by two-regressor LS per group (a = correction gain on the loop's own
        errors, G = motion gain), shares of the three terms, fraction of steps whose error shrinks, and the lag
        (pooled cross-correlation of D_k with m_(k+s), s = -4..4, inside runs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

ZGZ = ("zgz_global", "zgz_local")
READ_ROOTS = {"pfo": ROOT / "outputs/pfo/20261103/readings", "pfo2": ROOT / "outputs/pfo2/20261103/readings",
              "b2ub": ROOT / "outputs/b2ub/20261104/readings"}
NETS = {
    "B2 3407": ("pfo2", "pfo2_fgs_s3407"), "B2 3408": ("pfo2", "pfo2_fgs_s3408"), "B2 3409": ("pfo2", "pfo2_fgs_s3409"),
    "B2 3407 +label test": ("b2ub", "b2_s3407__label"), "B2 3408 +label test": ("b2ub", "b2_s3408__label"),
    "B2-gt label": ("b2ub", "b2gt_s3407__label"), "B2-gt r_s2d": ("b2ub", "b2gt_s3407"),
    "S43 3407": ("pfo", "s43_s3407"), "S43 3408": ("pfo", "s43_s3408"), "S43 3409": ("pfo", "s43_s3409"),
}
PERT = ("T10", "T30", "R5", "R15", "F05", "F15", "N")
GROUPS = ("T", "R", "F")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SHIFTS = tuple(range(-4, 5))
_CTX = {}


def context():
    if not _CTX:
        import torch
        import yaml
        from mano_layer import ManoLayer
        from tools.pfo.engine import sequences
        cfg = yaml.safe_load((ROOT / "configs/pfo2_20261103/eval/pfo2_fgs_s3407.yaml").read_text())
        root, seqs = sequences(cfg, ZGZ)
        _CTX["mano"] = ManoLayer(str(ROOT / "assets/mano_right.npz"), add_mean=False).eval()
        _CTX["betas"] = {}
        for s, d in seqs:
            aux = np.load(Path(root) / d / f"{s}_aux.npz", allow_pickle=True)
            _CTX["betas"][s] = torch.tensor(np.asarray(aux["betas"], np.float32)).view(1, -1)
    return _CTX


def hand_joints(x, seq):
    """(N, 21, 3) mm: articulation only (translation and root rotation zeroed)."""
    import torch
    from semkine import eval_track as ET
    c = context()
    p = np.array(x, np.float32, copy=True)
    p[:, 0:6] = 0.0
    j, _ = ET.mano_fk(c["mano"], p, c["betas"][seq], torch.device("cpu"))
    return j.astype(np.float64) * 1000.0


def rot_inc(a, b):
    """(N, 3) deg: log(R_a R_b^T)."""
    ra = Rotation.from_rotvec(np.asarray(a, np.float64)[:, 3:6])
    rb = Rotation.from_rotvec(np.asarray(b, np.float64)[:, 3:6])
    return np.rad2deg((ra * rb.inv()).as_rotvec())


class Spaces:
    """Cached per-state coordinates of one recording: T (mm), rotation objects, hand-frame joints (mm)."""

    def __init__(self, seq):
        self.seq = seq

    def T(self, x):
        return np.asarray(x, np.float64)[:, 0:3] * 1000.0

    def diff(self, group, a, b, ja=None, jb=None):
        if group == "T":
            return self.T(a) - self.T(b)
        if group == "R":
            return rot_inc(a, b)
        return (ja if ja is not None else hand_joints(a, self.seq)) - (jb if jb is not None else hand_joints(b, self.seq))


def flat(v):
    return v.reshape(len(v), -1)


def unit(v):
    """Per-sample norm divisor: per-joint RMS for (N, 21, 3) joint fields, the vector norm otherwise."""
    return float(v.shape[1]) if v.ndim == 3 else 1.0


def ls_gain(d, m):
    d2, m2, u = flat(d), flat(m), unit(d)
    den = float((m2 * m2).sum())
    g = float((d2 * m2).sum()) / max(den, 1e-12)
    n = d2 - g * m2
    per = lambda a: float(np.sqrt((a * a).sum(1).mean() / u))  # noqa: E731
    return {"G": g, "rms_D": per(d2), "rms_m": per(m2), "rms_injection": per(n), "rms_lag": per((g - 1.0) * m2)}


def low_motion(d, m, q=0.25):
    u = unit(d)
    s = np.sqrt((flat(m) ** 2).sum(1))
    sel = s <= np.quantile(s, q)
    return {"quantile": q, "rms_m": float(np.sqrt((flat(m)[sel] ** 2).sum(1).mean() / u)),
            "rms_D": float(np.sqrt((flat(d)[sel] ** 2).sum(1).mean() / u))}


def per_finger_gain(d, m):
    out = {}
    for f, name in enumerate(FINGERS):
        idx = list(range(1 + 4 * f, 5 + 4 * f))
        out[name] = ls_gain(d[:, idx], m[:, idx])["G"]
    return out


def states(root, tag, task):
    p = READ_ROOTS[root] / tag / task / "states.npz"
    return np.load(p, allow_pickle=False) if p.exists() else None


def e3_block(z, seq):
    sp = Spaces(seq)
    prev, pred, tgt = z[f"{seq}/gt_previous"], z[f"{seq}/prediction"], z[f"{seq}/target"]
    jp, jq, jt = hand_joints(prev, seq), hand_joints(pred, seq), hand_joints(tgt, seq)
    out = {}
    for g in GROUPS:
        D = sp.diff(g, pred, prev, jq, jp)
        M = sp.diff(g, tgt, prev, jt, jp)
        out[g] = {**ls_gain(D, M), "low_motion": low_motion(D, M)}
        if g == "F":
            out[g]["per_finger_G"] = per_finger_gain(D, M)
    return out


def pert_block(z, seq):
    sp = Spaces(seq)
    prev, base = z[f"{seq}/gt_previous"], z[f"{seq}/prediction"]
    jp, jb = hand_joints(prev, seq), hand_joints(base, seq)
    out = {}
    for kind in PERT:
        y = z[f"{seq}/{kind}/prediction"]
        x = prev.astype(np.float64) + z[f"{seq}/{kind}/delta"]
        jy, jx = hand_joints(y, seq), hand_joints(x, seq)
        blk = {}
        for g in GROUPS:
            rx = sp.diff(g, x, prev, jx, jp)
            u = unit(rx)
            dx = flat(rx)
            dy = flat(sp.diff(g, y, base, jy, jb))
            den = (dx * dx).sum(1)
            moved = den > 1e-9
            if moved.mean() > 0.5:
                rho = (dy * dx).sum(1)[moved] / den[moved]
                blk[g] = {"rho_mean": float(rho.mean()), "rho_median": float(np.median(rho)),
                          "pooled": float((dy * dx).sum() / max(den.sum(), 1e-12)),
                          "rms_in": float(np.sqrt(den.mean() / u)), "rms_out": float(np.sqrt((dy * dy).sum(1).mean() / u))}
            else:
                blk[g] = {"crosstalk_rms": float(np.sqrt((dy * dy).sum(1).mean() / u))}
        out[kind] = blk
    return out


def e1_block(z, seq):
    sp = Spaces(seq)
    inc, pred, tgt, gtp = z[f"{seq}/incoming"], z[f"{seq}/prediction"], z[f"{seq}/target"], z[f"{seq}/gt_previous"]
    run = z[f"{seq}/run_id"]
    el = z[f"{seq}/elapsed_ms"]
    ji, jq, jt, jg = hand_joints(inc, seq), hand_joints(pred, seq), hand_joints(tgt, seq), hand_joints(gtp, seq)
    out = {}
    for g in GROUPS:
        raw = sp.diff(g, pred, inc, jq, ji)
        u = unit(raw)
        D = flat(raw)
        E = flat(sp.diff(g, inc, gtp, ji, jg))
        M = flat(sp.diff(g, tgt, gtp, jt, jg))
        P = flat(sp.diff(g, pred, tgt, jq, jt))

        def fit(sel):
            X = np.stack([-E[sel].reshape(-1), M[sel].reshape(-1)], 1)
            c, *_ = np.linalg.lstsq(X, D[sel].reshape(-1), rcond=None)
            return float(c[0]), float(c[1])
        a, G = fit(np.ones(len(D), bool))
        early, late = el < 150, el >= 1000
        a_early, G_early = fit(early) if early.sum() > 10 else (float("nan"), float("nan"))
        a_late, G_late = fit(late) if late.sum() > 10 else (float("nan"), float("nan"))
        n = D - (-a * E + G * M)
        rms = lambda v: float(np.sqrt((v * v).sum(1).mean() / u))  # noqa: E731
        ne, npst = np.sqrt((E * E).sum(1)), np.sqrt((P * P).sum(1))
        lag = {}
        for s in SHIFTS:
            num = den_a = den_b = 0.0
            for r in np.unique(run):
                idx = np.nonzero(run == r)[0]
                if len(idx) <= abs(s) + 1:
                    continue
                if s >= 0:
                    a_, b_ = D[idx[:len(idx) - s]], M[idx[s:]]
                else:
                    a_, b_ = D[idx[-s:]], M[idx[:len(idx) + s]]
                num += float((a_ * b_).sum())
                den_a += float((a_ * a_).sum())
                den_b += float((b_ * b_).sum())
            lag[str(s)] = num / max(np.sqrt(den_a * den_b), 1e-12)
        peak = max(lag, key=lambda k: lag[k])
        out[g] = {"a": a, "G": G, "a_early": a_early, "G_early": G_early, "a_late": a_late, "G_late": G_late,
                  "rms_prev_err": rms(E), "rms_post_err": rms(P), "rms_D": rms(D),
                  "rms_correction_term": rms(-a * E), "rms_motion_term": rms(G * M), "rms_injection": rms(n),
                  "shrink_fraction": float((npst < ne).mean()), "lag_corr": lag, "lag_peak_shift": int(peak)}
    return out


def f1_block(root, tag):
    p = READ_ROOTS[root] / tag / "f1" / "report.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())["per_sequence"]
    return {seq: {k: r[seq][k] for k in ("gt_iter1", "gt_iter20", "mean_iter20", "loop_iter20", "mean_start") if k in r[seq]}
            for seq in ZGZ}


def run(out):
    res = {}
    for name, (root, tag) in NETS.items():
        res[name] = {}
        f1 = f1_block(root, tag)
        if f1:
            res[name]["f1"] = f1
        for task, fn in (("e3", e3_block), ("pert", pert_block), ("e1", e1_block)):
            z = states(root, tag, task)
            if z is None:
                continue
            res[name][task] = {seq: fn(z, seq) for seq in ZGZ}
        print(name, "done", sorted(res[name]), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    out.with_suffix(".md").write_text(table(res))
    return res


def table(res):
    L = ["# ACU D1: per-step update decomposition (zero training, existing readings)", ""]
    for seq in ZGZ:
        L += [f"## {seq}", "", "E3 (GT previous state): motion gain G / injection RMS / lag RMS / low-motion RMS |D| (truth RMS)", "",
              "| net | T: G / inj / lag mm | R: G / inj / lag deg | F: G / inj / lag mm | low-motion R deg | low-motion F mm | F per-finger G (T I M R P) |",
              "|---|---|---|---|---|---|---|"]
        for name, r in res.items():
            if "e3" not in r:
                continue
            b = r["e3"][seq]
            f = lambda g: f"{b[g]['G']:.2f} / {b[g]['rms_injection']:.2f} / {b[g]['rms_lag']:.2f}"  # noqa: E731
            lm = lambda g: f"{b[g]['low_motion']['rms_D']:.2f} ({b[g]['low_motion']['rms_m']:.2f})"  # noqa: E731
            pf = " ".join(f"{b['F']['per_finger_G'][k]:.2f}" for k in FINGERS)
            L.append(f"| {name} | {f('T')} | {f('R')} | {f('F')} | {lm('R')} | {lm('F')} | {pf} |")
        L += ["", "pert retention (pooled projection; T on T kinds, R on R kinds, F on F kinds; N = all three)", "",
              "| net | T10 | T30 | R5 | R15 | F05 | F15 | N: T / R / F | F15 cross-talk R deg / T mm |", "|---|---|---|---|---|---|---|---|---|"]
        for name, r in res.items():
            if "pert" not in r:
                continue
            b = r["pert"][seq]
            g = {"T10": "T", "T30": "T", "R5": "R", "R15": "R", "F05": "F", "F15": "F"}
            cells = [f"{b[k][g[k]]['pooled']:.2f}" for k in ("T10", "T30", "R5", "R15", "F05", "F15")]
            n = " / ".join(f"{b['N'][x]['pooled']:.2f}" for x in GROUPS)
            ct = f"{b['F15']['R'].get('crosstalk_rms', float('nan')):.2f} / {b['F15']['T'].get('crosstalk_rms', float('nan')):.2f}"
            L.append(f"| {name} | " + " | ".join(cells) + f" | {n} | {ct} |")
        L += ["", "E1 (own loop): D = -a e_prev + G m + n; RMS prev err -> post err; shrink fraction; lag peak shift (corr at 0 / peak);"
              " a early (< 150 ms into a segment) / late (>= 1 s)", "",
              "| net | R: a / G / inj / prev->post / shrink | F: a / G / inj / prev->post / shrink | R lag | F lag | R a early/late | F a early/late |",
              "|---|---|---|---|---|---|---|"]
        for name, r in res.items():
            if "e1" not in r:
                continue
            b = r["e1"][seq]
            f = lambda g: (f"{b[g]['a']:.2f} / {b[g]['G']:.2f} / {b[g]['rms_injection']:.2f} / "  # noqa: E731
                           f"{b[g]['rms_prev_err']:.2f}->{b[g]['rms_post_err']:.2f} / {b[g]['shrink_fraction']:.2f}")
            lg = lambda g: f"{b[g]['lag_peak_shift']} ({b[g]['lag_corr']['0']:.2f} / {b[g]['lag_corr'][str(b[g]['lag_peak_shift'])]:.2f})"  # noqa: E731
            el = lambda g: f"{b[g]['a_early']:.2f} / {b[g]['a_late']:.2f}"  # noqa: E731
            L.append(f"| {name} | {f('R')} | {f('F')} | {lg('R')} | {lg('F')} | {el('R')} | {el('F')} |")
        L += ["", "f1 (same packet fed 20 times; RA mm): from GT iter 1 / iter 20; mean-pose start -> iter 20; own-loop start iter 20", "",
              "| net | GT 1 | GT 20 | mean start | mean 20 | loop 20 |", "|---|---|---|---|---|---|"]
        for name, r in res.items():
            if "f1" not in r:
                continue
            b = r["f1"][seq]
            L.append(f"| {name} | {b.get('gt_iter1', float('nan')):.2f} | {b.get('gt_iter20', float('nan')):.2f} | "
                     f"{b.get('mean_start', float('nan')):.2f} | {b.get('mean_iter20', float('nan')):.2f} | {b.get('loop_iter20', float('nan')):.2f} |")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "outputs/acu/20261104/diag/d1"))
    a = p.parse_args(argv)
    run(Path(a.out))


if __name__ == "__main__":
    main()
