#!/usr/bin/env python3
"""DT3 round: paired comparison of any evaluation records against a reference, a two-fold selection check and
descriptive per-step diagnostics. Generalises `dt2_compare.py` (whose `metrics()` is reused by import),
`wscan_table.py` and `wscan_2fold.py`: the records are located by templates, so the variant may be another arm,
another evidence window, a filter record (`outputs/dt2/wscan/...`), an `--suffix`ed evalx record or a DT3 probe output.

    python tools/dt/dt3_cmp.py --runs-of dt_dz_l3 --seeds 3407 3408 3409 \\
        --ref  "w50=outputs/dt2/filter_sweep/{run}/raw.json" \\
        --var  "w150=outputs/dt2/wscan/{run}/raw_w150.json" --var "w200=outputs/dt2/wscan/{run}/raw_w200.json" \\
        --twofold w150,w200 --diag --json /tmp/cmp.json

`--ref` / `--var` are `LABEL=TEMPLATE` or `LABEL=ARM:TEMPLATE`; `{run}` is replaced by `<arm>_s<seed>` (arm = `--runs-of`
unless the spec carries its own, as in DT2's `--ref "base=dt_base:outputs/semkine/{run}/evalx_val_core_last_tf_pert.json"`).
A relative template is resolved against `--root` (default: this repository when it has `outputs/semkine`, else the live repo, so a
staging copy reads the live records). DT3 probe records (tools/dt/infer_probe.py) are
`outputs/dt3/<sub>/<run>_<core>_pw<window-ms>[_clip][_af<6hex>].json` (sub = ens-win | adabn | tta-pol | seed-ens), e.g.
`--var "ens=outputs/dt3/ens-win/{run}_ensw100-200-300_pw200.json"` or `outputs/dt3/adabn/{run}_adabn_i2_bs32_pw200.json`; the
script overwrites a record of the same name, so a record that enters a verdict needs its own `--tag`. seed-ens writes one
file named after the joined runs, which no `{run}` template pairs (give its literal path in the spec; the tool then warns that
every seed reads the same record). Everything is read-only; a seed is used for a variant only when both its record and
the reference's exist (the others are listed). Records without teacher forcing (`tf`) or amplification give NaN there.

Table: per variant the paired dRA of every seed (variant - reference, mm) and their mean, RA g / l, root rotation, ABS,
acc_err_ra, acc ratio ra, root speed ratio g / l, finger speed ratio (pooled), TF RA, amplification, the DT2 verdict and
guardrails G1-G6 (the same code lines as `dt2_compare.compare`), and the DT2 section-5 filter criteria as flags
(F1 acc_err_ra down >= 15 % against the reference, F2 acc ratio ra in [0.8, 1.2], F3 / F4 root speed ratio global / local in
[0.8, 1.2]; the 'out-of-fold RA not worse' part of section 5 is `--twofold`).

`--twofold L1,L2,...`: per seed and per direction, the variant with the lowest RA on one fold (folds = (end // 10000) % 2 of
each step of both zgz sequences, from the npz next to each json) is reported on the other fold against the reference on that
fold; the mean over the selections is the out-of-fold gain (wscan_2fold's logic). The reference is not a candidate unless
`--twofold-incl-ref` (that is wscan_2fold literally, whose W = 50 record is in its candidate list). Optimistic by construction
(the development set is the test set); the numbers are for guarding a choice, not for claiming it.

`--diag`: per sequence, from the npz (CPU only). (i) root increment response gain = least-squares slope of the predicted root
increments on the ground-truth ones; rotation increments are the so3 log of R_t R_{t-1}^T (axis-angle in state[3:6]), translation
increments state[0:3] differences, both within segments (`run`); the slope is centred per component (`gain`) and through the origin
(`gain0`, json only). (ii) lag in steps = argmax_k of the Pearson correlation between predicted increment magnitudes at step t and
ground-truth ones at step t - k, k in [-3, 3] (k > 0: the prediction lags the ground truth by k * 50 ms). (iii) Spearman rho of the
per-step RA error against the predicted innovation magnitude |root(pred_t) (-) root(pred_{t-1})| within the 50 ms event-count buckets
[0,500) / [500,2000) / [2000,inf). `speed_ratio_rot` (mean predicted / mean GT rotation-increment angle) is the quantity
`evalx.motion_drift` stores as root_speed_ratio and serves as a cross-check. Descriptive, no thresholds.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from dt2_compare import NAN, SEQS, mean, metrics as dt2_metrics                     # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LIVE = Path("/data1/lyq/code/mesh/EventHands1")
BLOCK_MS = 10_000                                   # fold = (end // BLOCK_MS) % 2 (evalx.BLOCK_MS, wscan_2fold)
KS = tuple(range(-3, 4))                            # cross-correlation lags (steps) of --diag
BUCKETS = ((0, 500), (500, 2000), (2000, 1 << 62))  # 50 ms event counts of --diag
BUCKET_NAMES = ("[0,500)", "[500,2000)", "[2000,inf)")
_ARM = re.compile(r"^([A-Za-z0-9_.\-]+):(.+)$")


# ---------------------------------------------------------------------------------------------------------------------
# specs and loading

def default_root() -> Path:
    return REPO if (REPO / "outputs" / "semkine").is_dir() else LIVE


def parse_spec(spec: str, default_arm):
    """`LABEL=TEMPLATE` or `LABEL=ARM:TEMPLATE` -> (label, arm, template). A template never has a colon before its first slash."""
    if "=" not in spec:
        raise SystemExit(f"bad spec {spec!r}: expected LABEL=TEMPLATE or LABEL=ARM:TEMPLATE")
    label, rest = spec.split("=", 1)
    m = _ARM.match(rest)
    arm, tpl = (m.group(1), m.group(2)) if m else (default_arm, rest)
    if not label or not tpl:
        raise SystemExit(f"bad spec {spec!r}")
    if arm is None and "{run}" in tpl:
        raise SystemExit(f"spec {spec!r} needs an arm: give --runs-of ARM or LABEL=ARM:TEMPLATE")
    return label, arm, tpl


def record_path(arm, tpl: str, seed: int, root: Path) -> Path:
    p = Path(tpl.replace("{run}", f"{arm}_s{seed}"))
    return p if p.is_absolute() else root / p


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def metrics_of(d: dict) -> dict:
    """`dt2_compare.metrics` on records that may lack the teacher-forced loop / amplification / a usable perturbation block
    (filter and probe records): those entries become NaN, everything else is dt2_compare's number unchanged."""
    d = dict(d)
    if not isinstance(d.get("amplification"), dict) or "mpjpe_ra_mm" not in d["amplification"]:
        d["amplification"] = {"mpjpe_ra_mm": NAN}
    tf = d.get("tf")
    if not isinstance(tf, dict) or "mpjpe_ra_mm" not in tf.get("overall", {}):
        d["tf"] = {"overall": {"mpjpe_ra_mm": NAN}}
    try:
        return dt2_metrics(d)
    except (KeyError, IndexError, TypeError, ZeroDivisionError):                     # a perturb block of another shape
        d["perturb"] = {}
        return dt2_metrics(d)


# ---------------------------------------------------------------------------------------------------------------------
# paired comparison (dt2_compare.compare with the file layout factored out)

def compare_rows(rows: list) -> dict:
    """`rows` = [(seed, metrics of the variant, metrics of the reference)] -> dt2_compare.compare's result dict (verdict, guards,
    dRA, means) plus `filter_flags`. The verdict / guardrail lines are dt2_compare's, line for line."""
    if not rows:
        return {"seeds": [], "status": "no results"}
    used = [r[0] for r in rows]
    d = {k: [r[1][k] - r[2][k] for r in rows] for k in rows[0][1]}
    A = {k: mean([r[1][k] for r in rows]) for k in rows[0][1]}
    B = {k: mean([r[2][k] for r in rows]) for k in rows[0][1]}
    dra = d["RA"]
    m = mean(dra)
    if m <= -1.1 and all(x < 0 for x in dra):
        verdict = "BETTER" if (len(dra) >= 3 or m <= -2.5) else "better-pending-3409"
    elif m >= 1.1:
        verdict = "WORSE"
    else:
        verdict = "tie-candidate" if (all(x < 0 for x in dra) and m <= -0.5) else "tie"
    rel = lambda k: (A[k] - B[k]) / B[k]                                             # noqa: E731
    guards = []
    if mean(d["ABS"]) > 2.0: guards.append(f"G1 ABS {mean(d['ABS']):+.2f}")
    if mean(d["RA_l"]) > 1.0: guards.append(f"G2 RA_local {mean(d['RA_l']):+.2f}")
    if rel("jit") > 0.05: guards.append(f"G3a jit {100*rel('jit'):+.1f}%")
    if rel("acc") > 0.05: guards.append(f"G3b acc_err {100*rel('acc'):+.1f}%")
    if rel("acc_ra") == rel("acc_ra") and rel("acc_ra") > 0.05: guards.append(f"G3c acc_err_ra {100*rel('acc_ra'):+.1f}%")
    if A["rsr_g"] < 0.9: guards.append(f"G4a root speed g {A['rsr_g']:.2f}")
    if mean(d["fsr"]) < -0.05: guards.append(f"G4b finger speed {mean(d['fsr']):+.3f}")
    if sum(d["fail"]) > 0: guards.append(f"G5 failures +{sum(d['fail']):.0f}")
    if mean(d["amp"]) > 0.05: guards.append(f"G6 amp {mean(d['amp']):+.3f}")
    return {"seeds": used, "verdict": verdict, "guards": guards, "dRA": dra, "dRA_mean": m,
            "arm_mean": A, "base_mean": B, "delta_mean": {k: mean(v) for k, v in d.items()},
            "filter_flags": filter_flags(A, B)}


def filter_flags(A: dict, B: dict) -> dict:
    """DT2 section 5 criteria on the seed means (variant A against reference B); ok is None when a number is NaN."""
    def flag(value, ok_fn):
        return {"value": value, "ok": None if value != value else bool(ok_fn(value))}
    drop = -(A["acc_ra"] - B["acc_ra"]) / B["acc_ra"] if B["acc_ra"] else NAN
    return {"F1_acc_err_ra_drop": flag(drop, lambda v: v >= 0.15),
            "F2_acc_ratio_ra": flag(A["accr_ra"], lambda v: 0.8 <= v <= 1.2),
            "F3_root_speed_g": flag(A["rsr_g"], lambda v: 0.8 <= v <= 1.2),
            "F4_root_speed_l": flag(A["rsr_l"], lambda v: 0.8 <= v <= 1.2)}


def flags_text(ff: dict) -> str:
    def one(name, f, fmt):
        return f"{name} n/a" if f["ok"] is None else f"{name} {fmt(f['value'])} {'ok' if f['ok'] else 'FAIL'}"
    return "; ".join([one("F1 acc_err_ra", ff["F1_acc_err_ra_drop"], lambda v: f"{-100 * v:+.1f}%"),
                      one("F2", ff["F2_acc_ratio_ra"], lambda v: f"{v:.2f}"),
                      one("F3", ff["F3_root_speed_g"], lambda v: f"{v:.2f}"),
                      one("F4", ff["F4_root_speed_l"], lambda v: f"{v:.2f}")])


def collect(spec, seeds, root: Path) -> dict:
    """{seed: (path, json)} of the records of one spec that exist (None json when missing)."""
    label, arm, tpl = spec
    out = {}
    for s in seeds:
        p = record_path(arm, tpl, s, root)
        out[s] = (p, load_json(p))
    return out


def paired(ref_recs: dict, var_recs: dict, seeds) -> tuple:
    """(rows, missing, warnings): a seed is paired when the variant's and the reference's records both exist and carry the
    numbers `dt2_compare.metrics` needs (older records may lack `jitter`; those seeds are skipped with a warning)."""
    rows, missing, warns = [], [], []
    for s in seeds:
        a, b = var_recs[s][1], ref_recs[s][1]
        if a is None or b is None:
            missing.append(s)
            continue
        try:
            rows.append((s, metrics_of(a), metrics_of(b)))
        except (KeyError, TypeError) as e:
            missing.append(s)
            warns.append(f"seed {s}: {var_recs[s][0]} or {ref_recs[s][0]} lacks {e!r}")
    return rows, missing, warns


def table(ref_label, ref_recs, results: list) -> list:
    L = ["| variant | seeds | dRA per seed | dRA mean | RA (var / ref) | RA g / l (var) | dRA g / l | rot deg | ABS | acc_err_ra "
         "| acc ratio ra | root speed g/l | finger speed | TF RA | amp | DT2 verdict | guardrails G1-G6 | DT2 s5 filter flags |",
         "|" + "---|" * 18]
    rm = [metrics_of(v[1]) for v in ref_recs.values() if v[1] is not None]
    if rm:
        R = {k: mean([x[k] for x in rm]) for k in rm[0]}
        rs = ",".join(str(s) for s, v in ref_recs.items() if v[1] is not None)
        L.append(f"| **{ref_label}** (ref) | {rs} | - | - | {R['RA']:.2f} | {R['RA_g']:.2f} / {R['RA_l']:.2f} | - | "
                 f"{R['rot']:.2f} | {R['ABS']:.1f} | {R['acc_ra']:.2f} | {R['accr_ra']:.2f} | {R['rsr_g']:.2f} / {R['rsr_l']:.2f} | "
                 f"{R['fsr']:.3f} | {R['TF']:.2f} | {R['amp']:.3f} | - | - | - |")
    for r in results:
        if not r["seeds"]:
            L.append(f"| {r['label']} | - | | | | | | | | | | | | | | no results | | |")
            continue
        A, B, D = r["arm_mean"], r["base_mean"], r["delta_mean"]
        L.append(f"| {r['label']} | {','.join(map(str, r['seeds']))} | {' / '.join(f'{x:+.2f}' for x in r['dRA'])} | "
                 f"**{r['dRA_mean']:+.2f}** | {A['RA']:.2f} / {B['RA']:.2f} | {A['RA_g']:.2f} / {A['RA_l']:.2f} | "
                 f"{D['RA_g']:+.2f} / {D['RA_l']:+.2f} | {A['rot']:.2f} ({D['rot']:+.2f}) | {A['ABS']:.1f} ({D['ABS']:+.1f}) | "
                 f"{A['acc_ra']:.2f} ({D['acc_ra']:+.2f}) | {A['accr_ra']:.2f} | {A['rsr_g']:.2f} / {A['rsr_l']:.2f} | "
                 f"{A['fsr']:.3f} ({D['fsr']:+.3f}) | {A['TF']:.2f} | {A['amp']:.3f} | {r['verdict']} | "
                 f"{'; '.join(r['guards']) or '-'} | {flags_text(r['filter_flags'])} |")
    return L


# ---------------------------------------------------------------------------------------------------------------------
# two-fold selection

def fold_ra(z, seqs=SEQS):
    """Mean per-step RA of fold 0 / 1 (fold = (end // 10000) % 2) over the steps of all `seqs` (wscan_2fold.fold_ra)."""
    out = {0: [], 1: []}
    ends = []
    for s in seqs:
        e, ra = z[f"model|{s}|end"], z[f"model|{s}|mpjpe_ra_mm"]
        f = (e // BLOCK_MS) % 2
        ends.append(e)
        for k in (0, 1):
            out[k].append(ra[f == k])
    return {k: float(np.concatenate(v).mean()) for k, v in out.items()}, np.concatenate(ends)


def npz_of(json_path: Path) -> Path:
    return json_path.with_suffix(".npz")


def twofold(npz_paths: dict, ref: str, cands: list, seeds, include_ref: bool = False) -> dict:
    """`npz_paths[label][seed]` -> Path. For each seed (all candidates + reference present) and each direction: pick the
    candidate with the lowest fold RA on the selection fold; gain = its RA on the other fold - the reference's RA there."""
    pool = ([ref] if include_ref else []) + list(cands)
    sels, skipped, warns = [], [], []
    for seed in seeds:
        paths = {l: npz_paths[l].get(seed) for l in [ref] + list(cands)}
        miss = [l for l, p in paths.items() if p is None or not p.exists()]
        if miss:
            skipped.append({"seed": seed, "missing": miss})
            continue
        fr, ends = {}, {}
        for l, p in paths.items():
            with np.load(p) as z:
                fr[l], ends[l] = fold_ra(z)
        bad = [l for l in cands if not np.array_equal(ends[l], ends[ref])]
        if bad:
            warns.append(f"seed {seed}: steps of {bad} differ from the reference's (folds are not paired)")
        for sel, rep in ((0, 1), (1, 0)):
            pick = min(pool, key=lambda l: fr[l][sel])
            sels.append({"seed": seed, "sel_fold": sel, "rep_fold": rep, "pick": pick, "ra_pick": fr[pick][rep],
                         "ra_ref": fr[ref][rep], "gain": fr[pick][rep] - fr[ref][rep],
                         "fold_ra": {l: fr[l] for l in fr}})
    gains = [s["gain"] for s in sels]
    return {"ref": ref, "candidates": list(cands), "include_ref": include_ref, "selections": sels, "skipped": skipped,
            "warnings": warns, "mean_gain": float(np.mean(gains)) if gains else NAN, "n_sel": len(gains),
            "n_neg": int(sum(g < 0 for g in gains)), "all_neg": bool(gains) and all(g < 0 for g in gains),
            "picks": [s["pick"] for s in sels]}


def twofold_text(tf: dict) -> list:
    L = []
    for s in tf["skipped"]:
        L.append(f"  s{s['seed']}: skipped, missing {s['missing']}")
    for w in tf["warnings"]:
        L.append(f"  WARNING {w}")
    if tf["n_sel"]:
        incl = " (reference is a candidate)" if tf["include_ref"] else ""
        per = ", ".join("%+.2f" % s["gain"] for s in tf["selections"])
        L.append(f"{','.join(tf['candidates'])}{incl}: out-of-fold dRA vs {tf['ref']}: {tf['mean_gain']:+.3f} mm "
                 f"(per selection: {per}); picked: {tf['picks']}; {tf['n_neg']}/{tf['n_sel']} selections negative")
        L += ["", "| seed | selected on fold | pick | RA of the pick on the other fold | RA of ref there | gain |", "|---|---|---|---|---|---|"]
        for s in tf["selections"]:
            L.append(f"| {s['seed']} | {s['sel_fold']} | {s['pick']} | {s['ra_pick']:.3f} | {s['ra_ref']:.3f} | {s['gain']:+.3f} |")
    return L


# ---------------------------------------------------------------------------------------------------------------------
# diagnostics

def _aa_to_R(aa: np.ndarray) -> np.ndarray:
    """(N, 3) axis-angle -> (N, 3, 3) (Rodrigues, as evalx.aa_to_R)."""
    th = np.linalg.norm(aa, axis=-1, keepdims=True).clip(1e-9)
    k = aa / th
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def _so3_log(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) -> (N, 3) rotation vector. theta = atan2(|v|, (tr-1)/2) with v = vee(R - R^T)/2 is accurate for small angles
    (increments of consecutive 50 ms steps); near pi it loses the axis, which consecutive steps never reach."""
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1) / 2
    nv = np.linalg.norm(v, axis=-1)
    th = np.arctan2(nv, (np.trace(R, axis1=-2, axis2=-1) - 1) / 2)
    f = np.where(nv < 1e-9, 1.0, th / np.maximum(nv, 1e-300))
    return v * f[:, None]


def root_increments(pred51: np.ndarray, gt51: np.ndarray, run: np.ndarray) -> dict:
    """Per step i = 1..N-1 (index i-1 of the outputs): rotation increment so3log(R_i R_{i-1}^T) (rad) and translation increment
    (m) of the prediction (p_*) and the ground truth (g_*); `valid` = step i and i-1 are in the same segment."""
    out = {"valid": run[1:] == run[:-1], "run": run[1:]}
    for n, a in (("p", pred51), ("g", gt51)):
        a = a.astype(np.float64)
        R = _aa_to_R(a[:, 3:6])
        out[n + "_rot"] = _so3_log(R[1:] @ np.swapaxes(R[:-1], -1, -2))
        out[n + "_tr"] = a[1:, 0:3] - a[:-1, 0:3]
    return out


def ls_slope(x: np.ndarray, y: np.ndarray, center: bool = True) -> float:
    """Least-squares slope of y on x pooled over the columns (n, c); centred per column, or through the origin."""
    if len(x) == 0:
        return NAN
    if center:
        x, y = x - x.mean(0), y - y.mean(0)
    den = float((x * x).sum())
    return float((x * y).sum() / den) if den > 0 else NAN


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return NAN
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else NAN


def xcorr_lag(mp, mg, valid, run, ks=KS) -> tuple:
    """Pearson correlation between mp[i] (prediction) and mg[i - k] (ground truth) over the pairs inside one segment, for each k;
    returns (lag, {k: corr}). lag = argmax over k (ties: smaller |k|). k > 0: the prediction lags the ground truth by k steps."""
    n, curve = len(mp), {}
    for k in ks:
        i = np.arange(max(0, k), min(n, n + k))
        j = i - k
        ok = valid[i] & valid[j] & (run[i] == run[j])
        curve[k] = _pearson(mp[i][ok], mg[j][ok])
    cand = [(k, c) for k, c in curve.items() if c == c]
    lag = min(cand, key=lambda kc: (-kc[1], abs(kc[0])))[0] if cand else None
    return lag, curve


def _rank(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n with the average rank for ties."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(_rank(a), _rank(b)) if len(a) >= 3 else NAN


def diag_seq(z, seq: str) -> dict:
    """The three diagnostics of one sequence from the npz arrays `model|<seq>|{pred,gt,run,count,mpjpe_ra_mm}`."""
    inc = root_increments(z[f"model|{seq}|pred"], z[f"model|{seq}|gt"], z[f"model|{seq}|run"])
    v, run = inc["valid"], inc["run"]
    err = z[f"model|{seq}|mpjpe_ra_mm"][1:].astype(np.float64)
    cnt = z[f"model|{seq}|count"][1:]
    res = {"n_steps": int(len(v) + 1), "n_inc": int(v.sum())}
    for key in ("rot", "tr"):
        gx, px = inc["g_" + key][v], inc["p_" + key][v]
        mp, mg = np.linalg.norm(inc["p_" + key], axis=-1), np.linalg.norm(inc["g_" + key], axis=-1)
        lag, curve = xcorr_lag(mp, mg, v, run)
        rho = {}
        for nm, (lo, hi) in zip(BUCKET_NAMES, BUCKETS):
            m = v & (cnt >= lo) & (cnt < hi)
            rho[nm] = {"n": int(m.sum()), "rho": spearman(err[m], mp[m])}
        res.update({f"gain_{key}": ls_slope(gx, px), f"gain0_{key}": ls_slope(gx, px, center=False),
                    f"speed_ratio_{key}": float(mp[v].mean() / max(mg[v].mean(), 1e-12)) if v.any() else NAN,
                    f"lag_{key}": lag, f"xcorr_{key}": {str(k): c for k, c in curve.items()}, f"rho_{key}": rho})
    return res


def diag_label(npz_paths: dict, seeds) -> dict:
    """{seed: {seq: diag_seq}} for the seeds whose npz exists."""
    out = {}
    for s in seeds:
        p = npz_paths.get(s)
        if p is None or not p.exists():
            continue
        with np.load(p) as z:
            out[s] = {q: diag_seq(z, q) for q in SEQS if f"model|{q}|pred" in z.files}
    return out


def _nm(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else NAN


def diag_text(diag: dict) -> list:
    L = ["| variant | seeds | sequence | gain rot | gain transl | lag rot (steps, per seed) | lag transl | speed ratio rot | "
         + " | ".join(f"rho rot {nm}" for nm in BUCKET_NAMES) + " | n " + " / ".join(BUCKET_NAMES) + " |", "|" + "---|" * 12]
    for label, per_seed in diag.items():
        seeds = sorted(per_seed)
        if not seeds:
            continue
        for q in SEQS:
            rs = [per_seed[s][q] for s in seeds if q in per_seed[s]]
            if not rs:
                continue
            rho = [_nm([r["rho_rot"][nm]["rho"] for r in rs]) for nm in BUCKET_NAMES]
            n = [int(np.mean([r["rho_rot"][nm]["n"] for r in rs])) for nm in BUCKET_NAMES]
            L.append(f"| {label} | {','.join(map(str, seeds))} | {q} | {_nm([r['gain_rot'] for r in rs]):.3f} | "
                     f"{_nm([r['gain_tr'] for r in rs]):.3f} | {'/'.join(str(r['lag_rot']) for r in rs)} | "
                     f"{'/'.join(str(r['lag_tr']) for r in rs)} | {_nm([r['speed_ratio_rot'] for r in rs]):.3f} | "
                     + " | ".join(f"{x:+.3f}" for x in rho) + f" | {' / '.join(map(str, n))} |")
    return L


# ---------------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True, help='"LABEL=TEMPLATE" or "LABEL=ARM:TEMPLATE"; {run} -> ARM_s<seed>')
    ap.add_argument("--var", action="append", required=True, help="same form; repeatable")
    ap.add_argument("--runs-of", default=None, help="default ARM for specs without their own")
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408, 3409])
    ap.add_argument("--root", default=None, help="base of relative templates (default: this repo if it has outputs/, else the live repo)")
    ap.add_argument("--twofold", action="append", default=[], help="LABEL1,LABEL2,...: two-fold selection among these variants (repeatable)")
    ap.add_argument("--twofold-incl-ref", action="store_true", help="the reference is also a candidate (wscan_2fold literally)")
    ap.add_argument("--diag", action="store_true", help="descriptive diagnostics from the npz of the reference and every variant")
    ap.add_argument("--json", default=None, help="dump everything here")
    a = ap.parse_args()
    root = Path(a.root) if a.root else default_root()
    ref_spec = parse_spec(a.ref, a.runs_of)
    var_specs = [parse_spec(v, a.runs_of) for v in a.var]
    for sp in [ref_spec] + var_specs:
        if "{run}" not in sp[2]:
            print(f"warning: template of {sp[0]!r} has no {{run}}: every seed reads the same record", file=sys.stderr)
    labels = [ref_spec[0]] + [v[0] for v in var_specs]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"labels must be unique, got {labels}")
    ref_recs = collect(ref_spec, a.seeds, root)
    results, var_recs = [], {}
    for spec in var_specs:
        recs = collect(spec, a.seeds, root)
        var_recs[spec[0]] = recs
        rows, missing, warns = paired(ref_recs, recs, a.seeds)
        r = compare_rows(rows)
        r.update({"label": spec[0], "arm": spec[1], "template": spec[2], "missing_seeds": missing, "warnings": warns})
        results.append(r)
    print(f"reference {ref_spec[0]} = {ref_spec[2]} (arm {ref_spec[1]}); root {root}; paired on available seeds; dRA = variant - reference\n")
    print("\n".join(table(ref_spec[0], ref_recs, results)))
    for r in results:
        if r["missing_seeds"]:
            print(f"\n{r['label']}: seeds {r['missing_seeds']} skipped (record missing or unusable for the variant or the reference)")
        for w in r["warnings"]:
            print(f"  WARNING {w}")
    out = {"ref": {"label": ref_spec[0], "arm": ref_spec[1], "template": ref_spec[2]}, "root": str(root),
           "seeds": a.seeds, "variants": results, "twofold": [], "diag": {}}
    all_specs = {s[0]: s for s in [ref_spec] + var_specs}
    all_recs = {ref_spec[0]: ref_recs, **var_recs}
    for tf_arg in a.twofold:
        cands = [x.strip() for x in tf_arg.split(",") if x.strip()]
        for c in cands:
            if c not in var_recs:
                raise SystemExit(f"--twofold: unknown variant {c!r} (have {list(var_recs)})")
        paths = {l: {s: npz_of(all_recs[l][s][0]) for s in a.seeds} for l in [ref_spec[0]] + cands}
        tf = twofold(paths, ref_spec[0], cands, a.seeds, a.twofold_incl_ref)
        out["twofold"].append(tf)
        print("\ntwo-fold selection (folds = (end // 10000) % 2 of both zgz sequences; optimistic, development set = test set)")
        print("\n".join(twofold_text(tf)))
    if a.diag:
        for l in labels:
            out["diag"][l] = diag_label({s: npz_of(all_recs[l][s][0]) for s in a.seeds}, a.seeds)
        print("\ndescriptive diagnostics (from the npz; descriptive only, no thresholds)\n")
        print("\n".join(diag_text(out["diag"])))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
