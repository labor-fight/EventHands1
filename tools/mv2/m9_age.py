#!/usr/bin/env python3
"""MV2 M9 / PAS + CAF (P5-E5-R2, P5-E6-R1): how the frozen S39 depends on the AGE of its two contexts, E3 single step.

SELECT only (ycy, ylf; base recordings), frozen S39 x 3 seeds (step 6000), CPU. E3 = the bare network `model(X, prev)` with
the W200 event image and prev = GT(end - 50 - a_p) (a_p = 0 is the plain E3 of `carry_probe` / `common.SeqSamples`), error =
root-aligned MPJPE (mm) to GT(end), pooled over a domain's samples.

  arm `aged`     prev = GT(end - 50 - a_p), a_p in {0, 10, 25, 50, 100, 200} ms
  arm `extrap`   the same stale GT + an oracle constant-velocity extrapolation by a_p (translation linear, root geodesic
                 left-composed, fingers linear in axis-angle; velocity from GT over the last 17 ms before the stale time):
                 the state AT end - 50, predicted. Upper bound, not deployable.
  arm `extrap_q` the same extrapolated all the way to the query time (a_p + 50 ms). Descriptive.
  evidence age   the W200 events masked by their age a_e = end - t: `late_X` keeps a_e >= X (evidence arrives X ms late),
                 `recent_X` keeps a_e < X (only the freshest X ms); the normalisation stays that of the 200 ms window.
  persistence    d(a_p) = RA error of holding GT(end - 50 - a_p); gain G(a_p) = d(a_p) - eps(a_p); a_0 = first zero crossing.
  capture age    a_cap(c) = the age at which the mean 3-D joint displacement of GT reaches c mm (c = 14, 19: KS4 / AURA B).

Tool validity: a_p = 0 (also `extrap` at 0, and the identity evidence mask `recent_200`) equals the plain E3 run bit for bit
(identical batching), and the plain run repeats bit for bit. Smoke prints shapes / counts / timings / the checks only.

    python tools/mv2/m9_age.py --smoke
    python tools/mv2/m9_age.py --out outputs/mv2/m9_age [--workers 3 --threads 8]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mv2 import common as C  # noqa: E402
from semkine import eval_track as ET  # noqa: E402
from semkine import events as EV  # noqa: E402

AGES = (0, 10, 25, 50, 100, 200)
MAX_AGE = max(AGES)
DELTA_V = 17                          # ms, oracle velocity baseline = one 60 Hz source frame
EVIDENCE = (("full", None), ("late_25", ("late", 25)), ("late_50", ("late", 50)), ("late_100", ("late", 100)),
            ("recent_25", ("recent", 25)), ("recent_50", ("recent", 50)), ("recent_100", ("recent", 100)),
            ("recent_150", ("recent", 150)))
IDENTITY_MASK = ("recent", 200)       # keeps every event of the W200 window: must equal the plain image bit for bit
ARMS = ("aged", "extrap", "extrap_q")
CHUNK = 64
CAP_AGES = np.arange(5, 405, 5)
CAP_C = (14.0, 19.0)
SEED_SD = {"local": 0.379, "global": 0.261}      # mm, KS0 A/A (#kstar-results-20261025)
PACKET_MS = 50                        # the evaluator's packet interval (W200 = 4 x overlap)
SELECT_SEQS = C.base_seqs(C.SELECT)


# ----------------------------------------------------------------------------------------------- states
def extrapolate_cv(pos51: np.ndarray, t_c: np.ndarray, horizon: np.ndarray, delta: int = DELTA_V) -> np.ndarray:
    """Constant-velocity extrapolation of the 51-d state at ms `t_c` by `horizon` ms (oracle: the velocity is read off the GT
    over [t_c - delta, t_c]). Rows with horizon 0 return `pos51[t_c]` exactly."""
    th_c = np.asarray(pos51[t_c], np.float64)
    th_p = np.asarray(pos51[t_c - delta], np.float64)
    h = np.asarray(horizon, np.float64)
    out = th_c.copy()
    out[:, :3] = th_c[:, :3] + (th_c[:, :3] - th_p[:, :3]) / delta * h[:, None]
    rc, rp = Rot.from_rotvec(th_c[:, 3:6]), Rot.from_rotvec(th_p[:, 3:6])
    omega = (rc * rp.inv()).as_rotvec() / delta
    out[:, 3:6] = (Rot.from_rotvec(omega * h[:, None]) * rc).as_rotvec()
    out[:, 6:] = th_c[:, 6:] + (th_c[:, 6:] - th_p[:, 6:]) / delta * h[:, None]
    zero = h == 0
    out[zero] = th_c[zero]
    return out.astype(np.float32)


def evidence_mask(kind_x, window: int = C.WINDOW):
    """`SeqSamples.lnes` transform for an evidence-age mask (None = keep everything through the plain path). The event of
    relative ms `ms_rel` in a `window`-ms window has age a_e = window - 1 - ms_rel; `late` keeps a_e >= X, `recent` a_e < X.
    The normalisation by the window is untouched (the surviving events keep their original timestamps)."""
    if kind_x is None:
        return None
    kind, X = kind_x

    def transform(x, y, p, ms_rel):
        a_e = (window - 1) - ms_rel
        keep = (a_e >= X) if kind == "late" else (a_e < X)
        return x[keep], y[keep], p[keep], ms_rel[keep]
    return transform


# ----------------------------------------------------------------------------------------------- measurement
@torch.no_grad()
def forward_e3(model, X: torch.Tensor, P: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
    outs = []
    for i in range(0, len(P), chunk):
        outs.append(model(X[i:i + chunk], torch.from_numpy(np.ascontiguousarray(P[i:i + chunk]))).float().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 51), np.float32)


def ra_err(model, P: np.ndarray, G: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """(N,) root-aligned MPJPE (mm) between the 51-d states P and G."""
    jp = C.root_aligned_joints(model, P, betas, torch.device("cpu"))
    jg = C.root_aligned_joints(model, G, betas, torch.device("cpu"))
    return np.linalg.norm(jp - jg, axis=-1).mean(-1) * 1000.0


def abs_joints(model, P: np.ndarray, betas: np.ndarray, chunk: int = 512) -> np.ndarray:
    out = []
    b = torch.tensor(betas, dtype=torch.float32).view(1, -1)
    with torch.no_grad():
        for i in range(0, len(P), chunk):
            p = torch.from_numpy(np.asarray(P[i:i + chunk], np.float32))
            out.append(model._fk(p, b.expand(len(p), -1))[1].numpy())
    return np.concatenate(out)


def subset(ss: "C.SeqSamples") -> np.ndarray:
    """Samples whose stale prev (up to MAX_AGE), the velocity baseline and a full 200 ms window all lie inside the run."""
    ok = (ss.ends - STEP_PREV - MAX_AGE - DELTA_V >= ss.run_a) & (ss.windows == C.WINDOW)
    return np.flatnonzero(ok)


STEP_PREV = C.STEP


def capture_ages(model, ss, idx, betas) -> Dict[str, dict]:
    """Per-sample first age (ms) at which the mean 3-D joint displacement GT(end) <-> GT(end - a) reaches c mm."""
    ends, run_a = ss.ends[idx], ss.run_a[idx]
    have = ends - int(CAP_AGES.max()) >= run_a
    out = {f"c{int(c)}": {"n": int(have.sum())} for c in CAP_C}
    if not have.any():
        return out
    e = ends[have]
    j_now = abs_joints(model, ss.pos51[e], betas)
    disp = np.zeros((len(e), len(CAP_AGES)), np.float32)
    for k, a in enumerate(CAP_AGES):
        j_then = abs_joints(model, ss.pos51[e - int(a)], betas)
        disp[:, k] = np.linalg.norm(j_now - j_then, axis=-1).mean(-1) * 1000.0
    for c in CAP_C:
        hit = disp >= c
        first = np.where(hit.any(1), CAP_AGES[np.argmax(hit, 1)], CAP_AGES.max() + 5)
        out[f"c{int(c)}"].update(median_ms=float(np.median(first)), p25_ms=float(np.percentile(first, 25)),
                                 p75_ms=float(np.percentile(first, 75)), censored_frac=float((first > CAP_AGES.max()).mean()))
    return out


def measure_seed(seed: int, seqs=SELECT_SEQS, every: int = 8, max_n: Optional[int] = None, ages=AGES,
                 evidence=EVIDENCE, threads: int = 4, chunk: int = CHUNK, with_arms: bool = True,
                 verbose: bool = False) -> dict:
    """All E3 conditions of one seed. Returns {'table': {domain: {arm: {evid: {age: mean_mm}}}}, 'n': {...}, 'checks': {...},
    'cap': {...}, 'per_seq': {...}, 'errors': {name: array}}."""
    torch.set_num_threads(threads)
    model, cfg = C.load_s39(seed, torch.device("cpu"))
    sums: Dict[str, Dict] = {}
    n_dom = {"local": 0, "global": 0}
    checks = {"a0_bit_exact": True, "extrap0_bit_exact": True, "identity_mask_bit_exact": True,
              "plain_repeat_bit_exact": True}
    per_seq: Dict[str, dict] = {}
    cap: Dict[str, List[dict]] = {"local": [], "global": []}
    errors: Dict[str, np.ndarray] = {}
    t0 = time.time()
    for seq in seqs:
        dom = "local" if seq.endswith("_local") else "global"
        ss = C.SeqSamples(seq, every=every, max_n=max_n)
        idx = subset(ss)
        if len(idx) == 0:
            continue
        model.set_hand_context(torch.from_numpy(ss.betas).view(1, -1), torch.from_numpy(ss.K).view(1, 3, 3))
        ends = ss.ends[idx]
        G = ss.gt[idx]
        betas = ss.betas
        n_dom[dom] += len(idx)
        # ---- plain E3 (independent construction) and its repeat
        X_plain = torch.from_numpy(np.stack([ET.build_lnes(ss._events, ss._offsets, int(ss.ends[i]), int(ss.windows[i]),
                                                           ss.channels) for i in idx]))
        P_plain = np.stack([np.asarray(ss.pos51[int(e) - C.STEP], np.float32) for e in ends])
        out_plain = forward_e3(model, X_plain, P_plain, chunk)
        checks["plain_repeat_bit_exact"] &= bool(np.array_equal(out_plain, forward_e3(model, X_plain, P_plain, chunk)))
        per_seq[seq] = {"n": int(len(idx))}
        cap[dom].append(capture_ages(model, ss, idx, betas))
        # ---- persistence d(a)
        for a in ages:
            P_a = np.stack([np.asarray(ss.pos51[int(e) - C.STEP - a], np.float32) for e in ends])
            sums.setdefault(dom, {}).setdefault("hold", {}).setdefault("full", {}).setdefault(a, []).append(
                ra_err(model, P_a, G, betas))
        # ---- evidence conditions x ages x arms
        for ev_name, ev_kind in evidence:
            X = X_plain if ev_kind is None else torch.from_numpy(ss.lnes(idx, transform=evidence_mask(ev_kind)))
            for a in ages:
                P_aged = np.stack([np.asarray(ss.pos51[int(e) - C.STEP - a], np.float32) for e in ends])
                arms = {"aged": P_aged}
                if with_arms and ev_name == "full":
                    t_c = ends - C.STEP - a
                    arms["extrap"] = extrapolate_cv(ss.pos51, t_c, np.full(len(idx), a))
                    arms["extrap_q"] = extrapolate_cv(ss.pos51, t_c, np.full(len(idx), a + C.STEP))
                for arm, P in arms.items():
                    out = forward_e3(model, X, P, chunk)
                    err = ra_err(model, out, G, betas)
                    sums.setdefault(dom, {}).setdefault(arm, {}).setdefault(ev_name, {}).setdefault(a, []).append(err)
                    errors[f"{seq}|{arm}|{ev_name}|{a}"] = err.astype(np.float32)
                    if a == 0 and arm in ("aged", "extrap") and ev_name == "full":
                        same = bool(np.array_equal(out, out_plain))
                        checks["a0_bit_exact" if arm == "aged" else "extrap0_bit_exact"] &= same
            if verbose:
                print(f"    seed {seed} {seq} evidence {ev_name} done ({time.time() - t0:.0f} s)", flush=True)
        # identity mask through the transform path
        X_id = torch.from_numpy(ss.lnes(idx, transform=evidence_mask(IDENTITY_MASK)))
        checks["identity_mask_bit_exact"] &= bool(np.array_equal(forward_e3(model, X_id, P_plain, chunk), out_plain))
    table = {dom: {arm: {ev: {str(a): float(np.concatenate(v).mean()) for a, v in by_age.items()}
                         for ev, by_age in by_ev.items()} for arm, by_ev in arms.items()} for dom, arms in sums.items()}
    return {"seed": seed, "table": table, "n": n_dom, "checks": checks, "cap": cap, "per_seq": per_seq,
            "secs": time.time() - t0, "errors": errors}


def _seed_task(args):
    seed, kw = args
    return measure_seed(seed, **kw)


# ----------------------------------------------------------------------------------------------- derived quantities
def derive(result: dict) -> dict:
    """Increments, persistence gain G, zero-crossing age a_0, capture ages and extrapolation recovery per seed / domain
    from the raw table (pure function of `result['seeds']`)."""
    out: Dict[str, dict] = {}
    for dom in ("local", "global"):
        per_seed = {}
        for s, r in result["seeds"].items():
            t = r["table"][dom]
            aged, hold = t["aged"]["full"], t["hold"]["full"]
            ages = sorted(int(a) for a in aged)
            eps = {a: aged[str(a)] for a in ages}
            G = {a: hold[str(a)] - aged[str(a)] for a in ages}
            a0 = None
            for a_prev, a in zip(ages[:-1], ages[1:]):
                if G[a_prev] > 0 and G[a] <= 0:
                    a0 = a_prev + (a - a_prev) * G[a_prev] / (G[a_prev] - G[a])
                    break
            rec = {}
            if "extrap" in t:
                for a in ages:
                    inc = eps[a] - eps[0]
                    rec[a] = (eps[a] - t["extrap"]["full"][str(a)]) / inc if (a > 0 and inc > 0) else None
            per_seed[s] = {"eps": eps, "hold": {a: hold[str(a)] for a in ages}, "G": G, "inc": {a: eps[a] - eps[0] for a in ages},
                           "a0_ms": a0, "G50_over_G0": (G[50] / G[0]) if (50 in G and G[0] != 0) else None,
                           "extrap_recovery": rec,
                           "evidence": {ev: {a: v[str(a)] - aged[str(a)] for a in ages if str(a) in v}
                                        for ev, v in t["aged"].items() if ev != "full"}}
        out[dom] = per_seed
    return out


def _cap_pool(result: dict, dom: str, c: str):
    # GT and the SELECT samples are the same for every seed: use the first seed's per-sequence medians, sample-weighted
    first = next(iter(result["seeds"].values()))
    rows = [(m[c]["median_ms"], m[c]["n"]) for m in first["cap"][dom] if "median_ms" in m[c]]
    if not rows:
        return None
    return float(sum(v * n for v, n in rows) / sum(n for _, n in rows))


# ----------------------------------------------------------------------------------------------- decision
def decide(result: dict) -> dict:
    """The pre-registered gates of PAS / CAF (tools/mv2/prereg_m9.md section 2).

    Validity (else `tool_defect`): a_p = 0 bit-exact against the plain E3 (aged, extrap, identity mask) and the plain run
    repeats bit for bit, every seed; >= 100 samples per domain.
    Axis (PAS, per domain; inc50 = eps(a_p = 50) - eps(a_p = 0), SD = 0.379 / 0.261): alive = all three seeds >= 2 SD
    (positive); dead = no seed >= 1 SD, or the three signs differ; else indeterminate. Overall: alive iff both domains alive;
    dead iff both domains dead; else mixed.
    CAF (per domain): G(50) / G(0) >= 0.5 in every seed and a_cap(19 mm) >= 100 ms => `frontier_ok`; all seeds G(50) <= 0 or
    a_cap(19 mm) < 50 ms => `positional_channels_downgrade` (either domain); else indeterminate.
    Extrapolation (descriptive): recovery at a_p = 50 >= 0.5 in every seed (placeholder 0.5, 'to be validated')."""
    out: Dict[str, object] = {"verdict": None, "notes": []}
    seeds = result["seeds"]
    for s, r in seeds.items():
        bad = [k for k, v in r["checks"].items() if not v]
        if bad:
            return {"verdict": "tool_defect", "notes": [f"seed {s}: {bad}"]}
        for dom in ("local", "global"):
            if r["n"][dom] < 100:
                return {"verdict": "tool_defect", "notes": [f"seed {s}: only {r['n'][dom]} {dom} samples"]}
    der = derive(result)
    out["derived_keys"] = sorted(der)
    axis, caf = {}, {}
    for dom in ("local", "global"):
        sd = SEED_SD[dom]
        inc = [der[dom][s]["inc"][50] for s in sorted(der[dom])]
        same_sign = all(v > 0 for v in inc) or all(v < 0 for v in inc)
        if all(v >= 2 * sd for v in inc):
            cls = "alive"
        elif max(inc) < 1 * sd or not same_sign:
            cls = "dead"
        else:
            cls = "indeterminate"
        axis[dom] = {"inc50_mm": inc, "threshold_alive_mm": 2 * sd, "threshold_dead_mm": sd, "same_sign": same_sign, "class": cls}
        g50 = [der[dom][s]["G50_over_G0"] for s in sorted(der[dom])]
        g_raw = [der[dom][s]["G"][50] for s in sorted(der[dom])]
        a_cap19 = _cap_pool(result, dom, "c19")
        ok = all(g is not None and g >= 0.5 for g in g50) and a_cap19 is not None and a_cap19 >= 2 * PACKET_MS
        down = all(g <= 0 for g in g_raw) or (a_cap19 is not None and a_cap19 < PACKET_MS)
        caf[dom] = {"G50_over_G0": g50, "G50_mm": g_raw, "a0_ms": [der[dom][s]["a0_ms"] for s in sorted(der[dom])],
                    "a_cap_19mm_ms": a_cap19, "a_cap_14mm_ms": _cap_pool(result, dom, "c14"),
                    "class": "frontier_ok" if ok else ("positional_channels_downgrade" if down else "indeterminate")}
        rec = [der[dom][s]["extrap_recovery"].get(50) for s in sorted(der[dom])]
        caf[dom]["extrap_recovery_50"] = rec
        caf[dom]["extrap_value"] = bool(all(r is not None and r >= 0.5 for r in rec))
    classes = [axis[d]["class"] for d in axis]
    out["axis"] = axis
    out["caf"] = caf
    out["axis_verdict"] = ("alive" if all(c == "alive" for c in classes) else
                           "dead" if all(c == "dead" for c in classes) else "mixed" if "alive" in classes or "dead" in classes
                           else "indeterminate")
    out["caf_verdict"] = ("positional_channels_downgrade" if any(caf[d]["class"] == "positional_channels_downgrade" for d in caf)
                          else "frontier_ok" if all(caf[d]["class"] == "frontier_ok" for d in caf) else "indeterminate")
    out["verdict"] = f"axis_{out['axis_verdict']}|caf_{out['caf_verdict']}"
    if out["axis_verdict"] == "dead":
        out["notes"].append("no accuracy lever in the previous-state age: M9 stays a pure systems contract; 'two-speed "
                            "improves accuracy' is permanently not claimed")
    return out


# ----------------------------------------------------------------------------------------------- main
def run(seeds, workers: int, **kw) -> dict:
    tasks = [(s, kw) for s in seeds]
    t0 = time.time()
    if workers <= 1:
        res = [_seed_task(t) for t in tasks]
    else:
        with mp.get_context("spawn").Pool(min(workers, len(tasks))) as pool:
            res = pool.map(_seed_task, tasks, chunksize=1)
    seeds_out = {}
    errors = {}
    for r in res:
        errors[str(r["seed"])] = r.pop("errors")
        seeds_out[str(r["seed"])] = r
    return {"seeds": seeds_out, "secs": time.time() - t0, "errors": errors}


def smoke(args) -> int:
    t0 = time.time()
    ok = True
    # ---- unit checks on synthetic states (no data)
    rng = np.random.default_rng(1)
    pos = np.zeros((600, 51), np.float32)
    tt = np.arange(600, dtype=np.float32)
    pos[:, :3] = np.stack([0.001 * tt, -0.002 * tt, 0.0005 * tt], 1)                        # constant translation velocity
    pos[:, 3:6] = np.stack([0.002 * tt, 0.0 * tt, 0.001 * tt], 1)                                # small rotation about a fixed axis
    pos[:, 6:] = 0.0005 * tt[:, None] * rng.standard_normal(45).astype(np.float32)[None]   # constant finger velocities
    t_c = np.array([300, 300, 450], dtype=np.int64)
    hz = np.array([0, 50, 100])
    ex = extrapolate_cv(pos, t_c, hz)
    exact0 = bool(np.array_equal(ex[0], pos[300]))
    lin_ok = bool(np.allclose(ex[1:, :3], pos[t_c[1:] + hz[1:], :3], atol=1e-5)
                  and np.allclose(ex[1:, 6:], pos[t_c[1:] + hz[1:], 6:], atol=1e-5)
                  and np.allclose(ex[1:, 3:6], pos[t_c[1:] + hz[1:], 3:6], atol=1e-4))
    print(f"[m9_age smoke] extrapolate_cv: horizon 0 exact={exact0}, exact for constant velocity={lin_ok}")
    ok &= exact0 and lin_ok
    x = np.array([5, 6, 7, 8], np.int64)
    ms4 = np.array([0, 1, 2, 3], np.float32)
    mask_ok = (evidence_mask(("late", 2), 4)(x, x, x % 2, ms4)[3].tolist() == [0.0, 1.0]
               and evidence_mask(("recent", 2), 4)(x, x, x % 2, ms4)[3].tolist() == [2.0, 3.0])
    print(f"  evidence mask keeps late (age >= X) / recent (age < X) events: {mask_ok}")
    ok &= mask_ok
    # ---- the tool path on a tiny FIT7 slice (never SELECT / zgz): shapes, counts, bit-exactness checks
    seq = ["ch_local"]
    r = measure_seed(3407, seqs=seq, every=args.every, max_n=args.max_n, ages=(0, 50, 200),
                     evidence=(("full", None), ("late_50", ("late", 50)), ("recent_100", ("recent", 100))),
                     threads=args.threads, with_arms=True)
    n = r["n"]["local"]
    print(f"  tool path ({seq[0]}, {n} samples, ages (0, 50, 200), 3 evidence conditions, 3 arms): {time.time() - t0:.1f} s")
    print(f"  checks: {r['checks']}")
    print(f"  table shape: domains={sorted(r['table'])} arms={sorted(r['table']['local'])} "
          f"evidence={sorted(r['table']['local']['aged'])} ages={sorted(r['table']['local']['aged']['full'])}")
    print(f"  capture-age fields: {sorted(r['cap']['local'][0]) if r['cap']['local'] else []}")
    ok &= all(r["checks"].values()) and n > 0
    # ---- derive / decide plumbing on a synthetic table (no effects)
    def fake_seed(s, inc50):
        tab = {}
        for dom in ("local", "global"):
            base = 4.0
            aged = {str(a): base + inc50 * a / 50 for a in AGES}
            hold = {str(a): base + 2.0 + 0.05 * a for a in AGES}
            tab[dom] = {"aged": {"full": aged}, "hold": {"full": hold},
                        "extrap": {"full": {str(a): base for a in AGES}}}
        return {"table": tab, "checks": {k: True for k in ("a0_bit_exact", "extrap0_bit_exact", "identity_mask_bit_exact",
                                                           "plain_repeat_bit_exact")},
                "n": {"local": 200, "global": 200}, "cap": {d: [{"c14": {"median_ms": 120.0, "n": 50},
                                                                 "c19": {"median_ms": 150.0, "n": 50}}] for d in ("local", "global")}}
    res_alive = {"seeds": {str(s): fake_seed(s, 2.0) for s in C.SEEDS}}
    res_dead = {"seeds": {str(s): fake_seed(s, 0.1) for s in C.SEEDS}}
    d1, d2 = decide(res_alive), decide(res_dead)
    plumb = d1["axis_verdict"] == "alive" and d2["axis_verdict"] == "dead"
    print(f"  decide() plumbing on synthetic tables: alive->{d1['axis_verdict']}, dead->{d2['axis_verdict']}")
    ok &= plumb
    out_dir = REPO / "outputs" / "mv2" / "smoke" / "m9_age"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke.json").write_text(json.dumps({"checks": r["checks"], "n": r["n"], "ok": bool(ok)}, indent=1))
    print(f"[m9_age smoke] {'PASS' if ok else 'FAIL'} ({time.time() - t0:.1f} s)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--max-n", type=int, default=6)
    a = ap.parse_args()
    if a.smoke:
        raise SystemExit(smoke(a))
    if not a.out:
        raise SystemExit("--out outputs/mv2/m9_age (formal) or --smoke")
    C.verify_protocol()
    out_dir = Path(a.out)
    res = run(C.SEEDS, a.workers, every=a.every, threads=a.threads)
    errors = res.pop("errors")
    out_dir.mkdir(parents=True, exist_ok=True)
    for s, e in errors.items():
        np.savez_compressed(out_dir / f"errors_seed{s}.npz", **e)
    res["derived"] = derive(res)
    res["decision"] = decide(res)
    C.save_json(out_dir / "result.json", res)
    C.receipt(out_dir, {"probe": "m9_age", "verdict": res["decision"]["verdict"]},
              scripts=["tools/mv2/m9_age.py", "tools/mv2/common.py"])
    print(json.dumps(res["decision"], indent=1, default=C._np))


if __name__ == "__main__":
    main()
