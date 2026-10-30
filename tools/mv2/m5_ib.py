#!/usr/bin/env python3
"""M5 probe 1: injection recovery IBC / IBR (P6-E1-R2, P6-E5-R3).

Per anchor (a window end on the evaluator's 50 ms grid with >= 1 s of valid history) and per case (injected root-rotation
bias of 2 / 5 / 10 deg about a camera axis x / y / z with a seeded sign, plus the zero-bias floor), the tracker trajectory
= GT + injected bias (+ nuisance noise, see `m5_core.make_noise`) is aligned to the REAL events of the causal windows
50 / 100 / 200 / 500 / 1000 ms ending at the anchor by the palm aligner (`m5_core.align6`).  The null is the same
trajectory against the time-shuffled events of the same window (positions fixed, ms stamps permuted).

Quantities per (case, window, variant): rho = 1 - residual / |b| (fraction removed; overshoot is not recovery), the
projection slope, the residual and correction angles, the posterior-SD identifiability gate.  Summaries: median rho,
fraction >= 0.5, time-to-threshold (smallest window with rho >= 0.5, censored = inf), paired real - null difference with an
anchor-clustered bootstrap CI.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from mv2 import m5_core as M

SIGNS = {}                                      # filled lazily per (seq, end, axis): seeded sign
PRIMARY = {"mode": "const", "evidence": "palm"}
THRESH = 0.5                                    # rho threshold of the time-to-threshold (half of the bias removed)
ARM_WINDOWS = (200, 1000)                       # windows of the sensitivity arms
_CACHE: Dict = {}


def _worker_init(threads: int):
    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)


def _geo() -> M.Geo:
    if "g" not in _CACHE:
        _CACHE["g"] = M.Geo()
    return _CACHE["g"]


def _subj(seq: str) -> M.Subject:
    if seq not in _CACHE:
        if len([k for k in _CACHE if k != "g"]) >= 2:                      # keep at most two recordings in memory
            for k in [k for k in _CACHE if k != "g"][:1]:
                del _CACHE[k]
        _CACHE[seq] = M.Subject.load(seq)
    return _CACHE[seq]


def case_list(seq: str, end: int, tiers: Sequence[float] = M.TIERS_DEG) -> List[dict]:
    """The balanced design of one anchor: the zero-bias floor + (tier x axis) with a seeded sign."""
    cases = [{"tier": 0.0, "axis": "none", "sign": 1}]
    for t in tiers:
        for ax in M.AXES:
            sg = 1 if (M.seed_of("sign", seq, end, t, ax) & 1) else -1
            cases.append({"tier": float(t), "axis": ax, "sign": sg})
    return cases


def _arms(case: dict) -> List[dict]:
    """(mode, evidence, windows, variants) runs of one case: the primary arm on every window with the shuffle null, the
    sensitivity arms (clean / AR(1) / iid nuisance, full-hand evidence) on the 5 deg tier only."""
    runs = [{"mode": "const", "evidence": "palm", "windows": M.WINDOWS_MS, "variants": ("real", "shuffle")}]
    if case["tier"] == 5.0:
        runs += [{"mode": m, "evidence": "palm", "windows": ARM_WINDOWS, "variants": ("real",)} for m in ("none", "ar", "iid")]
        runs += [{"mode": "const", "evidence": "full", "windows": ARM_WINDOWS, "variants": ("real",)}]
    return runs


def run_anchor(task: dict) -> List[dict]:
    """Worker: all solves of one anchor.  task = {seq, end, tiers?, windows?, quick?}"""
    seq, end = task["seq"], int(task["end"])
    g, sj = _geo(), _subj(seq)
    wmax = max(M.WINDOWS_MS)
    out: List[dict] = []
    for case in case_list(seq, end, task.get("tiers", M.TIERS_DEG)):
        for run in _arms(case):
            wins = [w for w in run["windows"] if w in task.get("windows", M.WINDOWS_MS)]
            if not wins:
                continue
            seed = M.seed_of(seq, end, case["tier"], case["axis"], run["mode"])
            rng = np.random.default_rng(seed)
            b = np.zeros(3) if case["axis"] == "none" else M.axis_vec(case["axis"], case["sign"], case["tier"])
            noise_rng = np.random.default_rng(M.seed_of(seq, end, case["tier"], case["axis"], run["mode"], "noise"))
            draws = None
            if run["mode"] in ("ar", "iid"):
                draws = M.noise_knots(end, end - wmax, noise_rng, rho=M.NOISE_RHO if run["mode"] == "ar" else 0.0)
            elif run["mode"] == "const":
                d = noise_rng.standard_normal(51) * M.sig51()
                d[3:6] = 0.0
            for w in wins:
                m_prev = end - w
                edges = M.window_edges(m_prev, end)
                gt = sj.pos51[edges]
                if run["mode"] == "none":
                    nz = None
                elif run["mode"] == "const":
                    nz = np.broadcast_to(d, (len(edges), 51)).copy()
                else:
                    nz = M.noise_at_edges(edges, end, draws)
                Xp = M.inject(gt, b, nz)
                btrue = M.true_offset(gt, Xp)
                ev, ms = sj.window_events(m_prev, end)
                pb = M.make_problem(g, Xp, sj.betas, sj.K, ev, ms, m_prev, end, run["evidence"])
                for variant in run["variants"]:
                    pbv = pb
                    if variant == "shuffle":
                        pbv = M.rewindow(pb, ev, M.shuffle_times(ms, np.random.default_rng(M.seed_of(seed, w, "shuf"))), m_prev, end)
                    t0 = time.time()
                    sv = M.align6(pbv)
                    rec = M.recovery(sv.theta, btrue)
                    rec.update({"seq": seq, "end": end, "tier": case["tier"], "axis": case["axis"], "sign": case["sign"],
                                "mode": run["mode"], "evidence": run["evidence"], "window": w, "variant": variant,
                                "sd_deg": sv.sd_deg, "gate_open": bool(sv.gate_open), "secs": time.time() - t0,
                                "n_events": pbv.n_events, "n_cells": int(pbv.win.crop.cells),
                                "theta": [float(x) for x in sv.theta], "finite": bool(torch.isfinite(sv.theta).all())})
                    out.append(rec)
    return out


# ----------------------------------------------------------------------------------------------- summaries
def _q(x, q):
    x = np.asarray(x, float)
    return float(np.nanpercentile(x, q)) if len(x) else float("nan")


def _cluster_median_ci(values: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = M.SEED):
    """Percentile 95 % CI of the median, resampling whole groups (anchors)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        bs.append(np.median(np.concatenate([values[idx[g]] for g in pick])))
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _p90_time(times: np.ndarray) -> float:
    """90th percentile of the time-to-threshold with censoring (inf counts as a failure): the smallest t such that at least 90 %
    of the cases reached the threshold by t; inf if fewer than 90 % ever did."""
    t = np.sort(np.asarray(times, float))
    k = int(math.ceil(0.9 * len(t))) - 1
    return float(t[k]) if len(t) else float("nan")


def subject_set(seq: str, select: Sequence[str]) -> str:
    return "SELECT" if seq.split("_")[0] in select else "FIT7"


def summarize(records: List[dict], select: Sequence[str]) -> dict:
    """The full IBC / IBR summary dict (JSON-able).  Records of every arm / evidence."""
    import collections
    R = [r for r in records if r["finite"]]
    n_bad = len(records) - len(R)
    out = {"n_records": len(records), "n_nonfinite": n_bad, "windows": list(M.WINDOWS_MS), "primary": {}, "arms": {},
           "floor": {}, "gate_inputs": {}}
    sets = ("SELECT", "FIT7")
    prim = [r for r in R if r["mode"] == "const" and r["evidence"] == "palm"]
    key = lambda r: (r["seq"], r["end"], r["tier"], r["axis"])
    by = collections.defaultdict(dict)                         # case -> {(window, variant): record}
    for r in prim:
        by[key(r)][(r["window"], r["variant"])] = r
    for S in sets:
        out["primary"][S] = {}
        for tier in M.TIERS_DEG:
            cases = [(k, v) for k, v in by.items() if k[2] == tier and subject_set(k[0], select) == S]
            tier_out = {"n_cases": len(cases), "windows": {}}
            for w in M.WINDOWS_MS:
                real = [(k, v[(w, "real")]) for k, v in cases if (w, "real") in v]
                null = {k: v[(w, "shuffle")] for k, v in cases if (w, "shuffle") in v}
                rho = np.array([r["rho"] for _, r in real])
                rho_null = np.array([null[k]["rho"] for k, _ in real if k in null])
                gate_open = np.array([r["gate_open"] for _, r in real])
                d = np.array([r["rho"] - null[k]["rho"] for k, r in real if k in null])
                grp = np.array([f"{k[0]}|{k[1]}" for k, _ in real if k in null])
                lo, hi = _cluster_median_ci(d, grp) if len(d) > 4 else (float("nan"), float("nan"))
                per_axis = {ax: float(np.median([r["rho"] for k, r in real if k[3] == ax])) if any(k[3] == ax for k, _ in real) else float("nan")
                            for ax in M.AXES}
                tier_out["windows"][str(w)] = {
                    "n": len(rho), "median_rho": float(np.median(rho)) if len(rho) else float("nan"),
                    "p25_rho": _q(rho, 25), "p75_rho": _q(rho, 75), "frac_ge_thresh": float((rho >= THRESH).mean()) if len(rho) else float("nan"),
                    "median_slope": float(np.median([r["slope"] for _, r in real])) if real else float("nan"),
                    "median_resid_deg": float(np.median([r["resid_deg"] for _, r in real])) if real else float("nan"),
                    "median_rho_null": float(np.median(rho_null)) if len(rho_null) else float("nan"),
                    "frac_ge_thresh_null": float((rho_null >= THRESH).mean()) if len(rho_null) else float("nan"),
                    "median_diff_real_minus_null": float(np.median(d)) if len(d) else float("nan"),
                    "diff_ci95": [lo, hi], "per_axis_median_rho": per_axis,
                    "gate_open_frac": float(gate_open.mean()) if len(gate_open) else float("nan"),
                    "median_rho_gated": float(np.median(np.where(gate_open, rho, 0.0))) if len(rho) else float("nan"),
                    "median_sd_deg": _q([r["sd_deg"] for _, r in real], 50),
                    "median_events": _q([r["n_events"] for _, r in real], 50),
                    "median_secs": _q([r["secs"] for _, r in real], 50)}
            # time to threshold per case (real and null)
            for variant in ("real", "shuffle"):
                T = []
                for k, v in cases:
                    t = math.inf
                    for w in M.WINDOWS_MS:
                        if (w, variant) in v and v[(w, variant)]["rho"] >= THRESH:
                            t = float(w)
                            break
                    T.append(t)
                T = np.array(T)
                tier_out[f"time_{variant}"] = {"p50": _p90_q(T, 0.5), "p90": _p90_time(T) if len(T) else float("nan"),
                                               "frac_reached": {str(w): float((T <= w).mean()) if len(T) else float("nan")
                                                                for w in M.WINDOWS_MS}}
            out["primary"][S][str(tier)] = tier_out
    # zero-bias floor: spurious correction angle of the aligner (what it hallucinates with nothing to recover)
    for S in sets:
        out["floor"][S] = {}
        for w in M.WINDOWS_MS:
            c = [r["corr_deg"] for r in prim if r["tier"] == 0.0 and r["variant"] == "real" and r["window"] == w and subject_set(r["seq"], select) == S]
            out["floor"][S][str(w)] = {"n": len(c), "median_corr_deg": _q(c, 50), "p90_corr_deg": _q(c, 90)}
    # sensitivity arms: tier 5, windows 200 / 1000, real only
    for arm, flt in (("clean(none)", lambda r: r["mode"] == "none" and r["evidence"] == "palm"),
                     ("ar1", lambda r: r["mode"] == "ar"), ("iid", lambda r: r["mode"] == "iid"),
                     ("full_hand", lambda r: r["mode"] == "const" and r["evidence"] == "full"),
                     ("palm_primary", lambda r: r["mode"] == "const" and r["evidence"] == "palm" and r["variant"] == "real" and r["tier"] == 5.0 and r["window"] in ARM_WINDOWS)):
        out["arms"][arm] = {}
        for S in sets:
            out["arms"][arm][S] = {}
            for w in ARM_WINDOWS:
                rr = [r for r in R if flt(r) and r["tier"] == 5.0 and r["window"] == w and r["variant"] == "real" and subject_set(r["seq"], select) == S]
                out["arms"][arm][S][str(w)] = {"n": len(rr), "median_rho": _q([r["rho"] for r in rr], 50),
                                               "frac_ge_thresh": float(np.mean([r["rho"] >= THRESH for r in rr])) if rr else float("nan"),
                                               "median_resid_deg": _q([r["resid_deg"] for r in rr], 50)}
    # sigma* stratification (descriptive): median rho of the primary 5 deg tier at 1000 ms by recording sigma* tertile
    ss = {}
    for r in prim:
        if r["tier"] == 5.0 and r["window"] == 1000 and r["variant"] == "real":
            sv = M.sigma_star(r["seq"])
            if sv is not None:
                ss.setdefault(r["seq"], []).append(r["rho"])
    if ss:
        sig = {s: M.sigma_star(s) for s in ss}
        cuts = np.percentile(list(sig.values()), [33.3, 66.7])
        strata = {"low": [], "mid": [], "high": []}
        for s, v in ss.items():
            strata["low" if sig[s] <= cuts[0] else ("mid" if sig[s] <= cuts[1] else "high")] += v
        out["sigma_star_strata"] = {k: {"n": len(v), "median_rho": _q(v, 50)} for k, v in strata.items()}
        out["sigma_star_strata"]["cuts"] = [float(c) for c in cuts]
    # gate inputs (SELECT, 5 deg, primary arm)
    g5 = out["primary"]["SELECT"].get("5.0", {})
    w1000 = g5.get("windows", {}).get("1000", {})
    out["gate_inputs"] = {
        "median_rho_5deg_1000ms": w1000.get("median_rho", float("nan")),
        "p90_time_ms_5deg": g5.get("time_real", {}).get("p90", float("nan")),
        "median_diff_vs_shuffle_5deg_1000ms": w1000.get("median_diff_real_minus_null", float("nan")),
        "diff_ci95_5deg_1000ms": w1000.get("diff_ci95", [float("nan"), float("nan")]),
        "median_rho_5deg_500ms": g5.get("windows", {}).get("500", {}).get("median_rho", float("nan")),
        "n_cases_5deg": g5.get("n_cases", 0)}
    return out


def _p90_q(T: np.ndarray, q: float) -> float:
    t = np.sort(np.asarray(T, float))
    if not len(t):
        return float("nan")
    return float(t[int(math.ceil(q * len(t))) - 1])


def decide(summary: dict) -> dict:
    """Gate of IBC / IBR on SELECT, 5 deg tier, primary arm (clean-offset injection with the constant T / finger nuisance):
    pass  = median rho(1000 ms) >= 0.5  and  p90 time-to-threshold <= 1000 ms  and  CI of the median (real - shuffled) > 0;
    kill  = median rho(1000 ms) < 0.3  or  p90 time > 1000 ms (censored)  or  the CI contains 0 (no better than shuffled);
    else descriptive.  Validity: any non-finite solve, or the planted-signal tool check failing, is a 'tool_defect'."""
    gi = summary["gate_inputs"]
    reasons_kill, reasons_fail = [], []
    med, p90 = gi["median_rho_5deg_1000ms"], gi["p90_time_ms_5deg"]
    lo = gi["diff_ci95_5deg_1000ms"][0]
    n_bad = summary.get("n_nonfinite", 0)
    tool_ok = (n_bad == 0) and bool(summary.get("planted_ok", True))
    if not tool_ok:
        return {"verdict": "tool_defect", "gate_inputs": gi, "n_nonfinite": n_bad, "planted_ok": summary.get("planted_ok")}
    if not (med == med):
        return {"verdict": "tool_defect", "reason": "no SELECT 5 deg cases", "gate_inputs": gi}
    if med < 0.3:
        reasons_kill.append("median rho < 0.3")
    if p90 > 1000:
        reasons_kill.append("p90 time-to-threshold > 1 s (censored)")
    if not (lo > 0):
        reasons_kill.append("CI of real - shuffled median includes 0")
    passed = med >= 0.5 and p90 <= 1000 and lo > 0
    if passed:
        verdict = "pass"
    elif reasons_kill:
        verdict = "kill"
    else:
        verdict = "descriptive"
    return {"verdict": verdict, "kill_reasons": reasons_kill, "gate_inputs": gi,
            "disposition": {"pass": "IBR certificate: palm alignment may open the 'M1 + root pathway' arm (with FIPA)",
                            "kill": "segment-level root bias only as an offline per-segment calibration",
                            "descriptive": "no claim"}[verdict]}
