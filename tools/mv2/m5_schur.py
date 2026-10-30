#!/usr/bin/env python3
"""M5 probe 2: bias-trajectory Schur identifiability spectrum (P3-E6-R2).

At the GT trajectory of each anchor window (50 / 200 / 500 / 1000 ms) the SBA-footprint Poisson Fisher (same likelihood as
the injection recovery, palm evidence, real events) of the joint parameter [b (6), s_0 .. s_{nk-1} (6 each)] is built with
the correction(t) = b + s(t) parametrisation (`m5_core.KnotParam`, knots every 100 ms).  Schur-complementing the
trajectory block gives the bias block S_b = (F_bb + P_bb) - (F_bs + P_bs)(F_ss + P_ss)^-1 (F_sb + P_sb):

    regime 'noprior'  : no prior at all                       -> S_b == 0 exactly (b is the constant mode of s): the theorem
    regime 'temporal' : P_b + label velocity / acceleration   -> S_b == P_b: no information about b without a level anchor
    regime 'full'     : P_b + velocity + acceleration + level prior on s (PREV_NOISE scale)   -> the gate quantity
    regime 'loose3'   : 'full' with the level prior 3x looser -> prior-dependence exponent of the posterior SD

and the worst-axis posterior SD of the bias rotation (deg).  The second half is the MAP recovery of an injected constant
root bias under the same joint model: real events vs time-shuffled events vs the 'temporal' (no level prior) regime.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Sequence

import numpy as np
import torch

from mv2 import m5_core as M
from mv2 import m5_ib as IB

SPEC_WINDOWS = (50, 200, 500, 1000)
MAP_WINDOWS = (200, 1000)
MAP_TIERS = (2.0, 5.0)
MAP_ITERS, MAP_POOLS, MAP_BLURS, MAP_MUS = 5, (2, 1, 1, 1, 1), (6.0, 3.0, 2.0, 1.0, 1.0), (0.5, 0.3, 0.2, 0.1, 0.05)
SD_TARGET_DEG = 2.0
LEVEL_LOOSE = 3.0
RANK_RATIO_TOL = 1e-2                           # ||S_b(noprior)||_F / ||F_bb||_F below this = numerically rank deficient (FD leakage ~1e-3)
REGIMES = ("noprior", "temporal", "full", "loose3")


def map_joint(pb: M.Problem, nk: int, prior: torch.Tensor):
    """Fixed-iteration safeguarded MAP of [b, s] from zero under `prior` (forward-difference Gauss-Newton)."""
    t0 = time.time()
    param = M.KnotParam(pb.T, nk)
    theta = torch.zeros(param.P)
    for it in range(MAP_ITERS):
        ev = M.evaluate(pb, param, theta, MAP_POOLS[it], central=False, blur=MAP_BLURS[it])
        H = ev.F + prior
        rhs = ev.g - prior @ theta.double()
        step = torch.linalg.solve(H + MAP_MUS[it] * torch.diag(H.diagonal().clamp(min=1e-9)), rhs).float()
        theta = M.safeguarded(pb, param, theta, M._cap_step(step, param), ev, prior)
    return theta, time.time() - t0


def spectrum_window(pb: M.Problem, nk: int, ls: dict) -> dict:
    F = M.fisher_joint(pb, nk, p=1)
    Fbb = F[:6, :6]
    out = {"nk": nk, "n_events": pb.n_events, "psd_min_eig": float(torch.linalg.eigvalsh(F).min()),
           "fisher_trace": float(torch.trace(F)), "regimes": {}}
    for reg in REGIMES:
        if reg == "noprior":
            prior = M.prior_precision(nk, ls, "noprior")
        elif reg == "temporal":
            prior = M.prior_precision(nk, ls, "temporal")
        elif reg == "full":
            prior = M.prior_precision(nk, ls, "full")
        else:
            prior = M.prior_precision(nk, ls, "full", level_mult=LEVEL_LOOSE)
        Sb = M.schur_b(F, prior)
        ev = torch.linalg.eigvalsh(0.5 * (Sb + Sb.T))
        sd = M.posterior_sd_deg(Sb) if reg != "noprior" else float("inf")
        out["regimes"][reg] = {"sd_deg": sd, "eig_min": float(ev[0]), "eig_max": float(ev[-1]),
                               "ratio_to_Fbb": float(torch.linalg.norm(Sb) / torch.linalg.norm(Fbb).clamp(min=1e-30))}
    # data-only information about b if the trajectory were known (no Schur), with the weak bias prior: the ceiling
    prior_b = M.prior_precision(nk, ls, "temporal")[:6, :6]
    out["sd_known_trajectory_deg"] = M.posterior_sd_deg(Fbb + prior_b)
    return out


def run_anchor(task: dict) -> dict:
    seq, end = task["seq"], int(task["end"])
    ls = task["label_stats"]
    g, sj = IB._geo(), IB._subj(seq)
    res = {"seq": seq, "end": end, "spectrum": {}, "map": []}
    for w in task.get("spec_windows", SPEC_WINDOWS):
        m_prev = end - w
        edges = M.window_edges(m_prev, end)
        X = torch.tensor(sj.pos51[edges], dtype=torch.float32)
        ev, ms = sj.window_events(m_prev, end)
        pb = M.make_problem(g, X, sj.betas, sj.K, ev, ms, m_prev, end, "palm")
        t0 = time.time()
        res["spectrum"][str(w)] = spectrum_window(pb, M.knot_count(w), ls)
        res["spectrum"][str(w)]["secs"] = time.time() - t0
    for w in task.get("map_windows", MAP_WINDOWS):
        m_prev = end - w
        edges = M.window_edges(m_prev, end)
        gt = sj.pos51[edges]
        ev, ms = sj.window_events(m_prev, end)
        nk = M.knot_count(w)
        for tier in task.get("map_tiers", MAP_TIERS):
            for ax in M.AXES:
                sg = 1 if (M.seed_of("sign", seq, end, tier, ax) & 1) else -1
                for mode in task.get("map_modes", ("const", "ar") if tier == 5.0 and w == 1000 else ("const",)):
                    nrng = np.random.default_rng(M.seed_of(seq, end, tier, ax, mode, "noise"))
                    if mode == "const":
                        d = nrng.standard_normal(51) * M.sig51()
                        d[3:6] = 0.0
                        nz = np.broadcast_to(d, (len(edges), 51)).copy()
                    else:
                        nz = M.noise_at_edges(edges, end, M.noise_knots(end, end - 1000, nrng, rho=M.NOISE_RHO))
                    b = M.axis_vec(ax, sg, tier)
                    Xp = M.inject(gt, b, nz)
                    btrue = M.true_offset(gt, Xp)
                    pb = M.make_problem(g, Xp, sj.betas, sj.K, ev, ms, m_prev, end, "palm")
                    variants = [("real", "full")] if mode == "ar" else [("real", "full"), ("shuffle", "full"), ("noprior", "temporal")]
                    for vname, reg in variants:
                        pbv = pb
                        if vname == "shuffle":
                            pbv = M.rewindow(pb, ev, M.shuffle_times(ms, np.random.default_rng(M.seed_of(seq, end, tier, ax, w, "shuf"))), m_prev, end)
                        prior = M.prior_precision(nk, ls, reg)
                        theta, secs = map_joint(pbv, nk, prior)
                        rec = M.recovery(theta[:6], btrue)
                        rec.update({"w": w, "tier": tier, "axis": ax, "mode": mode, "variant": vname, "secs": secs,
                                    "finite": bool(torch.isfinite(theta).all())})
                        res["map"].append(rec)
    return res


# ----------------------------------------------------------------------------------------------- summaries
def summarize(results: List[dict], select: Sequence[str]) -> dict:
    out = {"n_anchors": len(results), "spectrum": {}, "map": {}}
    sets = {"SELECT": [r for r in results if IB.subject_set(r["seq"], select) == "SELECT"],
            "FIT7": [r for r in results if IB.subject_set(r["seq"], select) == "FIT7"]}
    for S, rs in sets.items():
        out["spectrum"][S] = {}
        for w in SPEC_WINDOWS:
            ws = [r["spectrum"][str(w)] for r in rs if str(w) in r["spectrum"]]
            if not ws:
                continue
            sd = {reg: np.array([x["regimes"][reg]["sd_deg"] for x in ws]) for reg in REGIMES}
            full, loose = sd["full"], sd["loose3"]
            alpha = np.log(loose / full) / math.log(LEVEL_LOOSE)
            out["spectrum"][S][str(w)] = {
                "n": len(ws), "share_sd_le_2deg": float((full <= SD_TARGET_DEG).mean()),
                "share_sd_le_2deg_loose3": float((loose <= SD_TARGET_DEG).mean()),
                "median_sd_deg": float(np.median(full)), "p25_sd_deg": float(np.percentile(full, 25)), "p75_sd_deg": float(np.percentile(full, 75)),
                "median_sd_temporal_only_deg": float(np.median(sd["temporal"])),
                "median_sd_known_trajectory_deg": float(np.median([x["sd_known_trajectory_deg"] for x in ws])),
                "prior_dependence_exponent_median": float(np.median(alpha)),
                "rank_deficient_noprior_frac": float(np.mean([x["regimes"]["noprior"]["ratio_to_Fbb"] < RANK_RATIO_TOL for x in ws])),
                "rank_deficient_temporal_frac": float(np.mean([x["regimes"]["temporal"]["sd_deg"] >= 0.9 * math.degrees(math.radians(10.0)) for x in ws])),
                "median_events": float(np.median([x["n_events"] for x in ws])),
                "all_psd": bool(all(x["psd_min_eig"] > -1e-6 * max(1.0, abs(x["fisher_trace"])) for x in ws))}
    # MAP recovery
    for S, rs in sets.items():
        out["map"][S] = {}
        recs = [dict(m, seq=r["seq"], end=r["end"]) for r in rs for m in r["map"] if m["finite"]]
        for w in MAP_WINDOWS:
            out["map"][S][str(w)] = {}
            for tier in MAP_TIERS:
                blk = {}
                for mode in ("const", "ar"):
                    for variant in ("real", "shuffle", "noprior"):
                        x = [m for m in recs if m["w"] == w and m["tier"] == tier and m["mode"] == mode and m["variant"] == variant]
                        if not x:
                            continue
                        blk[f"{mode}/{variant}"] = {"n": len(x), "median_resid_deg": float(np.median([m["resid_deg"] for m in x])),
                                                    "median_rho": float(np.median([m["rho"] for m in x])),
                                                    "frac_resid_gt_b": float(np.mean([m["resid_deg"] > m["b_deg"] for m in x])),
                                                    "median_secs": float(np.median([m["secs"] for m in x]))}
                # real - shuffle paired difference of rho (const)
                key = lambda m: (m["seq"], m["end"], m["axis"])
                real = {key(m): m["rho"] for m in recs if m["w"] == w and m["tier"] == tier and m["mode"] == "const" and m["variant"] == "real"}
                shuf = {key(m): m["rho"] for m in recs if m["w"] == w and m["tier"] == tier and m["mode"] == "const" and m["variant"] == "shuffle"}
                ks = [k for k in real if k in shuf]
                if len(ks) > 4:
                    d = np.array([real[k] - shuf[k] for k in ks])
                    grp = np.array([f"{k[0]}|{k[1]}" for k in ks])
                    lo, hi = IB._cluster_median_ci(d, grp)
                    blk["diff_real_minus_shuffle"] = {"n": len(ks), "median": float(np.median(d)), "ci95": [lo, hi]}
                out["map"][S][str(w)][str(tier)] = blk
    return out


def decide(summary: dict) -> dict:
    """Gate (SELECT): share of 1 s windows with worst-axis bias posterior SD <= 2 deg (full regime) >= 0.5 and the MAP
    recovery residual median (5 deg, 1000 ms, real, constant-offset injection) <= 2 deg and MAP better than shuffled (CI > 0)
    and the posterior not prior-dominated (median prior-dependence exponent < 0.5);  kill: share < 0.10, or prior-dominated
    (share <= 2 deg falls below 0.10 when the level prior is 3x looser while the full regime passes the gate share), or
    the no-prior spectrum is NOT rank deficient (theory/tool failure -> tool_defect, not kill);  else descriptive."""
    sp = summary["spectrum"]["SELECT"].get("1000", {})
    mp = summary["map"]["SELECT"].get("1000", {}).get("5.0", {})
    if not sp:
        return {"verdict": "tool_defect", "reason": "no SELECT spectrum"}
    tool_ok = sp["rank_deficient_noprior_frac"] >= 0.95 and sp["all_psd"]
    if not tool_ok:
        return {"verdict": "tool_defect", "rank_deficient_noprior_frac": sp["rank_deficient_noprior_frac"], "all_psd": sp["all_psd"]}
    share = sp["share_sd_le_2deg"]
    alpha = sp["prior_dependence_exponent_median"]
    prior_dom = alpha >= 0.5
    real = mp.get("const/real", {})
    diff = mp.get("diff_real_minus_shuffle", {})
    map_resid = real.get("median_resid_deg", float("nan"))
    map_ok = bool(map_resid <= 2.0) and bool(diff.get("ci95", [float("nan")])[0] > 0)
    kill = share < 0.10 or (prior_dom and sp["share_sd_le_2deg_loose3"] < 0.10)
    if share >= 0.5 and map_ok and not prior_dom:
        verdict = "pass"
    elif kill:
        verdict = "kill"
    else:
        verdict = "descriptive"
    return {"verdict": verdict, "share_sd_le_2deg_1s": share, "share_sd_le_2deg_1s_loose3": sp["share_sd_le_2deg_loose3"],
            "prior_dependence_exponent": alpha, "prior_dominated": prior_dom, "map_resid_median_deg_5deg_1s": map_resid,
            "map_beats_shuffle": bool(diff.get("ci95", [float("nan")])[0] > 0),
            "rank_deficient_noprior_frac": sp["rank_deficient_noprior_frac"],
            "disposition": {"pass": "bias slow state may be estimated only when the Schur gate is open",
                            "kill": "bias slow state permanently frozen; segment bias = offline calibration / model-mismatch term",
                            "descriptive": "no claim"}[verdict]}
