#!/usr/bin/env python3
"""MV2 M5 probes (palm-rigid alignment + segment-level root bias PRA): entry point, smoke, decide.

    CUDA_VISIBLE_DEVICES="" python tools/mv2/m5_palm.py --smoke
    CUDA_VISIBLE_DEVICES="" python tools/mv2/m5_palm.py --out outputs/mv2 [--probe ib|schur|fipa|rca|all] [--workers 36 --threads 4]
    python tools/mv2/m5_palm.py decide --root outputs/mv2          # reads outputs/mv2/m5_*/result.json, prints the verdicts

Probes (tools/mv2/prereg_m5.md): m5_ib (IBC / IBR injection recovery), m5_schur (Schur identifiability spectrum + joint MAP),
m5_fipa (fixed-iteration static-graph palm alignment), m5_rca (root-block contraction audit of the frozen S39).  The formal
mode first verifies the MV2 freeze (it exits until `outputs/mv2/protocol.json` exists); `--smoke` prints shapes, counts,
timings and pass / fail of internal unit checks only, never a statistic of real data that could be an effect.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mv2 import common as MC  # noqa: E402
from mv2 import m5_core as M  # noqa: E402
from mv2 import m5_fipa as FP  # noqa: E402
from mv2 import m5_ib as IB  # noqa: E402
from mv2 import m5_rca as RC  # noqa: E402
from mv2 import m5_schur as SC  # noqa: E402

SCRIPTS = ["tools/mv2/m5_palm.py", "tools/mv2/m5_core.py", "tools/mv2/m5_synth.py", "tools/mv2/m5_ib.py",
           "tools/mv2/m5_schur.py", "tools/mv2/m5_fipa.py", "tools/mv2/m5_rca.py"]
PROBES = ("ib", "schur", "fipa", "rca")
FIT7_ANCHORS, SELECT_ANCHORS = 4, 10             # anchors per recording (evenly spaced eligible window ends)
FIPA_EVERY, FIPA_CHUNK = 4, 20                   # every 4th 50 ms endpoint (200 ms apart); samples per task
PLANTED_ANCHORS, PLANTED_RHO_MIN = 3, 0.5        # planted-signal tool validity: median rho of the 5 deg clean injection


# ----------------------------------------------------------------------------------------------- decide
def decide(result: dict) -> dict:
    """`result` = {'ib': summary, 'schur': summary, 'fipa': summary, 'rca': summary} (any subset).  Applies exactly the gates of
    prereg_m5.md per probe and the disposition table of plan section M5 (R3)."""
    d = {}
    if "ib" in result:
        d["ib"] = IB.decide(result["ib"])
    if "schur" in result:
        d["schur"] = SC.decide(result["schur"])
    if "fipa" in result:
        d["fipa"] = FP.decide(result["fipa"])
    if "rca" in result:
        d["rca"] = RC.decide(result["rca"])
    v = {k: x["verdict"] for k, x in d.items()}
    arms = {}
    if "ib" in v and "fipa" in v:
        arms["M1+root pathway with FIPA row (single variable)"] = bool(v["ib"] == "pass" and v["fipa"] == "pass")
    if "ib" in v and "schur" in v:
        arms["bias gate arm (Schur-gated slow state)"] = bool(v["ib"] == "pass" and v["schur"] == "pass")
    if "rca" in v:
        arms["root-block-restricted self-prev-state arm"] = bool(v["rca"] == "pass")
    d["m5"] = {"verdicts": v, "gpu_arms_released": arms,
               "any_tool_defect": any(x == "tool_defect" for x in v.values()),
               "status": ("tool_defect" if any(x == "tool_defect" for x in v.values()) else
                          ("condition_met" if any(arms.values()) else "offline_calibration_or_diagnostic_only")),
               "note": "each probe is gated on its own (SELECT); a pass releases a single-seed GPU debug arm only, never a claim"}
    return d


def load_results(root: Path) -> dict:
    out = {}
    for p in PROBES:
        f = Path(root) / f"m5_{p}" / "result.json"
        if f.exists():
            out[p] = json.loads(f.read_text())["summary"]
    return out


# ----------------------------------------------------------------------------------------------- validity checks
def planted_validity(g: M.Geo, seq: str = "ch_local", n_anchors: int = PLANTED_ANCHORS) -> dict:
    """Planted-signal check of the aligner on synthetic events drawn from the model's own mean field at the GT trajectory
    of a real FIT7 recording: median rho of 5 deg injections (3 axes x `n_anchors` moving anchors, constant nuisance) at 200
    and 1000 ms must be >= PLANTED_RHO_MIN.  No real event enters."""
    from mv2 import m5_synth as SY
    sj = M.Subject.load(seq)
    cand = sj.anchors(history_ms=1000, n=40)
    def move(e, w=200):
        a, b = sj.pos51[e - w], sj.pos51[e]
        return float(np.linalg.norm(b[3:6] - a[3:6]) + 10 * np.linalg.norm(b[:3] - a[:3]))
    anchors = sorted(cand, key=lambda e: -move(e))[:n_anchors]
    out = {"seq": seq, "anchors": [int(a) for a in anchors], "rho": {}, "secs": {}}
    ok = True
    for w in (200, 1000):
        rho = []
        t0 = time.time()
        for e in anchors:
            for ax in M.AXES:
                pb, bt = SY.planted_case(g, sj, e, w, ax, 1, 5.0, seed=M.seed_of("planted", e, ax, w), mode="const")
                rho.append(M.recovery(M.align6(pb).theta, bt)["rho"])
        out["rho"][str(w)] = [float(x) for x in rho]
        out["secs"][str(w)] = time.time() - t0
        ok &= bool(np.median(rho) >= PLANTED_RHO_MIN)
    out["median_rho"] = {w: float(np.median(v)) for w, v in out["rho"].items()}
    out["ok"] = bool(ok)
    return out


def synthetic_rca_results(n_runs=6, steps=140, lam=0.3, bias_sd=0.06, noise_sd=0.02, seed=1) -> list:
    """Synthetic closed-loop records with a known contraction and persistent bias (statistic-code test, no model)."""
    rng = np.random.default_rng(seed)
    res = []
    for rep in range(2):
        e_root, tau_root, d_root, run, first = [], [], [], [], []
        e_prev = np.zeros(3)
        for r in range(n_runs):
            b = rng.standard_normal(3) * bias_sd
            for k in range(steps):
                tau = b + rng.standard_normal(3) * noise_sd
                d = lam * e_prev if k > 0 else np.zeros(3)
                e = tau + d
                e_root.append(e), tau_root.append(tau), d_root.append(d), run.append(r), first.append(int(k == 0))
                e_prev = e
        n = len(run)
        z3, z45 = np.zeros((n, 3)), np.zeros((n, 45))
        ra_h = np.tile(np.linspace(4.0, 6.0, 8)[None], (n, 1)).astype(np.float32)
        res.append({"seq": f"syn{rep}_local", "seed": 3407, "n": n, "first": np.array(first), "run": np.array(run),
                    "e_root": np.array(e_root), "tau_root": np.array(tau_root), "d_root": np.array(d_root),
                    "e_tr": z3, "tau_tr": z3, "d_tr": z3, "e_f": z45, "tau_f": z45, "d_f": z45,
                    "ra_h": ra_h, "abs_h": ra_h + 1.0, "sh_ra": np.full(n, 6.5, np.float32), "sh_abs": np.full(n, 7.5, np.float32),
                    "e_root_sh": np.array(e_root)})
    return res


def fipa_jacobian_check(g: M.Geo) -> bool:
    """d residual / d (tau, omega) of `m5_fipa.linearize` against `torch.autograd.functional.jacobian` of the same residual with the
    association (segment, foot parameter, normal) held fixed, at a non-identity state."""
    rng = np.random.default_rng(7)
    x = torch.tensor(rng.standard_normal((1, 51)) * 0.2, dtype=torch.float32)
    x[0, :3] = torch.tensor([0.02, -0.01, 0.45])
    x[0, 3:6] = torch.tensor([0.1, -0.2, 0.15])
    V, J = g.fk(x, torch.zeros(1, 10))
    V, wr = V[0], J[0, 0]
    K = torch.tensor([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
    intr = torch.tensor([600.0, 600.0, 320.0, 240.0])
    tabs = FP.tabs_of(g)
    _, w_e, a2, b2 = FP.edge_state(V, tabs, 600.0, 600.0, 320.0, 240.0)
    j0 = rng.choice(len(w_e), size=64, p=(w_e / w_e.sum()).numpy())
    ev = (a2[j0] + 0.5 * (b2[j0] - a2[j0])).numpy() + rng.normal(0, 1.5, (64, 2))
    ev = torch.tensor(ev, dtype=torch.float32)
    Rm = FP.so3_exp(torch.tensor([0.02, 0.03, -0.01]))
    tau = torch.tensor([0.001, -0.002, 0.003])
    r, q, a, j, t_j, nj = FP.linearize(V, wr, tabs, ev, torch.ones(64), intr, Rm, tau)

    def r_of(delta):
        Rm2 = FP.so3_exp(delta[3:]) @ Rm
        Vh = wr + (V - wr) @ Rm2.T + tau + delta[:3]
        z = Vh[:, 2]
        P2 = torch.stack([600.0 * Vh[:, 0] / z + 320.0, 600.0 * Vh[:, 1] / z + 240.0], -1)
        foot = P2[tabs.e0[j]] * (1 - t_j)[:, None] + P2[tabs.e1[j]] * t_j[:, None]
        return ((ev - foot) * nj).sum(-1)
    Ja = torch.autograd.functional.jacobian(r_of, torch.zeros(6, dtype=torch.float32))
    mine = a / FP.UNITS
    return bool(float((Ja - mine).abs().max()) < 1e-2 * float(Ja.abs().max()))


def unit_tests(g: M.Geo) -> dict:
    """Synthetic / structural unit checks (no effect readings).  Returns {name: bool}."""
    out = {}
    # palm set
    out["palm_set_size_in_[100,200]"] = bool(100 <= g.n_palm <= 200)
    out["palm_edges_subset"] = bool(int(g.palm_edge.sum()) > 0 and int(g.palm_edge.sum()) < g.palm_edge.numel())
    # rigid correction about the wrist == FK with the left-multiplied root (translation untouched)
    x = torch.tensor(np.random.default_rng(0).standard_normal((1, 51)) * 0.2, dtype=torch.float32)
    betas = torch.zeros(1, 10)
    om = torch.tensor([[0.05, -0.08, 0.1]])
    V0, J0 = g.fk(x, betas)
    x2 = x.clone()
    x2[:, 3:6] = M.so3_log(M.so3_exp(om) @ M.so3_exp(x[:, 3:6]))
    V2, J2 = g.fk(x2, betas)
    Vr = M.apply_corr(V0, J0[:, 0], torch.zeros(1, 1, 3), om[None])[0]
    out["rigid_correction_equals_FK_root_left_multiply"] = bool(float((Vr - V2).abs().max()) < 1e-4 and float((J0[:, 0] - J2[:, 0]).abs().max()) < 1e-5)
    # knot interpolation reproduces constants and the constant mode aliases b
    kp = M.KnotParam(21, 5)
    th = torch.zeros(1, kp.P)
    th[0, 6 * 2 + 3:6 * 2 + 6] = 0.1                                # one knot rotated
    tau, om_ = kp.expand(th)
    out["knot_hat_partition_of_unity"] = bool(abs(float(kp.Phi.sum(1).min()) - 1) < 1e-6 and abs(float(kp.Phi.sum(1).max()) - 1) < 1e-6)
    # rotation-recovery metric
    th0 = torch.zeros(6)
    th0[3:6] = torch.tensor([0.0, 0.0, -0.08])
    r = M.recovery(th0, np.array([0.0, 0.0, 0.08]))
    out["recovery_metric_perfect"] = bool(abs(r["rho"] - 1) < 1e-6 and abs(r["slope"] - 1) < 1e-6)
    r = M.recovery(torch.tensor([0, 0, 0, 0.0, 0.0, 0.08]), np.array([0.0, 0.0, 0.08]))
    out["recovery_metric_overshoot_not_credited"] = bool(r["rho"] < 0 and r["slope"] < 0)
    th1 = torch.zeros(6)
    th1[3:6] = torch.tensor([0.0, 0.0, -0.4])
    r = M.recovery(th1, np.array([0.0, 0.0, 0.08]))
    out["recovery_metric_overshoot_penalised"] = bool(r["rho"] < 0 and r["slope"] > 1)
    # time-shuffled null keeps the multiset of (x, y, p) and of ms
    ms = np.arange(1000)
    sh = M.shuffle_times(ms, np.random.default_rng(1))
    out["shuffle_is_permutation"] = bool(np.array_equal(np.sort(sh), ms) and not np.array_equal(sh, ms))
    # priors: PSD, noprior aliasing in the algebra (S_b = 0 for F built as a pure function of b + s), level prior breaks it
    nk = 4
    ls = {"vel_rot": 0.05, "acc_rot": 0.05, "vel_trans": 0.01, "acc_trans": 0.01, "delta_ms": 100}
    J = torch.randn(6, 200, dtype=torch.float64)
    Jp = torch.zeros(6 * (1 + nk), 200 * nk, dtype=torch.float64)
    for k in range(nk):                                             # time bin group k sees correction b + s_k
        Jp[0:6, 200 * k:200 * (k + 1)] = J
        Jp[6 * (1 + k):6 * (2 + k), 200 * k:200 * (k + 1)] = J
    F = Jp @ Jp.T / 50.0
    Sb0 = M.schur_b(F, M.prior_precision(nk, ls, "noprior"))
    out["schur_noprior_is_zero"] = bool(float(torch.linalg.norm(Sb0)) / float(torch.linalg.norm(F[:6, :6])) < 1e-6)
    sd = {r_: M.posterior_sd_deg(M.schur_b(F, M.prior_precision(nk, ls, r_))) for r_ in ("temporal", "full")}
    sd3 = M.posterior_sd_deg(M.schur_b(F, M.prior_precision(nk, ls, "full", level_mult=3.0)))
    out["schur_temporal_only_returns_bias_prior"] = bool(abs(sd["temporal"] - 10.0) < 1e-3)
    out["schur_level_prior_finite_and_monotone"] = bool(math.isfinite(sd["full"]) and sd["full"] < sd["temporal"] and sd3 >= sd["full"])
    # RCA statistics on synthetic closed-loop records with known lambda and bias
    syn = synthetic_rca_results(lam=0.3)
    s = RC.summarize_set(syn, n_boot=200)
    out["rca_lambda_recovered"] = bool(abs(s["lambda"]["root"]["value"] - 0.3) < 0.02)
    out["rca_bias_share_above_null"] = bool(s["bias_share"]["20"]["share"] > 3 * s["bias_share"]["20"]["null_share"] and s["bias_share"]["20"]["null_share"] < 0.2)
    ph = RC.shapley_all(syn[0]["ra_h"])
    gap = syn[0]["ra_h"][:, 7] - syn[0]["ra_h"][:, 0]
    out["rca_shapley_sums_to_gap"] = bool(np.allclose(sum(ph.values()), gap, atol=1e-5))
    syn_hi = synthetic_rca_results(lam=1.05, steps=40)
    out["rca_lambda_gt1_detected"] = bool(RC.summarize_set(syn_hi, n_boot=200)["lambda"]["root"]["ci95"][0] > 1.0)
    # FIPA analytic Jacobian (rigid left-increment projective Jacobian) == autograd of the point-to-line residual at fixed association
    out["fipa_jacobian_matches_autograd"] = fipa_jacobian_check(g)
    # FIPA op counter self-test: a deliberately dynamic function must be flagged, the static graph must not
    def dyn(x):
        return x[x > 0].sum()
    tr = [FP.trace_ops(dyn, torch.randn(50) + k) for k in (0, 1, 2)]
    out["fipa_counter_flags_dynamic_op"] = bool(FP.count_dynamic_shape_ops(tr)["count"] > 0)
    return out


def decide_tests() -> dict:
    """The four gates on hand-made summaries: every verdict branch (pass / kill / descriptive / tool_defect) must be reachable."""
    out = {}
    gi = lambda med, p90, lo: {"gate_inputs": {"median_rho_5deg_1000ms": med, "p90_time_ms_5deg": p90, "diff_ci95_5deg_1000ms": [lo, lo + 0.2],
                                               "median_diff_vs_shuffle_5deg_1000ms": lo + 0.1, "median_rho_5deg_500ms": med, "n_cases_5deg": 100},
                               "n_nonfinite": 0, "planted_ok": True}
    out["ib_pass"] = IB.decide(gi(0.6, 500, 0.1))["verdict"] == "pass"
    out["ib_kill_low_median"] = IB.decide(gi(0.2, 500, 0.1))["verdict"] == "kill"
    out["ib_kill_p90"] = IB.decide(gi(0.6, math.inf, 0.1))["verdict"] == "kill"
    out["ib_kill_null"] = IB.decide(gi(0.6, 500, -0.05))["verdict"] == "kill"
    out["ib_descriptive"] = IB.decide(gi(0.4, 500, 0.1))["verdict"] == "descriptive"
    bad = gi(0.6, 500, 0.1)
    bad["planted_ok"] = False
    out["ib_tool_defect"] = IB.decide(bad)["verdict"] == "tool_defect"

    def sc(share, alpha, resid, ci_lo, loose=0.5, rank=1.0):
        return {"spectrum": {"SELECT": {"1000": {"share_sd_le_2deg": share, "share_sd_le_2deg_loose3": loose,
                                                  "prior_dependence_exponent_median": alpha, "rank_deficient_noprior_frac": rank, "all_psd": True}}},
                "map": {"SELECT": {"1000": {"5.0": {"const/real": {"median_resid_deg": resid}, "diff_real_minus_shuffle": {"ci95": [ci_lo, ci_lo + 0.2]}}}}}}
    out["schur_pass"] = SC.decide(sc(0.7, 0.2, 1.5, 0.1))["verdict"] == "pass"
    out["schur_kill_share"] = SC.decide(sc(0.05, 0.2, 1.5, 0.1))["verdict"] == "kill"
    out["schur_kill_prior_dominated"] = SC.decide(sc(0.7, 0.8, 1.5, 0.1, loose=0.05))["verdict"] == "kill"
    out["schur_descriptive"] = SC.decide(sc(0.3, 0.2, 1.5, 0.1))["verdict"] == "descriptive"
    out["schur_tool_defect"] = SC.decide(sc(0.7, 0.2, 1.5, 0.1, rank=0.5))["verdict"] == "tool_defect"

    def fp(cell_rot, k8_rot, noop=5.0, ref=6.0, ops=0, ok=True):
        cells = {f"K{k}_C{C}": {"rot_deg_median": cell_rot if k <= 4 else k8_rot} for k in FP.KS for C in FP.CAPACITIES}
        return {"cells": cells, "s39_tf": {"mean_of_seed_medians_rot_deg": ref}, "no_op": {"rot_deg_median": noop},
                "ops": {"count": ops, "counter_self_test": True}, "synthetic": {"ok": ok}}
    out["fipa_pass"] = FP.decide(fp(3.5, 3.0))["verdict"] == "pass"
    out["fipa_kill_k8"] = FP.decide(fp(7.0, 7.0))["verdict"] == "kill"
    out["fipa_kill_ops"] = FP.decide(fp(3.5, 3.0, ops=2))["verdict"] == "kill"
    out["fipa_descriptive"] = FP.decide(fp(5.5, 4.5))["verdict"] == "descriptive"
    out["fipa_no_vacuous_pass"] = FP.decide(fp(5.0, 5.0, noop=5.0, ref=9.0))["verdict"] != "pass"
    out["fipa_tool_defect"] = FP.decide(fp(3.5, 3.0, ok=False))["verdict"] == "tool_defect"

    def rc(share, lam_ub, bias, gap=1.0):
        return {"SELECT": {"exposure_gap_ra": {"gap_mm": gap, "root": {"share": share, "ci95": [share - .1, share + .1]}},
                           "lambda": {"root": {"value": lam_ub - 0.1, "ci95": [lam_ub - 0.2, lam_ub]}},
                           "bias_share": {"20": {"share": bias, "null_share": 0.05}}}, "validity": {"ok": True}}
    out["rca_pass"] = RC.decide(rc(0.7, 0.5, 0.6))["verdict"] == "pass"
    out["rca_kill_share"] = RC.decide(rc(0.3, 0.5, 0.6))["verdict"] == "kill"
    out["rca_kill_lambda"] = RC.decide(rc(0.7, 1.1, 0.6))["verdict"] == "kill"
    out["rca_descriptive_bias"] = RC.decide(rc(0.7, 0.5, 0.3))["verdict"] == "descriptive"
    out["rca_degenerate_gap"] = RC.decide(rc(0.7, 0.5, 0.6, gap=0.01))["verdict"] == "descriptive"
    full = decide({"ib": gi(0.6, 500, 0.1), "schur": sc(0.7, 0.2, 1.5, 0.1), "fipa": fp(3.5, 3.0), "rca": rc(0.7, 0.5, 0.6)})
    out["decide_all_pass_releases_three_arms"] = bool(all(full["m5"]["gpu_arms_released"].values()) and len(full["m5"]["gpu_arms_released"]) == 3)
    return out


def pool_roundtrip() -> dict:
    """spawn-pool round trip of every worker entry on tiny FIT7 tasks: shapes / finiteness / JSON round trip only."""
    out = {}
    seq = "ch_local"
    sj = M.Subject.load(seq)
    anc = sj.anchors(n=2)
    ls = M.label_stats([sj], delta_ms=100)
    pool = _pool(2, 2)
    try:
        recs = _flat(pool.imap_unordered(IB.run_anchor, [{"seq": seq, "end": int(e), "tiers": (5.0,), "windows": (50,)} for e in anc]))
        out["pool_ib_records"] = len(recs)
        res = list(pool.imap_unordered(SC.run_anchor, [{"seq": seq, "end": int(e), "label_stats": ls, "spec_windows": (50,), "map_windows": ()} for e in anc]))
        out["pool_schur_results"] = len(res)
        ends = MC.SeqSamples(seq, every=40).ends[:2]
        frecs = _flat(pool.imap_unordered(FP.run_samples, [{"seq": seq, "ends": [int(e) for e in ends], "capacities": (256,)}]))
        out["pool_fipa_records"] = len(frecs)
        s39 = pool.starmap(FP.s39_tf_root_errors, [(seq, 80, [3407])])
        out["pool_s39_tf_samples"] = int(len(s39[0][3407]["rot_deg"]))
        rr = list(pool.imap_unordered(RC.run_sequence, [{"seq": seq, "seed": 3407, "max_steps": 3}]))
        out["pool_rca_steps"] = int(rr[0]["n"])
    finally:
        pool.close()
        pool.join()
    s1 = IB.summarize(recs, ("ycy", "ylf"))
    s2 = SC.summarize(res, ("ycy", "ylf"))
    sets = RC.collect(rr, ("ycy", "ylf"))
    s3 = {k: RC.summarize_set(v, n_boot=50) for k, v in sets.items()}
    ok = True
    for obj in (s1, s2, s3):
        try:
            json.loads(json.dumps(obj, default=MC._np))
        except Exception:
            ok = False
    out["json_roundtrip_ok"] = ok
    return out


# ----------------------------------------------------------------------------------------------- smoke
def smoke(args) -> int:
    t_all = time.time()
    outdir = REPO / "outputs" / "mv2" / "smoke" / "m5_palm"
    outdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    rep = {"checks": {}, "timings_s": {}, "shapes": {}, "counts": {}}
    g = M.Geo()
    rep["counts"]["palm_vertices"] = g.n_palm
    rep["counts"]["palm_edges"] = int(g.palm_edge.sum())
    rep["counts"]["mesh_edges"] = int(g.palm_edge.numel())
    t0 = time.time()
    rep["checks"].update(unit_tests(g))
    rep["timings_s"]["unit_tests"] = time.time() - t0

    # ---- real slice (FIT7 ch_local, one anchor): full code path of IB / Schur, finite + shapes + timings only
    seq = "ch_local"
    sj = M.Subject.load(seq)
    anc = sj.anchors(n=1)
    rep["counts"]["anchor_eligible_ends_ch_local_first"] = int(anc[0]) if anc else -1
    end = int(anc[0])
    ls = M.label_stats([sj], delta_ms=100)
    rep["checks"]["label_stats_finite_positive"] = bool(all(math.isfinite(v) and v > 0 for k, v in ls.items() if k != "delta_ms"))
    t0 = time.time()
    recs = IB.run_anchor({"seq": seq, "end": end, "tiers": (5.0,), "windows": (50, 200)})
    rep["timings_s"]["ib_anchor_tiers5_windows50_200"] = time.time() - t0
    rep["counts"]["ib_records_smoke"] = len(recs)
    rep["checks"]["ib_all_finite"] = bool(all(r["finite"] and math.isfinite(r["rho"]) or r["tier"] == 0 for r in recs))
    rep["timings_s"]["ib_median_solve_by_window"] = {str(w): float(np.median([r["secs"] for r in recs if r["window"] == w])) for w in (50, 200) if any(r["window"] == w for r in recs)}
    summ = IB.summarize(recs, ("ycy", "ylf"))                        # exercises the summariser on FIT7 only; prints nothing about it
    rep["checks"]["ib_summarize_runs"] = bool("primary" in summ)
    t0 = time.time()
    res = SC.run_anchor({"seq": seq, "end": end, "label_stats": ls, "spec_windows": (50, 200), "map_windows": (200,), "map_tiers": (5.0,), "map_modes": ("const",)})
    rep["timings_s"]["schur_anchor_spec50_200_map200"] = time.time() - t0
    rep["counts"]["schur_map_records_smoke"] = len(res["map"])
    sp = res["spectrum"]
    rep["checks"]["schur_spectrum_finite_psd"] = bool(all(math.isfinite(v["regimes"]["full"]["sd_deg"]) and v["psd_min_eig"] > -1e-6 * abs(v["fisher_trace"]) for v in sp.values()))
    rep["checks"]["schur_noprior_rank_deficient_on_real_window"] = bool(all(v["regimes"]["noprior"]["ratio_to_Fbb"] < SC.RANK_RATIO_TOL for v in sp.values()))
    rep["checks"]["schur_level_prior_monotone_on_real_window"] = bool(all(v["regimes"]["loose3"]["sd_deg"] >= v["regimes"]["full"]["sd_deg"] - 1e-9 for v in sp.values()))
    rep["checks"]["schur_map_finite"] = bool(all(m["finite"] for m in res["map"]))
    rep["timings_s"]["schur_fisher_secs_by_window"] = {w: float(v["secs"]) for w, v in sp.items()}
    summ = SC.summarize([res], ("ycy", "ylf"))
    rep["checks"]["schur_summarize_runs"] = bool("spectrum" in summ)

    # ---- planted-signal tool validity (synthetic events) -- pass / fail only
    t0 = time.time()
    pv = planted_validity(g, "ch_local", n_anchors=3)
    rep["checks"]["planted_signal_recovery_ok"] = pv["ok"]
    rep["timings_s"]["planted_signal"] = time.time() - t0

    # ---- FIPA on a real slice + synthetic validity + op counter
    t0 = time.time()
    frecs = FP.run_samples({"seq": seq, "ends": [end], "capacities": (256, 512, 1024)})
    rep["timings_s"]["fipa_one_sample_all_cases_caps"] = time.time() - t0
    rep["counts"]["fipa_records_smoke"] = len(frecs)
    rep["checks"]["fipa_all_finite"] = bool(all(math.isfinite(r[f"rot_{k}"]) and math.isfinite(r[f"trans_{k}"]) for r in frecs for k in FP.KS))
    syn = fipa_synthetic(g, sj, end)
    rep["checks"]["fipa_planted_recovery_ok"] = bool(syn["ok"])
    ops = fipa_ops(g, sj, seq, [end - 50 * k for k in range(0, 6)])
    rep["counts"]["fipa_aten_ops_per_forward"] = ops["n_ops_per_forward"]
    rep["counts"]["fipa_data_dependent_shape_ops"] = ops["count"]
    rep["checks"]["fipa_static_graph_zero_dynamic_ops"] = bool(ops["count"] == 0)
    rep["checks"]["fipa_same_op_sequence_across_inputs"] = bool(ops["same_op_sequence"])

    # ---- RCA: tiny closed loop of the frozen S39 (3 steps of one run), shapes / finite / TF == single-step forward
    t0 = time.time()
    r = RC.run_sequence({"seq": seq, "seed": 3407, "max_steps": 3})
    rep["timings_s"]["rca_3steps_one_run_x_runs"] = time.time() - t0
    rep["counts"]["rca_steps_smoke"] = int(r["n"])
    rep["shapes"]["rca_arrays"] = {k: list(np.shape(v)) for k, v in r.items() if isinstance(v, np.ndarray)}
    rep["checks"]["rca_finite"] = bool(r["finite"])
    rep["checks"]["rca_tf_equals_single_step_forward"] = rca_tf_check(seq)
    # ---- decide() branches on hand-made summaries; spawn-pool round trip of every worker entry (no statistic printed)
    dt = decide_tests()
    rep["checks"].update({f"decide_{k}": v for k, v in dt.items()})
    t0 = time.time()
    pr = pool_roundtrip()
    rep["counts"].update(pr)
    rep["checks"]["pool_roundtrip_json_ok"] = bool(pr["json_roundtrip_ok"] and pr["pool_ib_records"] > 0 and pr["pool_rca_steps"] > 0)
    rep["timings_s"]["pool_roundtrip"] = time.time() - t0
    rep["timings_s"]["total"] = time.time() - t_all
    (outdir / "smoke.json").write_text(json.dumps(rep, indent=1, sort_keys=True, default=MC._np))
    fails = [k for k, v in rep["checks"].items() if not v]
    print(json.dumps({"checks_pass": len(rep["checks"]) - len(fails), "checks_total": len(rep["checks"]), "failed": fails,
                      "counts": rep["counts"], "timings_s": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rep["timings_s"].items()}},
                     indent=1, default=MC._np))
    return 0 if not fails else 1


def rca_tf_check(seq: str) -> bool:
    """The TF output used by the audit equals the plain single-step E3 forward of the frozen model (batch-3 vs batch-1 rounding allowed)."""
    ss = MC.SeqSamples(seq, every=8, max_n=2)
    model, _ = MC.load_s39(3407, torch.device("cpu"))
    model.set_hand_context(torch.tensor(ss.betas).view(1, -1), torch.tensor(ss.K).view(1, 3, 3))
    X = torch.from_numpy(ss.lnes(np.arange(len(ss))))
    P = torch.from_numpy(ss.prev)
    with torch.no_grad():
        a = model(X, P)
        b = torch.cat([model(X[i:i + 1], P[i:i + 1]) for i in range(len(ss))])
    return bool(float((a - b).abs().max()) < 1e-3)


def fipa_rim_events(g: M.Geo, sg, K, C: int, rng, frac_out: float = 0.25) -> np.ndarray:
    """Synthetic events for the static-graph aligner: points on the soft-silhouette palm edges of the reference pose with
    2 source-px normal jitter, plus `frac_out` uniform outliers in the palm box."""
    tabs = FP.tabs_of(g)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    _, w_e, a2, b2 = FP.edge_state(sg.V, tabs, fx, fy, cx, cy)
    w = w_e.numpy()
    ab = (b2 - a2).numpy()
    nrm = np.stack([-ab[:, 1], ab[:, 0]], -1)
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-6)
    n_in = int(C * (1 - frac_out))
    j = rng.choice(len(w), size=n_in, p=w / w.sum())
    pts = a2.numpy()[j] + rng.uniform(0, 1, (n_in, 1)) * ab[j] + nrm[j] * rng.normal(0, 2.0, (n_in, 1))
    uv = SB_project(sg.V[g.palm_idx], K).numpy()
    lo, hi = uv.min(0) - 20, uv.max(0) + 20
    return np.concatenate([pts, rng.uniform(lo, hi, (C - n_in, 2))]).astype(np.float32)


def fipa_synthetic(g: M.Geo, sj: M.Subject, end: int, n: int = 12) -> dict:
    """Planted-signal check of the static-graph aligner: synthetic events on the palm silhouette of the reference pose (+ 25 %
    outliers), injected 5 deg about z (in-plane roll) and 10 mm in x; the median residual at K = 4 / C = 512 must be <= 0.5 x
    the injection (rotation: 3.0 deg; translation: 5 mm)."""
    rng = np.random.default_rng(M.SEED)
    sg = FP.SampleGeom(g, sj, end)
    tabs = FP.tabs_of(g)
    K = sj.K
    intr = torch.tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=torch.float32)
    resid_rot, resid_tr = [], []
    for i in range(n):
        C = 512
        xy = fipa_rim_events(g, sg, K, C, rng)
        for kind in ("rot", "trans"):
            om0 = torch.tensor([0, 0, math.radians(5.0)], dtype=torch.float32) if kind == "rot" else torch.zeros(3)
            tau0 = torch.zeros(3) if kind == "rot" else torch.tensor([0.01, 0, 0], dtype=torch.float32)
            V0, wr0, R0 = FP.perturb(sg.V, sg.wr, om0, tau0)
            it = FP.fipa_forward(V0, wr0, tabs, torch.from_numpy(xy), torch.ones(C), intr, iters=4)
            Rm, tau = it[3]
            if kind == "rot":
                resid_rot.append(float(torch.rad2deg(FP.so3_log(Rm @ R0).norm())))
            else:
                resid_tr.append(float((tau0 + tau).norm() * 1000))
    ok = bool(np.median(resid_rot) <= 3.0 and np.median(resid_tr) <= 5.0)
    return {"ok": ok, "median_resid_rot_deg": float(np.median(resid_rot)), "median_resid_trans_mm": float(np.median(resid_tr))}


def SB_project(V, K):
    from semkine import sba as SB
    return SB.project_src(V, K)


def fipa_ops(g: M.Geo, sj: M.Subject, seq: str, ends) -> dict:
    """Op trace of one FIPA forward (K = 4, C = 512) on real windows of different event counts; counts dynamic-shape ops."""
    K = sj.K
    tabs = FP.tabs_of(g)
    intr = torch.tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=torch.float32)
    traces = []
    for e in ends:
        sg = FP.SampleGeom(g, sj, int(e))
        evs = FP.window_events_src(sj, int(e))
        om0 = torch.tensor([0.05, -0.04, 0.03], dtype=torch.float32)
        V0, wr0, R0 = FP.perturb(sg.V, sg.wr, om0, torch.zeros(3))
        xy, valid, _ = FP.event_bucket(evs, SB_project(V0[g.palm_idx], K).numpy(), 512)
        traces.append(FP.trace_ops(FP.fipa_forward, V0, wr0, tabs, torch.from_numpy(xy), torch.from_numpy(valid), intr, 4))
    out = FP.count_dynamic_shape_ops(traces)
    st = [FP.trace_ops(lambda x: x[x > 0].sum(), torch.randn(50) + k) for k in (0, 1, 2)]
    out["counter_self_test"] = bool(FP.count_dynamic_shape_ops(st)["count"] > 0)
    return out


# ----------------------------------------------------------------------------------------------- formal run
def _pool(workers: int, threads: int):
    return mp.get_context("spawn").Pool(workers, initializer=IB._worker_init, initargs=(threads,))


def _flat(it):
    out = []
    for x in it:
        out += x if isinstance(x, list) else [x]
    return out


def anchors_for(seqs, n):
    return {s: M.Subject.load(s).anchors(history_ms=1000, n=n) for s in seqs}


def write(out_root: Path, probe: str, summary: dict, extra: dict, t0: float, argv_extra: dict):
    d = out_root / f"m5_{probe}"
    MC.save_json(d / "result.json", {"summary": summary, "extra": extra, "wall_s": time.time() - t0, **argv_extra})
    MC.receipt(d, {"probe": f"m5_{probe}", **argv_extra}, SCRIPTS)


def run_formal(args) -> int:
    dev = bool(args.dev_fast)
    out_root = Path(args.out)
    if dev:
        # pipeline test only: no freeze check, tiny tasks, FIT7 subjects stand in for SELECT, refuses the official output dir
        if out_root.resolve() == (REPO / "outputs" / "mv2").resolve():
            raise SystemExit("--dev-fast must not write to outputs/mv2")
        proto = {}
        select = tuple(args.dev_select.split(","))
        fit7 = tuple(x for x in MC.FIT7 if x not in select)[:1]
    else:
        proto = MC.verify_protocol()
        select, fit7 = MC.SELECT, MC.FIT7
    probes = PROBES if args.probe == "all" else (args.probe,)
    fit_seqs, sel_seqs = MC.base_seqs(fit7), MC.base_seqs(select)
    ib_extra = {"windows": (50, 500, 1000), "tiers": (5.0,)} if dev else {}
    sc_extra = {"spec_windows": (50, 200, 1000), "map_windows": (200, 1000), "map_tiers": (5.0,), "map_modes": ("const",)} if dev else {}
    every, chunk = (FIPA_EVERY * 20, 3) if dev else (FIPA_EVERY, FIPA_CHUNK)
    rca_extra = {"max_steps": 3} if dev else {}
    n_boot = 100 if dev else RC.N_BOOT
    info = {"workers": args.workers, "threads": args.threads, "fit7_anchors": args.fit_anchors, "select_anchors": args.select_anchors,
            "protocol_sha": proto.get("prereg_sha256"), "dev_fast": dev, "select": list(select), "fit7": list(fit7)}
    pool = _pool(args.workers, args.threads)
    g = M.Geo()
    torch.set_num_threads(args.threads)
    anc = {}
    if "ib" in probes or "schur" in probes:
        anc = {**anchors_for(sel_seqs, args.select_anchors), **anchors_for(fit_seqs, args.fit_anchors)}
        info["anchors"] = {s: [int(e) for e in v] for s, v in anc.items()}
        order = sorted(anc, key=lambda s: (s.split("_")[0] not in select, not s.endswith("global")))     # costly tasks first
    if "ib" in probes:
        t0 = time.time()
        pv = planted_validity(g)
        tasks = [{"seq": s, "end": int(e), **ib_extra} for s in order for e in anc[s]]
        recs = _flat(pool.imap_unordered(IB.run_anchor, tasks, chunksize=1))
        summ = IB.summarize(recs, select)
        summ["planted_ok"], summ["planted"] = pv["ok"], pv
        write(out_root, "ib", summ, {"records": recs}, t0, info)
        print("ib", "done (dev)" if dev else IB.decide(summ)["verdict"], flush=True)
    if "schur" in probes:
        t0 = time.time()
        ls = M.label_stats([M.Subject.load(s) for s in fit_seqs])
        tasks = [{"seq": s, "end": int(e), "label_stats": ls, **(sc_extra if (s.split("_")[0] in select or dev) else {"map_windows": ()})}
                 for s in order for e in anc[s]]
        results = list(pool.imap_unordered(SC.run_anchor, tasks, chunksize=1))
        summ = SC.summarize(results, select)
        summ["label_stats"] = ls
        write(out_root, "schur", summ, {"results": results}, t0, info)
        print("schur", "done (dev)" if dev else SC.decide(summ)["verdict"], flush=True)
    if "fipa" in probes:
        t0 = time.time()
        tasks = []
        for s in sel_seqs:
            ends = MC.SeqSamples(s, every=every).ends
            tasks += [{"seq": s, "ends": [int(e) for e in ends[i:i + chunk]]} for i in range(0, len(ends), chunk)]
        recs = _flat(pool.imap_unordered(FP.run_samples, tasks, chunksize=1))
        s39 = {}
        parts = pool.starmap(FP.s39_tf_root_errors, [(s, every, list(MC.SEEDS)) for s in sel_seqs])
        for seed in MC.SEEDS:
            s39[seed] = {"rot_deg": np.concatenate([p[seed]["rot_deg"] for p in parts]), "trans_mm": np.concatenate([p[seed]["trans_mm"] for p in parts])}
        sj = M.Subject.load(sel_seqs[0])
        ops = fipa_ops(g, sj, sel_seqs[0], [int(e) for e in MC.SeqSamples(sel_seqs[0], every=every).ends[:8]])
        syn = fipa_synthetic(g, M.Subject.load("ch_local"), M.Subject.load("ch_local").anchors(n=1)[0])
        summ = FP.summarize(recs, s39, ops, syn)
        write(out_root, "fipa", summ, {"records": recs}, t0, info)
        print("fipa", "done (dev)" if dev else FP.decide(summ)["verdict"], flush=True)
    if "rca" in probes:
        t0 = time.time()
        tasks = [{"seq": s, "seed": sd, **rca_extra} for s in fit_seqs + sel_seqs for sd in MC.SEEDS]
        results = list(pool.imap_unordered(RC.run_sequence, tasks, chunksize=1))
        adir = out_root / "m5_rca" / "arrays"
        adir.mkdir(parents=True, exist_ok=True)
        for r in results:
            if r.get("n"):
                np.savez_compressed(adir / f"{r['seq']}_{r['seed']}.npz", **{k: v for k, v in r.items() if isinstance(v, np.ndarray)})
        sets = RC.collect(results, select)
        summ = {k: RC.summarize_set(v, n_boot=n_boot) for k, v in sets.items()}
        summ["per_seed_select"] = {str(sd): RC.summarize_set([r for r in sets["SELECT"] if r["seed"] == sd], n_boot=n_boot) for sd in MC.SEEDS}
        syn = RC.summarize_set(synthetic_rca_results(lam=0.3), n_boot=200)
        summ["validity"] = {"ok": bool(abs(syn["lambda"]["root"]["value"] - 0.3) < 0.02 and all(r["finite"] for r in results if r.get("n"))),
                            "synthetic_lambda": syn["lambda"]["root"]["value"]}
        write(out_root, "rca", summ, {"n_results": len(results)}, t0, info)
        print("rca", "done (dev)" if dev else RC.decide(summ)["verdict"], flush=True)
    pool.close()
    pool.join()
    if dev:
        dd = decide(load_results(out_root))                         # exercised, not printed (FIT7 stands in for SELECT)
        print("dev-fast pipeline finished; verdict dict keys:", sorted(dd))
    else:
        print(json.dumps(decide(load_results(out_root)), indent=1, default=MC._np))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "decide"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(REPO / "outputs" / "mv2"))
    ap.add_argument("--root", default=str(REPO / "outputs" / "mv2"))
    ap.add_argument("--probe", default="all", choices=("all",) + PROBES)
    ap.add_argument("--workers", type=int, default=36)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--select-anchors", type=int, default=SELECT_ANCHORS, dest="select_anchors")
    ap.add_argument("--fit-anchors", type=int, default=FIT7_ANCHORS, dest="fit_anchors")
    ap.add_argument("--dev-fast", action="store_true", dest="dev_fast", help="pipeline test: tiny tasks, no freeze check, FIT7 stands in for SELECT; never writes to outputs/mv2")
    ap.add_argument("--dev-select", default="lfz", dest="dev_select")
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) not in ("",):
        raise SystemExit('M5 probes are CPU only: run with CUDA_VISIBLE_DEVICES=""')
    if args.cmd == "decide":
        print(json.dumps(decide(load_results(Path(args.root))), indent=1, default=MC._np))
        return 0
    return smoke(args) if args.smoke else run_formal(args)


if __name__ == "__main__":
    sys.exit(main())
