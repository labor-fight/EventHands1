#!/usr/bin/env python3
"""Apply the registered K-star gates (docs/EXPERIMENT_RECORDS.md#kstar-prereg-20261025) to a stage's result.

The constants below are the pre-registered thresholds; they are frozen with this file (outputs/kstar/protocol.json).
Each stage writes outputs/kstar/<stage>/decision.json; the next stage reads the previous decisions and refuses to run on
a stopped ladder. A control that misbehaves makes the gate a tool defect (descriptive, one redesign under a new
pre-registration), never a pass.

    python tools/kstar/decide.py ks1 | ks2 | ks3 | ks4 | ks5
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kstar import common as KC  # noqa: E402

DIRS = {"ks0": "s0_aa", "ks1": "s1_fusion_ceiling", "ks2": "s2_stale", "ks3": "s3_corr_rstar", "ks3v": "s3_v0",
        "ks4": "s4_pafit", "ks4c": "s4_capture", "ks5": "s5_power"}

# KS1 fusion ceiling (zgz S39 archives; one-sided kills only)
KS1_ORACLE_KILL_MM = 0.6          # oracle increments: improvement < 0.6 mm in BOTH domains -> terminate accuracy line
KS1_RSTAR_KAPPA = "0.87"          # R_art-equivalent level of a measurement at r*
KS1_RSTAR_KILL_MM = 1.1           # zgz_local improvement at kappa 0.87 < 1.1 mm -> no fusion / no GPU
KS1_CV_FRACTION = 0.5             # run-held-out improvement < half the in-sample one -> tuning optimism -> same kill
KS1_POS_MIN_MM = 2.0              # positive control (white-noise anchor) must improve >= 2 mm (tool validity)
KS1_NEG_MAX_MM = 0.2              # negative control (shuffled increments) must improve < 0.2 mm (tool validity)
# KS2 staleness / persistence (SELECT local, each subject; art-only target, lag-2 persistence removed)
KS2_RHO_STALE_KILL = 0.70         # r(R1) / r(R0) >= 0.70 -> extrapolator
KS2_PARTIAL_KILL = 0.45           # partial r < 0.45 -> not a measurement beyond persistence
KS2_RHO_SHUFFLE_KILL = 0.50       # r(R3) / r(R0) >= 0.50 -> the head does not use event timing
# KS3 correlated r* (FIT + SELECT local base sequences, measured residual profile, independent-weight solver)
KS3_DEAD = 0.95                   # median r*_E3 > 0.95 -> local single-step route dead
KS3_GLOBAL_ONLY = 0.90            # 0.90 < r*_E3 <= 0.95 -> no local GPU work
KS3_V0_TOL = 0.02                 # zgz rho = 0 must reproduce the ledger's r* 0.825 within 0.02
# KS4 PA-fit (SELECT local, each subject, 14 mm image-plane offset)
KS4_BOFF_MIN = 0.30               # innovation readout slope on the offset < 0.30 -> no position information
KS4_DIFF_DROP = 0.10              # differential r drop > 0.10 vs clean -> capture not recoverable by the readout
# KS5 power
KS5_POWER_FACTOR = 1.5            # achievable gain < 1.5 x MDE (3 vs 3 seeds) -> GPU stage single-seed descriptive


def load(stage):
    p = KC.OUT / DIRS[stage] / "result.json"
    if not p.exists():
        raise SystemExit(f"{p} missing")
    return json.loads(p.read_text())


def prior(stage):
    p = KC.OUT / DIRS[stage] / "decision.json"
    return json.loads(p.read_text()) if p.exists() else None


def write(stage, dec):
    KC.save_json(KC.OUT / DIRS[stage] / "decision.json", dec)
    print(json.dumps(dec, indent=1, ensure_ascii=False))


def ks1():
    r = load("ks1")
    S = r["summary_delta_ra"]
    per = r["per"]
    v = r["validity"]
    pos_ok = all(e["positive_white_anchor"]["delta_ra"] <= -KS1_POS_MIN_MM for e in per.values())
    neg_ok = all(e["negative_shuffled"]["delta_ra"] > -KS1_NEG_MAX_MM for e in per.values())
    tool_ok = v["V0_ok"] and v["V1_ok"] and pos_ok and neg_ok
    oracle = {s: S[f"{s}|0.0"]["mean"] for s in KC.ZGZ}
    loc = S[f"zgz_local|{KS1_RSTAR_KAPPA}"]
    seeds_neg = all(x < 0 for x in loc["per_seed_mean"])
    cv = loc["per_seed_cv_mean"]
    cv_mean = sum(cv) / len(cv) if cv else None
    kill_a = all(oracle[s] > -KS1_ORACLE_KILL_MM for s in KC.ZGZ)
    kill_b = (loc["mean"] > -KS1_RSTAR_KILL_MM) or (not seeds_neg) or \
        (cv_mean is None) or (cv_mean > KS1_CV_FRACTION * loc["mean"])
    verdict = "tool_defect" if not tool_ok else ("kill_A" if kill_a else ("kill_B" if kill_b else "pass"))
    write("ks1", {"verdict": verdict, "tool_ok": tool_ok, "positive_ok": pos_ok, "negative_ok": neg_ok,
                  "oracle_delta_ra": oracle, "local_rstar_delta_ra": loc["mean"], "local_rstar_seeds": loc["per_seed_mean"],
                  "local_rstar_cv_mean": cv_mean, "kill_A": kill_a, "kill_B": kill_b,
                  "consequence": {"kill_A": "K-star accuracy line terminated; no GPU; KS2-KS4 run as descriptive diagnostics",
                                  "kill_B": "fusion / C4 accuracy claim deleted; no GPU; KS2-KS4 descriptive",
                                  "pass": "continue", "tool_defect": "gate descriptive; one redesign needs a new pre-registration"}[verdict]})


def ks2():
    R = load("ks2")["readings"]
    ctrl = {}
    for s in KC.SELECT_SUBJ:
        ctrl[s] = {"placebo_fails": R[f"{s}|local|PERS|art"]["partial_r"] < KS2_PARTIAL_KILL,
                   "positive_passes": R[f"{s}|local|POS|art"]["partial_r"] >= KS2_PARTIAL_KILL}
    tool_ok = all(c["placebo_fails"] and c["positive_passes"] for c in ctrl.values())
    subj = {}
    for s in KC.SELECT_SUBJ:
        rs = R[f"{s}|local|rho_R1|art|r"]["value"]
        rsh = R[f"{s}|local|rho_R3|art|r"]["value"]
        pr = R[f"{s}|local|R0|art"]["partial_r"]
        subj[s] = {"rho_stale": rs, "rho_shuffle": rsh, "partial_r": pr,
                   "kill": rs >= KS2_RHO_STALE_KILL or pr < KS2_PARTIAL_KILL or rsh >= KS2_RHO_SHUFFLE_KILL}
    kill = any(v["kill"] for v in subj.values())
    verdict = "tool_defect" if not tool_ok else ("kill" if kill else "pass")
    write("ks2", {"verdict": verdict, "controls": ctrl, "subjects": subj,
                  "consequence": {"kill": "C1 is a predictor / extrapolator: K-star terminated as a tracking scheme, no "
                                          "fusion, no GPU", "pass": "continue",
                                  "tool_defect": "gate descriptive; one redesign needs a new pre-registration"}[verdict]})


def _num(x):
    if isinstance(x, (int, float)):
        return float(x)
    return 1.0 if x.startswith(">") else 0.0


def ks3():
    v = load("ks3v")["validity"]
    R = load("ks3")
    prof = "measured_local" if any("measured_local" in e for k, e in R.items() if isinstance(e, dict) and "n_endpoints" in e) \
        else "rho0.7"
    vals = sorted(_num(e[prof]["ind"]["r_star_e3"]) for k, e in R.items()
                  if isinstance(e, dict) and "n_endpoints" in e and "local" in k)
    med = vals[len(vals) // 2] if len(vals) % 2 else 0.5 * (vals[len(vals) // 2 - 1] + vals[len(vals) // 2])
    tool_ok = bool(v["V0_ok"] and v["V1_ok"])
    if not tool_ok:
        verdict = "tool_defect"
    elif med > KS3_DEAD:
        verdict = "local_dead"
    elif med > KS3_GLOBAL_ONLY:
        verdict = "global_only"
    else:
        verdict = "pass"
    write("ks3", {"verdict": verdict, "profile": prof, "median_r_star_e3_local": med, "per_sequence": vals,
                  "validity": v, "gpu_target_r": med if verdict == "pass" else None})


def ks4():
    R = load("ks4")["readings"]
    subj = {}
    for s in KC.SELECT_SUBJ:
        b = R[f"{s}|local|t14|aug_innov|offset_slope"]["slope_y_on_x"]
        d14 = R[f"{s}|local|t14|aug_diff"]["r"]
        d0 = R[f"{s}|local|clean|aug_diff"]["r"]
        subj[s] = {"b_off_14mm": b, "diff_r_14mm": d14, "diff_r_clean": d0,
                   "no_position_info": b < KS4_BOFF_MIN, "capture_not_by_readout": d0 - d14 > KS4_DIFF_DROP}
    info_limited = any(v["no_position_info"] for v in subj.values())
    write("ks4", {"verdict": "information_limited" if info_limited else "pass", "subjects": subj,
                  "needs_capture_training": any(v["capture_not_by_readout"] for v in subj.values()),
                  "consequence": "information_limited: C1 can only refine near the anchor (differential); with KS1 the "
                                 "fusion has no lever on S39's slow error -> no GPU" if info_limited else "continue"})


def ks5():
    aa = load("ks0")["decision"]["seed_level"]
    k1 = load("ks1")["summary_delta_ra"][f"zgz_local|{KS1_RSTAR_KAPPA}"]["mean"]
    mde = aa["zgz_local"]["mde_3v3_mm"]
    gmax = -k1
    write("ks5", {"achievable_gain_local_mm": gmax, "mde_local_mm": mde,
                  "confirmatory_possible": gmax >= KS5_POWER_FACTOR * mde,
                  "consequence": "3-seed confirmatory comparison allowed" if gmax >= KS5_POWER_FACTOR * mde
                  else "GPU stage single-seed descriptive only"})


if __name__ == "__main__":
    KC.verify_protocol()
    {"ks1": ks1, "ks2": ks2, "ks3": ks3, "ks4": ks4, "ks5": ks5}[sys.argv[1]]()
