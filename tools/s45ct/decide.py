#!/usr/bin/env python3
"""S45-CT decisions (CPU only), both registered in docs/EXPERIMENT_RECORDS.md#s45ct-prereg-20261030.

  decide.py acceptance   step 0: T1d replay, preflights, corruption audits, proxy reproduction, T1d re-evaluation
                         and the discrimination check -> acceptance/acceptance.json (formal training needs PASS)
  decide.py final        per arm: R, C, the transfer / stability guards, its class, and the pre-written decision
                         -> summary.json, report.md

R = (S45 - arm) / (S45 - T1d) on the median over the nine training local recordings of each recording's mean E1
RA (fixed step 6000); C = RA after 20 feeds of the same packet from the training-mean finger pose / RA after 1 feed
(median over the same recordings).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.s45ct.common import FORMAL_ARMS, OUT, S45TR, now, read, require, save, sha, verify_protocol  # noqa: E402

TH = {"R_high": 0.6, "R_low": 0.3, "C_conv": 0.8, "C_div": 1.0, "R_reverse": -0.3, "e3_margin_mm": 0.5,
      "global_divergence_mm": 8.0, "discrimination_mm": 10.0}
C1SF_SO3 = ROOT / "outputs/c1sf/20261029/evaluations/so3"
STEPS = (5000, 5500, 6000)


def readout(net, task):
    """S45 / T1d at step 6000 are S45-TR's published readouts; everything else is this study's."""
    base = S45TR / "readouts" / net if net in ("s45", "t1d") else OUT / "readouts" / net
    return np.load(base / task / "states.npz", allow_pickle=False)


def name(net, step):
    return net if step == 6000 else f"{net}_{step}"


def e1_median(net, step=6000):
    z = readout(name(net, step), "tl_e1")
    recs = sorted({k.split("/")[0] for k in z.files})
    return float(np.median([z[f"{r}/mpjpe_ra_mm"].mean() for r in recs]))


def e3_median(net):
    z = readout(net, "tl_e3")
    recs = sorted({k.split("/")[0] for k in z.files})
    return float(np.median([z[f"{r}/mpjpe_ra_mm"].mean() for r in recs]))


def convergence(net):
    """RA at feeds 1 and 20 from the mean finger pose, medians over recordings (S45-TR readouts' r3 formula)."""
    from semkine import eval_track as ET
    from tools.s45loc.analyze import FK
    fk = FK()
    z = readout(net, "tl_f1")
    recs = sorted({k.split("/")[0] for k in z.files})

    def ra(q, k):
        a, b = fk.joints(q, z[f"{q}/mean_iter{k}"]), fk.joints(q, z[f"{q}/target"])
        return float(np.linalg.norm(ET.root_align(a) - ET.root_align(b), axis=-1).mean())
    r1, r20 = float(np.median([ra(q, 1) for q in recs])), float(np.median([ra(q, 20) for q in recs]))
    return {"ra_feed1_mm": r1, "ra_feed20_mm": r20, "C": r20 / r1}


def zgz_e1(net, step):
    path = (C1SF_SO3 / f"e1_step{step}_repeat1_none" if net == "s45" else OUT / "evaluations" / net / f"e1_step{step}")
    m = read(path / "report.json")["metrics"]
    return {d: m[d]["mpjpe_ra_mm"] for d in ("local", "global")}


def r_class(R):
    return "high" if R >= TH["R_high"] else ("low" if R <= TH["R_low"] else "mid")


def c_class(C):
    return "conv" if C <= TH["C_conv"] else ("div" if C >= TH["C_div"] else "mid")


# --------------------------------------------------------------------------- step 0
def acceptance():
    verify_protocol()
    target = OUT / "acceptance" / "acceptance.json"
    require(not target.exists(), "step-0 acceptance is read once")
    from tools.xcl import evaluate as XE
    checks = {}
    rec = {a: read(OUT / "receipts" / f"preflight_{a}.json") for a in ("t1dreplay",) + FORMAL_ARMS}
    checks["t1d_replay_exact"] = rec["t1dreplay"]["status"] == "PASS" and rec["t1dreplay"]["t1d_code_replay_exact"] is True
    checks["arm_preflights"] = all(rec[a]["status"] == "PASS" for a in FORMAL_ARMS)
    checks["audits"] = all(read(OUT / "acceptance" / f"audit_{m}.json")["status"] == "PASS" for m in FORMAL_ARMS)
    checks["proxy_reproduction"] = read(OUT / "acceptance" / "proxy.json")["status"] == "PASS"
    try:
        XE.compare_traces(OUT / "evaluations/t1d/e1_step6000/states.npz", S45TR / "evaluations/t1d/e1_step6000/states.npz")
        checks["t1d_reevaluation_bitwise"] = True
    except RuntimeError:
        checks["t1d_reevaluation_bitwise"] = False
    gaps = {str(s): e1_median("s45", s) - e1_median("t1d", s) for s in STEPS}
    primary = "R" if all(g >= TH["discrimination_mm"] for g in gaps.values()) else "C"
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "in_sample_gain_s45_minus_t1d_mm": gaps, "primary_metric": primary,
              "thresholds": TH, "computed_utc": now()}
    save(target, result, exclusive=True)
    print(result, flush=True)
    require(result["status"] == "PASS", "step-0 acceptance failed; no formal training")


# --------------------------------------------------------------------------- final
def a_e(net):
    """Descriptive: S45-RF P1's own-packet fit on the arm's E2 (fed S45's zgz E1 states)."""
    from tools.s45rf import p1 as P1
    from tools.s45rf.common import pose_mean
    with np.load(OUT / "evaluations" / net / "e2_step6000" / "states.npz", allow_pickle=False) as z:
        arrs = {k: z[k] for k in z.files}
    parts = [P1.block_sums(P1.design(arrs, None, s, P1.BLOCKS["finger"], pose_mean()), "D", ("Mc", "E", "Bm"))
             for s in ("zgz_global", "zgz_local")]
    a_m, ae, _ = P1.solve(parts)
    return {"a_m": float(a_m), "a_e": float(ae)}


def arm_class(R, C, transfer_ok, reverse, primary):
    if reverse:
        return "reverse"
    rc, cc = r_class(R), c_class(C)
    if primary == "C":
        rc = {"conv": "high", "div": "low", "mid": "mid"}[cc]
    if rc == "high" and cc == "conv":
        return "retain" if transfer_ok else "retain_sample_only"
    if rc == "low" and cc == "div":
        return "lose"
    if (rc == "high" and cc == "div") or (rc == "low" and cc == "conv"):
        return "inconsistent"
    return "partial"


def decide(cls, unstable, R):
    s, p = cls["sigma16"], cls["shuffle"]
    if "reverse" in (s, p):
        return "REVERSE"
    if any(unstable.values()):
        return "POWER_INSUFFICIENT"
    if "inconsistent" in (s, p):
        return "MECHANISM_MISMATCH"
    no_leak = ("lose", "retain_sample_only")
    if p == "retain":
        return "B2_LEAKAGE"
    if s == "retain" and p in no_leak:
        return "A_CORRESPONDENCE"
    if s == "retain" and p == "partial":
        return "A_CORRESPONDENCE" if R["sigma16"] - R["shuffle"] >= TH["R_low"] else "B2_LEAKAGE"
    if s == "lose":
        return "B1_EXACTNESS"
    if s == "partial" and p in no_leak:
        return "BOUNDARY"
    if s == "retain_sample_only" and p in no_leak:
        return "CONFOUND_DOMAIN"
    return "INSUFFICIENT_UNMAPPED"


NEXT = {
    "A_CORRESPONDENCE": "build T0' (part-correct, prior-independent correspondence) gated by the proxy I; stop the 8 mm "
                        "per-event gate, T1b with the current T0, N4-r2 (shelved)",
    "B2_LEAKAGE": "stop T0' and treating T1d as a correspondence bound; zero-training probe: can events predict the GT "
                  "occupancy",
    "B1_EXACTNESS": "stop the correspondence route and T0'; whether to continue the S45 recursive line is the user's call",
    "BOUNDARY": "one more arm, sigma8 (same design); T0' target provisionally <= 8 mm random error and part-correct",
    "CONFOUND_DOMAIN": "zero-training S45-LOC zn-shift readout on sigma16 before any T0' work",
    "REVERSE": "routes are trusted unconditionally: any predicted route needs calibrated confidence + gating (proxy F first)",
    "MECHANISM_MISMATCH": "zero-training S45-RF P1 E2 decomposition of the inconsistent arm; no conclusion",
    "POWER_INSUFFICIENT": "one more seed (3408) for the unstable arm and T1d, same thresholds on the two-seed means",
    "INSUFFICIENT_UNMAPPED": "report the readings; no decision",
}


def final():
    verify_protocol()
    acc = read(OUT / "acceptance" / "acceptance.json")
    require(acc["status"] == "PASS", "step-0 acceptance did not pass")
    primary = acc["primary_metric"]
    ref = {s: (e1_median("s45", s), e1_median("t1d", s)) for s in STEPS}
    s45_zgz = {s: zgz_e1("s45", s) for s in STEPS}
    s45_e3 = e3_median("s45")
    res, cls, unstable, Rv = {}, {}, {}, {}
    for arm in FORMAL_ARMS:
        Rs = {s: (ref[s][0] - e1_median(arm, s)) / (ref[s][0] - ref[s][1]) for s in STEPS}
        conv = convergence(arm)
        z = {s: zgz_e1(arm, s) for s in STEPS}
        transfer = sum(z[s]["local"] < s45_zgz[s]["local"] for s in STEPS)
        reverse = Rs[6000] <= TH["R_reverse"] or z[6000]["global"] > s45_zgz[6000]["global"] + TH["global_divergence_mm"]
        e3 = e3_median(arm)
        c = arm_class(Rs[6000], conv["C"], transfer == 3, reverse, primary)
        unstable[arm] = any(r_class(Rs[s]) != r_class(Rs[6000]) for s in (5000, 5500)) if primary == "R" else False
        cls[arm], Rv[arm] = c, Rs[6000]
        res[arm] = {"R": {str(s): v for s, v in Rs.items()}, "convergence": conv, "class": c,
                    "zgz_e1": {str(s): v for s, v in z.items()}, "transfer_local_better_than_s45": transfer,
                    "tl_e3_median_mm": e3, "e3_guard_ok": e3 <= s45_e3 + TH["e3_margin_mm"],
                    "late_checkpoint_unstable": unstable[arm], "e2_fit": a_e(arm)}
    decision = decide(cls, unstable, Rv)
    out = {"computed_utc": now(), "primary_metric": primary, "thresholds": TH, "arms": res,
           "references": {"tl_e1_median_mm": {str(s): {"s45": v[0], "t1d": v[1]} for s, v in ref.items()},
                          "s45_zgz_e1": {str(s): v for s, v in s45_zgz.items()}, "s45_tl_e3_median_mm": s45_e3,
                          "t1d_convergence": convergence("t1d"), "s45_convergence": convergence("s45")},
           "decision": decision, "next": NEXT[decision],
           "sigma16_retains_but_e3_guard_fails": cls["sigma16"] == "retain" and not res["sigma16"]["e3_guard_ok"],
           "single_seed_descriptive": True}
    save(OUT / "summary.json", out)
    lines = ["# S45-CT 结果（自动生成；单 seed 3407，固定 step 6000）", "",
             f"判读：**{decision}**（主指标 {primary}）→ {NEXT[decision]}", "",
             "| 臂 | R@6000 | R@5000 / 5500 | C（RA 喂1→喂20） | zgz local 优于 S45（/3） | 样本内 E3 | 类别 |",
             "|---|---|---|---|---|---|---|"]
    for arm, r in res.items():
        cv = r["convergence"]
        lines.append(f"| {arm} | {r['R']['6000']:+.2f} | {r['R']['5000']:+.2f} / {r['R']['5500']:+.2f} | "
                     f"{cv['C']:.2f}（{cv['ra_feed1_mm']:.1f}→{cv['ra_feed20_mm']:.1f}） | {r['transfer_local_better_than_s45']} | "
                     f"{r['tl_e3_median_mm']:.2f}{'' if r['e3_guard_ok'] else ' ⚠'} | {r['class']} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(out["decision"], out["next"], flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("action", choices=("acceptance", "final"))
    a = p.parse_args(argv)
    acceptance() if a.action == "acceptance" else final()


if __name__ == "__main__":
    main()
