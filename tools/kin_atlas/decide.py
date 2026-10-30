#!/usr/bin/env python3
"""KA step 1 decision (docs/EXPERIMENT_RECORDS.md#ka-prereg-20261028): reads outputs/kin_atlas/{synth,real}/result.json
and the unit-test receipt, writes outputs/kin_atlas/decision.json.

Validity (all required, else tool_defect -- descriptive only):
  V1 synthetic A/A (reference vs second threshold realisation): AUC >= 0.90 for atlas_kin and for the better box
  V2 synthetic shuffled motion labels (oop family): |AUC - 0.5| <= 0.05 for every representation
  V3 tests/test_kin_atlas.py passed (receipt written by outputs/kin_atlas/run_ka.sh)
Gates (box = the better of box16 / box24 on the same data, chosen per comparison -- conservative against the atlas):
  G1 synthetic, families oop + combined pooled per context: AUC(atlas_kin) - AUC(box) >= 0.05, bootstrap (contexts,
     2000) 95 % lower bound > 0
  G2 real (nuisance-different anchors): AUC(atlas_kin) - AUC(box) >= 0.03, 95 % lower bound > 0
  G3 real: AUC(atlas_kin) - AUC(atlas_rigid) > 0 with 95 % lower bound > 0 (articulated lifting matters)
Verdict: pass = G1 and G2 (go to step 2; G3 decides whether the joint / articulated part is needed or a rigid 3-D hand
frame suffices); fail_real = G2 fails (stop the line); inconsistent = G1 fails and G2 passes (descriptive, no claim).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.kin_atlas import common as C  # noqa: E402

G1_MIN, G2_MIN = 0.05, 0.03
V1_MIN, V2_TOL = 0.90, 0.05


def _pooled(fam_dict, fams):
    ctxs = sorted(set.intersection(*[set(fam_dict[f]) for f in fams]))
    return np.array([np.mean([fam_dict[f][c] for f in fams]) for c in ctxs]), np.array(ctxs)


def main():
    C.verify_protocol()
    syn = json.loads((C.OUT / "synth" / "result.json").read_text())
    real = json.loads((C.OUT / "real" / "result.json").read_text())
    tests = json.loads((C.OUT / "tests_receipt.json").read_text())
    out = {"validity": {}, "gates": {}}
    # V1 / V2
    aa = {rep: float(np.mean(list(syn["controls"][rep]["aa"].values()))) for rep in C.REPS}
    sh = {rep: float(np.mean(list(syn["controls"][rep]["shuffled_oop"].values()))) for rep in C.REPS}
    box_aa = max(aa["box16"], aa["box24"])
    out["validity"]["V1_aa"] = {"auc": aa, "pass": bool(aa["atlas_kin"] >= V1_MIN and box_aa >= V1_MIN)}
    out["validity"]["V2_shuffled"] = {"auc": sh, "pass": bool(all(abs(v - 0.5) <= V2_TOL for v in sh.values()))}
    out["validity"]["V3_tests"] = {"pass": bool(tests.get("passed"))}
    valid = all(v["pass"] for v in out["validity"].values())
    # G1 synthetic
    fam = syn["families"]
    kin, ctxs = _pooled(fam["atlas_kin"], ("oop", "combined"))
    boxes = {b: _pooled(fam[b], ("oop", "combined"))[0] for b in ("box16", "box24")}
    best = max(boxes, key=lambda b: boxes[b].mean())
    d = kin - boxes[best]
    ci = C.boot_ci(d, ctxs)
    out["gates"]["G1_synthetic"] = {"box_best": best, "auc_atlas_kin": float(kin.mean()), "auc_box": float(boxes[best].mean()),
                                    "diff": float(d.mean()), "ci95": list(ci),
                                    "pass": bool(d.mean() >= G1_MIN and ci[0] > 0)}
    out["descriptive_synthetic_by_family"] = {
        rep: {f: float(np.mean(list(fam[rep][f].values()))) for f in fam[rep]} for rep in C.REPS}
    # G2 / G3 real
    r = real["feats"]
    best_r = r["box_best"]
    d2 = r[f"diff|atlas_kin-{best_r}"]
    d3 = r["diff|atlas_kin-atlas_rigid"]
    out["gates"]["G2_real"] = {"box_best": best_r, "auc": {k: v["mean"] for k, v in r["auc"].items()},
                               "diff": d2["mean"], "ci95": d2["ci95"], "pass": bool(d2["mean"] >= G2_MIN and d2["ci95"][0] > 0)}
    out["gates"]["G3_articulation"] = {"diff": d3["mean"], "ci95": d3["ci95"], "pass": bool(d3["mean"] > 0 and d3["ci95"][0] > 0)}
    g1, g2 = out["gates"]["G1_synthetic"]["pass"], out["gates"]["G2_real"]["pass"]
    if not valid:
        verdict = "tool_defect"
    elif g1 and g2:
        verdict = "pass"
    elif not g2:
        verdict = "fail_real"
    else:
        verdict = "inconsistent"
    out["verdict"] = verdict
    out["articulated_part_needed"] = bool(out["gates"]["G3_articulation"]["pass"]) if verdict == "pass" else None
    C.save_json(C.OUT / "decision.json", out)
    print(json.dumps({"verdict": verdict, "gates": {k: v["pass"] for k, v in out["gates"].items()},
                      "validity": {k: v["pass"] for k, v in out["validity"].items()}}, indent=1))


if __name__ == "__main__":
    main()
