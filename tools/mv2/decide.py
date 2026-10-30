#!/usr/bin/env python3
"""Apply every MV2 probe's registered gate (docs/EXPERIMENT_RECORDS.md#mv2-prereg-20261026) and write
outputs/mv2/decisions.json with the GPU-release flags of the plan's section 3.

Each probe's own `decide()` is the gate; this file only routes results to them and combines the releases:
  M1 GPU  = pa_probe.decide_m1(pa1, pa2)["gpu_release_m1"] and the M9 cost account's sync / data-dependent-shape /
            graph-break gates on S39+SLR;
  M2      = factor arms inside M1 only: RGQ in {pass, weak}, MSCN pass;
  M3 / M4 / M5 = conditional single-variable arms on M1 (each released by its own probe; M3 additionally needs M1);
  M6, M8 (unless SRCA releases in-training self-supervision), M9, M10 = no GPU arm.
A missing result leaves its entry "pending".

    python tools/mv2/decide.py [--root outputs/mv2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mv2 import common as C  # noqa: E402


def load(root: Path, name: str):
    p = root / name / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(C.OUT))
    a = ap.parse_args()
    C.verify_protocol()
    root = Path(a.root)
    out = {}
    from mv2 import pa_probe as PA
    pa1, pa2, pa3 = load(root, "pa1"), load(root, "pa2"), load(root, "pa3")
    rgq, mscn = load(root, "rgq"), load(root, "mscn")
    out["M1"] = PA.decide_m1(pa1, pa2) if pa1 and pa2 else "pending"
    out["M1_pa3"] = "descriptive" if pa3 else "pending"
    out["M2"] = PA.decide_m2(rgq, mscn) if rgq and mscn else "pending"
    from mv2 import m3_probes as M3
    out["M3"] = {k: (M3.decide(r) if r else "pending") for k, r in
                 (("FBW", load(root, "fbw")), ("SMG", load(root, "smg")), ("EXPO", load(root, "expo")))}
    from mv2 import m4_contour as M4
    r4 = load(root, "m4_contour")
    out["M4"] = M4.decide(r4) if r4 else "pending"
    from mv2 import m5_palm as M5
    r5 = {p: load(root, f"m5_{p}") for p in ("ib", "schur", "fipa", "rca")}
    r5 = {p: v["summary"] for p, v in r5.items() if v}
    out["M5"] = M5.decide(r5) if r5 else "pending"
    from mv2 import m6_kscs as M6
    r6 = load(root, "m6")
    out["M6"] = M6.decide(r6) if r6 else "pending"
    from mv2 import m8_probes as M8
    out["M8"] = {k: (M8.decide(r) if r else "pending") for k, r in
                 (("SRCA", load(root, "srca")), ("TBC", load(root, "tbc")), ("PF", load(root, "pf")))}
    from mv2 import m9_psq, m9_age, m9_cost, m10_tta, m10_sdg, m10_sad
    for key, mod, name in (("M9_PSQ", m9_psq, "m9_psq"), ("M9_AGE", m9_age, "m9_age"), ("M9_COST", m9_cost, "m9_cost"),
                           ("M10_TTA", m10_tta, "m10_tta"), ("M10_SDG", m10_sdg, "m10_sdg"),
                           ("M10_SAD", m10_sad, "m10_sad")):
        r = load(root, name)
        if r and name == "m10_sdg":                      # amendment 3: SDG's gate reads its registered primary ROI block
            r = r[r.get("primary", "full")]
        out[key] = mod.decide(r) if r else "pending"
    # ---- GPU releases (plan section 3)
    rel = {}
    cost = out["M9_COST"]
    # amendment 1 (2026-10-26, prereg §5 tool defect): `gate_breaks` counts dynamo breaks of the whole model, and the
    # frozen S39 forward alone already has 6 (deployed) / 8 (default) breaks, so no S39-derived arm can reach 0; the gate is
    # descriptive (reported under "m9_cost_gate_breaks_descriptive"), the release keeps 0 host syncs and 0 DDS ops.
    cost_ok = isinstance(cost, dict) and all(cost.get("gates", {}).get(g, False)
                                             for g in ("gate_sync", "gate_dds"))
    out["m9_cost_gate_breaks_descriptive"] = (cost.get("gates", {}).get("gate_breaks") if isinstance(cost, dict) else None)
    m1 = out["M1"]
    rel["M1_slr_debug"] = bool(isinstance(m1, dict) and m1.get("gpu_release_m1") and cost_ok)
    m2 = out["M2"]
    rel["M2_rgq_factor"] = bool(rel["M1_slr_debug"] and isinstance(m2, dict) and m2["rgq"] in ("pass", "weak"))
    rel["M2_mscn_factor"] = bool(rel["M1_slr_debug"] and isinstance(m2, dict) and m2["mscn"] == "pass")
    m3 = out["M3"]
    fbw = m3.get("FBW") if isinstance(m3, dict) else None
    rel["M3_mpes_channel"] = bool(rel["M1_slr_debug"] and isinstance(fbw, dict)
                                  and (fbw.get("overall") or fbw.get("verdict")) == "pass")
    m4 = out["M4"]
    rel["M4_contour_channel"] = bool(isinstance(m4, dict) and m4.get("overall") == "pass")
    m5 = out["M5"]
    rel["M5_arms"] = (m5.get("m5", {}).get("gpu_arms_released", {}) if isinstance(m5, dict) else {})
    out["gpu_release"] = rel
    out["m9_cost_gates_ok_for_slr"] = cost_ok
    C.save_json(root / "decisions.json", out, exclusive=False)
    print(json.dumps({"gpu_release": rel}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
