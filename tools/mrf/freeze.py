#!/usr/bin/env python3
"""MRF freeze: source / config / reference identities into outputs/mrf/protocol.json, written once, after the
pre-registration (docs/EXPERIMENT_RECORDS.md#mrf-prereg-20261024) and before any MRF reading. model/model.py is not in
`files`: Stages A-C verify it against `model_wiring.pre` (the file AGG778 R2 froze), Stage D against `model_wiring.post`
(the same file with the frozen MODEL.XYZ_RF wiring, applied after AGG778 R2 is complete).

    python tools/mrf/freeze.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.agg778 import common as C  # noqa: E402
from tools.mrf import common as MC  # noqa: E402
from tools.mrf import wiring as W  # noqa: E402

PLAN = Path("/home/lyq/.claude/plans/xyz-c1-meshgnn-receptive-field-mrf.md")
XCL = "outputs/xcl/20261019/evaluations"
READOUTS = ("e1_step6000_repeat1_none", "e1_step6000_repeat2_none", "e2_step6000_repeat1_none",
            "e3_step6000_repeat1_none", "e3_step6000_repeat1_h3_zero", "e1_step5000_repeat1_none",
            "e1_step5500_repeat1_none", "gt_pose_step6000_repeat1_none", "gt_transl_step6000_repeat1_none")
FILES = [
    "semkine/mrf.py", "tests/test_mrf.py", "tools/mrf/run.sh",
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tools" / "mrf").glob("*.py")),
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "configs" / "mrf").rglob("*.yaml")),
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tools" / "agg778").glob("*.py")),
    "semkine/xyz_mesh.py", "semkine/xvm.py", "semkine/xag.py", "semkine/xag_sources.py", "semkine/eval_track.py",
    "semkine/dataset.py", "semkine/encoder.py", "semkine/events.py", "semkine/train.py",
    "model/config.py", "model/mano_layer.py", "model/pose_repr.py",
    "tools/xvm/common.py", "tools/xvm/omega_r.py", "tools/s44/evflow.py", "tools/xcl/evaluate.py",
    "tools/xcl/interventions.py", "tools/xcl/common.py", "tools/xcl/audit.py", "tools/xyz_c1_repro/train.py",
    "tools/make_s36_row.py", "tools/report_table.py", "tools/xdl/geometry.py", "assets/mano_right.npz",
    "configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml", "configs/xcl_20261019/xcl_a3f0_s3407.yaml",
    *[f"{XCL}/{r}/{m}/report.json" for r in ("xyz_c1_stable_s3407", "xcl_a3f0_s3407", "xag_paired_a0f1_s3407")
      for m in READOUTS],
    "outputs/semkine/xyz_c1_baseline_main_row.json", "outputs/semkine/dt3_dz_l3_w200cfg_main_row.json",
    f"{XCL}/xcl_a3f0_s3407/row_step6000_repeat1_none/xcl_a3f0_main_row_step6000.json",
    *sorted(str(p.relative_to(ROOT)) for p in (MC.OUT / "precheck_20261024").glob("*") if p.is_file()),
]


def main():
    if MC.PROTOCOL.exists():
        raise SystemExit("MRF protocol.json exists; the freeze is written once")
    missing = [f for f in FILES if not (ROOT / f).is_file()]
    if missing:
        raise SystemExit(f"missing frozen inputs: {missing}")
    files = {f: C.sha(ROOT / f) for f in FILES}
    external = {}
    for cond, m in C.MODELS.items():
        got = C.sha(ROOT / m["ckpt"])
        if got != m["sha256"]:
            raise SystemExit(f"{cond} checkpoint SHA differs from the registered identity")
        external[m["ckpt"]] = got
    external[str(C.MANIFEST)] = C.sha(C.MANIFEST)
    external[C.STRICT_TRACE["path"]] = C.sha(ROOT / C.STRICT_TRACE["path"])
    external[str(PLAN)] = C.sha(PLAN)
    wiring = W.shas()
    r2 = json.loads((ROOT / "outputs" / "agg778" / "r2" / "protocol.json").read_text())
    if r2["files"]["model/model.py"] != wiring["pre"]:
        raise SystemExit("model/model.py is not the file AGG778 R2 froze")
    proto = {"study": "MRF", "prereg": "docs/EXPERIMENT_RECORDS.md#mrf-prereg-20261024",
             "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "physical_gpus": [4, 5, 6, 7],
             "files": files, "model_wiring": wiring, "external": external, "models": C.MODELS,
             "seed_policy": {"B_probes": "3 seeds per main arm (one seed index = one initial value per family)",
                             "D": "single seed 3407 (user decision 2026-10-24)"},
             "disclosed_before_freeze": ["outputs/mrf/precheck_20261024 (descriptive, no accuracy)",
                                         "engineering smoke runs (losses only)"],
             "no_reading_before_freeze": True}
    C.save_json(MC.PROTOCOL, proto, exclusive=True)
    print(json.dumps({"files": len(files), "external": len(external), "model_wiring": wiring}))


if __name__ == "__main__":
    main()
