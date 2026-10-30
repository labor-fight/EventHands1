#!/usr/bin/env python3
"""AGG778 R2-1: freeze the R2 source / config / reference identities into outputs/agg778/r2/protocol.json (written
once, before any R2 probe reading; the pilot result it discloses is on selection subjects only).

    python tools/agg778/r2_freeze.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.agg778 import common as C  # noqa: E402

FILES = [
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tools" / "agg778").glob("*.py")),
    "tools/agg778/r2_run.sh", "tests/test_agg778_r2.py", "tests/test_agg778_model.py", "tests/test_agg778.py",
    "semkine/agg778.py", "semkine/xyz_mesh.py", "semkine/xvm.py", "semkine/xag.py", "semkine/xag_sources.py",
    "semkine/eval_track.py", "semkine/dataset.py", "semkine/encoder.py", "semkine/events.py", "semkine/train.py",
    "model/model.py", "model/config.py", "model/mano_layer.py", "model/pose_repr.py",
    "tools/xvm/common.py", "tools/xvm/omega_r.py", "tools/s44/evflow.py", "tools/xcl/evaluate.py",
    "tools/xcl/interventions.py", "tools/xcl/common.py", "tools/xcl/audit.py", "tools/xyz_c1_repro/train.py",
    "tools/make_s36_row.py", "tools/report_table.py", "tools/xdl/geometry.py",
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "configs" / "agg778_r2").rglob("*.yaml")),
    "configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml", "configs/xcl_20261019/xcl_a3f0_s3407.yaml",
    *[f"outputs/xcl/20261019/evaluations/{r}/{m}/report.json" for r in ("xyz_c1_stable_s3407", "xcl_a3f0_s3407")
      for m in ("e1_step6000_repeat1_none", "e2_step6000_repeat1_none", "e3_step6000_repeat1_none",
                "e3_step6000_repeat1_h3_zero", "e1_step5000_repeat1_none", "e1_step5500_repeat1_none")],
    "outputs/semkine/xyz_c1_baseline_main_row.json", "outputs/semkine/dt3_dz_l3_w200cfg_main_row.json",
    "outputs/xcl/20261019/evaluations/xcl_a3f0_s3407/row_step6000_repeat1_none/xcl_a3f0_main_row_step6000.json",
    "outputs/agg778/r1_source_archive/manifest.json", "outputs/agg778/r2/pilot/summary.json",
]


def main():
    out = C.OUT / "r2" / "protocol.json"
    if out.exists():
        raise SystemExit("R2 protocol.json exists; the freeze is written once")
    files = {f: C.sha(ROOT / f) for f in FILES}
    external = {}
    for cond, m in C.MODELS.items():
        got = C.sha(ROOT / m["ckpt"])
        if got != m["sha256"]:
            raise SystemExit(f"{cond} checkpoint SHA differs from the registered identity")
        external[m["ckpt"]] = got
    external[str(C.MANIFEST)] = C.sha(C.MANIFEST)
    pilot = json.loads((C.OUT / "r2" / "pilot" / "summary.json").read_text())
    if not pilot.get("pass"):
        raise SystemExit("the R2 pilot did not pass: redesign the downstream before pre-registering")
    proto = {"study": "AGG778-R2", "prereg": "docs/EXPERIMENT_RECORDS.md#agg778-r2-prereg-20261024",
             "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "physical_gpus": [4, 5, 6, 7],
             "files": files, "external": external, "models": C.MODELS,
             "pilot_pass": True, "no_probe_reading_before_freeze": True}
    C.save_json(out, proto, exclusive=True)
    print(json.dumps({"files": len(files), "external": len(external)}))


if __name__ == "__main__":
    main()
