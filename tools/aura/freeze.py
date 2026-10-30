#!/usr/bin/env python3
"""AURA freeze: source / config / reference identities into outputs/aura/protocol.json, written once, after the
pre-registration (docs/EXPERIMENT_RECORDS.md#aura-prereg-20261024) and before any AURA reading. model/model.py is not in
`files`: Stages A-C accept AGG778 R2's frozen file or that file with the MRF wiring (`model_wiring.pre`); Stage D
accepts the AURA-wired file of either (`model_wiring.post`), applied only after R2 and MRF are both complete.

    python tools/aura/freeze.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.agg778 import common as C  # noqa: E402
from tools.aura import common as AC  # noqa: E402
from tools.aura import wiring as W  # noqa: E402
from semkine import aura as AU  # noqa: E402

PLAN = Path("/home/lyq/.claude/plans/cosmic-launching-cupcake.md")
XCL = "outputs/xcl/20261019/evaluations/xyz_c1_stable_s3407"
READOUTS = ("e1_step6000_repeat1_none", "e1_step6000_repeat2_none", "e2_step6000_repeat1_none",
            "e3_step6000_repeat1_none", "e3_step6000_repeat1_h3_zero", "e1_step5000_repeat1_none",
            "e1_step5500_repeat1_none", "gt_pose_step6000_repeat1_none", "gt_transl_step6000_repeat1_none")
FILES = [
    "semkine/aura.py", "tests/test_aura.py", "tools/aura/run.sh",
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tools" / "aura").glob("*.py")),
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "configs" / "aura").rglob("*.yaml")),
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tools" / "agg778").glob("*.py")),
    "tools/mrf/wiring.py", "tools/mrf/gpu_queue.py",
    "semkine/xyz_mesh.py", "semkine/xag.py", "semkine/xag_sources.py", "semkine/eval_track.py", "semkine/dataset.py",
    "semkine/encoder.py", "semkine/events.py", "semkine/train.py", "semkine/xvm.py",
    "model/config.py", "model/mano_layer.py", "model/pose_repr.py",
    "tools/s44/evflow.py", "tools/xcl/evaluate.py", "tools/xcl/common.py", "tools/xcl/interventions.py",
    "tools/xcl/audit.py", "tools/xcl/sources/step1_source_labels.py", "tools/xcl/sources/step1_common.py",
    "tools/xyz_c1_repro/train.py", "tools/make_s36_row.py", "tools/report_table.py", "assets/mano_right.npz",
    "configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml",
    *[f"{XCL}/{m}/report.json" for m in READOUTS],
    "outputs/semkine/xyz_c1_baseline_main_row.json", "outputs/semkine/dt3_dz_l3_w200cfg_main_row.json",
    "outputs/sba/d/result.json",
    "outputs/deleted_20261019/xag_xa_legacy/files/tools/xa/step1_assoc_regimes.py",
]


def main():
    if AC.PROTOCOL.exists():
        raise SystemExit("AURA protocol.json exists; the freeze is written once")
    missing = [f for f in FILES if not (ROOT / f).is_file()]
    if missing:
        raise SystemExit(f"missing frozen inputs: {missing}")
    files = {f: C.sha(ROOT / f) for f in FILES}
    m = C.MODELS["A0"]
    if C.sha(ROOT / m["ckpt"]) != m["sha256"]:
        raise SystemExit("C1 checkpoint SHA differs from the registered identity")
    external = {m["ckpt"]: m["sha256"], str(C.MANIFEST): C.sha(C.MANIFEST),
                C.STRICT_TRACE["path"]: C.sha(ROOT / C.STRICT_TRACE["path"]), str(PLAN): C.sha(PLAN)}
    wiring = W.states(W.MODEL_PY.read_text())
    if wiring["pre"] != AC.MODEL_PRE:
        raise SystemExit(f"model/model.py pre-states differ from R2's file / MRF's post-wiring: {wiring['pre']}")
    proto = {"study": "AURA", "prereg": "docs/EXPERIMENT_RECORDS.md#aura-prereg-20261024",
             "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "physical_gpus": [4, 5, 6, 7],
             "files": files, "model_wiring": wiring, "external": external, "models": {"A0": m},
             "constants": {"R_MATCH_m": AU.R_MATCH, "SIGMA_STAR": {"zgz_local": 0.94, "zgz_global": 0.96},
                           "BAND_PX": AU.BAND_PX, "FRONT_PX": AU.FRONT_PX, "Z_TOL": AU.Z_TOL, "K": AU.K_CAND,
                           "SHELL_R": AU.SHELL_R, "VIS_TOL": AU.VIS_TOL},
             "seed_policy": {"C_probes": "3 seeds primary, 2 secondary, 1 descriptive; one seed base per family "
                                         "(T0s 50000, T6 50100, T6s12 50200, M / MR / shell / split / j16 50300, "
                                         "occ 50400) + seed index; loader 3407 (A/A group: 3408)",
                             "D": "single seed 3407"},
             "disclosed_before_freeze": [
                 "engineering smoke runs (Stage A / B on 32 / 4 packets, Stage C 3 steps x 8 packets, C2 16 endpoints)",
                 "R_MATCH from fit-subject GT anchor depths (dataset statistic)",
                 "shuffle locality on 12 zgz packets (median partner distance shufG 30 px, shufP 6.4 px)",
                 "independent reviews read the R2 pilot summary, R2 training-loss logs and run times (no R2 evaluation)"],
             "no_reading_before_freeze": True}
    C.save_json(AC.PROTOCOL, proto, exclusive=True)
    print(json.dumps({"files": len(files), "external": len(external), "model_wiring": wiring}))


if __name__ == "__main__":
    main()
