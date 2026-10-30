#!/usr/bin/env python3
"""Freeze the C1SF protocol before any C1SF reading (refuses to overwrite).

Records the SHA-256 of every C1SF source, config and every frozen module the runs import, the gate
thresholds decide.py reads, and the SHA of the pre-registration section in docs/EXPERIMENT_RECORDS.md.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.c1sf.common import (ARMS, CONFIGS, FORMAL_ARMS, OUT, PHYSICAL_GPUS, PROTOCOL, PROTOCOL_V1, RESOURCES,  # noqa: E402
                               ROOT, TRAIN_BASE, now, save, sha)

SECTIONS = (("<!-- EXPERIMENT_RECORD:C1SF-PREREG-20261029:BEGIN -->", "<!-- EXPERIMENT_RECORD:C1SF-PREREG-20261029:END -->"),
            ("<!-- EXPERIMENT_RECORD:C1SF-AMEND1-20261029:BEGIN -->", "<!-- EXPERIMENT_RECORD:C1SF-AMEND1-20261029:END -->"))
FROZEN = (
    "tools/c1sf/__init__.py", "tools/c1sf/common.py", "tools/c1sf/make_configs.py", "tools/c1sf/train.py",
    "tools/c1sf/evaluate.py", "tools/c1sf/chain.py", "tools/c1sf/decide.py", "tools/c1sf/freeze.py",
    "semkine/c1sf_zrel.py", "tests/test_c1sf.py",
    "configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml",
    "semkine/xyz_mesh.py", "model/model.py", "model/config.py", "model/mano_layer.py", "semkine/dataset.py",
    "semkine/domrand.py", "semkine/encoder.py", "semkine/eval_track.py", "semkine/train.py",
    "tools/xcl/evaluate.py", "tools/xcl/interventions.py", "tools/xcl/report.py", "tools/xdl/geometry.py",
    "tools/xyz_c1_repro/train.py", "tools/make_s36_row.py", "tools/report_table.py", "assets/mano_right.npz",
)
GATES = {
    "step1": {"subject_bias_max_ratio": 0.25, "structural_min_ratio": 0.5,
              "statistic": "median over the 9 base training recordings of (C1 E3 RA - identity RA), per domain, "
                           "divided by the same difference on zgz_<domain>"},
    "xcl_rule": {"threshold_mm": {"local": 5.0, "global": 8.0},
                 "rule": "tools/xcl/report.py direction_support: |E1 delta| >= threshold, E1/E2/E3 deltas strictly "
                         "the same sign, >= 2 of the step 5000/5500/6000 E1 deltas with that sign"},
    "mechanism": {"ood_e3_improvement_mm": 2.0, "ood_geometry_harm_max_mm": 1.0, "echo_geometry_harm_min_mm": 2.0,
                  "rule": "OOD supported if dz or zr: E3 local - C1 E3 local <= -2.0 and frozen geometry_zero "
                          "E3 local delta >= -1.0; echo supported if both dz and zr: geometry_zero delta <= -2.0 and "
                          "|E3 local change| < 2.0; otherwise undetermined"},
}


def main():
    require_fresh = not PROTOCOL.exists()
    if not require_fresh:
        raise SystemExit("protocol already frozen: " + str(PROTOCOL))
    text = (ROOT / "docs/EXPERIMENT_RECORDS.md").read_text()
    parts = []
    for begin, end in SECTIONS:
        if text.count(begin) != 1 or text.count(end) != 1:
            raise SystemExit("pre-registration / amendment section markers missing or duplicated")
        parts.append(text[text.index(begin):text.index(end) + len(end)])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prereg_section_r2.md").write_text("\n\n".join(parts) + "\n")
    files = list(FROZEN) + sorted(str(p.relative_to(ROOT)) for p in CONFIGS.glob("*.yaml"))
    protocol = {
        "study": "C1SF_20261029", "seed": 3407, "frozen_at_utc": now(),
        "prereg_anchor": "docs/EXPERIMENT_RECORDS.md#c1sf-prereg-20261029",
        "prereg_section_sha256": sha(OUT / "prereg_section_r2.md"),
        "amends": "outputs/c1sf/20261029/protocol.json", "amends_sha256": sha(PROTOCOL_V1),
        "amendment": "docs/EXPERIMENT_RECORDS.md#c1sf-amend1-20261029 (GPU 2-7; loader NUM_WORKERS 16 / PREFETCH_FACTOR 2; "
                     "r2 configs and preflights; driver starts after the v1 step 1; two trace evaluations per GPU)",
        "resources": RESOURCES,
        "arms": ARMS, "formal_arms": list(FORMAL_ARMS), "train_base_recordings": list(TRAIN_BASE),
        "physical_gpus": list(PHYSICAL_GPUS), "gates": GATES,
        "file_sha256": {f: sha(ROOT / f) for f in files},
        "readings_before_freeze": "v1 step 1 (step1.json), the evaluation pairing check (pairing_eval.json) and the v1 "
                                  "c1replay / dz preflight receipts (receipts_v1/) were read before this amendment; no arm reading",
    }
    save(PROTOCOL, protocol, exclusive=True)
    print(PROTOCOL, len(protocol["file_sha256"]), "files frozen")


if __name__ == "__main__":
    main()
