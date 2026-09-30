#!/usr/bin/env python3
"""S37 rotation-loss-weight arms (docs/S37_ROTW_CNNROOT_PREREG.md): single variable against S37."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("arm", ["s37_rotw10", "s37_rotw30"])
def test_rotw_differs_from_s37_only_in_lambda_r(arm):
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "diff_configs.py"),
         str(ROOT / "configs" / "semkine" / "s37_routed_s3407.yaml"),
         str(ROOT / "configs" / "semkine" / f"{arm}_s3407.yaml"),
         "--allow", "LOSS.LAMBDA_R", "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
