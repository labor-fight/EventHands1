#!/usr/bin/env python3
"""AGG778 R2 Stage D evaluation: the R1 evaluator `tools/agg778/d_eval.py` (the unchanged XCL `trace` / `make_main_row`),
with the R2 registry, receipts and protocol swapped in."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.agg778 import d_eval as E  # noqa: E402
from tools.agg778 import r2_d as R  # noqa: E402

R.patch_eval(E)

if __name__ == "__main__":
    E.main()
