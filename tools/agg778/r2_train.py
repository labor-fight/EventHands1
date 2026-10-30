#!/usr/bin/env python3
"""AGG778 R2 Stage D training: the R1 wrapper `tools/agg778/train.py`, unchanged, with the R2 registry swapped in
(`tools/agg778/r2_d.py`). Lightning re-executes this file for the second DDP rank, so the swap applies there too."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.agg778 import train as T  # noqa: E402
from tools.agg778 import r2_d as R  # noqa: E402

R.patch_train(T)

if __name__ == "__main__":
    T.main()
