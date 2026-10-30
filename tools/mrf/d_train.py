#!/usr/bin/env python3
"""MRF Stage D training: the AGG778 R1 wrapper `tools/agg778/train.py`, unchanged, with the MRF registry and protocol
check swapped in (`tools/mrf/d_stage.py`). Lightning re-executes this file for the second DDP rank, so the swap applies
there too."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.agg778 import train as T  # noqa: E402
from tools.mrf import d_stage as R  # noqa: E402

R.patch_train(T)

if __name__ == "__main__":
    T.main()
