#!/usr/bin/env python3
"""PMR registry-10 readings and acceptance (docs/EXPERIMENT_RECORDS.md#rdr-trunk-prereg-20261110): the frozen
tools/pmr/run.py unchanged, with its registry rebound to tools/pmr/arms10.py and RDR checkpoints enabled.

    python -m tools.pmr.run10 read --arm d1rv3 --seed 3407 --task T
    python -m tools.pmr.run10 accept --arm d1rv3 --phase preflight|formal --seed 3407
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import rdr_hooks as RH  # noqa: E402
from tools.pmr import arms10 as A2  # noqa: E402
from tools.pmr import run as R  # noqa: E402

RH.enable()
R.A = A2

if __name__ == "__main__":
    R.main()
