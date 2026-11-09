#!/usr/bin/env python3
"""PMR stage S6 readings and acceptance (docs/EXPERIMENT_RECORDS.md#pmr-s6-prereg-20261109): the frozen
tools/pmr/run.py unchanged, with its arm registry rebound to the S6 registry (tools/pmr/arms7.py).

    python -m tools.pmr.run7 read --arm d1hub --task T
    python -m tools.pmr.run7 accept --arm d1hub --phase preflight|formal
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms7 as A2  # noqa: E402
from tools.pmr import run as R  # noqa: E402

R.A = A2

if __name__ == "__main__":
    R.main()
