#!/usr/bin/env python3
"""PMR stage S4c readings and acceptance (docs/EXPERIMENT_RECORDS.md#pmr-s4c-prereg-20261109): the frozen
tools/pmr/run.py unchanged, with its arm registry rebound to the S4c registry (tools/pmr/arms6.py).

    python -m tools.pmr.run6 read --arm d1<refinement> --task T
    python -m tools.pmr.run6 accept --arm d1<refinement> --phase preflight|formal
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms6 as A2  # noqa: E402
from tools.pmr import run as R  # noqa: E402

R.A = A2

if __name__ == "__main__":
    R.main()
