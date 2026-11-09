#!/usr/bin/env python3
"""PMR stage S7 readings and acceptance (docs/EXPERIMENT_RECORDS.md#pmr-s7-prereg-20261109): the frozen
tools/pmr/run.py unchanged, with its arm registry rebound to the S7 registry (tools/pmr/arms8.py).

    python -m tools.pmr.run8 read --arm d1ghand<hp|hub> --task T
    python -m tools.pmr.run8 accept --arm d1ghand<hp|hub> --phase preflight|formal
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms8 as A2  # noqa: E402
from tools.pmr import run as R  # noqa: E402

R.A = A2

if __name__ == "__main__":
    R.main()
