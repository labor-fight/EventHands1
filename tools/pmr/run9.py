#!/usr/bin/env python3
"""PMR design D2 readings and acceptance (docs/EXPERIMENT_RECORDS.md#pmr-s9-prereg-20261109): the frozen
tools/pmr/run.py unchanged, with its arm registry rebound to the D2 registry (tools/pmr/arms9.py).

    python -m tools.pmr.run9 read --arm d1hubfull --task T
    python -m tools.pmr.run9 accept --arm d1hubfull --phase preflight|formal
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms9 as A2  # noqa: E402
from tools.pmr import run as R  # noqa: E402

R.A = A2

if __name__ == "__main__":
    R.main()
