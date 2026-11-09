#!/usr/bin/env python3
"""PMR stage S3b rounds rungs readings and acceptance (amendment 3) (docs/EXPERIMENT_RECORDS.md#pmr-amend3-20261109): the frozen
tools/pmr/run.py unchanged, with its arm registry rebound to the rounds-rung registry (tools/pmr/arms3.py).

    python -m tools.pmr.run3 read --arm d1l<k>b --task T
    python -m tools.pmr.run3 accept --arm d1l<k>b --phase preflight|formal
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms3 as A2  # noqa: E402
from tools.pmr import run as R  # noqa: E402

R.A = A2

if __name__ == "__main__":
    R.main()
