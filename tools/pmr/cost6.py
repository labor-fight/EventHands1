#!/usr/bin/env python3
"""PMR stage S8 cost reading (docs/EXPERIMENT_RECORDS.md#pmr-s8-prereg-20261109): tools/pmr/cost4.py's binding (cost2 with
tools/pmr/run.py rebound to the union registry) with the union extended by tools/pmr/arms_any3.py. GPU 4 exclusive.

    python -m tools.pmr.cost6 run --arm A [--seed S] --policy default|det
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms_any3  # noqa: E402,F401
from tools.pmr import arms_any2 as U  # noqa: E402
from tools.pmr import run as R  # noqa: E402

R.A = U

if __name__ == "__main__":
    from tools.pmr import cost2 as C2
    C2.main()
