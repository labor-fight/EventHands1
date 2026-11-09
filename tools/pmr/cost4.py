#!/usr/bin/env python3
"""PMR cost readings for any registered arm (docs/EXPERIMENT_RECORDS.md#pmr-s5-prereg-20261109): tools/pmr/cost2.py
(PFO's frozen harness with the PMR stage breakdown) unchanged, with tools/pmr/run.py's registry rebound to the union
registry (tools/pmr/arms_any.py) so arm_spec / prepare accept the S4 refinements. GPU 4 exclusive.

    python -m tools.pmr.cost4 run --arm A [--seed S] --policy default|det
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms_any as U  # noqa: E402
from tools.pmr import run as R  # noqa: E402

R.A = U

if __name__ == "__main__":
    from tools.pmr import cost2 as C2
    C2.main()
