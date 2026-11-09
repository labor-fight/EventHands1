#!/usr/bin/env python3
"""PMR stage S3c cost block for any registered arm (docs/EXPERIMENT_RECORDS.md#pmr-s3c-prereg-20261109):
tools/pmr/cost3.py unchanged, with tools/pmr/run.py's registry rebound to the union registry (tools/pmr/arms_any2.py) so
--arm accepts the S4 refinements (tools/pmr/cost4.py does the same for cost2). Every job exclusive on GPU 4.

    python -m tools.pmr.cost5 run --net s43 --impl exec|dense --policy default|det
    python -m tools.pmr.cost5 run --net b2 --policy default|det
    python -m tools.pmr.cost5 run --arm A [--fast] --policy default|det
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms_any2 as U  # noqa: E402
from tools.pmr import run as R  # noqa: E402

R.A = U

if __name__ == "__main__":
    from tools.pmr import cost3 as C3
    C3.main()
