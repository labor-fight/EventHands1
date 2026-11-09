#!/usr/bin/env python3
"""PMR stage S8 written verdict (docs/EXPERIMENT_RECORDS.md#pmr-s8-prereg-20261109): tools/pmr/analyze7.py's S5 rule
unchanged (module: RA(no-P control, s) - RA(X, s) > 0 for every seed; done: the three-seed means of X + rol5's E1 five all
below S43's engine values), with the union registry extended by tools/pmr/arms_any3.py and the S6 / S7 arms' controls
(D1's R-noprop / R-mlp: their node inputs are D1's) added. Outputs keep analyze7's names: decisions/s5_<arm>.json,
analysis/s5_<arm>.md, outputs/semkine/pmr_<arm>_rol5_3seed_main_row.json.

    python -m tools.pmr.analyze11 s5 --arm d1hub|d1ghandhp|d1ghandhub
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms_any3  # noqa: E402,F401
from tools.pmr import analyze7 as A7  # noqa: E402

for _arm in ("d1hub", "d1ghandhp", "d1ghandhub"):
    A7.NOPROP[_arm], A7.MLP[_arm] = "d1n", "d1m"

if __name__ == "__main__":
    A7.main()
