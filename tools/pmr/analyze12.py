#!/usr/bin/env python3
"""PMR design D2 written verdict (docs/EXPERIMENT_RECORDS.md#pmr-s9-prereg-20261109): tools/pmr/analyze7.py's S5 rule
unchanged (module: RA(no-P control, s) - RA(X, s) > 0 for every seed; done: the three-seed means of X + rol5's E1 five all
below S43's engine values), with the union registry extended by tools/pmr/arms_any4.py and d1hubfull's controls (D1's
R-noprop / R-mlp: its node inputs and readout are D1's) added. Outputs keep analyze7's names: decisions/s5_d1hubfull.json,
analysis/s5_d1hubfull.md, outputs/semkine/pmr_d1hubfull_rol5_3seed_main_row.json.

    python -m tools.pmr.analyze12 s5 --arm d1hubfull
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms_any4  # noqa: E402,F401
from tools.pmr import analyze7 as A7  # noqa: E402

A7.NOPROP["d1hubfull"], A7.MLP["d1hubfull"] = "d1n", "d1m"

if __name__ == "__main__":
    A7.main()
