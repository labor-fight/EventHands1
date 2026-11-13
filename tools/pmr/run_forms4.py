#!/usr/bin/env python3
"""PMR design D2 input-form readings (docs/EXPERIMENT_RECORDS.md#pmr-d2-screen-prereg-20261110): tools/pmr/run_forms.py
unchanged, with the union registry extended by tools/pmr/arms_any4.py so that d1hubfull_rol5 resolves.

    python -m tools.pmr.run_forms4 read --arm d1hubfull_rol5 --seed S --form str --task e1|e3
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms_any4  # noqa: E402,F401
from tools.pmr import run_forms as F  # noqa: E402

if __name__ == "__main__":
    F.main()
