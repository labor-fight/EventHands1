#!/usr/bin/env python3
"""PMR design D2 rollout readings and acceptance (docs/EXPERIMENT_RECORDS.md#pmr-s9-prereg-20261109): tools/pmr/run_rol2.py
unchanged, with the union registry extended by tools/pmr/arms_any4.py.

    python -m tools.pmr.run_rol4 read --arm A_rol5 --seed S --task T
    python -m tools.pmr.run_rol4 accept --arm A_rol5 --seed S --phase preflight|formal
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms_any4  # noqa: E402,F401
from tools.pmr import run_rol2 as R  # noqa: E402

if __name__ == "__main__":
    R.main()
