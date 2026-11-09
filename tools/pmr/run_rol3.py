#!/usr/bin/env python3
"""PMR stage S8 rollout readings and acceptance (docs/EXPERIMENT_RECORDS.md#pmr-s8-prereg-20261109): tools/pmr/run_rol2.py
unchanged, with the union registry extended by tools/pmr/arms_any3.py.

    python -m tools.pmr.run_rol3 read --arm A_rol5 --seed S --task T
    python -m tools.pmr.run_rol3 accept --arm A_rol5 --seed S --phase preflight|formal
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms_any3  # noqa: E402,F401
from tools.pmr import run_rol2 as R  # noqa: E402

if __name__ == "__main__":
    R.main()
