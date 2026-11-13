#!/usr/bin/env python3
"""PMR registry-10 rollout readings and acceptance: tools/pmr/run_rol2.py unchanged, union registry 5, RDR enabled.

    python -m tools.pmr.run_rol5 read --arm d1rv3_rol5 --seed 3407 --task T
    python -m tools.pmr.run_rol5 accept --arm d1rv3_rol5 --seed 3407 --phase preflight|formal
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import rdr_hooks as RH  # noqa: E402
from tools.pmr import arms_any5  # noqa: E402,F401
from tools.pmr import run_rol2 as R  # noqa: E402

RH.enable()

if __name__ == "__main__":
    R.main()
