#!/usr/bin/env python3
"""TPD driver 5 (docs/EXPERIMENT_RECORDS.md#tpd-amend6-20261106): tools/tpd/chain3.py unchanged with physical GPUs 4-7
only (user 2026-11-06: "GPU4-7, 4 cards"): trainings on the pairs (4, 5) / (6, 7), evaluations on GPUs 4-7, GPUs 0-3
never. Same verbs, queue / status / logs under outputs/tpd/20261105/$TPD_DRIVER_DIR (driver5).

    TPD_DRIVER_DIR=driver5 python tools/tpd/chain4.py add|run|drain|close|summary
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from tools.tpd import chain3 as C  # noqa: E402

C.PHYSICAL_GPUS = (4, 5, 6, 7)
C.PAIRS = ((4, 5), (6, 7))

if __name__ == "__main__":
    C.main()
