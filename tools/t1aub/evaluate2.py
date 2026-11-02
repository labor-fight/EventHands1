#!/usr/bin/env python3
"""T1A-UB rounds >= 2: the frozen round-1 evaluator (tools/t1aub/evaluate.py trace / readout / row, unchanged) with the
network resolution, configs and hooks of tools/t1aub/r2.py; same arguments and outputs as evaluate.py."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.t1aub import r2  # noqa: E402

r2.patch_common()
from tools.t1aub import evaluate as E  # noqa: E402

if __name__ == "__main__":
    E.main()
