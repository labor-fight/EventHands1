#!/usr/bin/env python3
"""T1A-UB round 5: the frozen round-1 evaluator (tools/t1aub/evaluate.py, unchanged) with network resolution through
tools/t1aub/r5.py (oracle cells), then tools/t1aub/r2.py (round->=2 cells), then the round-1 layer.

    python tools/t1aub/evaluate5.py {trace,readout,row} ...     (the arguments of evaluate.py)
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.t1aub import r5  # noqa: E402

r5.patch_common()
from tools.t1aub import evaluate as E  # noqa: E402

if __name__ == "__main__":
    E.main()
