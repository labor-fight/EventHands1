#!/usr/bin/env python3
"""T1A-UB rounds >= 6: the frozen round-1 evaluator (tools/t1aub/evaluate.py, unchanged) with tools/t1aub/r6.py's hooks
(router-location-feature arms) on top of tools/t1aub/r5.py's resolution (oracle cells, round->=2 cells, round-1 layer).

    python tools/t1aub/evaluate6.py {trace,readout,row} ...     (the arguments of evaluate.py)
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.t1aub import r6  # noqa: E402

r6.patch_common()
from tools.t1aub import evaluate as E  # noqa: E402

if __name__ == "__main__":
    E.main()
