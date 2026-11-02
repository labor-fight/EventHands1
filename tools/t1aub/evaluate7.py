#!/usr/bin/env python3
"""T1A-UB main rows and readings after engineering amendment 1 (docs/EXPERIMENT_RECORDS.md#t1aub-amend1-rows-20261102):
the frozen round-1 evaluator (tools/t1aub/evaluate.py, unchanged) with S45-SM's router cache made per device
(tools/t1aub/devfix.py) and network resolution / hooks through tools/t1aub/r6.py (round-6 location-feature arms) on top of
tools/t1aub/r5.py (oracle cells), tools/t1aub/r2.py (round->=2 cells) and the round-1 layer.

    python tools/t1aub/evaluate7.py {trace,readout,row} ...     (the arguments of evaluate.py)
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.t1aub import devfix, r6  # noqa: E402

devfix.install()
r6.patch_common()
from tools.t1aub import evaluate as E  # noqa: E402

if __name__ == "__main__":
    E.main()
