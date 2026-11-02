#!/usr/bin/env python3
"""T1A-UB latency breakdown after engineering amendment 1 (docs/EXPERIMENT_RECORDS.md#t1aub-amend1-rows-20261102): the
frozen tools/t1aub/breakdown.py with tools/t1aub/devfix.py and tools/t1aub/r6.py's resolution and hooks.

    python tools/t1aub/breakdown7.py --net N --block B
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.t1aub import devfix, r6  # noqa: E402

devfix.install()
r6.patch_common()
from tools.t1aub import breakdown as BD  # noqa: E402

if __name__ == "__main__":
    BD.main()
