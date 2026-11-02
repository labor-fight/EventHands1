#!/usr/bin/env python3
"""T1A-UB rounds >= 6: the frozen latency breakdown (tools/t1aub/breakdown.py, unchanged) with tools/t1aub/r6.py's hooks.

    python tools/t1aub/breakdown6.py --net N --block B
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.t1aub import r6  # noqa: E402

r6.patch_common()
from tools.t1aub import breakdown as BD  # noqa: E402

if __name__ == "__main__":
    BD.main()
