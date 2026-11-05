#!/usr/bin/env python3
"""PFO training of any registered arm (tools/pfo/arms3.py): tools/pfo/train2.py's procedure, unchanged, resolving the arm
through the data-driven registry. Lightning relaunches rank 1 with this script's argv, so both ranks resolve alike.

    python tools/pfo/train3.py --config configs/pfo_20261103/pfo_<arm>_s<seed>[_preflight].yaml
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.pfo import arms3 as A3  # noqa: E402
from tools.pfo import train2 as T2  # noqa: E402

T2.build_config, T2.config_path = A3.build_config, A3.config_path

if __name__ == "__main__":
    T2.main()
