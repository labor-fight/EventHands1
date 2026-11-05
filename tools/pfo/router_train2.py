#!/usr/bin/env python3
"""PFO router training of any registered router (tools/pfo/routers3.py): tools/pfo/router_train.py's procedure, unchanged.

    python tools/pfo/router_train2.py --router NAME --phase preflight|formal
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.pfo import router_train as RT  # noqa: E402
from tools.pfo import routers3 as R3  # noqa: E402

RT.ROUTERS.clear()
RT.ROUTERS.update(R3.ROUTERS3)

if __name__ == "__main__":
    RT.main()
