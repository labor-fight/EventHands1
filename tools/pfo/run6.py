#!/usr/bin/env python3
"""PFO router swaps / readings for any registered router (tools/pfo/routers3.py): tools/pfo/run3.py and
tools/pfo/router_read.py, unchanged, with the data-driven registry.

    python -m tools.pfo.run6 swap --trunk T --router R --task T
    python -m tools.pfo.run6 read --router R
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo import router_train as RT  # noqa: E402
from tools.pfo import routers3 as R3  # noqa: E402

RT.ROUTERS.clear()
RT.ROUTERS.update(R3.ROUTERS3)

if __name__ == "__main__":
    verb = sys.argv[1]
    if verb == "swap":
        from tools.pfo import run3
        run3.main(sys.argv[1:])
    elif verb == "read":
        from tools.pfo import router_read
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        router_read.main()
    else:
        raise SystemExit("usage: run6.py swap|read ...")
