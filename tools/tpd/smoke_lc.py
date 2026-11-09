#!/usr/bin/env python3
"""Engineering smoke of tools/tpd/router_lc.py on GPU under the deterministic flags (before the LC stage is frozen; not a
reading): the preflight recipe with the output root moved to TPD_SMOKE_DIR and TPD_SMOKE_STEPS steps. Lightning relaunches
rank 1 with this script's argv, so the redirection comes from the environment.

    TPD_SMOKE_DIR=<scratch> TPD_SMOKE_STEPS=8 python tools/tpd/smoke_lc.py --router lc --phase preflight
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.tpd import router_lc as R  # noqa: E402

if __name__ == "__main__":
    R.OUT = Path(os.environ["TPD_SMOKE_DIR"])
    R.STEPS["preflight"] = int(os.environ.get("TPD_SMOKE_STEPS", "8"))
    R.main()
