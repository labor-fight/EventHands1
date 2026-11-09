#!/usr/bin/env python3
"""Engineering smoke of tools/tpd/train_rolc.py (seed 3409 under tools/tpd/ns_seed.py) on GPU, deterministic flags, output
and config roots moved to TPD_SMOKE_DIR, TPD_SMOKE_STEPS steps (not a reading)."""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.tpd import ns_seed  # noqa: E402

ns_seed.enter()
from tools.tpd import arms_rol as A  # noqa: E402

if __name__ == "__main__":
    scratch = Path(os.environ["TPD_SMOKE_DIR"])
    steps = int(os.environ.get("TPD_SMOKE_STEPS", "6"))
    A.OUT, A.CONFIGS = scratch / "out", scratch / "configs"
    A.PREFLIGHT.update(MAX_STEPS=steps, SAVE_EVERY_N_STEPS=steps, VAL_CHECK_INTERVAL=steps, WARMUP_STEPS=2)
    if sys.argv[1] == "--arm":
        sys.argv = [sys.argv[0], "--config", str(A.write_config(sys.argv[2], 3409, "preflight"))]
    from tools.tpd import train_rol
    train_rol.main()
