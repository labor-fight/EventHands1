#!/usr/bin/env python3
"""Engineering smoke of tools/tpd/train_rol.py on GPU under the deterministic flags (before the stage is frozen; not a
reading): the registered preflight recipe with the output and config roots moved to a scratch directory and MAX_STEPS
cut. Lightning relaunches rank 1 with this script's argv, so the redirection comes from the environment.

    TPD_SMOKE_DIR=<scratch> TPD_SMOKE_STEPS=8 python tools/tpd/smoke_rol.py --arm rol3      (writes the config, runs)
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.tpd import arms_rol as A  # noqa: E402


def redirect():
    scratch = Path(os.environ["TPD_SMOKE_DIR"])
    steps = int(os.environ.get("TPD_SMOKE_STEPS", "8"))
    A.OUT, A.CONFIGS = scratch / "out", scratch / "configs"
    A.PREFLIGHT.update(MAX_STEPS=steps, SAVE_EVERY_N_STEPS=steps, VAL_CHECK_INTERVAL=steps, WARMUP_STEPS=2)


def main():
    redirect()
    if sys.argv[1] == "--arm":
        arm = sys.argv[2]
        sys.argv = [sys.argv[0], "--config", str(A.write_config(arm, 3407, "preflight"))]
    from tools.tpd import train_rol
    train_rol.main()


if __name__ == "__main__":
    main()
