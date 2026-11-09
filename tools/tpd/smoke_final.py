#!/usr/bin/env python3
"""Engineering smoke of tools/tpd/train_final.py on GPU under the deterministic flags (before the synthesis stage is
frozen; not a reading): an arm's preflight recipe with the output / config roots moved to TPD_SMOKE_DIR, TPD_SMOKE_STEPS
steps and TPD_SMOKE_WORKERS data workers. Lightning relaunches rank 1 with this script's argv, so the redirection comes
from the environment.

    TPD_SMOKE_DIR=<scratch> python tools/tpd/smoke_final.py --arm c0
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.tpd import arms_final as A  # noqa: E402

if __name__ == "__main__":
    scratch = Path(os.environ["TPD_SMOKE_DIR"])
    steps = int(os.environ.get("TPD_SMOKE_STEPS", "8"))
    workers = int(os.environ.get("TPD_SMOKE_WORKERS", "4"))
    A.OUT, A.CONFIGS = scratch / "out", scratch / "configs"
    native = A.build_config

    def build(arm, seed, phase="formal"):
        cfg = native(arm, seed, phase)
        cfg["TRAIN"].update(MAX_STEPS=steps, SAVE_EVERY_N_STEPS=steps, VAL_CHECK_INTERVAL=steps, NUM_WORKERS=workers, WARMUP_STEPS=2,
                            LIMIT_VAL_BATCHES=2)
        return cfg
    A.build_config = build
    if sys.argv[1] == "--arm":
        sys.argv = [sys.argv[0], "--config", str(A.write_config(sys.argv[2], 3407, "preflight"))]
    from tools.tpd import train_final
    train_final.main()
