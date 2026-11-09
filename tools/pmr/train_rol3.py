#!/usr/bin/env python3
"""PMR stage S8 rollout training (docs/EXPERIMENT_RECORDS.md#pmr-s8-prereg-20261109): tools/pmr/train_rol2.py unchanged, with
the union registry extended by tools/pmr/arms_any3.py in every DDP rank (Lightning relaunches rank 1 with this same argv).
Run with NCCL_P2P_DISABLE=1.

    python tools/pmr/train_rol3.py --config configs/pmr_20261109/pmr_<arm>_rol5_s<seed>[_preflight].yaml
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.pmr import arms_any3  # noqa: E402,F401  (extends arms_any2 before train_rol2 reads it)
from tools.pmr import train_rol2 as T  # noqa: E402

T.CODE = T.CODE + ("tools/pmr/train_rol3.py", "tools/pmr/arms_any3.py", "tools/pmr/arms7.py", "tools/pmr/arms8.py",
                   "semkine/pmr_trunk7.py", "semkine/pmr_trunk8.py")

if __name__ == "__main__":
    T.main()
