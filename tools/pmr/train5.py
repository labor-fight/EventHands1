#!/usr/bin/env python3
"""PMR stage S4b refinement training (tools/pmr/arms5.py, docs/EXPERIMENT_RECORDS.md#pmr-s4b-prereg-20261109): the frozen
tools/pmr/train.py unchanged, with its registry functions rebound to the S4b registry in every DDP rank (Lightning
relaunches rank 1 with this same argv). Run with NCCL_P2P_DISABLE=1.

    python tools/pmr/train5.py --config configs/pmr_20261109/pmr_d1<refinement>_s<seed>[_preflight].yaml
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

from tools.pmr import arms5 as A2  # noqa: E402
from tools.pmr import train as T  # noqa: E402

T.build_config, T.config_path, T.install_trunk, T.parse = A2.build_config, A2.config_path, A2.install_trunk, A2.parse
T.CODE = T.CODE + ("tools/pmr/train5.py", "tools/pmr/arms5.py", "semkine/pmr_trunk5.py")

if __name__ == "__main__":
    T.main()
