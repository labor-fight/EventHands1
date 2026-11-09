#!/usr/bin/env python3
"""PMR stage S3b rounds rungs training (amendment 3) (tools/pmr/arms3.py, docs/EXPERIMENT_RECORDS.md#pmr-amend3-20261109): the frozen
tools/pmr/train.py unchanged, with its registry functions rebound to the rounds-rung registry in every DDP rank (Lightning
relaunches rank 1 with this same argv). Run with NCCL_P2P_DISABLE=1.

    python tools/pmr/train3.py --config configs/pmr_20261109/pmr_d1l<k>b_s3407[_preflight].yaml
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

from tools.pmr import arms3 as A2  # noqa: E402
from tools.pmr import train as T  # noqa: E402

T.build_config, T.config_path, T.install_trunk, T.parse = A2.build_config, A2.config_path, A2.install_trunk, A2.parse
T.CODE = T.CODE + ("tools/pmr/train3.py", "tools/pmr/arms3.py", "semkine/pmr_trunk3.py")

if __name__ == "__main__":
    T.main()
