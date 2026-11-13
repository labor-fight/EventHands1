#!/usr/bin/env python3
"""ABL from-scratch training (docs/EXPERIMENT_RECORDS.md#abl-prereg-20261111): the frozen tools/pmr/train.py unchanged, with
its registry rebound to the union registry 6 (tools/pmr/arms_any6.py, registry 11 included) and its router block installed
by tools/pmr/ablrt.py (r_s2d, an RDR router, or none = S45's ray association), in every DDP rank (Lightning relaunches rank 1
with this same argv). Run with NCCL_P2P_DISABLE=1.

    python tools/pmr/train11.py --config configs/pmr_20261109/pmr_<arm>_s<seed>[_preflight].yaml
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

from tools.pmr import arms_any6  # noqa: E402,F401
from tools.pmr import ablrt as RT  # noqa: E402
from tools.pmr import arms_any2 as U  # noqa: E402
from tools.pmr import train as T  # noqa: E402

T.build_config, T.config_path, T.install_trunk, T.parse = U.build_config, U.config_path, U.install_trunk, U.parse
T.install_routes = RT.install_routes
T.sha = RT.safe_sha
T.CODE = T.CODE + ("tools/pmr/train11.py", "tools/pmr/arms11.py", "tools/pmr/arms_any6.py", "tools/pmr/ablrt.py",
                   "semkine/pmr_trunk11.py", "semkine/pmr_trunk2.py", "semkine/rdr_router.py", "semkine/rdr_hooks.py")

if __name__ == "__main__":
    T.main()
