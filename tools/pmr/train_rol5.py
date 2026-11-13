#!/usr/bin/env python3
"""PMR registry-10 rollout training (docs/EXPERIMENT_RECORDS.md#rdr-trunk-prereg-20261110): tools/pmr/train_rol2.py
unchanged, with the union registry extended by tools/pmr/arms_any5.py and the router block's name check replaced by the
registry-10 one (the hook kind stays "pfr"; semkine/rdr_hooks.py loads the RDR checkpoint). Run with NCCL_P2P_DISABLE=1.

    python tools/pmr/train_rol5.py --config configs/pmr_20261109/pmr_d1rv<k>_rol5_s<seed>[_preflight].yaml
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

from semkine import rdr_hooks as RH  # noqa: E402
from tools.pmr import arms_any5  # noqa: E402,F401
from tools.pmr import train_rol2 as T  # noqa: E402

RH.enable()
_require = T.require
_BLOCK = "PMR rollout arms keep B2's router block and the PMR trunk"


def require(cond, message=""):
    if message == _BLOCK:
        import yaml
        p = yaml.safe_load(Path(sys.argv[2]).read_text())["PFO"]
        cond = p["ROUTER_KIND"] == "pfr" and p["ASSOC"] == "pfa" and p["ROUTER"].startswith("rdr_") and p["TRUNK"] == "pmr"
        message = "registry-10 rollout arms: an RDR router behind the pfr kind, PFA, the PMR trunk"
    return _require(cond, message)


T.require = require
T.CODE = T.CODE + ("tools/pmr/train_rol5.py", "tools/pmr/arms_any5.py", "tools/pmr/arms10.py", "semkine/rdr_router.py",
                   "semkine/rdr_hooks.py")

if __name__ == "__main__":
    T.main()
