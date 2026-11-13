#!/usr/bin/env python3
"""ABL rollout training (docs/EXPERIMENT_RECORDS.md#abl-prereg-20261111): tools/pmr/train_rol2.py unchanged, with the union
registry 6 (rol1 included), the router block check generalised to r_s2d / RDR / none (tools/pmr/ablrt.py `block_ok`), RDR
checkpoints enabled and, for the ray-association arm, S45-SM's router install replaced by a no-op record, in every DDP rank.
Run with NCCL_P2P_DISABLE=1.

    python tools/pmr/train_rol6.py --config configs/pmr_20261109/pmr_<arm>_rol5|rol1_s<seed>[_preflight].yaml
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

from semkine import s45sm as SM  # noqa: E402  (before any hook)
from tools.pmr import arms_any6  # noqa: E402,F401
from semkine import rdr_hooks as RH  # noqa: E402
from tools.pmr import ablrt as RT  # noqa: E402
from tools.pmr import train_rol2 as T  # noqa: E402

RH.enable()
_require, _install = T.require, SM.install
_BLOCK = "PMR rollout arms keep B2's router block and the PMR trunk"


def require(cond, message=""):
    if message == _BLOCK:
        import yaml
        p = yaml.safe_load(Path(sys.argv[2]).read_text())["PFO"]
        cond = RT.block_ok(p) and p["TRUNK"] == "pmr"
        message = "ABL rollout arms: r_s2d / an RDR router behind the pfr kind with PFA, or the ray association; PMR trunk"
    return _require(cond, message)


def install(router, checkpoint, assoc, onhand_tau=0.5):
    if router == "none":
        return {"router": "none", "assoc": "ray"}
    return _install(router, checkpoint, assoc, onhand_tau)


T.require = require
T.sha = RT.safe_sha
SM.install = install
T.CODE = T.CODE + ("tools/pmr/train_rol6.py", "tools/pmr/arms_any6.py", "tools/pmr/arms11.py", "tools/pmr/ablrt.py",
                   "semkine/pmr_trunk11.py", "semkine/pmr_trunk2.py", "semkine/rdr_router.py", "semkine/rdr_hooks.py")

if __name__ == "__main__":
    T.main()
