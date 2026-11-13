#!/usr/bin/env python3
"""PMR registry-10 training (docs/EXPERIMENT_RECORDS.md#rdr-trunk-prereg-20261110): the frozen tools/pmr/train.py unchanged,
with its registry rebound to tools/pmr/arms10.py and the router installed from the arm's own RDR checkpoint, in every DDP
rank (Lightning relaunches rank 1 with this same argv). Run with NCCL_P2P_DISABLE=1.

    python tools/pmr/train10.py --config configs/pmr_20261109/pmr_d1rv<k>_s<seed>[_preflight].yaml
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

from tools.pmr import arms10 as A2  # noqa: E402
from tools.pmr import train as T  # noqa: E402


def install_routes(p):
    """The arm's RDR router behind the "pfr" hook kind, PFA, tau 0.5."""
    from semkine import pfo_router as PR
    from semkine import rdr_hooks as RH
    from semkine import s45sm as SM
    T.require(p["ROUTER_KIND"] == "pfr" and p["ASSOC"] == "pfa" and p["ROUTER"].startswith("rdr_"), "registry-10 router block")
    RH.enable()
    PR.register()
    state = SM.install(p["ROUTER_KIND"], p["ROUTER_CHECKPOINT"], p["ASSOC"], float(p["ONHAND_TAU"]))
    return {**state, "sample_sources": "semkine.s45sm.predicted_sources"}


T.build_config, T.config_path, T.install_trunk, T.parse = A2.build_config, A2.config_path, A2.install_trunk, A2.parse
T.install_routes = install_routes
T.CODE = T.CODE + ("tools/pmr/train10.py", "tools/pmr/arms10.py", "semkine/rdr_router.py", "semkine/rdr_hooks.py")

if __name__ == "__main__":
    T.main()
