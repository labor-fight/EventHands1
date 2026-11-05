#!/usr/bin/env python3
"""PFO phase-2b training (docs/EXPERIMENT_RECORDS.md#pfo-p2b-prereg-20261103): tools/pfo/train.py's procedure for the
phase-2b arms (tools/pfo/arms2.py): routing of the config's PFO block (sr1 or a registered PFO router, hard PFA), the
trunk variant (semkine/pfo_trunk.py or semkine/pfo_trunk2.py), and, when registered, the own-loop prior bank
(semkine/s45tr_prior.py `install_n4`); deterministic kernels; fresh-only outputs.

    python tools/pfo/train2.py --config configs/pfo_20261103/pfo_<arm>_s<seed>[_preflight].yaml
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

import yaml  # noqa: E402

from tools.pfo.arms2 import TRUNK2, build_config, config_path  # noqa: E402
from tools.pfo.common import now, require, save, sha  # noqa: E402
from tools.pfo.train import register_audit_module  # noqa: E402


def main():
    require(len(sys.argv) == 3 and sys.argv[1] == "--config", "usage: train2.py --config <registered config>")
    path = Path(sys.argv[2]).resolve()
    cfg = yaml.safe_load(path.read_text())
    p = cfg["PFO"]
    require(path == config_path(p["ARM"], p["SEED"], p["PHASE"]).resolve(), "unregistered config path")
    require(cfg == build_config(p["ARM"], p["SEED"], p["PHASE"]), "config differs from the registry")
    out = Path(cfg["TRAIN"]["OUTPUT_DIR"])
    if os.environ.get("LOCAL_RANK") in (None, "0"):
        require(not out.exists(), "fresh-only output already exists: " + str(out))
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    register_audit_module()
    from semkine import train as T
    from semkine import s45sm as SM
    T.XYZPairStreamCallback = T.U1aPairStreamCallback               # packets carry no XAG source
    if p["ROUTER_KIND"] == "pfr":
        from semkine import pfo_router as PR
        PR.register()
    state = SM.install(p["ROUTER_KIND"], p["ROUTER_CHECKPOINT"], p["ASSOC"], float(p["ONHAND_TAU"]))
    if p["TRUNK"] in TRUNK2:
        from semkine import pfo_trunk2 as PT
        module = "semkine/pfo_trunk2.py"
    else:
        from semkine import pfo_trunk as PT
        module = "semkine/pfo_trunk.py"
    PT.install(p["TRUNK"])
    bank = None
    if p.get("BANK"):
        from semkine import s45tr_prior as P
        P.install_n4(p["BANK"]["root"], float(p["BANK"]["p"]))
        bank = p["BANK"]
    if os.environ.get("LOCAL_RANK") in (None, "0"):
        out.mkdir(parents=True, exist_ok=False)
        save(out / "pfo_launch.json", {"pfo": p, "routing": state, "trunk": PT.installed(), "bank": bank,
                                       "config": str(path), "config_sha256": sha(path),
                                       "train_wrapper_sha256": sha(__file__), "trunk_module_sha256": sha(ROOT / module),
                                       "router_checkpoint_sha256": sha(p["ROUTER_CHECKPOINT"]),
                                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                       "started_utc": now()})
    sys.argv = [sys.argv[0], "--config", str(path)]
    T.main()


if __name__ == "__main__":
    main()
