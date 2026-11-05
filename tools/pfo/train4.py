#!/usr/bin/env python3
"""PFO training of any registered arm (tools/pfo/arms3.py) whose trunk is a semkine/pfo_trunk4.py variant string
("full", "nomp", "nomp+pos", ...; "base" is "full"): tools/pfo/train2.py's procedure with the composable trunk.

    python tools/pfo/train4.py --config configs/pfo_20261103/pfo_<arm>_s<seed>[_preflight].yaml
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

from tools.pfo.arms3 import build_config, config_path  # noqa: E402
from tools.pfo.common import now, require, save, sha  # noqa: E402
from tools.pfo.train import register_audit_module  # noqa: E402


def trunk4_variant(trunk):
    return "full" if trunk == "base" else trunk


def main():
    require(len(sys.argv) == 3 and sys.argv[1] == "--config", "usage: train4.py --config <registered config>")
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
    from semkine import pfo_trunk4 as PT
    T.XYZPairStreamCallback = T.U1aPairStreamCallback
    if p["ROUTER_KIND"] == "pfr":
        from semkine import pfo_router as PR
        PR.register()
    state = SM.install(p["ROUTER_KIND"], p["ROUTER_CHECKPOINT"], p["ASSOC"], float(p["ONHAND_TAU"]))
    PT.install(trunk4_variant(p["TRUNK"]))
    bank = None
    if p.get("BANK"):
        from semkine import s45tr_prior as P
        P.install_n4(p["BANK"]["root"], float(p["BANK"]["p"]))
        bank = p["BANK"]
    if os.environ.get("LOCAL_RANK") in (None, "0"):
        out.mkdir(parents=True, exist_ok=False)
        save(out / "pfo_launch.json", {"pfo": p, "routing": state, "trunk": PT.installed(), "bank": bank,
                                       "config": str(path), "config_sha256": sha(path),
                                       "train_wrapper_sha256": sha(__file__),
                                       "trunk_module_sha256": sha(ROOT / "semkine/pfo_trunk4.py"),
                                       "router_checkpoint_sha256": sha(p["ROUTER_CHECKPOINT"]),
                                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                       "started_utc": now()})
    sys.argv = [sys.argv[0], "--config", str(path)]
    T.main()


if __name__ == "__main__":
    main()
