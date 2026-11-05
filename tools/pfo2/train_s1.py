#!/usr/bin/env python3
"""PFO2 stage-1 training (docs/EXPERIMENT_RECORDS.md#pfo2-s1-prereg-20261103): one registered stage-1 arm
(tools/pfo2/reg1.py) x seed x phase, two GPUs, with tools/pfo2/train_det.py's deterministic policy.

    python tools/pfo2/train_s1.py --config configs/pfo2_20261103/pfo2_<arm>_s<seed>[_preflight|_head500].yaml

semkine/train.py unchanged; in every DDP rank: the frozen router + hard PFA of the config, the stage-1 trunk
(semkine/pfo2_trunk.py) with its context hook, the arm's auxiliary losses (semkine/pfo2_train.py), deterministic kernels
kept through Lightning (train_det's Trainer policy). Fresh-only output directories.
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

from tools.pfo.train import register_audit_module  # noqa: E402
from tools.pfo2.common import now, require, save, sha  # noqa: E402
from tools.pfo2.reg1 import build_config, config_path  # noqa: E402
from tools.pfo2.train_det import install_policy  # noqa: E402


def install_arm(p, losses, dual=None):
    """Trunk, context hook, losses and double forward of a stage-1 arm (the evaluation entry passes neither)."""
    from model import MNISTModel
    from semkine import pfo2_trunk as PT
    PT.install(p["TRUNK"])
    PT.install_context(MNISTModel)
    if losses:
        from semkine import pfo2_train as L
        unknown = set(losses) - {"disp"}
        require(not unknown, f"unknown losses {unknown}")
        L.install_disp_loss(MNISTModel, losses["disp"])
    if dual:
        from semkine import pfo2_dual as D
        D.install(MNISTModel, float(dual["lam_max"]), int(dual["start"]), int(dual["end"]))
    return PT


def main():
    require(len(sys.argv) == 3 and sys.argv[1] == "--config", "usage: train_s1.py --config <registered config>")
    path = Path(sys.argv[2]).resolve()
    cfg = yaml.safe_load(path.read_text())
    p = cfg["PFO"]
    require(path == config_path(p["ARM"], p["SEED"], p["PHASE"]).resolve(), "unregistered config path")
    require(cfg == build_config(p["ARM"], p["SEED"], p["PHASE"]), "config differs from the registry")
    out = Path(cfg["TRAIN"]["OUTPUT_DIR"])
    if os.environ.get("LOCAL_RANK") in (None, "0"):
        require(not out.exists(), "fresh-only output already exists: " + str(out))
    install_policy()
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
    T.XYZPairStreamCallback = T.U1aPairStreamCallback
    if p["ROUTER_KIND"] == "pfr":
        from semkine import pfo_router as PR
        PR.register()
    state = SM.install(p["ROUTER_KIND"], p["ROUTER_CHECKPOINT"], p["ASSOC"], float(p["ONHAND_TAU"]))
    PT = install_arm(p, cfg["S1"]["LOSSES"], cfg["S1"]["DUAL"])
    if os.environ.get("LOCAL_RANK") in (None, "0"):
        out.mkdir(parents=True, exist_ok=False)
        save(out / "pfo2_launch.json", {"pfo": p, "s1": cfg["S1"], "phase": p["PHASE"], "routing": state,
                                        "trunk": PT.installed(), "module": "pfo2", "bank": None, "config": str(path),
                                        "config_sha256": sha(path), "train_wrapper_sha256": sha(__file__),
                                        "trunk_module_sha256": sha(Path(PT.__file__)),
                                        "loss_module_sha256": sha(ROOT / "semkine/pfo2_train.py"),
                                        "dual_module_sha256": sha(ROOT / "semkine/pfo2_dual.py"),
                                        "router_checkpoint_sha256": sha(p["ROUTER_CHECKPOINT"]),
                                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                        "started_utc": now()})
    sys.argv = [sys.argv[0], "--config", str(path)]
    T.main()


if __name__ == "__main__":
    main()
