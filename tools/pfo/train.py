#!/usr/bin/env python3
"""PFO training (docs/EXPERIMENT_RECORDS.md#pfo-prereg-20261103, phase-2 sections): one arm x seed, two GPUs.

    python tools/pfo/train.py --config configs/pfo_20261103/pfo_<arm>_s<seed>[_preflight].yaml

`semkine/train.py` unchanged, with, in every DDP rank (Lightning relaunches rank 1 with this same argv):
  * the routing of the config's PFO block (frozen events-only router + hard PFA, `semkine/s45sm.py`, as pfasrdz trained)
  * the trunk variant (`semkine/pfo_trunk.py`) installed before the model is built
  * deterministic kernels (the S45 strict flags); the packets carry no XAG source (pair-stream audit of the base fields)
The config must equal the registry's generated config (tools/pfo/arms.py); output directories are fresh-only.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import types

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402

from tools.pfo.arms import build_config, config_path  # noqa: E402
from tools.pfo.common import now, require, save, sha  # noqa: E402


def parse():
    require(len(sys.argv) == 3 and sys.argv[1] == "--config", "usage: train.py --config <registered config>")
    path = Path(sys.argv[2]).resolve()
    cfg = yaml.safe_load(path.read_text())
    p = cfg["PFO"]
    require(path == config_path(p["ARM"], p["SEED"], p["PHASE"]).resolve(), "unregistered config path")
    require(cfg == build_config(p["ARM"], p["SEED"], p["PHASE"]), "config differs from the registry")
    return path, cfg


def register_audit_module():
    """semkine/train.py imports the retired `tools.xag.train_audit` for configs with MODEL.XAG_*: provide it."""
    import pytorch_lightning as pl
    package = sys.modules.get("tools.xag") or types.ModuleType("tools.xag")
    package.__path__ = []
    module = types.ModuleType("tools.xag.train_audit")

    class XAGTrainAudit(pl.Callback):
        def __init__(self, output_dir):
            self.output_dir = Path(output_dir)

        def on_fit_start(self, trainer, model):
            if trainer.global_rank == 0:
                save(self.output_dir / "pfo_binding.json", {"pfo": model.cfg["PFO"], "recorded_at_utc": now(),
                                                            "config_sha256": sha(model.cfg["_config_path"])})

    def write(path, rows):
        save(path, rows)
    module.XAGTrainAudit, module.write = XAGTrainAudit, write
    package.train_audit = module
    sys.modules["tools.xag"], sys.modules["tools.xag.train_audit"] = package, module


def main():
    path, cfg = parse()
    p = cfg["PFO"]
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
    from semkine import pfo_trunk as PT
    T.XYZPairStreamCallback = T.U1aPairStreamCallback               # packets carry no XAG source
    state = SM.install(p["ROUTER_KIND"], p["ROUTER_CHECKPOINT"], p["ASSOC"], float(p["ONHAND_TAU"]))
    PT.install(p["TRUNK"])
    if os.environ.get("LOCAL_RANK") in (None, "0"):
        out.mkdir(parents=True, exist_ok=False)
        save(out / "pfo_launch.json", {"pfo": p, "routing": state, "trunk": PT.installed(), "config": str(path),
                                       "config_sha256": sha(path), "train_wrapper_sha256": sha(__file__),
                                       "trunk_module_sha256": sha(ROOT / "semkine/pfo_trunk.py"),
                                       "router_checkpoint_sha256": sha(p["ROUTER_CHECKPOINT"]),
                                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                       "started_utc": now()})
    sys.argv = [sys.argv[0], "--config", str(path)]
    T.main()


if __name__ == "__main__":
    main()
