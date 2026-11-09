#!/usr/bin/env python3
"""PMR training of a registered arm (tools/pmr/arms.py, docs/EXPERIMENT_RECORDS.md#pmr-prereg-20261109): B2's training
procedure (tools/idp/train.py with its route `router`: semkine/train.py unchanged; deterministic flags; Lightning's
Trainer built with `deterministic=True` (#pfo2-amend1-20261103); r_s2d + PFA, tau 0.5 (semkine/s45sm.py `install`);
B2's pair-stream hashes) with the arm's PMR trunk installed in every DDP rank.

    python tools/pmr/train.py --config configs/pmr_20261109/pmr_<arm>_s<seed>[_preflight].yaml
Lightning relaunches rank 1 with this same argv (the trunk is installed again there).
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

from tools.pmr.arms import build_config, config_path, install_trunk, parse  # noqa: E402
from tools.pmr.common import now, require, save, sha  # noqa: E402

POLICY = {"trainer_deterministic": True, "benchmark": False}
CODE = ("tools/pmr/train.py", "tools/pmr/arms.py", "tools/pfo2/arms.py", "semkine/pmr_trunk.py", "semkine/s45sm.py",
        "semkine/s45sm_fast.py", "semkine/pfo_router.py", "semkine/train.py", "semkine/dataset.py", "semkine/xag.py",
        "semkine/xyz_mesh.py", "model/model.py")


def install_policy():
    import pytorch_lightning as pl
    native = pl.Trainer

    class PolicyRecord(pl.Callback):
        def on_fit_start(self, trainer, model):
            import torch
            if trainer.global_rank == 0:
                out = Path(model.cfg["TRAIN"]["OUTPUT_DIR"])
                save(out / "pmr_det.json", {**POLICY, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                            "recorded_at_utc": now()}, exclusive=True)

    class Trainer(native):
        def __init__(self, *args, **kwargs):
            kwargs.update(deterministic=True, benchmark=False)
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), PolicyRecord()]
            super().__init__(*args, **kwargs)
    pl.Trainer = Trainer


def install_routes(p):
    """B2's training hooks exactly (the deployed router r_s2d, PFA, tau 0.5)."""
    from semkine import pfo_router as PR
    from semkine import s45sm as SM
    require(p["ROUTER_KIND"] == "pfr" and p["ASSOC"] == "pfa" and p["ROUTER"] == "r_s2d", "PMR keeps B2's router block")
    PR.register()
    state = SM.install(p["ROUTER_KIND"], p["ROUTER_CHECKPOINT"], p["ASSOC"], float(p["ONHAND_TAU"]))
    return {**state, "sample_sources": "semkine.s45sm.predicted_sources"}


def main():
    require(len(sys.argv) == 3 and sys.argv[1] == "--config", "usage: train.py --config <registered config>")
    path = Path(sys.argv[2]).resolve()
    cfg = yaml.safe_load(path.read_text())
    p = cfg["PFO"]
    parse(p["ARM"])
    require(path == config_path(p["ARM"], p["SEED"], p["PHASE"]).resolve(), "unregistered config path")
    require(cfg == build_config(p["ARM"], p["SEED"], p["PHASE"]), "config differs from the registry")
    require(not cfg["DATA"].get("XAG_SOURCE_ROOT"), "PMR arms read no XAG source")
    out = Path(cfg["TRAIN"]["OUTPUT_DIR"])
    rank0 = os.environ.get("LOCAL_RANK") in (None, "0")
    if rank0:
        require(not out.exists(), "fresh-only output already exists: " + str(out))
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    install_policy()
    from tools.pfo.train import register_audit_module
    register_audit_module()
    from semkine import train as T
    T.XYZPairStreamCallback = T.U1aPairStreamCallback               # B2's pair-stream hashes (the 12 packet fields)
    routing = install_routes(p)
    trunk = install_trunk(cfg)
    if rank0:
        out.mkdir(parents=True, exist_ok=False)
        save(out / "pmr_launch.json", {"pfo": p, "pmr": cfg["PMR"], "phase": p["PHASE"], "routing": routing,
                                       "trunk": trunk, "config": str(path), "config_sha256": sha(path),
                                       "code_sha256": {c: sha(ROOT / c) for c in CODE},
                                       "router_checkpoint_sha256": sha(p["ROUTER_CHECKPOINT"]),
                                       "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                       "started_utc": now()})
    sys.argv = [sys.argv[0], "--config", str(path)]
    T.main()


if __name__ == "__main__":
    main()
