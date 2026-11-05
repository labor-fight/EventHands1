#!/usr/bin/env python3
"""ACU stage-1 training of a registered arm (tools/acu/arms.py, docs/EXPERIMENT_RECORDS.md#acu-s1-prereg-20261104).

The base's procedure (tools/pfo2/train.py for B2, tools/b2ub/train.py for B2-gt; semkine/train.py unchanged;
deterministic flags; Lightning's Trainer built with `deterministic=True`, #pfo2-amend1-20261103; B2's pair-stream hashes)
with one change: the trunk class is the CA trunk (semkine/acu_ca.py, same parameters and initial values as the base's
gshare trunk) and the face-route association is CA instead of PFA. The routes stay the base's: the deployed r_s2d router
(B2, a2) or the batch's XAG pseudo-sources through the native `semkine.xag.sample_sources` (B2-gt, a1).

    python tools/acu/train.py --config configs/acu_20261104/acu_<arm>_s<seed>[_preflight].yaml
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

from tools.acu.arms import build_config, config_path  # noqa: E402
from tools.acu.common import now, require, save, sha  # noqa: E402

POLICY = {"trainer_deterministic": True, "benchmark": False}
CODE = ("tools/acu/train.py", "tools/acu/arms.py", "semkine/acu_ca.py", "semkine/evf_render.py", "semkine/pfo_trunk4.py",
        "semkine/s45sm.py", "semkine/pfo_router.py", "semkine/train.py", "semkine/dataset.py", "semkine/xag.py",
        "model/model.py", "semkine/xyz_mesh.py")


def install_policy():
    import pytorch_lightning as pl
    native = pl.Trainer

    class PolicyRecord(pl.Callback):
        def on_fit_start(self, trainer, model):
            import torch
            if trainer.global_rank == 0:
                out = Path(model.cfg["TRAIN"]["OUTPUT_DIR"])
                save(out / "acu_det.json", {**POLICY, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                            "recorded_at_utc": now()}, exclusive=True)

    class Trainer(native):
        def __init__(self, *args, **kwargs):
            kwargs.update(deterministic=True, benchmark=False)
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), PolicyRecord()]
            super().__init__(*args, **kwargs)
    pl.Trainer = Trainer


def install_routes(p):
    """The base's training routes and PFA hooks (B2: the deployed router; B2-gt: the native pseudo-source sampler)."""
    from semkine import s45sm as SM
    if p["ROUTER"] == "label":
        from tools.b2ub.train import install_label_routes
        return install_label_routes()
    if p["ROUTER_KIND"] == "pfr":
        from semkine import pfo_router as PR
        PR.register()
    return SM.install(p["ROUTER_KIND"], p["ROUTER_CHECKPOINT"], p["ASSOC"], float(p["ONHAND_TAU"]))


def main():
    require(len(sys.argv) == 3 and sys.argv[1] == "--config", "usage: train.py --config <registered config>")
    path = Path(sys.argv[2]).resolve()
    cfg = yaml.safe_load(path.read_text())
    a = cfg["ACU"]
    p = cfg["PFO"]
    require(path == config_path(a["ARM"], p["SEED"], p["PHASE"]).resolve(), "unregistered config path")
    require(cfg == build_config(a["ARM"], p["SEED"], p["PHASE"]), "config differs from the registry")
    require(a["IFACE"] == "ca" and p["ASSOC"] == "pfa", "stage-1 arms are CA arms over a PFA base")
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
    from semkine import acu_ca as CA
    from model import MNISTModel
    T.XYZPairStreamCallback = T.U1aPairStreamCallback               # B2's pair-stream hashes (the 12 packet fields)
    routing = install_routes(p)
    trunk = CA.install(a["TRUNK"])
    hooks = CA.install_hooks()
    CA.install_intrinsics(MNISTModel)
    if rank0:
        out.mkdir(parents=True, exist_ok=False)
        save(out / "acu_launch.json", {"acu": a, "pfo": p, "phase": p["PHASE"], "routing": routing, "trunk": trunk,
                                       "hooks": hooks, "config": str(path), "config_sha256": sha(path),
                                       "code_sha256": {c: sha(ROOT / c) for c in CODE},
                                       "xag_source_root": cfg["DATA"].get("XAG_SOURCE_ROOT"),
                                       "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                       "started_utc": now()})
    sys.argv = [sys.argv[0], "--config", str(path)]
    T.main()


if __name__ == "__main__":
    main()
