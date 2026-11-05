#!/usr/bin/env python3
"""B2UB U3 training of B2-gt (tools/b2ub/arms.py, docs/EXPERIMENT_RECORDS.md#b2ub-prereg-20261104): tools/pfo2/train.py's
procedure for B2 (semkine/train.py unchanged; deterministic flags; Lightning's Trainer built with `deterministic=True`,
#pfo2-amend1-20261103; the gshare trunk of semkine/pfo_trunk4.py; the pair-stream hashes of tools/pfo2/train.py) with
one change: the routes come from the batch's XAG pseudo-sources instead of the router.

In every DDP rank: `semkine.xag.sample_sources` stays native (it reads the pseudo-sources the batches carry because the
config sets DATA.XAG_SOURCE_ROOT), and the association hooks are exactly B2's training hooks (semkine/s45sm.py
`install` with assoc "pfa": `pfa_association` and `empty_association`), without a router.

    python tools/b2ub/train.py --config configs/b2ub_20261104/b2ub_b2gt_s3407[_preflight].yaml
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

from tools.b2ub.arms import build_config, config_path  # noqa: E402
from tools.b2ub.common import now, require, save, sha  # noqa: E402

POLICY = {"trainer_deterministic": True, "benchmark": False}
CODE = ("tools/b2ub/train.py", "tools/b2ub/arms.py", "tools/pfo2/arms.py", "tools/pfo2/train.py", "semkine/s45sm.py",
        "semkine/pfo_trunk4.py", "semkine/train.py", "semkine/dataset.py", "semkine/xag.py")


def install_policy():
    import pytorch_lightning as pl
    native = pl.Trainer

    class PolicyRecord(pl.Callback):
        def on_fit_start(self, trainer, model):
            import torch
            if trainer.global_rank == 0:
                out = Path(model.cfg["TRAIN"]["OUTPUT_DIR"])
                save(out / "b2ub_det.json", {**POLICY, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                             "recorded_at_utc": now()}, exclusive=True)

    class Trainer(native):
        def __init__(self, *args, **kwargs):
            kwargs.update(deterministic=True, benchmark=False)
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), PolicyRecord()]
            super().__init__(*args, **kwargs)
    pl.Trainer = Trainer


def install_label_routes():
    """B2's training association hooks (semkine/s45sm.py `install`, assoc "pfa") with the native source sampler."""
    from semkine import s45sm as SM
    from semkine import xag as XAG
    from semkine import xyz_mesh as XM
    require(XAG.sample_sources is SM._NATIVE["sample_sources"] and XM.associate_rays is SM._NATIVE["associate_rays"]
            and XAG.route_association is SM._NATIVE["route_association"], "another hook already patched XAG / rays")
    XAG.route_association = SM.pfa_association
    XM.associate_rays = SM.empty_association
    return {"router": "label", "assoc": "pfa", "sample_sources": "semkine.xag.sample_sources (native)",
            "route_association": "semkine.s45sm.pfa_association", "associate_rays": "semkine.s45sm.empty_association"}


def main():
    require(len(sys.argv) == 3 and sys.argv[1] == "--config", "usage: train.py --config <registered config>")
    path = Path(sys.argv[2]).resolve()
    cfg = yaml.safe_load(path.read_text())
    p = cfg["PFO"]
    require(path == config_path(p["ARM"], p["SEED"], p["PHASE"]).resolve(), "unregistered config path")
    require(cfg == build_config(p["ARM"], p["SEED"], p["PHASE"]), "config differs from the registry")
    require(p["ROUTER"] == "label" and p["ASSOC"] == "pfa" and cfg["DATA"].get("XAG_SOURCE_ROOT"), "not a label-route arm")
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
    from tools.pfo2.train import install_trunk
    register_audit_module()
    from semkine import train as T
    T.XYZPairStreamCallback = T.U1aPairStreamCallback               # B2's pair-stream hashes (the 12 packet fields)
    routing = install_label_routes()
    PT = install_trunk(p["MODULE"], p["TRUNK"])
    if rank0:
        out.mkdir(parents=True, exist_ok=False)
        save(out / "b2ub_launch.json", {"pfo": p, "phase": p["PHASE"], "routing": routing, "trunk": PT.installed(),
                                        "module": p["MODULE"], "config": str(path), "config_sha256": sha(path),
                                        "code_sha256": {c: sha(ROOT / c) for c in CODE},
                                        "xag_source_root": cfg["DATA"]["XAG_SOURCE_ROOT"],
                                        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                        "started_utc": now()})
    sys.argv = [sys.argv[0], "--config", str(path)]
    T.main()


if __name__ == "__main__":
    main()
