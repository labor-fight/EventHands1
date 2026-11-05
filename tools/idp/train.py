#!/usr/bin/env python3
"""IDP training of a registered arm (tools/idp/arms.py, docs/EXPERIMENT_RECORDS.md#idp-prereg-20261105): B2's training
procedure (tools/pfo2/train.py: semkine/train.py unchanged; deterministic flags; Lightning's Trainer built with
`deterministic=True`, #pfo2-amend1-20261103; the gshare trunk; the pair-stream hashes) with up to two registered
changes in every DDP rank:
  PFO.TRAIN_ROUTES  router      B2's hooks exactly: semkine/s45sm.py `install` (r_s2d, PFA, tau 0.5)
                    label_part  the same hooks, then `semkine.xag.sample_sources` = tools/pfo/nets.py `mixed_sources` in
                                mode `label_part` (the router's on-hand set and template point, the XAG label's part;
                                B2UB's U1 definition, so the training routes equal the engine's test-time routes)
  PFO.FEAT          none        the gshare trunk (semkine/pfo_trunk4.py)
                    fore        semkine/idp_feat.py (the trunk plus the zero-initialised record feature)

    python tools/idp/train.py --config configs/idp_20261105/idp_<arm>_s3407[_preflight].yaml
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

from tools.idp.arms import ARMS, build_config, config_path  # noqa: E402
from tools.idp.common import now, require, save, sha  # noqa: E402

POLICY = {"trainer_deterministic": True, "benchmark": False}
CODE = ("tools/idp/train.py", "tools/idp/arms.py", "tools/pfo2/arms.py", "tools/pfo2/train.py", "tools/pfo/nets.py",
        "semkine/s45sm.py", "semkine/s45sm_fast.py", "semkine/pfo_router.py", "semkine/pfo_trunk4.py",
        "semkine/idp_feat.py", "semkine/train.py", "semkine/dataset.py", "semkine/xag.py")


def install_policy():
    import pytorch_lightning as pl
    native = pl.Trainer

    class PolicyRecord(pl.Callback):
        def on_fit_start(self, trainer, model):
            import torch
            if trainer.global_rank == 0:
                out = Path(model.cfg["TRAIN"]["OUTPUT_DIR"])
                save(out / "idp_det.json", {**POLICY, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                            "recorded_at_utc": now()}, exclusive=True)

    class Trainer(native):
        def __init__(self, *args, **kwargs):
            kwargs.update(deterministic=True, benchmark=False)
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), PolicyRecord()]
            super().__init__(*args, **kwargs)
    pl.Trainer = Trainer


def install_routes(p):
    """B2's training hooks; `label_part` additionally swaps the source sampler for the engine's mix (one process)."""
    import torch
    from semkine import pfo_router as PR
    from semkine import s45sm as SM
    from semkine import xag as XAG
    require(p["ROUTER_KIND"] == "pfr" and p["ASSOC"] == "pfa", "IDP arms keep B2's router block")
    PR.register()
    state = SM.install(p["ROUTER_KIND"], p["ROUTER_CHECKPOINT"], p["ASSOC"], float(p["ONHAND_TAU"]))
    state = {**state, "train_routes": p["TRAIN_ROUTES"], "sample_sources": "semkine.s45sm.predicted_sources"}
    if p["TRAIN_ROUTES"] == "label_part":
        from tools.pfo import nets as N
        N._MIX.update(mode="label_part", gen=torch.Generator().manual_seed(0))
        XAG.sample_sources = N.mixed_sources
        state.update(mix="label_part", sample_sources="tools.pfo.nets.mixed_sources")
    return state


def install_trunk(p):
    if p["FEAT"] == "fore":
        from semkine import idp_feat as IF
        return {"module": "idp_feat", "variant": IF.install(), "feat": "fore", "trunk": p["TRUNK"]}
    from tools.pfo2.train import install_trunk as base_install
    PT = base_install(p["MODULE"], p["TRUNK"])
    return {"module": p["MODULE"], "variant": PT.installed(), "feat": "none", "trunk": p["TRUNK"]}


def main():
    require(len(sys.argv) == 3 and sys.argv[1] == "--config", "usage: train.py --config <registered config>")
    path = Path(sys.argv[2]).resolve()
    cfg = yaml.safe_load(path.read_text())
    p = cfg["PFO"]
    require(p["ARM"] in ARMS, "unregistered arm")
    require(path == config_path(p["ARM"], p["SEED"], p["PHASE"]).resolve(), "unregistered config path")
    require(cfg == build_config(p["ARM"], p["SEED"], p["PHASE"]), "config differs from the registry")
    require((p["TRAIN_ROUTES"] == "label_part") == bool(cfg["DATA"].get("XAG_SOURCE_ROOT")), "source root vs routes")
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
    trunk = install_trunk(p)
    if rank0:
        out.mkdir(parents=True, exist_ok=False)
        save(out / "idp_launch.json", {"pfo": p, "phase": p["PHASE"], "routing": routing, "trunk": trunk,
                                       "config": str(path), "config_sha256": sha(path),
                                       "code_sha256": {c: sha(ROOT / c) for c in CODE},
                                       "router_checkpoint_sha256": sha(p["ROUTER_CHECKPOINT"]),
                                       "xag_source_root": cfg["DATA"].get("XAG_SOURCE_ROOT"),
                                       "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                       "started_utc": now()})
    sys.argv = [sys.argv[0], "--config", str(path)]
    T.main()


if __name__ == "__main__":
    main()
