#!/usr/bin/env python3
"""EVF training of an arm of the second registry (tools/evf/arms2.py, semkine/evf_model2.py); tools/evf/train.py's
procedure (frozen with round 1). EVF training of a registered arm: semkine/train.py unchanged, with the EVF model in place of
MNISTModel, Lightning's Trainer built with `deterministic=True` (#pfo2-amend1-20261103: without it Lightning undoes the
deterministic flag) and the phase's step count (`--max-steps`; the schedule stays the config's 6000 steps).

    python tools/evf/train2.py --config configs/evf_20261104/evf_<arm>_s<seed>[_<phase>].yaml

Run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (the job builders set it).
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

from tools.evf.arms2 import build_config, config_path, stop_step  # noqa: E402
from tools.evf.common import now, require, save, sha  # noqa: E402

POLICY = {"trainer_deterministic": True, "benchmark": False}
CODE = ("semkine/evf_model.py", "semkine/evf_model2.py", "semkine/evf_graph.py", "semkine/evf_net.py",
        "semkine/evf_nodes.py", "semkine/evf_render.py", "tools/evf/train2.py", "tools/evf/arms2.py")


def install_policy():
    import pytorch_lightning as pl
    native = pl.Trainer

    class PolicyRecord(pl.Callback):
        def on_fit_start(self, trainer, model):
            import torch
            if trainer.global_rank == 0:
                out = Path(model.cfg["TRAIN"]["OUTPUT_DIR"])
                save(out / "evf_det.json", {**POLICY, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                            "recorded_at_utc": now()}, exclusive=True)

    class Trainer(native):
        def __init__(self, *args, **kwargs):
            kwargs.update(deterministic=True, benchmark=False)
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), PolicyRecord()]
            super().__init__(*args, **kwargs)
    pl.Trainer = Trainer


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
                save(self.output_dir / "evf_binding.json", {"evf": model.cfg["EVF"], "recorded_at_utc": now(),
                                                            "config_sha256": sha(model.cfg["_config_path"])})

    def write(path, rows):
        save(path, rows)
    module.XAGTrainAudit, module.write = XAGTrainAudit, write
    package.train_audit = module
    sys.modules["tools.xag"], sys.modules["tools.xag.train_audit"] = package, module


def main():
    # the DDP launcher re-runs this script in rank 1 with the argv rank 0 hands to semkine/train.py (`--max-steps` added)
    argv = sys.argv[1:]
    require(len(argv) in (2, 4) and argv[0] == "--config" and (len(argv) == 2 or argv[2] == "--max-steps"),
            "usage: train.py --config <registered config>")
    path = Path(argv[1]).resolve()
    cfg = yaml.safe_load(path.read_text())
    e = cfg["EVF"]
    require(path == config_path(e["ARM"], e["SEED"], e["PHASE"]).resolve(), "unregistered config path")
    require(cfg == build_config(e["ARM"], e["SEED"], e["PHASE"]), "config differs from the registry")
    require(len(argv) == 2 or int(argv[3]) == stop_step(e["PHASE"]), "step count differs from the phase")
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
    register_audit_module()
    from semkine import train as T
    from semkine.evf_model2 import EVFModel2 as EVFModel
    T.XYZPairStreamCallback = T.U1aPairStreamCallback               # packets carry no XAG source
    T.MNISTModel = EVFModel
    if rank0:
        out.mkdir(parents=True, exist_ok=False)
        save(out / "evf_launch.json", {"evf": e, "config": str(path), "config_sha256": sha(path),
                                       "code_sha256": {c: sha(ROOT / c) for c in CODE},
                                       "stop_step": stop_step(e["PHASE"]),
                                       "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                       "started_utc": now()})
    sys.argv = [sys.argv[0], "--config", str(path), "--max-steps", str(stop_step(e["PHASE"]))]
    T.main()


if __name__ == "__main__":
    main()
