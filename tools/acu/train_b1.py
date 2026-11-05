#!/usr/bin/env python3
"""ACU stage-1b training of b1 (tools/acu/arms_b1.py, docs/EXPERIMENT_RECORDS.md#acu-s1b-prereg-20261104): B2's
procedure (tools/pfo2/train.py: semkine/train.py unchanged, the deployed r_s2d router with PFA, the gshare trunk of
semkine/pfo_trunk4.py, deterministic flags, Lightning's Trainer with deterministic=True) plus semkine/acu_b1.py's
dataset, collate, forward and trunk wrappers in every DDP rank.

    python tools/acu/train_b1.py --config configs/acu_20261104/acu_b1_s3407[_preflight].yaml
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

from tools.acu.arms_b1 import build_config, config_path  # noqa: E402
from tools.acu.common import now, require, save, sha  # noqa: E402

CODE = ("tools/acu/train_b1.py", "tools/acu/arms_b1.py", "semkine/acu_b1.py", "semkine/pfo_trunk4.py", "semkine/s45sm.py",
        "semkine/pfo_router.py", "semkine/train.py", "semkine/dataset.py", "semkine/events.py", "semkine/encoder.py",
        "model/model.py", "semkine/xyz_mesh.py")


def main():
    require(len(sys.argv) == 3 and sys.argv[1] == "--config", "usage: train_b1.py --config <registered config>")
    path = Path(sys.argv[2]).resolve()
    cfg = yaml.safe_load(path.read_text())
    a, p = cfg["ACU"], cfg["PFO"]
    require(path == config_path(a["ARM"], p["SEED"], p["PHASE"]).resolve(), "unregistered config path")
    require(cfg == build_config(a["ARM"], p["SEED"], p["PHASE"]), "config differs from the registry")
    require(a["UPDATE"] == "b1" and a["IFACE"] == "pfa", "not the b1 arm")
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
    from tools.acu.train import install_policy, install_routes
    from tools.pfo.train import register_audit_module
    from tools.pfo2.train import install_trunk
    install_policy()
    register_audit_module()
    from semkine import train as T
    from semkine import acu_b1 as B1
    from semkine import xyz_mesh as XM
    from model import MNISTModel
    T.XYZPairStreamCallback = T.U1aPairStreamCallback
    routing = install_routes(p)
    install_trunk(p["MODULE"], p["TRUNK"])
    trunk = B1.install_trunk(XM)
    B1.install_dataset()
    B1.install_collate(T)
    B1.install_model(MNISTModel)
    if rank0:
        out.mkdir(parents=True, exist_ok=False)
        save(out / "acu_launch.json", {"acu": a, "pfo": p, "phase": p["PHASE"], "routing": routing, "trunk": trunk,
                                       "b1": {"p_start": B1.P_START, "p_uniform": B1.P_UNIFORM, "tag": B1.TAG_B1,
                                              "age_unit_s": B1.AGE_UNIT_S},
                                       "config": str(path), "config_sha256": sha(path),
                                       "code_sha256": {c: sha(ROOT / c) for c in CODE},
                                       "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                       "started_utc": now()})
    sys.argv = [sys.argv[0], "--config", str(path)]
    T.main()


if __name__ == "__main__":
    main()
