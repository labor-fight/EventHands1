#!/usr/bin/env python3
"""ACU stage-2 training of a 2x2 cell (tools/acu/arms_s2.py, docs/EXPERIMENT_RECORDS.md#acu-s2-prereg-20261104).

B2's procedure (semkine/train.py unchanged; deterministic flags; Lightning's Trainer with deterministic=True; B2's
pair-stream hashes) on the causal pipeline of semkine/acu_causal.py in every DDP rank: the history-carrying training
packets (+ B1's prior time for b1 cells), the causal-stride nodes, the history-aware tokens, PR's routes (the frozen PR
router accepted by s1c), the cell's association (PFA or CA) and trunk (gshare / nomp, + B1's age term for b1 cells).

    python tools/acu/train_s2.py --config configs/acu_20261104/acu_s2_<arm>_s<seed>[_preflight].yaml
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

from tools.acu.arms_s2 import build_config, config_path  # noqa: E402
from tools.acu.common import OUT, now, read, require, save, sha  # noqa: E402

CODE = ("tools/acu/train_s2.py", "tools/acu/arms_s2.py", "semkine/acu_causal.py", "semkine/acu_pr.py", "semkine/acu_hist.py",
        "semkine/acu_ca.py", "semkine/acu_b1.py", "semkine/evf_render.py", "semkine/pfo_trunk4.py", "semkine/s45sm.py",
        "semkine/train.py", "semkine/dataset.py", "semkine/events.py", "semkine/encoder.py", "model/model.py",
        "semkine/xyz_mesh.py")


def install_trunk(a):
    from semkine import acu_b1 as B1
    from semkine import xyz_mesh as XM
    if a["IFACE"] == "ca":
        from semkine import acu_ca as CA
        CA.install(a["TRUNK"])
    else:
        from tools.pfo2.train import install_trunk as T
        T("trunk4", a["TRUNK"])
    name = XM.XYZMeshNet.__name__
    if a["UPDATE"] == "b1":
        name = B1.install_trunk(XM)
    return name


def main():
    require(len(sys.argv) == 3 and sys.argv[1] == "--config", "usage: train_s2.py --config <registered config>")
    path = Path(sys.argv[2]).resolve()
    cfg = yaml.safe_load(path.read_text())
    a, p = cfg["ACU"], cfg["PFO"]
    require(path == config_path(a["CELL"], p["SEED"], p["PHASE"], a["TRUNK"]).resolve(), "unregistered config path")
    require(cfg == build_config(a["CELL"], p["SEED"], p["PHASE"], a["TRUNK"]), "config differs from the registry")
    rec = read(Path(a["PR_CHECKPOINT"]).with_suffix(".json"))
    require(rec["status"] == "COMPLETE" and sha(a["PR_CHECKPOINT"]) == rec["checkpoint_sha256"], "PR router not accepted")
    require(read(OUT / "receipts" / "pr_formal.json")["status"] == "PASS", "PR router check missing")
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
    from tools.acu.train import install_policy
    from tools.pfo.train import register_audit_module
    install_policy()
    register_audit_module()
    from semkine import acu_causal as AC
    from semkine import train as T
    from model import MNISTModel
    T.XYZPairStreamCallback = T.U1aPairStreamCallback
    trunk = install_trunk(a)
    AC.install_dataset(b1=a["UPDATE"] == "b1")
    AC.install_collate(T)
    hooks = AC.install_model(MNISTModel, a["PR_CHECKPOINT"], a["IFACE"], float(a["ONHAND_TAU"]), b1=a["UPDATE"] == "b1")
    if rank0:
        out.mkdir(parents=True, exist_ok=False)
        save(out / "acu_launch.json", {"acu": a, "pfo": p, "phase": p["PHASE"], "trunk": trunk, "hooks": hooks,
                                       "pr_checkpoint_sha256": sha(a["PR_CHECKPOINT"]), "config": str(path),
                                       "config_sha256": sha(path), "code_sha256": {c: sha(ROOT / c) for c in CODE},
                                       "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                       "started_utc": now()})
    sys.argv = [sys.argv[0], "--config", str(path)]
    T.main()


if __name__ == "__main__":
    main()
