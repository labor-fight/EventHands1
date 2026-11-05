#!/usr/bin/env python3
"""PFO2 training of a registered arm (tools/pfo2/arms.py): tools/pfo/train4.py's procedure (semkine/train.py unchanged; in
every DDP rank the config's frozen router + hard PFA, the trunk variant, deterministic flags, no XAG source in the
packets) with the PFO2 registry and the arm's trunk module (semkine/pfo_trunk4.py, or semkine/pfo_trunk3.py for the
single-change reference runs).

    python tools/pfo2/train.py --config configs/pfo2_20261103/pfo2_<arm>_s<seed>[_preflight|_head500].yaml
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
from tools.pfo2.arms import build_config, config_path  # noqa: E402
from tools.pfo2.common import now, require, save, sha  # noqa: E402


def trunk_module(module):
    if module == "trunk3":
        from semkine import pfo_trunk3 as PT
    else:
        from semkine import pfo_trunk4 as PT
    return PT


def install_trunk(module, trunk):
    """Install the arm's trunk; "base" is trunk4's "full" (tools/pfo/train4.py `trunk4_variant`)."""
    PT = trunk_module(module)
    PT.install(("full" if trunk == "base" else trunk) if module == "trunk4" else trunk)
    return PT


def main():
    require(len(sys.argv) == 3 and sys.argv[1] == "--config", "usage: train.py --config <registered config>")
    path = Path(sys.argv[2]).resolve()
    cfg = yaml.safe_load(path.read_text())
    p = cfg["PFO"]
    phase = p["PHASE"]
    require(path == config_path(p["ARM"], p["SEED"], phase).resolve(), "unregistered config path")
    require(cfg == build_config(p["ARM"], p["SEED"], phase), "config differs from the registry")
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
    T.XYZPairStreamCallback = T.U1aPairStreamCallback
    if p["ROUTER_KIND"] == "pfr":
        from semkine import pfo_router as PR
        PR.register()
    state = SM.install(p["ROUTER_KIND"], p["ROUTER_CHECKPOINT"], p["ASSOC"], float(p["ONHAND_TAU"]))
    PT = install_trunk(p["MODULE"], p["TRUNK"])
    bank = None
    if p.get("BANK"):
        from semkine import s45tr_prior as P
        P.install_n4(p["BANK"]["root"], float(p["BANK"]["p"]))
        bank = p["BANK"]
    if os.environ.get("LOCAL_RANK") in (None, "0"):
        out.mkdir(parents=True, exist_ok=False)
        save(out / "pfo2_launch.json", {"pfo": p, "phase": phase, "routing": state, "trunk": PT.installed(),
                                        "module": p["MODULE"], "bank": bank, "config": str(path),
                                        "config_sha256": sha(path), "train_wrapper_sha256": sha(__file__),
                                        "trunk_module_sha256": sha(Path(PT.__file__)),
                                        "router_checkpoint_sha256": sha(p["ROUTER_CHECKPOINT"]),
                                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                        "started_utc": now()})
    sys.argv = [sys.argv[0], "--config", str(path)]
    T.main()


if __name__ == "__main__":
    main()
