#!/usr/bin/env python3
"""TPD synthesis / final-configuration training (semkine/tpd_final.py; the stage's registration names the arms): B2's
training procedure (semkine/train.py unchanged; deterministic flags; Trainer deterministic=True; the gshare trunk) with,
in every DDP rank, the config's FINAL block:
  CAUSAL   true: the trunk's nodes by the causal stride and its tokens with the causal fill (semkine/tpd_causal.py);
           every training / validation packet carries n_prev (semkine/tpd_final.py datasets)
  ROUTES   label_part: B2's hooks, then the label_part mix (tools/pfo/nets.py `mixed_sources`, IDP L1's training
           routes; the batches carry the XAG pseudo-sources) -- the mix evaluates the router at the trunk's nodes;
           router: semkine/s45sm_fast.py's deployed routes (the router at the trunk's nodes)
  ROLLOUT  true: ROL's K-packet samples (the config's ROL block) and step; false: B2's variable windows and step
The model is semkine/tpd_final.py `final_model_class()` (cosine to 0.02 of the peak, B2's floor); MODEL.INIT_FROM when set.

    python tools/tpd/train_final.py --config configs/tpd_20261105/<registered config>.yaml
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

from tools.tpd.common import now, require, save, sha  # noqa: E402
from tools.tpd.train_rol import install_policy, stream_callback  # noqa: E402

CODE = ("tools/tpd/train_final.py", "tools/tpd/train_rol.py", "semkine/tpd_final.py", "semkine/tpd_causal.py",
        "semkine/tpd_rol.py", "semkine/acu_pr.py", "tools/pfo/nets.py", "tools/pfo2/train.py", "semkine/s45sm.py",
        "semkine/s45sm_fast.py", "semkine/pfo_router.py", "semkine/pfo_trunk4.py", "semkine/xyz_mesh.py",
        "semkine/encoder.py", "semkine/train.py", "semkine/dataset.py", "semkine/events.py", "semkine/xag.py",
        "model/model.py")


def install_routes(p, mode):
    import torch
    from semkine import pfo_router as PR
    from semkine import s45sm as SM
    from semkine import s45sm_fast as FAST
    from semkine import xag as XAG
    require(p["ROUTER_KIND"] == "pfr" and p["ASSOC"] == "pfa", "the final configuration keeps B2's router block")
    PR.register()
    if mode == "label_part":
        state = SM.install(p["ROUTER_KIND"], p["ROUTER_CHECKPOINT"], p["ASSOC"], float(p["ONHAND_TAU"]))
        from tools.pfo import nets as N
        N._MIX.update(mode="label_part", gen=torch.Generator().manual_seed(0))
        XAG.sample_sources = N.mixed_sources
        return {**state, "train_routes": "label_part", "sample_sources": "tools.pfo.nets.mixed_sources"}
    require(mode == "router", "FINAL.ROUTES: label_part | router")
    state = FAST.install(p["ROUTER_KIND"], p["ROUTER_CHECKPOINT"], p["ASSOC"], float(p["ONHAND_TAU"]))
    return {**state, "train_routes": "router", "sample_sources": "semkine.s45sm_fast.predicted_sources"}


def main():
    require(len(sys.argv) == 3 and sys.argv[1] == "--config", "usage: train_final.py --config <registered config>")
    path = Path(sys.argv[2]).resolve()
    cfg = yaml.safe_load(path.read_text())
    p, f = cfg["PFO"], cfg["FINAL"]
    from tools.tpd.arms_final import build_config, config_path
    require(path == config_path(p["ARM"], p["SEED"], p["PHASE"]).resolve(), "unregistered config path")
    require(cfg == build_config(p["ARM"], p["SEED"], p["PHASE"]), "config differs from the registry")
    require((f["ROUTES"] == "label_part") == bool(cfg["DATA"].get("XAG_SOURCE_ROOT")), "source root vs routes")
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
    from semkine import tpd_final as TF
    from semkine import tpd_rol as R
    from semkine.events import EventPacket
    from tools.pfo2.train import install_trunk
    require(p["TRUNK"] == "gshare" and p["MODULE"] == "trunk4", "the gshare trunk")
    routing = install_routes(p, f["ROUTES"])
    PT = install_trunk(p["MODULE"], p["TRUNK"])
    geom = R.install_grad_geometry()
    rollout = bool(f["ROLLOUT"])
    T.XYZPairStreamCallback = stream_callback() if rollout else T.U1aPairStreamCallback
    native_build = T.build_dataset
    r = cfg["ROL"]

    def build(cfg_, split, components, train=None, **kwargs):
        ds = native_build(cfg_, split, components, train=train, **kwargs)
        if split == "train" and rollout:
            return TF.upgrade_rollout(ds, int(r["K"]), int(r["PACKET_MS"]))
        return TF.upgrade_plain(ds)
    T.build_dataset = build
    T.EventPacket = (EventPacket, R.RolloutItem)
    T.collate_packets = TF.collate_final
    T.MNISTModel = TF.final_model_class()
    if rank0:
        out.mkdir(parents=True, exist_ok=False)
        save(out / "tpd_launch.json", {"pfo": p, "final": f, "rol": r, "phase": p["PHASE"], "routing": routing,
                                       "trunk": PT.installed(), "geometry": geom, "config": str(path),
                                       "config_sha256": sha(path), "code_sha256": {c: sha(ROOT / c) for c in CODE},
                                       "router_checkpoint_sha256": sha(p["ROUTER_CHECKPOINT"]),
                                       "init_from": cfg["MODEL"].get("INIT_FROM"),
                                       "xag_source_root": cfg["DATA"].get("XAG_SOURCE_ROOT"),
                                       "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                       "started_utc": now()})
    sys.argv = [sys.argv[0], "--config", str(path)]
    T.main()


if __name__ == "__main__":
    main()
