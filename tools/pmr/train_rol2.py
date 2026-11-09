#!/usr/bin/env python3
"""PMR stage S5 rollout training (tools/pmr/arms_rol2.py, docs/EXPERIMENT_RECORDS.md#pmr-s5-prereg-20261109):
tools/pmr/train_rol.py (frozen) with the union registry (any PMR base arm). TPD's ROL
procedure (tools/tpd/train_rol.py: semkine/train.py unchanged; deterministic flags; Lightning's Trainer built with
`deterministic=True`; B2's router hooks r_s2d + PFA; semkine/tpd_rol.py's rollout dataset, rollout model class and
start check; the rollout stream / timing / start-check audits) with the PMR arm's trunk installed in every DDP rank in
place of B2's gshare trunk. The PMR trunk computes its geometry with gradient, so `install_grad_geometry` (B2's edge
geometry patch) is not needed and not installed. Runs with NCCL_P2P_DISABLE=1 (#pmr-prereg-20261109).

    python tools/pmr/train_rol2.py --config configs/pmr_20261109/pmr_<arm>_rol5_s<seed>[_preflight].yaml
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402

from tools.pmr import arms_any2 as PA  # noqa: E402
from tools.pmr.arms_rol2 import build_config, config_path, parse  # noqa: E402
from tools.pmr.common import now, require, save, sha  # noqa: E402

POLICY = {"trainer_deterministic": True, "benchmark": False}
CODE = ("tools/pmr/train_rol2.py", "tools/pmr/arms_rol2.py", "tools/pmr/arms_any2.py", "tools/pmr/common.py",
        "semkine/tpd_rol.py", "semkine/pmr_trunk.py", "tools/pfo2/arms.py", "tools/pfo/train.py", "semkine/s45sm.py",
        "semkine/pfo_router.py", "semkine/xyz_mesh.py", "semkine/train.py", "semkine/dataset.py", "semkine/events.py",
        "semkine/xag.py", "model/model.py", "model/mano_layer.py")
FIELDS = ("sequence_id", "t_start_us", "t_end_us", "target", "prev_state", "events", "ptr", "delta_t_s", "betas",
          "camera_K", "is_sequence_start", "is_sequence_end")


def callbacks():
    import pytorch_lightning as pl
    import torch

    class PolicyRecord(pl.Callback):
        def on_fit_start(self, trainer, model):
            if trainer.global_rank == 0:
                out = Path(model.cfg["TRAIN"]["OUTPUT_DIR"])
                save(out / "pmr_det.json", {**POLICY, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                            "recorded_at_utc": now()}, exclusive=True)

    class Timing(pl.Callback):
        def __init__(self):
            self.rows, self.t0 = [], None

        def _write(self, trainer, model):
            out = Path(model.cfg["TRAIN"]["OUTPUT_DIR"]) / f"rol_timing_rank{trainer.global_rank}.json"
            out.write_text(json.dumps({"steps": self.rows, "max_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                                       "max_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
                                       "written_utc": now()}, indent=1) + "\n")

        def on_train_batch_start(self, trainer, model, batch, batch_idx):
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            torch.cuda.synchronize()
            self.rows.append({"step": int(trainer.global_step), "s": round(time.perf_counter() - self.t0, 4),
                              "alloc_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)})
            if len(self.rows) <= 60 or len(self.rows) % 50 == 0:
                self._write(trainer, model)

        def on_train_end(self, trainer, model):
            self._write(trainer, model)
    return [PolicyRecord(), Timing()]


def install_policy():
    import pytorch_lightning as pl
    native = pl.Trainer

    class Trainer(native):
        def __init__(self, *args, **kwargs):
            kwargs.update(deterministic=True, benchmark=False)
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), *callbacks()]
            super().__init__(*args, **kwargs)
    pl.Trainer = Trainer


def stream_callback():
    import pytorch_lightning as pl

    class RolloutStream(pl.Callback):
        """The first two training batches of each rank: one hash over every packet of the rollout (12 B2 fields)."""

        def __init__(self, output_dir):
            self.output_dir = Path(output_dir)
            self.rows = []

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if len(self.rows) >= 2:
                return
            h = hashlib.sha256()
            for pk in batch.packets:
                for name in FIELDS:
                    h.update(getattr(pk, name).detach().cpu().contiguous().numpy().tobytes())
            self.rows.append({"step": int(trainer.global_step), "sha256": h.hexdigest(), "samples": batch.batch_size,
                              "packets": len(batch.packets), "events": [int(p.events.shape[0]) for p in batch.packets]})
            (self.output_dir / f"rollout_stream_rank{trainer.global_rank}.json").write_text(json.dumps(self.rows, indent=2))
    return RolloutStream


def main():
    require(len(sys.argv) == 3 and sys.argv[1] == "--config", "usage: train_rol.py --config <registered config>")
    path = Path(sys.argv[2]).resolve()
    cfg = yaml.safe_load(path.read_text())
    p = cfg["PFO"]
    parse(p["ARM"])
    require(path == config_path(p["ARM"], p["SEED"], p["PHASE"]).resolve(), "unregistered config path")
    require(cfg == build_config(p["ARM"], p["SEED"], p["PHASE"]), "config differs from the registry")
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
    from semkine import pfo_router as PR
    from semkine import s45sm as SM
    from semkine import tpd_rol as R
    from semkine.events import EventPacket
    require(p["ROUTER_KIND"] == "pfr" and p["ASSOC"] == "pfa" and p["ROUTER"] == "r_s2d" and p["TRUNK"] == "pmr",
            "PMR rollout arms keep B2's router block and the PMR trunk")
    PR.register()
    routing = SM.install(p["ROUTER_KIND"], p["ROUTER_CHECKPOINT"], p["ASSOC"], float(p["ONHAND_TAU"]))
    trunk = PA.install_trunk(cfg)
    geom = "pmr_trunk (gradient geometry built in)"
    T.XYZPairStreamCallback = stream_callback()
    native_build = T.build_dataset
    r = cfg["ROL"]

    def build(cfg_, split, components, train=None, **kwargs):
        ds = native_build(cfg_, split, components, train=train, **kwargs)
        return R.upgrade(ds, int(r["K"]), int(r["PACKET_MS"])) if split == "train" else ds
    T.build_dataset = build
    T.EventPacket = (EventPacket, R.RolloutItem)
    T.collate_packets = R.collate_any
    T.MNISTModel = R.rol_model_class()
    if rank0:
        out.mkdir(parents=True, exist_ok=False)
        save(out / "pmr_launch.json", {"pfo": p, "pmr": cfg["PMR"], "rol": r, "phase": p["PHASE"], "routing": routing,
                                       "trunk": trunk, "geometry": geom, "config": str(path),
                                       "config_sha256": sha(path), "code_sha256": {c: sha(ROOT / c) for c in CODE},
                                       "router_checkpoint_sha256": sha(p["ROUTER_CHECKPOINT"]),
                                       "init_from": cfg["MODEL"]["INIT_FROM"],
                                       "init_from_sha256": sha(cfg["MODEL"]["INIT_FROM"]),
                                       "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                       "started_utc": now()})
    sys.argv = [sys.argv[0], "--config", str(path)]
    T.main()


if __name__ == "__main__":
    main()
