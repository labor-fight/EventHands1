#!/usr/bin/env python3
"""CMP training entry (tools/cmp/arms.py, docs/EXPERIMENT_RECORDS.md#cmp-prereg-20261110): semkine/train.py unchanged,
with, in the one process of the run:
  model   the CMP class of CMP.MODEL (semkine/cmp_models.py, semkine/cmp_evhp.py) built in place of MNISTModel;
          MODEL.INIT_FROM warm-starts it from the seed's earlier-stage checkpoint (semkine/train.py's name-matched copy)
  data    sequence stages (CMP.STAGE: seq): the training dataset re-indexed into K packets (semkine/cmp_seq.py `upgrade`);
          frame stages: the dataset as built (legacy LNES, fast pipe)
  policy  deterministic flags, Lightning's Trainer with `deterministic=True` (CMP.DETERMINISTIC "warn": warn-only, for
          ops without a deterministic CUDA kernel), CUBLAS_WORKSPACE_CONFIG; fresh-only output
  audit   cmp_launch.json (config / code SHA, devices), cmp_timing.json (per-step wall time, peak CUDA memory)

    python tools/cmp/train.py --config configs/cmp_20261110/cmp_<arm>_s<seed>[_smoke].yaml
"""
from __future__ import annotations

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

from tools.cmp.arms import ARMS, EXT, build_config, config_path  # noqa: E402
from tools.cmp.common import now, require, save, sha  # noqa: E402

CODE = ("tools/cmp/train.py", "tools/cmp/arms.py", "tools/cmp/common.py", "semkine/cmp_models.py", "semkine/cmp_seq.py",
        "semkine/cmp_evhp.py", "semkine/cmp_seq_crop.py", "semkine/cmp_evhpc.py", "semkine/cmp_eho.py", "semkine/train.py", "semkine/dataset.py", "semkine/events.py", "semkine/domrand.py",
        "model/model.py", "model/mano_layer.py")


def callbacks():
    import pytorch_lightning as pl
    import torch

    class Timing(pl.Callback):
        def __init__(self):
            self.rows, self.t0 = [], None

        def _write(self, model):
            out = Path(model.cfg["TRAIN"]["OUTPUT_DIR"]) / "cmp_timing.json"
            out.write_text(json.dumps({"steps": self.rows, "max_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                                       "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                       "written_utc": now()}, indent=1) + "\n")

        def on_train_batch_start(self, trainer, model, batch, batch_idx):
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            torch.cuda.synchronize()
            self.rows.append({"step": int(trainer.global_step), "s": round(time.perf_counter() - self.t0, 4),
                              "alloc_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)})
            if len(self.rows) <= 40 or len(self.rows) % 100 == 0:
                self._write(model)

        def on_train_end(self, trainer, model):
            self._write(model)
    return [Timing()]


def install_policy(deterministic=True):
    import pytorch_lightning as pl
    native = pl.Trainer

    class Trainer(native):
        def __init__(self, *args, **kwargs):
            kwargs.update(deterministic=deterministic, benchmark=False)
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), *callbacks()]
            super().__init__(*args, **kwargs)
    pl.Trainer = Trainer


def main():
    require(len(sys.argv) == 3 and sys.argv[1] == "--config", "usage: train.py --config <registered config>")
    path = Path(sys.argv[2]).resolve()
    cfg = yaml.safe_load(path.read_text())
    c = cfg["CMP"]
    require(c["ARM"] in ARMS, "unregistered arm")
    require(path == config_path(c["ARM"], c["SEED"], c["PHASE"]).resolve(), "unregistered config path")
    require(cfg == build_config(c["ARM"], c["SEED"], c["PHASE"]), "config differs from the registry")
    out = Path(cfg["TRAIN"]["OUTPUT_DIR"])
    require(not out.exists(), "fresh-only output already exists: " + str(out))
    init = cfg["MODEL"].get("INIT_FROM")
    require(init is None or Path(init).exists(), f"missing INIT_FROM checkpoint {init}")
    import torch
    det = c.get("DETERMINISTIC", True)
    require(det in (True, "warn"), "CMP.DETERMINISTIC is true or warn")
    torch.use_deterministic_algorithms(True, warn_only=det == "warn")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    install_policy(det)
    from semkine import train as T
    from semkine.cmp_models import model_class
    T.MNISTModel = model_class(c["MODEL"])

    class Throughput(T.ThroughputCallback):
        """semkine/train.py's throughput log, also for the dict batches of the sequence stages."""

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            super().on_train_batch_end(trainer, pl_module, outputs,
                                       (batch["lnes"],) if isinstance(batch, dict) else batch, batch_idx)
    T.ThroughputCallback = Throughput
    ext = EXT.get(c["ARM"])                                      # amendment 6: an extension arm's own dataset
    if ext is not None and hasattr(ext, "build_dataset"):
        native_build = T.build_dataset

        def build(cfg_, split, components, train=None, **kwargs):
            return ext.build_dataset(native_build(cfg_, split, components, train=train, **kwargs), cfg_, split)
        T.build_dataset = build
    elif c["STAGE"] == "seq":
        from semkine import cmp_seq as S
        native_build = T.build_dataset
        s = c["SEQ"]

        def build(cfg_, split, components, train=None, **kwargs):
            ds = native_build(cfg_, split, components, train=train, **kwargs)
            if split != "train":
                return ds
            if "CROP" in c:                                       # amendment 1: EvHandPose's hand crop
                from semkine import cmp_seq_crop as SC
                return SC.upgrade(ds, int(s["K"]), int(s["PACKET_MS"]), int(s["VOXEL_BINS"]), int(c["CROP"]["SIZE"]),
                                  float(c["CROP"]["RATE"]), float(c["CROP"]["SHIFT"]))
            return S.upgrade(ds, int(s["K"]), int(s["PACKET_MS"]), int(s.get("VOXEL_BINS", 0)))
        T.build_dataset = build
    out.mkdir(parents=True, exist_ok=False)
    save(out / "cmp_launch.json", {"cmp": c, "config": str(path), "config_sha256": sha(path),
                                   "code_sha256": {f: sha(ROOT / f) for f in (*CODE, *getattr(ext, "CODE", ()))
                                                   if (ROOT / f).exists()},
                                   "init_from": init, "init_from_sha256": sha(init) if init else None,
                                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                   "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "pid": os.getpid(),
                                   "started_utc": now()}, exclusive=True)
    sys.argv = [sys.argv[0], "--config", str(path)]
    T.main()


if __name__ == "__main__":
    main()
