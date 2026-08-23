#!/usr/bin/env python3
"""S1 training entry point on the SemKine protocol.

Differences from `model/train_abs.py`, each one forced by a finding in the audit:

* data comes from `semkine.dataset.SemKineDataset`, so the subject-disjoint splits and the
  domain randomisation arm are reachable. With `INPUT_MODE: legacy_lnes` and
  `AUG.DOMRAND.ENABLED: false` the samples are bitwise identical to the legacy loader
  (`tests/test_s1_dataset.py::test_legacy_lnes_is_bitwise_identical`).
* checkpoints are always saved on a fixed step grid. Selection by `val_loss` is not offered,
  because `val_loss` is anti-correlated with the closed-loop metric this project reports
  (`experiment_history.md` §8.5: Spearman -0.07). Choosing among grid points is then a separate,
  explicit step (`tools/select_checkpoint.py`), scored by the recursive evaluator.
* `SEED` shifts the augmentation stream as well as the initialisation, so a "replicate" is a
  genuinely independent draw rather than the same data order with different weights.

The legacy trainer is untouched and remains the way to reproduce the frozen baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from config import load_config          # noqa: E402
from model import MNISTModel            # noqa: E402

from semkine import domrand as DR       # noqa: E402
from semkine.dataset import build_dataset   # noqa: E402
from semkine.events import EventPacket, collate_packets  # noqa: E402


class ThroughputCallback(pl.Callback):
    def __init__(self):
        self.t0 = None
        self.n = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank != 0:
            return
        if self.t0 is None:
            self.t0 = time.time()
            self.n = 0
        n = (batch.batch_size if hasattr(batch, "batch_size") else batch[0].shape[0])
        self.n += n * max(trainer.world_size, 1)
        if (batch_idx + 1) % 50 == 0:
            sps = self.n / max(time.time() - self.t0, 1e-6)
            mem = (torch.cuda.max_memory_allocated() / (1024 ** 3)
                   if torch.cuda.is_available() else 0.0)
            pl_module.log("train_samples_per_sec", sps, prog_bar=True, rank_zero_only=True)
            pl_module.log("cuda_max_mem_gb", mem, prog_bar=True, rank_zero_only=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--devices", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="overrides SEED, for replicates")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--no-logger", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["SEED"] = int(args.seed)
    tcfg = cfg["TRAIN"]
    if args.run_name:
        tcfg["RUN_NAME"] = args.run_name
    if args.output_dir:
        tcfg["OUTPUT_DIR"] = args.output_dir
    pl.seed_everything(int(cfg.get("SEED", 0)), workers=True)

    components = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    dr = DR.DomRandConfig.from_cfg(cfg)
    train_ds = build_dataset(cfg, "train", components, train=True)
    val_ds = build_dataset(cfg, "val_core", components, train=False)
    print(f"seed={cfg['SEED']} input_mode={train_ds.input_mode} domrand={dr.enabled}\n"
          f"train: {len(train_ds.sequences)} seqs / {len(train_ds)} samples\n"
          f"val_core: {len(val_ds.sequences)} seqs / {len(val_ds)} samples", flush=True)

    bsz = int(args.batch_size or tcfg["BATCH_SIZE_PER_GPU"])
    nw = int(args.num_workers if args.num_workers is not None else tcfg["NUM_WORKERS"])
    persistent = bool(tcfg.get("PERSISTENT_WORKERS", True)) and nw > 0

    def _collate(items):
        if isinstance(items[0], EventPacket):
            return collate_packets(items)
        return torch.utils.data.default_collate(items)

    raw = train_ds.input_mode != "legacy_lnes"
    train_loader = DataLoader(
        train_ds, batch_size=bsz, shuffle=True, num_workers=nw,
        pin_memory=bool(tcfg.get("PIN_MEMORY", True)) and nw > 0,
        persistent_workers=persistent,
        prefetch_factor=int(tcfg.get("PREFETCH_FACTOR", 2)) if nw > 0 else None,
        drop_last=True, collate_fn=_collate if raw else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(bsz, 512), shuffle=False, num_workers=min(nw, 4),
        pin_memory=bool(tcfg.get("PIN_MEMORY", True)) and nw > 0,
        persistent_workers=persistent and nw > 0, prefetch_factor=2 if nw > 0 else None,
        collate_fn=_collate if raw else None,
    )

    model = MNISTModel(cfg)
    warm = cfg["MODEL"].get("INIT_FROM")
    if warm:
        print(f"warm-started {model.init_from_abs_checkpoint(warm)} tensors from {warm}",
              flush=True)
    teacher_ckpt = cfg["MODEL"].get("DISTILL_CKPT")
    if teacher_ckpt:
        teacher_cfg = dict(cfg)
        tmodel = dict(cfg.get("MODEL", {}))
        tmodel.pop("ENCODER", None)
        tmodel.pop("ENCODER_HIDDEN", None)
        tmodel.pop("ENCODER_FEAT", None)
        tmodel.pop("ENCODER_CELL", None)
        tmodel.pop("DISTILL_WEIGHT", None)
        tmodel.pop("DISTILL_CKPT", None)
        tmodel.pop("ACTIVE_HEAD", None)
        tmodel.pop("ACTIVE_FEAT_DIM", None)
        tmodel.pop("ACTIVE_HIDDEN", None)
        teacher_cfg["MODEL"] = tmodel
        model.teacher = MNISTModel.load_from_checkpoint(
            teacher_ckpt, cfg=teacher_cfg, map_location="cpu").eval()
        for p in model.teacher.parameters():
            p.requires_grad_(False)
        print(f"distillation teacher loaded from {teacher_ckpt}", flush=True)

    out_dir = Path(tcfg["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    every = int(tcfg.get("SAVE_EVERY_N_STEPS") or 500)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir), filename=tcfg.get("RUN_NAME", "semkine") + "-{step}",
        monitor=None, save_last=True, save_top_k=-1, every_n_train_steps=every,
    )
    logger = False if args.no_logger else TensorBoardLogger(
        save_dir=str(out_dir), name="logs", version=tcfg.get("RUN_NAME", "semkine"))
    callbacks = [ckpt_cb, ThroughputCallback()]
    if logger:
        callbacks.insert(1, LearningRateMonitor(logging_interval="step"))

    devices = int(args.devices or tcfg["DEVICES"])
    max_steps = int(args.max_steps or tcfg["MAX_STEPS"])
    kwargs = dict(
        accelerator="gpu", devices=devices,
        strategy="ddp_find_unused_parameters_false" if devices > 1 else "auto",
        precision=tcfg.get("PRECISION", "bf16"), max_steps=max_steps,
        val_check_interval=int(tcfg["VAL_CHECK_INTERVAL"]),
        accumulate_grad_batches=int(tcfg.get("ACCUMULATE_GRAD_BATCHES", 1)),
        sync_batchnorm=bool(tcfg.get("SYNC_BATCHNORM", False)), logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True, log_every_n_steps=20, default_root_dir=str(out_dir),
        num_sanity_val_steps=0,
    )
    try:
        trainer = pl.Trainer(check_val_every_n_epoch=None, **kwargs)
    except TypeError:
        trainer = pl.Trainer(**kwargs)

    (out_dir / "training_metadata.json").write_text(json.dumps({
        "config_path": cfg.get("_config_path"), "run_name": tcfg.get("RUN_NAME"),
        "seed": int(cfg["SEED"]), "input_mode": train_ds.input_mode,
        "domrand": dr.__dict__, "train_sequences": [s for s, _ in train_ds.sequences],
        "val_core_sequences": [s for s, _ in val_ds.sequences],
        "train_samples": len(train_ds), "val_samples": len(val_ds),
        "batch_size_per_gpu": bsz, "devices": devices, "max_steps": max_steps,
        "save_every_n_steps": every, "lr": tcfg["LR"],
        "selection_policy": "fixed step grid; select by recursive RA on val_core",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }, indent=2))

    trainer.fit(model, train_loader, val_loader)
    print("done. last:", ckpt_cb.last_model_path, flush=True)


if __name__ == "__main__":
    main()
