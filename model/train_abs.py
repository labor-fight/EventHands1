#!/usr/bin/env python3
"""Train EventHands absolute-pose baseline from YAML config (DDP / bf16)."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import ConcatDataset, DataLoader

# Allow `python model/train_abs.py` from repo root or model/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config
from fastevc import build_hand_data51_datasets
from model import MNISTModel


class ThroughputCallback(pl.Callback):
    def __init__(self):
        self.t0 = None
        self.n = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank != 0:
            return
        bsz = batch[0].shape[0] * max(trainer.world_size, 1)
        if self.t0 is None:
            self.t0 = time.time()
            self.n = 0
        self.n += bsz
        if (batch_idx + 1) % 50 == 0:
            dt = time.time() - self.t0
            sps = self.n / max(dt, 1e-6)
            mem = (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
                if torch.cuda.is_available()
                else 0.0
            )
            pl_module.log("train_samples_per_sec", sps, prog_bar=True, rank_zero_only=True)
            pl_module.log("cuda_max_mem_gb", mem, prog_bar=True, rank_zero_only=True)


def seed_everything(seed: int):
    pl.seed_everything(seed, workers=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--devices", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="Override per-GPU batch")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--no-logger", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("SEED", 0)))

    mano = np.load(cfg["MANO"]["NPZ"])
    components = mano["hands_components"].astype(np.float32)

    print("Building train datasets...", flush=True)
    train_list = build_hand_data51_datasets(cfg, "train", components, train=True)
    print("Building val datasets...", flush=True)
    val_list = build_hand_data51_datasets(cfg, "val", components, train=False)
    # Fixed window for val for stability
    for ds in val_list:
        ds.fixed_window = int(cfg.get("EVAL", {}).get("WINDOW_MS", 100))
        ds.speed_aug = False
        ds.polarity_flip = False
        ds.pixel_polarity_swap = False

    train_ds = ConcatDataset(train_list)
    val_ds = ConcatDataset(val_list)
    print(f"train samples={len(train_ds)} val samples={len(val_ds)}", flush=True)

    tcfg = cfg["TRAIN"]
    bsz = int(args.batch_size or tcfg["BATCH_SIZE_PER_GPU"])
    nw = int(args.num_workers if args.num_workers is not None else tcfg["NUM_WORKERS"])
    persistent = bool(tcfg.get("PERSISTENT_WORKERS", True)) and nw > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=bsz,
        shuffle=True,
        num_workers=nw,
        pin_memory=bool(tcfg.get("PIN_MEMORY", True)) and nw > 0,
        persistent_workers=persistent,
        prefetch_factor=int(tcfg.get("PREFETCH_FACTOR", 2)) if nw > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(bsz, 512),
        shuffle=False,
        num_workers=min(nw, 4),
        pin_memory=bool(tcfg.get("PIN_MEMORY", True)) and nw > 0,
        persistent_workers=persistent and nw > 0,
        prefetch_factor=2 if nw > 0 else None,
    )

    model = MNISTModel(cfg)
    warm = cfg["MODEL"].get("INIT_FROM")
    if warm:
        n = model.init_from_abs_checkpoint(warm)
        print(f"Warm-started {n} tensors from {warm}", flush=True)

    out_dir = Path(tcfg["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_every = tcfg.get("SAVE_EVERY_N_STEPS")
    if save_every:
        # Fixed grid instead of top-k: val_loss is a poor proxy for the closed-loop
        # metric, so selecting checkpoints by it samples the trajectory unevenly.
        ckpt_cb = ModelCheckpoint(
            dirpath=str(out_dir),
            filename=tcfg.get("RUN_NAME", "abs") + "-{step}",
            monitor=None,
            save_last=True,
            save_top_k=-1,
            every_n_train_steps=int(save_every),
        )
    else:
        ckpt_cb = ModelCheckpoint(
            dirpath=str(out_dir),
            filename=tcfg.get("RUN_NAME", "abs") + "-{step}-val_loss={val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_last=True,
            save_top_k=3,
            every_n_train_steps=None,
        )
    if args.no_logger:
        logger = False
    else:
        logger = TensorBoardLogger(
            save_dir=str(out_dir), name="logs", version=tcfg.get("RUN_NAME", "abs")
        )

    devices = int(args.devices or tcfg["DEVICES"])
    max_steps = int(args.max_steps or tcfg["MAX_STEPS"])
    precision = tcfg.get("PRECISION", "bf16")
    # pytorch-lightning 1.9 uses "bf16" or 16
    if devices > 1:
        # PL 1.9 string strategy; find_unused_parameters false via env / default.
        strategy = "ddp_find_unused_parameters_false"
    else:
        strategy = "auto"

    trainer_kwargs = dict(
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_steps=max_steps,
        val_check_interval=int(tcfg["VAL_CHECK_INTERVAL"]),
        accumulate_grad_batches=int(tcfg.get("ACCUMULATE_GRAD_BATCHES", 1)),
        sync_batchnorm=bool(tcfg.get("SYNC_BATCHNORM", False)),
        logger=logger,
        callbacks=[ckpt_cb, LearningRateMonitor(logging_interval="step"), ThroughputCallback()],
        enable_progress_bar=True,
        log_every_n_steps=20,
        default_root_dir=str(out_dir),
        num_sanity_val_steps=0,
    )
    # PL 1.9: avoid epoch-based validation when using max_steps
    try:
        trainer = pl.Trainer(check_val_every_n_epoch=None, **trainer_kwargs)
    except TypeError:
        trainer = pl.Trainer(**trainer_kwargs)

    # Persist metadata
    meta = {
        "config_path": cfg.get("_config_path"),
        "pose_repr": cfg["MODEL"]["POSE_REPR"],
        "output_dim": cfg["MODEL"]["OUTPUT_DIM"],
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "batch_size_per_gpu": bsz,
        "devices": devices,
        "max_steps": max_steps,
        "lr": tcfg["LR"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    import json

    (out_dir / "training_metadata.json").write_text(json.dumps(meta, indent=2))

    print(
        f"Starting fit: devices={devices} strategy={strategy} bsz={bsz} "
        f"steps={max_steps} workers={nw}",
        flush=True,
    )
    trainer.fit(model, train_loader, val_loader)
    print("Training done. Best:", ckpt_cb.best_model_path, "last:", ckpt_cb.last_model_path, flush=True)


if __name__ == "__main__":
    main()
