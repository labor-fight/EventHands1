#!/usr/bin/env python3
"""S1 training entry point on the SemKine protocol.

Differences from `model/train_abs.py`, each one forced by a finding in the audit:

* data comes from `semkine.dataset.SemKineDataset`, so the subject-disjoint splits and the
  domain randomisation arm are reachable. With `INPUT_MODE: legacy_lnes` and
  `AUG.DOMRAND.ENABLED: false` the samples are bitwise identical to the legacy loader
  (`tests/test_s1_dataset.py::test_legacy_lnes_is_bitwise_identical`).
* checkpoints are always saved on a fixed step grid. Selection by `val_loss` is not offered,
  because `val_loss` is anti-correlated with the closed-loop metric this project reports
  (Spearman -0.07). Choosing among grid points is then a separate,
  explicit step (`tools/select_checkpoint.py`), scored by the recursive evaluator.
* `SEED` shifts the augmentation stream as well as the initialisation, so a "replicate" is a
  genuinely independent draw rather than the same data order with different weights.

The legacy trainer is untouched and remains the way to reproduce the frozen baseline.
"""
from __future__ import annotations

import argparse
import hashlib
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
from semkine.dataset import build_dataset, splits_manifest   # noqa: E402
from semkine.events import EventPacket, collate_packets  # noqa: E402


def _provenance(cfg_path):
    """Code and config identity of a run (x1001): commit, branch, dirty flag, config sha256."""
    import hashlib
    import subprocess

    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=REPO, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    dirty = git("status", "--porcelain", "--untracked-files=no")
    sha = None
    if cfg_path and Path(cfg_path).exists():
        sha = hashlib.sha256(Path(cfg_path).read_bytes()).hexdigest()
    return {"git_commit": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(dirty), "repo": str(REPO), "config_sha256": sha}


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
        # A batch is one packet batch, an unroll `(lead, main)` pair, or the legacy tuple of
        # tensors. Lightning may hand the pair back as a list, so do not test for `tuple`.
        head = batch
        if isinstance(batch, (tuple, list)) and hasattr(batch[-1], "batch_size"):
            head = batch[-1]
        n = (head.batch_size if hasattr(head, "batch_size") else head[0].shape[0])
        self.n += n * max(trainer.world_size, 1)
        if (batch_idx + 1) % 50 == 0:
            sps = self.n / max(time.time() - self.t0, 1e-6)
            mem = (torch.cuda.max_memory_allocated() / (1024 ** 3)
                   if torch.cuda.is_available() else 0.0)
            pl_module.log("train_samples_per_sec", sps, prog_bar=True, rank_zero_only=True)
            pl_module.log("cuda_max_mem_gb", mem, prog_bar=True, rank_zero_only=True)


class U1aPairStreamCallback(pl.Callback):
    """Record initial data-stream hashes on each rank, without exposing raw samples."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.rows = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if len(self.rows) >= 2:
            return
        h = hashlib.sha256()
        for name in ("sequence_id", "t_start_us", "t_end_us", "target", "prev_state", "events",
                     "ptr", "delta_t_s", "betas", "camera_K", "is_sequence_start", "is_sequence_end"):
            h.update(getattr(batch, name).detach().cpu().contiguous().numpy().tobytes())
        self.rows.append({"step": int(trainer.global_step), "sha256": h.hexdigest(),
                          "samples": batch.batch_size, "events": int(batch.events.shape[0])})
        (self.output_dir / f"pair_stream_rank{trainer.global_rank}.json").write_text(
            json.dumps(self.rows, indent=2))


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
    ap.add_argument("--resume", default=None,
                    help="Lightning checkpoint to continue from (model, optimizer, scheduler, "
                         "global_step); the fixed checkpoint grid continues from its step")
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
    # Name the split and its subjects on stdout. A run that silently trains on half the subjects
    # is indistinguishable from a correct one in every other line of this log.
    manifest = splits_manifest(cfg)
    manifest_name = manifest.name if manifest else "splits_semkine.json"
    train_subjects = sorted({s.split("_")[0] for s, _ in train_ds.sequences})
    print(f"seed={cfg['SEED']} input_mode={train_ds.input_mode} domrand={dr.enabled}\n"
          f"splits: {manifest_name}\n"
          f"train: {len(train_ds.sequences)} seqs / {len(train_ds)} samples / "
          f"{len(train_subjects)} subjects {train_subjects}\n"
          f"val_core: {len(val_ds.sequences)} seqs / {len(val_ds)} samples "
          f"{sorted({s.split('_')[0] for s, _ in val_ds.sequences})}", flush=True)

    bsz = int(args.batch_size or tcfg["BATCH_SIZE_PER_GPU"])
    nw = int(args.num_workers if args.num_workers is not None else tcfg["NUM_WORKERS"])
    persistent = bool(tcfg.get("PERSISTENT_WORKERS", True)) and nw > 0

    def _collate(items):
        if isinstance(items[0], EventPacket):
            return collate_packets(items)
        if isinstance(items[0], tuple) and isinstance(items[0][0], EventPacket):
            # S22 unroll pair: collate each half separately, so both stay ragged batches.
            return tuple(collate_packets(list(col)) for col in zip(*items))
        return torch.utils.data.default_collate(items)

    raw = train_ds.input_mode != "legacy_lnes"
    # Keep U1a's paired stream independent of the RNG consumed by 1 vs 17 heads.
    # Lightning's distributed sampler uses SEED; this generator also pins worker seeds.
    u1a = str(cfg["MODEL"].get("U1A_MODE", "off")).lower() != "off"
    data_generator = torch.Generator().manual_seed(int(cfg["SEED"])) if u1a else None
    train_loader = DataLoader(
        train_ds, batch_size=bsz, shuffle=True, num_workers=nw,
        pin_memory=bool(tcfg.get("PIN_MEMORY", True)) and nw > 0,
        persistent_workers=persistent,
        prefetch_factor=int(tcfg.get("PREFETCH_FACTOR", 2)) if nw > 0 else None,
        drop_last=True, collate_fn=_collate if raw else None, generator=data_generator,
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
        # The teacher is the frozen dense arm, so every student-side key has to go. Dropping them
        # by prefix rather than by name means adding a student key (S18 added three) cannot
        # silently build a teacher that refuses to load its own checkpoint.
        drop = ("ENCODER", "DISTILL_", "ACTIVE_")
        tmodel = {k: v for k, v in cfg.get("MODEL", {}).items()
                  if not any(k == d.rstrip("_") or k.startswith(d) for d in drop)}
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
    if u1a:
        callbacks.append(U1aPairStreamCallback(out_dir))
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
        enable_progress_bar=not u1a, log_every_n_steps=20, default_root_dir=str(out_dir),
        num_sanity_val_steps=0,
        # x1001: val_loss never selects a checkpoint; a few batches keep a sanity signal
        limit_val_batches=tcfg.get("LIMIT_VAL_BATCHES", 1.0),
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
        "splits_manifest": manifest_name, "train_subjects": train_subjects,
        "train_samples": len(train_ds), "val_samples": len(val_ds),
        "batch_size_per_gpu": bsz, "devices": devices, "max_steps": max_steps,
        "save_every_n_steps": every, "lr": tcfg["LR"],
        "selection_policy": "explicit last only (U1a)" if u1a else "fixed step grid; select by recursive RA on val_core",
        "data_generator_seed": int(cfg["SEED"]) if u1a else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "resumed_from": args.resume,
        "accumulate_grad_batches": int(tcfg.get("ACCUMULATE_GRAD_BATCHES", 1)),
        "effective_batch": bsz * devices * int(tcfg.get("ACCUMULATE_GRAD_BATCHES", 1)),
        "lr_schedule": str(tcfg.get("LR_SCHEDULE", "constant")),
        "provenance": _provenance(cfg.get("_config_path")),
        "argv": sys.argv,
    }, indent=2))
    # The resolved config (seed and paths applied) next to the checkpoints: select_checkpoint.py
    # reads the run directory's yaml before the training metadata's config_path.
    import yaml
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(
        {k: v for k, v in cfg.items() if k != "_config_path"}, sort_keys=False))

    if args.resume:
        print(f"resuming from {args.resume}", flush=True)
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
    print("done. last:", ckpt_cb.last_model_path, flush=True)


if __name__ == "__main__":
    main()
