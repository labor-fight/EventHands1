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
    """Code and config identity of a run: commit, branch, dirty flag, config sha256."""
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
        # A batch is one packet batch or the legacy tuple of tensors.
        n = (batch.batch_size if hasattr(batch, "batch_size") else batch[0].shape[0])
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


class XYZPairStreamCallback(U1aPairStreamCallback):
    """Audit the XAG studies' raw packets: the first batches' hashes (and source fields) of each rank."""

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if len(self.rows) >= 3:
            return
        packets = (batch,)
        xag = any(k.startswith("XAG_") for k in pl_module.cfg.get("MODEL", {}))
        h = hashlib.sha256()
        for packet in packets:
            for name in ("sequence_id", "t_start_us", "t_end_us", "target", "prev_state", "events",
                         "ptr", "delta_t_s", "betas", "camera_K", "is_sequence_start", "is_sequence_end"):
                h.update(getattr(packet, name).detach().cpu().contiguous().numpy().tobytes())
            if xag:
                for name in ("xag_source_index", "xag_source_ms", "xag_part", "xag_face", "xag_bary", "xag_valid"):
                    value = getattr(packet, name)
                    if value is None:
                        raise RuntimeError(f"XAG paired stream lacks {name}")
                    h.update(name.encode())
                    h.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.rows.append({"step": int(trainer.global_step), "sha256": h.hexdigest(),
                          "samples": packets[-1].batch_size,
                          "events": sum(int(packet.events.shape[0]) for packet in packets)})
        path = self.output_dir / f"pair_stream_rank{trainer.global_rank}.json"
        if xag:
            from tools.xag.train_audit import write
            write(path, self.rows)
        else:
            path.write_text(json.dumps(self.rows, indent=2))


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

    # SEMKINE_FAST_PIPE=1 / DATA.FAST_PIPE: the bit-for-bit fast sample path (semkine/dataset.py, "fast pipe")
    fast_requested = bool(getattr(train_ds, "fast_pipe", False))
    fast_active = bool(getattr(train_ds, "fast_pipe_active", False))
    if fast_requested and fast_active:
        print("fast_pipe: ON -- fast training-sample path, bitwise identical to the legacy one "
              "(SEMKINE_FAST_PIPE / DATA.FAST_PIPE)", flush=True)
    elif fast_requested:
        print(f"fast_pipe: requested but not applicable ({train_ds.fast_pipe_reason}); "
              f"using the legacy sample path", flush=True)
    bsz = int(args.batch_size or tcfg["BATCH_SIZE_PER_GPU"])
    nw = int(args.num_workers if args.num_workers is not None else tcfg["NUM_WORKERS"])
    persistent = bool(tcfg.get("PERSISTENT_WORKERS", True)) and nw > 0

    def _collate(items):
        if isinstance(items[0], EventPacket):
            return collate_packets(items)
        return torch.utils.data.default_collate(items)

    raw = train_ds.input_mode != "legacy_lnes"
    # The XAG arms pair their data stream with S45's: Lightning's distributed sampler uses SEED; this generator also
    # pins worker seeds.
    xag_study = "XAG_ASSOC" in cfg["MODEL"] or "XAG_GNN" in cfg["MODEL"]
    paired_stream = xag_study
    data_generator = torch.Generator().manual_seed(int(cfg["SEED"])) if paired_stream else None
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
    if xag_study:
        callbacks.append(XYZPairStreamCallback(out_dir))
        from tools.xag.train_audit import XAGTrainAudit
        callbacks.append(XAGTrainAudit(out_dir))
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
        enable_progress_bar=not paired_stream, log_every_n_steps=20, default_root_dir=str(out_dir),
        num_sanity_val_steps=0,
        # val_loss never selects a checkpoint; a few batches keep a sanity signal
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
        "fast_pipe": fast_active, "fast_pipe_requested": fast_requested,
        "selection_policy": ("fixed step6000 only (XAG)" if xag_study else
                             "fixed step grid; select by recursive RA on val_core"),
        "data_generator_seed": int(cfg["SEED"]) if paired_stream else None,
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
