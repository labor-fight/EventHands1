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
from semkine.checkpointing import (CommittedRandomSampler, CompleteCheckpoint,
                                  legacy_branch_provenance, reserve_branch_directory,
                                  LEGACY_BRANCH_MODE)  # noqa: E402


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
    recovery = ap.add_mutually_exclusive_group()
    recovery.add_argument("--resume", default=None,
                    help="Lightning checkpoint to continue from (model, optimizer, scheduler, "
                         "global_step); the fixed checkpoint grid continues from its step")
    recovery.add_argument("--branch-from", default=None,
                    help="start a NONEXACT legacy optimizer branch with fresh RNG and sample order")
    ap.add_argument("--branch-seed", type=int, default=None,
                    help="new RNG/sampler seed for --branch-from; keeps dataset/config SEED unchanged")
    ap.add_argument("--complete-checkpoints", action="store_true",
                    help="save per-rank RNG and committed shuffled samples for exact training continuation; "
                         "automatically enabled when resuming a checkpoint that contains this state")
    args = ap.parse_args()
    if (args.branch_from is None) != (args.branch_seed is None):
        ap.error("--branch-from and --branch-seed must be provided together")
    if args.branch_from and args.seed is not None:
        ap.error("--branch-from keeps config SEED unchanged; use only --branch-seed")

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["SEED"] = int(args.seed)
    tcfg = cfg["TRAIN"]
    if args.run_name:
        tcfg["RUN_NAME"] = args.run_name
    if args.output_dir:
        tcfg["OUTPUT_DIR"] = args.output_dir
    complete = args.complete_checkpoints or bool(tcfg.get("COMPLETE_CHECKPOINTS", False))
    branch_origin = None
    if args.branch_from:
        saved = torch.load(args.branch_from, map_location="cpu")
        branch_origin = legacy_branch_provenance(args.branch_from, saved, args.branch_seed)
        del saved
        if int(tcfg.get("ACCUMULATE_GRAD_BATCHES", 1)) != 1:
            raise ValueError("Legacy optimizer branches require ACCUMULATE_GRAD_BATCHES=1")
        effective_max_steps = int(args.max_steps or tcfg["MAX_STEPS"])
        if effective_max_steps <= branch_origin["source_global_step"]:
            raise ValueError("Branch max-steps is absolute and must exceed the source global_step")
        reserve_branch_directory(tcfg["OUTPUT_DIR"], branch_origin)
        complete = True
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu")
        saved_complete = CompleteCheckpoint.KEY in saved
        if complete and not saved_complete:
            raise ValueError("Old checkpoint has no RNG/sampler state: complete continuation is unavailable")
        complete = complete or saved_complete
        if saved_complete:
            branch_origin = saved[CompleteCheckpoint.KEY].get("legacy_branch")
        del saved
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
    devices = int(args.devices or tcfg["DEVICES"])
    max_steps = int(args.max_steps or tcfg["MAX_STEPS"])
    sampler_seed = branch_origin["branch_seed"] if branch_origin else int(cfg["SEED"])
    sampler = CommittedRandomSampler(train_ds, sampler_seed) if complete else None
    val_sampler = None
    if complete and devices > 1:
        # Disabling automatic train-sampler replacement also affects validation.
        # Sequential distributed indices match the previous Lightning wrapper.
        val_sampler = torch.utils.data.DistributedSampler(val_ds, num_replicas=devices,
                            rank=int(os.environ.get("LOCAL_RANK", 0)), shuffle=False, seed=int(cfg["SEED"]))

    def _collate(items):
        if isinstance(items[0], EventPacket):
            return collate_packets(items)
        if isinstance(items[0], tuple) and isinstance(items[0][0], EventPacket):
            # S22 unroll pair: collate each half separately, so both stay ragged batches.
            return tuple(collate_packets(list(col)) for col in zip(*items))
        return torch.utils.data.default_collate(items)

    raw = train_ds.input_mode != "legacy_lnes"
    train_loader = DataLoader(
        train_ds, batch_size=bsz, shuffle=sampler is None, sampler=sampler, num_workers=nw,
        pin_memory=bool(tcfg.get("PIN_MEMORY", True)) and nw > 0,
        persistent_workers=persistent,
        prefetch_factor=int(tcfg.get("PREFETCH_FACTOR", 2)) if nw > 0 else None,
        drop_last=True, collate_fn=_collate if raw else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(bsz, 512), shuffle=False, sampler=val_sampler, num_workers=min(nw, 4),
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
    if complete:
        # Pin effective settings and dataset indexing, not output-directory labels.
        normalized_cfg = json.loads(json.dumps(cfg, default=str))
        normalized_cfg.pop("_config_path", None)
        for key in ("OUTPUT_DIR", "RUN_NAME", "COMPLETE_CHECKPOINTS"):
            normalized_cfg["TRAIN"].pop(key, None)
        digest = lambda value: hashlib.sha256(value).hexdigest()
        payload_files = {}
        for ds in (train_ds, val_ds):
            for seq, split in ds.sequences:
                for suffix in ("_events.npy", "_tsub.npy", "_offsets.npy", "_aux.npz", ".meta"):
                    path = Path(cfg["DATA"]["ROOT"]) / split / (seq + suffix)
                    if suffix == "_tsub.npy" and not path.exists():
                        # legacy_lnes supports millisecond timestamps without this file.
                        # The raw dataset constructor already rejects a missing sidecar.
                        payload_files[str(path.resolve())] = None
                        continue
                    st = path.stat()
                    payload_files[str(path.resolve())] = [st.st_size, st.st_mtime_ns]
        source_files = sorted(str(path.relative_to(REPO)) for folder in ("semkine", "model")
                              for path in (REPO/folder).glob("*.py"))
        contract = dict(world_size=devices, batch_size=bsz, workers=nw, persistent_workers=persistent,
                        prefetch_factor=int(tcfg.get("PREFETCH_FACTOR", 2)) if nw else None,
                        accumulate_grad_batches=int(tcfg.get("ACCUMULATE_GRAD_BATCHES", 1)),
                        precision=str(tcfg.get("PRECISION", "bf16")), max_steps=max_steps,
                        val_check_interval=int(tcfg["VAL_CHECK_INTERVAL"]), save_every_n_steps=every,
                        seed=int(cfg["SEED"]), config_sha256=digest(json.dumps(normalized_cfg,sort_keys=True).encode()),
                        train_index_sha256=digest(train_ds.index.tobytes()),
                        val_index_sha256=digest(val_ds.index.tobytes()),
                        split_sha256=digest((manifest or Path(cfg["DATA"]["ROOT"])/"splits_semkine.json").read_bytes()),
                        mano_sha256=digest(Path(cfg["MANO"]["NPZ"]).read_bytes()),
                        teacher_sha256=digest(Path(teacher_ckpt).read_bytes()) if teacher_ckpt else None,
                        source_sha256={name:digest((REPO/name).read_bytes()) for name in source_files},
                        data_file_size_mtime=payload_files, dataset_randomness="pure(seed,index)",
                        deterministic_algorithms=True, cudnn_benchmark=False,
                        torch_version=torch.__version__, lightning_version=pl.__version__, cuda_version=torch.version.cuda,
                        numpy_version=np.__version__, logger_enabled=not args.no_logger)
        if branch_origin is not None:
            contract["legacy_branch"] = branch_origin
        ckpt_cb = CompleteCheckpoint(sampler, out_dir, tcfg.get("RUN_NAME", "semkine"), every, bsz,
                                    contract, legacy_branch=branch_origin if args.branch_from else None)
    else:
        ckpt_cb = ModelCheckpoint(
            dirpath=str(out_dir), filename=tcfg.get("RUN_NAME", "semkine") + "-{step}",
            monitor=None, save_last=True, save_top_k=-1, every_n_train_steps=every,
        )
    logger = False if args.no_logger else TensorBoardLogger(
        save_dir=str(out_dir), name="logs", version=tcfg.get("RUN_NAME", "semkine"))
    callbacks = [ckpt_cb, ThroughputCallback()]
    if logger:
        callbacks.insert(1, LearningRateMonitor(logging_interval="step"))

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
    if complete:
        # Exact sample/RNG recovery also needs deterministic graph gradients. In
        # this pinned Lightning version True enables PyTorch's strict mode and
        # the deterministic cuBLAS workspace; unsupported operations must fail.
        kwargs.update(enable_checkpointing=False, replace_sampler_ddp=False,
                      deterministic=True, benchmark=False)
    try:
        trainer = pl.Trainer(check_val_every_n_epoch=None, **kwargs)
    except TypeError:
        trainer = pl.Trainer(**kwargs)

    metadata_path = out_dir / "training_metadata.json"
    metadata_writer = not complete or int(os.environ.get("LOCAL_RANK", 0)) == 0
    if complete and metadata_writer and metadata_path.exists():
        if not args.resume:
            raise FileExistsError("Complete-checkpoint run already exists; choose a new output directory or --resume")
        metadata_path = out_dir / ("resume_metadata_" + str(time.time_ns()) + ".json")
    metadata = {
        "config_path": cfg.get("_config_path"), "run_name": tcfg.get("RUN_NAME"),
        "seed": int(cfg["SEED"]), "input_mode": train_ds.input_mode,
        "domrand": dr.__dict__, "train_sequences": [s for s, _ in train_ds.sequences],
        "val_core_sequences": [s for s, _ in val_ds.sequences],
        "splits_manifest": manifest_name, "train_subjects": train_subjects,
        "train_samples": len(train_ds), "val_samples": len(val_ds),
        "batch_size_per_gpu": bsz, "devices": devices, "max_steps": max_steps,
        "save_every_n_steps": every, "lr": tcfg["LR"],
        "selection_policy": "fixed step grid; select by recursive RA on val_core",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "resumed_from": args.resume,
        "branched_from": args.branch_from,
        "legacy_branch_provenance": branch_origin,
        "recovery_mode": (LEGACY_BRANCH_MODE if args.branch_from else
                          "complete_committed_step" if complete else "legacy_optimizer_state_without_rng_sampler"),
        "recovery_contract": contract if complete else None,
    }
    if metadata_writer:
        metadata_path.write_text(json.dumps(metadata, indent=2))

    if args.resume:
        print(f"resuming from {args.resume}; recovery="
              f"{'complete committed step' if complete else 'legacy: RNG/sampler unavailable'}", flush=True)
    if args.branch_from:
        print(f"NONEXACT_LEGACY_OPTIMIZER_BRANCH from {args.branch_from}; "
              f"source_step={branch_origin['source_global_step']} branch_seed={args.branch_seed}; "
              f"dataset_seed={cfg['SEED']} unchanged; new data epoch 0", flush=True)
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume or args.branch_from)
    print("done. last:", ckpt_cb.last_model_path, flush=True)


if __name__ == "__main__":
    main()
