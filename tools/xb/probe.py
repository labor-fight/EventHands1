#!/usr/bin/env python3
"""Train the preregistered S37 residual A/B/C/D probe without changing the main trainer.

Ordinary Lightning checkpoints retain MNISTModel's architecture/config and are therefore
readable by make_s36_row.py.  The run's config_resolved.yaml is the authoritative config.
The frozen common mesh modules are loaded from one fixed C0 checkpoint.  Only the new
raw-residual encoder and the two readout layers learn in ``--freeze probe`` mode.
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
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from config import load_config  # noqa: E402
from model import MNISTModel  # noqa: E402
from semkine.dataset import build_dataset, splits_manifest  # noqa: E402
from semkine.events import EventPacket, collate_packets  # noqa: E402
from semkine.train import ThroughputCallback, _provenance  # noqa: E402
from semkine import eval_track as ET  # noqa: E402
from semkine.dataset import sequences_for_split  # noqa: E402

PROBE_PREFIXES = ("xyz.residual.", "xyz.fc1.", "xyz.fc2.")


class RepeatedPermutationSampler(torch.utils.data.Sampler):
    """Cycle complete seeded permutations and keep every optimizer batch full.

    A sequence with fewer than 64 packets must not change the optimization budget through
    a short final batch. Each permutation includes every bank row once; batches may span
    permutation boundaries. Parameter initialization cannot affect this stream.
    """

    def __init__(self, size, samples, seed):
        self.size, self.samples, self.seed = int(size), int(samples), int(seed)
        if self.size <= 0:
            raise ValueError("empty bank")

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        emitted = 0
        while emitted < self.samples:
            order = torch.randperm(self.size, generator=generator).tolist()
            take = min(self.size, self.samples - emitted)
            yield from order[:take]
            emitted += take

    def __len__(self):
        return self.samples


def warm_start(model, path: str | Path) -> dict:
    """Copy all legacy tensors; fc1's extra *appended* feature columns start at zero.

    The input contract is [legacy 17 slots, prev48, extra residual features].  This makes
    initial predictions exactly the legacy model's and avoids an arbitrary residual gain.
    An unmatched legacy tensor is an error: a silently missing common backbone would
    invalidate the information probe.  Newly introduced tensors keep their seeded init.
    """
    source = torch.load(path, map_location="cpu")
    source = source.get("state_dict", source)
    own = model.state_dict()
    copied, expanded, rejected = [], [], []
    for name, value in source.items():
        if name not in own:
            rejected.append(name)
        elif own[name].shape == value.shape:
            own[name].copy_(value)
            copied.append(name)
        elif name == "xyz.fc1.weight" and value.ndim == 2 and own[name].shape[0] == value.shape[0] \
                and own[name].shape[1] > value.shape[1]:
            own[name].zero_()
            own[name][:, :value.shape[1]].copy_(value)
            expanded.append({"name": name, "old_shape": list(value.shape), "new_shape": list(own[name].shape),
                             "extra_init": "zero", "input_layout": "legacy_slots,prev48,new_residual"})
        else:
            rejected.append(name)
    if rejected:
        raise RuntimeError(f"warm start rejected legacy tensors: {rejected}")
    model.load_state_dict(own, strict=True)
    return {"checkpoint": str(Path(path).resolve()), "copied_tensors": copied,
            "expanded_tensors": expanded, "new_tensors": sorted(set(own) - set(source))}


def freeze_model(model, mode="probe", prefixes=PROBE_PREFIXES) -> dict:
    """Freeze by exact parameter-name prefixes, and disclose the complete trainable set."""
    if mode not in ("probe", "full"):
        raise ValueError(mode)
    trainable = []
    frozen = []
    for name, parameter in model.named_parameters():
        learn = mode == "full" or any(name.startswith(prefix) for prefix in prefixes)
        parameter.requires_grad_(learn)
        (trainable if learn else frozen).append((name, parameter.numel()))
    if not trainable:
        raise RuntimeError(f"no trainable parameters matched {prefixes}")
    if mode == "probe" and not any(name.startswith(prefixes[0]) for name, _ in trainable):
        raise RuntimeError(f"no new residual encoder matched {prefixes[0]}")
    return {"mode": mode, "prefixes": list(prefixes), "trainable_names": [n for n, _ in trainable],
            "trainable_params": sum(n for _, n in trainable), "frozen_params": sum(n for _, n in frozen),
            "total_params": sum(n for _, n in trainable + frozen)}


class StreamAudit(pl.Callback):
    """Paired training examples must not depend on the architecture's RNG consumption."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.rows = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_rank != 0 or len(self.rows) >= 3:
            return
        # A raw unroll pair audits both packets; plain batches audit one.
        packets = batch if isinstance(batch, (list, tuple)) else (batch,)
        digest = hashlib.sha256()
        for packet in packets:
            for name in ("sequence_id", "t_start_us", "t_end_us", "target", "prev_state", "events", "ptr",
                         "delta_t_s", "betas", "camera_K"):
                digest.update(getattr(packet, name).detach().cpu().contiguous().numpy().tobytes())
        self.rows.append({"step": int(trainer.global_step), "sha256": digest.hexdigest(),
                          "samples": int(packets[-1].batch_size)})
        (self.directory / "pair_stream.json").write_text(json.dumps(self.rows, indent=2))


def collate(items):
    if isinstance(items[0], EventPacket):
        return collate_packets(items)
    if isinstance(items[0], (tuple, list)) and isinstance(items[0][0], EventPacket):
        return tuple(collate_packets(list(part)) for part in zip(*items))
    return torch.utils.data.default_collate(items)


@torch.no_grad()
def prepare_bank(cfg, checkpoint, output, device, max_per_seq=64):
    """One common frozen C0 rollout over all 72 training sequences; no GT association.

    Event tokens, residual masses and already pooled common mesh slots are captured from
    the actual production forward.  Every sequence contributes up to its first 64 valid
    50 ms steps, including restarts at valid-run boundaries.  Labels never enter forward.
    """
    device = torch.device(device)
    model = MNISTModel.load_from_checkpoint(checkpoint, cfg=cfg, map_location="cpu").to(device).eval()
    if getattr(model, "xyz_assoc_prev", "self") != "self":
        raise ValueError("cache generation must use own-state association")
    if getattr(model.xyz, "residual", None) is not None:
        raise ValueError("cache must come from the fixed legacy C0 model")
    model.xyz_keep_aux = True
    root = Path(cfg["DATA"]["ROOT"])
    sequences = sequences_for_split(root, "train", splits_manifest(cfg))
    subjects = sorted({seq.split("_")[0] for seq, _ in sequences})
    if len(sequences) != 72 or len(subjects) != 9 or "zgz" in subjects:
        raise RuntimeError("bank generation requires the fixed 72-sequence nine-person training split")
    columns = {name: [] for name in ("slots", "tok", "r", "mask", "prev", "target", "betas", "camera_K")}
    sample_ids, sequence_summary = [], []
    rng = np.random.default_rng(0)
    max_nodes = int(model.xyz_max_nodes)
    started = time.monotonic()
    for seq, directory in sequences:
        events, offsets, aux, targets = ET.load_sequence(root, directory, seq)
        tsub_path = root / directory / f"{seq}_tsub.npy"
        tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
        beta = torch.as_tensor(np.array(aux["betas"]), dtype=torch.float32, device=device).view(1, -1)
        camera = torch.as_tensor(np.array(aux["camera_K"]), dtype=torch.float32, device=device).view(1, 3, 3)
        model.set_hand_context(beta, camera)
        count = 0
        for run_id, (a, b) in enumerate(np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)):
            prev = torch.from_numpy(targets[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)).view(1, -1).to(device)
            previous_end = int(a)
            for end in np.arange(a + 49, b, 50, dtype=np.int64):
                ev5 = ET._window_events(events, offsets, tsub, int(end), 50)
                packet = ET.make_eval_packet(ev5, prev, beta, camera, 50, device)
                # make_eval_packet.target is zero, so no current ground truth can reach association.
                prediction = model.forward_packet(packet)
                if len(ev5):
                    last = model.xyz_last
                    slots = last["slots"][0].float().cpu()
                    n = last["tok"].shape[1]
                    tok = torch.zeros(max_nodes, 7, dtype=torch.float32)
                    residual = torch.zeros(max_nodes, dtype=torch.float32)
                    mask = torch.zeros(max_nodes, dtype=torch.bool)
                    tok[:n] = last["tok"][0].float().cpu()
                    residual[:n] = last["r"][0].float().cpu()
                    mask[:n] = last["mask"][0].cpu()
                else:
                    slots = torch.zeros(model.xyz.N_SLOTS, model.xyz.slot_in)
                    tok = torch.zeros(max_nodes, 7)
                    residual, mask = torch.zeros(max_nodes), torch.zeros(max_nodes, dtype=torch.bool)
                values = {"slots": slots, "tok": tok, "r": residual, "mask": mask,
                          "prev": prev[0].float().cpu(), "target": torch.from_numpy(targets[end].copy()),
                          "betas": beta[0].cpu(), "camera_K": camera[0].cpu()}
                for name, value in values.items():
                    columns[name].append(value.clone())
                sample_ids.append({"seq": seq, "run_id": run_id, "previous_end_ms": previous_end,
                                   "end_ms": int(end), "n_raw_events": len(ev5)})
                prev, previous_end = prediction, int(end)
                count += 1
                if count >= max_per_seq:
                    break
            if count >= max_per_seq:
                break
        if not count:
            raise RuntimeError(f"training sequence has no valid W50 packet: {seq}")
        sequence_summary.append({"seq": seq, "packets": count})
        print(f"bank {seq}: {count} packets, cumulative={len(sample_ids)}, elapsed={time.monotonic()-started:.1f}s", flush=True)
    result = {name: torch.stack(values) for name, values in columns.items()}
    result["sample_ids"] = sample_ids
    result["metadata"] = {"checkpoint": str(Path(checkpoint).resolve()), "config_path": cfg["_config_path"],
                          "window_ms": 50, "step_ms": 50, "max_per_sequence": max_per_seq,
                          "subjects": subjects, "sequences": sequence_summary, "n_samples": len(sample_ids),
                          "init_rng_seed": 0, "association": "model own prev; target tensor zero",
                          "target_usage": "run initialization and supervision only", "token_dtype": "float32"}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    result["metadata"]["sha256"] = digest
    output.with_suffix(".json").write_text(json.dumps(result["metadata"], indent=2))
    print(f"bank saved {output}, sha256={digest}", flush=True)


def cached_forward(model, batch):
    """The production readout applied to a fixed common-feature bank."""
    slots, prev = batch["slots"], batch["prev"]
    xyz = model.xyz
    zs = torch.relu(xyz.slot_proj(slots.reshape(len(prev), -1, 1))).view(len(prev), -1)
    extra = xyz.residual(batch["tok"], batch["r"], batch["mask"])
    if isinstance(extra, tuple):
        extra = extra[0]
    features = torch.cat([zs, prev[:, 3:51].to(zs.dtype), extra.to(zs.dtype)], dim=-1)
    delta = xyz.fc2(torch.relu(xyz.fc1(features)))
    if model.zero_event_gate:
        delta = delta * batch["mask"].any(1).unsqueeze(-1).to(delta.dtype)
    return prev + delta


def save_cached_checkpoint(model, cfg, optimizer, step, directory):
    # Match Lightning's checkpoint format and MNISTModel.save_hyperparameters(self.cfg).
    checkpoint = {"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                  "hyper_parameters": dict(cfg), "pytorch-lightning_version": pl.__version__,
                  "global_step": int(step), "epoch": 0, "optimizer_states": [optimizer.state_dict()],
                  "xb_cached_training": True}
    path = directory / f"{cfg['TRAIN']['RUN_NAME']}-step={step}.ckpt"
    torch.save(checkpoint, path)
    last = directory / "last.ckpt"
    if last.is_symlink() or last.exists():
        last.unlink()
    last.symlink_to(path.name)


def train_cached(model, cfg, args, warm, frozen):
    """Equal-capacity heads, identical bank and sampling; common mesh layers stay frozen."""
    if args.freeze != "probe":
        raise ValueError("cached bank training can only train the residual and readout probe")
    device = torch.device(args.device)
    bank = torch.load(args.bank, map_location="cpu")
    meta = bank["metadata"]
    if len(meta["sequences"]) != 72 or len(meta["subjects"]) != 9 or "zgz" in meta["subjects"]:
        raise RuntimeError("cached probe requires the fixed nine-person training bank")
    if Path(meta["checkpoint"]).resolve() != Path(args.init_ckpt).resolve():
        raise ValueError("bank and model must use the same fixed common C0 checkpoint")
    model.to(device).train()
    n = len(bank["prev"])
    tensor_names = ("slots", "tok", "r", "mask", "prev", "target", "betas")
    tensors = torch.utils.data.TensorDataset(*(bank[name] for name in tensor_names), torch.arange(n))
    tcfg, seed = cfg["TRAIN"], int(cfg["SEED"])
    bsz, workers = int(tcfg["BATCH_SIZE_PER_GPU"]), int(tcfg["NUM_WORKERS"])
    loader = DataLoader(tensors, batch_size=bsz, shuffle=False, drop_last=True,
                        sampler=RepeatedPermutationSampler(n, bsz * int(tcfg["MAX_STEPS"]), seed),
                        generator=torch.Generator().manual_seed(seed), num_workers=workers,
                        pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    directory = Path(tcfg["OUTPUT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    resolved = directory / "config_resolved.yaml"
    resolved.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if k != "_config_path"}, sort_keys=False))
    bank_hash = json.loads(Path(args.bank).with_suffix(".json").read_text()).get("sha256")
    metadata = {"config_path": str(resolved.resolve()), "source_config_path": cfg["_config_path"],
                "run_name": tcfg["RUN_NAME"], "seed": seed, "input_mode": "raw_packed",
                "train_subjects": meta["subjects"], "train_sequences": [row["seq"] for row in meta["sequences"]],
                "val_core_sequences": [seq for seq, _ in sequences_for_split(Path(cfg["DATA"]["ROOT"]), "val_core", splits_manifest(cfg))],
                "train_samples": n, "bank": str(Path(args.bank).resolve()), "bank_sha256": bank_hash,
                "batch_size_per_gpu": bsz, "devices": 1, "max_steps": int(tcfg["MAX_STEPS"]),
                "accumulate_grad_batches": 1, "effective_batch": bsz,
                "save_every_n_steps": int(tcfg.get("SAVE_EVERY_N_STEPS", 500)), "warm_start": warm,
                "freeze": frozen, "selection_policy": "fixed last step; no validation selection",
                "data_stream": "complete repeated seeded permutations; batches wrap across permutation boundaries",
                "cache_policy": meta, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                "provenance": _provenance(cfg["_config_path"]), "argv": sys.argv}
    (directory / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(tcfg["LR"]))
    iterator, stream, logs = iter(loader), [], []
    started = time.monotonic()
    amp = str(tcfg.get("PRECISION", "32")).startswith("bf16") and device.type == "cuda"
    for step in range(1, int(tcfg["MAX_STEPS"]) + 1):
        try:
            values = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            values = next(iterator)
        if step <= 3:
            stream.append({"step": step, "indices": values[-1].tolist(),
                           "sha256": hashlib.sha256(values[-1].numpy().tobytes()).hexdigest()})
            (directory / "pair_stream.json").write_text(json.dumps(stream, indent=2))
        batch = {name: value.to(device, non_blocking=True) for name, value in zip(tensor_names, values[:-1])}
        warmup = int(tcfg.get("WARMUP_STEPS", 0))
        lr = float(tcfg["LR"]) * min(step / max(warmup, 1), 1.0)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            pred = cached_forward(model, batch)
            loss, parts = model._compute_loss(pred, batch["target"], batch["betas"])
        objective = loss.float().log10() if model.log10_loss else loss.float()
        if not bool(torch.isfinite(objective)):
            raise RuntimeError(f"non-finite cached probe loss at step {step}")
        objective.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0:
            row = {"step": step, "loss": float(loss.detach()), "objective": float(objective.detach()),
                   "lr": lr, "elapsed_s": time.monotonic() - started}
            logs.append(row)
            (directory / "training_curve.json").write_text(json.dumps(logs, indent=2))
            print(json.dumps(row), flush=True)
        if step % int(tcfg.get("SAVE_EVERY_N_STEPS", 500)) == 0 or step == int(tcfg["MAX_STEPS"]):
            save_cached_checkpoint(model, cfg, optimizer, step, directory)
    metadata["completed_steps"] = int(tcfg["MAX_STEPS"])
    metadata["elapsed_s"] = time.monotonic() - started
    (directory / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"cached probe done: {directory / 'last.ckpt'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--freeze", choices=("probe", "full"), default="probe")
    ap.add_argument("--trainable-prefix", action="append", default=None)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--run-name")
    ap.add_argument("--output-dir")
    ap.add_argument("--resume")
    ap.add_argument("--no-logger", action="store_true")
    ap.add_argument("--verify-only", action="store_true", help="build/load/freeze and print audit; no data or GPU")
    ap.add_argument("--prepare-bank", help="write the shared fixed C0 feature bank; no training")
    ap.add_argument("--bank", help="train on this shared fixed feature bank instead of recomputing MANO rays")
    ap.add_argument("--max-per-seq", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["SEED"] = args.seed
    seed = int(cfg["SEED"])
    tcfg = cfg["TRAIN"]
    for key, value in (("RUN_NAME", args.run_name), ("OUTPUT_DIR", args.output_dir),
                       ("BATCH_SIZE_PER_GPU", args.batch_size), ("NUM_WORKERS", args.num_workers),
                       ("MAX_STEPS", args.max_steps)):
        if value is not None:
            tcfg[key] = value
    tcfg["DEVICES"] = args.devices
    if args.prepare_bank:
        prepare_bank(cfg, args.init_ckpt, args.prepare_bank, args.device, args.max_per_seq)
        return
    pl.seed_everything(seed, workers=True)
    model = MNISTModel(cfg)
    warm = warm_start(model, args.init_ckpt)
    frozen = freeze_model(model, args.freeze, tuple(args.trainable_prefix or PROBE_PREFIXES))
    if args.verify_only:
        print(json.dumps({"warm_start": warm, "freeze": frozen}, indent=2), flush=True)
        return
    if args.bank:
        pl.seed_everything(seed, workers=True)
        train_cached(model, cfg, args, warm, frozen)
        return
    # Reset the shared stream after model construction; a separate loader generator pins
    # sampler and worker seeding independent of arm-specific parameter initialisation.
    pl.seed_everything(seed, workers=True)
    components = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    train_ds = build_dataset(cfg, "train", components, train=True)
    val_ds = build_dataset(cfg, "val_core", components, train=False)
    subjects = sorted({seq.split("_")[0] for seq, _ in train_ds.sequences})
    required = ["ch", "lfz", "lpc", "lr", "ly", "lyh", "lyq", "ycy", "ylf"]
    if subjects != required or len(train_ds.sequences) != 72:
        raise RuntimeError(f"fixed nine-person protocol required, got {subjects} / {len(train_ds.sequences)} seqs")
    if train_ds.input_mode != "raw_packed":
        raise ValueError("the residual probe requires DATA.INPUT_MODE=raw_packed")
    directory = Path(tcfg["OUTPUT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    bsz, workers = int(tcfg["BATCH_SIZE_PER_GPU"]), int(tcfg["NUM_WORKERS"])
    loader_kwargs = dict(num_workers=workers, pin_memory=workers > 0,
                         persistent_workers=workers > 0 and bool(tcfg.get("PERSISTENT_WORKERS", True)),
                         prefetch_factor=int(tcfg.get("PREFETCH_FACTOR", 2)) if workers > 0 else None,
                         collate_fn=collate)
    train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(seed), **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=min(bsz, 512), shuffle=False, **loader_kwargs)
    every = int(tcfg.get("SAVE_EVERY_N_STEPS", 500))
    checkpoint = ModelCheckpoint(dirpath=str(directory), filename=tcfg["RUN_NAME"] + "-{step}",
                                 monitor=None, save_last=True, save_top_k=-1, every_n_train_steps=every)
    logger = False if args.no_logger else TensorBoardLogger(str(directory), name="logs", version=tcfg["RUN_NAME"])
    trainer = pl.Trainer(accelerator="gpu", devices=args.devices,
                         strategy="ddp_find_unused_parameters_false" if args.devices > 1 else "auto",
                         precision=tcfg.get("PRECISION", "bf16"), max_steps=int(tcfg["MAX_STEPS"]),
                         val_check_interval=int(tcfg.get("VAL_CHECK_INTERVAL", every)),
                         check_val_every_n_epoch=None, num_sanity_val_steps=0,
                         limit_val_batches=tcfg.get("LIMIT_VAL_BATCHES", 1.0), logger=logger,
                         callbacks=[checkpoint, StreamAudit(directory), ThroughputCallback()],
                         accumulate_grad_batches=int(tcfg.get("ACCUMULATE_GRAD_BATCHES", 1)),
                         log_every_n_steps=20, default_root_dir=str(directory))
    resolved_path = directory / "config_resolved.yaml"
    resolved_path.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if k != "_config_path"}, sort_keys=False))
    # make_s36_row reads metadata.config_path, so point it to this fully resolved copy.
    metadata = {"config_path": str(resolved_path.resolve()), "source_config_path": cfg["_config_path"],
                "run_name": tcfg["RUN_NAME"], "seed": seed, "input_mode": train_ds.input_mode,
                "train_subjects": subjects, "train_sequences": [seq for seq, _ in train_ds.sequences],
                "val_core_sequences": [seq for seq, _ in val_ds.sequences],
                "splits_manifest": str(splits_manifest(cfg)), "train_samples": len(train_ds),
                "val_samples": len(val_ds), "batch_size_per_gpu": bsz, "devices": args.devices,
                "effective_batch": bsz * args.devices * int(tcfg.get("ACCUMULATE_GRAD_BATCHES", 1)),
                "max_steps": int(tcfg["MAX_STEPS"]), "save_every_n_steps": every,
                "selection_policy": "fixed last step; no validation selection", "warm_start": warm,
                "freeze": frozen, "data_generator_seed": seed, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                "provenance": _provenance(cfg["_config_path"]), "argv": sys.argv}
    (directory / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"seed": seed, "subjects": subjects, "train_samples": len(train_ds),
                      "freeze": frozen, "warm_start_expanded": warm["expanded_tensors"]}, indent=2), flush=True)
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
    print("done. last:", checkpoint.last_model_path, flush=True)


if __name__ == "__main__":
    main()
