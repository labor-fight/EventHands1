"""Paired four-arm training on chronological full-history stream lanes.

This is a real 6000-optimizer-step finetune, with the original S43 loss.
The frozen protocol and coordinator authorization must exist before training.
The CPU smoke command is engineering only and never runs an optimizer.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
import torch

from .core import RelConfig, RelationMemory
from .data import ContinuousRunLanes, EMATrace, FIT7, SELECT, representation_from_snapshot
from .geometry import GeometryConfig, ManoGeometry, event_intrinsics
from .model import load_relation_model


def digest_array(h, array):
    value = np.ascontiguousarray(array)
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                  default=lambda a: a.tolist() if hasattr(a, "tolist") else str(a)) + "\n")


class RelationRuntime:
    """Per-sequence observation memory and detached predicted-state context.

Used during optimization and available to replay: no representation builder
accepts targets, GT histories, GT visibility, or labels.
    """

    def __init__(self, model, rel_config=None, geometry_config=None, prep_workers=1):
        self.model, self.arm = model, model.arm
        self.rel_config = rel_config or RelConfig()
        self.geometry_config = geometry_config or GeometryConfig()
        self._thread_geometry = threading.local()
        self._pool = ThreadPoolExecutor(max_workers=int(prep_workers)) if int(prep_workers) > 1 else None
        self.memories, self.poses = {}, {}
        self.last_cutoffs = {}
        self.timing = {k: 0. for k in ("ingest_seconds", "geometry_seconds", "read_seconds", "commit_seconds")}
        self.queries = self.commits = self.resets = 0

    def _input_item(self, item):
        elapsed = {k: 0. for k in self.timing}
        sid = item["seq_id"]
        if item.get("reset", False):
            old = item.get("released_seq_id")
            tail = item.get("released_tail")
            if tail is not None and old in self.memories:
                if self.arm in ("full", "no_order"):
                    self.memories[old].ingest(tail["events"], watermark_ms=tail["cutoff_ms"],
                        packet_ordinal=tail["packet_ordinal"], source_range=tail["source_range"], time_unit="ms")
                else:
                    self.memories[old].ingest(tail["events"], tail["cutoff_ms"])
            if old in self.memories:
                del self.memories[old]
            self.poses.pop(old, None)
            self.last_cutoffs.pop(old, None)
            self.poses[sid] = np.asarray(item["initial_state"], np.float32).copy()
            self.last_cutoffs[sid] = item["start_ms"]
            if self.arm in ("full", "no_order"):
                memory = RelationMemory(self.rel_config, seq_id=sid)
                memory.reset(sid, boundary_ms=item["run_start_ms"])
                self.memories[sid] = memory
            elif self.arm == "enhanced":
                self.memories[sid] = EMATrace(item["run_start_ms"], self.rel_config.cell_size,
                                            self.rel_config.tau_ms)
        if sid not in self.poses:
            raise ValueError("sequence requires a real boundary reset before inputs")
        pose = self.poses[sid]
        started = time.perf_counter()
        if self.arm in ("full", "no_order"):
            memory = self.memories[sid]
            memory.ingest(item["events"], watermark_ms=item["cutoff_ms"],
                          packet_ordinal=item["packet_ordinal"], seq_id=sid,
                          source_range=item["source_range"], time_unit="ms")
            elapsed["ingest_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            if not hasattr(self._thread_geometry, "provider"):
                self._thread_geometry.provider = ManoGeometry(self.geometry_config)
            binding = self._thread_geometry.provider.bind(pose, item["betas"], event_intrinsics(item["K_cam"]), memory.topology)
            elapsed["geometry_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            if self.arm == "no_order":
                # Same polarity slots; never transpose p,q.  This erases
                # spatial order while keeping per-slot mass and scale.
                reverse = memory.topology.reverse[binding.edge_ids]
                union = np.union1d(binding.edge_ids, reverse)
                snap = memory.readout(item["cutoff_ms"], edge_ids=union)
                direct = np.searchsorted(union, binding.edge_ids)
                backward = np.searchsorted(union, reverse)
                symmetric = .5 * (snap.Z[direct] + snap.Z[backward])
                snap = dataclasses.replace(snap, Z=symmetric, edge_ids=binding.edge_ids)
            else:
                snap = memory.readout(item["cutoff_ms"], edge_ids=binding.edge_ids)
            geometry = binding.read(snap)
            elapsed["read_seconds"] += time.perf_counter() - started
        elif self.arm == "enhanced":
            values = self.memories[sid].ingest(item["events"], item["cutoff_ms"])
            elapsed["ingest_seconds"] += time.perf_counter() - started
            snap, geometry = SimpleNamespace(A=values), None
        else:
            snap, geometry = SimpleNamespace(A=None), None
        rep = representation_from_snapshot(item, snap, self.arm, geometry, self.rel_config.cell_size)
        rep["delta_commit_ms"] = item["cutoff_ms"] - self.last_cutoffs[sid]
        return rep, pose, item["betas"], item["K_cam"], snap, elapsed
    def inputs(self, items):
        started = time.perf_counter()
        if len(set(item["seq_id"] for item in items)) != len(items):
            raise ValueError("a parallel batch must contain independent sequence lanes")
        values = list(self._pool.map(self._input_item, items)) if self._pool else [self._input_item(item) for item in items]
        representations, previous, beta, camera, snapshots, timing = map(list, zip(*values))
        for elapsed in timing:
            for key, value in elapsed.items():
                self.timing[key] += value
        self.timing.setdefault("prep_wall_seconds", 0.)
        self.timing["prep_wall_seconds"] += time.perf_counter() - started
        self.queries += len(items)
        self.resets += sum(bool(item.get("reset")) for item in items)
        device = next(self.model.parameters()).device
        tensor = lambda v: torch.as_tensor(np.stack(v), device=device, dtype=torch.float32)
        return representations, tensor(previous), tensor(beta), tensor(camera), snapshots

    def commit(self, items, prediction, snapshots):
        started = time.perf_counter()
        predicted = prediction.detach().float().cpu().numpy()
        if predicted.shape != (len(items), 51) or not np.isfinite(predicted).all():
            raise ArithmeticError("invalid detached predicted-state commit")
        for item, pose, snap in zip(items, predicted, snapshots):
            sid = item["seq_id"]
            if self.arm in ("full", "no_order"):
                self.memories[sid].commit(snap)
            self.poses[sid] = pose.copy()
            self.last_cutoffs[sid] = item["cutoff_ms"]
            self.commits += 1
        self.timing["commit_seconds"] += time.perf_counter() - started

    def state_dict(self):
        return {"arm": self.arm, "poses": copy.deepcopy(self.poses), "last_cutoffs": dict(self.last_cutoffs),
                "memories": {sid: mem.state_dict() for sid, mem in self.memories.items()},
                "timing": dict(self.timing), "queries": self.queries, "commits": self.commits, "resets": self.resets}

    def load_state_dict(self, state):
        if state["arm"] != self.arm:
            raise ValueError("runtime checkpoint arm mismatch")
        factory = RelationMemory.from_state if self.arm in ("full", "no_order") else EMATrace.from_state
        self.memories = {sid: factory(mem) for sid, mem in state["memories"].items()}
        for key in ("poses", "last_cutoffs", "timing", "queries", "commits", "resets"):
            setattr(self, key, copy.deepcopy(state[key]))


def set_training_mode(model, freeze_bn=True):
    model.train()
    if freeze_bn:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


def lr_multiplier(step, steps, warmup):
    if step < warmup:
        return float(step + 1) / max(1, warmup)
    p = min(max((step - warmup) / float(max(1, steps - warmup)), 0.), 1.)
    return .02 + .98 * .5 * (1. + math.cos(math.pi * p))


def _source_files():
    root = Path(__file__).resolve().parents[2]
    # Freeze executable training dependencies, not unrelated replay/report
    # tools that can legitimately be implemented while this process runs.
    relative = ("tools/s43_rel/__init__.py", "tools/s43_rel/core.py", "tools/s43_rel/core_cpu.cpp",
        "tools/s43_rel/geometry.py", "tools/s43_rel/geometry_cpu.cpp", "tools/s43_rel/model.py",
        "tools/s43_rel/data.py", "tools/s43_rel/training.py", "tools/s43_rel/joint_reader.py", "tools/s43_integration/model.py",
        "tools/s43_br/data.py", "model/model.py", "model/mano_layer.py", "model/sparse_trunk.py",
        "model/pose_repr.py", "model/backbones.py", "model/render_fast.py", "semkine/events.py",
        "semkine/lie.py", "semkine/xag.py", "semkine/eval_track.py", "semkine/dataset.py")
    files = [root / path for path in relative]
    return {str(p): sha(p) for p in files}


def check_authorization(protocol_path, authorization_path, stage, config_path):
    """Coordinator-issued gates separate engineering from scientific training."""
    protocol_path, authorization_path = Path(protocol_path), Path(authorization_path)
    protocol, authorization = json.loads(protocol_path.read_text()), json.loads(authorization_path.read_text())
    if authorization.get("protocol_sha256") != sha(protocol_path):
        raise RuntimeError("authorization must bind the exact frozen protocol")
    if authorization.get(f"{stage}_training_authorized") is not True:
        raise RuntimeError(f"coordinator has not authorized {stage} scientific training")
    if authorization.get("mechanism_experiments_completed") is not True:
        raise RuntimeError("complete E1/E2/E3 before scientific training")
    configurations = authorization.get("config_sha256", {})
    if configurations.get(str(Path(config_path).resolve())) != sha(config_path):
        raise RuntimeError("training config is not frozen in the authorization")
    return protocol, authorization


def train(config_path, output_dir, *, stage, device, protocol, authorization, resume=None):
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    check_authorization(protocol, authorization, stage, config_path)
    rel, stage_cfg = cfg["REL"], cfg["STAGES"][stage]
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and resume is None:
        raise FileExistsError(f"preserve existing run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if str(device).startswith("cuda"):
        allowed = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not allowed or any(x not in {"4", "5", "6", "7"} for x in allowed.split(",")):
            raise RuntimeError("GPU training requires explicit CUDA_VISIBLE_DEVICES drawn only from physical 4-7")
    seed = int(rel["SEED"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(rel.get("CPU_THREADS", 4)))
    model, cfg = load_relation_model(cfg, device=device)
    subjects = FIT7 if stage == "small" else FIT7 + SELECT
    lanes = ContinuousRunLanes(rel["DATA_ROOT"], subjects=subjects,
                               batch_size=stage_cfg["BATCH_SIZE"], seed=seed)
    runtime = RelationRuntime(model, RelConfig(**cfg.get("MEMORY", {})), GeometryConfig(**cfg.get("GEOMETRY", {})), rel.get("PREP_WORKERS", 8))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=float(stage_cfg["LR"]))
    steps, clip = int(stage_cfg["STEPS"]), int(stage_cfg["CLIP_STEPS"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, steps, stage_cfg["WARMUP_STEPS"]))
    begin_step = 0
    cumulative_digest = hashlib.sha256()
    if resume:
        saved = torch.load(resume, map_location="cpu")
        if saved["arm"] != model.arm or saved["config_sha256"] != sha(config_path):
            raise ValueError("resume arm/config mismatch")
        if saved.get("reader", "points") != model.reader:
            raise ValueError("resume SUPPORT_READER mismatch")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        for value in optimizer.state.values():
            for key, tensor in value.items():
                if isinstance(tensor, torch.Tensor):
                    value[key] = tensor.to(device)
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        lanes.load_state_dict(saved["lanes"]); runtime.load_state_dict(saved["runtime"])
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        begin_step = saved["step"]
        # Per-step hashes remain comparable. The prior hash is explicitly
        # chained into the resumed trace, not falsely reconstructed as SHA state.
        cumulative_digest.update(saved["sample_digest"].encode())
    source_before = _source_files()
    dump(output_dir / "config_resolved.json", cfg)
    manifest = {"schema": "s43_relation_training_v1", "stage": stage, "arm": model.arm, "reader": model.reader,
                "seed": seed, "steps": steps, "batch_size": stage_cfg["BATCH_SIZE"], "clip_steps": clip,
                "TBPTT": 1, "supervised_outputs": steps * stage_cfg["BATCH_SIZE"] * clip,
                "historical_S43_supervised_samples": 6144000,
                "budget_fraction_of_historical": steps * stage_cfg["BATCH_SIZE"] * clip / 6144000.,
                "initialization": "same canonical S43 6000-step checkpoint; paired finetune",
                "base_checkpoint_sha256": sha(rel["BASE_CHECKPOINT"]), "params": model.counts(),
                "BN_statistics": "frozen" if rel.get("FREEZE_BN_STATS", True) else "trained",
                "loss": "unchanged original S43 so3_trans_fk and original log10 objective",
                "data": lanes.manifest(), "config_sha256": sha(config_path),
                "protocol_sha256": sha(protocol), "source_files": source_before,
                "checkpoint": "optimizer, scheduler, RNG, own-feedback, full numeric sensor states and lane cursors",
                "no_per_sample_Z_disk_cache": True, "resume": resume,
                "no_order": "0.5 * (Z_e,pq + Z_reverse(e),pq); no p/q exchange"}
    dump(output_dir / "manifest.json", manifest)
    log_path = output_dir / "curve.jsonl"
    start = time.perf_counter()
    bf16 = str(device).startswith("cuda") and rel.get("PRECISION", "bf16") == "bf16"
    for step in range(begin_step, steps):
        set_training_mode(model, rel.get("FREEZE_BN_STATS", True))
        optimizer.zero_grad(set_to_none=True)
        losses, step_digest = [], hashlib.sha256()
        network_seconds = 0.
        for phase in range(clip):
            batch = lanes.next_batch()
            for item in batch:
                step_digest.update(item["seq_id"].encode())
                digest_array(step_digest, np.asarray(item["source_range"], np.int64))
                digest_array(step_digest, item["events"])
                digest_array(step_digest, item["betas"])
                digest_array(step_digest, item["K_cam"])
                if item.get("released_tail") is not None:
                    digest_array(step_digest, item["released_tail"]["events"])
                    digest_array(step_digest, np.asarray(item["released_tail"]["source_range"], np.int64))
            reps, previous, betas, camera, snapshots = runtime.inputs(batch)
            # Labels are read only after every model input has been built.
            target = torch.as_tensor(lanes.targets(batch), device=device)
            network_started = time.perf_counter()
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=bf16):
                prediction = model(reps, previous, betas, camera)
                raw_loss, parts = model.loss(prediction, target, betas)
                objective = raw_loss.clamp_min(1e-12).log10() if model.base.log10_loss else raw_loss
            if not bool(torch.isfinite(objective)):
                raise ArithmeticError(f"nonfinite objective at step {step + 1}, clip phase {phase}")
            (objective / clip).backward()
            losses.append(float(raw_loss.detach()))
            runtime.commit(batch, prediction, snapshots)
            network_seconds += time.perf_counter() - network_started
        grad = float(torch.nn.utils.clip_grad_norm_(parameters, float(rel.get("GRAD_CLIP", 10.))))
        if not math.isfinite(grad):
            raise ArithmeticError("nonfinite gradient norm")
        optimizer.step(); scheduler.step()
        step_hash = step_digest.hexdigest(); cumulative_digest.update(step_hash.encode())
        record = {"step": step + 1, "loss": float(np.mean(losses)), "clip_raw_losses": losses,
                  "lr": optimizer.param_groups[0]["lr"], "grad_norm": grad,
                  "step_source_digest": step_hash, "sample_digest": cumulative_digest.hexdigest(),
                  "outputs": lanes.outputs, "events": lanes.events, "resets": runtime.resets,
                  "commits": runtime.commits, "seconds": time.perf_counter() - start,
                  "network_backward_seconds": network_seconds, **runtime.timing}
        with log_path.open("a") as file:
            file.write(json.dumps(record) + "\n")
        if step < 3 or (step + 1) % 10 == 0:
            print(json.dumps(dict(arm=model.arm, **record)), flush=True)
        if (step + 1) % int(stage_cfg.get("SAVE_EVERY", 500)) == 0 or step + 1 == steps:
            if _source_files() != source_before:
                raise RuntimeError("source changed during scientific training")
            checkpoint = {"schema": "s43_relation_training_v1", "arm": model.arm, "reader": model.reader, "cfg": cfg,
                          "config_sha256": sha(config_path), "step": step + 1,
                          "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                          "scheduler_state_dict": scheduler.state_dict(), "runtime": runtime.state_dict(),
                          "lanes": lanes.state_dict(), "python_rng": random.getstate(),
                          "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                          "cuda_rng": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else [],
                          "sample_digest": cumulative_digest.hexdigest()}
            temporary = output_dir / ".last.tmp.ckpt"
            torch.save(checkpoint, temporary); os.replace(temporary, output_dir / "last.ckpt")
            # Fixed grid evaluation checkpoints need weights, not 8 copies of
            # resident Z history per grid point. Only last.ckpt is resumable.
            grid = {k: v for k, v in checkpoint.items() if k in ("schema", "arm", "reader", "cfg", "step", "model_state_dict", "config_sha256")}
            torch.save(grid, output_dir / f"step={step + 1}.ckpt")
    dump(output_dir / "complete.json", dict(manifest, completed_steps=steps,
         actual_outputs=lanes.outputs, actual_events=lanes.events,
         seconds=time.perf_counter() - start, sample_digest=cumulative_digest.hexdigest(),
         checkpoint_sha256=sha(output_dir / "last.ckpt"), runtime_timing=runtime.timing,
         source_files_unchanged=_source_files() == source_before))


def smoke(config_path, output_path, threads=2):
    """Actual checkpoint forward/backward; no fitting and no SELECT/zgz read."""
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    torch.set_num_threads(threads)
    model, cfg = load_relation_model(cfg, device="cpu")
    lanes = ContinuousRunLanes(cfg["REL"]["DATA_ROOT"], subjects=("ch",), batch_size=2, seed=3407)
    runtime = RelationRuntime(model, RelConfig(**cfg.get("MEMORY", {})), GeometryConfig(**cfg.get("GEOMETRY", {})))
    batch = lanes.next_batch()
    reps, previous, betas, camera, snapshots = runtime.inputs(batch)
    model.eval()
    with torch.no_grad():
        old = model.base(torch.as_tensor(np.stack([r["lnes"] for r in reps])), previous, betas, camera)
    prediction = model(reps, previous, betas, camera)
    difference = float((prediction - old).abs().max())
    if difference != 0.:
        raise AssertionError("zero-initialized adapters must reproduce legacy exactly")
    target = torch.as_tensor(lanes.targets(batch))
    loss, _ = model.loss(prediction, target, betas)
    objective = loss.clamp_min(1e-12).log10() if model.base.log10_loss else loss
    objective.backward()
    gradients = {name: float(sum(p.grad.norm() for p in module.parameters() if p.grad is not None))
                 for name, module in (("sensor", model.sensor_adapter), ("typed", model.typed_adapter), ("base", model.base))}
    runtime.commit(batch, prediction, snapshots)
    # A second query consumes new observations and retains own feedback and Z.
    second = lanes.next_batch()
    reps2, previous2, *_ = runtime.inputs(second)
    if not np.array_equal(previous2.numpy(), prediction.detach().numpy()):
        raise AssertionError("second window did not use this arm's own detached prediction")
    dump(output_path, {"scope": "engineering forward/backward only, zero optimizer updates",
         "arm": model.arm, "legacy_max_abs": difference, "loss": float(loss.detach()),
         "finite": bool(torch.isfinite(prediction).all()), "gradient_norm_sums": gradients,
         "params": model.counts(), "own_feedback_exact": True, "history_reset_between_windows": False,
         "source_events": sum(len(b["events"]) for b in batch), "timing": runtime.timing,
         "GT_input": False, "target_role": "unchanged loss only", "zgz_read": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--config", required=True)
    smoke_parser.add_argument("--out", required=True)
    smoke_parser.add_argument("--threads", type=int, default=2)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--out-dir", required=True)
    train_parser.add_argument("--stage", choices=("small", "full"), required=True)
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--protocol", required=True)
    train_parser.add_argument("--authorization", required=True)
    train_parser.add_argument("--resume")
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.config, args.out, args.threads)
    else:
        train(args.config, args.out_dir, stage=args.stage, device=args.device,
              protocol=args.protocol, authorization=args.authorization, resume=args.resume)


if __name__ == "__main__":
    main()
