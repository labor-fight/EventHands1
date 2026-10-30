"""Complete causal recording replay with independent ingest/query/commit.

Every native valid-run millisecond is consumed exactly once, including the
two canonical warmup windows and unscored tails.  Query cadence and the W50
LNES observation window are independent.  No target is read until replay ends.
"""
from __future__ import annotations

import argparse
from collections import deque
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch

from tools.s43_br.data import DATA_ROOT, PUBLIC_INITIAL_STATE, SequenceReader
from tools.s43_integration.model import load_legacy
from tools.s43_integration.replay import _evaluation_functions, _metrics
from .core import RelConfig, RelationMemory
from .data import EMATrace, ContinuousRunLanes, last_lnes, representation_from_snapshot
from .geometry import GeometryConfig, ManoGeometry, event_intrinsics
from .model import load_relation_model
from .training import dump

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gpu_sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def validate_device(device):
    if torch.device(device).type == "cuda":
        allowed = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not allowed or any(value not in {"4", "5", "6", "7"} for value in allowed.split(",")):
            raise RuntimeError("set CUDA_VISIBLE_DEVICES to physical GPU 4-7 only")


class RecentWindow:
    """W50 raw support with an explicit rate-domain event cap.

The cap is checked at every complete native-ms group in a submitted packet,
so accept/reject does not depend on packet boundaries.  Over-domain packets
are rejected before observation state changes; real multiplicity is never
clipped.  Caller-owned input arrays and temporary query tensors are separate
from the bounded persistent support.
    """

    def __init__(self, boundary_ms, window_ms=50, max_window_events=262144):
        self.boundary_ms, self.window_ms = int(boundary_ms), int(window_ms)
        if isinstance(max_window_events, bool) or int(max_window_events) != max_window_events or max_window_events < 1:
            raise ValueError("max_window_events must be a positive integer")
        self.max_window_events = int(max_window_events)
        self.watermark_ms = int(boundary_ms)
        self.blocks = deque()
        self.groups = deque()
        self.event_count = self.peak_event_count = 0

    def prepare_ingest(self, events, watermark_ms):
        """Pure preflight: allocate bounded retained support, without mutation."""
        events = np.asarray(events)
        if events.ndim != 2 or events.shape[1] != 4:
            raise ValueError("rolling window expects events[N,4]")
        if not np.isfinite(watermark_ms) or int(watermark_ms) != watermark_ms:
            raise ValueError("rolling window uses complete integer-ms watermarks")
        if watermark_ms < self.watermark_ms:
            raise ValueError("rolling window watermark cannot rewind")
        if len(events) and (np.any(events[:, 2] < self.watermark_ms) or np.any(events[:, 2] >= watermark_ms)):
            raise ValueError("rolling window cannot consume duplicate/future events")
        if len(events):
            times = events[:, 2]
            if (not np.isfinite(times).all() or np.any(times != np.floor(times))
                    or np.any(times[1:] < times[:-1])):
                raise ValueError("rolling window requires sorted integer-ms groups")
        groups, count, peak = deque(self.groups), self.event_count, self.peak_event_count
        if len(events):
            starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
            stops = np.r_[starts[1:], len(times)]
            for begin, end in zip(starts, stops):
                stamp = int(times[begin])
                oldest = stamp + 1 - self.window_ms
                while groups and groups[0][0] < oldest:
                    count -= groups.popleft()[1]
                n = int(end - begin)
                groups.append((stamp, n)); count += n
                if count > self.max_window_events:
                    raise OverflowError("W50 raw support exceeds max_window_events; packet rejected before observation update")
                peak = max(peak, count)
        oldest = int(watermark_ms) - self.window_ms
        while groups and groups[0][0] < oldest:
            count -= groups.popleft()[1]
        blocks = deque(block for block in self.blocks if block[-1, 2] >= oldest)
        if blocks and blocks[0][0, 2] < oldest:
            blocks[0] = blocks[0][blocks[0][:, 2] >= oldest].copy()
        if len(events):
            keep = events[:, 2] >= oldest
            if np.any(keep):
                # int64 is the established native representation.  Its fixed
                # 32-byte event stride makes the persistent byte limit explicit.
                blocks.append(np.asarray(events[keep], dtype=np.int64).copy())
        return blocks, groups, int(watermark_ms), count, peak

    def apply_ingest(self, prepared):
        self.blocks, self.groups, self.watermark_ms, self.event_count, self.peak_event_count = prepared

    def ingest(self, events, watermark_ms):
        self.apply_ingest(self.prepare_ingest(events, watermark_ms))

    def read(self, cutoff_ms):
        if int(cutoff_ms) != self.watermark_ms:
            raise ValueError("W50 read requires the current complete prefix")
        if not self.blocks:
            return np.empty((0, 4), np.int64)
        out = np.concatenate(self.blocks)
        return out[out[:, 2] >= cutoff_ms - self.window_ms]

    @property
    def nbytes(self):
        return sum(block.nbytes for block in self.blocks)


class CausalRelationTracker:
    """Online twin of the training builder, with pure queries and explicit commits."""

    def __init__(self, model, rel_config=None, geometry_config=None, max_window_events=262144):
        self.model = model.eval()
        self.arm = model.arm
        self.rel_config = rel_config or RelConfig()
        self.provider = ManoGeometry(geometry_config or GeometryConfig()) if self.arm in ("full", "no_order") else None
        self.device = next(model.parameters()).device
        self.max_window_events = max_window_events
        self.active = False

    def reset(self, seq_id, boundary_ms, pose, betas, K_cam):
        self.seq_id, self.pose = seq_id, np.asarray(pose, np.float32).copy()
        self.betas, self.K_cam = np.asarray(betas, np.float32).copy(), np.asarray(K_cam, np.float32).copy()
        self.window = RecentWindow(boundary_ms, max_window_events=self.max_window_events)
        self.watermark_ms = self.last_commit_ms = int(boundary_ms)
        self.pose_version = self.observation_version = self.commits = 0
        self.last_ordinal, self.last_source_end = -1, None
        self.pending = None
        if self.arm in ("full", "no_order"):
            self.memory = RelationMemory(self.rel_config, seq_id)
            self.memory.reset(seq_id, boundary_ms)
        elif self.arm == "enhanced":
            self.memory = EMATrace(boundary_ms, self.rel_config.cell_size, self.rel_config.tau_ms)
        else:
            self.memory = None
        self.active = True

    def ingest(self, events, *, watermark_ms, packet_ordinal, source_range):
        if not self.active:
            raise ValueError("reset before ingest")
        events = np.asarray(events)
        lo, hi = (int(value) for value in source_range)
        if packet_ordinal <= self.last_ordinal or hi - lo != len(events) or (self.last_source_end is not None and lo < self.last_source_end):
            raise ValueError("source range/ordinal is duplicate or inconsistent")
        if watermark_ms < self.watermark_ms:
            raise ValueError("watermark rewind")
        # Overflow/malformed W50 support must fail before A/Z, EMA, source
        # ledger or pending predictions can change.  The final apply cannot
        # allocate or revalidate after the sensor core has advanced.
        prepared_window = self.window.prepare_ingest(events, watermark_ms)
        if self.arm in ("full", "no_order"):
            self.memory.ingest(events, watermark_ms=watermark_ms, packet_ordinal=packet_ordinal,
                               seq_id=self.seq_id, time_unit="ms", source_range=(lo, hi))
        elif self.arm == "enhanced":
            self.memory.ingest(events, watermark_ms)
        self.window.apply_ingest(prepared_window)
        self.watermark_ms, self.last_ordinal, self.last_source_end = int(watermark_ms), int(packet_ordinal), hi
        self.observation_version += 1
        self.pending = None

    def representation(self, cutoff_ms):
        if not self.active or cutoff_ms != self.watermark_ms:
            raise ValueError("query requires the current complete prefix")
        elapsed = {"geometry_seconds": 0., "readout_seconds": 0., "representation_seconds": 0.}
        started = time.perf_counter()
        if self.arm in ("full", "no_order"):
            binding = self.provider.bind(self.pose, self.betas, event_intrinsics(self.K_cam), self.memory.topology)
            elapsed["geometry_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            if self.arm == "no_order":
                reverse = self.memory.topology.reverse[binding.edge_ids]
                union = np.union1d(binding.edge_ids, reverse)
                snapshot = self.memory.readout(cutoff_ms, edge_ids=union)
                direct, backward = np.searchsorted(union, binding.edge_ids), np.searchsorted(union, reverse)
                snapshot = dataclasses.replace(snapshot, Z=.5 * (snapshot.Z[direct] + snapshot.Z[backward]),
                                               edge_ids=binding.edge_ids)
            else:
                snapshot = self.memory.readout(cutoff_ms, edge_ids=binding.edge_ids)
            geometry = binding.read(snapshot)
            elapsed["readout_seconds"] = time.perf_counter() - started
        elif self.arm == "enhanced":
            snapshot, geometry = SimpleNamespace(A=self.memory.A.copy()), None
        else:
            snapshot, geometry = SimpleNamespace(A=None), None
        started = time.perf_counter()
        observed = self.window.read(cutoff_ms)
        item = {"events": observed, "lnes": last_lnes(observed, cutoff_ms - 50, cutoff_ms),
                "start_ms": cutoff_ms - 50, "cutoff_ms": cutoff_ms, "seq_id": self.seq_id,
                "source_range": (0, 0), "packet_ordinal": self.last_ordinal}
        rep = representation_from_snapshot(item, snapshot, self.arm, geometry, self.rel_config.cell_size)
        rep["delta_commit_ms"] = cutoff_ms - self.last_commit_ms
        rep["meta"].update(observation_window_ms=50, ingest_watermark_ms=self.watermark_ms,
                           source_range_semantics="W50 read support, not a new source submission")
        elapsed["representation_seconds"] = time.perf_counter() - started
        return rep, snapshot, elapsed

    @torch.inference_mode()
    def query(self, cutoff_ms):
        """Does not modify the observation ledger or committed hand context."""
        rep, snapshot, costs = self.representation(cutoff_ms)
        gpu_sync(self.device)
        started = time.perf_counter()
        previous = torch.as_tensor(self.pose[None], device=self.device)
        beta = torch.as_tensor(self.betas[None], device=self.device)
        camera = torch.as_tensor(self.K_cam[None], device=self.device)
        gpu_sync(self.device)
        costs["context_transfer_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        prediction = self.model([rep], previous, beta, camera)
        gpu_sync(self.device)
        # The existing model performs sensor/typed H2D inside forward.  It is
        # included here, not falsely excluded from decode cost.
        costs["decode_including_observation_transfer_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        value = prediction[0].float().cpu().numpy().copy()
        gpu_sync(self.device)
        costs["prediction_transfer_seconds"] = time.perf_counter() - started
        if value.shape != (51,) or not np.isfinite(value).all():
            raise ArithmeticError("nonfinite/invalid prediction")
        result = {"state": value, "seq_id": self.seq_id, "cutoff_ms": int(cutoff_ms),
                  "target_ms": int(cutoff_ms) - 1, "pose_version": self.pose_version,
                  "observation_version": self.observation_version, "snapshot": snapshot, "costs": costs}
        self.pending = result
        return result

    def commit(self, result):
        if (result is not self.pending or result["seq_id"] != self.seq_id
                or result["pose_version"] != self.pose_version
                or result["observation_version"] != self.observation_version
                or result["cutoff_ms"] <= self.last_commit_ms):
            raise ValueError("stale, duplicate or foreign query commit")
        if self.arm in ("full", "no_order"):
            self.memory.commit(result["snapshot"])
        self.pose = result["state"].copy()
        self.last_commit_ms = result["cutoff_ms"]
        self.pose_version += 1
        self.commits += 1
        self.pending = None

    def canonical_commit(self, cutoff_ms, pose):
        """Identical public canonical warmup; observations remain continuous."""
        if cutoff_ms != self.watermark_ms or cutoff_ms <= self.last_commit_ms:
            raise ValueError("invalid canonical warmup commit")
        if self.arm in ("full", "no_order"):
            self.memory.commit(self.memory.readout(cutoff_ms, edge_ids=[], include_A=False))
        self.pose = np.asarray(pose, np.float32).copy()
        self.last_commit_ms = int(cutoff_ms)
        self.pose_version += 1
        self.commits += 1
        self.pending = None

    def evidence_digest(self):
        h = hashlib.sha256()
        if self.arm in ("full", "no_order"):
            for name in self.memory._array_names():
                h.update(getattr(self.memory, name).tobytes())
            h.update(str((self.memory.watermark_ms, self.memory.version, self.memory.commit_count)).encode())
        elif self.arm == "enhanced":
            h.update(self.memory.A.tobytes())
            h.update(str(self.memory.cutoff_ms).encode())
        h.update(str((self.observation_version, self.last_source_end, self.last_ordinal)).encode())
        return h.hexdigest()

    def memory_stats(self):
        if self.arm in ("full", "no_order"):
            value = self.memory.stats()
        elif self.arm == "enhanced":
            value = {"resident_bytes": self.memory.A.nbytes + self.memory.tau_ms.nbytes,
                     "shared_topology_bytes": 0}
        else:
            value = {"resident_bytes": 0, "shared_topology_bytes": 0}
        return dict(value, raw_window_bytes=self.window.nbytes,
                    raw_window_event_count=self.window.event_count,
                    raw_window_peak_events=self.window.peak_event_count,
                    max_window_events=self.window.max_window_events,
                    raw_window_payload_byte_limit=self.window.max_window_events * 32,
                    hand_context_bytes=self.pose.nbytes + self.betas.nbytes + self.K_cam.nbytes)

    def release(self):
        if self.arm in ("full", "no_order"):
            self.memory.release()
        self.memory = self.window = self.pending = None
        self.active = False


def duplicate_query_check(tracker, cutoff_ms):
    digest = tracker.evidence_digest()
    pose, version, commits = tracker.pose.copy(), tracker.pose_version, tracker.commits
    one = tracker.query(cutoff_ms)
    two = tracker.query(cutoff_ms)
    exact = np.array_equal(one["state"], two["state"])
    if not (exact and digest == tracker.evidence_digest() and np.array_equal(pose, tracker.pose)
            and version == tracker.pose_version and commits == tracker.commits):
        raise AssertionError("pure duplicate query changed observations/context/prediction")
    return two, {"cutoff_ms": cutoff_ms, "prediction_max_abs": float(np.max(np.abs(one["state"] - two["state"]))),
                 "evidence_unchanged": True, "hand_context_unchanged": True, "commit_count_unchanged": True}


def expected_source(reader, runs):
    h, count = hashlib.sha256(), 0
    for start, stop in runs:
        lo, hi = int(reader.offsets[start]), int(reader.offsets[stop])
        count += hi - lo
        for row in range(lo, hi, 65536):
            h.update(np.arange(row, min(hi, row + 65536), dtype=np.int64).tobytes())
    return count, h.hexdigest()


def source_snapshot(config, checkpoint, reader):
    paths = [ROOT / p for p in ("tools/s43_rel/replay.py", "tools/s43_rel/core.py", "tools/s43_rel/core_cpu.cpp",
        "tools/s43_rel/geometry.py", "tools/s43_rel/geometry_cpu.cpp", "tools/s43_rel/data.py", "tools/s43_rel/model.py",
        "tools/s43_rel/training.py", "tools/s43_rel/joint_reader.py", "tools/s43_integration/model.py", "tools/s43_integration/replay.py",
        "tools/s43_br/data.py", "model/model.py", "model/mano_layer.py", "model/pose_repr.py",
        "model/sparse_trunk.py", "model/backbones.py", "model/render_fast.py", "semkine/eval_track.py",
        "assets/mano_right.npz")]
    paths += [Path(config), reader.root / "splits_semkine.json"]
    if checkpoint is not None:
        paths.append(Path(checkpoint))
    for suffix in ("_events.npy", "_offsets.npy", "_aux.npz", ".meta"):
        paths.append(Path(str(reader.base) + suffix))
    return {str(p.resolve()): {"sha256": sha(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(set(paths))}


def summarize_latencies(seconds):
    if not seconds:
        return {"n": 0}
    value = np.asarray(seconds, np.float64) * 1000.
    return dict(n=len(value), mean_ms=float(value.mean()), p50_ms=float(np.percentile(value, 50)),
                p95_ms=float(np.percentile(value, 95)), p99_ms=float(np.percentile(value, 99)))


def replay(config, checkpoint, recording, out, *, hop=50, device="cpu", threads=2,
           duplicate_query=False, fixed_commit_ms=None, root=DATA_ROOT, max_window_events=262144):
    validate_device(device)
    if hop not in (25, 50, 100) or fixed_commit_ms not in (None, 50):
        raise ValueError("registered query intervals are 25/50/100ms; optional fixed commit is 50ms")
    if fixed_commit_ms is not None and hop > fixed_commit_ms:
        raise ValueError("fixed50 commit requires queries at25or50ms")
    prefix = Path(out)
    if prefix.with_suffix(".json").exists() or prefix.with_suffix(".npz").exists():
        raise FileExistsError(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(threads))
    load_started = time.perf_counter()
    model, cfg = load_relation_model(config, checkpoint=checkpoint, device=device)
    canonical, _ = load_legacy(cfg["REL"]["BASE_CHECKPOINT"], cfg["REL"]["BASE_CONFIG"], device)
    model.eval(); canonical.eval()
    tracker = CausalRelationTracker(model, RelConfig(**cfg.get("MEMORY", {})), GeometryConfig(**cfg.get("GEOMETRY", {})), max_window_events)
    reader = SequenceReader(recording, Path(root))
    runs = np.asarray(reader.runs, np.int64).reshape(-1, 2)
    if np.any(runs[:, 0] < 0) or np.any(runs[:, 1] < runs[:, 0]) or np.any(runs[:, 1] >= len(reader.offsets)):
        raise ValueError("invalid valid-run intervals")
    if len(runs) > 1 and np.any(runs[1:, 0] < runs[:-1, 1]):
        raise ValueError("valid runs overlap: duplicate source consumption forbidden")
    expected_count, expected_digest = expected_source(reader, runs)
    source_started = time.perf_counter()
    sources = source_snapshot(config, checkpoint, reader)
    for path in (cfg["REL"]["BASE_CHECKPOINT"], cfg["REL"]["BASE_CONFIG"]):
        p = Path(path); sources[str(p.resolve())] = {"sha256": sha(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
    source_hash_seconds = time.perf_counter() - source_started
    load_seconds = time.perf_counter() - load_started
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    account = {"count": 0, "digest": hashlib.sha256(), "ms_packets": 0}
    predictions, ends, run_ids, counts, latencies, audits, run_records = [], [], [], [], [], [], []
    costs = {k: 0. for k in ("reset_compile_seconds", "ingest_seconds", "geometry_seconds", "readout_seconds",
        "representation_seconds", "context_transfer_seconds", "decode_including_observation_transfer_seconds",
        "prediction_transfer_seconds", "commit_seconds", "warmup_seconds", "tail_seconds", "query_audit_seconds")}
    memories = []
    stream_started = time.perf_counter()

    def consume(start, stop, ordinal):
        for ms in range(start, stop):
            raw = ContinuousRunLanes._packet(reader, ms, ms + 1)
            tracker.ingest(raw["events"], watermark_ms=ms + 1, packet_ordinal=ordinal,
                           source_range=raw["source_range"])
            lo, hi = raw["source_range"]
            account["digest"].update(np.arange(lo, hi, dtype=np.int64).tobytes())
            account["count"] += hi - lo
            account["ms_packets"] += 1
            ordinal += 1
        return ordinal

    try:
        with torch.inference_mode():
            for run_id, (a, b) in enumerate(runs):
                a, b = int(a), int(b)
                if a == b:
                    run_records.append({"run_id": run_id, "start_ms": a, "stop_ms": b, "source_count": 0, "outputs": 0})
                    continue
                run_started = time.perf_counter(); before_count, before_outputs = account["count"], len(predictions)
                reset_started = time.perf_counter()
                tracker.reset(f"{recording}:run{run_id}", a, PUBLIC_INITIAL_STATE, reader.betas, reader.K_cam)
                costs["reset_compile_seconds"] += time.perf_counter() - reset_started
                cursor, ordinal, warmups = a, 0, 0
                warm_started = time.perf_counter()
                for _ in range(2):
                    if b - cursor < 50:
                        break
                    ordinal = consume(cursor, cursor + 50, ordinal)
                    cursor += 50
                    observed = tracker.window.read(cursor)
                    lnes = last_lnes(observed, cursor - 50, cursor)
                    previous = torch.as_tensor(tracker.pose[None], device=device)
                    beta = torch.as_tensor(reader.betas[None], device=device)
                    camera = torch.as_tensor(reader.K_cam[None], device=device)
                    history = canonical(torch.as_tensor(lnes[None], device=device), previous, beta, camera)
                    gpu_sync(device)
                    tracker.canonical_commit(cursor, history[0].float().cpu().numpy())
                    warmups += 1
                costs["warmup_seconds"] += time.perf_counter() - warm_started
                if warmups == 2:
                    while b - cursor >= hop:
                        gpu_sync(device); cycle_started = time.perf_counter(); before_query_count = account["count"]
                        started = time.perf_counter(); ordinal = consume(cursor, cursor + hop, ordinal)
                        costs["ingest_seconds"] += time.perf_counter() - started
                        cursor += hop
                        result = tracker.query(cursor)
                        for key, value in result["costs"].items():
                            costs[key] += value
                        should_commit = fixed_commit_ms is None or (cursor - a - 100) % fixed_commit_ms == 0
                        started = time.perf_counter()
                        if should_commit:
                            tracker.commit(result)
                        gpu_sync(device)
                        costs["commit_seconds"] += time.perf_counter() - started
                        latency = time.perf_counter() - cycle_started
                        latencies.append(latency)
                        predictions.append(result["state"]); ends.append(cursor - 1); run_ids.append(run_id)
                        counts.append(account["count"] - before_query_count)
                        if duplicate_query and len(predictions) == 1:
                            # Audit after commit at the *same observation prefix*;
                            # no extra feedback is submitted by these queries.
                            started = time.perf_counter()
                            _, check = duplicate_query_check(tracker, cursor)
                            costs["query_audit_seconds"] += time.perf_counter() - started
                            audits.append(check)
                        memories.append(tracker.memory_stats())
                        if len(predictions) % 100 == 0:
                            print(json.dumps({"recording": recording, "arm": model.arm, "outputs": len(predictions),
                                "source_events": account["count"], "seconds": time.perf_counter() - stream_started}), flush=True)
                tail_ms, tail_events = b - cursor, 0
                if cursor < b:
                    started = time.perf_counter(); before_tail = account["count"]
                    ordinal = consume(cursor, b, ordinal)
                    tail_events = account["count"] - before_tail
                    costs["tail_seconds"] += time.perf_counter() - started
                memories.append(tracker.memory_stats())
                tracker.release()
                consumed = account["count"] - before_count
                if consumed != int(reader.offsets[b] - reader.offsets[a]):
                    raise AssertionError("valid-run source count mismatch")
                run_records.append({"run_id": run_id, "start_ms": a, "stop_ms": b, "warmup_windows": warmups,
                    "source_count": consumed, "outputs": len(predictions) - before_outputs,
                    "tail_ms": tail_ms, "tail_events": tail_events, "seconds": time.perf_counter() - run_started})
        gpu_sync(device)
        stream_seconds = time.perf_counter() - stream_started
        pipeline_RSS_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        if account["count"] != expected_count or account["digest"].hexdigest() != expected_digest:
            raise AssertionError("full recording source count/digest mismatch")
        gpu_memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)} if torch.device(device).type == "cuda" else {}
        score_started = time.perf_counter()
        pred = np.asarray(predictions, np.float32).reshape(-1, 51)
        # First semantic target read, strictly after all online inference.
        target = np.stack([reader.target(end) for end in ends]) if ends else np.empty((0, 51), np.float32)
        metrics, per = _metrics(_evaluation_functions(), model.base.mano.cpu(), pred, target, reader.betas)
        scoring_seconds = time.perf_counter() - score_started
        np.savez_compressed(prefix.with_suffix(".npz"), pred=pred, gt=target,
            end_ms=np.asarray(ends, np.int64), run_id=np.asarray(run_ids, np.int64),
            n_new_events=np.asarray(counts, np.int64), latency_seconds=np.asarray(latencies), **per)
        # Stat plus cryptographic verification detects both in-place source
        # edits and atomic replacements. This cost is outside pipeline latency.
        verify_started = time.perf_counter()
        unchanged = {path: sha(path) == value["sha256"] for path, value in sources.items()}
        if not all(unchanged.values()):
            raise RuntimeError("source/config/checkpoint/data changed during replay")
        source_verify_seconds = time.perf_counter() - verify_started
        report = {"schema": "s43_relation_full_recording_replay_v1", "recording": recording,
            "arm": model.arm, "config": str(config), "checkpoint": None if checkpoint is None else str(checkpoint),
            "device": str(device), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "hop_ms": hop, "observation_window_ms": 50, "output_lag_ms": 1,
            "max_window_events": int(max_window_events),
            "raw_window_overflow_policy": "reject before observation update; no clipping; cap checked at every complete integer-ms prefix",
            "commit_schedule_ms": hop if fixed_commit_ms is None else fixed_commit_ms,
            "initialization": "public constant + same two frozen canonical S43 W50 predictions per valid run",
            "initialization_GT": False, "target_role": "scoring only after complete causal replay",
            "full_recording": True, "native_ms_packets": account["ms_packets"], "source_count": account["count"],
            "expected_source_count": expected_count, "source_id_digest": account["digest"].hexdigest(),
            "expected_source_id_digest": expected_digest, "all_valid_events_consumed_once": True,
            "outputs": len(pred), "metrics": metrics, "runs": run_records, "duplicate_query_checks": audits,
            "params": model.counts(), "stream_seconds": stream_seconds, "scoring_seconds": scoring_seconds,
            "load_seconds": load_seconds, "source_hash_seconds": source_hash_seconds,
            "source_verify_seconds": source_verify_seconds, "costs": costs,
            "pipeline_latency": summarize_latencies(latencies),
            "events_per_second_including_warmup_and_tails": expected_count / max(stream_seconds, 1e-12),
            "outputs_per_second_including_warmup_and_tails": len(pred) / max(stream_seconds, 1e-12),
            "latency_scope": "synchronized full playback service path: native ingestion/geometry/transfers/decode/commit; excludes actual sensor waiting and scoring",
            "RSS_peak_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "RSS_peak_through_pipeline_bytes": pipeline_RSS_peak,
            "cuda_memory": gpu_memory,
            "peak_resident_observation_bytes": max((m["resident_bytes"] for m in memories), default=0),
            "peak_shared_topology_bytes": max((m.get("shared_topology_bytes", 0) for m in memories), default=0),
            "peak_raw_window_bytes": max((m["raw_window_bytes"] for m in memories), default=0),
            "peak_raw_window_events": max((m["raw_window_peak_events"] for m in memories), default=0),
            "raw_window_payload_byte_limit": int(max_window_events) * 32,
            "source_files": sources, "source_files_unchanged": unchanged, "argv": sys.argv,
            "scientific_scope": "public-init W50/raw complete-flow evaluation; not historical GT-assisted official main row"}
        dump(prefix.with_suffix(".json"), report)
        return report
    except BaseException as error:
        dump(Path(str(prefix) + "_failed.json"), {"error": repr(error), "source_count": account["count"],
            "outputs": len(predictions), "seconds": time.perf_counter() - stream_started, "runs": run_records})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--recording", required=True)
    parser.add_argument("--hop", type=int, choices=(25, 50, 100), default=50)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--root", default=str(DATA_ROOT))
    parser.add_argument("--duplicate-query", action="store_true")
    parser.add_argument("--fixed-commit-ms", type=int, choices=(50,))
    parser.add_argument("--max-window-events", type=int, default=262144)
    args = parser.parse_args()
    replay(args.config, args.checkpoint, args.recording, args.out, hop=args.hop, device=args.device,
           threads=args.threads, duplicate_query=args.duplicate_query, fixed_commit_ms=args.fixed_commit_ms, root=args.root,
           max_window_events=args.max_window_events)


if __name__ == "__main__":
    main()
