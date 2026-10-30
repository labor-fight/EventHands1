"""Target-free full-pipeline service benchmark on declared causal prefixes.

Native recording prefixes and synthetic event-rate stress are separate modes.
This tool never reports pose accuracy and never reads a target pose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch

from tools.s43_br.data import DATA_ROOT, PUBLIC_INITIAL_STATE, recording_dirs
from tools.s43_integration.model import load_legacy
from .core import RelConfig
from .data import ContinuousRunLanes, last_lnes
from .geometry import GeometryConfig, ASSET
from .model import load_relation_model
from .replay import CausalRelationTracker, duplicate_query_check, gpu_sync, sha, summarize_latencies, validate_device
from .training import dump
from .deployment import decode_surface


class ObservationReader:
    """Resource-only observation mappings; no pose metadata or target API.

    K/beta and valid-run intervals retain the main experiment's external
    calibration/recording selection condition. Native tsub is intentionally
    unused because the representation contract is the offset's integer-ms bin.
    """
    def __init__(self, recording, root):
        self.root = Path(root)
        self.base = self.root / recording_dirs(self.root)[recording] / recording
        self.events = np.load(str(self.base) + "_events.npy", mmap_mode="r", allow_pickle=False)
        self.offsets = np.load(str(self.base) + "_offsets.npy", mmap_mode="r", allow_pickle=False)
        with np.load(str(self.base) + "_aux.npz", allow_pickle=False) as aux:
            self.betas = np.asarray(aux["betas"], np.float32).reshape(10).copy()
            self.K_cam = np.asarray(aux["camera_K"], np.float32).reshape(3, 3).copy()
            self.runs = np.asarray(aux["valid_runs_ms"], np.int64).copy()
        if self.events.ndim != 2 or self.events.shape[1] != 3 or int(self.offsets[-1]) != len(self.events):
            raise ValueError("invalid resource benchmark event/offset mappings")


def stream_sha(path):
    identity = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            identity.update(block)
    return identity.hexdigest()


def source_identity(config, checkpoint, cfg, reader):
    from .training import _source_files
    paths = [Path(path) for path in _source_files()]
    paths += [Path(__file__), Path(__file__).with_name("replay.py"), Path(__file__).with_name("deployment.py"),
              Path(config), Path(cfg["REL"]["BASE_CHECKPOINT"]), Path(cfg["REL"]["BASE_CONFIG"]),
              reader.root / "splits_semkine.json", ASSET]
    paths += [Path(str(reader.base) + suffix) for suffix in ("_events.npy", "_offsets.npy", "_aux.npz")]
    if checkpoint:
        paths.append(Path(checkpoint))
    return {str(path.resolve()): {"sha256": stream_sha(path), "bytes": path.stat().st_size}
            for path in sorted(set(paths))}


def benchmark(config, checkpoint, out, *, device="cpu", recording="ch_local", hop=50,
              queries=16, threads=2, synthetic_events_per50=None, seed=3407, root=DATA_ROOT):
    validate_device(device)
    if hop not in (25, 50, 100) or queries < 1:
        raise ValueError("positive query budget and registered query interval required")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(threads)
    started = time.perf_counter()
    model, cfg = load_relation_model(config, checkpoint, device)
    canonical, _ = load_legacy(cfg["REL"]["BASE_CHECKPOINT"], cfg["REL"]["BASE_CONFIG"], device)
    tracker = CausalRelationTracker(model, RelConfig(**cfg.get("MEMORY", {})), GeometryConfig(**cfg.get("GEOMETRY", {})))
    reader = ObservationReader(recording, Path(root))
    needed_ms = 100 + hop * queries
    eligible = [(int(a), int(b)) for a, b in reader.runs if int(b) - int(a) >= needed_ms]
    if not eligible:
        raise ValueError("no valid segment long enough for the declared target-free benchmark")
    start_ms, stop_ms = eligible[0]
    source_hash_started = time.perf_counter()
    sources = source_identity(config, checkpoint, cfg, reader)
    source_hash_seconds = time.perf_counter() - source_hash_started
    load_seconds = time.perf_counter() - started
    reset_started = time.perf_counter()
    tracker.reset(f"benchmark:{recording}:{start_ms}", start_ms, PUBLIC_INITIAL_STATE, reader.betas, reader.K_cam)
    reset_seconds = time.perf_counter() - reset_started
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rng = np.random.default_rng(seed)
    ordinal = source_count = 0
    source_digest = hashlib.sha256()

    def consume(a, b):
        nonlocal ordinal, source_count
        if synthetic_events_per50 is None:
            packets = [ContinuousRunLanes._packet(reader, ms, ms + 1) for ms in range(a, b)]
        else:
            n = int(round(synthetic_events_per50 * (b - a) / 50.))
            stamps = np.sort(rng.integers(a, b, n), kind="stable")
            stream = np.column_stack((rng.integers(0, 240, n), rng.integers(0, 180, n), stamps,
                                      rng.integers(0, 2, n))).astype(np.int64)
            packets = []
            row = source_count
            for ms in range(a, b):
                events = stream[stream[:, 2] == ms]
                packets.append({"events": events, "source_range": (row, row + len(events))})
                row += len(events)
        for ms, packet in zip(range(a, b), packets):
            tracker.ingest(packet["events"], watermark_ms=ms + 1, packet_ordinal=ordinal,
                           source_range=packet["source_range"])
            lo, hi = packet["source_range"]
            source_digest.update(np.arange(lo, hi, dtype=np.int64).tobytes())
            ordinal += 1
            source_count += len(packet["events"])

    with torch.inference_mode():
        cursor = start_ms
        warm_started = time.perf_counter()
        for _ in range(2):
            consume(cursor, cursor + 50); cursor += 50
            image = last_lnes(tracker.window.read(cursor), cursor - 50, cursor)
            prediction = canonical(torch.as_tensor(image[None], device=device),
                torch.as_tensor(tracker.pose[None], device=device), torch.as_tensor(reader.betas[None], device=device),
                torch.as_tensor(reader.K_cam[None], device=device))
            gpu_sync(device)
            tracker.canonical_commit(cursor, prediction[0].float().cpu().numpy())
        warmup_seconds = time.perf_counter() - warm_started
        records = []
        for index in range(queries):
            gpu_sync(device); before = time.perf_counter(); n_before = source_count
            ingest_started = time.perf_counter(); consume(cursor, cursor + hop); cursor += hop
            ingest_seconds = time.perf_counter() - ingest_started
            result = tracker.query(cursor)
            commit_started = time.perf_counter(); tracker.commit(result); gpu_sync(device)
            commit_seconds = time.perf_counter() - commit_started
            pose_service_seconds = time.perf_counter() - before
            surface, surface_costs = decode_surface(model, result["state"], reader.betas)
            wall = time.perf_counter() - before
            records.append(dict(index=index, cutoff_ms=cursor, events=source_count - n_before,
                end_to_end_seconds=wall, pose51_service_seconds=pose_service_seconds,
                ingest_seconds=ingest_seconds, commit_seconds=commit_seconds,
                **surface_costs, **result["costs"]))
        _, audit = duplicate_query_check(tracker, cursor)
    gpu_sync(device)
    resources = tracker.memory_stats()
    unchanged = {path: stream_sha(path) == identity["sha256"] for path, identity in sources.items()}
    if not all(unchanged.values()):
        raise RuntimeError("benchmark observation/config/weights/executable source changed during measurement")
    report = {"schema": "s43_relation_pipeline_benchmark_v2_surface", "arm": model.arm,
        "scope": "target-free resource benchmark only; no accuracy result and no full-recording claim",
        "input_kind": "native quantized recording prefix" if synthetic_events_per50 is None else "synthetic uniform sensor stress",
        "recording_geometry_context": recording, "synthetic_events_per50ms": synthetic_events_per50,
        "start_ms": start_ms, "cutoff_ms": cursor, "valid_run_stop_ms": stop_ms,
        "query_hop_ms": hop, "W_ms": 50, "queries": queries, "warmup_queries": 2,
        "seed": seed, "device": str(device), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "config_sha256": sha(config), "checkpoint_sha256": sha(checkpoint) if checkpoint else None,
        "benchmark_sha256": sha(__file__), "replay_sha256": sha(Path(__file__).with_name("replay.py")),
        "deployment_sha256": sha(Path(__file__).with_name("deployment.py")),
        "source_events_including_warmup": source_count, "source_id_digest": source_digest.hexdigest(),
        "load_seconds": load_seconds, "reset_and_compile_seconds": reset_seconds, "warmup_seconds": warmup_seconds,
        "source_hash_seconds_in_load": source_hash_seconds,
        "source_identities": sources, "source_files_unchanged": unchanged,
        "latency": summarize_latencies([r["end_to_end_seconds"] for r in records]),
        "pose51_latency": summarize_latencies([r["pose51_service_seconds"] for r in records]),
        "output_surface_shapes": {key:list(value.shape) for key,value in surface.items()},
        "output_surface_payload_bytes": sum(value.nbytes for value in surface.values()),
        "output_surface_units": "camera coordinates in meters; original MANO mean added once",
        "output_contract": "sensor_to_mesh778_joints21",
        "events_per_second_measured_queries": sum(r["events"] for r in records) / sum(r["end_to_end_seconds"] for r in records),
        "outputs_per_second_measured_queries": queries / sum(r["end_to_end_seconds"] for r in records),
        "records": records, "resources": resources,
        "RSS_peak_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else 0,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if torch.device(device).type == "cuda" else 0,
        "duplicate_query_check": audit, "GT_read": False, "pose_metadata_opened": False,
        "calibration_condition": "same externally supplied K/beta and valid-run metadata as accuracy protocol; aux beta origin remains conditional",
        "latency_scope": "synchronized native input load/ingest/geometry/read/transfers/network/commit/final MANO FK and surface D2H; physical sensor waiting and output serialization excluded",
        "params": model.counts()}
    tracker.release()
    dump(out, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--recording", default="ch_local")
    parser.add_argument("--hop", type=int, choices=(25, 50, 100), default=50)
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--synthetic-events-per50", type=int, choices=(10000, 50000, 150000))
    parser.add_argument("--root", default=str(DATA_ROOT))
    args = parser.parse_args()
    benchmark(args.config, args.checkpoint, args.out, device=args.device, recording=args.recording,
              hop=args.hop, queries=args.queries, threads=args.threads,
              synthetic_events_per50=args.synthetic_events_per50, root=args.root)


if __name__ == "__main__":
    main()
