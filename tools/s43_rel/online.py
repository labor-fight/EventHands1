"""Standalone label-free finite event-stream client for the S43 relation model.

Inputs are only an mmap event array, external calibration, configuration and
weights.  No dataset reader, valid-run metadata, target or ground-truth shape
is instantiated or accessed.  Output JSONL is an explicitly O(T) artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

from tools.s43_integration.model import load_legacy
from .core import RelConfig
from .geometry import GeometryConfig, ASSET
from .model import load_relation_model
from .replay import CausalRelationTracker, gpu_sync, sha, validate_device
from .deployment import decode_surface


PUBLIC_INITIAL = np.array([0., 0., .6] + [0.] * 48, np.float32)


class MMapEventStream:
    """Native-ms groups without an O(T) converted timestamp array.

For us input, only comparisons with whole-ms boundaries are used.  The
validation requires nondecreasing *quantized bins*, not a physical order of
sub-ms rows.  Consequently each boundary partitions the raw timestamp column
even when times inside one bin are permuted; boundary searchsorted is valid.
    """

    def __init__(self, path, time_unit="ms", start_ms=None, stop_ms=None, validation_chunk=65536):
        self.path = Path(path)
        self.events = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if self.events.ndim != 2 or self.events.shape[1] != 4 or self.events.dtype.kind not in "iu":
            raise ValueError("events must be an integer N-by-4 .npy array [x,y,t,p]")
        if not self.events.dtype.isnative:
            raise ValueError("native-endian event dtype required for mmap timestamp search without whole-array conversion")
        if time_unit not in ("ms", "us"):
            raise ValueError("time_unit must be ms or us")
        self.time_unit, self.scale = time_unit, 1 if time_unit == "ms" else 1000
        self.timestamps = self.events[:, 2]  # mmap strided view, not a converted copy
        previous = None
        for begin in range(0, len(self.events), validation_chunk):
            raw = np.asarray(self.timestamps[begin:begin + validation_chunk])
            if len(raw) and (raw.min() < 0 or raw.max() > np.iinfo(np.int64).max):
                raise ValueError("timestamps must be nonnegative signed-int64 values")
            bins = raw // self.scale  # bounded validation scratch only
            if len(bins) and ((previous is not None and bins[0] < previous) or np.any(bins[1:] < bins[:-1])):
                raise ValueError("out-of-order native-ms event groups; no sorting or dropping is performed")
            if len(bins):
                previous = int(bins[-1])
        if not len(self.events) and (start_ms is None or stop_ms is None):
            raise ValueError("an empty stream requires explicit --start-ms and --stop-ms")
        first = int(self.timestamps[0]) // self.scale if len(self.events) else int(start_ms)
        last = int(self.timestamps[-1]) // self.scale if len(self.events) else int(stop_ms) - 1
        # Endpoint rows can be reordered within the same native bin; their
        # quantized times still give the first and final group exactly.
        self.start_ms = first if start_ms is None else int(start_ms)
        self.stop_ms = last + 1 if stop_ms is None else int(stop_ms)
        if self.start_ms < 0 or self.stop_ms < self.start_ms or self.start_ms > first or self.stop_ms <= last:
            raise ValueError("declared half-open coverage must include every input event")
        self.cursor_ms, self.source_end = self.start_ms, 0

    def packet(self, ms):
        if ms != self.cursor_ms or ms >= self.stop_ms:
            raise ValueError("consume native millisecond packets exactly once in order")
        lo = int(np.searchsorted(self.timestamps, int(ms) * self.scale, side="left"))
        hi = int(np.searchsorted(self.timestamps, (int(ms) + 1) * self.scale, side="left"))
        if lo != self.source_end:
            raise RuntimeError("source search would skip or repeat input event rows")
        packet = np.array(self.events[lo:hi], dtype=np.int64, copy=True)
        if len(packet):
            # Never sort or retain an interpretation of within-bin timestamps.
            packet[:, 2] //= self.scale
            if np.any(packet[:, 2] != ms):
                raise ValueError("timestamp group is not a complete native-ms prefix")
            if (np.any(packet[:, 0] < 0) or np.any(packet[:, 0] >= 240)
                    or np.any(packet[:, 1] < 0) or np.any(packet[:, 1] >= 180)
                    or np.any((packet[:, 3] != 0) & (packet[:, 3] != 1))):
                raise ValueError("event x/y must be in 240x180 and stored polarity must be 0/1")
        self.cursor_ms, self.source_end = int(ms) + 1, hi
        return packet, (lo, hi)


def load_calibration(path):
    raw = json.loads(Path(path).read_text())
    if "K_cam" not in raw:
        raise ValueError("calibration must supply K_cam in the original640x480 grid")
    K = np.asarray(raw["K_cam"], np.float32)
    beta = np.asarray(raw.get("betas", [0.] * 10), np.float32)
    initial = np.asarray(raw.get("initial_state", PUBLIC_INITIAL), np.float32)
    if K.shape != (3, 3) or beta.shape != (10,) or initial.shape != (51,):
        raise ValueError("K_cam[3,3], optional betas[10], optional initial_state[51] required")
    if not all(np.isfinite(value).all() for value in (K, beta, initial)):
        raise ValueError("finite calibration and initial state required")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0., 0., 1.]):
        raise ValueError("K_cam must have positive focal lengths and standard final row")
    return K, beta, initial, {"K_source_grid": [640, 480], "K_event_scale": .375,
        "beta_source": "external calibration" if "betas" in raw else "explicit mean shape, all10betas0",
        "initial_state_source": "external calibration" if "initial_state" in raw else "public constant",
        "GT_shape_required": False}


def run(events, calibration, config, checkpoint, out, *, time_unit="ms", hop=50, device="cpu",
        threads=2, start_ms=None, stop_ms=None, max_window_events=262144):
    validate_device(device)
    if hop not in (25, 50, 100):
        raise ValueError("hop must be25/50/100ms")
    out = Path(out)
    jsonl = out if out.suffix == ".jsonl" else Path(str(out) + ".jsonl")
    summary_path = jsonl.with_suffix(".summary.json")
    if jsonl.exists() or summary_path.exists():
        raise FileExistsError("preserve existing client output")
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(threads))
    started = time.perf_counter()
    stream = MMapEventStream(events, time_unit, start_ms, stop_ms)
    K, beta, initial, conditions = load_calibration(calibration)
    model, cfg = load_relation_model(config, checkpoint, device)
    canonical, _ = load_legacy(cfg["REL"]["BASE_CHECKPOINT"], cfg["REL"]["BASE_CONFIG"], device)
    model.eval(); canonical.eval()
    tracker = CausalRelationTracker(model, RelConfig(**cfg.get("MEMORY", {})),
                                    GeometryConfig(**cfg.get("GEOMETRY", {})), max_window_events)
    sources = {str(Path(path).resolve()): {"sha256": sha(path), "bytes": Path(path).stat().st_size}
               for path in (events, calibration, config, checkpoint, __file__, ASSET, Path(__file__).with_name("deployment.py"),
                            Path(__file__).with_name("replay.py"), cfg["REL"]["BASE_CHECKPOINT"], cfg["REL"]["BASE_CONFIG"])}
    from .training import _source_files
    for path, identity in _source_files().items():
        sources[path] = {"sha256": identity, "bytes": Path(path).stat().st_size}
    loaded_seconds = time.perf_counter() - started
    tracker.reset(str(Path(events).resolve()), stream.start_ms, initial, beta, K)
    ordinal = native_packets = source_count = outputs = warmups = 0
    source_digest = hashlib.sha256()
    peak_raw_bytes = 0
    resource_peak = {"resident_bytes": 0, "raw_window_bytes": 0}
    services_sum = services_max = mesh_services_sum = 0.
    stream_started = time.perf_counter()

    def consume(stop):
        nonlocal ordinal, native_packets, source_count, peak_raw_bytes, resource_peak
        while stream.cursor_ms < stop:
            ms = stream.cursor_ms
            packet, source_range = stream.packet(ms)
            tracker.ingest(packet, watermark_ms=ms + 1, packet_ordinal=ordinal, source_range=source_range)
            lo, hi = source_range
            source_digest.update(np.arange(lo, hi, dtype=np.int64).tobytes())
            source_count += hi - lo
            ordinal += 1; native_packets += 1
            memory = tracker.memory_stats()
            peak_raw_bytes = max(peak_raw_bytes, memory["raw_window_bytes"])
            for key in ("resident_bytes", "raw_window_bytes"):
                resource_peak[key] = max(resource_peak[key], memory[key])

    try:
        with jsonl.open("x") as output, torch.inference_mode():
            warm_started = time.perf_counter()
            for _ in range(2):
                if stream.stop_ms - stream.cursor_ms < 50:
                    break
                consume(stream.cursor_ms + 50)
                cutoff = stream.cursor_ms
                raw = tracker.window.read(cutoff)
                # Exactly the same original W50 builder used by the tracker.
                from .data import last_lnes
                image = last_lnes(raw, cutoff - 50, cutoff)
                prediction = canonical(torch.as_tensor(image[None], device=device),
                    torch.as_tensor(tracker.pose[None], device=device), torch.as_tensor(beta[None], device=device),
                    torch.as_tensor(K[None], device=device))
                gpu_sync(device)
                tracker.canonical_commit(cutoff, prediction[0].float().cpu().numpy())
                warmups += 1
            warmup_seconds = time.perf_counter() - warm_started
            if warmups == 2:
                while stream.stop_ms - stream.cursor_ms >= hop:
                    gpu_sync(device); service_started = time.perf_counter(); before_count = source_count
                    consume(stream.cursor_ms + hop)
                    cutoff = stream.cursor_ms
                    result = tracker.query(cutoff)
                    tracker.commit(result); gpu_sync(device)
                    surface, surface_costs = decode_surface(model, result["state"], beta)
                    service = time.perf_counter() - service_started
                    services_sum += service; services_max = max(services_max, service)
                    mesh_services_sum += surface_costs["output_mesh_total_seconds"]
                    row = {"output_index": outputs, "evidence_cutoff_ms": cutoff, "t_query_ms": cutoff - 1,
                        "pose51": result["state"].tolist(), "pose_version": tracker.pose_version,
                        **{key:value.tolist() for key,value in surface.items()},
                        "output_surface_units": "camera coordinates in meters",
                        "observation_version": tracker.observation_version, "commit": True,
                        "source_events_total": source_count, "new_source_events": source_count - before_count,
                        "service_seconds": service, **surface_costs}
                    output.write(json.dumps(row) + "\n"); output.flush()
                    outputs += 1
            tail_ms = stream.stop_ms - stream.cursor_ms
            tail_count = source_count
            tail_started = time.perf_counter()
            consume(stream.stop_ms)
            tail_seconds = time.perf_counter() - tail_started
            tail_events = source_count - tail_count
        gpu_sync(device)
        stream_seconds = time.perf_counter() - stream_started
        if source_count != len(stream.events) or stream.source_end != len(stream.events):
            raise AssertionError("the client did not consume every external event exactly once")
        expected = hashlib.sha256()
        for row in range(0, len(stream.events), 65536):
            expected.update(np.arange(row, min(row + 65536, len(stream.events)), dtype=np.int64).tobytes())
        if source_digest.hexdigest() != expected.hexdigest():
            raise AssertionError("source row digest mismatch")
        tracker.release()
        unchanged = {path: sha(path) == identity["sha256"] for path, identity in sources.items()}
        if not all(unchanged.values()):
            raise RuntimeError("external source, calibration, configuration, weights or client source changed")
        summary = {"schema": "s43_relation_external_event_client_v1", "arm": model.arm, "reader": model.reader,
            "device": str(device), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "events": str(Path(events).resolve()), "calibration": str(Path(calibration).resolve()),
            "input_columns": ["x", "y", "t", "stored_polarity_0or1"], "input_time_unit": time_unit,
            "timestamp_contract": "integer-ms groups; us floored only in bounded packets; no sub-ms order used",
            "coverage_ms": [stream.start_ms, stream.stop_ms], "hop_ms": hop,
            "observation_window_ms": 50, "output_lag_ms": 1, "warmup_canonical_windows": warmups,
            "source_event_count": source_count, "expected_source_event_count": len(stream.events),
            "native_ms_packets": native_packets, "source_id_digest": source_digest.hexdigest(),
            "all_external_events_consumed_once": True, "tail_ms": tail_ms, "tail_events": tail_events,
            "outputs": outputs, "conditions": conditions, "GT_or_target_files_read": False,
            "mean_shape_scope": "beta0 client operability only; no pose-accuracy or shape-robustness claim",
            "stream_seconds": stream_seconds, "load_validate_hash_seconds": loaded_seconds,
            "warmup_seconds": warmup_seconds, "tail_seconds": tail_seconds,
            "service_mean_seconds": services_sum / outputs if outputs else None,
            "service_max_seconds": services_max if outputs else None,
            "output_mesh_mean_seconds": mesh_services_sum / outputs if outputs else None,
            "output_shapes": {"pose51":[51],"vertices_m":[778,3],"joints_m":[21,3]},
            "output_surface_payload_bytes": (778 + 21) * 3 * 4,
            "service_scope": "synchronized ingestion/query/commit/final MANO FK/surface transfer; excludes physical sensor waiting and output serialization/I/O",
            "max_window_events": max_window_events, "raw_window_payload_byte_limit": max_window_events * 32,
            "peak_raw_window_bytes": peak_raw_bytes, "memory_peaks": resource_peak,
            "output_log": str(jsonl), "output_log_bytes": jsonl.stat().st_size,
            "output_log_storage": "O(number_of_outputs); not resident representation memory",
            "timestamp_scratch": "at most65536timestamps during validation plus one native-ms packet; no O(T) timestamp copy",
            "source_identities": sources, "source_files_unchanged": unchanged}
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    except BaseException as error:
        failure = {"schema": "s43_relation_external_event_client_failed_v1", "error": repr(error),
            "source_events_successfully_consumed": source_count, "outputs": outputs,
            "current_watermark_ms": tracker.watermark_ms, "source_row_cursor": stream.source_end,
            "reset_required_to_retry_full_file": True, "GT_or_target_files_read": False}
        summary_path.write_text(json.dumps(failure, indent=2) + "\n")
        if tracker.active:
            tracker.release()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True)
    parser.add_argument("--time-unit", choices=("ms", "us"), default="ms")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hop", type=int, choices=(25, 50, 100), default=50)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--start-ms", type=int)
    parser.add_argument("--stop-ms", type=int)
    parser.add_argument("--max-window-events", type=int, default=262144)
    args = parser.parse_args()
    run(args.events, args.calibration, args.config, args.checkpoint, args.out, time_unit=args.time_unit,
        hop=args.hop, device=args.device, threads=args.threads, start_ms=args.start_ms,
        stop_ms=args.stop_ms, max_window_events=args.max_window_events)


if __name__ == "__main__":
    main()
