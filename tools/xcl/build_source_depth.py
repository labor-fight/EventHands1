#!/usr/bin/env python3
"""CPU-only XCL foreground depth sidecar; never writes an existing source array.

The per-ms FK/raster/source operation is the original frozen source operation.
Every newly computed part, face and barycentric row must exactly equal the old
cache before depth for that millisecond may be published. Unknown depth is NaN.
Partial smoke builds stay explicitly incomplete and cannot be used by A3.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from functools import lru_cache

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "sources"))

import numpy as np
import torch
from semkine.dataset import _read_meta51, sequences_for_split
from semkine.xag_sources import DEPTH_VERSION, SOURCE_VERSION, source_maps, source_barycentrics
from step1_source_labels import rasterize
from step1_common import face_part_table, fk_batch, render_intrinsics_f32
from mano_layer import ManoLayer


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def valid_milliseconds(aux, n_ms):
    mask = np.zeros(n_ms, bool)
    for start, end in np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2):
        mask[max(0, int(start)):min(n_ms, int(end))] = True
    return mask


def prepare(root, source_root, seq, legacy, mano):
    base, cache = root / legacy / seq, source_root / seq
    # Refuse to create any legacy source directories or modify their files.
    source_meta_path = cache / "metadata.json"
    source = json.loads(source_meta_path.read_text())
    if source.get("version") != SOURCE_VERSION or source.get("seq") != seq or not source.get("complete"):
        raise RuntimeError(f"complete immutable source cache required: {cache}")
    lock = (cache / ".depth_build.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    events = np.load(str(base) + "_events.npy", mmap_mode="r", allow_pickle=False)
    offsets = np.load(str(base) + "_offsets.npy", mmap_mode="r", allow_pickle=False)
    if source["n_events"] != len(events) or source["n_ms"] != len(offsets) - 1:
        raise ValueError(f"data/source dimensions differ: {seq}")
    if not np.array_equal(offsets, np.load(cache / "offsets.npy", mmap_mode="r", allow_pickle=False)):
        raise ValueError(f"data/source offsets differ: {seq}")
    with np.load(str(base) + "_aux.npz", allow_pickle=True) as aux:
        wanted = valid_milliseconds(aux, len(offsets) - 1)
    if not np.all(np.load(cache / "done_ms.npy", mmap_mode="r", allow_pickle=False)[wanted]):
        raise RuntimeError(f"source completion bitmap is incomplete: {seq}")
    paths = [Path(str(base) + suffix) for suffix in ("_events.npy", "_offsets.npy", "_aux.npz", ".meta")]
    tsub = Path(str(base) + "_tsub.npy")
    if tsub.exists():
        paths.append(tsub)
    paths += [mano, root / "splits_semkine.json"]
    inputs = {}
    for path in paths:
        resolved = str(path.resolve())
        spec = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if source["inputs"].get(resolved) != spec:
            raise ValueError(f"source input hash differs: {path}")
        inputs[resolved] = spec
    helpers = [Path(__file__), REPO / "tools" / "xcl" / "sources" / "step1_common.py",
               REPO / "tools" / "xcl" / "sources" / "step1_source_labels.py"]
    identity = {"version": DEPTH_VERSION, "source_version": SOURCE_VERSION, "seq": seq,
                "legacy_dir": legacy, "n_events": len(events), "n_ms": len(offsets) - 1,
                "source_metadata_sha256": sha256(source_meta_path), "inputs": inputs,
                "code_sha256": {str(p.relative_to(REPO)): sha256(p) for p in helpers},
                "contract": "float32 camera z at existing foreground source pixel; GT own-ms FK1; exact part/face/bary parity; unknown NaN; focal augmentation"}
    meta_path = cache / "depth_metadata.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if any(previous.get(k) != v for k, v in identity.items()):
            raise RuntimeError(f"sidecar/input identity mismatch; preserve and choose a new cache root: {cache}")
    else:
        for name, dtype, shape in (("depth", np.float32, (len(events),)),
                                   ("depth_done_ms", np.bool_, (len(offsets) - 1,))):
            destination = cache / (name + ".npy")
            if destination.exists():
                raise RuntimeError(f"unpublished sidecar exists, preserve it: {destination}")
            array = np.lib.format.open_memmap(destination, mode="w+", dtype=dtype, shape=shape)
            # Noncompleted milliseconds are unreadable; sparse zero pages are safe.
            if name == "depth_done_ms":
                array[:] = False
            array.flush()
            del array
        write_json(meta_path, {**identity, "complete": False})
    done = np.load(cache / "depth_done_ms.npy", mmap_mode="r+", allow_pickle=False)
    depth = np.load(cache / "depth.npy", mmap_mode="r", allow_pickle=False)
    if depth.dtype != np.float32 or depth.shape != (len(events),) or done.dtype != np.bool_ or done.shape != wanted.shape:
        raise ValueError(f"sidecar shape/dtype differs: {cache}")
    return identity, wanted, done, lock


_WORKER = {}


def init_worker(root, source_root, seq_dirs, mano):
    torch.set_num_threads(1)
    model = ManoLayer(mano, add_mean=False).eval()
    _, fpart, _ = face_part_table(model)
    _WORKER.update(root=Path(root), source_root=Path(source_root), seq_dirs=seq_dirs, mano=model,
                   faces=model.f.numpy().astype(np.int64), face_part=fpart)


@lru_cache(maxsize=2)
def worker_sequence(seq):
    g = _WORKER
    base = g["root"] / g["seq_dirs"][seq] / seq
    cache = g["source_root"] / seq
    with np.load(str(base) + "_aux.npz", allow_pickle=True) as aux:
        betas = torch.tensor(aux["betas"], dtype=torch.float32).reshape(1, -1)
        fxyc = tuple(float(x) for x in render_intrinsics_f32(aux["camera_K"]))
    return {"events": np.load(str(base) + "_events.npy", mmap_mode="r", allow_pickle=False),
            "offsets": np.load(str(base) + "_offsets.npy", mmap_mode="r", allow_pickle=False),
            "pos51": _read_meta51(str(base) + ".meta"), "betas": betas, "fxyc": fxyc,
            **{key: np.load(cache / (key + ".npy"), mmap_mode="r", allow_pickle=False)
               for key in ("part", "face", "bary")},
            "depth": np.load(cache / "depth.npy", mmap_mode="r+", allow_pickle=False)}


def process_chunk(task):
    seq, milliseconds = task
    start_time = time.monotonic()
    a, g = worker_sequence(seq), _WORKER
    n_events = n_valid = 0
    for frame in milliseconds:
        lo, hi = int(a["offsets"][frame]), int(a["offsets"][frame + 1])
        if hi == lo:
            continue
        vertices, _ = fk_batch(g["mano"], a["betas"], a["pos51"][frame:frame + 1])
        verts = vertices[0].numpy().astype(np.float64)
        Z, P, FID = rasterize(verts, g["faces"], g["face_part"], a["fxyc"])
        maps = source_maps(Z, P, FID)
        ev = a["events"][lo:hi]
        x, y = ev[:, 0].astype(np.int64), ev[:, 1].astype(np.int64)
        part, face, pixel = maps["part"][y, x], maps["face"][y, x], maps["pixel"][y, x]
        bary = source_barycentrics(verts, g["faces"], a["fxyc"], face, pixel)
        for key, current in (("part", part), ("face", face), ("bary", bary)):
            if not np.array_equal(a[key][lo:hi], current):
                raise ValueError(f"legacy source parity failure: {seq} ms={frame} field={key}")
        depth = maps["depth"][y, x]
        known = (part >= 0) & (face >= 0)
        if not np.all(np.isfinite(depth[known]) & (depth[known] > 0)) or not np.isnan(depth[~known]).all():
            raise ValueError(f"foreground depth validity differs: {seq} ms={frame}")
        a["depth"][lo:hi] = depth
        n_events += hi - lo
        n_valid += int(known.sum())
    a["depth"].flush()
    return seq, milliseconds, n_events, n_valid, time.monotonic() - start_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO / "data" / "hand_data51")
    parser.add_argument("--source-root", type=Path, default=REPO / "outputs" / "xag" / "sources")
    parser.add_argument("--mano", type=Path, default=REPO / "assets" / "mano_right.npz")
    parser.add_argument("--seq", nargs="+", help="default: all 72 train + 2 zgz")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--max-ms", type=int, help="partial smoke: first N valid milliseconds per sequence")
    parser.add_argument("--start-ms", type=int, default=0)
    parser.add_argument("--summary", type=Path, default=REPO / "outputs" / "xcl" / "20261019" / "depth_build_summary.json")
    parser.add_argument("--estimate", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_ms < 1 or args.start_ms < 0 or (args.max_ms is not None and args.max_ms < 1):
        parser.error("workers/chunk-ms/max-ms must be positive and start-ms nonnegative")
    seq_dirs = dict(sequences_for_split(args.data_root, "train", None) + sequences_for_split(args.data_root, "val", None))
    if args.seq:
        missing = set(args.seq) - set(seq_dirs)
        if missing:
            parser.error(f"sequences absent from fixed manifest: {sorted(missing)}")
        seq_dirs = {seq: seq_dirs[seq] for seq in args.seq}
    if args.estimate:
        count = valid_ms = 0
        for seq, legacy in seq_dirs.items():
            base = args.data_root / legacy / seq
            offsets = np.load(str(base) + "_offsets.npy", mmap_mode="r")
            count += int(offsets[-1])
            with np.load(str(base) + "_aux.npz", allow_pickle=True) as aux:
                valid_ms += int(valid_milliseconds(aux, len(offsets) - 1).sum())
        print(json.dumps({"sequences": len(seq_dirs), "events": count, "valid_ms": valid_ms,
                          "sidecar_GiB": (count * 4 + valid_ms) / 2 ** 30}, indent=2))
        return
    overall_start = time.monotonic()
    states, locks, tasks = {}, [], []
    for seq, legacy in seq_dirs.items():
        identity, wanted, done, lock = prepare(args.data_root, args.source_root, seq, legacy, args.mano)
        locks.append(lock)
        eligible = np.flatnonzero(wanted & (np.arange(len(wanted)) >= args.start_ms))
        if args.max_ms is not None:
            eligible = eligible[:args.max_ms]
        todo = eligible[~done[eligible]]
        states[seq] = identity, wanted, done
        tasks += [(seq, todo[i:i + args.chunk_ms].tolist()) for i in range(0, len(todo), args.chunk_ms)]
        print(json.dumps({"seq": seq, "requested_ms": len(eligible), "pending_ms": len(todo),
                          "events": identity["n_events"]}), flush=True)
    run_start = time.monotonic()
    n_ms = n_events = n_valid = 0
    worker_seconds = 0.0
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    if tasks:
        with mp.get_context("fork").Pool(args.workers, initializer=init_worker,
             initargs=(str(args.data_root), str(args.source_root), seq_dirs, str(args.mano))) as pool:
            for i, (seq, milliseconds, ne, nv, seconds) in enumerate(pool.imap_unordered(process_chunk, tasks, chunksize=1), 1):
                done = states[seq][2]
                done[milliseconds] = True
                done.flush()
                n_ms += len(milliseconds)
                n_events += ne
                n_valid += nv
                worker_seconds += seconds
                if i % max(1, min(100, len(tasks) // 20)) == 0 or i == len(tasks):
                    progress = {"chunks_done": i, "chunks_total": len(tasks), "ms_done": n_ms,
                                "events_done": n_events, "elapsed_s": time.monotonic() - run_start}
                    print(json.dumps(progress), flush=True)
                    write_json(args.summary.with_name(args.summary.stem + "_progress.json"), progress)
    completed = {}
    for seq, (identity, wanted, done) in states.items():
        completed[seq] = bool(np.all(done[wanted]))
        publish = {**identity, "complete": completed[seq], "completed_valid_ms": int(done[wanted].sum()),
                   "total_valid_ms": int(wanted.sum())}
        if completed[seq]:
            publish["outputs_sha256"] = {name: sha256(args.source_root / seq / name)
                                         for name in ("depth.npy", "depth_done_ms.npy")}
        write_json(args.source_root / seq / "depth_metadata.json", publish)
    summary = {"version": DEPTH_VERSION, "workers": args.workers, "chunk_ms": args.chunk_ms,
               "processed_ms": n_ms, "processed_events": n_events, "valid_sources": n_valid,
               "worker_seconds": worker_seconds, "compute_wall_seconds": time.monotonic() - run_start,
               "total_wall_seconds": time.monotonic() - overall_start, "complete": completed}
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
