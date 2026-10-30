#!/usr/bin/env python3
"""Build immutable-row XAG geometric pseudo-source caches on CPU.

Examples:
  python tools/xag/build_sources.py --estimate
  python tools/xag/build_sources.py --seq zgz_local --max-ms 200 --workers 8 --out-dir outputs/xag/source_benchmark
  python tools/xag/build_sources.py --workers 120  # resumes missing valid milliseconds

Original data are read-only. Cache arrays are sparse-created and each completed
millisecond is published only after its part/face/bary arrays have been flushed.
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
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))
sys.path.insert(0, str(REPO / "tools" / "xcl" / "sources"))

import numpy as np
import torch
from semkine.dataset import _read_meta51, sequences_for_split
from semkine.xag_sources import SOURCE_VERSION, source_maps, source_barycentrics
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


def inputs_for(root, legacy, seq, mano):
    base = root / legacy / seq
    paths = [Path(str(base) + suffix) for suffix in ("_events.npy", "_offsets.npy", "_aux.npz", ".meta")]
    optional = Path(str(base) + "_tsub.npy")
    if optional.exists():
        paths.append(optional)
    paths += [mano, root / "splits_semkine.json", Path(__file__).resolve(), REPO / "semkine" / "xag_sources.py",
              REPO / "tools" / "xcl" / "sources" / "step1_source_labels.py", REPO / "tools" / "xcl" / "sources" / "step1_common.py"]
    return {str(p.resolve()): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths}


def valid_milliseconds(aux, n_ms):
    mask = np.zeros(n_ms, bool)
    for start, end in np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2):
        mask[max(0, int(start)):min(n_ms, int(end))] = True
    return mask


def prepare(root, out, seq, legacy, mano):
    base, cache = root / legacy / seq, out / seq
    cache.mkdir(parents=True, exist_ok=True)
    lock = (cache / ".build.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    events = np.load(str(base) + "_events.npy", mmap_mode="r")
    offsets = np.load(str(base) + "_offsets.npy", mmap_mode="r")
    with np.load(str(base) + "_aux.npz", allow_pickle=True) as aux:
        wanted = valid_milliseconds(aux, len(offsets) - 1)
    identity = {"version": SOURCE_VERSION, "seq": seq, "legacy_dir": legacy,
                "n_events": len(events), "n_ms": len(offsets) - 1,
                "inputs": inputs_for(root, legacy, seq, mano),
                "contract": "preaugmentation per-event-ms GT mesh; 2px foreground contour; 10mm discontinuity; fixed face-majority part"}
    meta_path = cache / "metadata.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if any(previous.get(k) != v for k, v in identity.items()):
            raise RuntimeError(f"source/input hash mismatch; choose a new cache root: {cache}")
    else:
        specs = {"part": (np.int8, (len(events),)), "face": (np.int16, (len(events),)),
                 "bary": (np.float32, (len(events), 3)), "done_ms": (np.bool_, (len(offsets) - 1,))}
        for name, (dtype, shape) in specs.items():
            array = np.lib.format.open_memmap(cache / (name + ".npy"), mode="w+", dtype=dtype, shape=shape)
            # Sparse holes initially read as zero. Unpublished frames may not be loaded.
            if name == "done_ms":
                array[:] = False
            array.flush()
            del array
        np.save(cache / "offsets.npy", np.asarray(offsets), allow_pickle=False)
        write_json(meta_path, {**identity, "complete": False})
    done = np.load(cache / "done_ms.npy", mmap_mode="r+")
    return identity, wanted, done, lock


_WORKER = {}


def init_worker(root, out, seq_dirs, mano):
    torch.set_num_threads(1)
    model = ManoLayer(mano, add_mean=False).eval()
    _, fpart, _ = face_part_table(model)
    _WORKER.update(root=Path(root), out=Path(out), seq_dirs=seq_dirs, mano=model,
                   faces=model.f.numpy().astype(np.int64), face_part=fpart)


@lru_cache(maxsize=2)
def worker_sequence(seq):
    g = _WORKER
    base = g["root"] / g["seq_dirs"][seq] / seq
    cache = g["out"] / seq
    with np.load(str(base) + "_aux.npz", allow_pickle=True) as aux:
        betas = torch.tensor(aux["betas"], dtype=torch.float32).reshape(1, -1)
        fxyc = tuple(float(x) for x in render_intrinsics_f32(aux["camera_K"]))
    return {"events": np.load(str(base) + "_events.npy", mmap_mode="r"),
            "offsets": np.load(str(base) + "_offsets.npy", mmap_mode="r"),
            "pos51": _read_meta51(str(base) + ".meta"), "betas": betas, "fxyc": fxyc,
            **{key: np.load(cache / (key + ".npy"), mmap_mode="r+") for key in ("part", "face", "bary")}}


def process_chunk(task):
    seq, milliseconds = task
    start_time = time.monotonic()
    a, g = worker_sequence(seq), _WORKER
    ms = np.asarray(milliseconds, np.int64)
    nonempty = ms[a["offsets"][ms + 1] > a["offsets"][ms]]
    n_events = n_valid = 0
    if len(nonempty):
        for frame in nonempty:
            # FK batch shape changes float32 reduction order. Pin it to one
            # millisecond so workers/chunk sizes/restarts cannot change labels.
            vertices, _ = fk_batch(g["mano"], a["betas"], a["pos51"][frame:frame + 1])
            verts = vertices[0].numpy().astype(np.float64)
            Z, P, FID = rasterize(verts, g["faces"], g["face_part"], a["fxyc"])
            maps = source_maps(Z, P, FID)
            lo, hi = int(a["offsets"][frame]), int(a["offsets"][frame + 1])
            ev = a["events"][lo:hi]
            x, y = ev[:, 0].astype(np.int64), ev[:, 1].astype(np.int64)
            part, face, pixel = maps["part"][y, x], maps["face"][y, x], maps["pixel"][y, x]
            bary = source_barycentrics(verts, g["faces"], a["fxyc"], face, pixel)
            a["part"][lo:hi], a["face"][lo:hi], a["bary"][lo:hi] = part, face, bary
            n_events += hi - lo
            n_valid += int((face >= 0).sum())
        for key in ("part", "face", "bary"):
            a[key].flush()
    return seq, milliseconds, n_events, n_valid, time.monotonic() - start_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO / "data" / "hand_data51")
    parser.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "xag" / "sources")
    parser.add_argument("--mano", type=Path, default=REPO / "assets" / "mano_right.npz")
    parser.add_argument("--seq", nargs="+", help="default: all 72 train + 2 zgz")
    parser.add_argument("--workers", type=int, default=min(120, os.cpu_count() or 1))
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--max-ms", type=int, help="first N requested valid milliseconds per sequence (benchmark)")
    parser.add_argument("--start-ms", type=int, default=0)
    parser.add_argument("--estimate", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_ms < 1 or args.start_ms < 0 or (args.max_ms is not None and args.max_ms < 1):
        parser.error("workers/chunk-ms/max-ms must be positive and start-ms nonnegative")
    seq_dirs = dict(sequences_for_split(args.data_root, "train", None) + sequences_for_split(args.data_root, "val", None))
    if args.seq:
        missing = set(args.seq) - set(seq_dirs)
        if missing:
            parser.error(f"sequences absent from the fixed manifest: {sorted(missing)}")
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
                          "cache_bytes": count * 15, "cache_GiB": count * 15 / 2 ** 30}, indent=2))
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    overall_start = time.monotonic()
    states, locks, tasks = {}, [], []
    for seq, legacy in seq_dirs.items():
        identity, wanted, done, lock = prepare(args.data_root, args.out_dir, seq, legacy, args.mano)
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
    cpu_seconds = 0.0
    if tasks:
        with mp.get_context("fork").Pool(args.workers, initializer=init_worker,
             initargs=(str(args.data_root), str(args.out_dir), seq_dirs, str(args.mano))) as pool:
            for i, (seq, milliseconds, ne, nv, seconds) in enumerate(pool.imap_unordered(process_chunk, tasks, chunksize=1), 1):
                done = states[seq][2]
                done[milliseconds] = True
                done.flush()
                n_ms += len(milliseconds)
                n_events += ne
                n_valid += nv
                cpu_seconds += seconds
                if i % max(1, len(tasks) // 20) == 0 or i == len(tasks):
                    print(json.dumps({"chunks_done": i, "chunks_total": len(tasks), "ms_done": n_ms,
                                      "events_done": n_events, "elapsed_s": time.monotonic() - run_start}), flush=True)
    completed = {}
    for seq, (identity, wanted, done) in states.items():
        completed[seq] = bool(np.all(done[wanted]))
        write_json(args.out_dir / seq / "metadata.json", {**identity, "complete": completed[seq],
                   "completed_valid_ms": int(done[wanted].sum()), "total_valid_ms": int(wanted.sum())})
    summary = {"version": SOURCE_VERSION, "workers": args.workers, "chunk_ms": args.chunk_ms,
               "processed_ms": n_ms, "processed_events": n_events, "valid_sources": n_valid,
               "worker_seconds": cpu_seconds, "compute_wall_seconds": time.monotonic() - run_start,
               "total_wall_seconds": time.monotonic() - overall_start, "complete": completed}
    write_json(args.out_dir / "build_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
