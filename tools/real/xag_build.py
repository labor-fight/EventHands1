#!/usr/bin/env python3
"""XAG per-event geometric pseudo-sources of the real data (docs/EXPERIMENT_RECORDS.md#real-amend5-20261111), CPU only.

EvHandTrack's router (r_s2d, the ESMNet surface router) is trained on per-event surface labels: for every event the GT
MANO mesh of the event's own millisecond is z-buffered and the event takes the part / face / barycentrics of the nearest
foreground contour within 2 px (else the hand pixel under it, else no label). On the simulated data these caches were
built by tools/xag/build_sources.py (archived 2026-10-19 under outputs/deleted_20261019/xag_xa_legacy/files/tools/xag/,
with its helpers tools/xa/step1_source_labels.py `rasterize` and tools/xa/step1_common.py). This file is that builder
with its helpers inlined, unchanged except the frame: the real canvas n346 (346 x 260, render scale 346 / 640 on the
640 x 480 virtual-pinhole K) instead of 240 x 180 / 0.375, the real data root and its split manifest (splits_real.json),
and the output root. The label rules (semkine/xag_sources.py `source_maps` / `source_barycentrics`, SOURCE_VERSION) and
the cache layout read by semkine/xag_sources.py `load_sources` are the same.

    python tools/real/xag_build.py --estimate
    python tools/real/xag_build.py --workers 100            # all training sequences; resumes missing milliseconds
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
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mano_layer import ManoLayer  # noqa: E402
from pose_repr import decode_to_mano_inputs  # noqa: E402
from semkine.dataset import _read_meta51, sequences_for_split  # noqa: E402
from semkine.xag_sources import SOURCE_VERSION, source_barycentrics, source_maps  # noqa: E402

if os.environ.get("ERH_NAMESPACE") == "1":                          # EvRealHands (tools/erh/ns.py, tools/erh/run.py)
    from tools.erh import ns as _erh  # noqa: E402
    _erh.install()
from tools.real import ns  # noqa: E402  (tools/erh/ns.py re-targets it to EvRealHands before this import)

DATA_ROOT = ns.CANVAS["n346"]["root"]
SPLITS = ns.splits("n346")
OUT_DIR = ns.OUT / "xag_sources"
H, W = ns.CANVAS["n346"]["H"], ns.CANVAS["n346"]["W"]
RENDER_SCALE = W / 640.0
BARY_EPS = 1e-5
ARCHIVED = REPO / "outputs/deleted_20261019/xag_xa_legacy/files/tools"


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


# ------------------------------------------------------------------ tools/xa/step1_common.py (frame constants above)
def fk_batch(mano, betas, params51):
    with torch.no_grad():
        p = torch.from_numpy(np.ascontiguousarray(params51, dtype=np.float32))
        dec = decode_to_mano_inputs(p, "mano_full_axis_angle", mano.hands_components, mano.hands_mean)
        return mano(betas.expand(len(p), -1), dec["global_orient"], dec["local_full_aa"], dec["transl"])


def render_intrinsics_f32(camera_K):
    K = torch.tensor(np.asarray(camera_K), dtype=torch.float32).view(1, 3, 3)
    s = RENDER_SCALE
    return K[:, 0, 0] * s, K[:, 1, 1] * s, K[:, 0, 2] * s, K[:, 1, 2] * s


def face_part_table(mano):
    w = mano.weights.numpy().astype(np.float64)
    hard = w.argmax(1)
    f = mano.f.numpy()
    hj = hard[f]
    a, b, c = hj[:, 0], hj[:, 1], hj[:, 2]
    maj = np.where(a == b, a, np.where(a == c, a, np.where(b == c, b, -1)))
    tie = np.where(maj < 0)[0]
    for i in tie:
        cand = np.unique(hj[i])
        score = [w[f[i], j].sum() for j in cand]
        maj[i] = cand[int(np.argmax(score))]
    return hard, maj.astype(np.int64), int(len(tie))


# ------------------------------------------------------------------ tools/xa/step1_source_labels.py `rasterize`
def rasterize(verts, faces, face_part, fxyc, eps=BARY_EPS):
    fx, fy, cx, cy = fxyc
    z = verts[:, 2]
    okv = z > 1e-6
    zs = np.where(okv, z, 1.0)
    u = fx * verts[:, 0] / zs + cx
    v = fy * verts[:, 1] / zs + cy
    fu, fv, fz = u[faces], v[faces], z[faces]
    x0, x1, x2 = fu[:, 0], fu[:, 1], fu[:, 2]
    y0, y1, y2 = fv[:, 0], fv[:, 1], fv[:, 2]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    ok = okv[faces].all(1) & (np.abs(area) > 1e-9)
    ok &= (fu.max(1) >= 0) & (fu.min(1) <= W - 1) & (fv.max(1) >= 0) & (fv.min(1) <= H - 1)
    xl = np.clip(np.floor(fu.min(1)), 0, W - 1).astype(np.int64)
    xh = np.clip(np.ceil(fu.max(1)), 0, W - 1).astype(np.int64)
    yl = np.clip(np.floor(fv.min(1)), 0, H - 1).astype(np.int64)
    yh = np.clip(np.ceil(fv.max(1)), 0, H - 1).astype(np.int64)
    nw, nh = xh - xl + 1, yh - yl + 1
    n = np.where(ok, nw * nh, 0)
    total = int(n.sum())
    Z = np.full(H * W, np.inf)
    FID = np.full(H * W, -1, np.int32)
    if total == 0:
        return Z.reshape(H, W), np.full((H, W), -1, np.int8), FID.reshape(H, W)
    assert total < 30_000_000, total
    fid = np.repeat(np.arange(len(faces)), n)
    k = np.arange(total) - np.repeat(np.cumsum(n) - n, n)
    px = xl[fid] + k % nw[fid]
    py = yl[fid] + k // nw[fid]
    A = area[fid]
    dxp, dyp = px - x0[fid], py - y0[fid]
    b1 = (dxp * (y2 - y0)[fid] - (x2 - x0)[fid] * dyp) / A
    b2 = ((x1 - x0)[fid] * dyp - dxp * (y1 - y0)[fid]) / A
    b0 = 1.0 - b1 - b2
    inside = (b0 >= -eps) & (b1 >= -eps) & (b2 >= -eps)
    fid, px, py, b0, b1, b2 = fid[inside], px[inside], py[inside], b0[inside], b1[inside], b2[inside]
    zz = 1.0 / (b0 / fz[fid, 0] + b1 / fz[fid, 1] + b2 / fz[fid, 2])
    pix = py * W + px
    key = pix.astype(np.float64) + zz / 16.0
    order = np.argsort(key, kind="stable")
    ps = pix[order]
    first = np.ones(len(ps), bool)
    first[1:] = ps[1:] != ps[:-1]
    sel = order[first]
    Z[pix[sel]] = zz[sel]
    FID[pix[sel]] = fid[sel]
    FIDm = FID.reshape(H, W)
    PART = np.where(FIDm >= 0, face_part[np.maximum(FIDm, 0)], -1).astype(np.int8)
    return Z.reshape(H, W), PART, FIDm


# ------------------------------------------------------------------ tools/xag/build_sources.py
def inputs_for(root, legacy, seq, mano):
    base = root / legacy / seq
    paths = [Path(str(base) + suffix) for suffix in ("_events.npy", "_offsets.npy", "_aux.npz", ".meta")]
    optional = Path(str(base) + "_tsub.npy")
    if optional.exists():
        paths.append(optional)
    paths += [mano, SPLITS, Path(__file__).resolve(), REPO / "semkine" / "xag_sources.py"]
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
                "n_events": len(events), "n_ms": len(offsets) - 1, "frame": [W, H], "render_scale": RENDER_SCALE,
                "inputs": inputs_for(root, legacy, seq, mano),
                "contract": "preaugmentation per-event-ms GT mesh; 2px foreground contour; 10mm discontinuity; "
                            "fixed face-majority part; real canvas n346"}
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


CROP_MARGIN = 4          # > the 2 px contour radius + the 1 px 4-neighbourhood: no label outside depends on the crop


def frame_labels(Z, P, FID, x, y, crop=True):
    """`source_maps` read at the events (x, y): part, face and the full-frame source pixel of each event.

    With `crop`, `source_maps` runs on the hand's bounding box grown by CROP_MARGIN px (clipped to the frame) instead
    of the whole frame; every label is the same (`--verify` checks it bit for bit): outside the box no hand pixel lies
    within the 2 px contour radius, and inside it every 4-neighbour and 2 px shift the rules read is either inside the
    box or a non-hand pixel, which the box's padding reproduces (the frame's own border stays the frame's border)."""
    n = len(x)
    out_part, out_face, out_pix = np.full(n, -1, np.int8), np.full(n, -1, np.int16), np.full(n, -1, np.int64)
    hand = np.isfinite(Z)
    if not crop:
        maps = source_maps(Z, P, FID)
        return maps["part"][y, x], maps["face"][y, x], maps["pixel"][y, x]
    if not hand.any():
        return out_part, out_face, out_pix
    ys, xs = np.nonzero(hand)
    y0, y1 = max(0, int(ys.min()) - CROP_MARGIN), min(H, int(ys.max()) + CROP_MARGIN + 1)
    x0, x1 = max(0, int(xs.min()) - CROP_MARGIN), min(W, int(xs.max()) + CROP_MARGIN + 1)
    maps = source_maps(Z[y0:y1, x0:x1], P[y0:y1, x0:x1], FID[y0:y1, x0:x1])
    inside = (y >= y0) & (y < y1) & (x >= x0) & (x < x1)
    yy, xx = y[inside] - y0, x[inside] - x0
    out_part[inside], out_face[inside] = maps["part"][yy, xx], maps["face"][yy, xx]
    pc = maps["pixel"][yy, xx]
    wc = x1 - x0
    out_pix[inside] = np.where(pc >= 0, (y0 + pc // wc) * W + (x0 + pc % wc), -1)
    return out_part, out_face, out_pix


def process_chunk(task):
    seq, milliseconds = task
    start_time = time.monotonic()
    a, g = worker_sequence(seq), _WORKER
    ms = np.asarray(milliseconds, np.int64)
    nonempty = ms[a["offsets"][ms + 1] > a["offsets"][ms]]
    n_events = n_valid = 0
    if len(nonempty):
        for frame in nonempty:
            vertices, _ = fk_batch(g["mano"], a["betas"], a["pos51"][frame:frame + 1])
            verts = vertices[0].numpy().astype(np.float64)
            Z, P, FID = rasterize(verts, g["faces"], g["face_part"], a["fxyc"])
            lo, hi = int(a["offsets"][frame]), int(a["offsets"][frame + 1])
            ev = a["events"][lo:hi]
            x, y = ev[:, 0].astype(np.int64), ev[:, 1].astype(np.int64)
            part, face, pixel = frame_labels(Z, P, FID, x, y)
            bary = source_barycentrics(verts, g["faces"], a["fxyc"], face, pixel, width=W)
            a["part"][lo:hi], a["face"][lo:hi], a["bary"][lo:hi] = part, face, bary
            n_events += hi - lo
            n_valid += int((face >= 0).sum())
        for key in ("part", "face", "bary"):
            a[key].flush()
    return seq, milliseconds, n_events, n_valid, time.monotonic() - start_time


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--mano", type=Path, default=REPO / "assets" / "mano_right.npz")
    parser.add_argument("--seq", nargs="+", help="default: every training sequence of splits_real.json")
    parser.add_argument("--workers", type=int, default=min(100, os.cpu_count() or 1))
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--max-ms", type=int, help="first N requested valid milliseconds per sequence (benchmark)")
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--verify", type=int, metavar="N",
                        help="check the cropped labels against the whole-frame labels on N random frames, then exit")
    args = parser.parse_args()
    root = DATA_ROOT
    seq_dirs = dict(sequences_for_split(root, "train", SPLITS))
    if args.seq:
        missing = set(args.seq) - set(seq_dirs)
        if missing:
            parser.error(f"sequences absent from the training split: {sorted(missing)}")
        seq_dirs = {seq: seq_dirs[seq] for seq in args.seq}
    if args.verify:
        rng = np.random.default_rng(0)
        init_worker(str(root), str(args.out_dir), seq_dirs, str(args.mano))
        g, seqs, checked, labelled, t_full, t_crop = _WORKER, sorted(seq_dirs), 0, 0, 0.0, 0.0
        while checked < args.verify:
            seq = seqs[int(rng.integers(len(seqs)))]
            base = root / seq_dirs[seq] / seq
            offsets = np.load(str(base) + "_offsets.npy", mmap_mode="r")
            frame = int(rng.integers(len(offsets) - 1))
            lo, hi = int(offsets[frame]), int(offsets[frame + 1])
            if hi == lo:
                continue
            with np.load(str(base) + "_aux.npz", allow_pickle=True) as aux:
                betas = torch.tensor(aux["betas"], dtype=torch.float32).reshape(1, -1)
                fxyc = tuple(float(v) for v in render_intrinsics_f32(aux["camera_K"]))
            pos = _read_meta51(str(base) + ".meta")
            verts = fk_batch(g["mano"], betas, pos[frame:frame + 1])[0][0].numpy().astype(np.float64)
            Z, P, FID = rasterize(verts, g["faces"], g["face_part"], fxyc)
            ev = np.load(str(base) + "_events.npy", mmap_mode="r")[lo:hi]
            x, y = ev[:, 0].astype(np.int64), ev[:, 1].astype(np.int64)
            t0 = time.perf_counter()
            full = frame_labels(Z, P, FID, x, y, crop=False)
            t1 = time.perf_counter()
            crop = frame_labels(Z, P, FID, x, y, crop=True)
            t2 = time.perf_counter()
            t_full, t_crop = t_full + t1 - t0, t_crop + t2 - t1
            for a_, b_ in zip(full, crop):
                if not np.array_equal(np.asarray(a_).astype(np.int64), np.asarray(b_).astype(np.int64)):
                    raise SystemExit(f"crop differs from the whole frame: {seq} ms {frame}")
            checked += 1
            labelled += int((full[1] >= 0).sum())
        print(json.dumps({"frames": checked, "labelled_events": labelled, "identical": True,
                          "ms_per_frame_full": 1e3 * t_full / checked, "ms_per_frame_crop": 1e3 * t_crop / checked}))
        return
    if args.estimate:
        count = valid_ms = 0
        bad = []
        for seq, legacy in seq_dirs.items():
            base = root / legacy / seq
            offsets = np.load(str(base) + "_offsets.npy", mmap_mode="r")
            count += int(offsets[-1])
            with np.load(str(base) + "_aux.npz", allow_pickle=True) as aux:
                valid_ms += int(valid_milliseconds(aux, len(offsets) - 1).sum())
            if len(_read_meta51(str(base) + ".meta")) != len(offsets) - 1:
                bad.append(seq)
        print(json.dumps({"sequences": len(seq_dirs), "events": count, "valid_ms": valid_ms,
                          "cache_GiB": count * 15 / 2 ** 30, "meta_length_mismatch": bad}, indent=2))
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    overall_start = time.monotonic()
    states, locks, tasks = {}, [], []
    for seq, legacy in seq_dirs.items():
        identity, wanted, done, lock = prepare(root, args.out_dir, seq, legacy, args.mano)
        locks.append(lock)
        eligible = np.flatnonzero(wanted)
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
        # interleave the sequences so that every worker's two-entry sequence cache is reused across neighbouring chunks
        with mp.get_context("fork").Pool(args.workers, initializer=init_worker,
                                         initargs=(str(root), str(args.out_dir), seq_dirs, str(args.mano))) as pool:
            for i, (seq, milliseconds, ne, nv, seconds) in enumerate(
                    pool.imap_unordered(process_chunk, tasks, chunksize=1), 1):
                done = states[seq][2]
                done[milliseconds] = True
                done.flush()
                n_ms += len(milliseconds)
                n_events += ne
                n_valid += nv
                cpu_seconds += seconds
                if i % max(1, len(tasks) // 50) == 0 or i == len(tasks):
                    print(json.dumps({"chunks_done": i, "chunks_total": len(tasks), "ms_done": n_ms,
                                      "events_done": n_events, "valid": n_valid,
                                      "elapsed_s": round(time.monotonic() - run_start, 1)}), flush=True)
    completed = {}
    for seq, (identity, wanted, done) in states.items():
        completed[seq] = bool(np.all(done[wanted]))
        write_json(args.out_dir / seq / "metadata.json", {**identity, "complete": completed[seq],
                   "completed_valid_ms": int(done[wanted].sum()), "total_valid_ms": int(wanted.sum())})
    summary = {"version": SOURCE_VERSION, "frame": [W, H], "render_scale": RENDER_SCALE, "workers": args.workers,
               "chunk_ms": args.chunk_ms, "processed_ms": n_ms, "processed_events": n_events, "valid_sources": n_valid,
               "worker_seconds": cpu_seconds, "compute_wall_seconds": time.monotonic() - run_start,
               "total_wall_seconds": time.monotonic() - overall_start, "complete": completed,
               "archived_builder": str(ARCHIVED / "xag/build_sources.py")}
    write_json(args.out_dir / "build_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
