#!/usr/bin/env python3
"""Convert hand_data AEDAT4 + mesh GT into EventHands .evc/.meta + memmap events."""
from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, RotationSpline

SRC_W, SRC_H = 640, 480
OUT_W, OUT_H = 240, 180
MS_HZ = 1000
NCOMPS = 51
MAGIC = bytes([4, 13])


def continuous_valid_runs(valid: np.ndarray) -> List[Tuple[int, int]]:
    """Return half-open [start, end) runs of True in a 1D boolean array."""
    runs = []
    n = len(valid)
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i + 1
        while j < n and valid[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def interpolate_pose_ms(
    t_ann_s: np.ndarray,
    transl: np.ndarray,
    global_orient: np.ndarray,
    hand_pose: np.ndarray,
    valid: np.ndarray,
    n_ms: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate GT to 1 ms grid.
    Returns:
      params51: (n_ms, 51) float32  [t3, R3, residual45]
      valid_ms: (n_ms,) bool
    """
    params51 = np.zeros((n_ms, NCOMPS), dtype=np.float32)
    valid_ms = np.zeros(n_ms, dtype=bool)
    t_ms = np.arange(n_ms, dtype=np.float64) / MS_HZ

    for a, b in continuous_valid_runs(valid):
        if b - a < 2:
            # Single frame: mark only the nearest ms sample
            if b - a == 1:
                idx = int(round(t_ann_s[a] * MS_HZ))
                if 0 <= idx < n_ms:
                    params51[idx, 0:3] = transl[a]
                    params51[idx, 3:6] = global_orient[a]
                    params51[idx, 6:51] = hand_pose[a]
                    valid_ms[idx] = True
            continue

        ta = t_ann_s[a:b]
        t0 = ta[0]
        t1 = ta[-1]
        # Inclusive ms range covered by this continuous run
        i0 = int(np.ceil(t0 * MS_HZ - 1e-9))
        i1 = int(np.floor(t1 * MS_HZ + 1e-9))
        i0 = max(0, i0)
        i1 = min(n_ms - 1, i1)
        if i1 < i0:
            continue
        query = t_ms[i0 : i1 + 1]

        # Translation: PCHIP per axis
        tr = np.zeros((len(query), 3), dtype=np.float64)
        for d in range(3):
            tr[:, d] = PchipInterpolator(ta, transl[a:b, d])(query)

        # Global orient: RotationSpline
        go_spline = RotationSpline(ta, Rotation.from_rotvec(global_orient[a:b]))
        go = go_spline(query).as_rotvec()

        # Local 15 joints: independent RotationSpline per joint
        hp = np.zeros((len(query), 45), dtype=np.float64)
        for j in range(15):
            rv = hand_pose[a:b, 3 * j : 3 * j + 3]
            rs = RotationSpline(ta, Rotation.from_rotvec(rv))
            hp[:, 3 * j : 3 * j + 3] = rs(query).as_rotvec()

        params51[i0 : i1 + 1, 0:3] = tr.astype(np.float32)
        params51[i0 : i1 + 1, 3:6] = go.astype(np.float32)
        params51[i0 : i1 + 1, 6:51] = hp.astype(np.float32)
        valid_ms[i0 : i1 + 1] = True

    return params51, valid_ms


def project_pca6(residual45: np.ndarray, components: np.ndarray) -> np.ndarray:
    """alpha6 = residual @ pinv(C[:6]) with C shape (6, 45)."""
    C6 = components[:6]
    pinv = np.linalg.pinv(C6)  # (45, 6)
    return residual45 @ pinv


def write_meta(path: Path, params51: np.ndarray) -> None:
    """Write EventHands .meta: int32 ncomps + (float64[ncomps] + magic)*N."""
    n = params51.shape[0]
    with path.open("wb") as f:
        f.write(struct.pack("<i", NCOMPS))
        # Pack as structured for speed
        row = np.empty(
            n,
            dtype=[("data", "<f8", NCOMPS), ("m0", "u1"), ("m1", "u1")],
        )
        row["data"] = params51.astype(np.float64)
        row["m0"] = 4
        row["m1"] = 13
        row.tofile(f)


def events_to_evc_and_memmap(
    aedat_path: Path,
    n_ms: int,
    events_npy: Path,
    offsets_npy: Path,
    evc_path: Path,
) -> Dict:
    """Read AEDAT4, bin to 1ms, scale to 240x180, write memmap + .evc."""
    from dv import AedatFile

    # First pass: count events per ms bin
    counts = np.zeros(n_ms, dtype=np.int64)
    total = 0
    with AedatFile(str(aedat_path)) as f:
        for packet in f["events"].numpy():
            ts = packet["timestamp"]  # us, relative to stream start (~0)
            ms = (ts // 1000).astype(np.int64)
            # Clip to [0, n_ms)
            mask = (ms >= 0) & (ms < n_ms)
            if not np.any(mask):
                continue
            ms = ms[mask]
            np.add.at(counts, ms, 1)
            total += int(mask.sum())

    offsets = np.zeros(n_ms + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    assert offsets[-1] == total

    # Memory-map destination
    events = np.lib.format.open_memmap(
        events_npy, mode="w+", dtype=np.uint8, shape=(total, 3)
    )
    # Cursor within each bin
    cursor = offsets[:-1].copy()

    with AedatFile(str(aedat_path)) as f:
        for packet in f["events"].numpy():
            ts = packet["timestamp"]
            ms = (ts // 1000).astype(np.int64)
            mask = (ms >= 0) & (ms < n_ms)
            if not np.any(mask):
                continue
            xs = packet["x"][mask].astype(np.int32)
            ys = packet["y"][mask].astype(np.int32)
            ps = packet["polarity"][mask].astype(np.uint8)
            ms = ms[mask]
            # Scale 640x480 -> 240x180
            xs = (xs * OUT_W // SRC_W).astype(np.uint8)
            ys = (ys * OUT_H // SRC_H).astype(np.uint8)
            # Polarity: dv uses True/False; EventHands uses 0/1 as channel index
            # In original EventHands: p=0 or 1; aedat readers in predict3 use
            # 0 if polarity else 1. We'll keep True->1, False->0 matching dv polarity.
            # Check: original fastevc uses p as channel. We'll use polarity as-is (0/1).
            ps = ps.astype(np.uint8)

            # Sort into bins while preserving order within each packet.
            # Events arrive time-ordered; write sequentially into each bin.
            # Group by ms for vectorized write.
            order = np.argsort(ms, kind="mergesort")
            ms_s = ms[order]
            xs_s = xs[order]
            ys_s = ys[order]
            ps_s = ps[order]
            # Find contiguous runs of same ms
            if len(ms_s) == 0:
                continue
            cuts = np.flatnonzero(np.diff(ms_s)) + 1
            starts = np.r_[0, cuts]
            ends = np.r_[cuts, len(ms_s)]
            for s, e in zip(starts, ends):
                bin_id = int(ms_s[s])
                dst = int(cursor[bin_id])
                n = e - s
                events[dst : dst + n, 0] = xs_s[s:e]
                events[dst : dst + n, 1] = ys_s[s:e]
                events[dst : dst + n, 2] = ps_s[s:e]
                cursor[bin_id] = dst + n

    events.flush()
    assert np.all(cursor == offsets[1:]), "bin fill mismatch"

    np.save(offsets_npy, offsets)

    # Write .evc from memmap (compatible with evcreader)
    with evc_path.open("wb") as f:
        sep = struct.pack("<HBB", 0, 0, 255)
        for i in range(n_ms):
            a, b = int(offsets[i]), int(offsets[i + 1])
            if b > a:
                # x as uint16, y uint8, p uint8
                chunk = events[a:b]
                # Pack as little-endian: H B B
                packed = np.empty(b - a, dtype=[("x", "<u2"), ("y", "u1"), ("p", "u1")])
                packed["x"] = chunk[:, 0]
                packed["y"] = chunk[:, 1]
                packed["p"] = chunk[:, 2]
                packed.tofile(f)
            f.write(sep)

    return {
        "n_events": int(total),
        "n_ms": int(n_ms),
        "events_per_ms_mean": float(counts.mean()) if n_ms else 0.0,
    }


def process_sequence(
    src_root: Path,
    out_root: Path,
    split: str,
    seq: str,
    mano_components: np.ndarray,
    skip_existing: bool = True,
) -> Dict:
    seq_dir = src_root / split / seq
    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    out_evc = out_dir / f"{seq}.evc"
    out_meta = out_dir / f"{seq}.meta"
    out_events = out_dir / f"{seq}_events.npy"
    out_offsets = out_dir / f"{seq}_offsets.npy"
    out_aux = out_dir / f"{seq}_aux.npz"
    out_done = out_dir / f"{seq}.done.json"

    if skip_existing and out_done.exists() and out_evc.exists() and out_meta.exists():
        return json.loads(out_done.read_text())

    mesh = np.load(seq_dir / "mesh" / f"{seq}_mesh.npz")
    eh12 = np.load(seq_dir / "mesh" / f"{seq}_eventhands12.npz")
    with open(seq_dir / "event" / f"{seq}_source_timestamps.csv") as f:
        rows = list(csv.DictReader(f))
    t0_ns = int(rows[0]["header_stamp_ns"])

    ts_ns = mesh["timestamp_ns"]
    t_ann_s = (ts_ns - t0_ns) / 1e9
    assert abs(t_ann_s[0]) < 1e-3, f"{seq}: annotation not aligned to event t0"

    # Duration: cover annotations and a bit of event tail
    n_ms = int(np.ceil(t_ann_s[-1] * MS_HZ)) + 1
    # Also read frame_times end if available
    ft_path = seq_dir / "event" / f"{seq}_events-frame_times.txt"
    if ft_path.exists():
        ft = np.loadtxt(ft_path, comments="#")
        n_ms = max(n_ms, int(np.ceil(float(ft[-1, 2]) * MS_HZ)))

    valid_ann = mesh["valid"].astype(bool) & (~mesh["interpolated"].astype(bool))
    transl = mesh["camera_transl_right_canonical_m"].astype(np.float64)
    go = mesh["global_orient_rotvec"].astype(np.float64)
    hp = mesh["hand_pose_residual_aa"].astype(np.float64)
    betas = mesh["betas"].astype(np.float32)
    K = mesh["camera_K"].astype(np.float32)

    # --- Sanity: recompute 12D at annotation frames and match eh12 ---
    alpha6 = project_pca6(hp, mano_components)
    params12_re = np.concatenate([alpha6, transl, go], axis=1).astype(np.float32)
    params12_gt = eh12["params_12"].astype(np.float32)
    # Only compare valid non-interpolated frames
    mask = valid_ann
    if mask.any():
        diff = np.abs(params12_re[mask] - params12_gt[mask]).max()
        if diff > 1e-4:
            raise RuntimeError(
                f"{seq}: params_12 mismatch max_abs={diff:.6g} (atol 1e-4)"
            )

    # betas constant within sequence (float32 noise can push std slightly above 1e-5)
    beta0 = betas[valid_ann][0] if valid_ann.any() else betas[0]
    beta_range = float(
        (betas[valid_ann].max(axis=0) - betas[valid_ann].min(axis=0)).max()
    ) if valid_ann.any() else 0.0
    if beta_range > 1e-4:
        raise RuntimeError(f"{seq}: betas not constant in sequence (range={beta_range})")

    params51, valid_ms = interpolate_pose_ms(
        t_ann_s, transl, go, hp, valid_ann, n_ms
    )

    # GT frame -> ms index mapping
    gt_ms_idx = np.clip(np.rint(t_ann_s * MS_HZ).astype(np.int64), 0, n_ms - 1)

    aedat = seq_dir / "event" / f"{seq}_events.aedat4"
    ev_stats = events_to_evc_and_memmap(
        aedat, n_ms, out_events, out_offsets, out_evc
    )
    write_meta(out_meta, params51)

    # Continuous valid ms runs for sampling
    runs = continuous_valid_runs(valid_ms)

    category = "global" if "global" in seq else ("local" if "local" in seq else "other")
    aux = {
        "betas": beta0,
        "camera_K": K[0] if K.ndim == 3 else K,
        "valid_ms": valid_ms,
        "gt_ms_idx": gt_ms_idx,
        "gt_valid": valid_ann,
        "gt_timestamp_ns": ts_ns,
        "t0_ns": np.int64(t0_ns),
        "n_ms": np.int64(n_ms),
        "valid_runs_ms": np.asarray(runs, dtype=np.int64).reshape(-1, 2)
        if runs
        else np.zeros((0, 2), dtype=np.int64),
        "category": np.array(category),
        "split": np.array(split),
        "canonical_hand": np.array("right"),
        "params_layout": np.array(
            ["t[0:3]", "R_root[3:6]", "hand_pose_residual_aa[6:51]"]
        ),
    }
    np.savez_compressed(out_aux, **aux)

    summary = {
        "seq": seq,
        "split": split,
        "category": category,
        "n_ms": n_ms,
        "n_ann": int(len(ts_ns)),
        "n_valid_ann": int(valid_ann.sum()),
        "n_valid_ms": int(valid_ms.sum()),
        "n_events": ev_stats["n_events"],
        "params12_max_abs_err": float(diff) if mask.any() else 0.0,
        "betas": beta0.tolist(),
    }
    out_done.write_text(json.dumps(summary, indent=2))
    print(
        f"[{split}/{seq}] ms={n_ms} events={ev_stats['n_events']} "
        f"valid_ms={valid_ms.sum()} params12_err={summary['params12_max_abs_err']:.2e}",
        flush=True,
    )
    return summary


def _worker(args):
    src, out, split, seq, components, skip = args
    try:
        return process_sequence(src, out, split, seq, components, skip)
    except Exception as e:
        return {"seq": seq, "split": split, "error": repr(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("/data1/lyq/data/hand_data/hand_data"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "hand_data51",
    )
    ap.add_argument(
        "--mano",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "mano_right.npz",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sequences", nargs="*", default=None, help="Optional seq filter")
    ap.add_argument("--no-skip", action="store_true")
    args = ap.parse_args()

    mano = np.load(args.mano)
    components = mano["hands_components"].astype(np.float64)

    jobs = []
    for split in ("train", "val"):
        split_dir = args.src / split
        if not split_dir.exists():
            continue
        for seq in sorted(p.name for p in split_dir.iterdir() if p.is_dir()):
            if args.sequences and seq not in args.sequences:
                continue
            jobs.append(
                (args.src, args.out, split, seq, components, not args.no_skip)
            )

    print(f"Processing {len(jobs)} sequences with {args.workers} workers -> {args.out}")
    results = []
    if args.workers <= 1:
        for j in jobs:
            results.append(_worker(j))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_worker, j) for j in jobs]
            for fut in as_completed(futs):
                results.append(fut.result())

    errors = [r for r in results if "error" in r]
    if errors:
        print("ERRORS:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    # Write summary + splits
    trials = {}
    splits = {"train": {"trials": [], "ms_frame_ranges": {}}, "val": {"trials": [], "ms_frame_ranges": {}}}
    for r in sorted(results, key=lambda x: (x["split"], x["seq"])):
        trials[r["seq"]] = r
        splits[r["split"]]["trials"].append(r["seq"])
        splits[r["split"]]["ms_frame_ranges"][r["seq"]] = [0, r["n_ms"]]

    summary = {
        "format": "EventHands-compatible .evc/.meta + memmap",
        "dataset": "hand_data",
        "event_resolution_src": [SRC_W, SRC_H],
        "output_resolution": [OUT_W, OUT_H],
        "event_frame_rate_hz": MS_HZ,
        "target_dims": NCOMPS,
        "target_layout": [
            "trans_x",
            "trans_y",
            "trans_z",
            "global_orient_x",
            "global_orient_y",
            "global_orient_z",
            *[f"hand_pose_residual_aa[{i}]" for i in range(45)],
        ],
        "target_note": (
            "meta is always 51D [t, R_root, residual45]. "
            "12D experiment derives [alpha6,t,R] via alpha6=residual@pinv(C[:6])."
        ),
        "trials": trials,
        "splits": splits,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "splits.json").write_text(json.dumps(splits, indent=2))
    print(f"Done. Wrote {args.out}/summary.json with {len(results)} sequences.")


if __name__ == "__main__":
    main()
