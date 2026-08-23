#!/usr/bin/env python3
"""Recover the sub-millisecond event timestamp that `prepare_hand_data.py` discarded.

`prepare_hand_data.py` bins events to 1 ms (`ts // 1000`) and stores only `(x, y, p)`, so
`data/hand_data51` cannot answer "when inside this millisecond did the event fire". The raw
AEDAT4 files still have the microsecond stamps, and the binning loop is deterministic: packets
arrive time-ordered, `np.argsort(ms, kind="mergesort")` is stable, and each bin is filled
sequentially. Replaying the identical loop and writing `ts % 1000` therefore produces an array
that is row-aligned with `<seq>_events.npy`.

Output: `<seq>_tsub.npy`, `(N,) uint16`, microseconds within the millisecond, so the absolute
timestamp of event `i` in bin `m` is `m * 1000 + tsub[i]` microseconds.

The alignment is verified, not assumed: `(x, y, p)` are recomputed alongside and compared
row-by-row against the stored array. A mismatch aborts the sequence rather than writing a
silently misaligned file.

Usage:
  python tools/extract_subms.py                      # all sequences, skip existing
  python tools/extract_subms.py --sequences zgz_local --no-skip
  python tools/extract_subms.py --verify-only        # re-check existing files
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SRC_W, SRC_H = 640, 480
OUT_W, OUT_H = 240, 180


def _iter_binned(aedat_path: Path, n_ms: int):
    """Yield (ms, x, y, p, tsub) per packet, exactly as prepare_hand_data.py sees them."""
    from dv import AedatFile

    with AedatFile(str(aedat_path)) as f:
        for packet in f["events"].numpy():
            ts = packet["timestamp"]
            ms = (ts // 1000).astype(np.int64)
            mask = (ms >= 0) & (ms < n_ms)
            if not np.any(mask):
                continue
            xs = (packet["x"][mask].astype(np.int32) * OUT_W // SRC_W).astype(np.uint8)
            ys = (packet["y"][mask].astype(np.int32) * OUT_H // SRC_H).astype(np.uint8)
            ps = packet["polarity"][mask].astype(np.uint8)
            tsub = (ts[mask] % 1000).astype(np.uint16)
            yield ms[mask], xs, ys, ps, tsub


def extract_sequence(src_root: Path, out_root: Path, split: str, seq: str,
                     skip_existing: bool = True, verify_only: bool = False) -> dict:
    out_dir = out_root / split
    events_npy = out_dir / f"{seq}_events.npy"
    offsets_npy = out_dir / f"{seq}_offsets.npy"
    tsub_npy = out_dir / f"{seq}_tsub.npy"
    if not events_npy.exists():
        return {"seq": seq, "split": split, "error": "missing events.npy"}

    events = np.load(events_npy, mmap_mode="r")
    offsets = np.load(offsets_npy)
    n_ms = len(offsets) - 1
    total = int(offsets[-1])

    if tsub_npy.exists() and skip_existing and not verify_only:
        t = np.load(tsub_npy, mmap_mode="r")
        if len(t) == total:
            return {"seq": seq, "split": split, "n_events": total, "skipped": True}

    aedat = src_root / split / seq / "event" / f"{seq}_events.aedat4"
    if not aedat.exists():
        return {"seq": seq, "split": split, "error": f"missing {aedat}"}

    if verify_only:
        if not tsub_npy.exists():
            return {"seq": seq, "split": split, "error": "no tsub to verify"}
        tsub = np.load(tsub_npy, mmap_mode="r")
        if len(tsub) != total:
            return {"seq": seq, "split": split, "error": f"len {len(tsub)} != {total}"}
    else:
        tsub = np.lib.format.open_memmap(
            tsub_npy, mode="w+", dtype=np.uint16, shape=(total,)
        )

    # Replay the binning with an identical cursor, checking (x, y, p) as we go.
    cursor = offsets[:-1].copy()
    mismatch = 0
    checked = 0
    for ms, xs, ys, ps, ts_sub in _iter_binned(aedat, n_ms):
        order = np.argsort(ms, kind="mergesort")
        ms_s, xs_s, ys_s, ps_s, tt_s = ms[order], xs[order], ys[order], ps[order], ts_sub[order]
        if len(ms_s) == 0:
            continue
        cuts = np.flatnonzero(np.diff(ms_s)) + 1
        for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(ms_s)]):
            bin_id = int(ms_s[s])
            dst = int(cursor[bin_id])
            n = e - s
            stored = np.asarray(events[dst : dst + n])
            got = np.stack([xs_s[s:e], ys_s[s:e], ps_s[s:e]], axis=1)
            if stored.shape == got.shape:
                mismatch += int((stored != got).any(axis=1).sum())
                checked += n
            else:
                mismatch += n
            if not verify_only:
                tsub[dst : dst + n] = tt_s[s:e]
            cursor[bin_id] = dst + n

    if not np.all(cursor == offsets[1:]):
        if not verify_only:
            tsub_npy.unlink(missing_ok=True)
        return {"seq": seq, "split": split, "error": "bin fill mismatch"}
    if mismatch:
        if not verify_only:
            tsub_npy.flush()
            tsub_npy.unlink(missing_ok=True)
        return {"seq": seq, "split": split, "error": f"{mismatch}/{checked} (x,y,p) rows differ"}

    if not verify_only:
        tsub.flush()
        # Non-decreasing absolute timestamps inside every bin is the contract S1 needs.
        t_all = np.load(tsub_npy, mmap_mode="r")
        bad = 0
        step = max(n_ms // 2000, 1)
        for m in range(0, n_ms, step):
            a, b = int(offsets[m]), int(offsets[m + 1])
            if b - a > 1:
                bad += int((np.diff(np.asarray(t_all[a:b]).astype(np.int64)) < 0).sum())
        if bad:
            return {"seq": seq, "split": split, "error": f"{bad} non-monotonic tsub in-bin"}

    out = {"seq": seq, "split": split, "n_events": total, "n_ms": n_ms,
           "rows_checked": checked, "mismatch": mismatch, "verified": True}
    print(f"[{split}/{seq}] events={total} checked={checked} mismatch={mismatch}", flush=True)
    return out


def _worker(a):
    try:
        return extract_sequence(*a)
    except Exception as e:  # noqa: BLE001 - report, do not kill the pool
        return {"seq": a[3], "split": a[2], "error": repr(e)}


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("/data1/lyq/data/hand_data/hand_data"))
    ap.add_argument("--out", type=Path, default=repo / "data" / "hand_data51")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--no-skip", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    splits = json.loads((args.out / "splits.json").read_text())
    jobs = []
    for split in ("train", "val"):
        for seq in splits.get(split, {}).get("trials", []):
            if args.sequences and seq not in args.sequences:
                continue
            jobs.append((args.src, args.out, split, seq, not args.no_skip, args.verify_only))

    print(f"{'Verifying' if args.verify_only else 'Extracting'} {len(jobs)} sequences "
          f"with {args.workers} workers", flush=True)
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
    done = [r for r in results if "error" not in r]
    n_ev = sum(r.get("n_events", 0) for r in done)
    print(f"\nok={len(done)} errors={len(errors)} total_events={n_ev:,} "
          f"(~{n_ev * 2 / 1e9:.2f} GB of uint16)")
    for e in errors:
        print("ERROR", e, file=sys.stderr)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
