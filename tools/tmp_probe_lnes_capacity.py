#!/usr/bin/env python3
"""Does a longer evidence window actually put more evidence into the tensor the network reads?

`docs/PLAN_SELECTION_VERDICT_20260825.md` §10.3 measured that the 5 ms recursive penalty is a
*window* effect, not an update-rate effect: holding W at 50 ms while stepping at 5 ms recovers 81%
of the loss. That makes W the lever, and §10.4 turned it into the pre-registered gate -- can a
longer effective window push the 14.88 mm single-step bias down?

Before spending a retrain on that gate, there is a cheaper question that can invalidate it outright.
`build_lnes` writes one timestamp per (pixel, polarity) and lets later events overwrite earlier
ones, so its capacity is bounded by 180*240*2 slots, not by W. If occupancy saturates in W, then
extending the window adds events that the representation discards by construction, the gate is
untestable on LNES no matter how the timestamps are normalised, and the case for a persistent-state
encoder stops being a preference and becomes the only way to raise the evidence ceiling.

Reports, per W: raw events in the window, occupied slots, the overwrite fraction those two imply,
and the `tval` distribution that tells us whether the temporal code is still being used or has
collapsed towards the most recent millisecond.

Temporary: `/tmp`-class diagnostic per the experiment contract; delete after the verdict lands.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from semkine import eval_track as ET                              # noqa: E402
from semkine.dataset import sequences_for_split, splits_manifest   # noqa: E402


def measure(events, offsets, end: int, window: int) -> dict:
    """Occupancy and overwrite for one LNES build, without allocating the float image twice."""
    start = end - window + 1
    a0, a1 = int(offsets[start]), int(offsets[end + 1])
    n_ev = a1 - a0
    if n_ev <= 0:
        return {"n_events": 0, "occupied": 0, "overwrite_frac": 0.0,
                "tval_mean": 0.0, "tval_p10": 0.0, "occupancy_frac": 0.0}
    ev = np.asarray(events[a0:a1])
    counts = np.diff(offsets[start:end + 2]).astype(np.int64)
    tval = np.repeat(np.arange(window, dtype=np.float32), counts) / float(window)
    # Flat slot index; last writer wins, exactly as build_lnes does.
    slot = ((ev[:, 1].astype(np.int64) * ET.W + ev[:, 0].astype(np.int64)) * 2
            + np.clip(ev[:, 2].astype(np.int64), 0, 1))
    img = np.zeros(ET.H * ET.W * 2, np.float32)
    img[slot] = tval                      # numpy fancy-index assignment keeps the last write
    occ = int(np.count_nonzero(img))
    kept = img[img > 0]
    return {"n_events": int(n_ev), "occupied": occ,
            "overwrite_frac": float(1.0 - occ / max(n_ev, 1)),
            "occupancy_frac": occ / float(ET.H * ET.W * 2),
            "tval_mean": float(kept.mean()) if occ else 0.0,
            "tval_p10": float(np.percentile(kept, 10)) if occ else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val_core")
    ap.add_argument("--config", default="configs/semkine/s1_track_domrand.yaml",
                    help="supplies the data root and the subject split manifest")
    ap.add_argument("--windows", default="5,10,20,50,100,200,400")
    ap.add_argument("--samples", type=int, default=60,
                    help="evaluation instants per sequence, spread over the valid runs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    wins = [int(x) for x in a.windows.split(",")]
    from config import load_config
    cfg = load_config(a.config)
    root = Path(cfg["DATA"]["ROOT"])

    agg = {w: [] for w in wins}
    for seq, d in sequences_for_split(root, a.split, splits_manifest(cfg)):
        events, offsets, aux, _ = ET.load_sequence(root, d, seq)
        runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
        # Only instants where every window fits inside one valid run, so all W see the same instant.
        cand = []
        for lo, hi in runs:
            lo2 = int(lo) + max(wins)
            if hi - lo2 > 1:
                cand.append(np.linspace(lo2, int(hi) - 1, a.samples, dtype=np.int64))
        if not cand:
            continue
        ends = np.unique(np.concatenate(cand))
        per_w = {}
        for w in wins:
            ms = [measure(events, offsets, int(e), w) for e in ends]
            per_w[w] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
            agg[w].append(per_w[w])
        print(f"  {seq}: " + "  ".join(
            f"W{w}:{per_w[w]['occupancy_frac']*100:.1f}%/{per_w[w]['overwrite_frac']*100:.0f}%ovr"
            for w in wins), flush=True)

    pooled = {w: {k: float(np.mean([d[k] for d in agg[w]])) for k in agg[w][0]}
              for w in wins if agg[w]}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"split": a.split, "windows": wins, "samples": a.samples, "pooled": pooled}, indent=2))
    print(f"\n{'W(ms)':>6} {'events':>10} {'occupied':>9} {'occ%':>7} {'overwrite%':>11} {'tval_mean':>10}")
    for w in wins:
        if w not in pooled:
            continue
        p = pooled[w]
        print(f"{w:>6} {p['n_events']:>10.0f} {p['occupied']:>9.0f} "
              f"{p['occupancy_frac']*100:>6.2f}% {p['overwrite_frac']*100:>10.1f}% {p['tval_mean']:>10.3f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
