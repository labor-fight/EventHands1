#!/usr/bin/env python3
"""S38: how much of an encoder's computation one new event changes (the asynchronous-update footprint).

An asynchronous runtime recomputes, per incoming event, only what that event changes; its cost is bounded
only if that set is bounded. This measures the set, without claiming a runtime: for real 50 ms packets
(zgz_global and zgz_local), one event is appended at the end of the packet (a copy of a random event of the
packet, moved to the packet's last timestamp, so it is a plausible newest event), and every intermediate of
the encoder is compared with and without it.

  sparse pyramid  sites whose feature changed, per level (exact comparison), and the MACs of recomputing
                  only those sites (each changed site's own kernel), against a full forward
  S37 graph       sampled nodes whose position in the time-uniform sample changed, and nodes whose final
                  feature changed

    python tools/s38/async_footprint.py --runs outputs/semkine/s38_spmeas_2k_s3407 outputs/semkine/rt_s37_2k_s3407
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "x1001")]
import evalx as EX                                                    # noqa: E402
from config import load_config                                       # noqa: E402
from model import MNISTModel                                         # noqa: E402
from semkine import eval_track as ET                                 # noqa: E402
from semkine.sparse_pyramid import SparsePyramid                     # noqa: E402

STEP = 50


def pyramid_trace(enc: SparsePyramid, ev: torch.Tensor, dt: torch.Tensor):
    """Per level: (sorted keys, features after the level's blocks); plus the pooled vector."""
    lv, f = enc._level0(ev, dt)
    h = enc.embed(f)
    out = []
    for i in range(enc.n_levels):
        for blk in enc.stages[i]:
            h = blk(h, lv.neighbours())
        out.append((lv.key, h))
        if i < enc.n_levels - 1:
            nxt, inv, slot = lv.parent()
            h = enc.downs[i](h, nxt.key.numel(), inv, slot)
            lv = nxt
    return out


def changed_sites(a, b):
    """Sites of `b` (the trace with the extra event) whose feature differs from `a`'s, or that are new."""
    ka, ha = a
    kb, hb = b
    pos = torch.searchsorted(ka, kb).clamp(max=max(len(ka) - 1, 0))
    present = ka[pos] == kb
    same = present & (ha[pos] == hb).all(-1)
    return int((~same).sum()), len(kb)


def site_macs(enc: SparsePyramid, level: int) -> int:
    """MACs of recomputing one site of `level` (its blocks and, above level 0, its share of the down step)."""
    c = enc.channels
    m = sum(9 * c[level] * c[level] for _ in enc.stages[level])
    if level > 0:
        m += 4 * c[level - 1] * c[level]
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--packets", type=int, default=40)
    ap.add_argument("--out", default=str(REPO / "outputs" / "s38" / "reports" / "async_footprint.json"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    res = {}
    for rd in a.runs:
        run = Path(rd)
        cfg = load_config(next(iter(sorted(run.glob("*.yaml")))))
        ck, step, _ = EX.find_ckpt(run, "last")
        m = MNISTModel.load_from_checkpoint(str(ck), cfg=cfg, map_location="cpu").to(dev).eval()
        enc = m.event_encoder
        root = Path(cfg["DATA"]["ROOT"])
        ent = {}
        for seq in ("zgz_global", "zgz_local"):
            events, offsets, aux, _ = ET.load_sequence(root, "val", seq)
            tsub = np.load(root / "val" / f"{seq}_tsub.npy", mmap_mode="r")
            runs_ms = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
            ends = [e for (s0, s1) in runs_ms for e in range(s0 + STEP - 1, s1, STEP)]
            stats = []
            for end in rng.choice(ends, size=min(a.packets, len(ends)), replace=False):
                ev = torch.from_numpy(ET._window_events(events, offsets, tsub, int(end), STEP)).to(dev)
                if len(ev) < 10:
                    continue
                new = ev[int(rng.integers(len(ev)))].clone()
                new[3] = ev[:, 3].max()
                ev2 = torch.cat([ev, new[None]], 0)
                dt = torch.tensor([STEP * 1e-3], device=dev)
                ptr = torch.tensor([0, len(ev)], device=dev)
                ptr2 = torch.tensor([0, len(ev2)], device=dev)
                with torch.no_grad():
                    if isinstance(enc, SparsePyramid):
                        ta, tb = pyramid_trace(enc, ev, dt), pyramid_trace(enc, ev2, dt)
                        per = [changed_sites(x, y) for x, y in zip(ta, tb)]
                        inc = sum(n * site_macs(enc, i) for i, (n, _) in enumerate(per))
                        full = sum(tot * site_macs(enc, i) for i, (_, tot) in enumerate(per))
                        stats.append({"events": len(ev), "changed": [p[0] for p in per], "sites": [p[1] for p in per],
                                      "trunk_macs_incremental": inc, "trunk_macs_full": full})
                    else:
                        src_a, mask_a = enc._sample(ev, ptr)
                        src_b, mask_b = enc._sample(ev2, ptr2)
                        both = mask_a & mask_b                               # slots live in both samples
                        moved = int(((src_a != src_b) & both).sum()) + int((mask_a != mask_b).sum())
                        _, ha, *_ = enc(ev, ptr, dt, None, return_nodes=True)
                        _, hb, *_ = enc(ev2, ptr2, dt, None, return_nodes=True)
                        n = min(ha.shape[1], hb.shape[1])
                        changed = int((~(ha[0, :n] == hb[0, :n]).all(-1)).sum()) + abs(ha.shape[1] - hb.shape[1])
                        stats.append({"events": len(ev), "nodes": int(mask_b.sum()), "resampled_nodes": moved,
                                      "changed_node_features": changed})
            if isinstance(enc, SparsePyramid):
                ch = np.array([s["changed"] for s in stats])
                si = np.array([s["sites"] for s in stats])
                ratio = np.array([s["trunk_macs_incremental"] / max(s["trunk_macs_full"], 1) for s in stats])
                ent[seq] = {"packets": len(stats), "events_median": float(np.median([s["events"] for s in stats])),
                            "changed_sites_median_per_level": np.median(ch, 0).tolist(),
                            "changed_sites_max_per_level": ch.max(0).tolist(),
                            "sites_median_per_level": np.median(si, 0).tolist(),
                            "incremental_over_full_trunk_macs_median": float(np.median(ratio)),
                            "incremental_over_full_trunk_macs_max": float(ratio.max())}
            else:
                ent[seq] = {"packets": len(stats), "events_median": float(np.median([s["events"] for s in stats])),
                            "nodes_median": float(np.median([s["nodes"] for s in stats])),
                            "resampled_nodes_median": float(np.median([s["resampled_nodes"] for s in stats])),
                            "changed_node_features_median": float(np.median([s["changed_node_features"] for s in stats]))}
            print(run.name, seq, json.dumps(ent[seq]), flush=True)
        res[run.name] = ent
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
