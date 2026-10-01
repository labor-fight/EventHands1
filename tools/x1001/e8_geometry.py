#!/usr/bin/env python3
"""x1001 E8 pre-check: how far back in time does a node's 3-hop causal neighbourhood reach under
each candidate neighbour rule, and what does the neighbour search cost at training batch size?

Weight-independent (graph geometry only). Packets: every 6th protocol packet of zgz_global and
zgz_local (dev), S37 stride sampling, <= 2048 nodes. Rules:
  W32       the S37 rule: k = 8 nearest among the 32 preceding nodes, metric (x/W, y/H, t_norm)
  W256      the same with 256 preceding nodes (the earlier D candidate)
  ALL_t{s}  k = 8 nearest among *all* earlier nodes of the packet, metric (x/W, y/H, s * t_norm)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))
sys.path.insert(0, str(REPO / "tools" / "x1001"))
import diag_export as DE                                              # noqa: E402
from semkine.encoder import event_tokens                              # noqa: E402
from semkine.event_gnn import EventGNN, T_COL                         # noqa: E402


def knn_all(p, mask, k, s):
    """k nearest *earlier* nodes over the whole packet; p (B, N, 3) graph coordinates."""
    q = p.clone()
    q[..., 2] = q[..., 2] * s
    B, N, _ = q.shape
    d = torch.cdist(q, q) ** 2                                         # (B, N, N)
    i = torch.arange(N, device=p.device)
    causal = i.unsqueeze(0) < i.unsqueeze(1)                           # j < i
    valid = causal.unsqueeze(0) & mask.unsqueeze(1) & mask.unsqueeze(2)
    d = d.masked_fill(~valid, float("inf"))
    near = d.topk(min(k, N), dim=-1, largest=False)
    emask = torch.isfinite(near.values).to(p.dtype)
    idx = near.indices * (emask > 0)
    return idx


def knn_window(p, mask, k, window):
    enc = EventGNN(max_nodes=p.shape[1], k=k, window=window)
    idx, _, emask = enc._edges(p, mask)
    return idx * (emask > 0), emask


def spans(idx, ok, ev, mask):
    B = idx.shape[0]
    t = ev[..., 3] * 1000.0
    x, y = ev[..., 1], ev[..., 2]

    def nb(v, fill):
        return v.gather(1, idx.reshape(B, -1)).reshape(idx.shape).masked_fill(~ok, fill)
    tmin, xmin, ymin, xmax, ymax = t, x, y, x, y
    for _ in range(3):
        tmin, xmin, ymin, xmax, ymax = (torch.minimum(t, nb(tmin, 1e9).min(-1).values),
                                        torch.minimum(x, nb(xmin, 1e9).min(-1).values),
                                        torch.minimum(y, nb(ymin, 1e9).min(-1).values),
                                        torch.maximum(x, nb(xmax, -1e9).max(-1).values),
                                        torch.maximum(y, nb(ymax, -1e9).max(-1).values))
    span = (t - tmin).masked_fill(~mask, float("nan"))
    rad = (0.5 * torch.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)).masked_fill(~mask, float("nan"))
    xj = x.gather(1, idx.reshape(B, -1)).reshape(idx.shape)
    yj = y.gather(1, idx.reshape(B, -1)).reshape(idx.shape)
    el = torch.sqrt((xj - x.unsqueeze(-1)) ** 2 + (yj - y.unsqueeze(-1)) ** 2).masked_fill(~ok, float("nan"))
    tj = t.gather(1, idx.reshape(B, -1)).reshape(idx.shape)
    dtj = (t.unsqueeze(-1) - tj).masked_fill(~ok, float("nan"))
    return span, rad, el, dtj


@torch.no_grad()
def main():
    dev = torch.device("cuda")
    enc = EventGNN(max_nodes=2048)
    rules = {"W32": ("win", 32), "W256": ("win", 256)}
    for s in (1.0, 0.3, 0.1, 0.03):
        rules[f"ALL_t{s:g}"] = ("all", s)
    out = {}
    for seq in ("zgz_global", "zgz_local"):
        sd = DE.SeqData("val", seq)
        rows = sd.rows()[::6]
        acc = {r: {"span3_ms": [], "rad3_px": [], "edge_px": [], "edge_dt_ms": []} for r in rules}
        for i0 in range(0, len(rows), 8):
            evs = [sd.ev5(e) for _, _, _, e in rows[i0:i0 + 8]]
            prev = torch.zeros(len(evs), 51, device=dev)
            b = torch.from_numpy(np.stack([sd.betas] * len(evs))).to(dev)
            K = torch.from_numpy(np.stack([sd.K] * len(evs))).to(dev)
            batch = DE.make_batch(evs, prev, b, K, dev)
            tok = event_tokens(batch.events, batch.ptr, batch.delta_t_s, 180, 240)
            src, mask = enc._sample(batch.events, batch.ptr)
            B, N = mask.shape
            flat = src.reshape(-1)
            ev = batch.events[flat].reshape(B, N, -1)
            tn = tok[flat, T_COL].reshape(B, N)
            p = torch.stack([ev[..., 1] / 240, ev[..., 2] / 180, tn], -1) * mask.unsqueeze(-1)
            for r, (kind, arg) in rules.items():
                if kind == "win":
                    idx, emask = knn_window(p.float(), mask, 8, arg)
                    ok = (emask > 0) & mask.unsqueeze(-1)
                else:
                    idx = knn_all(p.float(), mask, 8, arg)
                    i = torch.arange(N, device=dev)
                    ok = (i.unsqueeze(0).unsqueeze(-1) > idx) & mask.unsqueeze(-1)   # earlier node exists
                    ok = ok & (torch.arange(N, device=dev).view(1, N, 1) > 0)
                span, rad, el, dtj = spans(idx, ok, ev.float(), mask)
                acc[r]["span3_ms"].append(span[mask].cpu().numpy())
                acc[r]["rad3_px"].append(rad[mask].cpu().numpy())
                acc[r]["edge_px"].append(el[ok].cpu().numpy())
                acc[r]["edge_dt_ms"].append(dtj[ok].cpu().numpy())
        out[seq] = {r: {k: {"median": float(np.nanmedian(np.concatenate(v))),
                            "p90": float(np.nanpercentile(np.concatenate(v), 90))} for k, v in d.items()}
                    for r, d in acc.items()}
    # cost of the neighbour search at a training micro-batch (512 dense packets, 2048 nodes)
    cost = {}
    p = torch.rand(512, 2048, 3, device=dev)
    p[..., 2] = torch.sort(p[..., 2], dim=1).values
    mask = torch.ones(512, 2048, dtype=torch.bool, device=dev)
    for r, (kind, arg) in rules.items():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if kind == "win":
            knn_window(p, mask, 8, arg)
        else:
            for c in range(0, 512, 32):
                knn_all(p[c:c + 32], mask[c:c + 32], 8, arg)
        torch.cuda.synchronize()
        cost[r] = round((time.perf_counter() - t0) * 1000, 1)
    out["knn_ms_per_512_packets"] = cost
    dst = Path("/data1/lyq/code/mesh/EventHands1_x1001/diag/e8_geometry.json")
    dst.write_text(json.dumps(out, indent=1))
    for seq in ("zgz_global", "zgz_local"):
        print(seq)
        for r, d in out[seq].items():
            print(f"  {r:10s} span3 {d['span3_ms']['median']:6.2f} ms (p90 {d['span3_ms']['p90']:6.2f})  "
                  f"rad3 {d['rad3_px']['median']:5.1f} px  edge {d['edge_px']['median']:5.1f} px  "
                  f"edge dt {d['edge_dt_ms']['median']:6.2f} ms")
    print("knn ms / 512 packets:", cost)


if __name__ == "__main__":
    main()
