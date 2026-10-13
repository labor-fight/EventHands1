#!/usr/bin/env python3
"""D: EventGNN neighbourhood coverage, ENCODER_WINDOW 32 vs 256, on real zgz packets (CPU, no model).

Replicates EventGNN._sample/_edges (stride to <=2048, k=8 nearest among the W preceding nodes in
(x/W, y/H, t/dt)). Reports 1/3-hop time span (ms), radius (px), ancestors, edge stats, share of
edges beyond offset 32, under random dilution of zgz_global and on zgz_local; plus in-ms order bias.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(8)
ROOT = Path("/data1/lyq/code/mesh/EventHands1")
sys.path.insert(0, str(ROOT))
from semkine.eval_track import _window_events, load_sequence  # noqa: E402

OUT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "hand_data51"


def packets(seq, n_pk=30, step=50):
    ev, off, _aux, _pos = load_sequence(DATA, "val", seq)
    tp = DATA / "val" / f"{seq}_tsub.npy"
    tsub = np.load(tp, mmap_mode="r") if tp.exists() else None
    ends = list(range(step - 1, off.shape[0] - 1, step))
    ends = [ends[i] for i in np.linspace(0, len(ends) - 1, n_pk).astype(int)]
    return [np.asarray(_window_events(ev, off, tsub, e, step), dtype=np.float32) for e in ends]


def graph(ev5, W, k=8, max_nodes=2048, dts=0.05):
    n = ev5.shape[0]
    nk = min(n, max_nodes)
    step = np.float32(n) / np.float32(nk)
    src = np.minimum((np.arange(nk, dtype=np.float32) * step).astype(np.int64), n - 1)
    nd = torch.from_numpy(ev5[src])
    x, y, t = nd[:, 1], nd[:, 2], nd[:, 3]
    p = torch.stack([x / 240.0, y / 180.0, t / dts], -1)
    N = nk
    cand = torch.arange(N)[:, None] - torch.arange(1, W + 1)[None]
    ok = cand >= 0
    cand = cand.clamp(min=0)
    d = ((p[cand] - p[:, None]) ** 2).sum(-1).masked_fill(~ok, float("inf"))
    near = d.topk(min(k, W), dim=-1, largest=False)
    idx = cand.gather(1, near.indices)
    em = torch.isfinite(near.values)
    return x, y, t, idx, em, near.indices + 1


def stats(ev5, W, n_q=128):
    if ev5.shape[0] < 16:
        return None
    x, y, t, idx, em, offs = graph(ev5, W)
    N = x.shape[0]
    rows = torch.arange(N)[:, None].expand_as(idx)
    elen = torch.hypot(x[idx] - x[rows], y[idx] - y[rows])[em]
    edt = ((t[rows] - t[idx]) * 1e3)[em]
    A = torch.zeros(N, N)
    A[rows[em], idx[em]] = 1.0
    q = torch.linspace(N // 4, N - 1, n_q).long().unique()  # skip the packet head (truncated past)
    R = torch.zeros(q.numel(), N)
    R[torch.arange(q.numel()), q] = 1.0
    out = {"edge_px": elen.numpy(), "edge_dt_ms": edt.numpy(),
           "frac_edge_dt_ge1ms": float((edt >= 0.5).float().mean()),
           "frac_edge_off_gt32": float((offs[em] > 32).float().mean()), "n_nodes": N}
    for h in (1, 2, 3):
        R = ((R + R @ A) > 0).float()
        m = R > 0
        tq, xq, yq = t[q][:, None], x[q][:, None], y[q][:, None]
        span = ((tq - torch.where(m, t[None], tq)).amax(1) * 1e3)
        rad = torch.where(m, torch.hypot(x[None] - xq, y[None] - yq), torch.zeros(1)).amax(1)
        out[f"h{h}_span_ms"] = span.numpy()
        out[f"h{h}_rad_px"] = rad.numpy()
        out[f"h{h}_anc"] = m.sum(1).float().numpy()
    return out


def med(v):
    return float(np.median(np.concatenate(v))) if len(v) else None


def run_set(pks, W):
    acc = {}
    for ev5 in pks:
        s = stats(ev5, W)
        if s is None:
            continue
        for kk, v in s.items():
            acc.setdefault(kk, []).append(np.atleast_1d(v))
    r = {kk: med(v) for kk, v in acc.items()}
    r["edge_px_p90"] = float(np.percentile(np.concatenate(acc["edge_px"]), 90))
    r["h3_span_ms_p90"] = float(np.percentile(np.concatenate(acc["h3_span_ms"]), 90))
    return r


def order_bias(pks):
    """Spearman(index, y) and (index, x) inside each 1-ms bucket of raw events (raster-order check)."""
    ry, rx = [], []
    for ev5 in pks[:10]:
        ms = np.floor(ev5[:, 3] * 1e3 + 1e-6).astype(int)
        for b in np.unique(ms):
            sel = np.nonzero(ms == b)[0]
            if sel.size < 50:
                continue
            r_i = np.arange(sel.size)
            for arr, col in ((ry, 2), (rx, 1)):
                v = ev5[sel, col]
                rv = np.argsort(np.argsort(v, kind="stable"), kind="stable")
                arr.append(np.corrcoef(r_i, rv)[0, 1])
    return {"spearman_idx_y_med": float(np.median(ry)) if ry else None,
            "spearman_idx_x_med": float(np.median(rx)) if rx else None, "n_buckets": len(ry)}


def main():
    rng = np.random.default_rng(0)
    g = packets("zgz_global")
    l = packets("zgz_local")
    res = {"order_bias_global": order_bias(g), "order_bias_local": order_bias(l),
           "n_events_med": {"global": float(np.median([p.shape[0] for p in g])),
                            "local": float(np.median([p.shape[0] for p in l]))}}
    for W in (32, 256):
        for f in (1.0, 0.1, 0.03):
            pk = g if f == 1.0 else [p[rng.random(p.shape[0]) < f] for p in g]
            res[f"W{W}_global_keep{f}"] = run_set(pk, W)
        res[f"W{W}_local"] = run_set(l, W)
        print(W, "done", flush=True)
    (OUT / "nbr_window_probe.json").write_text(json.dumps(res, indent=1))
    keys = ["n_nodes", "edge_px", "edge_dt_ms", "frac_edge_dt_ge1ms", "frac_edge_off_gt32",
            "h1_span_ms", "h1_rad_px", "h3_span_ms", "h3_span_ms_p90", "h3_rad_px", "h3_anc"]
    print(json.dumps({k: res[k] for k in ("order_bias_global", "order_bias_local", "n_events_med")}))
    for name, r in res.items():
        if name.startswith("W"):
            print(name, " ".join(f"{k}={r[k]:.3g}" for k in keys if r.get(k) is not None))


if __name__ == "__main__":
    main()
