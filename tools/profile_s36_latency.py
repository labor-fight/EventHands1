#!/usr/bin/env python3
"""Stage-level latency breakdown of S36 `forward_packet` on real 50 ms packets.

Same packets and machine as `tools/make_s36_row.py` (600 packets of lyq_local, batch=1,
min of 3 pass-means after warmup), but each stage of the step is timed on its own with
cached inputs, so the sum of stages can be checked against the whole. A batched-32
encoder probe separates "the math is expensive" from "2048 nodes at batch=1 cannot
fill the GPU": if per-packet time collapses under batching, the cost is launch/serial
overhead, not FLOPs.

    CUDA_VISIBLE_DEVICES=0 python tools/profile_s36_latency.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from config import load_config                       # noqa: E402
from model import MNISTModel                         # noqa: E402
from semkine import eval_track as ET                 # noqa: E402
from semkine.dataset import sequences_for_split, splits_manifest      # noqa: E402
from semkine.encoder import event_tokens, query_render   # noqa: E402
from semkine.events import EV_X, EV_Y, EventPacketBatch  # noqa: E402
from semkine.event_gnn import T_COL                  # noqa: E402

STEP_MS = 50


def build_packets(cfg, root, seqs, device, max_packets=600):
    seq, d = next((s, d) for s, d in seqs if s == "lyq_local")
    events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
    tsub_p = root / d / f"{seq}_tsub.npy"
    tsub = np.load(tsub_p, mmap_mode="r") if tsub_p.exists() else None
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    packets = []
    for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
        prev = torch.from_numpy(pos51[a].copy()).view(1, -1).to(device)
        for end in np.arange(a + STEP_MS - 1, b, STEP_MS, dtype=np.int64):
            ev5 = ET._window_events(events, offsets, tsub, int(end), STEP_MS)
            packets.append(ET.make_eval_packet(ev5, prev, betas, K, STEP_MS, device))
            if len(packets) >= max_packets:
                return packets, betas, K
    return packets, betas, K


def timeit(fn, items, passes=3, warm=100):
    for it in items[:warm]:
        fn(it)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(passes):
        t0 = time.perf_counter()
        for it in items:
            fn(it)
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / len(items) * 1e3)
    return best


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda")
    run = REPO / "outputs/semkine/s36_eventgnn_s3407"
    sel = json.loads(next(run.glob("selection_val_core_step50*.json")).read_text())["selected"]
    cfg = load_config(json.loads((run / "training_metadata.json").read_text())["config_path"])
    root = Path(cfg["DATA"]["ROOT"])
    seqs = sequences_for_split(root, "val_core", splits_manifest(cfg))   # zgz under the current protocol
    model = MNISTModel.load_from_checkpoint(sel["ckpt"], cfg=cfg,
                                            map_location=device).to(device).eval()
    enc = model.event_encoder
    packets, betas, K = build_packets(cfg, root, seqs, device)
    model.set_hand_context(betas, K)
    n_ev = int(np.mean([int(b.events.shape[0]) for b in packets]))
    print(f"{len(packets)} packets, mean {n_ev} events")

    # ---- stage inputs, cached once so each timing loop runs only its own stage
    def render(b):
        bf, kf = model._resolve_betas_K(b.prev_state, b.betas, b.camera_K)
        rend = model._render_prev(b.prev_state.float(), bf, kf)
        return query_render(rend.to(dtype=b.events.dtype), b.events)

    extras = [render(b) for b in packets]
    toks = [torch.cat([event_tokens(b.events, b.ptr, b.delta_t_s, enc.height, enc.width),
                       e], dim=-1) for b, e in zip(packets, extras)]

    def sample(arg):
        b, tok = arg
        src, mask = enc._sample(b.events, b.ptr)
        flat = src.reshape(-1)
        feat = tok[flat].reshape(1, -1, enc.in_dim) * mask.unsqueeze(-1)
        ev = b.events[flat].reshape(1, -1, b.events.shape[-1])
        p = torch.stack([ev[..., EV_X] / enc.width, ev[..., EV_Y] / enc.height,
                         feat[..., T_COL] * enc.t_scale], dim=-1) * mask.unsqueeze(-1)
        return feat, p, mask

    sampled = [sample(a) for a in zip(packets, toks)]
    graphs = [enc._edges(p.to(torch.float32), m) for _, p, m in sampled]

    def message_pool(arg):
        (feat, _, mask), (idx, dp, emask) = arg
        h = torch.relu(enc.embed(feat)) * mask.unsqueeze(-1)
        for layer in enc.layers:
            h = (h + layer(h, idx, dp.to(h.dtype), emask.to(h.dtype))) * mask.unsqueeze(-1)
        live = mask.sum(1, keepdim=True).clamp_min(1.0).to(h.dtype)
        mean = h.sum(1) / live
        peak = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
        any_node = mask.any(1, keepdim=True).to(h.dtype)
        return enc.proj(torch.cat([mean, peak * any_node], dim=-1)) * any_node

    feats = [message_pool(a) for a in zip(sampled, graphs)]

    def decode(arg):
        b, feat = arg
        out = model._decode_active(feat, b.prev_state)
        out = out + model.prev_mlp(b.prev_state.to(out.dtype))
        empty = (b.counts <= 0).unsqueeze(-1)
        return torch.where(empty, torch.zeros_like(out), out) + b.prev_state.to(out.dtype)

    # ---- timings
    t_all = timeit(model.forward_packet, packets)
    t_render = timeit(render, packets)
    t_enc = timeit(lambda a: enc(a[0].events, a[0].ptr, a[0].delta_t_s, a[1]),
                   list(zip(packets, extras)))
    t_tok = timeit(lambda a: torch.cat(
        [event_tokens(a[0].events, a[0].ptr, a[0].delta_t_s, enc.height, enc.width), a[1]],
        dim=-1), list(zip(packets, extras)))
    t_sample = timeit(sample, list(zip(packets, toks)))
    t_edges = timeit(lambda a: enc._edges(a[1].to(torch.float32), a[2]),
                     [(f, p, m) for f, p, m in sampled])
    t_mp = timeit(message_pool, list(zip(sampled, graphs)))
    t_dec = timeit(decode, list(zip(packets, feats)))

    # ---- batched-32 encoder probe: same math, one packet's worth of nodes x 32
    B = 32
    merged = []
    for i0 in range(0, len(packets) - B + 1, B):
        chunk = packets[i0:i0 + B]
        evs, ptr, off = [], [0], 0
        for b in chunk:
            evs.append(b.events)
            off += int(b.events.shape[0])
            ptr.append(off)
        merged.append((torch.cat(evs), torch.tensor(ptr, device=device),
                       torch.full((B,), STEP_MS * 1e-3, device=device),
                       torch.cat([e for e in extras[i0:i0 + B]])))
    t_b32 = timeit(lambda a: enc(*a), merged, warm=len(merged)) / B

    stages = {"full_forward": t_all, "render_prev+query": t_render,
              "encoder_total": t_enc, "  tokens": t_tok, "  sample": t_sample,
              "  knn_edges": t_edges, "  message+pool": t_mp,
              "decode_heads": t_dec, "encoder_batched32_per_packet": t_b32}
    print(f"\n{'stage':<32}{'ms':>8}   % of full")
    for k, v in stages.items():
        print(f"{k:<32}{v:8.3f}   {v / t_all * 100:5.1f}%")
    glue = t_all - t_render - t_enc - t_dec
    print(f"{'(glue / python between stages)':<32}{glue:8.3f}   {glue / t_all * 100:5.1f}%")


if __name__ == "__main__":
    main()
