#!/usr/bin/env python3
"""Sub-task E (2026-09-30): batch-1, per-packet, real zgz packets, closed-loop latency of S37 routed.

Stages are timed with torch.cuda.synchronize() between them; the staged forward replays
model.forward_packet's routed branch call by call and is checked bitwise against it. A second
closed-loop pass calls forward_packet directly (one sync after) for the unstaged total. Then the
per-frame CNN (LNES + ResNet18) and the candidate add-on (MLP 768-256-256-9 + SVD->SO(3) + one
rotation interpolation). Queue simulation on the measured per-packet service times.

    CUDA_VISIBLE_DEVICES=7 python .experiments/arch_decision_20260930/E/bench_latency.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (REPO, REPO / "model", REPO / "tools", REPO / ".experiments" / "s37_debug_20260928"):
    sys.path.insert(0, str(_p))
from probe_decomp import STEP, aa_to_R                            # noqa: E402
from probe_rot import R_to_aa                                     # noqa: E402
from probe_s37_route import load_run                              # noqa: E402
from semkine import eval_track as ET                              # noqa: E402
from semkine import events as EV                                  # noqa: E402
from semkine.dataset import sequences_for_split, splits_manifest  # noqa: E402
from semkine.encoder import event_tokens                          # noqa: E402
from semkine.events import EV_X, EV_Y                             # noqa: E402
from semkine.event_gnn import T_COL                               # noqa: E402
from semkine.routed_readout import pool_joint_evidence, route_front_vertex_lbs  # noqa: E402

WARM = 30
sync = torch.cuda.synchronize
now = time.perf_counter


def q(x):
    x = np.asarray(x, np.float64)
    return {"p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "mean": float(x.mean()), "max": float(x.max()), "n": int(len(x))}


class S:
    def __init__(self, root, d, seq, device):
        self.name = seq
        self.events, self.offsets, aux, self.pos51 = ET.load_sequence(root, d, seq)
        p = root / d / f"{seq}_tsub.npy"
        self.tsub = np.load(p, mmap_mode="r") if p.exists() else None
        self.betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
        self.K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
        self.runs = []
        for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
            ends = np.arange(a + STEP - 1, b, STEP, dtype=np.int64)
            if len(ends):
                self.runs.append((int(a), ends))


@torch.no_grad()
def staged_forward(m, batch, t):
    """forward_packet, routed branch, with a sync + timestamp after each stage (appended to t)."""
    enc = m.event_encoder
    events, ptr, prev = batch.events, batch.ptr, batch.prev_state
    B = int(ptr.numel() - 1)
    if events.shape[0] == 0:
        out = m.forward_packet(batch)
        sync(); t.extend([now()] * 7)
        return out
    # 1 graph: token7 + stride sampling + kNN edges
    tok = event_tokens(events, ptr, batch.delta_t_s, enc.height, enc.width)
    src, mask = enc._sample(events, ptr)
    N = mask.shape[1]
    flat = src.reshape(-1)
    feat = tok[flat].reshape(B, N, enc.in_dim) * mask.unsqueeze(-1)
    ev = events[flat].reshape(B, N, -1)
    p = torch.stack([ev[..., EV_X] / enc.width, ev[..., EV_Y] / enc.height,
                     feat[..., T_COL] * enc.t_scale], dim=-1) * mask.unsqueeze(-1)
    idx, dp, emask = enc._edges(p.to(torch.float32), mask)
    sync(); t.append(now())
    # 2 EdgeConv
    h = torch.relu(enc.embed(feat)) * mask.unsqueeze(-1)
    for layer in enc.layers:
        h = (h + layer(h, idx, dp.to(h.dtype), emask.to(h.dtype))) * mask.unsqueeze(-1)
    sync(); t.append(now())
    # 3a packet readout (mean+max+proj) and the return_nodes edge summary
    any_node = mask.any(1, keepdim=True).to(h.dtype)
    live = mask.sum(1, keepdim=True).clamp_min(1.0).to(h.dtype)
    mean = h.sum(1) / live
    peak = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
    fvec = enc.proj(torch.cat([mean, peak * any_node], dim=-1)) * any_node
    n_e = emask.sum(-1, keepdim=True).clamp_min(1.0)
    dp32 = dp.to(torch.float32)
    g = torch.cat([(dp32 * emask.unsqueeze(-1)).sum(2) / n_e,
                   (dp32.norm(dim=-1) * emask).sum(-1, keepdim=True) / n_e], dim=-1) * mask.unsqueeze(-1)
    px = ev[..., EV_X].to(torch.float32) * mask
    py = ev[..., EV_Y].to(torch.float32) * mask
    sync(); t.append(now())
    # 4a routing: MANO FK + projection of prev
    betas_f, k_f = m._resolve_betas_K(prev, batch.betas, batch.camera_K)
    uv, z = m._project_prev(prev, betas_f, k_f)
    sync(); t.append(now())
    # 4b routing: front-vertex LBS dispatch
    a, dist, _ = route_front_vertex_lbs(px, py, mask, uv, z, m.mano.weights, m.route_band_px, m.route_front_k)
    sync(); t.append(now())
    # 3b per-joint pooling
    evidence, count = pool_joint_evidence(h, a, mask)
    livej = mask.sum(1).clamp_min(1).to(torch.float32)
    m.route_stats = {"route_frac_routed": ((a.sum(-1) > 0).sum(1).to(torch.float32) / livej).mean(),
                     "route_joints_hit": (count > 0).sum(1).to(torch.float32).mean()}
    sync(); t.append(now())
    # 5 heads + prev_mlp + delta
    out = m._decode_active(fvec, prev, evidence, root_add=None)
    pm = m.prev_mlp(prev.to(out.dtype))
    if not m.prev_mlp_root:
        pm = torch.cat([torch.zeros_like(pm[:, :6]), pm[:, 6:]], dim=-1)
    out = out + pm
    delta = out
    if m.zero_event_gate:
        empty = (batch.counts <= 0).unsqueeze(-1)
        delta = torch.where(empty, torch.zeros_like(delta), delta)
    out = delta + prev.to(out.dtype)
    sync(); t.append(now())
    return out


STAGES = ["cpu_window_pack_h2d", "graph_token7_sample_knn", "edgeconv", "pool_readout_mean_max_proj",
          "route_fk_project", "route_dispatch_lbs", "pool_joint_evidence", "heads_prevmlp_delta"]


@torch.no_grad()
def closed_loop(m, cfg, s, device, mode, rng):
    """mode 'staged' or 'direct'. Returns per-packet dicts (after WARM packets) and predictions."""
    m.set_hand_context(s.betas, s.K)
    rec, preds, n = [], [], 0
    for a, ends in s.runs:
        prev = torch.from_numpy(s.pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)).view(1, -1).to(device)
        for end in ends:
            sync()
            t = [now()]
            ev5 = ET._window_events(s.events, s.offsets, s.tsub, int(end), STEP)
            batch = ET.make_eval_packet(ev5, prev, s.betas, s.K, STEP, device)
            sync(); t.append(now())
            if mode == "staged":
                pred = staged_forward(m, batch, t)
            else:
                pred = m.forward_packet(batch)
                sync(); t.append(now())
            if n >= WARM:
                d = {"n_ev": int(len(ev5)), "total_ms": (t[-1] - t[0]) * 1e3, "fwd_ms": (t[-1] - t[1]) * 1e3}
                if mode == "staged":
                    for i, k in enumerate(STAGES):
                        d[k] = (t[i + 1] - t[i]) * 1e3
                else:
                    d["cpu_window_pack_h2d"] = (t[1] - t[0]) * 1e3
                rec.append(d)
            preds.append(pred)
            prev = pred
            n += 1
    return rec, torch.cat(preds)


@torch.no_grad()
def cnn_bench(c, ccfg, s, device):
    ch = EV.event_channels(ccfg)
    zero = torch.zeros(1, 51, device=device)
    rec, n = [], 0
    for a, ends in s.runs:
        for end in ends:
            sync(); t0 = now()
            x_np = ET.build_lnes(s.events, s.offsets, int(end), STEP, ch)
            t1 = now()
            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            sync(); t2 = now()
            c(x, zero)
            sync(); t3 = now()
            if n >= WARM:
                rec.append({"cpu_lnes_ms": (t1 - t0) * 1e3, "h2d_ms": (t2 - t1) * 1e3,
                            "resnet_fwd_ms": (t3 - t2) * 1e3, "total_ms": (t3 - t0) * 1e3})
            n += 1
    return rec


def project_so3(M):
    U, _, Vh = torch.linalg.svd(M)
    d = torch.det(U @ Vh)
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    return U @ D @ Vh


@torch.no_grad()
def addon_bench(device, iters=1000):
    torch.manual_seed(0)
    mlp = torch.nn.Sequential(torch.nn.Linear(768, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256),
                              torch.nn.ReLU(), torch.nn.Linear(256, 9)).to(device).eval()
    x = torch.randn(1, 768, device=device)
    prev_aa = torch.randn(1, 3, device=device) * 0.5
    res = {}
    for svd_dev in ("cuda", "cpu"):
        rec = {"mlp": [], "svd_so3": [], "interp": [], "total": []}
        for i in range(iters + WARM):
            sync(); t0 = now()
            M = mlp(x).view(1, 3, 3)
            sync(); t1 = now()
            R = project_so3(M if svd_dev == "cuda" else M.cpu()).to(device)
            sync(); t2 = now()
            Rp = aa_to_R(prev_aa)
            rel = R_to_aa(Rp.transpose(-1, -2) @ R)
            out_aa = R_to_aa(Rp @ aa_to_R(0.5 * rel))
            sync(); t3 = now()
            if i >= WARM:
                for k, v in (("mlp", t1 - t0), ("svd_so3", t2 - t1), ("interp", t3 - t2), ("total", t3 - t0)):
                    rec[k].append(v * 1e3)
        res[f"svd_on_{svd_dev}"] = {k: q(v) for k, v in rec.items()}
        _ = out_aa
    return res


def queue_sim(service_ms, gap_ms):
    S_ = np.asarray(service_ms, np.float64)
    n = len(S_)
    arr = np.arange(n) * gap_ms
    start, fin = np.zeros(n), np.zeros(n)
    f = -1e18
    for i in range(n):
        start[i] = max(arr[i], f)
        fin[i] = start[i] + S_[i]
        f = fin[i]
    e2e = fin - arr
    wait = start - arr
    backlog_end = int(((arr <= arr[-1]) & (fin > arr[-1])).sum())
    # backlog trend: packets waiting (arrived, not started) at each arrival instant
    waiting = np.array([int(((arr[:i + 1]) > -1).sum() - np.searchsorted(np.sort(start), arr[i], side="right"))
                        for i in range(0, n, max(1, n // 20))])
    return {"gap_ms": gap_ms, "n": n, "utilization": float(S_.sum() / (arr[-1] + gap_ms)),
            "rho_mean_service_over_gap": float(S_.mean() / gap_ms),
            "backlog_at_end_pkts": backlog_end, "max_wait_ms": float(wait.max()),
            "queue_len_samples": waiting.tolist(),
            "e2e_ms": q(e2e), "final_lag_ms": float(e2e[-1])}


def main():
    device = torch.device("cuda")
    torch.set_num_threads(8)
    t_all = now()
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "warmup_packets": WARM,
           "note": "fp32 eval, batch 1, closed loop (prev = own output), sync between stages"}
    m, cfg, sel = load_run("outputs/semkine/s37_routed_s3407", device)
    res["s37_ckpt"] = sel["ckpt"]
    assert m.routed and m.predict_delta and m.prevpos_embed and m.active_head
    assert not (m.prev_render or m.root_innovation or m.fk_graph or m.mesh_graph or m.xyz_mesh or m.mesh_query)
    assert m.route_prev_override is None and not m.ablate_evidence
    root = Path(cfg["DATA"]["ROOT"])
    seqs = [S(root, d, s, device) for s, d in sequences_for_split(root, "val_core", splits_manifest(cfg))]
    res["runs_per_seq"] = {s.name: {"n_runs": len(s.runs), "n_packets": int(sum(len(e) for _, e in s.runs)),
                                    "run_len_packets": [int(len(e)) for _, e in s.runs]} for s in seqs}
    # warm the kernels once on the first sequence (discarded)
    closed_loop(m, cfg, seqs[0], device, "direct", np.random.default_rng(0))
    res["s37"] = {}
    series = {}
    for s in seqs:
        torch.cuda.reset_peak_memory_stats()
        base_alloc = torch.cuda.memory_allocated()
        rd, pd = closed_loop(m, cfg, s, device, "direct", np.random.default_rng(0))
        peak_d = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        rs, ps = closed_loop(m, cfg, s, device, "staged", np.random.default_rng(0))
        peak_s = torch.cuda.max_memory_allocated()
        same = bool(torch.equal(pd, ps))
        maxdiff = float((pd - ps).abs().max())
        r = res["s37"][s.name] = {
            "bitwise_equal_staged_vs_direct": same, "max_abs_diff": maxdiff,
            "direct": {"total_incl_cpu": q([x["total_ms"] for x in rd]),
                       "forward_packet": q([x["fwd_ms"] for x in rd]),
                       "cpu_window_pack_h2d": q([x["cpu_window_pack_h2d"] for x in rd]),
                       "events_per_packet": q([x["n_ev"] for x in rd])},
            "staged": {k: q([x[k] for x in rs]) for k in STAGES},
            "staged_forward_sum": q([x["fwd_ms"] for x in rs]),
            "staged_total_incl_cpu": q([x["total_ms"] for x in rs]),
            "peak_mem_alloc_MiB": {"direct": peak_d / 2**20, "staged": peak_s / 2**20,
                                   "model_resident": base_alloc / 2**20},
            "peak_mem_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20,
        }
        series[s.name] = [x["total_ms"] for x in rd]
        print(s.name, json.dumps({"direct_total": r["direct"]["total_incl_cpu"], "fwd": r["direct"]["forward_packet"],
                                  "equal": same}), flush=True)
    del m
    torch.cuda.empty_cache()
    c, ccfg, csel = load_run("outputs/semkine/s37diag_cnnabs_s3407", device)
    res["cnn_ckpt"] = csel["ckpt"]
    res["cnn"] = {}
    cnn_bench(c, ccfg, seqs[0], device)
    for s in seqs:
        torch.cuda.reset_peak_memory_stats()
        rc = cnn_bench(c, ccfg, s, device)
        res["cnn"][s.name] = {k: q([x[k] for x in rc]) for k in rc[0]}
        res["cnn"][s.name]["peak_mem_alloc_MiB"] = torch.cuda.max_memory_allocated() / 2**20
        series[f"cnn:{s.name}"] = [x["total_ms"] for x in rc]
        print("cnn", s.name, json.dumps(res["cnn"][s.name]["total_ms"]), flush=True)
    del c
    torch.cuda.empty_cache()
    res["addon"] = addon_bench(device)
    print("addon", json.dumps({k: v["total"] for k, v in res["addon"].items()}), flush=True)
    res["queue"] = {}
    for name, sv in series.items():
        res["queue"][name] = {str(gp): queue_sim(sv, gp) for gp in (5.0, 10.0, 50.0)}
    res["seconds"] = round(now() - t_all, 1)
    (HERE / "latency_results.json").write_text(json.dumps(res, indent=1))
    (HERE / "service_series_ms.json").write_text(json.dumps(series))
    print("done", res["seconds"], flush=True)


if __name__ == "__main__":
    main()
