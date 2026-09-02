#!/usr/bin/env python3
"""Produce a main-table row for an `event_gnn`-family tracking run (S36 layout).

Accuracy re-runs the exact selection protocol (`select_checkpoint.py`: val_core of the
5v2v3 manifest, 50 ms recursive steps, rng seed 0, the checkpoint each seed's selection
JSON picked) and aggregates the 8 sequences into the table's local / global columns,
frames-weighted, root-aligned. The reproduced overall RA is asserted against the
selection JSON so a protocol drift cannot go unnoticed.

Cost columns follow `tools/make_main_table.py`: batch-1 latency on this machine scaled
so that EventHands-Full = 1.75 ms, thop MACs for the frontend only (the CNN rows count
conv1+resnet only; heads add ~0.0005 G and are noted), params = every nn.Parameter.
The one necessary deviation: an event frontend's latency is data dependent, so instead
of a fixed random input it replays real 50 ms packets from a val sequence and reports
the mean per step (min over 3 passes); the anchor stays min-over-rounds like the table.

The head-MACs footnote assumes the S36 head layout (ACTIVE_HEAD + PREVPOS_EMBED), which
S37 shares verbatim; the script refuses runs configured otherwise rather than printing a
silently wrong number.

    CUDA_VISIBLE_DEVICES=0 python tools/make_s36_row.py                     # S36
    CUDA_VISIBLE_DEVICES=0 python tools/make_s36_row.py --run s37_fkdirect  # S37
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from config import load_config                                    # noqa: E402
from mano_layer import ManoLayer                                   # noqa: E402
from model import MNISTModel                                      # noqa: E402
from semkine import eval_track as ET                              # noqa: E402
from semkine.dataset import sequences_for_split                   # noqa: E402

SEEDS = (3407, 3408)
MANIFEST = "_retired_splits_semkine_5v2v3.json"
STEP_MS = 50
FULL_PUBLISHED_MS = 1.75
LOCAL = lambda s: "_local" in s          # noqa: E731
METRICS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")


def wmean(rows, key):
    n = sum(r["n_frames"] for r in rows)
    return sum(r[key] * r["n_frames"] for r in rows) / max(n, 1)


def eval_seed(run_name: str, seed: int, device):
    run = REPO / f"outputs/semkine/{run_name}_s{seed}"
    sel = json.loads((run / f"selection_val_core_step{STEP_MS}_"
                            f"_retired_splits_semkine_5v2v3.json").read_text())["selected"]
    cfg = load_config(json.loads((run / "training_metadata.json").read_text())["config_path"])
    root = Path(cfg["DATA"]["ROOT"])
    mani = root / MANIFEST
    seqs = sequences_for_split(root, "val_core", mani)
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    model = MNISTModel.load_from_checkpoint(sel["ckpt"], cfg=cfg,
                                            map_location=device).to(device).eval()
    rng = np.random.default_rng(0)
    res = [ET.track_sequence(model, mano, cfg, root, d, s, STEP_MS, device, rng, 1.0, 1.0, None)
           for s, d in seqs]
    res = [r for r in res if r]
    agg = {"step": sel["step"], "ckpt": sel["ckpt"],
           "overall": {k: wmean(res, k) for k in METRICS},
           "local": {k: wmean([r for r in res if LOCAL(r["seq"])], k) for k in METRICS},
           "global": {k: wmean([r for r in res if not LOCAL(r["seq"])], k) for k in METRICS},
           "per_seq": {r["seq"]: {k: r[k] for k in METRICS} for r in res}}
    drift = abs(agg["overall"]["mpjpe_ra_mm"] - sel["mpjpe_ra_mm"])
    print(f"  s{seed}: step={sel['step']} RA={agg['overall']['mpjpe_ra_mm']:.4f} "
          f"(selection said {sel['mpjpe_ra_mm']:.4f}, drift {drift:.4f} mm)")
    assert drift < 0.05, "re-run does not reproduce the selection protocol"
    del model
    torch.cuda.empty_cache()
    return agg, cfg, seqs, root


@torch.no_grad()
def latency_anchor(device, iters=300, rounds=8):
    m = MNISTModel(load_config(REPO / "configs/eventhands_abs_full51.yaml")).to(device).eval()
    x = torch.rand(1, 180, 240, 2, device=device)
    prev = torch.randn(1, 51, device=device) * 0.02
    prev[:, 2] += 0.4
    for _ in range(iters):
        m(x, prev)
    torch.cuda.synchronize()
    best = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            m(x, prev)
        torch.cuda.synchronize()
        best.append((time.perf_counter() - t0) / iters * 1e3)
    del m
    torch.cuda.empty_cache()
    return min(best)


@torch.no_grad()
def latency_s36(cfg, ckpt, seqs, root, device, max_packets=600, passes=3):
    """Mean forward_packet time over real 50 ms packets of one val sequence, staged on GPU."""
    model = MNISTModel.load_from_checkpoint(ckpt, cfg=cfg, map_location=device).to(device).eval()
    seq, d = next((s, d) for s, d in seqs if s == "lyq_local")
    events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
    tsub_p = root / d / f"{seq}_tsub.npy"
    tsub = np.load(tsub_p, mmap_mode="r") if tsub_p.exists() else None
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    model.set_hand_context(betas, K)
    packets, counts = [], []
    for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
        prev = torch.from_numpy(pos51[a].copy()).view(1, -1).to(device)
        for end in np.arange(a + STEP_MS - 1, b, STEP_MS, dtype=np.int64):
            ev5 = ET._window_events(events, offsets, tsub, int(end), STEP_MS)
            packets.append(ET.make_eval_packet(ev5, prev, betas, K, STEP_MS, device))
            counts.append(len(ev5))
            if len(packets) >= max_packets:
                break
        if len(packets) >= max_packets:
            break
    for b_ in packets[:100]:
        model.forward_packet(b_)
    torch.cuda.synchronize()
    means = []
    for _ in range(passes):
        t0 = time.perf_counter()
        for b_ in packets:
            model.forward_packet(b_)
        torch.cuda.synchronize()
        means.append((time.perf_counter() - t0) / len(packets) * 1e3)
    del model
    torch.cuda.empty_cache()
    return min(means), int(np.mean(counts)), len(packets)


def macs_s36(cfg):
    from thop import profile
    mc = cfg.get("MODEL", {})
    assert bool(mc.get("ACTIVE_HEAD")) and bool(mc.get("PREVPOS_EMBED")), \
        "head-MACs formula below is for the S36/S37 head layout only"
    m = MNISTModel(cfg).eval()
    enc = copy.deepcopy(m.event_encoder)
    n_ev = 4096                               # > max_nodes, so the padded width is saturated
    ev = torch.zeros(n_ev, 5)
    ev[:, 1] = torch.rand(n_ev) * 240
    ev[:, 2] = torch.rand(n_ev) * 180
    ev[:, 3] = torch.linspace(0, 0.05, n_ev)
    ev[:, 4] = (torch.rand(n_ev) > 0.5).float()
    ptr = torch.tensor([0, n_ev])
    dt = torch.tensor([0.05])
    extra = torch.zeros(n_ev, enc.in_dim - 7)
    macs, _ = profile(enc, inputs=(ev, ptr, dt, extra), verbose=False)
    feat = int(mc.get("ENCODER_FEAT", 512))
    hid = int(mc.get("ACTIVE_HIDDEN", 64))
    heads = feat * 6 + 15 * ((feat + 3) * hid + hid * 3) + 2 * 51 * 64
    n_params = sum(p.numel() for p in m.parameters())
    n_enc = sum(p.numel() for p in enc.parameters())
    return macs, heads, n_params, n_enc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="s36_eventgnn",
                    help="run prefix under outputs/semkine/, without the _s<seed> suffix")
    args = ap.parse_args()
    device = torch.device("cuda")
    print(f"== {args.run}: accuracy (selection protocol, val_core 5v2v3, 50 ms recursive) ==")
    per_seed, cfg, seqs, root = {}, None, None, None
    for seed in SEEDS:
        agg, cfg, seqs, root = eval_seed(args.run, seed, device)
        per_seed[seed] = agg

    mean2 = {blk: {k: float(np.mean([per_seed[s][blk][k] for s in SEEDS]))
                   for k in METRICS} for blk in ("overall", "local", "global")}

    print("== cost (same machine, main-table method) ==")
    anchor = latency_anchor(device)
    raw_lat, mean_events, n_pk = latency_s36(cfg, per_seed[SEEDS[0]]["ckpt"], seqs, root, device)
    lat = raw_lat / anchor * FULL_PUBLISHED_MS
    macs, head_macs, n_params, n_enc = macs_s36(cfg)
    print(f"  Full anchor raw {anchor:.3f} ms | {args.run} raw {raw_lat:.3f} ms over {n_pk} packets "
          f"(mean {mean_events} events) -> scaled {lat:.2f} ms")
    print(f"  MACs frontend {macs/1e9:.3f} G (+ heads {head_macs/1e6:.2f} M) | "
          f"params {n_params/1e6:.2f} M (frontend {n_enc/1e6:.2f} M)")

    row = {"per_seed": per_seed, "two_seed_mean": mean2,
           "latency_ms_scaled_full1p75": lat, "latency_ms_raw": raw_lat,
           "anchor_full_raw_ms": anchor, "latency_packets": n_pk,
           "latency_mean_events_per_packet": mean_events,
           "macs_frontend": macs, "macs_heads": head_macs,
           "params_total": n_params, "params_frontend": n_enc,
           "protocol": {"split": "val_core", "manifest": MANIFEST, "step_ms": STEP_MS,
                        "note": "accuracy = two-seed mean of each seed's selected step"}}
    out = REPO / f"outputs/semkine/{args.run}_main_row.json"
    out.write_text(json.dumps(row, indent=2))

    m = mean2
    print("\n| 网络结构 | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global |"
          " RA-MPJPE(递推) | Latency | FLOPs/step | Params |")
    print("|---|---|---|---|---|---|---|---|---|")
    print(f"| {args.run} (双种子均值) | {m['local']['mpjpe_ra_mm']:.2f} "
          f"| {m['global']['mpjpe_ra_mm']:.2f} | {m['local']['mpvpe_ra_mm']:.2f} "
          f"| {m['global']['mpvpe_ra_mm']:.2f} | {m['overall']['mpjpe_ra_mm']:.2f} "
          f"| {lat:.2f} ms | {macs/1e9:.3f} G | {n_params/1e6:.2f} M |")
    for s in SEEDS:
        p = per_seed[s]
        print(f"|   └ s{s} (step {p['step']}) | {p['local']['mpjpe_ra_mm']:.2f} "
              f"| {p['global']['mpjpe_ra_mm']:.2f} | {p['local']['mpvpe_ra_mm']:.2f} "
              f"| {p['global']['mpvpe_ra_mm']:.2f} | {p['overall']['mpjpe_ra_mm']:.2f} "
              f"| — | — | — |")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
