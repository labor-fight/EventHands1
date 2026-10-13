#!/usr/bin/env python3
"""trk6 / S42: end-to-end latency of the deployed tracking chain, step by step, on real zgz packets.

One step = what the closed loop of `tools/tracking/evalx.py run_sequence` does at deployment, batch 1, fp32, eager:
  lnes   W-ms LNES of the step (clipped to the segment start) built on the host (`semkine.eval_track.build_lnes`) + H2D
  net    the tracker call `AdaptiveFilter(model)(x, prev, n_events=count50)`: previous-state FK + render (+ crop for
         MODEL.GLIMPSE), trunk, prev_mlp, prev + delta, the event-count adaptive filter on the device
  d2h    the filtered state copied to the host (the consumer's copy); the state fed back stays on the device
Every step ends with a CUDA synchronize. Reported per sequence and pooled: mean / p50 / p90 / p99 (ms), after a
warm-up pass over the same steps. Run it on an idle GPU, one model at a time.
    python tools/trk6/bench_chain.py --run-dir R --ckpt step=6000 [--window-ms 200] [--filter-spec F] [--max-steps 800]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "model")); sys.path.insert(0, str(REPO / "tools"))
from config import load_config                                        # noqa: E402
from model import MNISTModel                                          # noqa: E402
from semkine import eval_track as ET                                  # noqa: E402
from semkine import events as EV                                      # noqa: E402
from semkine.anchored import AdaptiveFilter                           # noqa: E402
from semkine.dataset import sequences_for_split                       # noqa: E402
from tracking.evalx import find_ckpt                                  # noqa: E402

STEP = 50


def steps_of(aux, max_steps):
    out = []
    for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
        ends = np.arange(a + STEP - 1, b, STEP, dtype=np.int64)
        for k, e in enumerate(ends):
            out.append((int(a), int(e), k == 0))
            if len(out) >= max_steps:
                return out
    return out


@torch.no_grad()
def bench(net, cfg, device, window, max_steps, passes):
    root = Path(cfg["DATA"]["ROOT"])
    ev_ch = EV.event_channels(cfg)
    res = {}
    for seq, d in sequences_for_split(root, "val_core"):
        events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
        betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
        K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
        net.set_hand_context(betas, K)
        plan = steps_of(aux, max_steps)
        rng = np.random.default_rng(0)
        times = []
        for p in range(passes + 1):                       # pass 0 = warm-up
            t_l, t_n, t_d = [], [], []
            for a, end, first in plan:
                if first:
                    prev = torch.from_numpy(pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)).view(1, -1).to(device)
                    net.reset_state(prev)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                n_ev = int(offsets[end + 1] - offsets[end - STEP + 1])
                w = min(window, end + 1 - a)
                x = torch.from_numpy(ET.build_lnes(events, offsets, end, w, ev_ch)).unsqueeze(0).to(device)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                pred = net(x, prev, n_events=n_ev)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                _ = pred.cpu()
                t3 = time.perf_counter()
                prev = pred
                t_l.append(t1 - t0); t_n.append(t2 - t1); t_d.append(t3 - t2)
            if p > 0:
                times.append(np.stack([t_l, t_n, t_d], 1) * 1e3)
        res[seq] = np.concatenate(times)
    return res


def summ(a):
    tot = a.sum(1)
    q = lambda v: {"mean": float(v.mean()), "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)),  # noqa: E731
                   "p99": float(np.percentile(v, 99))}
    return {"total": q(tot), "lnes_h2d": q(a[:, 0]), "net_filter": q(a[:, 1]), "d2h": q(a[:, 2]), "n": int(len(a))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--window-ms", type=int, default=200)
    ap.add_argument("--filter-spec", default=str(REPO / "outputs/dt3/gate_specs/ramp1000.json"))
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    run = Path(a.run_dir)
    cfg = load_config(next(iter(sorted(run.glob("*.yaml")))))
    ck, step, _ = find_ckpt(run, a.ckpt)
    model = MNISTModel.load_from_checkpoint(str(ck), cfg=cfg, map_location=device).to(device).eval()
    spec = json.loads(Path(a.filter_spec).read_text())
    net = AdaptiveFilter.from_spec(model, AdaptiveFilter.canonical_spec(spec)).to(device).eval()
    res = bench(net, cfg, device, a.window_ms, a.max_steps, a.passes)
    out = {"run": run.name, "ckpt": str(ck), "step": step, "window_ms": a.window_ms, "gpu": torch.cuda.get_device_name(),
           "params": int(sum(p.numel() for p in model.parameters())),
           "per_seq": {k: summ(v) for k, v in res.items()}, "pooled": summ(np.concatenate(list(res.values())))}
    p = out["pooled"]
    print(f"{run.name} step {step}: total p50 {p['total']['p50']:.3f} p90 {p['total']['p90']:.3f} p99 {p['total']['p99']:.3f} "
          f"mean {p['total']['mean']:.3f} ms | lnes+h2d p50 {p['lnes_h2d']['p50']:.3f} | net+filter p50 "
          f"{p['net_filter']['p50']:.3f} | d2h p50 {p['d2h']['p50']:.3f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
