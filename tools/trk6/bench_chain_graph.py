#!/usr/bin/env python3
"""trk6 / S42: `bench_chain.py` with the static-shape trunk (`conv1` -> ResNet trunk, batch 1) replayed from a CUDA graph.

The deployed step is the same as in bench_chain (host LNES + H2D, previous-state FK + splat eager, trunk, prev_mlp,
prev + delta, the count-gain filter on the device, D2H); only `rn(conv1(x))` is captured once per input shape and replayed
(`torch.cuda.CUDAGraph`), which removes its ~60 kernel launches. Both models get exactly the same treatment. Before timing,
the graphed tracker is run in closed loop over the same steps as the eager one and the outputs are compared (max |diff|
over every step is printed and stored; the timing is only reported if it is <= 1e-4).
    python tools/trk6/bench_chain_graph.py --run-dir R --ckpt step=6000 [--max-steps 800] [--out F]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "model")); sys.path.insert(0, str(REPO / "tools"))
from config import load_config                                        # noqa: E402
from model import MNISTModel                                          # noqa: E402
from semkine import eval_track as ET                                  # noqa: E402
from semkine import events as EV                                      # noqa: E402
from semkine.anchored import AdaptiveFilter                           # noqa: E402
from semkine.dataset import sequences_for_split                       # noqa: E402
from tracking.evalx import find_ckpt                                  # noqa: E402
from trk6.bench_chain import bench, steps_of, summ                    # noqa: E402


class GraphedTrunk(nn.Module):
    """`rn(conv1(x))` replayed from one CUDA graph per input shape (batch 1 inference only)."""

    def __init__(self, conv1, rn):
        super().__init__()
        self.conv1, self.rn = conv1, rn
        self.graphs = {}

    @torch.no_grad()
    def forward(self, x):
        key = tuple(x.shape)
        if key not in self.graphs:
            sin = torch.zeros_like(x)
            sin.copy_(x)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.rn(self.conv1(sin))
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                sout = self.rn(self.conv1(sin))
            self.graphs[key] = (g, sin, sout)
        g, sin, sout = self.graphs[key]
        sin.copy_(x)
        g.replay()
        return sout.clone()


@torch.no_grad()
def closed_loop_outputs(net, cfg, device, window, max_steps):
    root = Path(cfg["DATA"]["ROOT"])
    ev_ch = EV.event_channels(cfg)
    outs = []
    for seq, d in sequences_for_split(root, "val_core"):
        events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
        net.set_hand_context(torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1),
                             torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3))
        rng = np.random.default_rng(0)
        for a, end, first in steps_of(aux, max_steps):
            if first:
                prev = torch.from_numpy(pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)).view(1, -1).to(device)
                net.reset_state(prev)
            x = torch.from_numpy(ET.build_lnes(events, offsets, end, min(window, end + 1 - a), ev_ch)).unsqueeze(0).to(device)
            prev = net(x, prev, n_events=int(offsets[end + 1] - offsets[end - 49]))
            outs.append(prev.double().cpu())
    return torch.cat(outs)


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
    run = Path(a.run_dir)
    cfg = load_config(next(iter(sorted(run.glob("*.yaml")))))
    ck, step, _ = find_ckpt(run, a.ckpt)
    model = MNISTModel.load_from_checkpoint(str(ck), cfg=cfg, map_location=device).to(device).eval()
    spec = AdaptiveFilter.canonical_spec(json.loads(Path(a.filter_spec).read_text()))
    if getattr(model, "root_head_kind", "none") != "none" or getattr(model, "encoder_name", ""):
        raise SystemExit("graphed trunk needs the dense rn(conv1(x)) path")
    eager = closed_loop_outputs(AdaptiveFilter.from_spec(model, spec).to(device).eval(), cfg, device, a.window_ms, a.max_steps)
    model.rn = GraphedTrunk(model.conv1, model.rn)
    model.conv1 = nn.Identity()
    net = AdaptiveFilter.from_spec(model, spec).to(device).eval()
    graphed = closed_loop_outputs(net, cfg, device, a.window_ms, a.max_steps)
    diff = float((eager - graphed).abs().max())
    print(f"{run.name}: closed-loop max |graphed - eager| over {len(eager)} steps = {diff:.3g}")
    if diff > 1e-4:
        raise SystemExit("graphed trunk does not reproduce the eager tracker; no timing reported")
    res = bench(net, cfg, device, a.window_ms, a.max_steps, a.passes)
    out = {"run": run.name, "ckpt": str(ck), "step": step, "window_ms": a.window_ms, "gpu": torch.cuda.get_device_name(),
           "variant": "trunk CUDA graph", "max_abs_diff_vs_eager": diff,
           "per_seq": {k: summ(v) for k, v in res.items()}, "pooled": summ(np.concatenate(list(res.values())))}
    p = out["pooled"]
    print(f"{run.name} step {step} [graphed trunk]: total p50 {p['total']['p50']:.3f} p90 {p['total']['p90']:.3f} "
          f"p99 {p['total']['p99']:.3f} mean {p['total']['mean']:.3f} ms | net+filter p50 {p['net_filter']['p50']:.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
