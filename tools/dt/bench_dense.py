#!/usr/bin/env python3
"""DT round formal runtime measurement of DENSE (LNES) arms (docs/DT_RENDER_TRACK_PREREG.md section 4).

`tools/s38/bench.py` times `forward_packet` of the event-graph arms; the dense arms of this round
(ResNet18 / LNES + rendered previous state) have a plain `model(x, prev)` forward, so this is its counterpart with the same
definitions: run it on an idle machine, pinned to physical cores of one NUMA node, e.g.

    CUDA_VISIBLE_DEVICES=7 taskset -c 63-66 python tools/dt/bench_dense.py --arms dt_base=outputs/semkine/dt_base_s3407 \
        dt_w05=outputs/semkine/dt_w05_s3407 --graph --out outputs/dt/reports/bench_dense.json

Per arm, on real 50 ms packets (lyq_local, the sequence every recorded latency used, and zgz_global; timing only):
  forward   batch-1 `model(x, prev)`, back-to-back, one sync at the end of a pass (evalx.latency_model's definition);
            `rounds` interleaved rounds over all arms, median and min, raw and on the 1.75 ms full-model scale
            (evalx.latency_anchor, measured before and after)
  loop      the deployed closed loop: each step's output is read back to the host before the next packet (what
            `evalx.run_sequence` does), per-step mean
  graph     (--graph) the same forward replayed as one CUDA graph (`model/render_fast.py` GraphedForward: sync-free render,
            bit-identical output, batch 1, inference only), forward and loop; a deployment form, reported next to the eager
            numbers and never instead of them (every other arm in the project is measured eager)
  cost      parameters (every nn.Parameter) and thop MACs of the dense forward (evalx.macs_of)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools"), str(REPO / "tools" / "tracking")]
import evalx as EX                                                    # noqa: E402
from config import load_config                                       # noqa: E402
from model import MNISTModel                                         # noqa: E402
from semkine import eval_track as ET                                 # noqa: E402
from semkine import events as EV                                             # noqa: E402

STEP = 50


def staged_items(cfg, dev, d, seq, n):
    """`n` real 50 ms LNES packets spread over the sequence's valid runs, on the GPU, with the sequence's betas / K."""
    root = Path(cfg["DATA"]["ROOT"])
    events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=dev).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=dev).view(1, 3, 3)
    ev_ch = EV.event_channels(cfg)
    ends = [(int(a), int(e)) for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
            for e in np.arange(a + STEP - 1, b, STEP, dtype=np.int64)]
    pick = np.linspace(0, len(ends) - 1, min(n, len(ends))).round().astype(int)
    items = []
    for i in pick:
        a, end = ends[i]
        prev = torch.from_numpy(pos51[end - STEP + 1].copy()).view(1, -1).to(dev)
        x = torch.from_numpy(ET.build_lnes(events, offsets, end, STEP, ev_ch)).unsqueeze(0).to(dev)
        items.append((x, prev))
    return items, betas, K


@torch.no_grad()
def forward_ms(fn, items, warm=50):
    for it in items[:warm]:
        fn(*it)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for it in items:
        fn(*it)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / len(items) * 1e3


@torch.no_grad()
def loop_ms(fn, items, warm=50):
    """Closed loop: the next packet's state is this packet's output, read back to the host each step."""
    prev = items[0][1]
    for x, _ in items[:warm]:
        prev = fn(x, prev)
        prev.cpu()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for x, _ in items:
        prev = fn(x, prev)
        prev.cpu()
    return (time.perf_counter() - t0) / len(items) * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="name=RUN_DIR (last checkpoint) or name=CONFIG.yaml")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--packets", type=int, default=600)
    ap.add_argument("--graph", action="store_true", help="also the CUDA-graph deployment form (needs model/render_fast.py)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    smi = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    env = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "affinity": sorted(os.sched_getaffinity(0)),
           "torch_threads": torch.get_num_threads(), "gpu": torch.cuda.get_device_name(0),
           "other_compute_processes": smi, "load_avg": os.getloadavg(), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(json.dumps(env), flush=True)
    models = {}
    for spec in a.arms:
        name, path = spec.split("=", 1)
        p = Path(path)
        if p.suffix == ".yaml":
            cfg = load_config(p)
            m = MNISTModel(cfg)
        else:
            cfg = load_config(next(iter(sorted(p.glob("*.yaml")))))
            ck, step, _ = EX.find_ckpt(p, "last")
            m = MNISTModel.load_from_checkpoint(str(ck), cfg=cfg, map_location="cpu")
        models[name] = (cfg, m.to(dev).eval())
    seqs = {"lyq_local": ("train", "lyq_local"), "zgz_global": ("val", "zgz_global")}
    items = {}
    for name, (cfg, m) in models.items():
        for tag, (d, s) in seqs.items():
            items[(name, tag)] = staged_items(cfg, dev, d, s, a.packets)
    graphs, graph_equal = {}, {}
    if a.graph:
        from render_fast import GraphedForward
        for name, (cfg, m) in models.items():
            pk, betas, K = items[(name, "lyq_local")]
            gf = GraphedForward(m)
            try:
                gf.capture(pk[0][0], pk[0][1], betas, K)
            except Exception as e:                 # e.g. CAM_PLANES: the standalone replica has a different input layout
                print(f"graph capture not applicable for {name}: {type(e).__name__}", flush=True)
                graph_equal[name] = 0
                continue
            # the standalone graph replays the DEFAULT dense forward: it is only valid for arms whose eager output it
            # reproduces bit for bit (checked here on 40 real packets of both sequences); other arms get no graph row
            m.set_hand_context(betas, K)
            gf.set_context(betas, K)
            same = 0
            with torch.no_grad():
                for x, p in pk[:20]:
                    same += int(torch.equal(m(x, p), gf(x, p)))
            pk2, betas2, K2 = items[(name, "zgz_global")]
            m.set_hand_context(betas2, K2)
            gf.set_context(betas2, K2)
            with torch.no_grad():
                for x, p in pk2[:20]:
                    same += int(torch.equal(m(x, p), gf(x, p)))
            graph_equal[name] = same
            print(f"graph vs eager bit-identical packets for {name}: {same}/40", flush=True)
            if same == 40:
                graphs[name] = gf
    anchor0 = EX.latency_anchor(dev)
    kinds = lambda n: (("eager", "graph") if (a.graph and n in graphs) else ("eager",))                  # noqa: E731
    fwd = {(n, t, k): [] for n in models for t in seqs for k in kinds(n)}
    for r in range(a.rounds):
        for name, (cfg, m) in models.items():
            for tag in seqs:
                pk, betas, K = items[(name, tag)]
                m.set_hand_context(betas, K)
                fwd[(name, tag, "eager")].append(forward_ms(lambda x, p: m(x, p), pk))
                if name in graphs:
                    graphs[name].set_context(betas, K)
                    fwd[(name, tag, "graph")].append(forward_ms(graphs[name], pk))
        print(f"round {r}: " + ", ".join(f"{n}/{t}/{k} {v[-1]:.2f}" for (n, t, k), v in fwd.items()), flush=True)
    loop = {}
    for name, (cfg, m) in models.items():
        for tag in seqs:
            pk, betas, K = items[(name, tag)]
            m.set_hand_context(betas, K)
            loop[(name, tag, "eager")] = float(np.median([loop_ms(lambda x, p: m(x, p), pk) for _ in range(3)]))
            if name in graphs:
                graphs[name].set_context(betas, K)
                loop[(name, tag, "graph")] = float(np.median([loop_ms(graphs[name], pk) for _ in range(3)]))
    anchor1 = EX.latency_anchor(dev)
    anchor = min(anchor0, anchor1)
    res = {"env": env, "anchor_ms": [anchor0, anchor1], "graph": bool(a.graph), "arms": {}}
    for name, (cfg, m) in models.items():
        ent = {"params": int(sum(p.numel() for p in m.parameters())), "macs": EX.macs_of(cfg)}
        for tag in seqs:
            ent[tag] = {}
            for kind in kinds(name):
                v = fwd[(name, tag, kind)]
                ent[tag][kind] = {"forward_raw_median": float(np.median(v)), "forward_raw_min": float(min(v)),
                                  "forward_scaled_median": float(np.median(v)) * 1.75 / anchor,
                                  "loop_raw": loop[(name, tag, kind)], "loop_scaled": loop[(name, tag, kind)] * 1.75 / anchor,
                                  "rounds": v}
        ent["graph_bit_identical_packets_of_40"] = graph_equal.get(name)
        res["arms"][name] = ent
    ref = next(iter(models))
    for name, ent in res["arms"].items():
        ent["ratio_to_" + ref] = {t: {k: ent[t][k]["forward_raw_median"] / res["arms"][ref][t][k]["forward_raw_median"]
                                      for k in ent[t] if k in res["arms"][ref][t]} for t in seqs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for name, ent in res["arms"].items():
        print(f"{name}: params {ent['params'] / 1e6:.3f} M, MACs {ent['macs'] / 1e9:.3f} G; " + "; ".join(
            f"{t}/{k}: fwd {ent[t][k]['forward_raw_median']:.2f} ms raw ({ent[t][k]['forward_scaled_median']:.2f} scaled, "
            f"x{ent['ratio_to_' + ref][t][k]:.3f}), loop {ent[t][k]['loop_raw']:.2f} ms" for t in seqs for k in ent[t]), flush=True)


if __name__ == "__main__":
    main()
