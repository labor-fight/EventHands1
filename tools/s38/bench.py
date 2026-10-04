#!/usr/bin/env python3
"""S38 formal runtime measurement (docs/S38_ROOT_TRACKING_PREREG.md section 5): the budget is S37's runtime in the
same environment, measured interleaved, in isolation.

Run on an idle machine, pinned to physical cores of one NUMA node, e.g.
    CUDA_VISIBLE_DEVICES=0 taskset -c 0-3 python tools/s38/bench.py --arms rt_s37=RUN ... --out outputs/s38/reports/bench.json
where RUN is a run directory (its last checkpoint is loaded; weights do not change the cost).

Per arm, on real 50 ms packets (lyq_local, the sequence every recorded latency used, and zgz_global, the
highest event rate of the evaluation subject; timing only):
  forward     batch-1 `forward_packet`, back-to-back, one sync at the end of a pass (evalx.latency_model's
              definition); `rounds` interleaved rounds over all arms, median and min reported, raw and on the
              1.75 ms full-model scale (evalx.latency_anchor, measured before and after)
  loop        the deployed closed loop: each step's output read back to the host before the next packet
              (what `evalx.run_sequence` does), per-step mean
  MACs        thop over `forward_packet` on the standard synthetic packet (make_s36_row, the comparable column)
              and on the real packets (median / p95 / max), Linear / Conv / BN only, as for every arm
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
import make_s36_row as MR                                             # noqa: E402
from config import load_config                                       # noqa: E402
from model import MNISTModel                                         # noqa: E402
from semkine import eval_track as ET                                 # noqa: E402

STEP = 50


def packets(cfg, dev, d, seq, n):
    root = Path(cfg["DATA"]["ROOT"])
    events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
    tsub = np.load(root / d / f"{seq}_tsub.npy", mmap_mode="r")
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=dev).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=dev).view(1, 3, 3)
    # protocol steps of every valid run, `n` of them evenly spread over the sequence (the first seconds of a
    # recording are a near-still hand with few events, and would understate an event-dependent cost)
    ends = [int(e) for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
            for e in np.arange(a + STEP - 1, b, STEP, dtype=np.int64)]
    pick = np.linspace(0, len(ends) - 1, min(n, len(ends))).round().astype(int)
    out = []
    for end in (ends[i] for i in pick):
        prev = torch.from_numpy(pos51[end - STEP + 1].copy()).view(1, -1).to(dev)
        out.append(ET.make_eval_packet(ET._window_events(events, offsets, tsub, end, STEP), prev, betas, K, STEP, dev))
    return out, betas, K


@torch.no_grad()
def forward_ms(model, items):
    for it in items[:50]:
        model.forward_packet(it)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for it in items:
        model.forward_packet(it)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / len(items) * 1e3


@torch.no_grad()
def loop_ms(model, items):
    """Closed loop: the next packet's state is this packet's output, read back to the host each step."""
    prev = items[0].prev_state
    for it in items[:50]:
        it.prev_state = prev
        prev = model.forward_packet(it)
        prev.cpu()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for it in items:
        it.prev_state = prev
        prev = model.forward_packet(it)
        prev.cpu()
    return (time.perf_counter() - t0) / len(items) * 1e3


def macs_real(cfg, items, limit):
    from thop import profile

    class _W(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, batch):
            return self.m.forward_packet(batch)

    m = MNISTModel(cfg).eval()
    w = _W(m)
    out = []
    for it in items[:: max(1, len(items) // limit)][:limit]:
        cpu = it.to("cpu")
        if hasattr(m, "set_hand_context"):
            m.set_hand_context(cpu.betas, cpu.camera_K)
        macs, _ = profile(w, inputs=(cpu,), verbose=False)
        out.append(float(macs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="name=RUN_DIR (last checkpoint) or name=CONFIG.yaml")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--packets", type=int, default=600)
    ap.add_argument("--macs-packets", type=int, default=60)
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
            pk, betas, K = packets(cfg, dev, d, s, a.packets)
            items[(name, tag)] = (pk, betas, K)
    anchor0 = EX.latency_anchor(dev)
    fwd = {(n, t): [] for n in models for t in seqs}
    for r in range(a.rounds):
        for name, (cfg, m) in models.items():
            for tag in seqs:
                pk, betas, K = items[(name, tag)]
                m.set_hand_context(betas, K)
                fwd[(name, tag)].append(forward_ms(m, pk))
        print(f"round {r}: " + ", ".join(f"{n}/{t} {v[-1]:.2f}" for (n, t), v in fwd.items()), flush=True)
    loop = {}
    for name, (cfg, m) in models.items():
        for tag in seqs:
            pk, betas, K = items[(name, tag)]
            m.set_hand_context(betas, K)
            loop[(name, tag)] = float(np.median([loop_ms(m, pk) for _ in range(3)]))
    anchor1 = EX.latency_anchor(dev)
    anchor = min(anchor0, anchor1)
    res = {"env": env, "anchor_ms": [anchor0, anchor1], "arms": {}}
    for name, (cfg, m) in models.items():
        ent = {"params": int(sum(p.numel() for p in m.parameters())),
               "macs_synthetic": MR.macs_forward_packet(cfg)[0]}
        for tag in seqs:
            v = fwd[(name, tag)]
            ent[tag] = {"forward_raw_median": float(np.median(v)), "forward_raw_min": float(min(v)),
                        "forward_scaled_median": float(np.median(v)) * 1.75 / anchor,
                        "loop_raw": loop[(name, tag)], "loop_scaled": loop[(name, tag)] * 1.75 / anchor,
                        "rounds": v}
            mr = macs_real(cfg, items[(name, tag)][0], a.macs_packets)
            ent[tag]["macs_real"] = {"median": float(np.median(mr)), "p95": float(np.percentile(mr, 95)),
                                     "max": float(max(mr))}
        res["arms"][name] = ent
    ref = next(iter(models))
    for name, ent in res["arms"].items():
        ent["ratio_to_" + ref] = {t: ent[t]["forward_raw_median"] / res["arms"][ref][t]["forward_raw_median"]
                                  for t in seqs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for name, ent in res["arms"].items():
        print(f"{name}: params {ent['params'] / 1e6:.3f} M, MACs synthetic {ent['macs_synthetic'] / 1e9:.3f} G; " + "; ".join(
            f"{t}: fwd {ent[t]['forward_raw_median']:.2f} ms raw ({ent[t]['forward_scaled_median']:.2f} scaled, "
            f"x{ent['ratio_to_' + ref][t]:.3f}), loop {ent[t]['loop_raw']:.2f} ms, MACs real median "
            f"{ent[t]['macs_real']['median'] / 1e9:.3f} / p95 {ent[t]['macs_real']['p95'] / 1e9:.3f} G" for t in seqs), flush=True)


if __name__ == "__main__":
    main()
