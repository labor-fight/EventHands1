#!/usr/bin/env python3
"""x1001 latency acceptance: single stream, real-time replay, full processing chain.

Packets arrive every `--arrival-ms` (5 ms) on a wall clock; each one asks for the pose at its
arrival time from the trailing `--window-ms` (50 ms) of real events. One worker serves them FIFO.
Per packet the timed chain is: slice the window from the event store (memmap, as a ring buffer
would hold it) -> build the model input on the CPU (raw-event packet, or LNES) -> host-to-device
-> forward (including any state-dependent rendering / FK the model does) -> device-to-host of the
51D output. Latency = completion - arrival. Reported: p50 / p95 / p99, deadline (`--deadline-ms`,
7 ms) miss rate, max queue length (backlog); and, for a latest-only policy (when the worker frees
up it takes the newest packet and drops older queued ones), the drop rate and its latencies.

Run on an otherwise idle machine, pinned to 4 physical cores of the GPU's NUMA node:
  CUDA_VISIBLE_DEVICES=0 taskset -c 0-3 python tools/x1001/latency_bench.py --run-dir RUN [--seq zgz_global]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))
from config import load_config                                        # noqa: E402
from model import MNISTModel                                          # noqa: E402
from semkine import eval_track as ET                                  # noqa: E402
from semkine import events as EV                                      # noqa: E402

DATA = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51")


def load(run_dir: Path, ckpt: str, device):
    cfg = load_config(next(iter(sorted(run_dir.glob("*.yaml")))))
    if ckpt == "selected":
        sel = json.loads(sorted(run_dir.glob("selection_val_core_step50*.json"))[0].read_text())["selected"]
        path = sel["ckpt"]
    else:
        path = str(next(run_dir.glob(f"*-step={ckpt}.ckpt")))
    m = MNISTModel.load_from_checkpoint(path, cfg=cfg, map_location=device).to(device).eval()
    return m, cfg, path


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="selected")
    ap.add_argument("--seq", default="zgz_global")
    ap.add_argument("--arrival-ms", type=float, default=5.0)
    ap.add_argument("--deadline-ms", type=float, default=7.0)
    ap.add_argument("--window-ms", type=int, default=50)
    ap.add_argument("--window-mode", choices=("fixed", "adaptive"), default="fixed",
                    help="adaptive: shortest window >= --window-ms holding >= --min-events, <= --max-window-ms "
                         "(the rule evalx.py evaluates; the window search is inside the timed chain)")
    ap.add_argument("--min-events", type=int, default=2000)
    ap.add_argument("--max-window-ms", type=int, default=300)
    ap.add_argument("--seconds", type=float, default=20.0, help="replayed stream length")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device("cuda")
    run = Path(a.run_dir)
    model, cfg, ckpt = load(run, a.ckpt, device)
    raw = bool(getattr(model, "encoder_name", ""))
    ev_ch = EV.event_channels(cfg)
    d = "val" if a.seq.startswith("zgz") else "train"
    events, offsets, aux, pos51 = ET.load_sequence(DATA, d, a.seq)
    tsub_p = DATA / d / f"{a.seq}_tsub.npy"
    tsub = np.load(tsub_p, mmap_mode="r") if tsub_p.exists() else None
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    if hasattr(model, "set_hand_context"):
        model.set_hand_context(betas, K)
    a0, b0 = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)[0]
    start_ms = int(a0) + max(a.window_ms, a.max_window_ms if a.window_mode == "adaptive" else 0)
    n = int(a.seconds * 1000 / a.arrival_ms)
    ends = (start_ms + np.arange(n) * a.arrival_ms).astype(np.int64)
    ends = ends[ends < b0]
    prev = torch.from_numpy(pos51[a0].copy()).view(1, -1).to(device)

    def window(end_ms: int) -> int:
        w = int(min(a.window_ms, end_ms + 1))
        if a.window_mode == "adaptive":
            while (w < a.max_window_ms and w < end_ms + 1
                   and offsets[end_ms + 1] - offsets[end_ms - w + 1] < a.min_events):
                w = min(w + 10, a.max_window_ms, end_ms + 1)
        return w

    def process(end_ms: int):
        nonlocal prev
        w = window(int(end_ms))
        if raw:
            ev5 = ET._window_events(events, offsets, tsub, int(end_ms), w)
            out = model.forward_packet(ET.make_eval_packet(ev5, prev, betas, K, w, device))
        else:
            x = torch.from_numpy(ET.build_lnes(events, offsets, int(end_ms), w, ev_ch)).unsqueeze(0).to(device)
            out = model(x, prev)
        prev = out
        return out.cpu()                                            # D2H synchronises

    for e in ends[:50]:                                             # warm-up (not timed)
        process(int(e))
    prev = torch.from_numpy(pos51[a0].copy()).view(1, -1).to(device)
    service = []
    for e in ends:                                                  # isolated service times
        t0 = time.perf_counter()
        process(int(e))
        service.append((time.perf_counter() - t0) * 1e3)
    service = np.asarray(service)

    def replay(latest_only: bool):
        nonlocal prev
        prev = torch.from_numpy(pos51[a0].copy()).view(1, -1).to(device)
        T0 = time.perf_counter() + 0.05
        arrive = T0 + np.arange(len(ends)) * a.arrival_ms / 1e3
        lat, queue_max, dropped, i = [], 0, 0, 0
        while i < len(ends):
            now = time.perf_counter()
            if now < arrive[i]:
                while time.perf_counter() < arrive[i]:
                    pass
                now = time.perf_counter()
            waiting = int(np.searchsorted(arrive, now, side="right")) - i     # arrived, not served
            queue_max = max(queue_max, waiting)
            if latest_only and waiting > 1:
                dropped += waiting - 1
                i += waiting - 1
            process(int(ends[i]))
            lat.append((time.perf_counter() - arrive[i]) * 1e3)
            i += 1
        return np.asarray(lat), queue_max, dropped

    fifo, q_fifo, _ = replay(False)
    lo, q_lo, drop = replay(True)
    pct = lambda x: {f"p{p}": float(np.percentile(x, p)) for p in (50, 95, 99)} | {"max": float(x.max()), "mean": float(x.mean())}  # noqa: E731
    res = {"run": run.name, "ckpt": ckpt, "seq": a.seq, "arrival_ms": a.arrival_ms, "deadline_ms": a.deadline_ms,
           "window_ms": a.window_ms, "window_mode": a.window_mode, "min_events": a.min_events,
           "max_window_ms": a.max_window_ms, "packets": int(len(ends)), "threads": a.threads,
           "gpu": torch.cuda.get_device_name(0), "precision": "fp32",
           "median_events_per_window": float(np.median([offsets[e + 1] - offsets[e - a.window_ms + 1] for e in ends])),
           "service_ms": pct(service), "utilisation": float(service.mean() / a.arrival_ms),
           "fifo": {"latency_ms": pct(fifo), "deadline_miss": float((fifo > a.deadline_ms).mean()), "max_queue": q_fifo},
           "latest_only": {"latency_ms": pct(lo), "deadline_miss": float((lo > a.deadline_ms).mean()),
                           "max_queue": q_lo, "dropped": int(drop), "drop_rate": float(drop / len(ends))}}
    print(json.dumps(res, indent=1))
    out = Path(a.out) if a.out else run / f"latency_{a.seq}_{a.arrival_ms:g}ms_{a.window_mode}.json"
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
