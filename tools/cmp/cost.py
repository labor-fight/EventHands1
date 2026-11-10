#!/usr/bin/env python3
"""CMP cost block (docs/EXPERIMENT_RECORDS.md#cmp-prereg-20261110): PFO's cost rules (tools/pfo/cost.py) for the four
CMP baselines, and EvHandTrack's own cost tool (tools/evhandtrack/cost.py, unchanged) for S43's sparse executor and
EvHandTrack's streaming form, all written under outputs/cmp/20261110/costs/<tag>/. One GPU among 4-7, nothing else running.

    python -m tools.cmp.cost run --arm eh|emrol5|eg|ep|epc|eho --seed 3407 [--policy default]
    python -m tools.cmp.cost evhandtrack --net s43|stream [--policy default]
    python -m tools.cmp.cost block                 # the whole block in this order: s43, stream, eh, emrol5, eg, ep, epc

Per CMP baseline (packets: the first 600 50 ms packets of lyq_local, prev = GT at the run start; the deployed step of
tools/cmp/nets.py, i.e. the evaluated form):
  params     every parameter of the network (EvHandPose: FlowNet + Backbone)
  macs       executed Linear / Conv multiply-accumulates of one update on the actual tensors (PFO's MacCounter) plus the
             GRU's (3 x (in + hidden) x hidden per recurrent step, analytic); a sliding form re-runs its recurrent part over
             the window at every update, which is what is counted; median / mean / p90 over the packets
  latency    row form: inputs staged on the GPU, 100 warm-up packets, mean over 600, min of 3 passes, x 1.75 / the
             EventHands-Full anchor (tools/make_s36_row.py `latency_anchor`); end to end: own closed loop from the raw
             arrays in host memory (window cut, LNES / voxel grid, H2D, network, D2H), per-step synchronised, 3 passes
  memory     peak CUDA allocation of a step above the resident weights
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.cmp.arms import ARMS  # noqa: E402
from tools.cmp.common import OUT, now, read, require, save, sha  # noqa: E402

POLICIES = ("default", "det")
BLOCK = (("evhandtrack", "s43"), ("evhandtrack", "stream"), ("eh", 3407), ("emrol5", 3407), ("eg", 3407), ("ep", 3407),
         ("epc", 3407))


def policy_flags(policy):
    import torch
    from tools.xcl import evaluate as XE
    XE.configure_environment()
    if policy == "det":
        XE.set_policy()
    else:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)


def run(arm, seed, policy="default", smoke=False):
    import torch
    from tools.pfo import cost as C
    from tools.cmp import nets as CN
    from tools.cmp.arms import SMOKE_STEPS, checkpoint, config_path
    from semkine.cmp_seq import voxel_grid
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, range(2, 8))), "one GPU among 2-7")
    require(policy in POLICIES, "policy default | det")
    policy_flags(policy)
    if smoke:                                   # engineering smoke: smoke checkpoint, 150 packets, costs_smoke/
        ck = checkpoint(arm, seed, SMOKE_STEPS, "smoke")
        rec = {"status": "PASS", "checkpoint": str(ck), "checkpoint_sha256": sha(ck),
               "config": str(config_path(arm, seed, "smoke"))}
        C.N_PACKETS = 150
    else:
        rec = read(OUT / "receipts" / f"cmp_{arm}_s{seed}.json")
    require(rec["status"] == "PASS" and sha(rec["checkpoint"]) == rec["checkpoint_sha256"], "unaccepted checkpoint")
    tag = f"cmp_{arm}_s{seed}__{policy}"
    out = OUT / ("costs_smoke" if smoke else "costs") / tag
    require(not out.exists(), "preserve existing cost reading: " + str(out))
    out.mkdir(parents=True)
    device = torch.device("cuda")
    spec = {"kind": "cmp", "model": ARMS[arm]["model"], "checkpoint": rec["checkpoint"], "config": rec["config"],
            "form": "main"}
    net = CN.load(tag, device=device, spec=spec)
    model, kind = net.model, net.model_kind
    launch = {"arm": arm, "seed": seed, "policy": policy, "identity": net.identity, "checkpoint_sha256": rec["checkpoint_sha256"],
              "cost_sha256": sha(__file__), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_utc": now(),
              "torch": torch.__version__, "device_name": torch.cuda.get_device_name(0)}
    save(out / "launch.json", launch, exclusive=True)
    ctx, items = C.lyq_packets(net.cfg, device)
    net.bind(ctx)
    H, W = int(net.cfg["DATA"]["HEIGHT"]), int(net.cfg["DATA"]["WIDTH"])
    groups = {}
    for name, p in model.named_parameters():
        groups[name.split(".")[0]] = groups.get(name.split(".")[0], 0) + p.numel()
    params_total = int(sum(groups.values()))
    history = net.history if kind in ("eg", "ep") else 1

    def build(end, a, prev=None):
        """Host part of an update: window cut + representation + H2D (epc: + the crop around `prev`, its FK on the GPU)."""
        if kind == "epc":
            x, vox, crop = net.crop_inputs(ctx, end, prev)
            return {"x": x, "vox": vox, "crop": crop, "a": a}
        window = min(net.window, end - a + 1) if kind in ("eh", "eho") else 50
        item = {"x": net.lnes(ctx, end, window), "a": a}
        if kind == "ep":
            xs, ys, ps, us = CN.window_raw(ctx, end, 50)
            item["vox"] = torch.from_numpy(voxel_grid(xs, ys, ps, us, 50, model.voxel_bins, H, W)).unsqueeze(0).to(device)
        return item

    state = {"hist": [], "a": None, "gru_steps": 0}

    def step(item, prev):
        if kind in ("eh", "em"):
            return model(item["x"], prev)
        if kind == "eho":                               # amendment 4: the 12-D output decoded to the 51-D state
            return model.state(model(item["x"], prev))
        if state["a"] != item["a"]:
            state["hist"], state["a"] = [], item["a"]
        if kind == "epc":
            state["hist"] = (state["hist"] + [(model.packet_features(item["x"], item["vox"]), item["crop"])])[-history:]
            return model.sequence_state(torch.stack([f for f, _ in state["hist"]], 1),
                                        torch.stack([c for _, c in state["hist"]], 1))
        f = model.features(item["x"]) if kind == "eg" else model.packet_features(item["x"], item["vox"])
        state["hist"] = (state["hist"] + [f])[-history:]
        feats = torch.stack(state["hist"], 1)
        state["gru_steps"] = feats.shape[1]
        return model.run(feats)[0][:, -1] if kind == "eg" else model.sequence_state(feats)

    def gru_macs():
        if kind != "eg":
            return 0
        g = model.gru
        return 3 * (g.input_size + g.hidden_size) * g.hidden_size * state["gru_steps"]

    with torch.no_grad():
        staged = [(build(e, a, torch.from_numpy(p).view(1, -1).to(device)), torch.from_numpy(p).view(1, -1).to(device))
                  for e, p, a in items]
        for it, prev in staged[:100]:
            step(it, prev)
        torch.cuda.synchronize()
        means = []
        for _ in range(3):
            state["a"] = None
            t0 = time.perf_counter()
            for it, prev in staged:
                step(it, prev)
            torch.cuda.synchronize()
            means.append((time.perf_counter() - t0) / len(staged) * 1e3)
        row_raw = min(means)
        anchor = C.XE_anchor(device)
        e2e, gpu_only = [], []
        for _ in range(3):
            state["a"], prev_np, last_a = None, None, None
            for end, gt_a, a in items:
                if a != last_a:
                    prev_np, last_a = gt_a, a
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                prev = torch.from_numpy(prev_np).view(1, -1).to(device)
                it = build(end, a, prev)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                y = step(it, prev)
                prev_np = y.detach().float().cpu().numpy()[0].copy()
                t2 = time.perf_counter()
                e2e.append((t2 - t0) * 1e3)
                gpu_only.append((t2 - t1) * 1e3)
        state["a"] = None
        base = torch.cuda.memory_allocated()
        peak = 0
        for it, prev in staged:
            torch.cuda.reset_peak_memory_stats()
            step(it, prev)
            peak = max(peak, torch.cuda.max_memory_allocated() - base)
        state["a"] = None
        mc = C.MacCounter(list(model.named_modules()))
        macs, macs_by = [], {}
        for it, prev in staged:
            mc.reset()
            step(it, prev)
            macs.append(mc.total + gru_macs())
            for k, v in mc.by.items():
                macs_by[k] = macs_by.get(k, 0) + v
        mc.close()
    macs_by = {k: v / len(staged) for k, v in macs_by.items()}
    if kind == "eg":
        macs_by["gru(analytic)"] = 3 * (model.gru.input_size + model.gru.hidden_size) * model.gru.hidden_size * history
    e2e, gpu_only, macs = np.asarray(e2e), np.asarray(gpu_only), np.asarray(macs, np.float64)
    n_ev = [int(ctx.offsets[e + 1] - ctx.offsets[e - 49]) for e, _, _ in items]
    pct = lambda a, q: float(np.percentile(a, q))  # noqa: E731
    report = {**launch, "status": "COMPLETE", "finished_utc": now(), "n_packets": len(items), "history_packets": history,
              "params_total": params_total, "params_by_group": groups,
              "macs_layers": {"median": float(np.median(macs)), "mean": float(macs.mean()), "p90": pct(macs, 90)},
              "macs_by_module_mean": macs_by, "macs_mano_fk": 0,
              "events_per_packet": {"mean": float(np.mean(n_ev)), "median": float(np.median(n_ev))},
              "latency_row": {"raw_ms": row_raw, "anchor_ms": anchor, "scaled_ms": row_raw * 1.75 / anchor, "passes": means},
              "latency_e2e_ms": {q: pct(e2e, v) for q, v in (("p50", 50), ("p90", 90), ("p99", 99))} |
                                {"max": float(e2e.max()), "mean": float(e2e.mean())},
              "latency_gpu_step_ms": {q: pct(gpu_only, v) for q, v in (("p50", 50), ("p90", 90), ("p99", 99))} |
                                     {"max": float(gpu_only.max()), "mean": float(gpu_only.mean())},
              "memory_peak_step_mib": peak / 2 ** 20, "memory_weights_mib": base / 2 ** 20}
    save(out / "report.json", report, exclusive=True)
    print("COMPLETE", out, {k: report[k] for k in ("params_total", "macs_layers", "latency_row", "latency_e2e_ms")},
          flush=True)


def evhandtrack(net, policy="default"):
    """tools/evhandtrack/cost.py unchanged, output root moved to the CMP study."""
    from tools.evhandtrack import cost as EC
    EC.OUT = OUT
    EC.run(net, policy)


def block(policy="default"):
    """The whole cost block, one reading per subprocess (each reading is one network per process, as the harness wants)."""
    import subprocess
    for a, b in BLOCK:
        args = (["evhandtrack", "--net", b] if a == "evhandtrack" else ["run", "--arm", a, "--seed", str(b)])
        cmd = [sys.executable, "-m", "tools.cmp.cost", *args, "--policy", policy]
        print(now(), " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(ROOT), check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True, choices=sorted(ARMS))
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--policy", choices=POLICIES, default="default")
    r.add_argument("--smoke", action="store_true", help="engineering smoke on the smoke checkpoint (costs_smoke/)")
    e = sub.add_parser("evhandtrack")
    e.add_argument("--net", required=True, choices=("s43", "packet", "stream"))
    e.add_argument("--policy", choices=POLICIES, default="default")
    b = sub.add_parser("block")
    b.add_argument("--policy", choices=POLICIES, default="default")
    a = p.parse_args(argv)
    if a.verb == "run":
        run(a.arm, a.seed, a.policy, a.smoke)
    elif a.verb == "evhandtrack":
        evhandtrack(a.net, a.policy)
    else:
        block(a.policy)


if __name__ == "__main__":
    main()
