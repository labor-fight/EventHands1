#!/usr/bin/env python3
"""EVF cost for the arms of the second registry (tools/evf/arms2.py, semkine/evf_model2.py; tools/evf/cost.py's CPU
count, frozen with round 1). EVF cost (docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104).

    python -m tools.evf.cost2 macs --arm A            CPU, before any training of A: executed MACs per packet on the
                                                     600 lyq_local packets of the main-row rule and on a full-load
                                                     packet (EVF.MAX_NODES nodes); refuses (exit 3) above 2x S43
    python -m tools.evf.cost2 run --arm A --seed S --phase P --step N [--policy det|default] [--tag T]
                                                     GPU cost reading of a trained arm in an exclusive block: PFO's
                                                     rules (tools/pfo/cost.py: params, MACs on the actual tensors, row
                                                     latency with the x 1.75 / EventHands-Full anchor, end-to-end own
                                                     closed loop p50 / p90 / p99, peak memory), plus the full-load packet
MACs: every Linear layer on its actual input (node counts vary per packet) + MANO FK (analytic, PFO's count) + the
pooling contraction; the rasteriser / distance transforms are comparisons, reported apart as element operations.
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

from tools.evf import arms2 as A  # noqa: E402
from tools.evf.common import OUT, now, read, require, save, sha  # noqa: E402

from semkine.evf_net import N_SLOTS  # noqa: E402

S43_EXEC_MACS = 128.3e6          # outputs/pfo/20261103/costs/s43_s3407__exec__det/report.json, median over the packets
BUDGET = 2.0 * S43_EXEC_MACS
N_PACKETS = 600


class LinearMacs:
    def __init__(self, model):
        import torch
        self.total, self.by = 0, {}
        self.handles = []
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                self.handles.append(m.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def f(m, inp, out):
            n = int(inp[0].numel() // m.in_features) * m.in_features * m.out_features
            self.total += n
            key = ".".join(name.split(".")[:2])
            self.by[key] = self.by.get(key, 0) + n
        return f

    def reset(self):
        self.total, self.by = 0, {}

    def close(self):
        for h in self.handles:
            h.remove()


def build_model(arm, seed=3407, device="cpu"):
    import torch
    from semkine.evf_model2 import EVFModel2 as EVFModel
    torch.manual_seed(seed)
    cfg = A.build_config(arm, seed, "formal")
    cfg["MANO"]["NPZ"] = str(ROOT / cfg["MANO"]["NPZ"]) if not os.path.isabs(cfg["MANO"]["NPZ"]) else cfg["MANO"]["NPZ"]
    return EVFModel(cfg).to(device).eval(), cfg


def lyq_items(cfg, device):
    from tools.pfo.cost import lyq_packets
    return lyq_packets(cfg, device)


def full_load_packet(model, ctx, device):
    """A packet with more than EVF.MAX_NODES nodes in distinct cells (the cap keeps MAX_NODES of them)."""
    import torch
    from semkine import eval_track as ET
    o = model.evf
    H, W, v = model.render_h, model.render_w, int(o["VOXEL_PX"])
    need = int(o["MAX_NODES"]) + 100
    cells = []
    for tb in range(int(o["TIME_BINS"])):
        for p in (0, 1):
            for gy in range(H // v):
                for gx in range(W // v):
                    cells.append((tb, p, gy * v, gx * v))
                    if len(cells) >= need * 4:
                        break
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(cells), need, replace=False))
    dt = 0.05
    rows = [(cells[i][3], cells[i][2], (cells[i][0] + 0.5) / int(o["TIME_BINS"]) * dt, cells[i][1]) for i in pick]
    rows.sort(key=lambda r: r[2])
    ev = np.zeros((len(rows), 5), np.float32)
    ev[:, 1:] = np.asarray(rows, np.float32)
    return ev


def macs(arm):
    import torch
    out = OUT / "costs" / f"macs_{arm}.json"
    require(not out.exists(), "measured once: " + str(out))
    from semkine import eval_track as ET
    from tools.pfo.cost import mano_fk_macs
    model, cfg = build_model(arm)
    device = torch.device("cpu")
    ctx, items = lyq_items(cfg, device)
    model.set_hand_context(ctx.betas, ctx.K)
    counter = LinearMacs(model)
    fk = mano_fk_macs(model.mano)
    per, nodes, by = [], [], {}
    t0 = time.time()
    with torch.no_grad():
        for end, prev_np, _ in items:
            ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, end, 50)
            pk = ET.make_eval_packet(ev5, torch.from_numpy(prev_np).view(1, -1), ctx.betas, ctx.K, 50, device)
            counter.reset()
            model.forward_packet(pk)
            k = int(model.evf_stats["kept"][0])
            pool = k * (N_SLOTS * model.net.val.out_features)          # graph: L0 count, an upper bound
            per.append(counter.total + (fk + pool) * int(model.evf["ITERS"]))
            nodes.append(k)
            for key, v in counter.by.items():
                by[key] = by.get(key, 0) + v
        ev5 = full_load_packet(model, ctx, device)
        pk = ET.make_eval_packet(ev5, torch.from_numpy(items[0][1]).view(1, -1), ctx.betas, ctx.K, 50, device)
        counter.reset()
        model.forward_packet(pk)
        kf = int(model.evf_stats["kept"][0])
        full = counter.total + (fk + kf * N_SLOTS * model.net.val.out_features) * int(model.evf["ITERS"])
    counter.close()
    per = np.asarray(per, np.float64)
    params = int(sum(p.numel() for p in model.net.parameters()))
    rep = {"arm": arm, "evf": model.evf, "params_net": params, "n_packets": len(per),
           "macs": {"median": float(np.median(per)), "mean": float(per.mean()), "p90": float(np.percentile(per, 90)),
                    "max": float(per.max())},
           "macs_full_load": float(full), "nodes_full_load": kf,
           "nodes": {"median": float(np.median(nodes)), "p90": float(np.percentile(nodes, 90)), "max": int(max(nodes))},
           "macs_by_module_mean": {k: v / len(per) for k, v in by.items()}, "macs_mano_fk": fk,
           "s43_exec_macs": S43_EXEC_MACS, "budget_2x": BUDGET,
           "ratio_median": float(np.median(per)) / S43_EXEC_MACS, "ratio_full_load": float(full) / S43_EXEC_MACS,
           "within_budget": bool(np.median(per) <= BUDGET and full <= BUDGET), "cpu_seconds": time.time() - t0,
           "measured_at_utc": now()}
    save(out, rep, exclusive=True)
    print("MACs median %.1f M (x%.2f S43), full load %.1f M (x%.2f), nodes median %d, params %d" % (
        rep["macs"]["median"] / 1e6, rep["ratio_median"], full / 1e6, rep["ratio_full_load"], rep["nodes"]["median"],
        params), "WITHIN" if rep["within_budget"] else "OVER BUDGET", flush=True)
    sys.exit(0 if rep["within_budget"] else 3)


def _spec_model(arm, seed, phase, step, device):
    """An accepted EVF run's checkpoint (either registry) loaded like the readout engine loads it."""
    from tools.evf import run as R1
    from tools.evf import run2 as R2
    from tools.evf import nets2 as N2
    from tools.evf import arms as A1
    reg = R2 if arm in A.ARMS else R1
    require(arm in A.ARMS or arm in A1.ARMS, "unregistered EVF arm " + arm)
    tag, sp = reg.arm_spec(arm, seed, phase, step)
    net = N2.load_evf(tag, {**read(sp), "checkpoint": Path(read(sp)["checkpoint"]), "config": Path(read(sp)["config"])},
                      device)
    return tag, sp, net


def run(arm, seed, phase, step, policy, tag=None):
    """GPU cost reading of a trained EVF arm (exclusive block): tools/pfo/cost.py's rules (see the module docstring)."""
    import torch
    from tools.xcl import evaluate as XE
    from semkine import eval_track as ET
    from tools.pfo.cost import POLICIES, XE_anchor, mano_fk_macs
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, range(2, 8))), "one GPU among 2-7")
    require(policy in POLICIES, "policy det | default")
    XE.configure_environment()
    if policy == "det":
        XE.set_policy()
    else:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
    device = torch.device("cuda")
    rtag, sp, net = _spec_model(arm, seed, phase, step, device)
    out = OUT / "costs" / (tag or f"final__{rtag}__{policy}")
    require(not out.exists(), "preserve existing cost reading: " + str(out))
    out.mkdir(parents=True)
    model = net.model
    launch = {"arm": arm, "seed": seed, "phase": phase, "step": step, "policy": policy, "spec": str(sp),
              "checkpoint_sha256": sha(read(sp)["checkpoint"]), "cost_sha256": sha(__file__),
              "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_utc": now(), "torch": torch.__version__,
              "device_name": torch.cuda.get_device_name(0)}
    save(out / "launch.json", launch, exclusive=True)
    ctx, items = lyq_items(net.cfg, device)
    net.bind(ctx)
    fk = mano_fk_macs(model.mano)
    params = int(sum(p.numel() for p in model.parameters()))

    def build(end, prev_np):
        prev = torch.from_numpy(prev_np).view(1, -1).to(device)
        ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, end, 50)
        return ET.make_eval_packet(ev5, prev, ctx.betas, ctx.K, 50, device)

    with torch.no_grad():
        staged = [build(e, p) for e, p, _ in items]
        for it in staged[:100]:
            model.forward_packet(it)
        torch.cuda.synchronize()
        means = []
        for _ in range(3):
            t0 = time.perf_counter()
            for it in staged:
                model.forward_packet(it)
            torch.cuda.synchronize()
            means.append((time.perf_counter() - t0) / len(staged) * 1e3)
        row_raw = min(means)
        anchor = XE_anchor(device)
        e2e, gpu_only = [], []
        for _ in range(3):
            prev_np, last_a = None, None
            for end, gt_a, a in items:
                if a != last_a:
                    prev_np, last_a = gt_a, a
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                it = build(end, prev_np)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                y = model.forward_packet(it)
                prev_np = y.detach().cpu().numpy()[0].copy()
                t2 = time.perf_counter()
                e2e.append((t2 - t0) * 1e3)
                gpu_only.append((t2 - t1) * 1e3)
        base = torch.cuda.memory_allocated()
        peak = 0
        for it in staged:
            torch.cuda.reset_peak_memory_stats()
            model.forward_packet(it)
            peak = max(peak, torch.cuda.max_memory_allocated() - base)
        counter = LinearMacs(model)
        macs, nodes = [], []
        for it in staged:
            counter.reset()
            model.forward_packet(it)
            k = int(model.evf_stats["kept"][0])
            nodes.append(k)
            macs.append(counter.total + (fk + k * N_SLOTS * model.net.val.out_features) * int(model.evf["ITERS"]))
        ev5 = full_load_packet(model, ctx, device)
        pk = ET.make_eval_packet(ev5, torch.from_numpy(items[0][1]).view(1, -1).to(device), ctx.betas, ctx.K, 50, device)
        counter.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.forward_packet(pk)
        torch.cuda.synchronize()
        full_ms = (time.perf_counter() - t0) * 1e3
        kf = int(model.evf_stats["kept"][0])
        full = counter.total + (fk + kf * N_SLOTS * model.net.val.out_features) * int(model.evf["ITERS"])
        counter.close()
    e2e, gpu_only, macs = np.asarray(e2e), np.asarray(gpu_only), np.asarray(macs, np.float64)
    pct = lambda a, q: float(np.percentile(a, q))  # noqa: E731
    report = {**launch, "status": "COMPLETE", "finished_utc": now(), "n_packets": len(items), "params_total": params,
              "macs_layers": {"median": float(np.median(macs)), "mean": float(macs.mean()), "p90": pct(macs, 90)},
              "macs_full_load": float(full), "nodes_full_load": kf, "latency_full_load_ms": full_ms,
              "nodes": {"median": float(np.median(nodes)), "max": int(max(nodes))}, "macs_mano_fk": fk,
              "latency_row": {"raw_ms": row_raw, "anchor_ms": anchor, "scaled_ms": row_raw * 1.75 / anchor,
                              "passes": means},
              "latency_e2e_ms": {q: pct(e2e, v) for q, v in (("p50", 50), ("p90", 90), ("p99", 99))} |
                                {"max": float(e2e.max()), "mean": float(e2e.mean())},
              "latency_gpu_step_ms": {q: pct(gpu_only, v) for q, v in (("p50", 50), ("p90", 90), ("p99", 99))} |
                                     {"max": float(gpu_only.max()), "mean": float(gpu_only.mean())},
              "memory_peak_step_mib": peak / 2 ** 20, "memory_weights_mib": base / 2 ** 20}
    save(out / "report.json", report, exclusive=True)
    print("COMPLETE", out, {k: report[k] for k in ("params_total", "macs_layers", "macs_full_load", "latency_row",
                                                   "latency_e2e_ms")}, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    m = sub.add_parser("macs")
    m.add_argument("--arm", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True)
    r.add_argument("--seed", type=int, default=3407)
    r.add_argument("--phase", default="formal")
    r.add_argument("--step", type=int, default=6000)
    r.add_argument("--policy", default="det")
    r.add_argument("--tag")
    a = p.parse_args(argv)
    if a.verb == "macs":
        macs(a.arm)
    else:
        run(a.arm, a.seed, a.phase, a.step, a.policy, a.tag)


if __name__ == "__main__":
    main()
