#!/usr/bin/env python3
"""ACU stage-3 cost block (docs/EXPERIMENT_RECORDS.md#acu-s3-prereg-20261104): every network on the same GPU (4, exclusive
job), the same 600 lyq_local packets and the same rules as PFO's frozen harness (tools/pfo/cost.py), output root
outputs/acu/20261104/costs/<tag>/.

    python -m tools.acu.cost_s3 ref --net s43|b2 --policy default|det
        S43 (sparse executor) and B2 (PFO2 fgs 3407, r_s2d) through tools/pfo/cost.py `run`, unchanged
    python -m tools.acu.cost_s3 run --cell C [--seed S] [--trunk T] --policy default|det
        a stage-2 cell in its deployed causal form: every packet carries its stream history (`eval_batch`), PR routes,
        the cell's association. Readings as tools/pfo/cost.py: parameters (trunk + PR), executed Linear / Conv MACs on the
        actual tensors (PR on its actual nodes) + MANO FK apart, CA's rasterisation / JFA as operation counts (not MACs),
        row-form latency (600 staged packets, 100 warm-up, min of 3 passes, x 1.75 / anchor), end-to-end latency (own loop
        from the raw arrays: window + history cut, H2D, network, D2H), memory; plus the split of a step into its per-event
        stages (PR, tokens, association; what an asynchronous runtime does as events arrive) and the rest (what remains
        at the readout), synchronised stage timers (descriptive)
Output frequency is 20 Hz for every network (50 ms readouts); the buffering delay of an event is the time to the next
readout (mean 25 ms) for all of them and is reported as such, not as computation.
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

from tools.acu.common import OUT, now, read, require, save, sha  # noqa: E402

N_PACKETS = 600
POLICIES = ("default", "det")


def ref(net, policy):
    from tools.pfo import cost as C
    C.OUT = OUT
    if net == "s43":
        C.run("s43_s3407", "exec", None, f"ref__s43__exec__{policy}", policy)
    else:
        from tools.pfo2 import run as R
        from tools.pfo2 import run2 as R2
        name, sp = R2.arm_spec("fgs", 3407, 6000)
        R.prepare(sp)
        C.run(name, "fast", str(sp), f"ref__b2__fast__{policy}", policy)


class Timers:
    def __init__(self):
        self.t = {}

    def wrap(self, name, fn):
        import torch

        def g(*a, **k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            self.t.setdefault(name, []).append((time.perf_counter() - t0) * 1e3)
            return out
        return g


def run(cell, seed, trunk, policy):
    import torch
    from semkine import acu_causal as AC
    from semkine import encoder as ENC
    from semkine import xag as XAG
    from tools.acu import run_s2 as R
    from tools.pfo import cost as C
    from tools.pfo import nets as NM
    from tools.xcl import evaluate as XE
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, range(2, 8))), "one GPU among 2-7")
    require(policy in POLICIES, "policy")
    XE.configure_environment()
    if policy == "det":
        XE.set_policy()
    else:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
    name, sp = R.spec(cell, seed, trunk)
    tag = f"s3__{name}__{policy}"
    out = OUT / "costs" / tag
    require(not out.exists(), "preserve existing cost reading: " + str(out))
    out.mkdir(parents=True)
    R.install(cell, trunk)
    device = torch.device("cuda")
    s = read(sp)
    net = NM.load(name, device=device, spec={**s, "checkpoint": Path(s["checkpoint"]), "config": Path(s["config"])})
    model = net.model
    pr = AC._net(device)
    launch = {"net": name, "cell": cell, "seed": seed, "trunk": trunk, "policy": policy, "identity": net.identity,
              "checkpoint_sha256": sha(s["checkpoint"]), "cost_sha256": sha(__file__), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "started_utc": now(), "torch": torch.__version__, "device_name": torch.cuda.get_device_name(0)}
    save(out / "launch.json", launch, exclusive=True)
    ctx, items = C.lyq_packets(net.cfg, device)
    net.bind(ctx)
    groups = {}
    for n, p in model.named_parameters():
        key = ".".join(n.split(".")[:2]) if n.startswith("xyz.") else n.split(".")[0]
        groups[key] = groups.get(key, 0) + p.numel()
    for n, p in pr.named_parameters():
        groups["pr." + n.split(".")[0]] = groups.get("pr." + n.split(".")[0], 0) + p.numel()
    params_total = int(sum(groups.values()))

    def build(end, prev_np):
        prev = torch.from_numpy(prev_np).view(1, -1).to(device)
        return AC.eval_batch(ctx, int(end), AC.run_start(ctx.aux, int(end)), prev, device)

    def forward(b):
        return model.forward_packet(b)
    with torch.no_grad():
        staged = [build(e, p) for e, p, _ in items]
        for it in staged[:100]:
            forward(it)
        torch.cuda.synchronize()
        means = []
        for _ in range(3):
            t0 = time.perf_counter()
            for it in staged:
                forward(it)
            torch.cuda.synchronize()
            means.append((time.perf_counter() - t0) / len(staged) * 1e3)
        row_raw = min(means)
        anchor = C.XE_anchor(device)
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
                y = forward(it)
                prev_np = y.detach().cpu().numpy()[0].copy()
                t2 = time.perf_counter()
                e2e.append((t2 - t0) * 1e3)
                gpu_only.append((t2 - t1) * 1e3)
        base = torch.cuda.memory_allocated()
        peak = 0
        for it in staged:
            torch.cuda.reset_peak_memory_stats()
            forward(it)
            peak = max(peak, torch.cuda.max_memory_allocated() - base)
        mc = C.MacCounter(list(model.named_modules()) + [("pr." + n, m) for n, m in pr.named_modules()])
        macs, macs_by, nodes, render_ops = [], {}, [], []
        for it in staged:
            mc.reset()
            forward(it)
            macs.append(mc.total)
            for k, v in mc.by.items():
                macs_by[k] = macs_by.get(k, 0) + v
            nodes.append(int(AC.PR.causal_stride(it.ptr, it.k_win())[1].sum()))
            if R.A.CELLS[cell][0] == "ca":
                # rasterisation pixel tests (candidate windows) and JFA reads: 6 groups x 7 passes x 9 reads per pixel
                render_ops.append(6 * 7 * 9 * 180 * 240)
        mc.close()
        macs_by = {k: v / len(staged) for k, v in macs_by.items()}
        # per-event stages vs the readout (synchronised timers on the hooks; descriptive)
        tm = Timers()
        native = {"sources": XAG.sample_sources, "tokens": ENC.event_tokens, "route": XAG.route_association}
        XAG.sample_sources = tm.wrap("pr_routes", native["sources"])
        ENC.event_tokens = tm.wrap("tokens", native["tokens"])
        XAG.route_association = tm.wrap("association", native["route"])
        total_t = []
        for it in staged[:200]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            forward(it)
            torch.cuda.synchronize()
            total_t.append((time.perf_counter() - t0) * 1e3)
        XAG.sample_sources, ENC.event_tokens, XAG.route_association = native["sources"], native["tokens"], native["route"]
    e2e, gpu_only, macs = np.asarray(e2e), np.asarray(gpu_only), np.asarray(macs, np.float64)
    pct = lambda a, q: float(np.percentile(a, q))  # noqa: E731
    per_event = {k: float(np.mean(v)) for k, v in tm.t.items()}
    split = {"per_event_stages_ms": per_event, "per_event_total_ms": float(sum(per_event.values())),
             "step_total_ms": float(np.mean(total_t)), "readout_rest_ms": float(np.mean(total_t) - sum(per_event.values()))}
    n_ev = [int(ctx.offsets[e + 1] - ctx.offsets[e - 49]) for e, _, _ in items]
    report = {**launch, "status": "COMPLETE", "finished_utc": now(), "n_packets": len(items), "params_total": params_total,
              "params_by_group": groups,
              "macs_layers": {"median": float(np.median(macs)), "mean": float(macs.mean()), "p90": pct(macs, 90)},
              "macs_by_module_mean": macs_by, "macs_mano_fk": C.mano_fk_macs(model.mano),
              "render_ops_per_packet": float(np.mean(render_ops)) if render_ops else 0.0,
              "nodes_per_packet": {"mean": float(np.mean(nodes)), "median": float(np.median(nodes))},
              "events_per_packet": {"mean": float(np.mean(n_ev)), "median": float(np.median(n_ev))},
              "latency_row": {"raw_ms": row_raw, "anchor_ms": anchor, "scaled_ms": row_raw * 1.75 / anchor, "passes": means},
              "latency_e2e_ms": {q: pct(e2e, v) for q, v in (("p50", 50), ("p90", 90), ("p99", 99))} |
                                {"max": float(e2e.max()), "mean": float(e2e.mean())},
              "latency_gpu_step_ms": {q: pct(gpu_only, v) for q, v in (("p50", 50), ("p90", 90), ("p99", 99))} |
                                     {"max": float(gpu_only.max()), "mean": float(gpu_only.mean())},
              "split_ms": split, "buffer_delay_ms_mean": 25.0, "output_hz": 20,
              "memory_peak_step_mib": peak / 2 ** 20, "memory_weights_mib": base / 2 ** 20}
    save(out / "report.json", report, exclusive=True)
    print("COMPLETE", out, {k: report[k] for k in ("params_total", "macs_layers", "latency_row", "latency_e2e_ms", "split_ms")},
          flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("ref")
    r.add_argument("--net", required=True, choices=("s43", "b2"))
    r.add_argument("--policy", required=True, choices=POLICIES)
    c = sub.add_parser("run")
    c.add_argument("--cell", required=True)
    c.add_argument("--seed", type=int, default=3407)
    c.add_argument("--trunk", default="gshare")
    c.add_argument("--policy", required=True, choices=POLICIES)
    a = p.parse_args(argv)
    if a.verb == "ref":
        ref(a.net, a.policy)
    else:
        run(a.cell, a.seed, a.trunk, a.policy)


if __name__ == "__main__":
    main()
