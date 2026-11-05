#!/usr/bin/env python3
"""PFO unified cost harness (docs/EXPERIMENT_RECORDS.md#pfo-prereg-20261103): the same packets, GPU and rules for every
network, one network x implementation per process; run inside an exclusive block (no other GPU job, no heavy CPU job).

    python -m tools.pfo.cost run --net N [--impl fast|slow|dense|exec] [--spec FILE] [--tag T]

Packets: the first 600 50 ms packets of lyq_local (the packets of every recorded main-row latency) and, for the MAC
distribution, all 2590 zgz endpoints with the GT previous state. Readings:
  params      every parameter the deployed step needs (trunk + router), per module group
  macs        executed multiply-accumulates per packet of the deployed implementation: Linear / Conv layers counted on
              the actual tensors (variable node / record counts), the router's dense trunk once, S43's sparse executor
              on its active sites (its own counter); + MANO FK and the main functional contractions (`macs_functional`)
              reported apart; median / mean / p90 over the packets
  latency     (a) row form: each network's main-row rule (packets pre-staged on the GPU, prev = GT at the run start, 100
              warm-up packets, mean over 600, min of 3 passes, x 1.75 / EventHands-Full anchor); (b) end-to-end: own
              closed loop on the 600 packets from the raw event arrays in host memory (window cut, representation, H2D,
              network, state update, D2H of the state), one synchronised step at a time, p50 / p90 / p99 / max over 3
              passes; (c) per-stage breakdown (synchronised stage timers; descriptive, includes timer overhead)
  memory      peak CUDA allocation during a step above the resident weights, max over the packets
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo.common import NETS, OUT, now, read, require, save, sha  # noqa: E402

N_PACKETS = 600
FK_MACS = None          # filled by `mano_fk_macs`


def mano_fk_macs(mano):
    """Analytic MACs of one MANO forward (shape blend, joint regression, pose blend, LBS, vertex transform)."""
    V = int(mano.v_template.shape[0]) if hasattr(mano, "v_template") else 778
    nb = 10
    J = 16
    return int(V * 3 * nb + J * V * 3 + V * 3 * 135 + V * J * 12 + V * 12 + 21 * V * 3)


class MacCounter:
    """Forward hooks counting Linear / Conv1d / Conv2d multiply-accumulates on the actual tensors."""

    def __init__(self, modules):
        import torch
        self.total = 0
        self.by = {}
        self.handles = []
        for name, m in modules:
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
                self.handles.append(m.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        import torch

        def f(m, inp, out):
            if isinstance(m, torch.nn.Linear):
                n = int(inp[0].numel() // m.in_features) * m.in_features * m.out_features
            elif isinstance(m, torch.nn.Conv2d):
                n = int(out.numel()) * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
            else:
                n = int(out.numel()) * (m.in_channels // m.groups) * m.kernel_size[0]
            self.total += n
            key = name.split(".")[0] if not name.startswith("xyz.") else ".".join(name.split(".")[:2])
            self.by[key] = self.by.get(key, 0) + n
        return f

    def reset(self):
        self.total, self.by = 0, {}

    def close(self):
        for h in self.handles:
            h.remove()


class Stages:
    """Synchronised wall-clock stage timers (inclusive); `add(name)` wraps a callable, `module(name, m)` an nn.Module."""

    def __init__(self):
        self.t = {}
        self.depth = 0
        self.on = False

    @contextmanager
    def time(self, name):
        import torch
        if not self.on:
            yield
            return
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - t0) * 1e3

    def wrap(self, name, fn):
        def g(*a, **k):
            with self.time(name):
                return fn(*a, **k)
        g.__wrapped__ = fn
        return g

    def module(self, name, m):
        import torch
        state = {}

        def pre(mod, inp):
            if self.on:
                torch.cuda.synchronize()
                state["t0"] = time.perf_counter()

        def post(mod, inp, out):
            if self.on:
                torch.cuda.synchronize()
                self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - state["t0"]) * 1e3
        m.register_forward_pre_hook(pre)
        m.register_forward_hook(post)


def lyq_packets(cfg, device):
    """The 600 lyq_local packets of the main-row rule: (end, prev = GT at the run start)."""
    from semkine import eval_track as ET
    from tools.pfo.nets import sequence
    from tools.pfo.engine import sequences
    root, seqs = sequences(cfg, ("lyq_local",))
    ctx = sequence(root, seqs[0][1], "lyq_local", device)
    items = []
    for a, b in np.asarray(ctx.aux["valid_runs_ms"], np.int64).reshape(-1, 2):
        for end in np.arange(a + 49, b, 50, dtype=np.int64):
            items.append((int(end), np.asarray(ctx.gt[int(a)], np.float32).copy(), int(a)))
            if len(items) >= N_PACKETS:
                return ctx, items
    return ctx, items


POLICIES = ("det", "default")


def run(net_name, impl, spec=None, tag=None, policy="det"):
    import torch
    from tools.xcl import evaluate as XE
    from tools.pfo import nets as NM
    from semkine import eval_track as ET
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, range(2, 8))), "one GPU among 2-7")
    require(policy in POLICIES, "policy det | default")
    XE.configure_environment()
    if policy == "det":
        XE.set_policy()                 # the deterministic evaluation flags of every S45 row
    else:
        torch.set_num_threads(4)        # PyTorch's default kernels (S43's recorded rows); same thread count
        torch.set_num_interop_threads(1)
    spec_obj = read(spec) if spec else None
    if spec_obj:
        spec_obj = {**spec_obj, "checkpoint": Path(spec_obj["checkpoint"]), "config": Path(spec_obj["config"])}
    kind = (spec_obj or NETS[net_name])["kind"]
    impl = impl or ("dense" if kind == "s43" else "fast")
    out = OUT / "costs" / (tag or f"{net_name}__{impl}__{policy}")
    require(not out.exists(), "preserve existing cost reading: " + str(out))
    out.mkdir(parents=True)
    device = torch.device("cuda")
    st = Stages()
    if kind != "s43":
        from semkine import s45sm_fast as FAST
        if impl == "slow":
            # the original routing path: s45sm (no graph, sorted event image), same routes
            orig = FAST.install
            FAST.install = lambda *a, **k: __import__("semkine.s45sm", fromlist=["install"]).install(*a, **k)
        net = NM.load(net_name, device=device, spec=spec_obj)
        if impl == "slow":
            FAST.install = orig
    else:
        net = NM.load(net_name, device=device, spec=spec_obj)
    model = net.model
    launch = {"net": net_name, "impl": impl, "policy": policy, "identity": net.identity, "checkpoint_sha256": sha(net.identity["checkpoint"]),
              "cost_sha256": sha(__file__), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_utc": now(),
              "torch": torch.__version__, "device_name": torch.cuda.get_device_name(0)}
    save(out / "launch.json", launch, exclusive=True)
    ctx, items = lyq_packets(net.cfg, device)
    net.bind(ctx)
    mano = model.mano if hasattr(model, "mano") else None
    fk = mano_fk_macs(mano) if mano is not None else 0
    # ---------------------------------------------------------------- params
    router = None
    if kind != "s43":
        from semkine import s45sm as SM
        router = SM._net(device)
    groups = {}
    for name, p in model.named_parameters():
        key = ".".join(name.split(".")[:2]) if name.startswith("xyz.") else name.split(".")[0]
        groups[key] = groups.get(key, 0) + p.numel()
    if router is not None:
        for name, p in router.named_parameters():
            key = "router." + name.split(".")[0]
            groups[key] = groups.get(key, 0) + p.numel()
    params_total = int(sum(groups.values()))
    # ---------------------------------------------------------------- step functions
    ev_ch = None
    exe = None
    if kind == "s43":
        from semkine import events as EV
        from sparse_trunk import SparseExecutor
        ev_ch = EV.event_channels(net.cfg)
        exe = SparseExecutor(model.conv1, model.rn)

    def build(end, prev_np):
        """Host part of a step: window cut + representation input + H2D."""
        prev = torch.from_numpy(prev_np).view(1, -1).to(device)
        if kind == "s43":
            x = torch.from_numpy(ET.build_lnes(ctx.events, ctx.offsets, end, 50, ev_ch)).unsqueeze(0).to(device)
            return (x, prev)
        ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, end, 50)
        return ET.make_eval_packet(ev5, prev, ctx.betas, ctx.K, 50, device)

    counter = {"macs": 0}

    def s43_exec(x, prev):
        rend = model._render_prev(prev.float(), ctx.betas, ctx.K).to(x.dtype)
        y, m, _ = exe.run(torch.cat([x, rend], -1)[0], count_macs=counter.get("count", False))
        counter["macs"] = m
        delta = y[None] + model.prev_mlp(prev.to(y.dtype))
        if x.abs().sum() <= 0:
            delta = torch.zeros_like(delta)
        return delta + prev

    def forward(item):
        if kind == "s43":
            return s43_exec(*item) if impl == "exec" else model(*item)
        return model.forward_packet(item)

    with torch.no_grad():
        # ------------------------------------------------------------ row form (each network's main-row rule)
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
        anchor = XE_anchor(device)
        # ------------------------------------------------------------ end to end (own closed loop, per-step sync)
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
        # ------------------------------------------------------------ memory
        base = torch.cuda.memory_allocated()
        peak = 0
        for it in staged:
            torch.cuda.reset_peak_memory_stats()
            forward(it)
            peak = max(peak, torch.cuda.max_memory_allocated() - base)
        # ------------------------------------------------------------ MACs on the lyq packets and the zgz E3 packets
        macs, macs_by, macs_alt = [], {}, []
        if kind == "s43":
            # executed MACs of the implementation: the sparse executor's active-site count (exec) or the dense layers
            # (dense, masked cuDNN); the other count is kept in `macs_alt` (sparse <-> dense)
            counter["count"] = True
            mc = MacCounter(list(model.named_modules()))
            for it in staged:
                mc.reset()
                s43_exec(*it)
                sparse = counter["macs"] + mc.total
                mc.reset()
                model(*it)
                dense = mc.total
                macs.append(sparse if impl == "exec" else dense)
                macs_alt.append(dense if impl == "exec" else sparse)
            mc.close()
            counter["count"] = False
            trunk_const = 0
        else:
            from semkine import s45sm_fast as FAST
            mc_r = MacCounter([("router." + n, m) for n, m in router.named_modules()])
            img = FAST.event_image(staged[0].events, staged[0].ptr, staged[0].delta_t_s).to(router.stem[0].weight.dtype)
            router.trunk(img)                                                     # eager: the dense trunk, once
            trunk_const = mc_r.total
            mc_r.close()
            mc = MacCounter(list(model.named_modules()) + [("router." + n, m) for n, m in router.named_modules()
                                                           if not n.startswith(("stem", "stages", "lateral", "smooth"))])
            for it in staged:
                mc.reset()
                forward(it)
                macs.append(trunk_const + mc.total)
                for k, v in mc.by.items():
                    macs_by[k] = macs_by.get(k, 0) + v
            mc.close()
            macs_by = {k: v / len(staged) for k, v in macs_by.items()}
            macs_by["router.trunk(dense)"] = trunk_const
        # ------------------------------------------------------------ per-stage breakdown (descriptive)
        breakdown = stage_breakdown(net, model, kind, impl, staged, st, forward, s43_exec, exe)
    e2e, gpu_only, macs = np.asarray(e2e), np.asarray(gpu_only), np.asarray(macs, np.float64)
    n_ev = [int(ctx.offsets[e + 1] - ctx.offsets[e - 49]) for e, _, _ in items]
    pct = lambda a, q: float(np.percentile(a, q))  # noqa: E731
    report = {**launch, "status": "COMPLETE", "finished_utc": now(), "n_packets": len(items),
              "params_total": params_total, "params_by_group": groups,
              "macs_layers": {"median": float(np.median(macs)), "mean": float(macs.mean()), "p90": pct(macs, 90)},
              "macs_by_module_mean": macs_by, "macs_mano_fk": fk,
              "macs_alt_s43": ({"which": "dense" if impl == "exec" else "sparse_active_sites",
                                "median": float(np.median(macs_alt)), "mean": float(np.mean(macs_alt))} if macs_alt else None),
              "events_per_packet": {"mean": float(np.mean(n_ev)), "median": float(np.median(n_ev))},
              "latency_row": {"raw_ms": row_raw, "anchor_ms": anchor, "scaled_ms": row_raw * 1.75 / anchor, "passes": means},
              "latency_e2e_ms": {q: pct(e2e, v) for q, v in (("p50", 50), ("p90", 90), ("p99", 99))} |
                                {"max": float(e2e.max()), "mean": float(e2e.mean())},
              "latency_gpu_step_ms": {q: pct(gpu_only, v) for q, v in (("p50", 50), ("p90", 90), ("p99", 99))} |
                                     {"max": float(gpu_only.max()), "mean": float(gpu_only.mean())},
              "memory_peak_step_mib": peak / 2 ** 20, "memory_weights_mib": base / 2 ** 20,
              "breakdown_ms_mean": breakdown}
    save(out / "report.json", report, exclusive=True)
    print("COMPLETE", out, {k: report[k] for k in ("params_total", "macs_layers", "latency_row", "latency_e2e_ms")},
          flush=True)


def XE_anchor(device):
    from tools.make_s36_row import latency_anchor
    return latency_anchor(device)


def stage_breakdown(net, model, kind, impl, staged, st, forward, s43_exec, exe):
    """Mean synchronised wall time per stage (ms) over the staged packets."""
    import torch
    if kind == "s43":
        orig_render = model._render_prev
        model._render_prev = st.wrap("render_prev", orig_render)
        if impl == "exec":
            exe_run = exe.run
            exe.run = st.wrap("sparse_trunk", exe_run)
            st.module("prev_mlp", model.prev_mlp)
        else:
            model.rn.pooled = st.wrap("trunk", model.rn.pooled)
            st.module("adapter", model.conv1)
            st.module("prev_mlp", model.prev_mlp)
        total_name = "step_total"
    else:
        from semkine import encoder as ENC
        from semkine import s45sm_fast as FAST
        from semkine import xag as XAG
        from semkine import xyz_mesh as XM
        ENC.event_tokens = st.wrap("tokens", ENC.event_tokens)
        FAST.event_image = st.wrap("router_image", FAST.event_image)
        FAST.trunk = st.wrap("router_cnn", FAST.trunk)
        XAG.sample_sources = st.wrap("router_total", XAG.sample_sources)
        FAST.route = st.wrap("route_decode", FAST.route)
        model._fk = st.wrap("mano_fk", model._fk)
        XAG.route_association = st.wrap("pfa", XAG.route_association)
        XM.associate_rays = st.wrap("assoc_empty", XM.associate_rays)
        x = model.xyz
        st.module("xyz_total", x)
        st.module("phi", x.rel)
        st.module("vin", x.vin)
        st.module("edge", x.edge)
        for i, layer in enumerate(x.layers):
            st.module("gnn", layer)
        XM.lbs_slot_pool = st.wrap("slot_pool", XM.lbs_slot_pool)
        st.module("readout", x.slot_proj)
        st.module("readout", x.fc1)
        st.module("readout", x.fc2)
        total_name = "step_total"
    st.on = True
    for it in staged:
        with st.time(total_name):
            forward(it)
    st.on = False
    return {k: v / len(staged) for k, v in st.t.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--net", required=True)
    r.add_argument("--impl", choices=("fast", "slow", "dense", "exec"))
    r.add_argument("--spec")
    r.add_argument("--tag")
    r.add_argument("--policy", choices=POLICIES, default="det")
    a = p.parse_args(argv)
    run(a.net, a.impl, a.spec, a.tag, a.policy)


if __name__ == "__main__":
    main()
