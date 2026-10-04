#!/usr/bin/env python3
"""DT2 package D: the per-step timing ladder of the deployed tracking loop (zero training).

One step = one 50 ms packet's raw events in host memory -> the filtered 51-D state read back on the host. Times are the
raw wall-clock of that step in ms (no 1.75 / anchor scaling), p50 / p90 / p99 / min, global (zgz_global: dense packets)
and local (zgz_local: sparse packets, a segment restart every ~13 steps) apart, in two paces:

    b2b    back to back: the next packet starts when the previous state is on the host
    hz20   a 20 Hz cadence: 45 ms of idleness (time.sleep) before every packet; the sleep is not in the step time (clocks
           drop, caches go cold: what a live 50 ms stream sees)

The variants, each adding one thing to the previous (all use the dt_dz_l3 checkpoint, batch 1, fp32, recommended filter
spec; `v0`..`v3` filter with `FilteredTracker` on the host, as before DT2, with the gains of the spec evaluated per packet):

    v0   today: host `build_lnes` + H2D + eager forward (original render, 17 host syncs) + host filter + readout
    v1   v0 + `deploy_fast.enable_fast_render` (no host syncs, per-sequence MANO cache)
    v2   v1 + the forward replayed as a CUDA graph (`render_fast.GraphedForward`, the model's own forward)
    v3   v2 + the GPU LNES (`GpuLNES`: pinned events, one H2D, scatter kernels)
    v4   v3 + the GPU filter (`AdaptiveFilter` device path, state resident on the GPU): `DeployTracker(scope="model")`,
         the LNES and the filter are eager kernels around the graph
    v4g  v4 with the filter inside the graph (`scope="filter"`)
    v5   v4g with the GPU LNES inside the graph too (`scope="all"`): one replay per packet

`v0`/`v1` outputs equal each other and `v2`/`v3` bit for bit (same host filter, bit-identical stages); `v4`/`v4g`/`v5`
equal each other bit for bit and the recorded filter_eval of the same spec (tests/test_dt_deploy.py); the run prints the
max |difference| of every variant's output sequence to v0's as a sanity line.

Also (`--latency-row`, default on): `evalx.latency_model` -- the main table's own latency protocol (eager, batch 1, the bare
tracker, staged inputs) -- for the original model and for the model after `enable_fast_render`, with the 1.75 / anchor
scaling of the table, as an optional "eager + no host sync" row. The main table's Latency column is NOT changed by any of
this; the ladder above is the separate "deployment form" row.

The GPU may be shared with training: then the numbers are indicative only (kernels queue behind other processes' kernels).
The official numbers want an idle GPU:

    taskset -c <cores> nice -n 5 python tools/dt/bench_loop.py --run-dir outputs/semkine/dt_dz_l3_s3407 --gpu <2-7> \\
        --cores <cores> --variants v0,v1,v2,v3,v4,v4g,v5 --packets 300 --rounds 5 --tag idle
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ALL_VARIANTS = ("v0", "v1", "v2", "v3", "v4", "v4g", "v5")
DESCR = {
    "v0": "today: host LNES + H2D + eager fwd + host filter",
    "v1": "v0 + fast render (no host syncs, MANO cache)",
    "v2": "v1 + CUDA graph forward",
    "v3": "v2 + GPU LNES",
    "v4": "v3 + GPU filter (outside graph)",
    "v4g": "v4, filter inside graph",
    "v5": "v4g, LNES inside graph too",
}


def _parse_cores(spec: str):
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


# the GPU / CPU restrictions are applied before torch is imported
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=int, default=None)
_pre.add_argument("--cores", default=None)
_known, _ = _pre.parse_known_args()
if _known.gpu is not None:
    if _known.gpu in (0, 1):
        sys.exit("GPU 0 and 1 are not ours: pass --gpu 2..7")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_known.gpu)
if _known.cores:
    os.sched_setaffinity(0, _parse_cores(_known.cores))
    try:
        if os.nice(0) < 5:
            os.nice(5 - os.nice(0))
    except OSError:
        pass
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    sys.exit("set CUDA_VISIBLE_DEVICES (or --gpu N): this tool never touches a GPU it was not told to")

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402

sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "tracking")]
import deploy_fast as DF                                               # noqa: E402
import evalx as EX                                                     # noqa: E402
from config import load_config                                         # noqa: E402
from model import MNISTModel                                           # noqa: E402
from render_fast import GraphedForward                                 # noqa: E402
from semkine import events as EV                                       # noqa: E402
from semkine import eval_track as ET                                   # noqa: E402
from semkine.anchored import FilteredTracker                           # noqa: E402
from semkine.dataset import sequences_for_split                        # noqa: E402

WINDOW = 50
PACE_S = 0.045


# ------------------------------------------------------------------------------------------------ packets
def build_plan(cfg, root, seqs, kind: str, n_packets: int):
    """The first `n_packets` protocol steps of the sequences of `kind` ('global' / 'local'), events already in host memory:
    per step the segment-start flag and initial state, the (N, 3) uint8 events, the cumulative per-ms bounds, and the
    `offsets` slice the host splat takes (`build_lnes`). Betas / K of the first such sequence."""
    rng = np.random.default_rng(0)
    steps, betas, K = [], None, None
    for s, d in seqs:
        if ("_local" in s) != (kind == "local"):
            continue
        events, offsets, aux, pos51 = ET.load_sequence(root, d, s)
        betas, K = aux["betas"], aux["camera_K"]
        for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
            ends = np.arange(a + WINDOW - 1, b, WINDOW, dtype=np.int64)
            if not len(ends):
                continue
            prev0 = (pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)).astype(np.float32)
            for i, end in enumerate(ends):
                a0, a1 = int(offsets[end - WINDOW + 1]), int(offsets[end + 1])
                steps.append({"seg_start": i == 0, "prev0": prev0, "ev": np.ascontiguousarray(events[a0:a1]),
                              "bounds": DF.packet_bounds(offsets, int(end), WINDOW),
                              "off": (np.asarray(offsets[end - WINDOW + 1:end + 2]) - a0).astype(np.int64)})
                if len(steps) >= n_packets:
                    break
            if len(steps) >= n_packets:
                break
        if len(steps) >= n_packets:
            break
    if not steps:
        raise RuntimeError(f"no {kind} packets in {[s for s, _ in seqs]}")
    return {"kind": kind, "betas": betas, "K": K, "steps": steps,
            "events": np.array([len(st["ev"]) for st in steps])}


# ------------------------------------------------------------------------------------------------ variants
class _GraphModel(torch.nn.Module):
    """`FilteredTracker`'s `trk_model` when the forward is a CUDA graph replay."""

    encoder_name = ""

    def __init__(self, gf):
        super().__init__()
        self.gf = gf

    def forward(self, x, prevpos, betas=None, camera_K=None):
        return self.gf(x, prevpos)


class HostVariant:
    """v0..v3: the pre-DT2 loop (`FilteredTracker`: the filter on the host) with the stages swapped in one by one."""

    def __init__(self, name, model, cfg, spec, fast, graph, gpu_lnes, device):
        self.name, self.model, self.device = name, copy.deepcopy(model), device
        self.channels = EV.event_channels(cfg)
        self.gains = DF.HostGains(spec)
        self.fast, self.graph, self.gpu_lnes = fast, graph, gpu_lnes
        if fast:
            DF.enable_fast_render(self.model)
        self.glnes = DF.GpuLNES(device, WINDOW, 180, 240, self.channels) if gpu_lnes else None
        self.gf = GraphedForward(self.model) if graph else None
        trk = _GraphModel(self.gf) if graph else self.model
        self.ft = FilteredTracker(trk, 1.0, 1.0, 1.0)                  # the gains are set per packet (HostGains)
        self.prev_t = None

    def begin_sequence(self, betas, K):
        b = torch.as_tensor(betas, dtype=torch.float32, device=self.device).view(1, -1)
        k = torch.as_tensor(K, dtype=torch.float32, device=self.device).view(1, 3, 3)
        self.ft.set_hand_context(b, k)
        if self.graph:
            if not self.gf.ready:
                x0 = torch.zeros(1, 180, 240, 2 * len(self.channels), device=self.device)
                self.gf.capture(x0, torch.zeros(1, 51, device=self.device), b, k)
            self.gf.set_context(b, k)

    def begin_segment(self, prev0):
        self.prev_t = torch.from_numpy(prev0).view(1, -1).to(self.device)

    def step(self, st):
        ev, bounds = st["ev"], st["bounds"]
        if self.gpu_lnes:
            x = self.glnes.lnes(ev, bounds)
        else:
            x = torch.from_numpy(ET.build_lnes(ev, st["off"], WINDOW - 1, WINDOW, self.channels)).unsqueeze(0).to(self.device)
        self.ft.a_root, self.ft.a_rest, self.ft.a_trans = self.gains(len(ev))
        with torch.no_grad():
            pred = self.ft(x, self.prev_t)
        self.prev_t = pred
        return pred.cpu().numpy()[0]


class DeployVariant:
    """v4 / v4g / v5: `DeployTracker`."""

    def __init__(self, name, model, cfg, spec, scope, device):
        self.name = name
        self.tr = DF.DeployTracker(copy.deepcopy(model), spec, window=WINDOW, channels=EV.event_channels(cfg),
                                   graph=True, scope=scope, device=device)

    def begin_sequence(self, betas, K):
        self.tr.begin_sequence(betas, K)

    def begin_segment(self, prev0):
        self.tr.begin_segment(prev0)

    def step(self, st):
        return self.tr.step_events(st["ev"], st["bounds"])


def make_variant(name, model, cfg, spec, device):
    if name == "v0":
        return HostVariant(name, model, cfg, spec, False, False, False, device)
    if name == "v1":
        return HostVariant(name, model, cfg, spec, True, False, False, device)
    if name == "v2":
        return HostVariant(name, model, cfg, spec, True, True, False, device)
    if name == "v3":
        return HostVariant(name, model, cfg, spec, True, True, True, device)
    return DeployVariant(name, model, cfg, spec, {"v4": "model", "v4g": "filter", "v5": "all"}[name], device)


# ------------------------------------------------------------------------------------------------ measuring
def run_pass(var, plan, pace: bool, keep_out: bool = False):
    var.begin_sequence(plan["betas"], plan["K"])
    ts, outs = [], []
    for st in plan["steps"]:
        if st["seg_start"]:
            var.begin_segment(st["prev0"])
        if pace:
            time.sleep(PACE_S)
        t0 = time.perf_counter()
        out = var.step(st)
        ts.append((time.perf_counter() - t0) * 1e3)
        if keep_out:
            outs.append(out)
    return ts, (np.stack(outs) if keep_out else None)


def stats(ts):
    a = np.asarray(ts)
    return {"n": int(len(a)), "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "min": float(a.min()), "mean": float(a.mean()), "max": float(a.max())}


def count_host_syncs(var, plan, steps: int = 4):
    """Host synchronisations per step, by torch's sync debug mode ("warn": every `.item()`, boolean index, `.cpu()` and
    explicit stream synchronize is one warning). Independent of the load of a shared GPU, unlike the times. A CUDA graph
    replay is not a synchronisation; its one remaining sync is the readout."""
    import warnings
    var.begin_sequence(plan["betas"], plan["K"])
    sts = plan["steps"][:steps + 2]
    var.begin_segment(sts[0]["prev0"])
    var.step(sts[0])                                          # a step outside the count
    torch.cuda.synchronize()
    n = 0
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for st in sts[1:1 + steps]:
                var.step(st)
        n = sum("synchronizing CUDA operation" in str(x.message) for x in w)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    return n / float(steps)


def gpu_state(idx_visible: int = 0):
    try:
        phys = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[idx_visible]
        out = subprocess.run(["nvidia-smi", f"--id={phys}", "--query-gpu=name,utilization.gpu,memory.used,memory.total,clocks.sm",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        return f"GPU {phys}: {out}"
    except Exception as e:                                    # noqa: BLE001
        return f"GPU state unavailable ({e!r})"


def latency_rows(model, cfg, device):
    """evalx.latency_model rows (the main table's protocol): the bare tracker, eager, batch 1 -- as it is and after
    `enable_fast_render`; raw ms, the anchor, and the table's 1.75 / anchor scaling."""
    anchor = EX.latency_anchor(device)
    m0 = copy.deepcopy(model)
    m1 = DF.enable_fast_render(copy.deepcopy(model))
    raw = {"main-table form (original)": [], "eager + no host sync (fast render)": []}
    for _ in range(2):                                         # alternate
        raw["main-table form (original)"].append(EX.latency_model(m0, cfg, device))
        raw["eager + no host sync (fast render)"].append(EX.latency_model(m1, cfg, device))
    rows = {}
    for k, v in raw.items():
        lat = min(v)
        rows[k] = {"raw_ms": lat, "anchor_ms": anchor, "scaled_ms_full1p75": lat * 1.75 / anchor, "all_raw_ms": v}
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="a run directory with config_resolved.yaml and checkpoints")
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--gpu", type=int, default=None, help="physical GPU (2..7); sets CUDA_VISIBLE_DEVICES")
    ap.add_argument("--cores", default=None, help="CPU affinity, e.g. 0-3,72-75 (also nice +5)")
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS), help=f"comma list of {ALL_VARIANTS}")
    ap.add_argument("--packets", type=int, default=300, help="packets per pass, per kind (global / local)")
    ap.add_argument("--rounds", type=int, default=3, help="alternating rounds (every variant once per round)")
    ap.add_argument("--modes", default="b2b,hz20")
    ap.add_argument("--kinds", default="global,local")
    ap.add_argument("--spec", default=None, help="filter spec JSON (default: the DT2 recommendation)")
    ap.add_argument("--latency-row", dest="latency_row", action="store_true", default=True)
    ap.add_argument("--no-latency-row", dest="latency_row", action="store_false")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default=None, help="json path (default outputs/dt2/reports/D_bench_<tag>.json)")
    a = ap.parse_args()
    variants = [v for v in a.variants.split(",") if v]
    bad = [v for v in variants if v not in ALL_VARIANTS]
    if bad:
        sys.exit(f"unknown variants {bad}; known {ALL_VARIANTS}")
    modes, kinds = a.modes.split(","), a.kinds.split(",")
    spec = json.loads(a.spec) if a.spec else DF.RECOMMENDED_SPEC
    torch.set_num_threads(2)
    device = torch.device("cuda")

    run = Path(a.run_dir)
    run = run if run.is_absolute() else REPO / run
    cfg = load_config(run / "config_resolved.yaml")
    ckpt, step, _ = EX.find_ckpt(run, a.ckpt)
    model = MNISTModel.load_from_checkpoint(str(ckpt), cfg=cfg, map_location=device).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    seqs = sequences_for_split(root, "val_core", Path(cfg["DATA"]["SPLITS_MANIFEST"]))
    plans = {k: build_plan(cfg, root, seqs, k, a.packets) for k in kinds}
    for k, p in plans.items():
        e = p["events"]
        print(f"{k}: {len(p['steps'])} packets, events/packet median {int(np.median(e))} p90 {int(np.percentile(e, 90))} max {int(e.max())}",
              flush=True)
    state0 = gpu_state()
    print(state0, f"| affinity {sorted(os.sched_getaffinity(0))}", flush=True)

    vs = {}
    for name in variants:
        vs[name] = make_variant(name, model, cfg, spec, device)
        for k in kinds:                                          # warm-up (graph capture, library init, clocks)
            run_pass(vs[name], {**plans[k], "steps": plans[k]["steps"][:min(60, len(plans[k]["steps"]))]}, False)
    outs = {}
    for name in variants:                                        # sanity: the output sequences (b2b, global then local)
        outs[name] = {k: run_pass(vs[name], plans[k], False, keep_out=True)[1] for k in kinds}
    ref = outs.get("v0")
    print("max |output - v0 output| over the packets (state units; same-stage variants must be 0):", flush=True)
    sanity = {}
    for name in variants:
        if ref is not None:
            sanity[name] = {k: float(np.abs(outs[name][k] - ref[k]).max()) for k in kinds}
            print(f"  {name:4s} {sanity[name]}", flush=True)

    syncs = {name: {k: count_host_syncs(vs[name], plans[k]) for k in kinds} for name in variants}
    print("host syncs per step (torch sync debug mode, warn):", flush=True)
    for name in variants:
        print(f"  {name:4s} {syncs[name]}", flush=True)

    acc = {(v, k, m): [] for v in variants for k in kinds for m in modes}
    for r in range(a.rounds):
        order = variants[r % len(variants):] + variants[:r % len(variants)]
        for name in order:
            for k in kinds:
                for m in modes:
                    ts, _ = run_pass(vs[name], plans[k], m == "hz20")
                    acc[(name, k, m)] += ts
        print(f"round {r + 1}/{a.rounds} done", flush=True)
    state1 = gpu_state()

    table = {}
    for (v, k, m), ts in acc.items():
        table.setdefault(v, {}).setdefault(k, {})[m] = stats(ts)
    lat = latency_rows(model, cfg, device) if a.latency_row else None

    print(f"\nper-step ms (raw), {a.rounds} alternating rounds x {a.packets} packets; step = events in host memory -> 51-D state on the host")
    print(f"{state0}\n{state1}")
    for m in modes:
        print(f"\n[{m}] " + ("back to back" if m == "b2b" else "20 Hz cadence (45 ms idle before each packet)"))
        print("| variant | description | kind | n | p50 | p90 | p99 | min |")
        print("|---|---|---|---|---|---|---|---|")
        for v in variants:
            for k in kinds:
                s = table[v][k][m]
                print(f"| {v} | {DESCR[v]} | {k} | {s['n']} | {s['p50']:.2f} | {s['p90']:.2f} | {s['p99']:.2f} | {s['min']:.2f} |")
    print("\nhost synchronisations per step (sync debug mode): " + ", ".join(
        f"{v} {syncs[v][kinds[0]]:g}" for v in variants))
    if lat:
        print("\nevalx.latency_model (eager, batch 1, bare tracker; the main table's protocol):")
        for k, v in lat.items():
            print(f"  {k}: raw {v['raw_ms']:.3f} ms, anchor {v['anchor_ms']:.3f} ms, scaled (x1.75/anchor) {v['scaled_ms_full1p75']:.3f} ms")

    out = Path(a.out) if a.out else REPO / "outputs" / "dt2" / "reports" / f"D_bench_{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "tag": a.tag, "run": run.name, "ckpt": str(ckpt), "step": step, "spec": spec, "packets": a.packets, "rounds": a.rounds,
        "pace_ms": PACE_S * 1e3, "variants": {v: DESCR[v] for v in variants}, "table": table, "latency_model": lat,
        "sanity_max_abs_vs_v0": sanity, "host_syncs_per_step": syncs, "gpu_before": state0, "gpu_after": state1,
        "affinity": sorted(os.sched_getaffinity(0)), "torch": torch.__version__,
        "events_per_packet": {k: {"median": int(np.median(p["events"])), "p90": int(np.percentile(p["events"], 90)),
                                  "max": int(p["events"].max())} for k, p in plans.items()},
        "command": " ".join(sys.argv)}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
