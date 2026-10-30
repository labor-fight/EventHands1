#!/usr/bin/env python3
"""MV2 M9 / SORL + LBA (P5-E10-R1, P5-E2-R2): zero-data static accounting of the executable form of S39 and S39 + SLR.

Synthetic tensors only (a random sparse event image, a near-mean previous pose, zero betas, the default K) and randomly
initialised models built from their configs; no data file, no checkpoint is read. For each variant

  * a `TorchDispatchMode` counts every aten call (total / view-only / allocation-only / kernel-launching = `launches`),
    the host synchronisations (`aten._local_scalar_dense`, `is_nonzero`, `equal`: .item() / bool(tensor) / float(tensor))
    and the data-dependent-shape ops (nonzero, unique*, masked_select, boolean-mask indexing / index_put);
  * `torch._dynamo.explain` and `torch.compile(fullgraph=True, backend="eager")` count the graph breaks (CPU, trace only);
  * a static source grep of the forward path lists `.item()` / `.cpu()` / `.numpy()` / `nonzero` / `unique` tokens;
  * CPU forward latency (batch 1, no_grad) p50 / p99 at 2 and 8 threads, interleaved model order, one fresh pinned process
    per thread count, with an A/A control (S39 against a deep copy of itself) to size the harness noise.

Variants: S39 = configs/dt/dt_dz_l3.yaml (MNISTModel); S39+rootanc = configs/dt/dt_dz_l3_rootanc_2k.yaml (MNISTModel);
S39+SLR = configs/mv2/mv2_slr.yaml (`semkine.slr.SLRModel`). Controls with known answers (a toy module, a data-dependent toy)
validate the counter and the graph-break detector before any variant reading.

    python tools/mv2/m9_cost.py --smoke        # toy modules only are printed; the variants are exercised silently
    python tools/mv2/m9_cost.py --out outputs/mv2/m9_cost
"""
from __future__ import annotations

import argparse
import collections
import inspect
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mv2 import common as C  # noqa: E402

VARIANTS = {
    "S39": ("configs/dt/dt_dz_l3.yaml", "model.MNISTModel"),
    "S39+rootanc": ("configs/dt/dt_dz_l3_rootanc_2k.yaml", "model.MNISTModel"),
    "S39+SLR": ("configs/mv2/mv2_slr.yaml", "semkine.slr.SLRModel"),
}
LAUNCH_FACTOR = 3.0                 # launches <= 3 x (S39 + rootanc)  (S43 sparse 304 kernels vs dense 216: 1.4x)
CPU8_INCREMENT_MAX = 0.50           # PROPOSED engineering margin, not a ledger threshold (P5-E10-R1)
KILL_INCREMENT = 1.00               # > +100 % => the Latency claim is deleted (plan, M9 SORL)
AA_NOISE_MAX = 0.10                 # |A/A p50 ratio - 1| above this makes the latency gate undetermined
SYNC_OPS = {"aten._local_scalar_dense", "aten.is_nonzero", "aten.equal", "aten.item"}
DDS_OPS = {"aten.nonzero", "aten.unique", "aten._unique", "aten._unique2", "aten.unique_dim", "aten.unique_consecutive",
           "aten.masked_select", "aten.argwhere"}
INDEX_OPS = {"aten.index", "aten.index_put", "aten.index_put_", "aten._index_put_impl_"}
ALLOC_ONLY = {"aten.empty", "aten.empty_strided", "aten.empty_like", "aten.new_empty", "aten.new_empty_strided"}
STATIC_TOKENS = (r"\.item\(", r"\.cpu\(", r"\.numpy\(", r"\.tolist\(", r"\bnonzero\(", r"\bunique\(", r"masked_select",
                 r"torch\.where\(\s*[A-Za-z_0-9\.]+\s*\)", r"\bnp\.")
SEED_INPUT = 20261026


# ----------------------------------------------------------------------------------------------- op counter
class OpCounter(TorchDispatchMode):
    """Counts the aten calls reaching the dispatcher; the op itself runs unchanged."""

    def __init__(self):
        super().__init__()
        self.calls = collections.Counter()
        self.view = 0
        self.alloc = 0
        self.sync = collections.Counter()
        self.dds = collections.Counter()
        self.bool_index = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func.overloadpacket)
        self.calls[name] += 1
        rets = func._schema.returns
        if any(r.alias_info is not None and not r.alias_info.is_write for r in rets):
            self.view += 1
        elif name in ALLOC_ONLY:
            self.alloc += 1
        if name in SYNC_OPS:
            self.sync[name] += 1
        if name in DDS_OPS:
            self.dds[name] += 1
        if name in INDEX_OPS and len(args) > 1 and isinstance(args[1], (list, tuple)):
            if any(isinstance(t, torch.Tensor) and t.dtype in (torch.bool, torch.uint8) for t in args[1]):
                self.bool_index += 1
        return func(*args, **kwargs)

    def summary(self, top: int = 12) -> dict:
        total = int(sum(self.calls.values()))
        return {"total_calls": total, "view_calls": self.view, "alloc_calls": self.alloc,
                "launches": total - self.view - self.alloc,
                "sync_ops": {k: int(v) for k, v in self.sync.items()}, "sync_count": int(sum(self.sync.values())),
                "dds_ops": {k: int(v) for k, v in self.dds.items()}, "bool_index": int(self.bool_index),
                "dds_count": int(sum(self.dds.values())) + int(self.bool_index),
                "top_ops": self.calls.most_common(top)}


@torch.no_grad()
def count_ops(fn, *inputs) -> dict:
    fn(*inputs)                                              # warm-up outside the counter (lazy state, caches)
    with OpCounter() as oc:
        fn(*inputs)
    return oc.summary()


# ----------------------------------------------------------------------------------------------- dynamo
def graph_breaks(module: nn.Module, *inputs) -> dict:
    """Graph breaks of one forward by `torch._dynamo.explain` and the verdict of `fullgraph=True` (backend 'eager': trace only,
    no C++ compiler involved)."""
    import torch._dynamo as dynamo
    out: Dict[str, object] = {}
    try:
        dynamo.reset()
        with torch.no_grad():
            ex = dynamo.explain(module)(*inputs)
        out.update(graph_count=int(ex.graph_count), graph_break_count=int(ex.graph_break_count),
                   break_reasons=[str(r.reason)[:160] for r in ex.break_reasons[:4]], op_count=int(ex.op_count))
    except Exception as err:                                                     # pragma: no cover
        out.update(graph_count=None, graph_break_count=None, explain_error=f"{type(err).__name__}: {str(err)[:200]}")
    try:
        dynamo.reset()
        comp = torch.compile(module, fullgraph=True, backend="eager")
        with torch.no_grad():
            comp(*inputs)
        out["fullgraph_ok"] = True
    except Exception as err:
        out.update(fullgraph_ok=False, fullgraph_error=f"{type(err).__name__}: {str(err).splitlines()[0][:200] if str(err) else ''}")
    finally:
        dynamo.reset()
    return out


# ----------------------------------------------------------------------------------------------- static grep
def static_scan(objs) -> dict:
    hits, scanned = [], 0
    for o in objs:
        try:
            lines, start = inspect.getsourcelines(o)
            fname = Path(inspect.getsourcefile(o)).relative_to(REPO)
        except (OSError, TypeError, ValueError):
            continue
        scanned += 1
        for k, line in enumerate(lines):
            code = line.split("#")[0]
            for tok in STATIC_TOKENS:
                if re.search(tok, code):
                    hits.append({"file": str(fname), "line": start + k, "token": tok, "text": line.strip()[:120]})
    return {"scanned_functions": scanned, "hits": hits, "n_hits": len(hits)}


# ----------------------------------------------------------------------------------------------- models and inputs
def build_variant(name: str, seed: int = 0, deployed: bool = False):
    """A randomly initialised variant. `deployed=True` applies `model/deploy_fast.py::enable_fast_render` (the host-sync-free,
    bit-identical render + Python-int MANO chain of the DT2 deployment form; opt-in, instance-level); the default form is what
    the evaluators call (`model(x, prev)`), whose MANO chain indexes with CUDA/CPU tensors (`.item()` per joint)."""
    from config import load_config
    cfg_path, cls_path = VARIANTS[name]
    mod, cls = cls_path.rsplit(".", 1)
    Cls = getattr(__import__(mod, fromlist=[cls]), cls)
    torch.manual_seed(seed)
    model = Cls(cfg=load_config(REPO / cfg_path)).eval()
    model.set_hand_context(torch.zeros(1, 10), torch.tensor([[603.4507, 0, 325.09183], [0, 602.95654, 242.09796],
                                                             [0, 0, 1]]).view(1, 3, 3))
    if deployed:
        from deploy_fast import enable_fast_render
        enable_fast_render(model)
    return model


def make_inputs(batch: int = 1):
    g = torch.Generator().manual_seed(SEED_INPUT)
    x = (torch.rand(batch, C.H, C.W, 2, generator=g) * (torch.rand(batch, C.H, C.W, 2, generator=g) < 0.05)).float()
    x[:, 90, 120, 0] = 0.5                                                      # never an empty image (zero-event gate)
    P = torch.randn(batch, 51, generator=g) * 0.02
    return x, P


def forward_sources(model):
    objs = [type(model).forward]
    for n in ("_resolve_betas_K", "_render_prev", "_render_prev_chunks", "_render_chunk", "_splat", "_fk", "_intrinsics",
              "_trunk_with_root_head", "slr_geometry"):
        if hasattr(model, n):
            objs.append(getattr(type(model), n))
    if hasattr(model, "slr_readout"):
        objs.append(type(model.slr_readout).forward)
    from semkine import aura, xyz_mesh
    import backbones
    objs += [aura.visible, xyz_mesh.vertex_normals, backbones.resnet_forward_taps]
    return objs


def profile_form(name: str, deployed: bool, with_dynamo: bool = True) -> dict:
    model = build_variant(name, deployed=deployed)
    x, P = make_inputs()
    rec: Dict[str, object] = {"params": int(sum(p.numel() for p in model.parameters()))}
    try:
        rec["ops"] = count_ops(model, x, P)
    except Exception as err:                                                     # pragma: no cover
        rec["ops_error"] = f"{type(err).__name__}: {str(err)[:200]}"
    if with_dynamo:
        rec["dynamo"] = graph_breaks(model, x, P)
    rec["static"] = static_scan(forward_sources(model))
    return rec


def profile_variant(name: str, with_dynamo: bool = True) -> dict:
    """{'default': form the evaluators run, 'deployed': `enable_fast_render` form}; a form that cannot be built carries
    {'error': ...} (then that variant has no deployed reading: itself a finding)."""
    out: Dict[str, object] = {"variant": name}
    for form, dep in (("default", False), ("deployed", True)):
        try:
            out[form] = profile_form(name, dep, with_dynamo)
        except Exception as err:
            out[form] = {"error": f"{type(err).__name__}: {str(err)[:240]}"}
    return out


# ----------------------------------------------------------------------------------------------- latency
def _lat_init(threads: int, cores: Optional[List[int]]):
    if cores:
        os.sched_setaffinity(0, set(cores))
    torch.set_num_threads(threads)


def _lat_task(names: List[str], reps: int, warmup: int):
    models = {}
    for n in names:
        for form, dep in (("default", False), ("deployed", True)):
            try:
                models[f"{n}|{form}"] = build_variant(n, deployed=dep)
            except Exception:
                pass
    for form, dep in (("default", False), ("deployed", True)):                    # A/A control: an independent twin of S39
        models[f"S39|{form}|AA"] = build_variant("S39", deployed=dep)
    x, P = make_inputs()
    order = list(models)
    ts = {n: [] for n in order}
    load0 = os.getloadavg()[0]
    with torch.no_grad():
        for _ in range(warmup):
            for n in order:
                models[n](x, P)
        for r in range(reps):
            for n in order[r % len(order):] + order[:r % len(order)]:            # rotate the order every rep
                t0 = time.perf_counter()
                models[n](x, P)
                ts[n].append((time.perf_counter() - t0) * 1e3)
    q = lambda v: {"p50_ms": float(np.percentile(v, 50)), "p99_ms": float(np.percentile(v, 99)),   # noqa: E731
                   "mean_ms": float(np.mean(v)), "n": len(v)}
    return {"threads": torch.get_num_threads(), "loadavg_before": load0, "loadavg_after": os.getloadavg()[0],
            "stats": {n: q(v) for n, v in ts.items()}}


def run_latency(threads: int, reps: int, warmup: int, names=tuple(VARIANTS)) -> dict:
    allowed = sorted(os.sched_getaffinity(0))
    cores = allowed[-threads:] if len(allowed) >= threads else None
    with mp.get_context("spawn").Pool(1, initializer=_lat_init, initargs=(threads, cores)) as pool:
        out = pool.apply(_lat_task, (list(names), reps, warmup))
    out["cores"] = cores
    return out


# ----------------------------------------------------------------------------------------------- controls
class _Toy(nn.Module):
    """3 kernel ops (mul, add, sum), 0 syncs: the counter's positive control."""

    def forward(self, x):
        return (x * 2.0 + 1.0).sum()


class _ToyDDS(nn.Module):
    """1 nonzero, 1 boolean-mask index, 1 .item(): known data-dependent / sync counts and a graph break."""

    def forward(self, x):
        idx = torch.nonzero(x > 0)
        sel = x[x > 0]
        v = sel.sum().item()
        return idx.float().sum() + v


def controls() -> dict:
    x = torch.rand(8) - 0.3
    clean = count_ops(_Toy(), x)
    dirty = count_ops(_ToyDDS(), x)
    gb_clean = graph_breaks(_Toy(), x)
    gb_dirty = graph_breaks(_ToyDDS(), x)
    ok = (clean["launches"] == 3 and clean["sync_count"] == 0 and clean["dds_count"] == 0
          and dirty["dds_ops"].get("aten.nonzero", 0) >= 1 and dirty["bool_index"] >= 1
          and dirty["sync_ops"].get("aten._local_scalar_dense", 0) >= 1
          and gb_clean.get("graph_break_count") == 0 and gb_clean.get("fullgraph_ok") is True
          and (gb_dirty.get("graph_break_count") or 0) >= 1 and gb_dirty.get("fullgraph_ok") is False)
    return {"ok": bool(ok), "clean": {k: clean[k] for k in ("launches", "sync_count", "dds_count", "total_calls")},
            "dirty": {k: dirty[k] for k in ("launches", "sync_count", "dds_count", "bool_index")},
            "dynamo_clean": {k: gb_clean.get(k) for k in ("graph_break_count", "fullgraph_ok")},
            "dynamo_dirty": {k: gb_dirty.get(k) for k in ("graph_break_count", "fullgraph_ok")}}


# ----------------------------------------------------------------------------------------------- decision
def _usable(rec: dict) -> bool:
    return isinstance(rec, dict) and "ops" in rec and "dynamo" in rec


def decide(result: dict) -> dict:
    """Gates of SORL / LBA (tools/mv2/prereg_m9.md section 3) on the SLR variant, S39 and S39 + rootanc as references.

    The form read is the DEPLOYED one (`enable_fast_render`, the DT2 host-sync-free form a graph replays) for every variant; if
    S39+SLR cannot be built in it, `gate_deployable` is False and the other gates read the default form.
    tool validity (else `tool_defect`): the counter / dynamo controls return their known answers; S39 and S39+rootanc built.
    gate_deployable S39+SLR builds under `enable_fast_render`
    gate_sync       host syncs (item / bool() / equal) + data-dependent ops == 0, and no static .item() / .cpu() / .numpy() hits in
                    semkine/slr.py
    gate_dds        data-dependent-shape ops (nonzero / unique / masked_select / boolean-mask index) == 0
    gate_breaks     dynamo graph breaks == 0 and the fullgraph trace ok
    gate_launches   launches <= 3 x max(launches of S39, S39+rootanc) (same form)
    gate_cpu8       8-thread p50 increment of S39+SLR over S39 <= +50 % (PROPOSED value)
    Latency claim deleted: any sync / data-dependent op, not deployable, or a CPU8 increment > +100 %. If the S39 A/A control
    differs by more than 10 % the latency gate is `undetermined` (noisy machine) and the verdict cannot be `pass`.
    Descriptive: the counts of S39 itself and of S39+SLR in the default form, so a count can be attributed to SLR or to the base."""
    if not result.get("controls", {}).get("ok", False):
        return {"verdict": "tool_defect", "notes": ["counter / dynamo controls did not return their known answers"]}
    var = result["variants"]
    forms = {}
    for n in VARIANTS:
        rec = var.get(n, {})
        forms[n] = {f: rec.get(f) for f in ("default", "deployed")}
    for n in ("S39", "S39+rootanc"):
        if not (_usable(forms[n]["deployed"]) or _usable(forms[n]["default"])):
            return {"verdict": "tool_defect", "notes": [f"reference variant {n} did not run"]}
    slr_dep = _usable(forms["S39+SLR"]["deployed"])
    if not slr_dep and not _usable(forms["S39+SLR"]["default"]):
        return {"verdict": "tool_defect", "notes": ["S39+SLR did not run in any form"]}
    form = "deployed" if slr_dep else "default"

    def pick(n):
        rec = forms[n][form]
        return rec if _usable(rec) else (forms[n]["default"] if _usable(forms[n]["default"]) else forms[n]["deployed"])
    slr, base, root = pick("S39+SLR"), pick("S39"), pick("S39+rootanc")
    ref = max(base["ops"]["launches"], root["ops"]["launches"])
    slr_static = [h for h in slr["static"]["hits"] if h["file"].endswith("slr.py")]
    gates = {
        "gate_deployable": bool(slr_dep),
        "gate_sync": slr["ops"]["sync_count"] == 0 and slr["ops"]["dds_count"] == 0 and
                     not any(re.search(r"item|cpu|numpy|tolist", h["token"]) for h in slr_static),
        "gate_dds": slr["ops"]["dds_count"] == 0,
        "gate_breaks": slr["dynamo"].get("graph_break_count") == 0 and slr["dynamo"].get("fullgraph_ok") is True,
        "gate_launches": slr["ops"]["launches"] <= LAUNCH_FACTOR * ref,
    }
    out: Dict[str, object] = {
        "form_read": form, "gates": gates, "ref_launches": ref, "slr_launches": slr["ops"]["launches"],
        "launch_ratio": slr["ops"]["launches"] / max(ref, 1),
        "base_counts": {"sync": base["ops"]["sync_count"], "dds": base["ops"]["dds_count"],
                        "breaks": base["dynamo"].get("graph_break_count")},
        "slr_default_form_counts": ({"sync": forms["S39+SLR"]["default"]["ops"]["sync_count"],
                                     "dds": forms["S39+SLR"]["default"]["ops"]["dds_count"],
                                     "launches": forms["S39+SLR"]["default"]["ops"]["launches"]}
                                    if _usable(forms["S39+SLR"]["default"]) else None),
        "notes": []}
    if not slr_dep:
        out["notes"].append("S39+SLR cannot be built under enable_fast_render: " +
                            str((forms["S39+SLR"]["deployed"] or {}).get("error")))
    lat = result.get("latency")
    undetermined = lat is None
    inc = {}
    if lat is not None:
        for thr, r in lat.items():
            st = r["stats"]
            k_s, k_slr, k_aa, k_root = (f"S39|{form}", f"S39+SLR|{form}", f"S39|{form}|AA", f"S39+rootanc|{form}")
            if k_slr not in st or k_s not in st or k_aa not in st:
                undetermined = True
                out["notes"].append(f"{thr}-thread latency: missing {form}-form models")
                continue
            aa = st[k_aa]["p50_ms"] / st[k_s]["p50_ms"] - 1.0
            inc[thr] = {"aa_p50": aa, "slr_p50": st[k_slr]["p50_ms"] / st[k_s]["p50_ms"] - 1.0,
                        "slr_p99": st[k_slr]["p99_ms"] / st[k_s]["p99_ms"] - 1.0,
                        "rootanc_p50": (st[k_root]["p50_ms"] / st[k_s]["p50_ms"] - 1.0) if k_root in st else None}
            if abs(aa) > AA_NOISE_MAX:
                undetermined = True
                out["notes"].append(f"{thr}-thread A/A control differs by {aa:+.1%}: noisy machine")
        out["latency_increment"] = inc
        if "8" in inc and not undetermined:
            gates["gate_cpu8"] = inc["8"]["slr_p50"] <= CPU8_INCREMENT_MAX
    else:
        out["notes"].append("no latency section")
    if "gate_cpu8" not in gates:
        gates["gate_cpu8"] = None
    delete = (not gates["gate_deployable"]) or (not gates["gate_sync"]) or (not gates["gate_dds"]) or (
        "8" in inc and inc["8"]["slr_p50"] > KILL_INCREMENT)
    out["latency_claim_deleted"] = bool(delete)
    if any(v is False for v in gates.values()):
        out["verdict"] = "fail:" + ",".join(k for k, v in gates.items() if v is False)
    elif any(v is None for v in gates.values()):
        out["verdict"] = "undetermined"
    else:
        out["verdict"] = "pass"
    if out["verdict"].startswith("fail") and delete:
        out["notes"].append("Latency claim of M1 deleted; rewrite with static masks (plan, SORL kill rule)")
    return out


# ----------------------------------------------------------------------------------------------- main
def smoke(args) -> int:
    t0 = time.time()
    torch.set_num_threads(args.threads)
    ctl = controls()
    print(f"[m9_cost smoke] counter / dynamo controls ok={ctl['ok']}")
    print(f"  toy (x*2+1).sum(): launches={ctl['clean']['launches']} (expect 3), syncs={ctl['clean']['sync_count']}, "
          f"dds={ctl['clean']['dds_count']}, total aten calls={ctl['clean']['total_calls']}")
    print(f"  toy data-dependent: launches={ctl['dirty']['launches']}, syncs={ctl['dirty']['sync_count']}, "
          f"dds ops incl. bool-index={ctl['dirty']['dds_count']} (bool-index {ctl['dirty']['bool_index']}); "
          f"dynamo clean {ctl['dynamo_clean']}, dirty {ctl['dynamo_dirty']}")
    ok = ctl["ok"]
    # the variants are exercised silently: no op count, break count or latency of S39 / rootanc / SLR is printed
    for n in VARIANTS:
        try:
            rec = profile_variant(n, with_dynamo=not args.no_dynamo)
            ran = all(("ops" in rec[f] and (args.no_dynamo or "dynamo" in rec[f]) and rec[f]["static"]["scanned_functions"] > 0)
                      or "error" in rec[f] for f in ("default", "deployed")) and "ops" in rec["default"]
        except Exception as err:                                                 # pragma: no cover
            ran = False
            print(f"  variant {n}: ERROR {type(err).__name__}: {str(err)[:160]}")
        print(f"  variant {n}: code path executed = {ran}")
        ok &= ran
    lat = run_latency(2, reps=6, warmup=2)
    print(f"  latency harness: {len(lat['stats'])} models x {next(iter(lat['stats'].values()))['n']} reps at {lat['threads']} threads, "
          f"cores {lat['cores']}; harness wall {time.time() - t0:.1f} s")
    ok &= len(lat["stats"]) >= len(VARIANTS) + 1
    def fv(launches, sync=0, dds=0):
        rec = {"ops": {"launches": launches, "sync_count": sync, "dds_count": dds},
               "dynamo": {"graph_break_count": 0, "fullgraph_ok": True}, "static": {"hits": []}}
        return {"default": rec, "deployed": rec}
    fake_var = {n: fv(100 + 10 * i) for i, n in enumerate(VARIANTS)}
    keys = [f"{n}|{f}" for n in VARIANTS for f in ("default", "deployed")] + ["S39|default|AA", "S39|deployed|AA"]
    fake_lat = {t: {"stats": {k: {"p50_ms": 10.0 + (1.0 if k.startswith("S39+SLR") else 0.0), "p99_ms": 12.0} for k in keys}}
                for t in ("2", "8")}
    d = decide({"controls": ctl, "variants": fake_var, "latency": fake_lat})
    d_bad = decide({"controls": ctl, "variants": {**fake_var, "S39+SLR": fv(1000, sync=1)}, "latency": fake_lat})
    print(f"  decide() plumbing on synthetic counts: clean -> {d['verdict']}, sync+launch overrun -> {d_bad['verdict']}")
    ok &= d["verdict"] == "pass" and d_bad["verdict"].startswith("fail:gate_sync,gate_launches")
    out_dir = REPO / "outputs" / "mv2" / "smoke" / "m9_cost"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke.json").write_text(json.dumps({"controls": ctl, "ok": bool(ok)}, indent=1))
    print(f"[m9_cost smoke] {'PASS' if ok else 'FAIL'} ({time.time() - t0:.1f} s)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--no-dynamo", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        raise SystemExit(smoke(a))
    if not a.out:
        raise SystemExit("--out outputs/mv2/m9_cost (formal) or --smoke")
    C.verify_protocol()
    out_dir = Path(a.out)
    torch.set_num_threads(a.threads)
    result = {"controls": controls(), "variants": {n: profile_variant(n) for n in VARIANTS}, "latency": {}}
    for thr in (2, 8):
        result["latency"][str(thr)] = run_latency(thr, a.reps, a.warmup)
    result["decision"] = decide(result)
    C.save_json(out_dir / "result.json", result)
    C.receipt(out_dir, {"probe": "m9_cost", "verdict": result["decision"]["verdict"]},
              scripts=["tools/mv2/m9_cost.py", "tools/mv2/common.py"])
    print(json.dumps(result["decision"], indent=1, default=C._np))


if __name__ == "__main__":
    main()
