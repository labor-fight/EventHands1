#!/usr/bin/env python3
"""EvHandTrack cost block, kinematic-tree identity (docs/EXPERIMENT_RECORDS.md#evhandtrack-d1-20261110): PFO's frozen
cost harness (tools/pfo/cost.py: the first 600 lyq_local packets, parameters incl. the router, executed Linear / Conv MACs
on the actual tensors, row-form and end-to-end latency) with PMR's stage breakdown (tools/pmr/cost2.py), on EvHandTrack
seed 3407 (PMR run `pmr_d1r_rol5_s3407`, step 1500; --arm d1ghand_rol5 reads the ghand + rol5 record) and S43's sparse
executor in the same block; output root outputs/evhandtrack/costs/<tag>/ (or --out <root>); one GPU among 2-7, nothing
else on it.

    python -m tools.evhandtrack.cost2 run --net s43 --policy default
    python -m tools.evhandtrack.cost2 run --net packet|stream [--fast] [--arm d1r_rol5|d1ghand_rol5] --policy default

--fast patches the trunk with semkine/pmr_fast.py (the same function in fewer kernels) after loading. The packet run
first records the largest reference-vs-fast increment difference over the first 100 lyq_local packets (tools/pmr/cost3.py's
check, refused at >= 1e-3) in <root>/costs/evhandtrack_s3407__fast_check.json; the stream run requires that record. The
streaming form's hooks are tools/evhandtrack/cost.py's (the state is prepared in the host part of the step, right after
the window cut). The B2 + rol5 cost tool, tools/evhandtrack/cost.py, is unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo.common import now, read, require, save, sha  # noqa: E402

OUT = ROOT / "outputs/evhandtrack"
PMR = ROOT / "outputs/pmr/20261109"
ARMS = ("d1r_rol5", "d1ghand_rol5")                  # EvHandTrack (#evhandtrack-d1-20261110); the ghand + rol5 record
SEED = 3407
POLICIES = ("default", "det")
CHECK_PACKETS = 100
FAST_MAX = 1e-3


def check_path(out):
    return out / "costs" / f"evhandtrack_s{SEED}__fast_check.json"


def fast_check(net, out, run):
    """tools/pmr/cost3.py `fast_check` with the record under this block's root."""
    import copy
    import torch
    from tools.pfo import cost as C
    from semkine import pmr_fast as F
    ctx, items = C.lyq_packets(net.cfg, torch.device("cuda"))
    net.bind(ctx)
    ref = net.model
    fast = copy.deepcopy(ref)
    F.fastify(fast.xyz)
    diffs = []
    with torch.no_grad():
        for end, gt_a, _ in items[:CHECK_PACKETS]:
            a = net.step(ctx, end, gt_a)
            net.model = fast
            b = net.step(ctx, end, gt_a)
            net.model = ref
            diffs.append(float(abs(a - b).max()))
    save(check_path(out), {"run": run, "packets": len(diffs), "max_abs_diff": max(diffs),
                           "mean_max_abs_diff": sum(diffs) / len(diffs), "written_utc": now()}, exclusive=True)
    return max(diffs)


def run(net, policy, fast=False, out=OUT, arm=ARMS[0]):
    from tools.pfo import cost as C
    from tools.pfo import nets as NM
    C.OUT = out
    if net == "s43":
        C.run("s43_s3407", "exec", None, f"s43_s3407__exec__{policy}", policy)
        return
    from tools.pmr import cost2 as C2
    from tools.pmr import run as R
    run_name = f"pmr_{arm}_s{SEED}"
    spec = PMR / "specs" / f"{run_name}.json"
    sp = read(spec)
    receipt = read(PMR / "receipts" / f"{run_name}.json")
    require(receipt["status"] == "PASS", run_name + ": rol5 acceptance")
    require(sha(sp["checkpoint"]) == sp["checkpoint_sha256"] == receipt["checkpoint_sha256_by_step"]["1500"],
            run_name + ": checkpoint changed since acceptance")
    R.prepare(spec)
    C.stage_breakdown = C2.pmr_stage_breakdown
    if fast:
        from semkine import pmr_fast as F
        if net == "stream":
            c = read(check_path(out))
            require(c["run"] == run_name and c["max_abs_diff"] < FAST_MAX, "the packet run's fast check")
        native = NM.load

        def load(*a, **k):
            n = native(*a, **k)
            if net == "packet":
                d = fast_check(n, out, run_name)
                require(d < FAST_MAX, f"fast path differs from the reference by {d}")
            F.fastify(n.model.xyz)
            return n
        NM.load = load
    if net == "stream":
        from semkine import tpd_str as TS
        from tools.evhandtrack import cost as EC
        TS.install()
        EC.stream_cost_hooks()                  # wraps NM.load after the fast patch: load -> fastify -> retoken
    tag = f"evhandtrack_s{SEED}__{net}__{'pmrfast' if fast else 'ref'}__{policy}"
    C.run(run_name, "fast", str(spec), tag, policy)
    if net == "stream":
        nodes = EC._NODES
        save(out / "costs" / tag / "stream.json",
             {"stream": True, "budget": TS.BUDGET, "window_ms": TS.WINDOW_MS, "n_packets_built": len(nodes),
              "nodes_mean": sum(nodes) / max(len(nodes), 1), "nodes_max": max(nodes, default=0),
              "written_utc": now()}, exclusive=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--net", required=True, choices=("s43", "packet", "stream"))
    r.add_argument("--fast", action="store_true")
    r.add_argument("--policy", required=True, choices=POLICIES)
    r.add_argument("--out", default=str(OUT))
    r.add_argument("--arm", default=ARMS[0], choices=ARMS)
    a = p.parse_args(argv)
    require(not (a.fast and a.net == "s43"), "--fast is for the EvHandTrack trunk")
    return run(a.net, a.policy, a.fast, Path(a.out), a.arm)


if __name__ == "__main__":
    main()
