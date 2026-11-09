#!/usr/bin/env python3
"""PMR stage S3c cost block (docs/EXPERIMENT_RECORDS.md#pmr-s3c-prereg-20261109): S43, B2 and the PMR arm in one block
through PFO's frozen cost harness (tools/pfo/cost.py: the first 600 lyq_local packets, parameters incl. the router,
executed Linear / Conv MACs on the actual tensors, MANO FK apart, row-form and end-to-end latency, memory), output root
outputs/pmr/20261109/costs/final__<name>__<impl>__<policy>/; every job exclusive on GPU 4.

    python -m tools.pmr.cost3 run --net s43 --impl exec|dense --policy default|det
    python -m tools.pmr.cost3 run --net b2 --policy default|det
    python -m tools.pmr.cost3 run --arm A [--seed S] [--fast] --policy default|det
        a PMR arm (accepted step 6000; a rollout fine-tune has its base arm's architecture); --fast patches the network
        with semkine/pmr_fast.py (the same function in fewer kernels) after loading, and first records the largest
        difference between the reference and the fast increments over the first 100 packets (fast_check.json)
The PMR stage breakdown is tools/pmr/cost2.py's.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr.common import OUT, SEED, now, require, save  # noqa: E402

POLICIES = ("default", "det")
CHECK_PACKETS = 100


def fast_check(net, tag):
    """Reference vs fast increments on the first CHECK_PACKETS lyq_local packets (GPU, evaluation precision)."""
    import torch
    from tools.pfo import cost as C
    from semkine import pmr_fast as F
    import copy
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
    out = OUT / "costs" / f"{tag}_fast_check.json"
    save(out, {"packets": len(diffs), "max_abs_diff": max(diffs), "mean_max_abs_diff": sum(diffs) / len(diffs),
               "written_utc": now()}, exclusive=True)
    return max(diffs)


def run(net, impl, policy, arm=None, seed=SEED, fast=False):
    from tools.pfo import cost as C
    from tools.pfo import nets as NM
    from tools.pmr import cost2 as C2
    from tools.pmr import run as R
    C.OUT = OUT
    if net == "s43":
        require(impl in ("exec", "dense"), "S43 implementations: exec | dense")
        name, sp = "s43_s3407", None
    elif net == "b2":
        from tools.pmr.common import b2_spec, check_b2
        check_b2()
        name, sp = f"b2_s{SEED}", OUT / "specs" / f"b2_s{SEED}.json"
        require(sp.exists(), "B2's PMR spec (written by the K1 reading)")
        R.prepare(sp)
        impl = "fast"
    else:
        name, sp = R.arm_spec(arm, seed)
        R.prepare(sp)
        C.stage_breakdown = C2.pmr_stage_breakdown
        impl = "pmrfast" if fast else "fast"
        if fast:
            from semkine import pmr_fast as F
            native = NM.load

            def load(*a, **k):
                n = native(*a, **k)
                d = fast_check(n, f"final__{name}__pmrfast__{policy}")
                require(d < 1e-3, f"fast path differs from the reference by {d}")
                F.fastify(n.model.xyz)
                return n
            NM.load = load
    C.run(name, "fast" if impl == "pmrfast" else impl, str(sp) if sp else None, f"final__{name}__{impl}__{policy}", policy)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--net", default="arm", choices=("s43", "b2", "arm"))
    r.add_argument("--impl", default="fast")
    r.add_argument("--arm")
    r.add_argument("--seed", type=int, default=SEED)
    r.add_argument("--fast", action="store_true")
    r.add_argument("--policy", required=True, choices=POLICIES)
    a = p.parse_args(argv)
    require((a.net == "arm") == (a.arm is not None), "--arm goes with --net arm")
    return run(a.net, a.impl, a.policy, a.arm, a.seed, a.fast)


if __name__ == "__main__":
    main()
