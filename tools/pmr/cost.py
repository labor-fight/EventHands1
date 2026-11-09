#!/usr/bin/env python3
"""PMR cost readings (docs/EXPERIMENT_RECORDS.md#pmr-prereg-20261109) through PFO's frozen cost harness
(tools/pfo/cost.py: the first 600 lyq_local packets, parameters incl. the router, executed Linear / Conv MACs on the
actual tensors (the PMR node maps are grouped 1 x 1 convolutions, so they are counted), MANO FK apart, row-form and
end-to-end latency, memory) with its output root redirected to outputs/pmr/20261109/costs/<tag>/; every job exclusive on
GPU 4 (no other GPU job).

    python -m tools.pmr.cost run --arm A [--seed S] --policy default|det      an accepted arm at step 6000
    python -m tools.pmr.cost run --net b2 --policy default|det                 B2 (PFO2 fgs 3407, step 6000), same block
    python -m tools.pmr.cost run --net s43 --impl exec|dense --policy ...      S43 (s43n_sp seed 3407)
Tags <name>__<impl>__<policy>.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr.common import OUT, SEED, require  # noqa: E402

POLICIES = ("default", "det")


def run(net, impl, policy, arm=None, seed=SEED):
    from tools.pfo import cost as C
    from tools.pmr import run as R
    C.OUT = OUT
    if net == "s43":
        require(impl in ("exec", "dense"), "S43 implementations: exec | dense")
        name, sp = "s43_s3407", None
    elif net == "b2":
        from tools.pmr.common import b2_spec, check_b2
        check_b2()
        name, sp = f"b2_s{SEED}", OUT / "specs" / f"b2_s{SEED}.json"
        require(sp.exists(), "B2's PMR spec is written by `run b2` (the K1 reading) first")
        R.prepare(sp)
        impl = "fast"
    else:
        name, sp = R.arm_spec(arm, seed)
        R.prepare(sp)
        impl = "fast"
    C.run(name, impl, str(sp) if sp else None, f"{name}__{impl}__{policy}", policy)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--net", default="arm", choices=("s43", "b2", "arm"))
    r.add_argument("--impl", default="fast")
    r.add_argument("--arm")
    r.add_argument("--seed", type=int, default=SEED)
    r.add_argument("--policy", required=True, choices=POLICIES)
    a = p.parse_args(argv)
    require((a.net == "arm") == (a.arm is not None), "--arm goes with --net arm")
    return run(a.net, a.impl, a.policy, a.arm, a.seed)


if __name__ == "__main__":
    main()
