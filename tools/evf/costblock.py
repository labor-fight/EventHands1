#!/usr/bin/env python3
"""EVF final cost block (docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104 deliverable 4): the comparators through PFO's
frozen cost harness (tools/pfo/cost.py: the first 600 lyq_local packets, parameters, executed Linear / Conv MACs on the
actual tensors, MANO FK apart, row-form and end-to-end latency, memory) with its output root redirected to
outputs/evf/20261104/costs/<tag>/, and the EVF networks through tools/evf/cost2.py `run`; every job exclusive on GPU 4.

    python -m tools.evf.costblock run --net s43 --impl exec|dense --policy default|det
    python -m tools.evf.costblock run --net f --policy ...          PFO's F, seed 3407, step 6000
    python -m tools.evf.costblock run --net b2 --policy ...         B2 = PFO2's fgs, seed 3407, step 6000
    python -m tools.evf.costblock jobs --name N --after KEY [...] --evf ARM:PHASE:STEP [...] --out FILE
        the block: PyTorch-default flags first (the table's policy), then the deterministic ones; S43 sparse executor,
        the EVF networks, B2, F, S43 masked dense; chained one after another after the given jobs
Tags final__<name>__<impl>__<policy>.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.evf.common import OUT, PY, require  # noqa: E402

POLICIES = ("default", "det")


def run(net, impl, policy):
    from tools.pfo import cost as C
    C.OUT = OUT
    if net == "s43":
        require(impl in ("exec", "dense"), "S43 implementations: exec | dense")
        name, sp = "s43_s3407", None
    elif net == "f":
        from tools.pfo2 import run as R
        name, sp = R.pfo_spec("f1", 3407, 6000)
        R.prepare(sp)
        impl = "fast"
    else:
        from tools.pfo2 import run as R
        from tools.pfo2 import run2 as R2
        name, sp = R2.arm_spec("fgs", 3407, 6000)
        R.prepare(sp)
        impl = "fast"
    C.run(name, impl, str(sp) if sp else None, f"final__{name}__{impl}__{policy}", policy)


def jobs(name, after, evf):
    out, last = [], None
    for policy in POLICIES:
        cmds = [("s43:exec", ["tools.evf.costblock", "run", "--net", "s43", "--impl", "exec"])]
        for item in evf:
            arm, phase, step = item.split(":")
            cmds.append((f"evf:{arm}", ["tools.evf.cost2", "run", "--arm", arm, "--phase", phase, "--step", step]))
        cmds += [("b2", ["tools.evf.costblock", "run", "--net", "b2"]), ("f", ["tools.evf.costblock", "run", "--net", "f"]),
                 ("s43:dense", ["tools.evf.costblock", "run", "--net", "s43", "--impl", "dense"])]
        for label, args in cmds:
            key = f"{name}:cost:{label}:{policy}"
            out.append({"key": key, "kind": "exclusive", "command": [PY, "-m", *args, "--policy", policy],
                        "deps": [last] if last else list(after), "mem_gib": 15, "note": "EVF final cost block"})
            last = key
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--net", required=True, choices=("s43", "f", "b2"))
    r.add_argument("--impl", default="fast")
    r.add_argument("--policy", required=True, choices=POLICIES)
    j = sub.add_parser("jobs")
    j.add_argument("--name", required=True)
    j.add_argument("--after", action="append", required=True)
    j.add_argument("--evf", action="append", required=True)
    j.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.verb == "run":
        return run(a.net, a.impl, a.policy)
    o = Path(a.out)
    require(not o.exists(), "job file exists")
    J = jobs(a.name, a.after, a.evf)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(J, indent=1) + "\n")
    print(len(J), "jobs ->", o)


if __name__ == "__main__":
    main()
