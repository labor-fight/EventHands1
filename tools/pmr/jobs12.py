#!/usr/bin/env python3
"""PMR stage S6 job list (docs/EXPERIMENT_RECORDS.md#pmr-s6-prereg-20261109), written once and appended with
`chain3.py add`. Generating the list writes the registered configs.

    python -m tools.pmr.jobs12 s6 --freeze s6 --out outputs/pmr/20261109/jobs/s6.json

Freeze check; d1hub's preflight (pair, 40 steps) and its acceptance (tools/pmr/train7.py / run7.py); the formal training
(pair, seed 3407, 6000 steps); acceptance; e1 / e3 / pert, f1 after e1; the written decision (tools/pmr/analyze9.py), after
the S4 + S4b + S4c joint selection too.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms7 as A7  # noqa: E402
from tools.pmr.common import PY, ROOT as RR, SEED, require  # noqa: E402
from tools.pmr.jobs import TASKS, TRAIN_MEM, cpu, gpu  # noqa: E402

ARM = "d1hub"


def train(key, phase, deps):
    return {"key": key, "kind": "pair", "deps": list(deps), "mem_gib": TRAIN_MEM,
            "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "NCCL_P2P_DISABLE=1", PY,
                        str(RR / "tools/pmr/train7.py"), "--config", str(A7.write_config(ARM, SEED, phase))],
            "note": f"{ARM} {phase}"}


def s6(freeze):
    p = "s6"
    J = [cpu(f"{p}:freeze", ["verify", "--stage", freeze], [], "manifest unchanged", module="tools.pmr.freeze"),
         train(f"{p}:pre:{ARM}", "preflight", [f"{p}:freeze"])]
    J.append(cpu(f"{p}:pre:accept:{ARM}", ["accept", "--arm", ARM, "--phase", "preflight"], [f"{p}:pre:{ARM}"],
                 "preflight acceptance", module="tools.pmr.run7"))
    J.append(train(f"{p}:train:{ARM}", "formal", [f"{p}:pre:accept:{ARM}"]))
    J.append(cpu(f"{p}:accept:{ARM}", ["accept", "--arm", ARM, "--phase", "formal"], [f"{p}:train:{ARM}"], "acceptance",
                 module="tools.pmr.run7"))
    reads = []
    for task in TASKS:
        J.append(gpu(f"{p}:read:{ARM}:{task}", ["read", "--arm", ARM, "--task", task], [f"{p}:accept:{ARM}"],
                     module="tools.pmr.run7"))
        reads.append(J[-1]["key"])
    J.append(gpu(f"{p}:read:{ARM}:f1", ["read", "--arm", ARM, "--task", "f1"], [f"{p}:read:{ARM}:e1"],
                 module="tools.pmr.run7"))
    reads.append(J[-1]["key"])
    J.append(cpu(f"{p}:decide", ["s6"], reads + ["s4c:joint"], "written decision", module="tools.pmr.analyze9"))
    keys = [j["key"] for j in J]
    require(len(keys) == len(set(keys)), "duplicate job keys")
    return J


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("s6")
    s.add_argument("--freeze", required=True)
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "written once: " + str(out))
    jobs = s6(a.freeze)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
