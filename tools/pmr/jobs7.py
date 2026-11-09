#!/usr/bin/env python3
"""PMR stage S4 job list (docs/EXPERIMENT_RECORDS.md#pmr-s4-prereg-20261109), written once and appended with
`chain.py add`. Generating the list writes the registered configs.

    python -m tools.pmr.jobs7 s4 --freeze s4 --out outputs/pmr/20261109/jobs/s4.json

Freeze check; one preflight (pair, 40 steps) per arm and its acceptance (tools/pmr/train4.py / run4.py); the four
formal trainings (pair, seed 3407, 6000 steps); acceptance; e1 / e3 / pert, f1 after e1; the written selection
(tools/pmr/analyze6.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms4 as A4  # noqa: E402
from tools.pmr.common import PY, ROOT as RR, SEED, require  # noqa: E402
from tools.pmr.jobs import TASKS, TRAIN_MEM, cpu, gpu  # noqa: E402

ORDER = ("d1dual", "d1hp", "d1dualn", "d1dualm")


def train(key, arm, phase, deps):
    return {"key": key, "kind": "pair", "deps": list(deps), "mem_gib": TRAIN_MEM,
            "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "NCCL_P2P_DISABLE=1", PY,
                        str(RR / "tools/pmr/train4.py"), "--config", str(A4.write_config(arm, SEED, phase))],
            "note": f"{arm} {phase}"}


def s4(freeze):
    p = "s4"
    J = [cpu(f"{p}:freeze", ["verify", "--stage", freeze], [], "manifest unchanged", module="tools.pmr.freeze")]
    for arm in ORDER:
        J.append(train(f"{p}:pre:{arm}", arm, "preflight", [f"{p}:freeze"]))
        J.append(cpu(f"{p}:pre:accept:{arm}", ["accept", "--arm", arm, "--phase", "preflight"], [f"{p}:pre:{arm}"],
                     "preflight acceptance", module="tools.pmr.run4"))
    for arm in ORDER:
        J.append(train(f"{p}:train:{arm}", arm, "formal", [f"{p}:pre:accept:{arm}"]))
        J.append(cpu(f"{p}:accept:{arm}", ["accept", "--arm", arm, "--phase", "formal"], [f"{p}:train:{arm}"],
                     "acceptance", module="tools.pmr.run4"))
    reads = []
    for arm in ORDER:
        for task in TASKS:
            J.append(gpu(f"{p}:read:{arm}:{task}", ["read", "--arm", arm, "--task", task], [f"{p}:accept:{arm}"],
                         module="tools.pmr.run4"))
            reads.append(J[-1]["key"])
        J.append(gpu(f"{p}:read:{arm}:f1", ["read", "--arm", arm, "--task", "f1"], [f"{p}:read:{arm}:e1"],
                     module="tools.pmr.run4"))
        reads.append(J[-1]["key"])
    J.append(cpu(f"{p}:select", ["s4"], reads, "written selection", module="tools.pmr.analyze6"))
    keys = [j["key"] for j in J]
    require(len(keys) == len(set(keys)), "duplicate job keys")
    return J


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("s4")
    s.add_argument("--freeze", required=True)
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "written once: " + str(out))
    jobs = s4(a.freeze)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
