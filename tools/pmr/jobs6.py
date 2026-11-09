#!/usr/bin/env python3
"""PMR engineering amendment 3 job list (docs/EXPERIMENT_RECORDS.md#pmr-amend3-20261109), written once and appended with
`chain.py add`: the rounds rungs d1l1b / d1l2b (tools/pmr/arms3.py, tools/pmr/train3.py, run3.py) in place of d1l1 /
d1l2, and the ladder table (tools/pmr/analyze5.py) in place of the skipped `s3b:table`.

    python -m tools.pmr.jobs6 amend3 --freeze amend3 --out outputs/pmr/20261109/jobs/amend3.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms3 as A3  # noqa: E402
from tools.pmr.common import PY, ROOT as RR, SEED, require  # noqa: E402
from tools.pmr.jobs import TASKS, TRAIN_MEM, cpu, gpu  # noqa: E402

ORDER = ("d1l1b", "d1l2b")
S3B_READS = [f"s3b:read:{a}:{t}" for a in ("d1g0", "d1h0", "d1a0") for t in TASKS + ("f1",)]


def train(key, arm, phase, deps):
    return {"key": key, "kind": "pair", "deps": list(deps), "mem_gib": TRAIN_MEM,
            "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "NCCL_P2P_DISABLE=1", PY,
                        str(RR / "tools/pmr/train3.py"), "--config", str(A3.write_config(arm, SEED, phase))],
            "note": f"{arm} {phase}"}


def amend3(freeze):
    p = "a3"
    J = [cpu(f"{p}:freeze", ["verify", "--stage", freeze], [], "manifest unchanged", module="tools.pmr.freeze")]
    for arm in ORDER:
        J.append(train(f"{p}:pre:{arm}", arm, "preflight", [f"{p}:freeze"]))
        J.append(cpu(f"{p}:pre:accept:{arm}", ["accept", "--arm", arm, "--phase", "preflight"], [f"{p}:pre:{arm}"],
                     "preflight acceptance", module="tools.pmr.run3"))
    for arm in ORDER:
        J.append(train(f"{p}:train:{arm}", arm, "formal", [f"{p}:pre:accept:{arm}"]))
        J.append(cpu(f"{p}:accept:{arm}", ["accept", "--arm", arm, "--phase", "formal"], [f"{p}:train:{arm}"],
                     "acceptance", module="tools.pmr.run3"))
    reads = []
    for arm in ORDER:
        for task in TASKS:
            J.append(gpu(f"{p}:read:{arm}:{task}", ["read", "--arm", arm, "--task", task], [f"{p}:accept:{arm}"],
                         module="tools.pmr.run3"))
            reads.append(J[-1]["key"])
        J.append(gpu(f"{p}:read:{arm}:f1", ["read", "--arm", arm, "--task", "f1"], [f"{p}:read:{arm}:e1"],
                     module="tools.pmr.run3"))
        reads.append(J[-1]["key"])
    J.append(cpu(f"{p}:table", ["ladder"], reads + S3B_READS, "descriptive ladder table", module="tools.pmr.analyze5"))
    keys = [j["key"] for j in J]
    require(len(keys) == len(set(keys)), "duplicate job keys")
    return J


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("amend3")
    s.add_argument("--freeze", required=True)
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "written once: " + str(out))
    jobs = amend3(a.freeze)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
