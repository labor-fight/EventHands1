#!/usr/bin/env python3
"""PMR stage S2 job list (docs/EXPERIMENT_RECORDS.md#pmr-prereg-20261109, "过筛之后"; the design's S2 section), written
once and appended with `chain.py add`. Generating the list writes the registered configs it trains.

    python -m tools.pmr.jobs3 s2 --design d1 --freeze s2d1 --out outputs/pmr/20261109/jobs/s2_d1.json

R and R-noprop at seeds 3408 / 3409 from scratch (B2's recipe, 6000 steps, no ROL): freeze check; the four formal
trainings (pair; the code path of the accepted seed-3407 preflight and formal runs, so no new preflight; acceptance
checks the data stream against B2's formal run of the same seed); acceptance; e1 / e3 / pert, f1 after e1; the written
confirmation (tools/pmr/analyze2.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms as A  # noqa: E402
from tools.pmr.jobs import TASKS, cpu, gpu, train  # noqa: E402
from tools.pmr.common import require  # noqa: E402

S2_SEEDS = (3408, 3409)


def s2(design, freeze):
    require(design in A.DESIGNS, "unregistered design")
    p = f"s2:{design}"
    arms = (f"{design}r", f"{design}n")
    J = [cpu(f"{p}:freeze", ["verify", "--stage", freeze], [], "manifest unchanged", module="tools.pmr.freeze")]
    for seed in S2_SEEDS:
        for arm in arms:
            J.append(train(f"{p}:train:{arm}:s{seed}", arm, "formal", [f"{p}:freeze"], seed))
            J.append(cpu(f"{p}:accept:{arm}:s{seed}", ["accept", "--arm", arm, "--phase", "formal", "--seed", str(seed)],
                         [f"{p}:train:{arm}:s{seed}"], "acceptance"))
    reads = []
    for seed in S2_SEEDS:
        for arm in arms:
            for task in TASKS:
                J.append(gpu(f"{p}:read:{arm}:s{seed}:{task}", ["read", "--arm", arm, "--seed", str(seed), "--task", task],
                             [f"{p}:accept:{arm}:s{seed}"]))
                reads.append(J[-1]["key"])
            J.append(gpu(f"{p}:read:{arm}:s{seed}:f1", ["read", "--arm", arm, "--seed", str(seed), "--task", "f1"],
                         [f"{p}:read:{arm}:s{seed}:e1"]))
            reads.append(J[-1]["key"])
    J.append(cpu(f"{p}:confirm", ["s2", "--design", design], reads, "written confirmation", module="tools.pmr.analyze2"))
    keys = [j["key"] for j in J]
    require(len(keys) == len(set(keys)), "duplicate job keys")
    seen = set()
    for j in J:
        require(all(d in seen for d in j["deps"]), "dependency before its job: " + j["key"])
        seen.add(j["key"])
    return J


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("s2")
    s.add_argument("--design", required=True)
    s.add_argument("--freeze", required=True)
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "written once: " + str(out))
    jobs = s2(a.design, a.freeze)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
