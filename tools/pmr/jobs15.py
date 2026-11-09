#!/usr/bin/env python3
"""PMR stage S8 repair job list (docs/EXPERIMENT_RECORDS.md#pmr-amend8-20261109), written once and appended with
`chain4.py add`. The rol5 fine-tunes of d1hub at seeds 3407 / 3408 failed on borrowed pairs: in gpu-borrow's system
service a process may open 1024 files, and a K = 5 rollout batch passes more tensors between DataLoader workers than that
("received 0 items of ancdata"). From scratch trainings stay below it and keep running on borrowed pairs.

    python -m tools.pmr.jobs15 s8r2 --seeds 3407,3408 --out outputs/pmr/20261109/jobs/s8_r2.json

Per failed seed: the same rollout training (tools/pmr/train_rol3.py, same config) under the key suffix :r2 on the direct
pairs (4, 5) / (6, 7) only, its acceptance and readings (tools/pmr/run_rol3.py); then the cost reading and the written
verdict of S8 under :r2 (the originals were skipped), after every S8 reading.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import jobs14 as J14  # noqa: E402
from tools.pmr.common import PY, require  # noqa: E402
from tools.pmr.jobs import TASKS, cpu, gpu  # noqa: E402

ARM = "d1hub"
NAME = f"{ARM}_rol5"
DIRECT = [[4, 5], [6, 7]]


def s8r2(seeds):
    require(seeds and set(seeds) <= {3407, 3408, 3409}, "seeds")
    p = "s8"
    J, reads = [], []
    for seed in (3408, 3409):
        reads += [f"{p}:read:{ARM}:s{seed}:{t}" for t in (*TASKS, "f1")]
    for seed in (3407, 3408, 3409):
        if seed not in seeds:
            reads += [f"{p}:rread:{NAME}:s{seed}:{t}" for t in (*TASKS, "f1")]
            continue
        # seed 3407's base acceptance (s6:accept:d1hub) ended EXIT_0 in driver 3 and is not in driver 4's queue
        base = [] if seed == 3407 else [f"{p}:accept:{ARM}:s{seed}"]
        job = J14.rollout(f"{p}:rtrain:{NAME}:s{seed}:r2", NAME, seed, "formal", [f"{p}:rpre:accept:{NAME}", *base])
        J.append({**job, "pairs": DIRECT})
        J.append(cpu(f"{p}:raccept:{NAME}:s{seed}:r2", ["accept", "--arm", NAME, "--seed", str(seed), "--phase", "formal"],
                     [job["key"]], "acceptance", module="tools.pmr.run_rol3"))
        for task in TASKS:
            J.append(gpu(f"{p}:rread:{NAME}:s{seed}:{task}:r2", ["read", "--arm", NAME, "--seed", str(seed), "--task", task],
                         [f"{p}:raccept:{NAME}:s{seed}:r2"], module="tools.pmr.run_rol3"))
            reads.append(J[-1]["key"])
        J.append(gpu(f"{p}:rread:{NAME}:s{seed}:f1:r2", ["read", "--arm", NAME, "--seed", str(seed), "--task", "f1"],
                     [f"{p}:rread:{NAME}:s{seed}:e1:r2"], module="tools.pmr.run_rol3"))
        reads.append(J[-1]["key"])
    J.append({"key": f"{p}:cost:{ARM}:r2", "kind": "exclusive", "deps": reads, "mem_gib": 15,
              "note": "cost reading of the arm (default flags)",
              "command": [PY, "-m", "tools.pmr.cost6", "run", "--arm", ARM, "--policy", "default"]})
    J.append(cpu(f"{p}:verdict:r2", ["s5", "--arm", ARM], [f"{p}:cost:{ARM}:r2"], "written verdict",
                 module="tools.pmr.analyze11"))
    keys = [j["key"] for j in J]
    require(len(keys) == len(set(keys)), "duplicate job keys")
    return J


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("s8r2")
    s.add_argument("--seeds", required=True)
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "written once: " + str(out))
    jobs = s8r2([int(x) for x in a.seeds.split(",")])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
