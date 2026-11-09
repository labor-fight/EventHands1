#!/usr/bin/env python3
"""PMR stage S8 job list (docs/EXPERIMENT_RECORDS.md#pmr-s8-prereg-20261109): S5's pipeline (tools/pmr/jobs8.py) for an S6 /
S7 arm, written once and appended with `chain3.py add`. Generating the list writes the registered configs.

    python -m tools.pmr.jobs14 s8 --arm d1hub|d1ghandhp|d1ghandhub --freeze s8 --out outputs/pmr/20261109/jobs/s8.json

From scratch (the arm's stage entry, B2's recipe, 6000 steps): X at seeds 3408 / 3409 (its node inputs are D1's, so D1's
R-noprop seeds are its controls); acceptance; e1 / e3 / pert, f1 after e1. Rollout fine-tunes (tools/pmr/train_rol3.py,
rol5): one preflight at 3407, X + rol5 at 3407 / 3408 / 3409 on any pair (a rol5 fine-tune takes ~63 GiB, within a
borrowed job's cap); acceptance (tools/pmr/run_rol3.py); readings; X's cost reading (tools/pmr/cost6.py, GPU 4 exclusive,
default flags); the written verdict (tools/pmr/analyze11.py, S5's rule).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms_any3  # noqa: E402,F401
from tools.pmr import arms_any2 as U  # noqa: E402
from tools.pmr import arms_rol2 as R2  # noqa: E402
from tools.pmr.common import PY, ROOT as RR, require  # noqa: E402
from tools.pmr.jobs import TASKS, TRAIN_MEM, cpu, gpu  # noqa: E402
from tools.pmr.jobs4 import ROL_MEM  # noqa: E402

#: the stage whose entries train / read an arm from scratch and whose acceptance holds its seed 3407
ENTRY = {"d1hub": ("tools/pmr/train7.py", "tools.pmr.run7", "s6"),
         "d1ghandhp": ("tools/pmr/train8.py", "tools.pmr.run8", "s7"),
         "d1ghandhub": ("tools/pmr/train8.py", "tools.pmr.run8", "s7")}


def scratch(key, arm, seed, deps):
    return {"key": key, "kind": "pair", "deps": list(deps), "mem_gib": TRAIN_MEM,
            "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "NCCL_P2P_DISABLE=1", PY,
                        str(RR / ENTRY[arm][0]), "--config", str(U.write_config(arm, seed, "formal"))],
            "note": f"{arm} formal s{seed}"}


def rollout(key, name, seed, phase, deps):
    return {"key": key, "kind": "pair", "deps": list(deps), "mem_gib": ROL_MEM,
            "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "NCCL_P2P_DISABLE=1", PY,
                        str(RR / "tools/pmr/train_rol3.py"), "--config", str(R2.write_config(name, seed, phase))],
            "note": f"{name} {phase} s{seed}"}


def s8(arm, freeze):
    require(arm in ENTRY, "S8 takes an S6 / S7 arm")
    _, run_entry, stage = ENTRY[arm]
    p = "s8"

    def base_accept(seed):
        return f"{stage}:accept:{arm}" if seed == 3407 else f"{p}:accept:{arm}:s{seed}"
    J = [cpu(f"{p}:freeze", ["verify", "--stage", freeze], [], "manifest unchanged", module="tools.pmr.freeze")]
    for seed in (3408, 3409):
        J.append(scratch(f"{p}:train:{arm}:s{seed}", arm, seed, [f"{p}:freeze"]))
        J.append(cpu(f"{p}:accept:{arm}:s{seed}", ["accept", "--arm", arm, "--phase", "formal", "--seed", str(seed)],
                     [f"{p}:train:{arm}:s{seed}"], "acceptance", module=run_entry))
    reads = []
    for seed in (3408, 3409):
        for task in TASKS:
            J.append(gpu(f"{p}:read:{arm}:s{seed}:{task}", ["read", "--arm", arm, "--seed", str(seed), "--task", task],
                         [f"{p}:accept:{arm}:s{seed}"], module=run_entry))
            reads.append(J[-1]["key"])
        J.append(gpu(f"{p}:read:{arm}:s{seed}:f1", ["read", "--arm", arm, "--seed", str(seed), "--task", "f1"],
                     [f"{p}:read:{arm}:s{seed}:e1"], module=run_entry))
        reads.append(J[-1]["key"])
    name = f"{arm}_rol5"
    J.append(rollout(f"{p}:rpre:{name}", name, 3407, "preflight", [f"{p}:freeze", base_accept(3407)]))
    J.append(cpu(f"{p}:rpre:accept:{name}", ["accept", "--arm", name, "--seed", "3407", "--phase", "preflight"],
                 [f"{p}:rpre:{name}"], "preflight acceptance", module="tools.pmr.run_rol3"))
    for seed in (3407, 3408, 3409):
        J.append(rollout(f"{p}:rtrain:{name}:s{seed}", name, seed, "formal", [f"{p}:rpre:accept:{name}", base_accept(seed)]))
        J.append(cpu(f"{p}:raccept:{name}:s{seed}", ["accept", "--arm", name, "--seed", str(seed), "--phase", "formal"],
                     [f"{p}:rtrain:{name}:s{seed}"], "acceptance", module="tools.pmr.run_rol3"))
        for task in TASKS:
            J.append(gpu(f"{p}:rread:{name}:s{seed}:{task}", ["read", "--arm", name, "--seed", str(seed), "--task", task],
                         [f"{p}:raccept:{name}:s{seed}"], module="tools.pmr.run_rol3"))
            reads.append(J[-1]["key"])
        J.append(gpu(f"{p}:rread:{name}:s{seed}:f1", ["read", "--arm", name, "--seed", str(seed), "--task", "f1"],
                     [f"{p}:rread:{name}:s{seed}:e1"], module="tools.pmr.run_rol3"))
        reads.append(J[-1]["key"])
    J.append({"key": f"{p}:cost:{arm}", "kind": "exclusive", "deps": reads, "mem_gib": 15,
              "note": "cost reading of the arm (default flags)",
              "command": [PY, "-m", "tools.pmr.cost6", "run", "--arm", arm, "--policy", "default"]})
    J.append(cpu(f"{p}:verdict", ["s5", "--arm", arm], [f"{p}:cost:{arm}"], "written verdict", module="tools.pmr.analyze11"))
    keys = [j["key"] for j in J]
    require(len(keys) == len(set(keys)), "duplicate job keys")
    return J


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("s8")
    s.add_argument("--arm", required=True, choices=tuple(ENTRY))
    s.add_argument("--freeze", required=True)
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "written once: " + str(out))
    jobs = s8(a.arm, a.freeze)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
