#!/usr/bin/env python3
"""PMR design D2 job list (docs/EXPERIMENT_RECORDS.md#pmr-s9-prereg-20261109), written once and appended with
`chain4.py add`. Generating the list writes the registered configs.

    python -m tools.pmr.jobs16 s9 --freeze s9 --out outputs/pmr/20261109/jobs/s9.json

d1hubfull from scratch (tools/pmr/train9.py, B2's recipe, 6000 steps) at seeds 3407 / 3408 / 3409 after one preflight
(pair, 40 steps); acceptance (tools/pmr/run9.py); e1 / e3 / pert, f1 after e1. Its node inputs and readout are D1's, so
D1's R-noprop / R-mlp (three seeds) are the controls. Rollout fine-tunes (tools/pmr/train_rol4.py, rol5) on the direct
pairs only (#pmr-amend8-20261109): one preflight at 3407, then 3407 / 3408 / 3409; acceptance (tools/pmr/run_rol4.py);
readings; the cost reading (tools/pmr/cost7.py, GPU 4 exclusive, default flags); the written verdict (tools/pmr/analyze12.py,
S5's rule).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms_any4  # noqa: E402,F401
from tools.pmr import arms9 as A9  # noqa: E402
from tools.pmr import arms_rol2 as R2  # noqa: E402
from tools.pmr.common import PY, ROOT as RR, require  # noqa: E402
from tools.pmr.jobs import TASKS, TRAIN_MEM, cpu, gpu  # noqa: E402
from tools.pmr.jobs4 import ROL_MEM  # noqa: E402

ARM = "d1hubfull"
DIRECT = [[4, 5], [6, 7]]


def env_cmd(script, config):
    return ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "NCCL_P2P_DISABLE=1", PY, str(RR / script),
            "--config", str(config)]


def s9(freeze):
    p = "s9"
    J = [cpu(f"{p}:freeze", ["verify", "--stage", freeze], [], "manifest unchanged", module="tools.pmr.freeze"),
         {"key": f"{p}:pre:{ARM}", "kind": "pair", "deps": [f"{p}:freeze"], "mem_gib": TRAIN_MEM,
          "command": env_cmd("tools/pmr/train9.py", A9.write_config(ARM, 3407, "preflight")), "note": f"{ARM} preflight"}]
    J.append(cpu(f"{p}:pre:accept:{ARM}", ["accept", "--arm", ARM, "--phase", "preflight"], [f"{p}:pre:{ARM}"],
                 "preflight acceptance", module="tools.pmr.run9"))
    reads = []
    for seed in (3407, 3408, 3409):
        J.append({"key": f"{p}:train:{ARM}:s{seed}", "kind": "pair", "deps": [f"{p}:pre:accept:{ARM}"], "mem_gib": TRAIN_MEM,
                  "command": env_cmd("tools/pmr/train9.py", A9.write_config(ARM, seed, "formal")),
                  "note": f"{ARM} formal s{seed}"})
        J.append(cpu(f"{p}:accept:{ARM}:s{seed}", ["accept", "--arm", ARM, "--phase", "formal", "--seed", str(seed)],
                     [f"{p}:train:{ARM}:s{seed}"], "acceptance", module="tools.pmr.run9"))
        for task in (*TASKS, "f1"):
            dep = f"{p}:read:{ARM}:s{seed}:e1" if task == "f1" else f"{p}:accept:{ARM}:s{seed}"
            J.append(gpu(f"{p}:read:{ARM}:s{seed}:{task}", ["read", "--arm", ARM, "--seed", str(seed), "--task", task],
                         [dep], module="tools.pmr.run9"))
            reads.append(J[-1]["key"])
    name = f"{ARM}_rol5"
    J.append({"key": f"{p}:rpre:{name}", "kind": "pair", "deps": [f"{p}:accept:{ARM}:s3407"], "mem_gib": ROL_MEM,
              "pairs": DIRECT, "command": env_cmd("tools/pmr/train_rol4.py", R2.write_config(name, 3407, "preflight")),
              "note": f"{name} preflight"})
    J.append(cpu(f"{p}:rpre:accept:{name}", ["accept", "--arm", name, "--seed", "3407", "--phase", "preflight"],
                 [f"{p}:rpre:{name}"], "preflight acceptance", module="tools.pmr.run_rol4"))
    for seed in (3407, 3408, 3409):
        J.append({"key": f"{p}:rtrain:{name}:s{seed}", "kind": "pair", "mem_gib": ROL_MEM, "pairs": DIRECT,
                  "deps": [f"{p}:rpre:accept:{name}", f"{p}:accept:{ARM}:s{seed}"],
                  "command": env_cmd("tools/pmr/train_rol4.py", R2.write_config(name, seed, "formal")),
                  "note": f"{name} formal s{seed}"})
        J.append(cpu(f"{p}:raccept:{name}:s{seed}", ["accept", "--arm", name, "--seed", str(seed), "--phase", "formal"],
                     [f"{p}:rtrain:{name}:s{seed}"], "acceptance", module="tools.pmr.run_rol4"))
        for task in (*TASKS, "f1"):
            dep = f"{p}:rread:{name}:s{seed}:e1" if task == "f1" else f"{p}:raccept:{name}:s{seed}"
            J.append(gpu(f"{p}:rread:{name}:s{seed}:{task}", ["read", "--arm", name, "--seed", str(seed), "--task", task],
                         [dep], module="tools.pmr.run_rol4"))
            reads.append(J[-1]["key"])
    J.append({"key": f"{p}:cost:{ARM}", "kind": "exclusive", "deps": reads, "mem_gib": 15,
              "note": "cost reading of the arm (default flags)",
              "command": [PY, "-m", "tools.pmr.cost7", "run", "--arm", ARM, "--policy", "default"]})
    J.append(cpu(f"{p}:verdict", ["s5", "--arm", ARM], [f"{p}:cost:{ARM}"], "written verdict", module="tools.pmr.analyze12"))
    keys = [j["key"] for j in J]
    require(len(keys) == len(set(keys)), "duplicate job keys")
    return J


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("s9")
    s.add_argument("--freeze", required=True)
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "written once: " + str(out))
    jobs = s9(a.freeze)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
