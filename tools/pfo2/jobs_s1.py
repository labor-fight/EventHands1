#!/usr/bin/env python3
"""PFO2 stage-1 job lists (docs/EXPERIMENT_RECORDS.md#pfo2-s1-prereg-20261103) from a plan file (tools/pfo2/decide_s1.py
format plus "replay": the arm that trains B2's recipe through semkine/pfo2_trunk.py with no change).

    python -m tools.pfo2.jobs_s1 --plan tools/pfo2/plan_s1.json --name s1 --out FILE

  * the replay arm's preflight must equal B2's preflight bitwise (tools/pfo2/replay_s1.py) before any stage-1 training
  * every gnn / main arm (queue order: the full gnn arms, then the self controls, then the rewire controls, then main):
    40-step preflight (seed 3407), formal trainings at the plan's seeds (tools/pfo2/train_s1.py, deterministic),
    acceptance (tools/pfo2/run_s1.py), the standard readings
  * the stage-1 rules once every reading exists, the base's included (plan "base_deps"; tools/pfo2/decide_s1.py --final)
Trainings may use every pair the policy allows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo2 import jobs as J  # noqa: E402
from tools.pfo2.common import PY, ROOT as R, read, require  # noqa: E402
from tools.pfo2.reg1 import write_config  # noqa: E402


def train(key, arm, seed, phase, deps=()):
    return {"key": key, "kind": "pair", "deps": list(deps), "mem_gib": J.TRAIN_MEM, "pairs": J.ALL,
            "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY, str(R / "tools/pfo2/train_s1.py"),
                        "--config", str(write_config(arm, seed, phase))]}


def readings(name, arm, seed, accept_key):
    jobs, e1 = [], None
    for task, step in J.READINGS:
        k = f"{name}:eval:{arm}:{seed}:{task}@{step}"
        jobs.append(J.gpu(k, ["tools.pfo2.run_s1", "eval", "--arm", arm, "--seed", str(seed), "--task", task, "--step",
                              str(step)], [accept_key] + ([e1] if task == "f1" else [])))
        if (task, step) == ("e1", 6000):
            e1 = k
    return jobs


def build(plan, name, plan_path):
    gnn = plan.get("gnn", {}).values()
    arms = [fam[r] for r in ("full", "self", "rewire") for fam in gnn] + list(plan.get("main", {}).values())
    L = [train(f"{name}:pre:{plan['replay']}", plan["replay"], 3407, "preflight")]
    gate = f"{name}:cmp:replay"
    L.append(J.cpu(gate, ["tools.pfo2.replay_s1", "--arm", plan["replay"], "--base", plan["base"]],
                   [f"{name}:pre:{plan['replay']}"]))
    final = []
    for arm in arms:
        pre = f"{name}:pre:{arm}"
        L.append(train(pre, arm, 3407, "preflight", [gate]))
        for s in plan["seeds"]:
            t = f"{name}:train:{arm}:{s}"
            L.append(train(t, arm, s, "formal", [pre]))
            acc = f"{name}:accept:{arm}:{s}"
            L.append(J.cpu(acc, ["tools.pfo2.run_s1", "accept", "--arm", arm, "--seed", str(s)], [t]))
            L += readings(name, arm, s, acc)
            final += [f"{name}:eval:{arm}:{s}:{t_}@6000" for t_ in ("e1", "e3", "pert")]
    L.append(J.cpu(f"{name}:decide", ["tools.pfo2.decide_s1", "--plan", str(plan_path), "--final"],
                   final + list(plan.get("base_deps", []))))
    keys = [j["key"] for j in L]
    require(len(keys) == len(set(keys)), "duplicate keys")
    return L


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "job file exists")
    jobs = build(read(a.plan), a.name, Path(a.plan).resolve())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
