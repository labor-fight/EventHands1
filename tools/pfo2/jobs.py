#!/usr/bin/env python3
"""PFO2 job lists for tools/pfo2/chain.py (docs/EXPERIMENT_RECORDS.md#pfo2-prereg-20261103).

    python -m tools.pfo2.jobs s0 --out FILE       the stage-0 batch (written once; appended with `chain.py add`)

Stage 0:
  * F replay through PFO2's entry: the 40-step preflight (bitwise against PFO's F preflight; gates f1mp's training) and
    the first 500 formal steps (bitwise against PFO's F step-500 checkpoint; audit)
  * f1mp seeds 3407 / 3408 (trainings on pairs 2-3 / 4-5), acceptance, the standard readings
  * the base forms F + l1 (fl1) and F + gshare (fgs): preflights and the single-change reference runs (rl1 / rgs:
    preflight + the first 500 steps at seeds 3407-3409) on pair 6-7; the f1mp rule (`decide mp`) opens exactly one of
    the two formal chains (three seeds), whose acceptance needs the bitwise checks against the reference
  * eval-only readings of the terminated PFO runs f1self (seeds 3407 / 3408, steps 5000 / 5500) and f1noise (3407, 4000)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo2.arms import write_config  # noqa: E402
from tools.pfo2.common import PY, ROOT as R, SEEDS, require  # noqa: E402

READINGS = (("e1", 6000), ("e3", 6000), ("e1", 5000), ("e1", 5500), ("tl_e1", 6000), ("pert", 6000), ("f1", 6000))
FULL = ("e1", "e3", "tl_e1", "pert", "f1")
TRAIN_MEM = 130
SIDE = [[6, 7]]
MAIN = [[2, 3], [4, 5]]
ALL = [[2, 3], [4, 5], [6, 7]]


def train(key, arm, seed, phase, deps=(), pairs=None, note=""):
    job = {"key": key, "kind": "pair", "deps": list(deps), "mem_gib": TRAIN_MEM, "note": note,
           "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY, str(R / "tools/pfo2/train.py"),
                       "--config", str(write_config(arm, seed, phase))]}
    if pairs:
        job["pairs"] = pairs
    return job


def cpu(key, args, deps=(), note=""):
    return {"key": key, "kind": "cpu", "deps": list(deps), "command": [PY, "-m", *args], "note": note}


def gpu(key, args, deps=(), note=""):
    return {"key": key, "kind": "gpu", "deps": list(deps), "command": [PY, "-m", *args], "note": note}


def arm_readings(pre, arm, seed, accept_key):
    jobs, e1 = [], None
    for task, step in READINGS:
        k = f"{pre}:eval:{arm}:{seed}:{task}@{step}"
        jobs.append(gpu(k, ["tools.pfo2.run", "eval", "--arm", arm, "--seed", str(seed), "--task", task, "--step",
                            str(step)], [accept_key] + ([e1] if task == "f1" else [])))
        if (task, step) == ("e1", 6000):
            e1 = k
    return jobs


def stage0():
    J = []
    # F replay through the PFO2 entry (gate of f1mp's training) and its first 500 formal steps (audit)
    J.append(train("s0:pre:f1rep", "f1rep", 3407, "preflight", pairs=MAIN, note="F preflight replay"))
    J.append(cpu("s0:cmp:f1rep:pre", ["tools.pfo2.replay", "f1rep", "--phase", "preflight"], ["s0:pre:f1rep"]))
    J.append(train("s0:pre:f1mp", "f1mp", 3407, "preflight", ["s0:cmp:f1rep:pre"], MAIN))
    for s in (3407, 3408):
        J.append(train(f"s0:train:f1mp:{s}", "f1mp", s, "formal", ["s0:pre:f1mp"], MAIN))
        J.append(cpu(f"s0:accept:f1mp:{s}", ["tools.pfo2.run", "accept", "--arm", "f1mp", "--seed", str(s)],
                     [f"s0:train:f1mp:{s}"]))
        J += arm_readings("s0", "f1mp", s, f"s0:accept:f1mp:{s}")
    J.append(train("s0:head:f1rep", "f1rep", 3407, "head500", ["s0:cmp:f1rep:pre"], SIDE, "F first-500 replay"))
    J.append(cpu("s0:cmp:f1rep:head", ["tools.pfo2.replay", "f1rep", "--phase", "head500"], ["s0:head:f1rep"]))
    # the two base forms: preflights and single-change references (pair 6-7), then one formal chain opened by the rule
    for arm, ref in (("fl1", "rl1"), ("fgs", "rgs")):
        J.append(train(f"s0:pre:{ref}", ref, 3407, "preflight", ["s0:cmp:f1rep:pre"], SIDE))
        J.append(train(f"s0:pre:{arm}", arm, 3407, "preflight", ["s0:cmp:f1rep:pre"], SIDE))
        J.append(cpu(f"s0:cmp:{arm}:pre", ["tools.pfo2.replay", "base", "--arm", arm, "--seed", "3407", "--phase",
                                          "preflight"], [f"s0:pre:{ref}", f"s0:pre:{arm}"]))
        for s in SEEDS:
            J.append(train(f"s0:head:{ref}:{s}", ref, s, "head500", [f"s0:cmp:{arm}:pre"], SIDE))
    # eval-only readings of the terminated PFO runs (descriptive, not at step 6000)
    for arm, seed, step, tasks in (("f1self", 3407, 5000, ("e1",)), ("f1self", 3407, 5500, FULL),
                                   ("f1self", 3408, 5000, ("e1",)), ("f1self", 3408, 5500, FULL),
                                   ("f1noise", 3407, 4000, FULL)):
        e1 = None
        for task in tasks:
            k = f"s0:pfo:{arm}:{seed}:{task}@{step}"
            J.append(gpu(k, ["tools.pfo2.run", "evalpfo", "--arm", arm, "--seed", str(seed), "--task", task,
                             "--step", str(step)], [e1] if task == "f1" else []))
            if task == "e1":
                e1 = k
    # the f1mp rule, then exactly one base chain
    J.append(cpu("s0:decide:mp", ["tools.pfo2.decide", "mp"],
                 [f"s0:eval:f1mp:{s}:e1@6000" for s in (3407, 3408)]))
    for arm, ref, form in (("fl1", "rl1", "l1"), ("fgs", "rgs", "gshare")):
        g = f"s0:gate:{arm}"
        J.append(cpu(g, ["tools.pfo2.decide", "gate", "--rule", "mp", "--open", form], ["s0:decide:mp"],
                     "exit 3 = chain closed by the written rule"))
        for s in SEEDS:
            t = f"s0:train:{arm}:{s}"
            J.append(train(t, arm, s, "formal", [g, f"s0:cmp:{arm}:pre"], ALL))
            c = f"s0:cmp:{arm}:{s}:head"
            J.append(cpu(c, ["tools.pfo2.replay", "base", "--arm", arm, "--seed", str(s), "--phase", "head500"],
                         [t, f"s0:head:{ref}:{s}"]))
            acc = f"s0:accept:{arm}:{s}"
            J.append(cpu(acc, ["tools.pfo2.run", "accept", "--arm", arm, "--seed", str(s)], [t, c]))
            J += arm_readings("s0", arm, s, acc)
    keys = [j["key"] for j in J]
    require(len(keys) == len(set(keys)), "duplicate keys")
    return J


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("s0",))
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "job file exists")
    jobs = stage0()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
