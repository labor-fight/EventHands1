#!/usr/bin/env python3
"""EVF job lists from round 2 on: tools/evf/jobs2.py's builders (frozen) with acceptance by tools/evf/run3.py
(training prior-distribution arms); readings by run2, rules by decide2. (Original docstring follows.)

EVF job lists of the second registry (tools/evf/arms2.py: train2 / run2 / decide2; tools/evf/jobs.py's builders,
frozen with round 1). EVF job lists for tools/evf/chain.py (docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104), written once per stage and
appended with `chain.py add`; never all seeds at once.

    python -m tools.evf.jobs3 eng --out FILE                engineering: two 3-step runs (bitwise), 50-step smokes
    python -m tools.evf.jobs3 screen --round R --arms a,b --out FILE [--control c]
        one screening round: B2 seed 3407 at step 2000 (e1 / e3 / pert, once per study), each arm's 2000-step run at
        seed 3407 with acceptance and the same readings, the written screen, and for every passing arm (a gate, exit 3
        = closed by the rule) the formal 6000-step run, its readings and the single-seed effectiveness rule;
        `--control` arms are trained and read like the others but never screened (comparison only)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.evf.arms2 import write_config  # noqa: E402
from tools.evf.common import PY, ROOT as R, SCREEN_SEED, require  # noqa: E402

TRAIN_MEM = 130
SCREEN_TASKS = ("e1", "e3", "pert")
FORMAL_READINGS = (("e1", 6000), ("e3", 6000), ("pert", 6000), ("tl_e1", 6000), ("f1", 6000), ("e1", 5000),
                   ("e1", 5500))


def train(key, arm, seed, phase, deps=(), pairs=None, note=""):
    job = {"key": key, "kind": "pair", "deps": list(deps), "mem_gib": TRAIN_MEM, "note": note,
           "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY, str(R / "tools/evf/train2.py"),
                       "--config", str(write_config(arm, seed, phase))]}
    if pairs:
        job["pairs"] = pairs
    return job


def cpu(key, args, deps=(), note=""):
    return {"key": key, "kind": "cpu", "deps": list(deps), "command": [PY, "-m", *args], "note": note}


def gpu(key, args, deps=(), note=""):
    return {"key": key, "kind": "gpu", "deps": list(deps), "command": [PY, "-m", *args], "note": note}


def engineering(arm, control, freeze):
    J = [cpu("eng:freeze", ["tools.evf.freeze", "verify", "--stage", freeze])]
    J.append(train(f"eng:smoke3:{arm}", arm, SCREEN_SEED, "smoke3", ["eng:freeze"]))
    J.append(train(f"eng:smoke3b:{arm}", arm, SCREEN_SEED, "smoke3b", ["eng:freeze"]))
    J.append(cpu(f"eng:cmp3:{arm}", ["tools.evf.replay2", "smoke3", "--arm", arm, "--seed", str(SCREEN_SEED)],
                 [f"eng:smoke3:{arm}", f"eng:smoke3b:{arm}"]))
    for a in (arm, control):
        J.append(train(f"eng:smoke50:{a}", a, SCREEN_SEED, "smoke50", [f"eng:cmp3:{arm}"]))
    return J


def screen_round(rnd, arms, controls, freeze, b2_once):
    s = str(SCREEN_SEED)
    J = [cpu(f"{rnd}:freeze", ["tools.evf.freeze", "verify", "--stage", freeze])]
    b2 = []
    if b2_once:
        for task in SCREEN_TASKS:
            k = f"b2:{task}@2000"
            J.append(gpu(k, ["tools.evf.run2", "evalb2", "--seed", s, "--task", task, "--step", "2000"]))
            b2.append(k)
    else:
        b2 = [f"b2:{task}@2000" for task in SCREEN_TASKS]
    for arm in list(arms) + list(controls):
        t, acc = f"{rnd}:screen:{arm}", f"{rnd}:accept:{arm}:screen"
        J.append(train(t, arm, SCREEN_SEED, "screen", [f"{rnd}:freeze"]))
        J.append(cpu(acc, ["tools.evf.run3", "accept", "--arm", arm, "--seed", s, "--phase", "screen"], [t]))
        reads = []
        for task in SCREEN_TASKS:
            k = f"{rnd}:eval:{arm}:screen:{task}@2000"
            J.append(gpu(k, ["tools.evf.run2", "eval", "--arm", arm, "--seed", s, "--phase", "screen", "--task", task,
                             "--step", "2000"], [acc]))
            reads.append(k)
        if arm in controls:
            continue
        d, g = f"{rnd}:decide:screen:{arm}", f"{rnd}:gate:screen:{arm}"
        J.append(cpu(d, ["tools.evf.decide2", "screen", "--arm", arm], reads + b2))
        J.append(cpu(g, ["tools.evf.decide2", "gate", "--arm", arm], [d], "exit 3 = stopped by the written screen"))
        f, facc = f"{rnd}:formal:{arm}", f"{rnd}:accept:{arm}:formal"
        J.append(train(f, arm, SCREEN_SEED, "formal", [g]))
        J.append(cpu(facc, ["tools.evf.run3", "accept", "--arm", arm, "--seed", s, "--phase", "formal"], [f]))
        J.append(cpu(f"{rnd}:prefix:{arm}", ["tools.evf.replay2", "prefix", "--arm", arm, "--seed", s], [facc]))
        e1, keys = None, []
        for task, step in FORMAL_READINGS:
            k = f"{rnd}:eval:{arm}:formal:{task}@{step}"
            J.append(gpu(k, ["tools.evf.run2", "eval", "--arm", arm, "--seed", s, "--phase", "formal", "--task", task,
                             "--step", str(step)], [facc] + ([e1] if task == "f1" else [])))
            if (task, step) == ("e1", 6000):
                e1 = k
            keys.append(k)
        J.append(cpu(f"{rnd}:decide:effective:{arm}", ["tools.evf.decide2", "effective", "--arm", arm], keys[:3]))
    keys = [j["key"] for j in J]
    require(len(keys) == len(set(keys)), "duplicate keys")
    return J


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("eng", "screen"))
    p.add_argument("--out", required=True)
    p.add_argument("--round", default="r1")
    p.add_argument("--arms", default="a")
    p.add_argument("--control", default="")
    p.add_argument("--freeze", required=True)
    p.add_argument("--b2-once", action="store_true", help="add B2's step-2000 readings (first screening round)")
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "job file exists")
    arms = [x for x in a.arms.split(",") if x]
    controls = [x for x in a.control.split(",") if x]
    jobs = engineering(arms[0], controls[0], a.freeze) if a.what == "eng" else \
        screen_round(a.round, arms, controls, a.freeze, a.b2_once)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
