#!/usr/bin/env python3
"""TPD stage ROL job list for tools/tpd/chain.py (docs/EXPERIMENT_RECORDS.md#tpd-rol-prereg-20261105), written once and
appended with `chain.py add`. Generating the list writes the registered configs.

    python -m tools.tpd.jobs_rol rol --out outputs/tpd/20261105/jobs/rol.json

Order (the driver dispatches in queue order): freeze check; B2 3407's e1 through this study's entry and the K1 check;
the four preflights (pair, 40 steps) and their acceptance; the plan (step times, memory; rol5 allowed?); the formal
fine-tunes rol3, rol3d, rol1 (three pairs), then rol5 (wrapped by tools/tpd/maybe.py: a no-op unless the plan allows it);
acceptance; the rol3 / rol3d stream pairing check; the readings (e1 / e3 / pert, f1 after e1); the seed-3407 readout;
the seed-3408 runs of the chosen arm (and rol1 when the base was rebased), each wrapped by maybe.py; the seed-3408
readout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd.arms_rol import ARMS, write_config  # noqa: E402
from tools.tpd.common import PY, ROOT as R, require  # noqa: E402

TRAIN_MEM = 130
SEED_SCREEN, SEED_CONFIRM = 3407, 3408
ORDER = ("rol3", "rol3d", "rol1", "rol5")
TASKS = ("e1", "e3", "pert")


def wrap(condition, command):
    return [PY, "-m", "tools.tpd.maybe", condition, "--", *command] if condition else command


def train(key, arm, seed, phase, deps, condition=None):
    cmd = ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY, str(R / "tools/tpd/train_rol.py"), "--config",
           str(write_config(arm, seed, phase))]
    return {"key": key, "kind": "pair", "deps": list(deps), "mem_gib": TRAIN_MEM, "command": wrap(condition, cmd),
            "note": f"{arm} s{seed} {phase}"}


def gpu(key, args, deps, note="", condition=None):
    return {"key": key, "kind": "gpu", "deps": list(deps), "note": note,
            "command": wrap(condition, [PY, "-m", "tools.tpd.run_rol", *args])}


def cpu(key, args, deps, note="", module="tools.tpd.run_rol", condition=None):
    return {"key": key, "kind": "cpu", "deps": list(deps), "note": note,
            "command": wrap(condition, [PY, "-m", module, *args])}


def readings(arm, seed, deps, condition, sfx=""):
    J, keys = [], []
    for task in TASKS:
        key = f"read:{arm}{sfx}:{task}"
        J.append(gpu(key, ["read", "--arm", arm, "--seed", str(seed), "--task", task], deps, "reading", condition))
        keys.append(key)
    key = f"read:{arm}{sfx}:f1"
    J.append(gpu(key, ["read", "--arm", arm, "--seed", str(seed), "--task", "f1"], [f"read:{arm}{sfx}:e1"], "reading",
                 condition))
    return J, keys + [key]


def stage(seed=SEED_SCREEN):
    require(seed == SEED_SCREEN, "the screen runs seed 3407")
    J = [cpu("eng:freeze", ["verify", "--stage", "rol"], [], "manifest unchanged", module="tools.tpd.freeze")]
    J.append(gpu("k:b2", ["b2", "--seed", "3407", "--task", "e1"], ["eng:freeze"], "K1 reading"))
    J.append(cpu("k:check", ["check"], ["k:b2"], "K1"))
    for arm in ORDER:
        J.append(train(f"pre:{arm}", arm, seed, "preflight", ["eng:freeze"]))
        J.append(cpu(f"pre:accept:{arm}", ["accept", "--arm", arm, "--seed", str(seed), "--phase", "preflight"],
                     [f"pre:{arm}"], "preflight acceptance"))
    J.append(cpu("plan", ["plan"], [f"pre:accept:{a}" for a in ORDER], "step times / memory / rol5",
                 module="tools.tpd.analyze_rol"))
    for arm in ORDER:
        cond = "rol5" if arm == "rol5" else None
        deps = [f"pre:accept:{arm}", "k:check"] + (["plan"] if cond else [])
        J.append(train(f"train:{arm}", arm, seed, "formal", deps, cond))
        J.append(cpu(f"accept:{arm}", ["accept", "--arm", arm, "--seed", str(seed), "--phase", "formal"],
                     [f"train:{arm}"], "acceptance", condition=cond))
    J.append(cpu("pair", ["pair"], ["accept:rol3", "accept:rol3d"], "rol3 / rol3d stream"))
    reads = ["k:check", "pair"]
    for arm in ORDER:
        part, keys = readings(arm, seed, [f"accept:{arm}"], "rol5" if arm == "rol5" else None)
        J += part
        reads += keys
    J.append(cpu("final:analyze", ["all"], reads, "seed-3407 readout", module="tools.tpd.analyze_rol"))
    # seed 3408 of the chosen arm (and rol1 when rebased): every job is a no-op unless the written rule opens it
    reads2 = ["final:analyze"]
    for arm in ("rol3", "rol5", "rol3d", "rol1"):
        cond = f"s2:{arm}"
        J.append(train(f"train:{arm}:s{SEED_CONFIRM}", arm, SEED_CONFIRM, "formal", ["final:analyze"], cond))
        J.append(cpu(f"accept:{arm}:s{SEED_CONFIRM}", ["accept", "--arm", arm, "--seed", str(SEED_CONFIRM), "--phase",
                                                      "formal"], [f"train:{arm}:s{SEED_CONFIRM}"], "acceptance",
                     condition=cond))
        part, keys = readings(arm, SEED_CONFIRM, [f"accept:{arm}:s{SEED_CONFIRM}"], cond, f":s{SEED_CONFIRM}")
        J += part
        reads2 += keys
    J.append(cpu("final:analyze2", ["all2"], reads2, "seed-3408 readout", module="tools.tpd.analyze_rol"))
    keys = [j["key"] for j in J]
    require(len(keys) == len(set(keys)), "duplicate job keys")
    seen = set()
    for j in J:
        require(all(d in seen for d in j["deps"]), "dependency before its job: " + j["key"])
        seen.add(j["key"])
    return J


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("rol",))
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "written once: " + str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = stage()
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
