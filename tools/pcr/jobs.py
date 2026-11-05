#!/usr/bin/env python3
"""PCR job list for tools/pcr/chain.py (docs/EXPERIMENT_RECORDS.md#pcr-prereg-20261105), written once.

    python -m tools.pcr.jobs r1 --out outputs/pcr/20261105/jobs/r1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pcr.common import PY, ROOT as R, ZGZ, require  # noqa: E402

TRAIN_MEM = 150                          # a router training (PFO's declaration for its router trainings)
ROUTERS = ("pcr", "pcrb")


def train(key, router, phase, deps):
    return {"key": key, "kind": "pair", "deps": list(deps), "mem_gib": TRAIN_MEM,
            "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY, str(R / "tools/pcr/train.py"),
                        "--router", router, "--phase", phase], "note": f"{router} {phase}"}


def gpu(key, args, deps, note=""):
    return {"key": key, "kind": "gpu", "deps": list(deps), "command": [PY, "-m", "tools.pcr.run", *args], "note": note}


def cpu(key, args, deps, note="", module="tools.pcr.run", mem=None):
    j = {"key": key, "kind": "cpu", "deps": list(deps), "command": [PY, "-m", module, *args], "note": note}
    if mem:
        j["mem_gib"] = mem
    return j


def r1(freeze="r1"):
    J = [cpu("eng:freeze", ["verify", "--stage", freeze], [], "manifest unchanged", module="tools.pcr.freeze"),
         cpu("eng:consistency", ["consistency"], ["eng:freeze"], "prior map vs events", mem=40)]
    for r in ROUTERS:
        J.append(train(f"pre:{r}", r, "preflight", ["eng:consistency"]))
        J.append(cpu(f"pre:accept:{r}", ["accept", "--router", r, "--phase", "preflight"], [f"pre:{r}"], "preflight"))
    for r in ROUTERS:
        J.append(train(f"train:{r}", r, "formal", [f"pre:accept:{r}"]))
        J.append(cpu(f"accept:{r}", ["accept", "--router", r, "--phase", "formal"], [f"train:{r}"], "acceptance"))
    J.append(gpu("k:l1:r_s2d:e1", ["read", "--trunk", "l1", "--router", "r_s2d", "--task", "e1"], ["eng:freeze"], "K"))
    J.append(cpu("k:check", ["check"], ["k:l1:r_s2d:e1"], "K"))
    readings = ["k:check"]
    for task in ("e3", "pert"):
        J.append(gpu(f"read:l1:r_s2d:{task}", ["read", "--trunk", "l1", "--router", "r_s2d", "--task", task], ["k:check"]))
        readings.append(f"read:l1:r_s2d:{task}")
    J.append(gpu("read:l1:r_s2d:f1", ["read", "--trunk", "l1", "--router", "r_s2d", "--task", "f1"], ["k:check"]))
    readings.append("read:l1:r_s2d:f1")
    for seq in ZGZ:
        J.append(gpu(f"probe:r_s2d:{seq}", ["probe", "--router", "r_s2d", "--seq", seq, "--prior", "loop"], ["eng:freeze"]))
        readings.append(f"probe:r_s2d:{seq}")
    for r in ROUTERS:
        for seq in ZGZ:
            for prior in ("loop", "gt"):
                key = f"probe:{r}:{seq}:{prior}"
                J.append(gpu(key, ["probe", "--router", r, "--seq", seq, "--prior", prior], [f"accept:{r}"]))
                readings.append(key)
        for task in ("e1", "e3", "pert"):
            key = f"read:l1:{r}:{task}"
            J.append(gpu(key, ["read", "--trunk", "l1", "--router", r, "--task", task], [f"accept:{r}", "k:check"]))
            readings.append(key)
        J.append(gpu(f"read:l1:{r}:f1", ["read", "--trunk", "l1", "--router", r, "--task", "f1"], [f"read:l1:{r}:e1"]))
        readings.append(f"read:l1:{r}:f1")
        J.append(gpu(f"read:b2:{r}:e1", ["read", "--trunk", "b2", "--router", r, "--task", "e1"], [f"accept:{r}"]))
        readings.append(f"read:b2:{r}:e1")
    J.append(cpu("final:analyze", ["all"], readings, "written readouts", module="tools.pcr.analyze"))
    keys = [j["key"] for j in J]
    require(len(keys) == len(set(keys)), "duplicate job keys")
    seen = set()
    for j in J:
        require(all(d in seen for d in j["deps"]), "dependency before its job: " + j["key"])
        seen.add(j["key"])
    return J


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("r1",))
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "written once: " + str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = r1()
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
