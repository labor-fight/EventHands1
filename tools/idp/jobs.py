#!/usr/bin/env python3
"""IDP job list for tools/idp/chain.py (docs/EXPERIMENT_RECORDS.md#idp-prereg-20261105), written once and appended with
`chain.py add`. Generating the list writes the six registered configs (three arms x preflight / formal).

    python -m tools.idp.jobs r1 --out outputs/idp/20261105/jobs/r1.json

Order (the driver dispatches in queue order): freeze check; the two engine-check readings of B2 (K1 router, K2
label_part); the K check; the three preflights (pair) and their acceptance; the three formal trainings (pair, each gated
on its preflight acceptance and the K check; a label_part training declares 260 GiB of host memory, so the two of them
never run together); acceptance; the readings (e1 / e3 / pert, f1 after e1; the label_part arms also e1 with the
deployed router); the readout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.idp.arms import ARMS, write_config  # noqa: E402
from tools.idp.common import PY, ROOT as R, SEED, require  # noqa: E402

TRAIN_MEM_XAG = 260                      # a training that reads the XAG pseudo-sources (S45 studies' declaration)
TRAIN_MEM = 130
ORDER = ("l3", "l1", "l2")
TASKS = ("e1", "e3", "pert")


def train(key, arm, phase, deps):
    mem = TRAIN_MEM_XAG if ARMS[arm]["train_routes"] == "label_part" else TRAIN_MEM
    return {"key": key, "kind": "pair", "deps": list(deps), "mem_gib": mem,
            "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY, str(R / "tools/idp/train.py"),
                        "--config", str(write_config(arm, SEED, phase))],
            "note": f"{arm} {phase}"}


def gpu(key, args, deps, note=""):
    return {"key": key, "kind": "gpu", "deps": list(deps), "command": [PY, "-m", "tools.idp.run", *args], "note": note}


def cpu(key, args, deps, note="", module="tools.idp.run"):
    return {"key": key, "kind": "cpu", "deps": list(deps), "command": [PY, "-m", module, *args], "note": note}


def r1(freeze="r1"):
    J = [cpu("eng:freeze", ["verify", "--stage", freeze], [], "manifest unchanged", module="tools.idp.freeze")]
    J.append(gpu("k:b2:router", ["b2", "--task", "e1", "--route", "router"], ["eng:freeze"], "K1"))
    J.append(gpu("k:b2:label_part", ["b2", "--task", "e1", "--route", "label_part"], ["eng:freeze"], "K2"))
    J.append(cpu("k:check", ["check"], ["k:b2:router", "k:b2:label_part"], "K1 / K2"))
    for arm in ORDER:
        J.append(train(f"pre:{arm}", arm, "preflight", ["eng:freeze"]))
        J.append(cpu(f"pre:accept:{arm}", ["accept", "--arm", arm, "--phase", "preflight"], [f"pre:{arm}"], "preflight"))
    for arm in ORDER:
        J.append(train(f"train:{arm}", arm, "formal", [f"pre:accept:{arm}", "k:check"]))
        J.append(cpu(f"accept:{arm}", ["accept", "--arm", arm, "--phase", "formal"], [f"train:{arm}"], "acceptance"))
    readings = ["k:check"]
    for arm in ORDER:
        for task in TASKS:
            key = f"read:{arm}:{task}"
            J.append(gpu(key, ["read", "--arm", arm, "--task", task], [f"accept:{arm}"], "reading"))
            readings.append(key)
        key = f"read:{arm}:f1"
        J.append(gpu(key, ["read", "--arm", arm, "--task", "f1"], [f"read:{arm}:e1"], "reading"))
        readings.append(key)
        if ARMS[arm]["train_routes"] == "label_part":
            key = f"read:{arm}:deployed:e1"
            J.append(gpu(key, ["read", "--arm", arm, "--task", "e1", "--deployed"], [f"accept:{arm}"], "exposure"))
            readings.append(key)
    J.append(cpu("final:analyze", ["all"], readings, "written readouts", module="tools.idp.analyze"))
    keys = [j["key"] for j in J]
    require(len(keys) == len(set(keys)), "duplicate job keys")
    seen = set()
    for j in J:
        require(all(d in seen for d in j["deps"]), "dependency before its job: " + j["key"])
        seen.add(j["key"])
    return J


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
