#!/usr/bin/env python3
"""ACU job lists for tools/acu/chain.py (docs/EXPERIMENT_RECORDS.md#acu-s1-prereg-20261104), written once per stage and
appended with `chain.py add`.

    python -m tools.acu.jobs s1a --out outputs/acu/20261104/jobs/s1a.json

s1a: the CA screens on seed 3407 (a1 = B2-gt + CA, a2 = B2 + CA): freeze check, preflight (40 steps) and its acceptance,
formal training (6000 steps) and its acceptance, readings e1 / e3 / pert / f1 / tl_e1 with the deployed router (r_s2d)
and, for a1, also with the label identity; the stage-1a decision (tools/acu/decide.py s1a) after every reading.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.acu.arms import write_config  # noqa: E402
from tools.acu.common import PY, ROOT as R, SCREEN_SEED, require  # noqa: E402

TRAIN_MEM = {"label": 260, "router": 130}          # a training that reads the XAG pseudo-sources declares 260 GiB
READ_TASKS = ("e1", "e3", "pert")


def train(key, arm, phase, routes, deps, pairs=None):
    j = {"key": key, "kind": "pair", "deps": list(deps), "mem_gib": TRAIN_MEM[routes],
         "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY, str(R / "tools/acu/train.py"),
                     "--config", str(write_config(arm, SCREEN_SEED, phase))],
         "note": f"{arm} {phase}"}
    if pairs:
        j["pairs"] = pairs
    return j


def gpu(key, args, deps, note=""):
    return {"key": key, "kind": "gpu", "deps": list(deps), "command": [PY, "-m", "tools.acu.run", *args], "note": note}


def cpu(key, args, deps, note="", module="tools.acu.run"):
    return {"key": key, "kind": "cpu", "deps": list(deps), "command": [PY, "-m", module, *args], "note": note}


def s1a(freeze="s1a"):
    J = [cpu("s1a:freeze", ["verify", "--stage", freeze], [], "manifest unchanged", module="tools.acu.freeze")]
    readings = []
    for arm, routes, pair in (("a1", "label", [[2, 3]]), ("a2", "router", [[4, 5]])):
        J.append(train(f"{arm}:pre", arm, "preflight", routes, ["s1a:freeze"], pair))
        J.append(cpu(f"{arm}:pre:accept", ["accept", "--arm", arm, "--phase", "preflight"], [f"{arm}:pre"]))
        J.append(train(f"{arm}:train", arm, "formal", routes, [f"{arm}:pre:accept"], pair))
        J.append(cpu(f"{arm}:accept", ["accept", "--arm", arm, "--phase", "formal"], [f"{arm}:train"]))
        routes_read = ("router", "label") if arm == "a1" else ("router",)
        for route in routes_read:
            for task in READ_TASKS:
                key = f"{arm}:{route}:{task}"
                J.append(gpu(key, ["eval", "--arm", arm, "--task", task, "--route", route], [f"{arm}:accept"], arm))
                readings.append(key)
            key = f"{arm}:{route}:f1"
            J.append(gpu(key, ["eval", "--arm", arm, "--task", "f1", "--route", route], [f"{arm}:{route}:e1"], arm))
            readings.append(key)
        key = f"{arm}:{routes_read[-1] if arm == 'a1' else 'router'}:tl_e1"
        J.append(gpu(key, ["eval", "--arm", arm, "--task", "tl_e1", "--route", "label" if arm == "a1" else "router"],
                     [f"{arm}:accept"], arm))
        readings.append(key)
    J.append(cpu("s1a:decide", ["s1a"], readings, "stage-1a decision (written rules)", module="tools.acu.decide"))
    return J


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("s1a",))
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    jobs = {"s1a": s1a}[a.stage]()
    out = Path(a.out)
    require(not out.exists(), "job list written once: " + str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
