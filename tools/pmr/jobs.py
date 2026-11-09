#!/usr/bin/env python3
"""PMR job lists for tools/pmr/chain.py (docs/EXPERIMENT_RECORDS.md#pmr-prereg-20261109), written once and appended with
`chain.py add`. Generating a list writes the registered configs it trains.

    python -m tools.pmr.jobs screen --design d1 --freeze s1d1 [--k1] --out outputs/pmr/20261109/jobs/s1_d1.json

The S1 screen of one design (seed 3407), in queue order: freeze check; [K1: B2's e1 through this study's entry, bitwise
against PFO2's]; the three preflights (pair, 40 steps) and their acceptance against B2's preflight data stream; the three
formal trainings (pair, 6000 steps, each gated on its preflight acceptance [and K1]); acceptance; the readings
(e1 / e3 / pert, f1 after e1; R also e1 / e3 with every message zeroed at test time); the mechanism reading (d) of R and
of R-noprop (capture on a GPU, analysis on the CPU); the cost block (GPU 4 exclusive, PyTorch default flags: B2, R,
R-noprop, R-mlp), then the written screen.
Trainings run with NCCL_P2P_DISABLE=1: the GPU peer-to-peer transport hangs on this machine since 2026-11-08 (a bare
2-GPU all-reduce hangs, with the variable it completes); for a 2-rank all-reduce each element is one sum, so the
transport does not change the numerics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms as A  # noqa: E402
from tools.pmr.common import PY, ROOT as R, SEED, require  # noqa: E402

TRAIN_MEM = 130
TASKS = ("e1", "e3", "pert")


def train(key, arm, phase, deps, seed=SEED):
    return {"key": key, "kind": "pair", "deps": list(deps), "mem_gib": TRAIN_MEM,
            "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "NCCL_P2P_DISABLE=1", PY,
                        str(R / "tools/pmr/train.py"),
                        "--config", str(A.write_config(arm, seed, phase))],
            "note": f"{arm} {phase} s{seed}"}


def gpu(key, args, deps, note="", module="tools.pmr.run"):
    return {"key": key, "kind": "gpu", "deps": list(deps), "command": [PY, "-m", module, *args], "note": note}


def cpu(key, args, deps, note="", module="tools.pmr.run"):
    return {"key": key, "kind": "cpu", "deps": list(deps), "command": [PY, "-m", module, *args], "note": note}


def exclusive(key, args, deps, note=""):
    return {"key": key, "kind": "exclusive", "deps": list(deps), "mem_gib": 15,
            "command": [PY, "-m", "tools.pmr.cost", *args], "note": note}


def screen(design, freeze, k1):
    require(design in A.DESIGNS, "unregistered design")
    p = f"s1:{design}"
    arms = [f"{design}{c}" for c in A.VARIANTS]
    J = [cpu(f"{p}:freeze", ["verify", "--stage", freeze], [], "manifest unchanged", module="tools.pmr.freeze")]
    gate = []
    if k1:
        J.append(gpu("k1:b2", ["b2", "--task", "e1"], [f"{p}:freeze"], "K1 reading"))
        J.append(cpu("k1:check", ["check"], ["k1:b2"], "K1"))
        gate = ["k1:check"]
    for arm in arms:
        J.append(train(f"{p}:pre:{arm}", arm, "preflight", [f"{p}:freeze"]))
        J.append(cpu(f"{p}:pre:accept:{arm}", ["accept", "--arm", arm, "--phase", "preflight"], [f"{p}:pre:{arm}"],
                     "preflight acceptance"))
    for arm in arms:
        J.append(train(f"{p}:train:{arm}", arm, "formal", [f"{p}:pre:accept:{arm}", *gate]))
        J.append(cpu(f"{p}:accept:{arm}", ["accept", "--arm", arm, "--phase", "formal"], [f"{p}:train:{arm}"],
                     "acceptance"))
    reads = []
    for arm in arms:
        for task in TASKS:
            J.append(gpu(f"{p}:read:{arm}:{task}", ["read", "--arm", arm, "--task", task], [f"{p}:accept:{arm}"]))
            reads.append(J[-1]["key"])
        J.append(gpu(f"{p}:read:{arm}:f1", ["read", "--arm", arm, "--task", "f1"], [f"{p}:read:{arm}:e1"]))
        reads.append(J[-1]["key"])
    r_arm, n_arm = f"{design}r", f"{design}n"
    for task in ("e1", "e3"):
        J.append(gpu(f"{p}:read:{r_arm}:nomsg:{task}", ["read", "--arm", r_arm, "--task", task, "--messages", "off"],
                     [f"{p}:accept:{r_arm}"], "frozen message ablation (descriptive)"))
        reads.append(J[-1]["key"])
    for arm in (r_arm, n_arm):
        J.append(gpu(f"{p}:mech:cap:{arm}", ["capture", "--arm", arm], [f"{p}:accept:{arm}"], "mechanism (d)",
                     module="tools.pmr.mech"))
        J.append(cpu(f"{p}:mech:an:{arm}", ["analyze", "--arm", arm], [f"{p}:mech:cap:{arm}"], "mechanism (d)",
                     module="tools.pmr.mech"))
        reads.append(J[-1]["key"])
    last = list(reads)
    if k1:
        last.append("k1:check")
    costs = [("b2", ["run", "--net", "b2", "--policy", "default"])] if k1 else []
    costs += [(arm, ["run", "--arm", arm, "--policy", "default"]) for arm in arms]
    prev = None
    for name, args in costs:
        key = f"{p}:cost:{name}"
        J.append(exclusive(key, args, [prev] if prev else last, "cost block (default flags)"))
        prev = key
    J.append(cpu(f"{p}:screen", ["screen", "--design", design], [prev], "written screen", module="tools.pmr.analyze"))
    keys = [j["key"] for j in J]
    require(len(keys) == len(set(keys)), "duplicate job keys")
    seen = set()
    for j in J:
        require(all(d in seen for d in j["deps"]), "dependency before its job: " + j["key"])
        seen.add(j["key"])
    return J


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--design", required=True)
    s.add_argument("--freeze", required=True)
    s.add_argument("--k1", action="store_true")
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "written once: " + str(out))
    jobs = screen(a.design, a.freeze, a.k1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
