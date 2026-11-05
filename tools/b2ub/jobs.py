#!/usr/bin/env python3
"""B2UB job list for tools/b2ub/chain.py (docs/EXPERIMENT_RECORDS.md#b2ub-prereg-20261104), written once and appended
with `chain.py add`. Every reading job depends on the engine checks it is interpreted against; the final readout depends
on every reading and check.

    python -m tools.b2ub.jobs r1 --out outputs/b2ub/20261104/jobs/r1.json

Order (the driver dispatches in queue order): freeze check; U3 preflight (pair); engine checks K1-K3; U3 preflight
acceptance; U3 formal training (pair, gated on the preflight acceptance and K1-K3); U1 and U2 readings of B2; K4; U3
acceptance, readings and clamps; the final readout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.b2ub.arms import write_config  # noqa: E402
from tools.b2ub.common import PY, ROOT as R, SEEDS, U3_SEED, require  # noqa: E402
from tools.b2ub.engine import CLAMPS  # noqa: E402

TRAIN_MEM = 260                          # a training that reads the XAG pseudo-sources (S45 studies' declaration)
MIXES = ("label", "label_loc", "label_part", "label_onhand")
U1_TASKS = ("e1", "e3", "pert")


def train(key, phase, deps):
    return {"key": key, "kind": "pair", "deps": list(deps), "mem_gib": TRAIN_MEM,
            "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY, str(R / "tools/b2ub/train.py"),
                        "--config", str(write_config("b2gt", U3_SEED, phase))],
            "note": f"U3 B2-gt {phase}"}


def gpu(key, args, deps, note=""):
    return {"key": key, "kind": "gpu", "deps": list(deps), "command": [PY, "-m", "tools.b2ub.run", *args], "note": note}


def cpu(key, args, deps, note="", module="tools.b2ub.run"):
    return {"key": key, "kind": "cpu", "deps": list(deps), "command": [PY, "-m", module, *args], "note": note}


def r1(freeze="r1"):
    J = [cpu("eng:freeze", ["verify", "--stage", freeze], [], "manifest unchanged", module="tools.b2ub.freeze")]
    J.append(train("u3:pre:train", "preflight", ["eng:freeze"]))
    s = str(U3_SEED)
    J.append(gpu("k:k1", ["u1", "--seed", s, "--route", "router", "--task", "e1"], ["eng:freeze"], "K1"))
    J.append(gpu("k:k2", ["f1label"], ["eng:freeze"], "K2"))
    J.append(gpu("k:k3:none", ["u2", "--seed", s, "--clamp", "none"], ["eng:freeze"], "K3 / U2"))
    J.append(gpu("k:k3:trf", ["u2", "--seed", s, "--clamp", "TRF"], ["eng:freeze"], "K3 / U2"))
    J.append(cpu("k:check", ["check", "--what", "k123"], ["k:k1", "k:k2", "k:k3:none", "k:k3:trf"], "K1-K3"))
    J.append(cpu("u3:pre:accept", ["accept", "--phase", "preflight"], ["u3:pre:train"], "U3 preflight"))
    J.append(train("u3:train", "formal", ["u3:pre:accept", "k:check"]))
    readings = ["k:check"]
    for seed in SEEDS:
        for route in MIXES:
            for task in U1_TASKS:
                key = f"u1:{route}:{seed}:{task}"
                J.append(gpu(key, ["u1", "--seed", str(seed), "--route", route, "--task", task], ["k:check"], "U1"))
                readings.append(key)
        key = f"u1:label:{seed}:f1"
        J.append(gpu(key, ["u1", "--seed", str(seed), "--route", "label", "--task", "f1"], [f"u1:label:{seed}:e1"], "U1"))
        readings.append(key)
    for seed in SEEDS:
        for clamp in CLAMPS:
            if seed == U3_SEED and clamp in ("none", "TRF"):
                continue                                             # K3's readings are U2's
            key = f"u2:dep:{seed}:{clamp}"
            J.append(gpu(key, ["u2", "--seed", str(seed), "--clamp", clamp], ["k:check"], "U2 deployed"))
            readings.append(key)
        for clamp in CLAMPS:
            key = f"u2:lab:{seed}:{clamp}"
            J.append(gpu(key, ["u2", "--seed", str(seed), "--clamp", clamp, "--route", "label"], ["k:check"], "U2 label"))
            readings.append(key)
    for seed in SEEDS:
        key = f"k:k4:{seed}"
        J.append(cpu(key, ["check", "--what", "k4", "--seed", str(seed)], [f"u2:lab:{seed}:none", f"u1:label:{seed}:e1"],
                     "K4"))
        readings.append(key)
    J.append(cpu("u3:accept", ["accept", "--phase", "formal"], ["u3:train"], "U3 acceptance"))
    u3 = [("label", "e1", 6000), ("label", "e3", 6000), ("label", "pert", 6000), ("label", "tl_e1", 6000),
          ("label", "e1", 5000), ("label", "e1", 5500), ("router", "e1", 6000), ("router", "e3", 6000),
          ("router", "pert", 6000)]
    for route, task, step in u3:
        key = f"u3:{route}:{task}@{step}"
        J.append(gpu(key, ["u3", "--task", task, "--route", route, "--step", str(step)], ["u3:accept"], "U3"))
        readings.append(key)
    J.append(gpu("u3:label:f1@6000", ["u3", "--task", "f1", "--route", "label"], ["u3:label:e1@6000"], "U3"))
    readings.append("u3:label:f1@6000")
    for clamp in CLAMPS:
        key = f"u3c:{clamp}"
        J.append(gpu(key, ["u3c", "--clamp", clamp], ["u3:accept"], "U3 clamps"))
        readings.append(key)
    J.append(cpu("k:u3k4", ["check", "--what", "u3k4"], ["u3c:none", "u3:label:e1@6000"], "K4 B2-gt"))
    readings.append("k:u3k4")
    J.append(cpu("final:analyze", ["all"], readings, "written readouts", module="tools.b2ub.analyze"))
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
