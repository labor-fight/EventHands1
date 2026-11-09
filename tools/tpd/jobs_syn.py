#!/usr/bin/env python3
"""TPD synthesis job list for tools/tpd/chain.py (docs/EXPERIMENT_RECORDS.md#tpd-syn-prereg-20261105), written once.

    python -m tools.tpd.jobs_syn syn --out outputs/tpd/20261105/jobs/syn.json

Every training / reading is wrapped by tools/tpd/maybe_syn.py (a no-op unless the written rule opens it): seed 3407 of the
synthesis arm carrying the chosen ROL arm (after ROL's seed-3407 readout) and of the base c0, their preflights first;
the screen (after ROL's seed-3408 verdict); seed 3408 of both when it passes; the keep decision; seed 3409 of the final
arm; the final gate; the asynchronous-equivalence receipt of the final arm. Declared host memory: c0 200 GiB (B2's 16
workers, XAG sources), rollout arms 160 GiB (12 workers).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd.arms_final import ARMS, write_config  # noqa: E402
from tools.tpd.common import PY, ROOT as R, require  # noqa: E402

ROLL = ("c3d", "c3", "c5")
MEM = {"c0": 200, "c3d": 160, "c3": 160, "c5": 160}
TASKS = ("e1", "e3", "pert")


def wrap(cond, cmd):
    return [PY, "-m", "tools.tpd.maybe_syn", cond, "--", *cmd]


def train(key, arm, seed, phase, deps, cond):
    cmd = ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY, str(R / "tools/tpd/train_final.py"), "--config",
           str(write_config(arm, seed, phase))]
    return {"key": key, "kind": "pair", "deps": list(deps), "mem_gib": MEM[arm], "command": wrap(cond, cmd),
            "note": f"{arm} s{seed} {phase}"}


def cpu(key, args, deps, cond=None, module="tools.tpd.run_final", note=""):
    cmd = [PY, "-m", module, *args]
    return {"key": key, "kind": "cpu", "deps": list(deps), "command": wrap(cond, cmd) if cond else cmd, "note": note}


def gpu(key, args, deps, cond, note="reading"):
    return {"key": key, "kind": "gpu", "deps": list(deps), "command": wrap(cond, [PY, "-m", "tools.tpd.run_final", *args]),
            "note": note}


def runs(arm, seed, cond, deps, preflight):
    J, reads = [], []
    sfx = f"{arm}:s{seed}"
    if preflight:
        J.append(train(f"syn:pre:{arm}", arm, seed, "preflight", deps, cond))
        J.append(cpu(f"syn:pre:accept:{arm}", ["accept", "--arm", arm, "--seed", str(seed), "--phase", "preflight"],
                     [f"syn:pre:{arm}"], cond))
        deps = [f"syn:pre:accept:{arm}"]
    J.append(train(f"syn:train:{sfx}", arm, seed, "formal", deps, cond))
    J.append(cpu(f"syn:accept:{sfx}", ["accept", "--arm", arm, "--seed", str(seed), "--phase", "formal"],
                 [f"syn:train:{sfx}"], cond))
    for t in TASKS:
        J.append(gpu(f"syn:read:{sfx}:{t}", ["read", "--arm", arm, "--seed", str(seed), "--task", t],
                     [f"syn:accept:{sfx}"], cond))
        reads.append(f"syn:read:{sfx}:{t}")
    J.append(gpu(f"syn:read:{sfx}:f1", ["read", "--arm", arm, "--seed", str(seed), "--task", "f1"],
                 [f"syn:read:{sfx}:e1"], cond))
    reads.append(f"syn:read:{sfx}:f1")
    return J, reads


def stage():
    J = [cpu("syn:freeze", ["verify", "--stage", "syn"], [], module="tools.tpd.freeze", note="manifest unchanged")]
    reads1 = []
    for arm in ROLL:
        part, reads = runs(arm, 3407, f"chosen:{ARMS[arm]['rol']}", ["syn:freeze", "final:analyze"], True)
        J += part
        reads1 += reads
    part, reads = runs("c0", 3407, "any_chosen", ["syn:freeze", "final:analyze"], True)
    J += part
    reads1 += reads
    J.append(cpu("syn:screen", ["screen"], reads1 + ["final:analyze2"], cond="any_chosen", module="tools.tpd.analyze_syn",
                 note="written screen"))
    reads2 = []
    for arm in ("c0",) + ROLL:
        part, reads = runs(arm, 3408, f"s2:{arm}", ["syn:screen"], False)
        J += part
        reads2 += reads
    J.append(cpu("syn:keep", ["keep"], reads2, cond="any_chosen", module="tools.tpd.analyze_syn", note="keep decision"))
    reads3 = []
    for arm in ROLL:
        part, reads = runs(arm, 3409, f"s3:{arm}", ["syn:keep"], False)
        J += part
        reads3 += reads
    J.append(cpu("syn:final", ["final"], reads3, cond="s3:any", module="tools.tpd.analyze_syn", note="final gate"))
    for arm in ROLL:
        sp = str(R / "outputs/tpd/20261105/specs" / f"tpd_{arm}_s3407.json")
        J.append(cpu(f"syn:async:{arm}", ["--name", f"final_{arm}", "--spec", sp, "--causal"], ["syn:final"],
                     cond=f"s3:{arm}", module="tools.tpd.asynccheck", note="async receipt"))
    keys = [j["key"] for j in J]
    require(len(keys) == len(set(keys)), "duplicate job keys")
    return J


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("syn",))
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
