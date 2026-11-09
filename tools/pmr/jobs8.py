#!/usr/bin/env python3
"""PMR stage S5 job list (docs/EXPERIMENT_RECORDS.md#pmr-s5-prereg-20261109): the refinement S4 selected, written once and
appended with `chain.py add`. Generating the list writes the registered configs.

    python -m tools.pmr.jobs8 s5 --arm d1dual|d1hp|d1pos --freeze s5 --out outputs/pmr/20261109/jobs/s5.json

From scratch (tools/pmr/train4.py, B2's recipe, 6000 steps): the refinement X and, when its node inputs differ from D1's
(dual, pos), its no-P control at seeds 3408 / 3409 (hp reuses D1's R-noprop seeds); acceptance (tools/pmr/run4.py /
run5.py); e1 / e3 /
pert, f1 after e1. Rollout fine-tunes (tools/pmr/train_rol2.py, rol5): one preflight per new rollout arm at 3407, X + rol5
at 3407 / 3408 / 3409, and for a refinement with its own controls (dual) its R-noprop / R-mlp + rol5 at 3407 (the paper
table; hp reuses D1's); acceptance (tools/pmr/run_rol2.py); readings; the cost reading of the refined architecture
(tools/pmr/cost4.py, GPU 4 exclusive, default flags); the written verdict (tools/pmr/analyze7.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms_any as U  # noqa: E402
from tools.pmr import arms_rol2 as R2  # noqa: E402
from tools.pmr.common import PY, ROOT as RR, require  # noqa: E402
from tools.pmr.jobs import TASKS, TRAIN_MEM, cpu, gpu  # noqa: E402
from tools.pmr.jobs4 import ROL_MEM  # noqa: E402

CONTROLS = {"d1dual": ("d1dualn", "d1dualm"), "d1hp": None, "d1pos": ("d1posn", "d1posm")}
#: the stage whose entries and registry train / read a refinement from scratch
ENTRY = {"d1dual": ("tools/pmr/train4.py", "tools.pmr.run4", "s4"), "d1hp": ("tools/pmr/train4.py", "tools.pmr.run4", "s4"),
         "d1pos": ("tools/pmr/train5.py", "tools.pmr.run5", "s4b")}
#: a control's refinement (its stage and entries are its refinement's)
CONTROL_OF = {c: x for x, cs in CONTROLS.items() if cs for c in cs}


def scratch(key, arm, seed, deps, entry):
    return {"key": key, "kind": "pair", "deps": list(deps), "mem_gib": TRAIN_MEM,
            "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "NCCL_P2P_DISABLE=1", PY,
                        str(RR / entry), "--config", str(U.write_config(arm, seed, "formal"))],
            "note": f"{arm} formal s{seed}"}


def rollout(key, name, seed, phase, deps):
    return {"key": key, "kind": "pair", "deps": list(deps), "mem_gib": ROL_MEM,
            "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "NCCL_P2P_DISABLE=1", PY,
                        str(RR / "tools/pmr/train_rol2.py"), "--config", str(R2.write_config(name, seed, phase))],
            "note": f"{name} {phase} s{seed}"}


def base_accept(arm, seed):
    if seed == 3407:
        return f"{ENTRY[arm][2]}:accept:{arm}" if arm in ENTRY else f"{ENTRY[CONTROL_OF[arm]][2]}:accept:{arm}"
    return f"s5:accept:{arm}:s{seed}"


def s5(arm, freeze):
    require(arm in CONTROLS, "S5 takes an S4 refinement")
    ctrl = CONTROLS[arm]
    train_entry, run_entry, _ = ENTRY[arm]
    p = "s5"
    J = [cpu(f"{p}:freeze", ["verify", "--stage", freeze], [], "manifest unchanged", module="tools.pmr.freeze")]
    new = [arm] + ([ctrl[0]] if ctrl else [])
    for seed in (3408, 3409):
        for a in new:
            J.append(scratch(f"{p}:train:{a}:s{seed}", a, seed, [f"{p}:freeze"], train_entry))
            J.append(cpu(f"{p}:accept:{a}:s{seed}", ["accept", "--arm", a, "--phase", "formal", "--seed", str(seed)],
                         [f"{p}:train:{a}:s{seed}"], "acceptance", module=run_entry))
    reads = []
    for seed in (3408, 3409):
        for a in new:
            for task in TASKS:
                J.append(gpu(f"{p}:read:{a}:s{seed}:{task}", ["read", "--arm", a, "--seed", str(seed), "--task", task],
                             [f"{p}:accept:{a}:s{seed}"], module=run_entry))
                reads.append(J[-1]["key"])
            J.append(gpu(f"{p}:read:{a}:s{seed}:f1", ["read", "--arm", a, "--seed", str(seed), "--task", "f1"],
                         [f"{p}:read:{a}:s{seed}:e1"], module=run_entry))
            reads.append(J[-1]["key"])
    fine = [(f"{arm}_rol5", s, arm) for s in (3407, 3408, 3409)]
    if ctrl:
        fine += [(f"{c}_rol5", 3407, c) for c in ctrl]
    for base in [arm] + (list(ctrl) if ctrl else []):
        name = f"{base}_rol5"
        J.append(rollout(f"{p}:rpre:{name}", name, 3407, "preflight", [f"{p}:freeze", base_accept(base, 3407)]))
        J.append(cpu(f"{p}:rpre:accept:{name}", ["accept", "--arm", name, "--seed", "3407", "--phase", "preflight"],
                     [f"{p}:rpre:{name}"], "preflight acceptance", module="tools.pmr.run_rol2"))
    for name, seed, base in fine:
        J.append(rollout(f"{p}:rtrain:{name}:s{seed}", name, seed, "formal",
                         [f"{p}:rpre:accept:{name}", base_accept(base, seed)]))
        J.append(cpu(f"{p}:raccept:{name}:s{seed}", ["accept", "--arm", name, "--seed", str(seed), "--phase", "formal"],
                     [f"{p}:rtrain:{name}:s{seed}"], "acceptance", module="tools.pmr.run_rol2"))
        for task in TASKS:
            J.append(gpu(f"{p}:rread:{name}:s{seed}:{task}", ["read", "--arm", name, "--seed", str(seed), "--task", task],
                         [f"{p}:raccept:{name}:s{seed}"], module="tools.pmr.run_rol2"))
            reads.append(J[-1]["key"])
        J.append(gpu(f"{p}:rread:{name}:s{seed}:f1", ["read", "--arm", name, "--seed", str(seed), "--task", "f1"],
                     [f"{p}:rread:{name}:s{seed}:e1"], module="tools.pmr.run_rol2"))
        reads.append(J[-1]["key"])
    J.append({"key": f"{p}:cost:{arm}", "kind": "exclusive", "deps": reads, "mem_gib": 15,
              "note": "cost reading of the refined architecture (default flags)",
              "command": [PY, "-m", "tools.pmr.cost4", "run", "--arm", arm, "--policy", "default"]})
    J.append(cpu(f"{p}:verdict", ["s5", "--arm", arm], [f"{p}:cost:{arm}"], "written verdict", module="tools.pmr.analyze7"))
    keys = [j["key"] for j in J]
    require(len(keys) == len(set(keys)), "duplicate job keys")
    return J


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("s5")
    s.add_argument("--arm", required=True, choices=tuple(CONTROLS))
    s.add_argument("--freeze", required=True)
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "written once: " + str(out))
    jobs = s5(a.arm, a.freeze)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
