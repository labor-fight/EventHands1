#!/usr/bin/env python3
"""ACU stage-3 planners (docs/EXPERIMENT_RECORDS.md#acu-s3-prereg-20261104). Each planner is itself a driver job: it reads
the written decision it depends on, writes the next configs and job list once, and appends them to the queue.

    python -m tools.acu.jobs_s3 plan_a      after s2:decide: the 2x2 winner W with the nomp trunk (seed 3407): preflight,
                                            acceptance, formal training, acceptance, readings; then `decide_s3 trunk`
                                            and plan_b
    python -m tools.acu.jobs_s3 plan_b      after the trunk decision: W with the chosen trunk at seeds 3408 / 3409
                                            (preflight, training, acceptance, readings); the cost block (GPU 4 exclusive:
                                            S43 sparse executor, B2, W's deployed causal form; default flags, then the
                                            deterministic ones); the main rows; `decide_s3 final`; the final tables
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.acu import arms_s2 as A  # noqa: E402
from tools.acu.common import OUT, PY, ROOT as R, require  # noqa: E402

PAIRS = [[4, 5], [6, 7]]


def _train(key, cell, seed, phase, trunk, deps):
    return {"key": key, "kind": "pair", "deps": deps, "mem_gib": 200, "pairs": PAIRS,
            "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY, str(R / "tools/acu/train_s2.py"),
                        "--config", str(A.write_config(cell, seed, phase, trunk))], "note": f"s3 {cell} {trunk} {seed} {phase}"}


def _cpu(key, args, deps, module, mem=8):
    return {"key": key, "kind": "cpu", "deps": deps, "mem_gib": mem, "command": [PY, "-m", module, *args]}


def _gpu(key, args, deps, module="tools.acu.run_s2"):
    return {"key": key, "kind": "gpu", "deps": deps, "command": [PY, "-m", module, *args]}


def _cell_block(prefix, cell, seed, trunk, deps):
    J = [_train(f"{prefix}:pre", cell, seed, "preflight", trunk, deps),
         _cpu(f"{prefix}:pre:accept", ["accept", "--cell", cell, "--seed", str(seed), "--trunk", trunk, "--phase", "preflight"],
              [f"{prefix}:pre"], "tools.acu.run_s2", mem=24),
         _train(f"{prefix}:train", cell, seed, "formal", trunk, [f"{prefix}:pre:accept"]),
         _cpu(f"{prefix}:accept", ["accept", "--cell", cell, "--seed", str(seed), "--trunk", trunk, "--phase", "formal"],
              [f"{prefix}:train"], "tools.acu.run_s2", mem=24)]
    reads = []
    base = ["eval", "--cell", cell, "--seed", str(seed), "--trunk", trunk, "--task"]
    for task in ("e1", "e3", "pert", "tl_e1"):
        J.append(_gpu(f"{prefix}:{task}", base + [task], [f"{prefix}:accept"]))
        reads.append(f"{prefix}:{task}")
    J.append(_gpu(f"{prefix}:f1", base + ["f1"], [f"{prefix}:e1"]))
    reads.append(f"{prefix}:f1")
    return J, reads


def _write_and_add(name, jobs):
    out = OUT / "jobs" / f"{name}.json"
    require(not out.exists(), "job list written once: " + str(out))
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    import subprocess
    subprocess.run([PY, str(R / "tools/acu/chain_r2.py"), "add", str(out)], check=True)
    print(len(jobs), "jobs added from", out, flush=True)


def plan_a():
    from tools.acu.decide_s3 import winner
    w = winner()
    J, reads = _cell_block("s3a:nomp", w, 3407, "nomp", [])
    J.append(_cpu("s3a:decide", ["trunk"], reads, "tools.acu.decide_s3"))
    J.append(_cpu("s3:plan_b", ["plan_b"], ["s3a:decide"], "tools.acu.jobs_s3"))
    _write_and_add("s3a", J)


def plan_b():
    from tools.acu.decide_s3 import chosen
    w, t = chosen()
    J, reads = [], []
    for seed in (3408, 3409):
        j, r = _cell_block(f"s3b:{seed}", w, seed, t, [])
        J += j
        reads += r
    trainings = [f"s3b:{s}:train" for s in (3408, 3409)]
    last = reads[-1]
    cost = []
    for policy in ("default", "det"):
        for label, module, args in (("s43", "tools.acu.cost_s3", ["ref", "--net", "s43", "--policy", policy]),
                                    ("b2", "tools.acu.cost_s3", ["ref", "--net", "b2", "--policy", policy]),
                                    ("acu", "tools.acu.cost_s3", ["run", "--cell", w, "--seed", "3407", "--trunk", t,
                                                                  "--policy", policy])):
            key = f"s3b:cost:{label}:{policy}"
            J.append({"key": key, "kind": "exclusive", "deps": [cost[-1] if cost else last] + trainings, "mem_gib": 15,
                      "command": [PY, "-m", module, *args], "note": "cost block (GPU 4 exclusive)"})
            cost.append(key)
    tag = lambda s: str(OUT / "readings" / A.run_name(w, s, trunk=t))  # noqa: E731
    costdir = str(OUT / "costs" / f"s3__{A.run_name(w, 3407, trunk=t)}__default")
    J.append(_cpu("s3b:row", ["row", "--name", "final", "--label", f"ACU {w} ({'/'.join(A.CELLS[w])}, {t}, PR, causal; 3 seeds)",
                              "--read", tag(3407), "--read", tag(3408), "--read", tag(3409), "--cost", costdir],
                  reads + cost, "tools.acu.final"))
    J.append(_cpu("s3b:decide", ["final"], reads + cost, "tools.acu.decide_s3"))
    J.append(_cpu("s3b:asynccheck", [], [f"s3b:{s}:accept" for s in (3408, 3409)], "tools.acu.asynccheck", mem=24))
    mechs = ",".join([f"ACU {w} 3407={tag(3407)}", f"ACU {w} 3408={tag(3408)}", f"ACU {w} 3409={tag(3409)}",
                      f"B2 3407={OUT.parent.parent / 'pfo2/20261103/readings/pfo2_fgs_s3407'}",
                      f"S43 3407={OUT.parent.parent / 'pfo/20261103/readings/s43_s3407'}"]
                     + [f"2x2 {c}={OUT / 'readings' / A.run_name(c, 3407)}" for c in A.CELLS])
    J.append(_cpu("s3b:tables", ["tables", "--rows", "@evf_s43,@evf_b2,final", "--mech", mechs],
                  ["s3b:row", "s3b:decide", "s3b:asynccheck"], "tools.acu.final", mem=16))
    _write_and_add("s3b", J)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("plan_a", "plan_b"))
    a = p.parse_args(argv)
    plan_a() if a.verb == "plan_a" else plan_b()


if __name__ == "__main__":
    main()
