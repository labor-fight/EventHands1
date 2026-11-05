#!/usr/bin/env python3
"""ACU stage-2 job list (docs/EXPERIMENT_RECORDS.md#acu-s2-prereg-20261104) for tools/acu/chain_r2.py: freeze check; the gate
(s1c PR_OK, exit 3 closes stage 2 by rule); per 2x2 cell preflight + acceptance, formal training + acceptance (b1 cells:
sample pairing; ca cells: data stream equal to their pfa partner), readings e1 / e3 / pert / tl_e1 / f1 with PR routes; the
written 2x2 decision.

    python -m tools.acu.jobs_s2 --out outputs/acu/20261104/jobs/s2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.acu.arms_s2 import CELLS, write_config  # noqa: E402
from tools.acu.common import PY, ROOT as R, require  # noqa: E402

PAIRS = [[4, 5], [6, 7]]
SEED = 3407
PARTNER = {"c10": "c00", "c11": "c01"}


def jobs():
    def train(key, cell, phase, deps):
        return {"key": key, "kind": "pair", "deps": deps, "mem_gib": 200, "pairs": PAIRS,
                "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY, str(R / "tools/acu/train_s2.py"),
                            "--config", str(write_config(cell, SEED, phase))], "note": f"s2 {cell} {phase}"}

    def cpu(key, args, deps, module="tools.acu.run_s2", mem=8):
        return {"key": key, "kind": "cpu", "deps": deps, "mem_gib": mem, "command": [PY, "-m", module, *args]}

    def gpu(key, cell, task, deps):
        return {"key": key, "kind": "gpu", "deps": deps,
                "command": [PY, "-m", "tools.acu.run_s2", "eval", "--cell", cell, "--task", task], "note": f"s2 {cell}"}
    J = [cpu("s2:freeze", ["verify", "--stage", "s2"], [], module="tools.acu.freeze", mem=2),
         cpu("s2:gate", ["gate"], ["s2:freeze", "s1c:decide"], module="tools.acu.decide_s2", mem=2)]
    reads = []
    for cell in ("c00", "c10", "c01", "c11"):
        J.append(train(f"s2:{cell}:pre", cell, "preflight", ["s2:gate"]))
        J.append(cpu(f"s2:{cell}:pre:accept", ["accept", "--cell", cell, "--phase", "preflight"], [f"s2:{cell}:pre"]
                     + ([f"s2:{PARTNER[cell]}:pre"] if cell in PARTNER else []), mem=24))
    for cell in ("c00", "c10", "c01", "c11"):
        J.append(train(f"s2:{cell}:train", cell, "formal", [f"s2:{cell}:pre:accept"]))
    for cell in ("c00", "c10", "c01", "c11"):
        J.append(cpu(f"s2:{cell}:accept", ["accept", "--cell", cell, "--phase", "formal"], [f"s2:{cell}:train"]
                     + ([f"s2:{PARTNER[cell]}:train"] if cell in PARTNER else []), mem=24))
        for task in ("e1", "e3", "pert", "tl_e1"):
            J.append(gpu(f"s2:{cell}:{task}", cell, task, [f"s2:{cell}:accept"]))
            reads.append(f"s2:{cell}:{task}")
        J.append(gpu(f"s2:{cell}:f1", cell, "f1", [f"s2:{cell}:e1"]))
        reads.append(f"s2:{cell}:f1")
    J.append(cpu("s2:decide", ["s2"], reads, module="tools.acu.decide_s2"))
    return J


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "job list written once: " + str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    J = jobs()
    out.write_text(json.dumps(J, indent=1) + "\n")
    print(len(J), "jobs ->", out)


if __name__ == "__main__":
    main()
