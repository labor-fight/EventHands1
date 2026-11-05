#!/usr/bin/env python3
"""ACU stage-1b job list (docs/EXPERIMENT_RECORDS.md#acu-s1b-prereg-20261104) for tools/acu/chain_r2.py: freeze check, b1
preflight and acceptance, formal training and acceptance, readings e1 / e3 / pert / f1 / tl_e1 (deployed r_s2d router),
the s1b decision.

    python -m tools.acu.jobs_b1 --out outputs/acu/20261104/jobs/s1b.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.acu.arms_b1 import write_config  # noqa: E402
from tools.acu.common import PY, ROOT as R, SCREEN_SEED, require  # noqa: E402

PAIRS = [[4, 5], [6, 7]]


def jobs():
    def train(key, phase, deps):
        return {"key": key, "kind": "pair", "deps": deps, "mem_gib": 130, "pairs": PAIRS,
                "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY, str(R / "tools/acu/train_b1.py"),
                            "--config", str(write_config("b1", SCREEN_SEED, phase))], "note": f"b1 {phase}"}

    def cpu(key, args, deps, module="tools.acu.run_b1", note=""):
        return {"key": key, "kind": "cpu", "deps": deps, "command": [PY, "-m", module, *args], "note": note}

    def gpu(key, task, deps):
        return {"key": key, "kind": "gpu", "deps": deps, "command": [PY, "-m", "tools.acu.run_b1", "eval", "--task", task],
                "note": "b1"}
    J = [cpu("s1b:freeze", ["verify", "--stage", "s1b"], [], module="tools.acu.freeze", note="manifest unchanged"),
         train("b1:pre", "preflight", ["s1b:freeze"]),
         cpu("b1:pre:accept", ["accept", "--phase", "preflight"], ["b1:pre"]),
         train("b1:train", "formal", ["b1:pre:accept"]),
         cpu("b1:accept", ["accept", "--phase", "formal"], ["b1:train"])]
    reads = []
    for task in ("e1", "e3", "pert", "tl_e1"):
        J.append(gpu(f"b1:{task}", task, ["b1:accept"]))
        reads.append(f"b1:{task}")
    J.append(gpu("b1:f1", "f1", ["b1:e1"]))
    reads.append("b1:f1")
    J.append(cpu("s1b:decide", ["s1b"], reads, module="tools.acu.decide_b1", note="stage-1b decision (written rules)"))
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
