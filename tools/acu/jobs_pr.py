#!/usr/bin/env python3
"""ACU s1c job list (docs/EXPERIMENT_RECORDS.md#acu-s1c-prereg-20261104) for tools/acu/chain_r2.py: freeze check, PR router
preflight + check, formal training + check, prefix-equivalence on real packets (CPU), identity readings (zgz_local,
zgz_global, ylf_local), the written s1c decision.

    python -m tools.acu.jobs_pr --out outputs/acu/20261104/jobs/s1c.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.acu.common import PY, ROOT as R, require  # noqa: E402

PAIRS = [[4, 5], [6, 7]]


def jobs():
    def train(key, phase, deps):
        return {"key": key, "kind": "pair", "deps": deps, "mem_gib": 260, "pairs": PAIRS,
                "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY, str(R / "tools/acu/router_pr.py"),
                            "--phase", phase], "note": f"PR router {phase}"}

    def cpu(key, args, deps, module="tools.acu.run_pr", mem=8):
        return {"key": key, "kind": "cpu", "deps": deps, "mem_gib": mem, "command": [PY, "-m", module, *args]}

    def gpu(key, args, deps):
        return {"key": key, "kind": "gpu", "deps": deps, "command": [PY, "-m", "tools.acu.run_pr", *args]}
    J = [cpu("s1c:freeze", ["verify", "--stage", "s1c"], [], module="tools.acu.freeze", mem=2),
         train("pr:pre", "preflight", ["s1c:freeze"]),
         cpu("pr:pre:check", ["check", "--phase", "preflight"], ["pr:pre"]),
         train("pr:train", "formal", ["pr:pre:check"]),
         cpu("pr:check", ["check", "--phase", "formal"], ["pr:train"]),
         cpu("pr:equiv", ["equiv"], ["pr:check"], mem=16)]
    reads = []
    for seq in ("zgz_local", "zgz_global", "ylf_local"):
        J.append(gpu(f"pr:read:{seq}", ["read", "--seq", seq], ["pr:check"]))
        reads.append(f"pr:read:{seq}")
    J.append(cpu("s1c:decide", ["decide"], reads + ["pr:equiv"]))
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
