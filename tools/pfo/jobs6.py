#!/usr/bin/env python3
"""PFO final cost block (docs/EXPERIMENT_RECORDS.md#pfo-p3-prereg-20261103): one exclusive job per network x
implementation x policy on GPU 4 alone (tools/pfo/chain.py), chained in a fixed order after the given jobs; the
default-flag pass first (the table's policy), then the deterministic pass. Networks:
  S43 s3407   sparse executor (the budget reference) and masked dense (the strict secondary reference)
  F           tools/pfo/run7.py cost (composable trunk + its deployed router)
  pfasrdz     fast implementation
  B, B - MP   tools/pfo/run2.py cost (the phase-2a arms b / nomp)
all at seed 3407 (a cost reading does not depend on the seed). The phase-1 S43 / pfasrdz readings are kept; their
re-measurements here are tagged `final__<net>__<impl>__<policy>`.

    python -m tools.pfo.jobs6 cost --after KEY [--after KEY ...] --out FILE
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo.common import PY, require  # noqa: E402

FINAL = "f1"
POLICIES = ("default", "det")


def commands(policy):
    tag = lambda net, impl: f"final__{net}__{impl}__{policy}"  # noqa: E731
    return [
        ("s43_s3407:exec", [PY, "-m", "tools.pfo.cost", "run", "--net", "s43_s3407", "--impl", "exec", "--policy", policy,
                            "--tag", tag("s43_s3407", "exec")]),
        (f"{FINAL}:3407", [PY, "-m", "tools.pfo.run7", "cost", "--arm", FINAL, "--seed", "3407", "--policy", policy]),
        ("s43_s3407:dense", [PY, "-m", "tools.pfo.cost", "run", "--net", "s43_s3407", "--impl", "dense", "--policy",
                             policy, "--tag", tag("s43_s3407", "dense")]),
        ("pfasrdz:fast", [PY, "-m", "tools.pfo.cost", "run", "--net", "pfasrdz", "--impl", "fast", "--policy", policy,
                          "--tag", tag("pfasrdz", "fast")]),
        ("b:3407", [PY, "-m", "tools.pfo.run2", "cost", "--arm", "b", "--seed", "3407", "--policy", policy]),
        ("nomp:3407", [PY, "-m", "tools.pfo.run2", "cost", "--arm", "nomp", "--seed", "3407", "--policy", policy]),
    ]


def cost(after):
    jobs, last = [], None
    for policy in POLICIES:
        for name, cmd in commands(policy):
            key = f"p3:cost:{name}:{policy}"
            jobs.append({"key": key, "kind": "exclusive", "command": cmd, "deps": [last] if last else list(after),
                         "mem_gib": 15, "note": "final cost block (#pfo-p3-prereg-20261103)"})
            last = key
    return jobs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("cost",))
    p.add_argument("--after", action="append", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "job file exists")
    jobs = cost(a.after)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
