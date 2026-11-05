#!/usr/bin/env python3
"""PFO2 final cost block (docs/EXPERIMENT_RECORDS.md#pfo2-prereg-20261103 acceptance): one exclusive job per network x
implementation x policy on GPU 4 alone (tools/pfo2/chain.py), chained in a fixed order after the given jobs; the
PyTorch-default pass first (the table's policy), then the deterministic pass. Networks: S43 s3407 (sparse executor =
the reference, masked dense = the strict secondary reference), PFO's F, the base B2 and the given PFO2 candidates, all
at seed 3407 (a cost reading does not depend on the seed).

    python -m tools.pfo2.jobs_cost --name N --after KEY [--after KEY ...] --cand ARM [--cand ARM ...] --out FILE
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo2.common import PY, require  # noqa: E402

POLICIES = ("default", "det")


def commands(policy, base, cands):
    c = lambda *a: [PY, "-m", "tools.pfo2.cost", "run", *a, "--policy", policy]  # noqa: E731
    out = [("s43:exec", c("--net", "s43", "--impl", "exec"))]
    out += [(f"{a}:3407", c("--net", "pfo2", "--arm", a, "--seed", "3407")) for a in cands]
    out += [(f"{base}:3407", c("--net", "pfo2", "--arm", base, "--seed", "3407")),
            ("F:3407", c("--net", "pfo", "--arm", "f1", "--seed", "3407")),
            ("s43:dense", c("--net", "s43", "--impl", "dense"))]
    return out


def block(name, after, base, cands):
    jobs, last = [], None
    for policy in POLICIES:
        for label, cmd in commands(policy, base, cands):
            key = f"{name}:cost:{label}:{policy}"
            jobs.append({"key": key, "kind": "exclusive", "command": cmd, "deps": [last] if last else list(after),
                         "mem_gib": 15, "note": "PFO2 final cost block"})
            last = key
    return jobs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--after", action="append", required=True)
    p.add_argument("--base", default="fgs")
    p.add_argument("--cand", action="append", default=[])
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "job file exists")
    jobs = block(a.name, a.after, a.base, a.cand)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
