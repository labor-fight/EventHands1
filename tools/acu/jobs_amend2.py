#!/usr/bin/env python3
"""ACU engineering amendment 2 (docs/EXPERIMENT_RECORDS.md#acu-amend2-20261104): re-registration of the jobs skipped after
the PR preflight failure (`pr:pre` FAILED_PRESERVED at 18:55 UTC: CUDA has no deterministic floating-point cumsum).

    python -m tools.acu.jobs_amend2 write     jobs/amend2.json (once), then `chain_r2.py add` it

Every job of jobs/s1c.json, jobs/s2.json and jobs/s3_plan.json, in the same order, under the key prefix "r2:"
(dependencies inside the set renamed the same way), each also depending on the freeze check of this amendment; the
commands are unchanged except that every entry point that runs PR goes through the deterministic window sums
(`rewrite`), and the stage-3 planner is tools/acu/jobs_s3r2.py (the same planners, the planned jobs rewritten likewise).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.acu.common import OUT, PY, require  # noqa: E402

PREFIX = "r2:"
SOURCES = ("s1c", "s2", "s3_plan")
FREEZE = {"key": PREFIX + "amend2:freeze", "kind": "cpu", "deps": [], "mem_gib": 4,
          "command": [PY, "-m", "tools.acu.freeze", "verify", "--stage", "amend2"],
          "note": "engineering amendment 2: deterministic PR window sums"}
SCRIPTS = {"tools/acu/router_pr.py": "tools/acu/router_prd.py", "tools/acu/train_s2.py": "tools/acu/train_s2d.py"}
DET_MODULES = ("tools.acu.run_pr", "tools.acu.run_s2", "tools.acu.cost_s3", "tools.acu.asynccheck")
PLANNERS = {"tools.acu.jobs_s3": "tools.acu.jobs_s3r2"}


def rewrite(command):
    """The command with PR's deterministic sums installed (scripts -> wrappers, modules -> detrun, planner -> r2)."""
    out = []
    for i, c in enumerate(command):
        for old, new in SCRIPTS.items():
            if c == str(ROOT / old):
                c = str(ROOT / new)
        if i and command[i - 1] == "-m" and c in DET_MODULES:
            out += ["tools.acu.detrun", c]
            continue
        if i and command[i - 1] == "-m" and c in PLANNERS:
            c = PLANNERS[c]
        out.append(c)
    return out


def rename(jobs, extra_deps=()):
    keys = {j["key"] for j in jobs}
    require(not any(k.startswith(PREFIX) for k in keys), "already renamed")
    new = []
    for j in jobs:
        deps = [PREFIX + d if d in keys else d for d in j.get("deps", [])]
        new.append({**j, "key": PREFIX + j["key"], "deps": deps + [d for d in extra_deps if d not in deps],
                    "command": rewrite(j["command"])})
    return new


def build():
    jobs = []
    for s in SOURCES:
        jobs += json.loads((OUT / "jobs" / f"{s}.json").read_text())
    for j in jobs:
        j.pop("added_at_utc", None)
    return [FREEZE] + rename(jobs, extra_deps=(FREEZE["key"],))


def write():
    out = OUT / "jobs" / "amend2.json"
    require(not out.exists(), "job list written once: " + str(out))
    jobs = build()
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("write",))
    p.parse_args(argv)
    write()


if __name__ == "__main__":
    main()
