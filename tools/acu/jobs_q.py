#!/usr/bin/env python3
"""ACU PR revision job list (docs/EXPERIMENT_RECORDS.md#acu-pr2-prereg-20261104) for tools/acu/chain_r2.py: the registered
s1c / s2 / s3-plan job lists (tools/acu/jobs_pr.py, tools/acu/jobs_s2.py, jobs/s3_plan.json; frozen, unchanged) rebuilt
under the namespace "q" (tools/acu/ns_q.py: PR2 in place of PR, configs/acu_20261104q, outputs/acu/20261104q).

    python -m tools.acu.jobs_q write      outputs/acu/20261104/jobs/q.json (once), then `chain_r2.py add` it

Every job keeps its kind, GPUs, memory and arguments; keys get the prefix "q:" (dependencies inside the set renamed the
same way); the frozen stages' own freeze checks are replaced by `q:freeze` (manifests/pr2.json); entry points run under the
namespace (`rewrite`: router_pr.py -> router_pr2.py, train_s2.py -> train_s2q.py, modules -> qrun, the stage-3 planner
-> jobs_s3q). The PR2 router preflight (and so every other job of the set) also waits for `q:gate0`, which closes the
set by rule (exit 3) unless the main s1c decision (`r2:s1c:decide`) is PR_WEAK.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.acu import ns_q  # noqa: E402

ns_q.enter()

from tools.acu import jobs_pr, jobs_s2  # noqa: E402
from tools.acu.common import PY, require  # noqa: E402

PREFIX = "q:"
MAIN_S1C = "r2:s1c:decide"
FREEZE = {"key": PREFIX + "freeze", "kind": "cpu", "deps": [], "mem_gib": 4,
          "command": [PY, "-m", "tools.acu.freeze", "verify", "--stage", "pr2"], "note": "PR revision PR2: freeze check"}
GATE0 = {"key": PREFIX + "gate0", "kind": "cpu", "deps": [FREEZE["key"], MAIN_S1C], "mem_gib": 2,
         "command": [PY, "-m", "tools.acu.pr2", "gate0"], "note": "PR2 only after the main s1c says PR_WEAK"}
DROPPED = ("s1c:freeze", "s2:freeze", "s3:freeze")
SCRIPTS = {"tools/acu/router_pr.py": "tools/acu/router_pr2.py", "tools/acu/train_s2.py": "tools/acu/train_s2q.py"}
PLANNERS = {"tools.acu.jobs_s3": "tools.acu.jobs_s3q"}
OUTSIDE = ("tools.acu.freeze", "tools.acu.pr2", "tools.acu.qrun", "tools.acu.jobs_s3q")


def rewrite(command):
    """The command under the namespace "q"."""
    out = []
    for i, c in enumerate(command):
        for old, new in SCRIPTS.items():
            if c == str(ROOT / old):
                c = str(ROOT / new)
        if i and command[i - 1] == "-m" and c.startswith("tools.acu."):
            if c in PLANNERS:
                c = PLANNERS[c]
            elif c not in OUTSIDE:
                out += ["tools.acu.qrun", c]
                continue
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
    """jobs_pr + jobs_s2 (configs written under the namespace) + the stage-3 plan, renamed; freeze checks replaced."""
    src = jobs_pr.jobs() + jobs_s2.jobs() + json.loads((ns_q.base_out() / "jobs" / "s3_plan.json").read_text())
    for j in src:
        j.pop("added_at_utc", None)
    keys = {j["key"] for j in src}
    require(all(d in keys or d in DROPPED for j in src for d in j.get("deps", [])), "a dependency outside the lists")
    kept = [{**j, "deps": [d for d in j.get("deps", []) if d not in DROPPED]} for j in src if j["key"] not in DROPPED]
    jobs = rename(kept, extra_deps=(FREEZE["key"],))
    for j in jobs:
        if j["key"] == PREFIX + "pr:pre":
            j["deps"].append(GATE0["key"])
    return [FREEZE, GATE0] + jobs


def write():
    out = ns_q.base_out() / "jobs" / "q.json"
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
