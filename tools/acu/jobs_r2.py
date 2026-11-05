#!/usr/bin/env python3
"""ACU stage-1a job list, revision 2 (docs/EXPERIMENT_RECORDS.md#acu-amend1-20261104): tools/acu/jobs.py's s1a with the
trainings on GPU pairs 4-5 (a1) and 6-7 (a2) and the freeze check against the amended manifest; nothing else changes.

    python -m tools.acu.jobs_r2 s1a --out outputs/acu/20261104/jobs/s1a_r2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.acu import jobs as J1  # noqa: E402
from tools.acu.common import require  # noqa: E402

PAIRS_R2 = {"a1": [[4, 5]], "a2": [[6, 7]]}


def s1a():
    jobs = J1.s1a(freeze="s1a_amend1")
    for j in jobs:
        if j["kind"] == "pair":
            arm = j["key"].split(":")[0]
            j["pairs"] = PAIRS_R2[arm]
    return jobs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("s1a",))
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    jobs = s1a()
    out = Path(a.out)
    require(not out.exists(), "job list written once: " + str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
