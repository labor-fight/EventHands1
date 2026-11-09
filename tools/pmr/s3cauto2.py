#!/usr/bin/env python3
"""PMR stage S3c registration, second attempt (docs/EXPERIMENT_RECORDS.md#pmr-amend6-20261109): `s3c:auto` wrote
outputs/pmr/20261109/jobs/s3c.json and then failed to append it, because the cost block depends on every S3c reading of D1
and two of them (the plain and causal asynchronous checks) had finished in driver 2 before its stop, so driver 3's queue
does not hold their keys. This appends the same job list with every dependency that is absent from the target queue
and finished EXIT_0 in an earlier driver dropped (it is satisfied); any other unknown dependency stops it.
decisions/s3c_auto2.json records the outcome (once).

    PMR_DRIVER_DIR=driver3 python -m tools.pmr.s3cauto2
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import chain3 as C3  # noqa: E402
from tools.pmr.common import OUT, now, read, require, save  # noqa: E402

EARLIER = ("driver1", "driver2")


def plan(jobs, queued, finished):
    """jobs with dependencies outside `queued` dropped when they are in `finished`; returns (jobs, dropped)."""
    own = {j["key"] for j in jobs}
    out, dropped = [], set()
    for j in jobs:
        deps = []
        for d in j["deps"]:
            if d in queued or d in own:
                deps.append(d)
            else:
                require(d in finished, f"{j['key']} depends on {d}, neither queued nor finished")
                dropped.add(d)
        out.append({**j, "deps": deps})
    return out, sorted(dropped)


def main():
    C = C3.C
    rec = OUT / "decisions" / "s3c_auto2.json"
    require(not rec.exists(), "registered once")
    require(not (OUT / "decisions" / "s3c_auto.json").exists(), "s3c:auto registered already")
    jobs = read(OUT / "jobs" / "s3c.json")
    queued = {j["key"] for j in C.load_queue()["jobs"]}
    require(not ({j["key"] for j in jobs} & queued), "S3c final-form jobs queued already")
    finished = {k for d in EARLIER for k, v in read(OUT / d / "status.json")["jobs"].items() if v["status"] == "EXIT_0"}
    jobs, dropped = plan(jobs, queued, finished)
    C.add(jobs)
    save(rec, {"jobs": len(jobs), "dropped_finished_deps": dropped, "driver": str(C.D2), "written_utc": now()},
         exclusive=True)
    print(len(jobs), "jobs appended; dropped finished deps:", dropped)


if __name__ == "__main__":
    main()
