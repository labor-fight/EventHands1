#!/usr/bin/env python3
"""Move driver 5's pending D2 screen to mixed-pair driver 6 (PMR amendment 10).

Drain driver 5 first, then run ``python -m tools.pmr.jobs18 move --from driver5``.
Training commands, configurations, job keys and completed records are preserved.
The two pending rol5 jobs lose their direct-only pair restriction after the new
borrow_job2 wrapper raises the inherited file-descriptor soft limit.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.pmr.common import OUT, PY, now, read, require, save, sha  # noqa: E402
from tools.pmr.migrate import plan as pending_plan  # noqa: E402

FREEZE_KEY = "s9x:freeze:amend10"
ROL_KEYS = {"s9x:rpre:d1hubfull_rol5", "s9x:rtrain:d1hubfull_rol5:s3407"}


def plan(queue, status, source="driver5"):
    """Build a new queue without mutating the source; record scheduling changes."""
    require(status["status"] == "DRAINED", "the source driver must finish draining")
    require(not any(v["status"] == "RUNNING" for v in status["jobs"].values()), "source still has running jobs")
    waits, moved = pending_plan(queue, status, source)
    require(not waits, "drained source cannot need wait jobs")
    freeze = {"key": FREEZE_KEY, "kind": "cpu", "deps": [], "mem_gib": 4,
              "command": [PY, "-m", "tools.pmr.freeze", "verify", "--stage", "amend10"],
              "note": "mixed GPU pairs and inherited FD limit; #pmr-amend10-20261110"}
    jobs, changes = [freeze], []
    for original in moved:
        job = copy.deepcopy(original)
        require(job["key"] != FREEZE_KEY, "amendment already queued")
        if job["key"] in ROL_KEYS and "pairs" in job:
            require(job["pairs"] == [[5, 7]], "unexpected rol5 pair restriction")
            changes.append({"key": job["key"], "field": "pairs", "before": job.pop("pairs"), "after": None,
                            "reason": "mixed pairs authorized; borrow_job2 raises the inherited NOFILE soft limit"})
        job["deps"] = [FREEZE_KEY, *job["deps"]]
        jobs.append(job)
    require(len(jobs) > 1, "no pending jobs to migrate")
    return jobs, changes


def move(source):
    from tools.pmr import chain6 as C6
    src, target = OUT / source, C6.C.D2
    require(source == "driver5" and target.name == "driver6", "only driver5 -> driver6 is authorized")
    require((src / "DRAIN").exists(), "drain driver5 first")
    require(not target.exists(), "target must be fresh")
    queue, status = read(src / "queue.json"), read(src / "status.json")
    jobs, changes = plan(queue, status, source)
    path = OUT / "jobs/driver6_migrated.json"
    require(not path.exists(), "migration written once")
    save(path, jobs, exclusive=True)
    C6.C.add(jobs)
    save(target / "policy.json", {"train_pairs": [list(p) for p in C6.C.PAIRS],
                                  "note": "amend10: any pair in pool 0,1,2,5,7; any borrowed member wraps the whole pair"},
         exclusive=True)
    save(target / "migration.json", {"from": str(src), "written_utc": now(), "moved": len(jobs) - 1,
                                     "completed": [k for k, v in status["jobs"].items() if v["status"] == "EXIT_0"],
                                     "source_queue_sha256": sha(src / "queue.json"),
                                     "source_status_sha256": sha(src / "status.json"), "jobs_file": str(path),
                                     "scheduling_changes": changes, "added_gate": FREEZE_KEY}, exclusive=True)
    print(json.dumps({"target": str(target), "moved": len(jobs) - 1, "changes": changes}, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verb", choices=("move",))
    parser.add_argument("--from", dest="source", required=True, choices=("driver5",))
    args = parser.parse_args(argv)
    move(args.source)


if __name__ == "__main__":
    main()
