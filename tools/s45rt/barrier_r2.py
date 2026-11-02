#!/usr/bin/env python3
"""S45-RT amendment-1 barrier / milestone (docs/EXPERIMENT_RECORDS.md#s45rt-amend1-20261031): exit 0 once every job
key listed in the given JSON file is terminal in the shared driver (tools/s45rt/chain_r2.py), whatever its outcome.

Used (i) as each study's barrier before its decision (only its own readings, so the other study's jobs in the shared
queue never delay it), (ii) before the main rows (every training of the queue, so latency is measured on a quiet
machine), and (iii) as milestones that order the two studies' trainings by priority.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.s45rt.common import OUT, read, require  # noqa: E402

TERMINAL = ("EXIT_0", "FAILED_PRESERVED", "SKIPPED_DEP_FAILED")


def main():
    require(len(sys.argv) == 2, "usage: barrier_r2.py <keys.json>")
    keys = json.loads(Path(sys.argv[1]).read_text())
    require(isinstance(keys, list) and keys and all(isinstance(k, str) for k in keys), "a non-empty list of job keys")
    while True:
        q, s = read(OUT / "queue_r2.json"), read(OUT / "status_r2.json")
        queued = {j["key"] for j in q["jobs"]}
        require(set(keys) <= queued, "barrier waits on unknown jobs: " + ", ".join(sorted(set(keys) - queued)))
        left = [k for k in keys if s["jobs"].get(k, {}).get("status") not in TERMINAL]
        if not left:
            print("terminal:", len(keys), "jobs", flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
