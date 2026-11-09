#!/usr/bin/env python3
"""Move the pending jobs of the drained TPD driver (outputs/tpd/20261105/{queue,status}.json) into driver 2's queue
(docs/EXPERIMENT_RECORDS.md#tpd-amend1-20261105): same keys, kinds, commands, notes and memory declarations; a dependency
the old driver finished (EXIT_0) is dropped, one still pending is kept; a pending job with a failed or skipped dependency
is not moved (it would have been skipped). Once.

    python -m tools.tpd.migrate_driver
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from tools.tpd.common import OUT, now, read, require, save  # noqa: E402


def main():
    from tools.tpd import chain2 as C2
    old_q, old_s = read(OUT / "queue.json"), read(OUT / "status.json")
    require(old_s["status"] == "DRAINED", "the old driver must be drained first")
    require(not C2.QUEUE.exists(), "driver 2's queue exists already")
    st = {k: v["status"] for k, v in old_s["jobs"].items()}
    pending = [j for j in old_q["jobs"] if st.get(j["key"], "PENDING") == "PENDING"]
    keys = {j["key"] for j in pending}
    moved, dropped = [], []
    for j in pending:
        bad = [d for d in j["deps"] if st.get(d) in ("FAILED_PRESERVED", "SKIPPED_DEP_FAILED")]
        if bad:
            dropped.append({"key": j["key"], "failed_deps": bad})
            continue
        deps = [d for d in j["deps"] if d in keys]
        require(all(st.get(d) in ("EXIT_0", "PENDING", None) for d in j["deps"]), "unexpected dependency state")
        moved.append({k: v for k, v in {**j, "deps": deps}.items() if k in ("key", "kind", "command", "deps", "note",
                                                                             "mem_gib", "pairs")})
    out = OUT / "jobs" / "driver2_migrated.json"
    require(not out.exists(), "migrated once")
    out.write_text(json.dumps(moved, indent=1) + "\n")
    C2.add(moved)
    save(OUT / "driver2" / "migration.json", {"moved": len(moved), "dropped": dropped, "jobs_file": str(out),
                                              "old_status_jobs": len(st), "written_utc": now()}, exclusive=True)
    print(len(moved), "moved;", len(dropped), "dropped", dropped)


if __name__ == "__main__":
    main()
