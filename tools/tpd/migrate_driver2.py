#!/usr/bin/env python3
"""Move the unfinished jobs of a stopped TPD driver-2 state directory into a fresh one (docs/EXPERIMENT_RECORDS.md
#tpd-amend2-20261105): pending jobs and jobs left RUNNING by the stop (their processes were killed; their run directories
are archived first) are re-queued unchanged; finished (EXIT_0) dependencies are dropped. Once per target.

    TPD_DRIVER_DIR=<new> python -m tools.tpd.migrate_driver2 --from driver2
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from tools.tpd.common import OUT, now, read, require, save  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="src", required=True)
    a = p.parse_args()
    from tools.tpd import chain2 as C2
    src = OUT / a.src
    require(C2.D2 != src and not C2.QUEUE.exists(), "the target state directory must be fresh")
    q, s = read(src / "queue.json"), read(src / "status.json")
    require(s["status"] in ("STOPPED_PRESERVED", "DRAINED"), "the source driver has not stopped")
    st = {k: v["status"] for k, v in s["jobs"].items()}
    redo = [k for k, v in st.items() if v == "RUNNING"]
    todo = [j for j in q["jobs"] if st.get(j["key"], "PENDING") in ("PENDING", "RUNNING")]
    keys = {j["key"] for j in todo}
    moved = []
    for j in todo:
        require(all(st.get(d, "PENDING") in ("EXIT_0", "PENDING", "RUNNING") for d in j["deps"]), "failed dependency")
        moved.append({k: v for k, v in {**j, "deps": [d for d in j["deps"] if d in keys]}.items()
                      if k in ("key", "kind", "command", "deps", "note", "mem_gib", "pairs")})
    out = OUT / "jobs" / f"{os.environ.get('TPD_DRIVER_DIR')}_migrated.json"
    require(not out.exists(), "migrated once")
    out.write_text(json.dumps(moved, indent=1) + "\n")
    C2.add(moved)
    save(C2.D2 / "migration.json", {"from": str(src), "moved": len(moved), "rerun_after_kill": redo,
                                    "jobs_file": str(out), "written_utc": now()}, exclusive=True)
    print(len(moved), "moved; re-run:", redo)


if __name__ == "__main__":
    main()
