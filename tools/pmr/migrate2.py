#!/usr/bin/env python3
"""Move the unfinished jobs of a stopped PMR driver into a fresh one (docs/EXPERIMENT_RECORDS.md#pmr-amend5-20261109):
jobs still PENDING and jobs left RUNNING by the stop are re-queued unchanged (dependencies on finished jobs dropped).
The output directory of every training left RUNNING (TRAIN.OUTPUT_DIR of its --config), if present, is moved first to
outputs/pmr/20261109/formal_failed/<source>_stop/<run> (preserved), so the fresh re-run starts clean. Once per target.

    PMR_DRIVER_DIR=driver3 python -m tools.pmr.migrate2 --from driver2
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr.borrow_job import output_dir  # noqa: E402
from tools.pmr.common import OUT, now, read, require, save  # noqa: E402
from tools.pmr.migrate import FIELDS  # noqa: E402


def plan(queue, status):
    """(jobs to re-queue, keys left RUNNING by the stop)."""
    st = {k: v["status"] for k, v in status["jobs"].items()}
    redo = [j["key"] for j in queue["jobs"] if st.get(j["key"]) == "RUNNING"]
    todo = [j for j in queue["jobs"] if st.get(j["key"], "PENDING") in ("PENDING", "RUNNING")]
    keys = {j["key"] for j in todo}
    moved = []
    for j in todo:
        require(all(d in keys or st.get(d) == "EXIT_0" for d in j["deps"]), "failed dependency of " + j["key"])
        moved.append({k: v for k, v in {**j, "deps": [d for d in j["deps"] if d in keys]}.items() if k in FIELDS})
    return moved, redo


def archive(job, dest_root):
    """Move a stopped training's output directory aside; returns the new path or None."""
    if "--config" not in job["command"]:
        return None
    out = output_dir(job["command"])
    if not out.exists():
        return None
    dest = dest_root / out.name
    require(not dest.exists(), "archive exists: " + str(dest))
    dest_root.mkdir(parents=True, exist_ok=True)
    shutil.move(str(out), str(dest))
    return str(dest)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="src", required=True)
    a = p.parse_args(argv)
    from tools.pmr import chain3 as C3
    C = C3.C
    src = OUT / a.src
    require(C.D2 != src and not C.QUEUE.exists(), "the target state directory must be fresh")
    q, s = read(src / "queue.json"), read(src / "status.json")
    require(s["status"] in ("STOPPED_PRESERVED", "DRAINED"), "the source driver has not stopped")
    moved, redo = plan(q, s)
    by_key = {j["key"]: j for j in q["jobs"]}
    archived = {k: archive(by_key[k], OUT / "formal_failed" / f"{a.src}_stop") for k in redo}
    out = OUT / "jobs" / f"{C.D2.name}_migrated.json"
    require(not out.exists(), "migrated once")
    out.write_text(json.dumps(moved, indent=1) + "\n")
    C.add(moved)
    save(C.D2 / "migration.json", {"from": str(src), "moved": len(moved), "rerun_after_stop": redo,
                                   "archived": archived, "jobs_file": str(out), "written_utc": now()}, exclusive=True)
    print(len(moved), "moved; re-run:", redo, "archived:", archived)


if __name__ == "__main__":
    main()
