#!/usr/bin/env python3
"""Move the pending jobs of a draining PMR driver into driver 4 (docs/EXPERIMENT_RECORDS.md#pmr-amend7-20261109):
tools/pmr/migrate.py's rule with driver 4 as the target. The source carries its DRAIN file and its running jobs run to their
end there; every PENDING job is appended unchanged, a dependency on a job that already ended EXIT_0 is dropped, one on a
job still running there becomes the CPU job `d4:wait:<key>` (tools/pmr/migrate.py `wait`: exits 0 only if that job ends
EXIT_0). Once per target.

    PMR_DRIVER_DIR=driver4 python -m tools.pmr.migrate3 --from driver3
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import migrate as M  # noqa: E402
from tools.pmr.common import OUT, now, read, require, save  # noqa: E402

PREFIX = "d4:wait:"


def plan(queue, status, source):
    """migrate.py's plan with driver 4's wait-job prefix."""
    native = M.wait_key
    M.wait_key = lambda key: PREFIX + key
    try:
        waits, moved = M.plan(queue, status, source)
    finally:
        M.wait_key = native
    for w in waits:
        w["note"] = w["note"].replace("#pmr-amend4-20261109", "#pmr-amend7-20261109")
    return waits, moved


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="src", required=True)
    a = p.parse_args(argv)
    require(os.environ.get("PMR_DRIVER_DIR") not in (None, "", a.src), "set PMR_DRIVER_DIR to the target")
    from tools.pmr import chain4 as C4
    C = C4.C
    src = OUT / a.src
    require(C.D2 != src and not C.QUEUE.exists(), "the target state directory must be fresh")
    require((src / "DRAIN").exists(), "drain the source driver first")
    waits, moved = plan(read(src / "queue.json"), read(src / "status.json"), a.src)
    out = OUT / "jobs" / f"{C.D2.name}_migrated.json"
    require(not out.exists(), "migrated once")
    out.write_text(json.dumps(waits + moved, indent=1) + "\n")
    C.add(waits + moved)
    save(C.D2 / "migration.json", {"from": str(src), "waits": [w["key"] for w in waits], "moved": len(moved),
                                   "jobs_file": str(out), "written_utc": now()}, exclusive=True)
    print(len(waits), "waits;", len(moved), "moved")


if __name__ == "__main__":
    main()
