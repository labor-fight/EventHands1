#!/usr/bin/env python3
"""Move the pending jobs of a draining PMR driver into driver 2 (docs/EXPERIMENT_RECORDS.md#pmr-amend4-20261109).

  move  (once per target) the source driver must carry its DRAIN file: its running jobs run to their end there. Every job
        still PENDING in it is appended unchanged to the target queue; a dependency on a job that already ended EXIT_0 is
        dropped, one on a job still running there becomes the CPU job `d2:wait:<key>`, which waits for that job and exits
        0 only if it ends EXIT_0 (so a failure there skips its dependants here, as it would have there).
  wait  the body of a `d2:wait:<key>` job.

    PMR_DRIVER_DIR=driver2 python -m tools.pmr.migrate move --from driver1
    python -m tools.pmr.migrate wait --driver driver1 --key <key>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from tools.pmr.common import OUT, PY, now, read, require, save  # noqa: E402

FIELDS = ("key", "kind", "command", "deps", "note", "mem_gib", "pairs")
TERMINAL = ("EXIT_0", "FAILED_PRESERVED", "SKIPPED_DEP_FAILED")
POLL_S = 30


def wait_key(key):
    return "d2:wait:" + key


def plan(queue, status, source):
    """(wait jobs, moved jobs) for the source driver's queue and status."""
    st = {k: v["status"] for k, v in status["jobs"].items()}
    todo = [j for j in queue["jobs"] if st.get(j["key"], "PENDING") == "PENDING"]
    keys = {j["key"] for j in todo}
    running = [j["key"] for j in queue["jobs"] if st.get(j["key"]) == "RUNNING"]
    waits = [{"key": wait_key(k), "kind": "cpu", "deps": [], "mem_gib": 1,
              "command": [PY, "-m", "tools.pmr.migrate", "wait", "--driver", source, "--key", k],
              "note": f"waits for {k} in {source} (#pmr-amend4-20261109)"} for k in running]
    moved = []
    for j in todo:
        deps = []
        for d in j["deps"]:
            if d in keys:
                deps.append(d)
            elif st.get(d) == "RUNNING":
                deps.append(wait_key(d))
            else:
                require(st.get(d) == "EXIT_0", f"{j['key']} depends on {d} ({st.get(d)})")
        moved.append({k: v for k, v in {**j, "deps": deps}.items() if k in FIELDS})
    return waits, moved


def move(source):
    from tools.pmr import chain2 as C2
    src = OUT / source
    require(C2.C.D2 != src and not C2.C.QUEUE.exists(), "the target state directory must be fresh")
    require((src / "DRAIN").exists(), "drain the source driver first")
    waits, moved = plan(read(src / "queue.json"), read(src / "status.json"), source)
    out = OUT / "jobs" / f"{C2.C.D2.name}_migrated.json"
    require(not out.exists(), "migrated once")
    out.write_text(json.dumps(waits + moved, indent=1) + "\n")
    C2.C.add(waits + moved)
    save(C2.C.D2 / "migration.json", {"from": str(src), "waits": [w["key"] for w in waits], "moved": len(moved),
                                      "jobs_file": str(out), "written_utc": now()}, exclusive=True)
    print(len(waits), "waits;", len(moved), "moved")


def wait(driver, key):
    path = OUT / driver / "status.json"
    while True:
        s = read(path)
        v = s["jobs"].get(key, {}).get("status")
        if v in TERMINAL:
            print(key, v, now(), flush=True)
            return 0 if v == "EXIT_0" else 1
        if s["status"] != "RUNNING":
            print(key, v, "but", driver, s["status"], now(), flush=True)
            return 1
        time.sleep(POLL_S)


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="verb", required=True)
    m = sub.add_parser("move")
    m.add_argument("--from", dest="src", required=True)
    w = sub.add_parser("wait")
    w.add_argument("--driver", required=True)
    w.add_argument("--key", required=True)
    a = p.parse_args(argv)
    if a.verb == "move":
        require(os.environ.get("PMR_DRIVER_DIR") not in (None, "", a.src), "set PMR_DRIVER_DIR to the target")
        move(a.src)
        return 0
    return wait(a.driver, a.key)


if __name__ == "__main__":
    sys.exit(main())
