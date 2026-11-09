#!/usr/bin/env python3
"""Hold a training that was already running on GPUs 0-3 when the borrowing rule arrived to gpu-borrow's yield rule
(docs/EXPERIMENT_RECORDS.md#pmr-amend4-20261109). gpu-borrow cannot adopt a running process, so this watcher applies its
rule to the job's process group: as soon as a process of the priority user (lmx), or any compute context outside the
group, appears on one of the job's cards (or the cards' state stays unreadable for DEBOUNCE polls), the whole group is
killed (SIGKILL, gpu-borrow's default zero grace). The driver then records the job FAILED_PRESERVED and it is re-run fresh
through gpu-borrow. Exits once the group is gone. Run with gpu-borrow's interpreter (NVML):

    ~/.local/share/gpu-borrow/venv/bin/python tools/pmr/borrow_guard.py --key K --pgid P --gpus 2,3 --log <file.jsonl>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pwd
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.gpu_borrow.monitor import MonitorError, NvidiaMonitor  # noqa: E402
from tools.gpu_borrow.supervisor import select_gpus  # noqa: E402

POLL_S = 0.5
DEBOUNCE = 6


def live_group(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def owned(pid, pgid):
    try:
        return os.getpgid(pid) == pgid
    except ProcessLookupError:
        return False


def blockers(gpu, pgid, priority_uid):
    return [p for p in gpu.processes if not owned(p.pid, pgid) and (p.uid == priority_uid or p.kind != "G")]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--key", required=True)
    p.add_argument("--pgid", type=int, required=True)
    p.add_argument("--gpus", required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--priority-user", default="lmx")
    a = p.parse_args(argv)
    priority_uid = pwd.getpwnam(a.priority_user).pw_uid
    a.log.parent.mkdir(parents=True, exist_ok=True)
    log = a.log.open("a", buffering=1)

    def event(name, **fields):
        log.write(json.dumps({"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": name,
                              "key": a.key, "pgid": a.pgid, **fields}) + "\n")

    monitor = NvidiaMonitor()
    gpus = select_gpus(monitor.snapshot(), a.gpus)
    uuids = tuple(g.uuid for g in gpus)
    event("watching", gpus=[g.index for g in gpus], backend=monitor.backend, priority_uid=priority_uid)
    errors = 0
    try:
        while live_group(a.pgid):
            try:
                snap = monitor.snapshot(uuids)
                found = {g.index: [(q.pid, q.uid, q.kind) for q in blockers(g, a.pgid, priority_uid)]
                         for g in snap.values()}
                errors = 0
            except MonitorError as exc:
                errors += 1
                found = {}
                if errors < DEBOUNCE:
                    time.sleep(POLL_S)
                    continue
                event("monitor_error", error=str(exc))
            if errors >= DEBOUNCE or any(found.values()):
                if live_group(a.pgid):
                    os.killpg(a.pgid, signal.SIGKILL)
                event("yielded", blockers={str(k): v for k, v in found.items() if v})
                return 0
            time.sleep(POLL_S)
        event("job_ended")
        return 0
    finally:
        monitor.close()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
