#!/usr/bin/env python3
"""Least-loaded placement for the shared packer table (docs/EXPERIMENT_RECORDS.md#real-amend5-run-20261111).

The packers (tools/real/pack.py, frozen) start a ready GPU task on whichever card polls first with room under
policy.json's max_jobs, so a new task can land on a card already running three while another card is idle. Measured on
these cards (2026-11-11): an EvHandTrack base training alone runs ~4.5 k steps / h, two on a card ~2.7 k each, three
~1.5 k each. This daemon keeps policy.json's max_jobs at

    min(MAX_JOBS, lowest number of running GPU tasks among the live cards that could take the highest-priority
        ready GPU task + 1)

(all live cards when no GPU task is ready), so the next task goes to a least-loaded card. "Could take" mirrors the
packer's own rules (per-card memory-estimate cap, per-arm cap of trainings, free GPU memory with the margin). A GPU
process outside the table (e.g. a router run by hand on two cards) counts as one more task on its card (since 22:15 UTC;
the packers compare only their own tasks with max_jobs, so this only moves the threshold). It never
starts, stops or edits a task; on exit (SIGTERM / SIGINT) it puts max_jobs back to MAX_JOBS. Every change is logged on
stdout; policy.json's edits list records only the start and the stop.

Stop it (kill its process group) before tools/real/move.py, which edits max_jobs itself.

    setsid nohup python tools/real/steer.py > outputs/real/20261110/pack/steer.log 2>&1 &
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from tools.real import pack as P  # noqa: E402

MAX_JOBS = 4
PERIOD_S = 5
LIVE_S = 120


def say(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def gpu_free():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    free = {}
    for line in out.strip().splitlines():
        i, u, f = [x.strip() for x in line.split(",")]
        free[i] = free[u] = float(f) / 1024
    return free


def external_load(st):
    """Per card, the GPU processes that belong to no running task of the table (e.g. the two ranks of a router run by
    hand): each counts as one more task on its card."""
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True,
                          text=True, check=True).stdout
    index = {u.strip(): i.strip() for i, u in (r.split(",") for r in gpus.strip().splitlines())}
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True).stdout
    kids = {}
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                kids.setdefault(int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1]), []).append(int(p))
            except (OSError, IndexError, ValueError):
                pass
    own = set()
    for t in st["tasks"].values():
        if t["status"] == "RUNNING" and t.get("pid"):
            todo = [t["pid"]]
            while todo:
                q = todo.pop()
                own.add(q)
                todo += kids.get(q, [])
    ext = {}
    for r in apps.strip().splitlines():
        u, pid = [x.strip() for x in r.split(",")]
        if pid.isdigit() and int(pid) not in own:
            ext[index.get(u, u)] = ext.get(index.get(u, u), 0) + 1
    by_card = {}
    for c, info in st["cards"].items():
        dev = str(info.get("device", c))
        by_card[c] = ext.get(index.get(dev, dev), 0)
    return by_card


def set_max_jobs(value, why=None):
    pol = json.loads(P.POLICY.read_text())
    old = pol.get("max_jobs")
    if old == value and why is None:
        return old
    pol["max_jobs"] = value
    if why is not None:
        pol.setdefault("edits", []).append({"at": P.now(), "change": f"max_jobs {old} -> {value}", "why": why})
    tmp = P.POLICY.with_suffix(".steer.tmp")
    tmp.write_text(json.dumps(pol, indent=1) + "\n")
    os.replace(tmp, P.POLICY)
    return old


def target(st, pol, free, ext=None):
    T = st["tasks"]
    live = [c for c, info in st["cards"].items()
            if info.get("alive", True) and time.time() - info.get("beat", 0) < LIVE_S and "exited" not in info]
    if not live:
        return None, None
    mine = {c: [k for k, t in T.items() if t["status"] == "RUNNING" and t.get("card") == c and t["gpu"]] for c in live}
    load = {c: len(v) + (ext or {}).get(c, 0) for c, v in mine.items()}
    ready = [(k, t) for k, t in sorted(T.items(), key=lambda kv: kv[1]["prio"])
             if t["status"] == "PENDING" and t["gpu"] and P.deps_state(st, t) == "ready"]
    if not ready:
        return min(MAX_JOBS, min(load.values()) + 1), None
    key, t = ready[0]
    need = pol["est_gib"][t["est"]]
    ok = []
    for c in live:
        if load[c] >= MAX_JOBS:
            continue
        if sum(pol["est_gib"][T[k]["est"]] for k in mine[c]) + need > pol["mem_cap_gib"]:
            continue
        cap = pol.get("max_per_arm", {}).get(t["arm"]) if t["kind"] in ("smoke", "train") else None
        if cap is not None and sum(1 for k in mine[c] if T[k]["arm"] == t["arm"]
                                   and T[k]["kind"] in ("smoke", "train")) >= cap:
            continue
        dev = st["cards"][c].get("device", c)
        if free.get(str(dev), free.get(c, 0.0)) < need + pol["free_margin_gib"]:
            continue
        ok.append(c)
    pool = ok or live
    return min(MAX_JOBS, min(load[c] for c in pool) + 1), key


def main():
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    set_max_jobs(json.loads(P.POLICY.read_text()).get("max_jobs", MAX_JOBS),
                 "tools/real/steer.py started: max_jobs follows the least-loaded live card (+1, at most "
                 f"{MAX_JOBS}) so a new task goes to the emptiest card; changes logged in pack/steer.log")
    say("steer start")
    last = None
    while not stop:
        try:
            st = json.loads(P.STATE.read_text())
            pol = P.policy()
            value, key = target(st, pol, gpu_free(), external_load(st))
            if value is not None and value != pol.get("max_jobs"):
                old = set_max_jobs(value)
                say(f"max_jobs {old} -> {value}" + (f" (next ready GPU task {key})" if key else " (no ready GPU task)"))
            elif (value, key) != last and key:
                say(f"max_jobs {value} (next ready GPU task {key})")
            last = (value, key)
        except Exception as e:                          # a transient read error must not stop the placement
            say("error:", repr(e))
        for _ in range(PERIOD_S):
            if stop:
                break
            time.sleep(1)
    set_max_jobs(MAX_JOBS, f"tools/real/steer.py stopped: max_jobs back to {MAX_JOBS}")
    say("steer stop; max_jobs", MAX_JOBS)


if __name__ == "__main__":
    main()
