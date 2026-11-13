#!/usr/bin/env python3
"""Card packer of the real study (docs/EXPERIMENT_RECORDS.md#real-prereg-20261110).

One packer runs inside each `gpu-borrow run --gpus <k>` job (one borrowed card, one borrow job: the trainings it starts
are processes of that job, so they never count as "another job" on the card). The packers of the cards share one task
table (outputs/real/20261110/pack/state.json, flock) and each starts the ready tasks on its own card while the card has
room; batch sizes and every recipe stay as registered (the official recipes fix them).

  tasks    smoke (seed 3407, 20 steps, every arm, chained epxf -> epx and epf -> ehpe; then a 30-endpoint reading of
           eho / epx / ehpe) -> gate -> formal trainings of the 5 arms x 3 seeds (epx after epxf's accept, ehpe after
           epf's accept), accept (CPU), E1 readings (eho, epx, ehpe), table (CPU)
  order    ready tasks by priority: readings, epf, ehpe, epxf, epx, eho (the EventHPE chain is the critical path)
  room     policy.json (re-read every loop): at most max_jobs tasks per card, at most max_per_arm[arm] trainings of
           one arm per card (EventHPE, the long pole: one per card), the summed memory estimates of its running tasks +
           the new one <= mem_cap_gib, the card's free memory >= estimate + free_margin_gib, and min_start_gap_s between
           two starts on a card
  failure  a failed task is kept (log, rc) and not retried; its dependants are skipped; a failed smoke closes the gate
  preempt  gpu-borrow kills the whole job; on restart the packer queues its card's unfinished tasks again; a card whose
           heartbeat is older than 15 min loses its unfinished tasks to the other cards the same way. Amendment 2
           (#real-amend2-20261110, user: resume allowed): a formal training keeps its run directory and
           tools/real/train.py resumes it from the newest loadable checkpoint; smokes, readings and accepts are archived
           to pack/preempted/ and rerun fresh
  stop     pack/STOP: no new starts; the packer exits once its running tasks end

    python tools/real/pack.py init          # write the task table (once)
    python tools/real/pack.py run --card K  # inside gpu-borrow run --gpus K
    python tools/real/pack.py status
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from tools.real import ns  # noqa: E402

PY = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"
PACK = ns.OUT / "pack"
STATE = PACK / "state.json"
LOCK = PACK / "lock"
POLICY = PACK / "policy.json"
LOGS = ns.OUT / "logs"
DEFAULT_POLICY = {"max_jobs": 4, "mem_cap_gib": 40.0, "free_margin_gib": 3.0, "min_start_gap_s": 40,
                  "max_per_arm": {"ehpe": 1},
                  "est_gib": {"eho": 5.0, "epxf": 5.0, "epx": 7.0, "epf": 11.0, "ehpe": 15.0, "e1": 4.0,
                              "e1smoke": 4.0}}
PRIORITY = {"e1": 0, "epf": 1, "ehpe": 2, "epxf": 3, "epx": 4, "eho": 5}
READ_ARMS = ("eho", "epx", "ehpe")
DEP = {"epx": "epxf", "ehpe": "epf"}
STALE_S = 900


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def cfg(arm, seed, phase):
    return str(ns.CONFIGS / (f"cmp_{arm}_s{seed}" + ("" if phase == "formal" else f"_{phase}") + ".yaml"))


def run_dir(arm, seed, phase):
    return ns.OUT / "runs" / (f"cmp_{arm}_s{seed}" + ("" if phase == "formal" else f"_{phase}"))


def tasks():
    t = {}

    def add(key, kind, cmd, deps=(), arm=None, gpu=True, outputs=(), prio=9, est="e1"):
        t[key] = {"kind": kind, "cmd": cmd, "deps": list(deps), "arm": arm, "gpu": gpu, "outputs": [str(o) for o in outputs],
                  "prio": prio, "est": est, "status": "PENDING"}
    s0 = 3407
    for arm in ns.ARMS:
        deps = [f"smoke:{DEP[arm]}"] if arm in DEP else []
        add(f"smoke:{arm}", "smoke", [PY, "tools/real/train.py", "--config", cfg(arm, s0, "smoke")], deps, arm,
            outputs=[run_dir(arm, s0, "smoke")], prio=PRIORITY[arm], est=arm)
    for arm in READ_ARMS:
        add(f"rsmoke:{arm}", "rsmoke", [PY, "tools/real/run.py", "e1", "--arm", arm, "--seed", str(s0), "--phase",
                                        "smoke", "--limit", "30"], [f"smoke:{arm}"], arm,
            outputs=[ns.OUT / "smoke" / "readings" / f"cmp_{arm}_s{s0}_smoke"], prio=0, est="e1smoke")
    add("gate", "gate", None, [k for k in t], gpu=False)
    for seed in ns.SEEDS:
        for arm in ns.ARMS:
            deps = ["gate"] + ([f"accept:{DEP[arm]}:{seed}"] if arm in DEP else [])
            add(f"train:{arm}:{seed}", "train", [PY, "tools/real/train.py", "--config", cfg(arm, seed, "formal")], deps,
                arm, outputs=[run_dir(arm, seed, "formal")], prio=PRIORITY[arm], est=arm)
            add(f"accept:{arm}:{seed}", "accept", [PY, "tools/real/run.py", "accept", "--arm", arm, "--seed", str(seed)],
                [f"train:{arm}:{seed}"], arm, gpu=False, outputs=[ns.OUT / "receipts" / f"cmp_{arm}_s{seed}.json"])
        for arm in READ_ARMS:
            add(f"e1:{arm}:{seed}", "e1", [PY, "tools/real/run.py", "e1", "--arm", arm, "--seed", str(seed)],
                [f"accept:{arm}:{seed}"], arm, outputs=[ns.OUT / "readings" / f"cmp_{arm}_s{seed}"], prio=0, est="e1")
    add("table", "table", [PY, "tools/real/table.py"], [k for k in t if k.startswith("e1:")], gpu=False,
        outputs=[ns.OUT / "table.md"])
    return t


@contextmanager
def locked():
    PACK.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            st = json.loads(STATE.read_text())
            yield st
            tmp = STATE.with_suffix(".tmp")
            tmp.write_text(json.dumps(st, indent=1) + "\n")
            os.replace(tmp, STATE)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def policy():
    try:
        p = {**DEFAULT_POLICY, **json.loads(POLICY.read_text())}
    except FileNotFoundError:
        p = dict(DEFAULT_POLICY)
    return p


def free_gib():
    """Free memory of this job's card (CUDA_VISIBLE_DEVICES = its UUID under gpu-borrow)."""
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        out = subprocess.run(["nvidia-smi", "-i", dev, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        return float(out[0]) / 1024.0
    except Exception:                                                      # noqa: BLE001
        return 0.0


def archive(task, key):
    for o in task.get("outputs", []):
        p = Path(o)
        if p.exists():
            dst = PACK / "preempted" / f"{p.name}__{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dst))


def requeue(st, card, reason):
    for key, t in st["tasks"].items():
        if t["status"] == "RUNNING" and t.get("card") == card:
            if t["kind"] != "train":                                  # amendment 2: trainings resume in place
                archive(t, key)
            t.update(status="PENDING", card=None, pid=None)
            t.setdefault("requeued", []).append({"at": now(), "reason": reason})


def deps_state(st, t):
    ds = [st["tasks"][d]["status"] for d in t["deps"]]
    if any(s in ("FAILED", "SKIPPED") for s in ds):
        return "dead"
    return "ready" if all(s == "DONE" for s in ds) else "wait"


def cmd_init():
    PACK.mkdir(parents=True, exist_ok=True)
    if STATE.exists():
        raise SystemExit("task table exists: " + str(STATE))
    STATE.write_text(json.dumps({"created_utc": now(), "tasks": tasks(), "cards": {}}, indent=1) + "\n")
    if not POLICY.exists():
        POLICY.write_text(json.dumps(DEFAULT_POLICY, indent=1) + "\n")
    print("task table written:", len(tasks()), "tasks")


def run_cpu(key, t):
    LOGS.mkdir(parents=True, exist_ok=True)
    if t["cmd"] is None:
        return 0
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    with open(LOGS / f"{key.replace(':', '_')}.log", "a") as log:
        return subprocess.run(t["cmd"], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env).returncode


def cmd_run(card):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
    if soft < want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    card = str(card)
    LOGS.mkdir(parents=True, exist_ok=True)
    with locked() as st:
        requeue(st, card, "packer restart (preempted or killed)")
        st["cards"][card] = {"started": now(), "beat": time.time(), "pid": os.getpid(),
                             "device": os.environ.get("CUDA_VISIBLE_DEVICES")}
    procs = {}
    last_start = 0.0
    while True:
        # reap own children
        for key, (p, log) in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            p.wait()
            log.close()
            with locked() as st:
                t = st["tasks"][key]
                t.update(status="DONE" if rc == 0 else "FAILED", rc=rc, ended=now())
            del procs[key]
        stop = (PACK / "STOP").exists()
        pol = policy()
        started = None
        cpu_job = None
        with locked() as st:
            st["cards"][card]["beat"] = time.time()
            for c, info in st["cards"].items():                          # a dead card's unfinished tasks
                if c != card and time.time() - info.get("beat", 0) > STALE_S and info.get("alive", True):
                    requeue(st, c, f"card {c} heartbeat stale > {STALE_S} s")
                    info["alive"] = False
            T = st["tasks"]
            for key, t in T.items():                                      # dependants of failures
                if t["status"] == "PENDING" and deps_state(st, t) == "dead":
                    t.update(status="SKIPPED", ended=now())
            gate = T["gate"]
            if gate["status"] == "PENDING" and deps_state(st, gate) == "ready":
                gate.update(status="DONE", ended=now())
            if not stop:
                for key, t in sorted(T.items(), key=lambda kv: kv[1]["prio"]):
                    if t["status"] != "PENDING" or deps_state(st, t) != "ready" or t["gpu"]:
                        continue
                    t.update(status="RUNNING", card=card, started=now())
                    cpu_job = key
                    break
                if cpu_job is None and time.time() - last_start >= pol["min_start_gap_s"]:
                    mine = [k for k, t in T.items() if t["status"] == "RUNNING" and t.get("card") == card and t["gpu"]]
                    used = sum(pol["est_gib"][T[k]["est"]] for k in mine)
                    if len(mine) < pol["max_jobs"]:
                        free = None
                        for key, t in sorted(T.items(), key=lambda kv: kv[1]["prio"]):
                            if t["status"] != "PENDING" or not t["gpu"] or deps_state(st, t) != "ready":
                                continue
                            need = pol["est_gib"][t["est"]]
                            if used + need > pol["mem_cap_gib"]:
                                continue
                            cap = pol.get("max_per_arm", {}).get(t["arm"]) if t["kind"] in ("smoke", "train") else None
                            if cap is not None and sum(1 for k in mine if T[k]["arm"] == t["arm"]
                                                       and T[k]["kind"] in ("smoke", "train")) >= cap:
                                continue
                            free = free_gib() if free is None else free
                            if free < need + pol["free_margin_gib"]:
                                continue
                            t.update(status="RUNNING", card=card, started=now(), attempt=t.get("attempt", 0) + 1)
                            started = key
                            break
            pending = [k for k, t in T.items() if t["status"] in ("PENDING", "RUNNING")]
        if cpu_job is not None:
            t = json.loads(STATE.read_text())["tasks"][cpu_job]
            rc = run_cpu(cpu_job, t)
            with locked() as st:
                st["tasks"][cpu_job].update(status="DONE" if rc == 0 else "FAILED", rc=rc, ended=now())
            continue
        if started is not None:
            t = json.loads(STATE.read_text())["tasks"][started]
            log = open(LOGS / f"{started.replace(':', '_')}.log", "a")
            log.write(f"== {now()} card {card} device {os.environ.get('CUDA_VISIBLE_DEVICES')} attempt "
                      f"{t.get('attempt')}\n")
            log.flush()
            env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
            p = subprocess.Popen(t["cmd"], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env)
            procs[started] = (p, log)
            with locked() as st:
                st["tasks"][started]["pid"] = p.pid
            last_start = time.time()
            continue
        if not procs and (stop or not [k for k in pending
                                        if json.loads(STATE.read_text())["tasks"][k]["status"] == "PENDING"]):
            break
        time.sleep(5)
    with locked() as st:
        st["cards"][card]["exited"] = now()
        T = st["tasks"]
        failed = [k for k, t in T.items() if t["status"] in ("FAILED", "SKIPPED")]
    print("packer exit, card", card, "failed/skipped:", failed, flush=True)
    sys.exit(1 if failed else 0)


def cmd_status():
    st = json.loads(STATE.read_text())
    by = {}
    for k, t in st["tasks"].items():
        by.setdefault(t["status"], []).append(k)
    for s, ks in sorted(by.items()):
        print(f"{s:8s} {len(ks):3d}  " + " ".join(ks if s != "PENDING" else ks[:8] + (["..."] if len(ks) > 8 else [])))
    for c, info in st["cards"].items():
        print("card", c, {k: v for k, v in info.items() if k != "beat"}, "beat age %.0f s" % (time.time() - info["beat"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verb", choices=("init", "run", "status"))
    ap.add_argument("--card")
    a = ap.parse_args()
    if a.verb == "init":
        cmd_init()
    elif a.verb == "run":
        cmd_run(a.card)
    else:
        cmd_status()


if __name__ == "__main__":
    main()
