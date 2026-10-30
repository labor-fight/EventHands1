#!/usr/bin/env python3
"""XVM launcher: pin one job to physical GPU 4/5/6/7 (or the pair 4,5 / 6,7), refuse anything else.

    python tools/xvm/run.py --gpu 6 --name p1 -- python tools/xvm/ladder.py --probe p1

Before launching it checks that no process of another user holds the target GPU(s). The job runs detached under
nohup with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, its log goes to outputs/xvm/logs/<name>.log and its state (launched /
exit code) to outputs/xvm/status.json (`--wait` blocks and records the exit code; otherwise `tools/xvm/run.py
--poll` refreshes the status of finished jobs).
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import getpass
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "xvm"
STATUS = OUT / "status.json"
ALLOWED = {"4", "5", "6", "7", "4,5", "6,7"}
PYTHON = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _locked_update(fn):
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / ".status.lock"
    with open(lock, "w") as fl:
        fcntl.flock(fl, fcntl.LOCK_EX)
        st = json.loads(STATUS.read_text()) if STATUS.exists() else {"jobs": {}}
        fn(st)
        tmp = STATUS.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1, sort_keys=True))
        tmp.replace(STATUS)
        return st


def foreign_processes(gpus):
    """Processes of other users on the given physical GPU indices."""
    idx = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    uuid = {l.split(",")[0].strip(): l.split(",")[1].strip() for l in idx.strip().splitlines()}
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True).stdout
    me = getpass.getuser()
    bad = []
    for line in apps.strip().splitlines():
        if not line.strip():
            continue
        g, pid = (x.strip() for x in line.split(","))
        if g not in {uuid[i] for i in gpus}:
            continue
        user = subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True, text=True).stdout.strip()
        if user and user != me:
            bad.append((pid, user))
    return bad


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def poll():
    def upd(st):
        for name, j in st["jobs"].items():
            if j.get("state") == "running" and not pid_alive(j["pid"]):
                rc_file = Path(j["log"]).with_suffix(".rc")
                j["exit_code"] = int(rc_file.read_text()) if rc_file.exists() else None
                j["state"] = "done" if j["exit_code"] == 0 else "failed"
                j["finished"] = now()
    return _locked_update(upd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu")
    ap.add_argument("--name")
    ap.add_argument("--poll", action="store_true")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.poll:
        st = poll()
        for n, j in sorted(st["jobs"].items()):
            print(f"{n:28s} gpu{j['gpu']:4s} {j['state']:8s} {j.get('exit_code')} {j['log']}")
        return
    if a.gpu not in ALLOWED:
        raise SystemExit(f"XVM uses physical GPU 4-7 only (got {a.gpu!r})")
    if not a.name or not a.cmd:
        raise SystemExit("--name and a command are required")
    cmd = a.cmd[1:] if a.cmd[0] == "--" else a.cmd
    if cmd[0] == "python":
        cmd[0] = PYTHON
    gpus = a.gpu.split(",")
    bad = foreign_processes(gpus)
    if bad:
        raise SystemExit(f"GPU {a.gpu} holds processes of other users: {bad}")
    st = json.loads(STATUS.read_text()) if STATUS.exists() else {"jobs": {}}
    if a.name in st["jobs"] and st["jobs"][a.name]["state"] == "running" and pid_alive(st["jobs"][a.name]["pid"]):
        raise SystemExit(f"job {a.name} is still running")
    log = OUT / "logs" / f"{a.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID", XVM_GPU=a.gpu,
               PYTHONDONTWRITEBYTECODE="1")
    rc = log.with_suffix(".rc")
    if rc.exists():
        rc.unlink()
    shell = f"cd {shlex.quote(str(REPO))} && {' '.join(shlex.quote(c) for c in cmd)} > {shlex.quote(str(log))} 2>&1; " \
            f"echo $? > {shlex.quote(str(rc))}"
    p = subprocess.Popen(["nohup", "bash", "-c", shell], env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

    def upd(s):
        s["jobs"][a.name] = {"gpu": a.gpu, "pid": p.pid, "cmd": cmd, "log": str(log), "launched": now(),
                             "state": "running"}
    _locked_update(upd)
    print(f"launched {a.name} on GPU {a.gpu}: pid {p.pid}, log {log}")
    if a.wait:
        p.wait()
        poll()


if __name__ == "__main__":
    main()
