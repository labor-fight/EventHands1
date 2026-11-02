#!/usr/bin/env python3
"""S45-PX driver, amendment 1 (docs/EXPERIMENT_RECORDS.md#s45px-amend1-20261030): engineering only.

Resumes tools/s45px/run.py's registered job graph after the first driver has ended:
  * the two preflight trainings (EXIT_0) are carried over and never rerun;
  * their acceptance runs again with the corrected departure step (accept_r1.py) under new keys
    `accept_r1:preflight:<arm>`; the superseded failed acceptances stay in the first status;
  * formal trainings start from train_r1.py and their acceptance is accept_r1.py.
Every other job, dependency, command and resource rule is run.py's. Status: status_r2.json, logs: logs_r2/.
`touch <OUT>/STOP` stops it.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.frozen import accepts  # noqa: E402
from tools.s45px import run as R1  # noqa: E402
from tools.s45px.common import OUT, PHYSICAL_GPUS, now, read, require, save, verify_protocol  # noqa: E402

STATUS, LOGS, MANIFEST = OUT / "status_r2.json", OUT / "logs_r2", OUT / "manifests" / "amend1.json"
RENAME = {f"accept:preflight:{a}": f"accept_r1:preflight:{a}" for a in R1.ORDER}
CARRIED = tuple(f"preflight:{a}" for a in R1.ORDER)


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def jobs():
    J = []
    for key, kind, cmd, deps in R1.jobs():
        deps = [RENAME.get(d, d) for d in deps]
        arm = key.rsplit(":", 1)[-1]
        if key in RENAME:
            key, cmd = RENAME[key], R1.tool("accept_r1.py", "--arm", arm, "--phase", "preflight")
        elif key.startswith("formal:"):
            cmd = [str(ROOT / "tools/s45px/train_r1.py") if c.endswith("tools/s45px/train.py") else c for c in cmd]
        elif key.startswith("accept:formal:"):
            cmd = R1.tool("accept_r1.py", "--arm", arm, "--phase", "formal")
        J.append((key, kind, cmd, deps))
    return J


def prepare():
    verify_protocol()
    require(accepts(__file__, read(MANIFEST)["file_sha256"]["tools/s45px/run_r2.py"]), "amendment-1 driver changed")
    first = read(R1.STATUS)
    require(first["status"] in ("COMPLETE", "COMPLETE_WITH_PRESERVED_FAILURES") and not alive(first["pid"]),
            "the first driver must have ended")
    require(not STATUS.exists(), "fresh-only driver; preserve the previous status")
    for key in CARRIED:
        require(first["jobs"][key]["status"] == "EXIT_0", "preflight training did not finish: " + key)
    for key in RENAME:
        require(first["jobs"][key]["status"] == "FAILED_PRESERVED", "only the superseded acceptances are replaced")
    return first


def run():
    first = prepare()
    LOGS.mkdir(parents=True, exist_ok=True)
    graph = jobs()
    commands = {k: cmd for k, _, cmd, _ in graph}
    state = {}
    for key, kind, _, deps in graph:
        carried = key in CARRIED
        state[key] = {"status": "EXIT_0" if carried else "PENDING", "kind": kind, "deps": deps, "carried": carried}
    running = {}
    status = {"status": "RUNNING", "pid": os.getpid(), "started_utc": now(), "amendment": "s45px-amend1-20261030",
              "jobs": state}
    save(STATUS, status)
    while True:
        if R1.STOP.exists():
            for proc, _ in running.values():
                proc.terminate()
            status.update(status="STOPPED", ended_utc=now())
            save(STATUS, status)
            raise SystemExit("STOP file present")
        for key, (proc, gpus) in list(running.items()):
            if proc.poll() is not None:
                running.pop(key)
                state[key].update(status="EXIT_0" if proc.returncode == 0 else "FAILED_PRESERVED",
                                  exit_code=proc.returncode, ended_utc=now())
        for key, s in state.items():
            if s["status"] == "PENDING" and any(state[d]["status"] in ("FAILED_PRESERVED", "SKIPPED_DEP_FAILED") for d in s["deps"]):
                s.update(status="SKIPPED_DEP_FAILED", ended_utc=now())
        shared = R1.ct_alive()
        gpus_ok, pairs_ok = (R1.WINDOW_GPUS, R1.WINDOW_PAIRS) if shared else (PHYSICAL_GPUS, R1.PAIRS)
        used = {}
        for _, gpus in running.values():
            for g in gpus:
                used[g] = used.get(g, 0) + (2 if len(gpus) == 2 else 1)
        pair_running = any(state[k]["kind"] == "pair" for k in running)
        row_running = any(state[k]["kind"] == "row" for k in running)
        status["shared_with_s45ct"] = shared
        for key, kind, _, deps in graph:
            s = state[key]
            if s["status"] != "PENDING" or not all(state[d]["status"] == "EXIT_0" for d in deps):
                continue
            gpus = None
            if kind == "cpu":
                gpus = ()
            elif kind == "pair" and not row_running and not (shared and pair_running):
                free = [p for p in pairs_ok if all(used.get(g, 0) == 0 for g in p)]
                if free and R1.mem_available_gib() >= R1.MIN_FREE_GIB:
                    gpus = next((p for p in free if all(R1.gpu_used_mib(g) < 1024 for g in p)), None)
            elif kind == "gpu" and not row_running:
                gpus = next(((g,) for g in gpus_ok if used.get(g, 0) < 2), None)
            elif kind == "row":
                gpus = (R1.ROW_GPU,) if not running and not shared else None
            if gpus is None:
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)))
            log = (LOGS / (key.replace(":", "_") + ".log")).open("w")
            running[key] = (subprocess.Popen(commands[key], cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT), gpus)
            s.update(status="RUNNING", gpus=list(gpus), started_utc=now(), shared_with_s45ct=shared)
            for g in gpus:
                used[g] = used.get(g, 0) + (2 if len(gpus) == 2 else 1)
            pair_running = pair_running or kind == "pair"
            if kind == "row":
                break
        save(STATUS, status)
        if not running and not any(s["status"] == "PENDING" and all(state[d]["status"] == "EXIT_0" for d in s["deps"])
                                   for s in state.values()):
            failed = [k for k, s in state.items() if s["status"] != "EXIT_0"]
            status.update(status="COMPLETE" if not failed else "COMPLETE_WITH_PRESERVED_FAILURES", ended_utc=now())
            save(STATUS, status)
            break
        time.sleep(10)


if __name__ == "__main__":
    run()
