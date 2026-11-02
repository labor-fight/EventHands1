#!/usr/bin/env python3
"""S45-CT driver, amendment 1 (docs/EXPERIMENT_RECORDS.md#s45ct-amend1-20261030): resources only.

Resumes tools/s45ct/run.py's registered job graph after the first driver has ended:
  * jobs that finished EXIT_0 under the first driver are carried over and never rerun;
  * the two preflights killed for host memory (t1dreplay, sigma16) rerun fresh in their original directories, after
    the failed ones are moved to preflight_failed/<run>_attempt1/ (kept);
  * every S45-CT training (all of them load the XAG sources) runs at most MAX_PAIRS = 2 at a time.
Commands, dependencies, GPU slots, row serialisation and fail-closed handling are run.py's. Status: status_r2.json,
logs: logs_r2/. `touch <OUT>/STOP` stops it.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.s45ct import run as R1  # noqa: E402
from tools.frozen import accepts  # noqa: E402
from tools.s45ct.common import OUT, PHYSICAL_GPUS, now, output_dir, read, require, run_name, save, verify_protocol  # noqa: E402

MAX_PAIRS = 2
RERUN = {"preflight:t1dreplay": "t1dreplay", "preflight:sigma16": "sigma16"}
STATUS, LOGS, MANIFEST = OUT / "status_r2.json", OUT / "logs_r2", OUT / "manifests" / "amend1.json"


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def prepare():
    verify_protocol()
    require(accepts(__file__, read(MANIFEST)["file_sha256"]["tools/s45ct/run_r2.py"]), "amendment-1 driver changed")
    first = read(R1.STATUS)
    require(first["status"] in ("COMPLETE", "COMPLETE_WITH_PRESERVED_FAILURES") and not alive(first["pid"]),
            "the first driver must have ended")
    require(not STATUS.exists(), "fresh-only driver; preserve the previous status")
    for key, arm in RERUN.items():
        require(first["jobs"][key]["status"] == "FAILED_PRESERVED", "only the memory-killed preflights are rerun")
        src = output_dir(arm, "preflight")
        dst = OUT / "preflight_failed" / f"{run_name(arm, 'preflight')}_attempt1"
        if src.exists():
            require(not dst.exists(), "failed preflight already archived: " + str(dst))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
    return first


def run():
    first = prepare()
    busy = [g for g in PHYSICAL_GPUS if R1.gpu_busy(g)]
    require(not busy, f"GPUs {busy} are in use; S45-CT waits for idle cards among 2-7")
    LOGS.mkdir(parents=True, exist_ok=True)
    graph = R1.jobs()
    commands = {k: cmd for k, _, cmd, _ in graph}
    state = {}
    for key, kind, _, deps in graph:
        old = first["jobs"][key]["status"]
        carried = old == "EXIT_0" or (old == "FAILED_PRESERVED" and key not in RERUN)   # never rerun, never retried
        state[key] = {"status": old if carried else "PENDING", "kind": kind, "deps": deps, "carried_from_r1": carried}
    running = {}
    status = {"status": "RUNNING", "pid": os.getpid(), "started_utc": now(), "amendment": "s45ct-amend1-20261030",
              "max_pairs": MAX_PAIRS, "jobs": state}
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
        used = {}
        for _, gpus in running.values():
            for g in gpus:
                used[g] = used.get(g, 0) + (2 if len(gpus) == 2 else 1)
        pairs_running = sum(1 for k in running if state[k]["kind"] == "pair")
        row_running = any(state[k]["kind"] == "row" for k in running)
        for key, kind, _, deps in graph:
            s = state[key]
            if s["status"] != "PENDING" or not all(state[d]["status"] == "EXIT_0" for d in deps):
                continue
            gpus = None
            if kind == "cpu":
                gpus = ()
            elif kind == "pair" and pairs_running < MAX_PAIRS:
                gpus = next((p for p in R1.PAIRS if all(used.get(g, 0) == 0 for g in p)), None)
            elif kind == "gpu" and not row_running:
                gpus = next(((g,) for g in PHYSICAL_GPUS if used.get(g, 0) < 2), None)
            elif kind == "row":
                gpus = (R1.ROW_GPU,) if not running else None
            if gpus is None:
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)))
            log = (LOGS / (key.replace(":", "_") + ".log")).open("w")
            running[key] = (subprocess.Popen(commands[key], cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT), gpus)
            s.update(status="RUNNING", gpus=list(gpus), started_utc=now())
            for g in gpus:
                used[g] = used.get(g, 0) + (2 if len(gpus) == 2 else 1)
            pairs_running += kind == "pair"
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
