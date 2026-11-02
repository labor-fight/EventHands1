#!/usr/bin/env python3
"""S45-CT driver, amendment 2 (docs/EXPERIMENT_RECORDS.md#s45ct-amend2-20261030): resources only.

Amendment 1 allowed two S45-CT trainings at a time; both formal trainings were killed for host memory right after
their first full validation (persistent training + validation workers of two XAG-source runs). As pre-registered in
amendment 1, the trainings now run one pair at a time (MAX_PAIRS = 1). This driver resumes amendment 1's state:
  * jobs that finished EXIT_0 under either earlier driver are carried over and never rerun;
  * formal sigma16 and formal shuffle rerun fresh in their original directories, after the killed ones are moved to
    formal_failed/<run>_attempt1/ (kept); every other failure stays failed (never retried).
Commands, dependencies, GPU slots, row serialisation and fail-closed handling are run.py's. Status: status_r3.json,
logs: logs_r3/. `touch <OUT>/STOP` stops it.
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

MAX_PAIRS = 1
RERUN = {"formal:sigma16": "sigma16", "formal:shuffle": "shuffle"}
PREVIOUS, STATUS, LOGS = OUT / "status_r2.json", OUT / "status_r3.json", OUT / "logs_r3"
MANIFEST = OUT / "manifests" / "amend2.json"


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def prepare():
    verify_protocol()
    require(accepts(__file__, read(MANIFEST)["file_sha256"]["tools/s45ct/run_r3.py"]), "amendment-2 driver changed")
    prev = read(PREVIOUS)
    require(prev["status"] in ("COMPLETE", "COMPLETE_WITH_PRESERVED_FAILURES") and not alive(prev["pid"]),
            "the amendment-1 driver must have ended")
    require(not STATUS.exists(), "fresh-only driver; preserve the previous status")
    for key, arm in RERUN.items():
        require(prev["jobs"][key]["status"] == "FAILED_PRESERVED", "only the memory-killed formal trainings are rerun")
        src = output_dir(arm, "formal")
        dst = OUT / "formal_failed" / f"{run_name(arm)}_attempt1"
        if src.exists():
            require(not dst.exists(), "failed training already archived: " + str(dst))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
    return prev


def run():
    prev = prepare()
    busy = [g for g in PHYSICAL_GPUS if R1.gpu_busy(g)]
    require(not busy, f"GPUs {busy} are in use; S45-CT waits for idle cards among 2-7")
    LOGS.mkdir(parents=True, exist_ok=True)
    graph = R1.jobs()
    commands = {k: cmd for k, _, cmd, _ in graph}
    state = {}
    for key, kind, _, deps in graph:
        old = prev["jobs"][key]["status"]
        carried = old == "EXIT_0" or (old == "FAILED_PRESERVED" and key not in RERUN)   # never rerun, never retried
        state[key] = {"status": old if carried else "PENDING", "kind": kind, "deps": deps, "carried": carried}
    running = {}
    status = {"status": "RUNNING", "pid": os.getpid(), "started_utc": now(), "amendment": "s45ct-amend2-20261030",
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
