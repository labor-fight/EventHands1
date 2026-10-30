#!/usr/bin/env python3
"""S45-PX driver (docs/EXPERIMENT_RECORDS.md#s45px-prereg-20261030); physical GPUs 2-7 only.

  run.py freeze   after the ledger section is written: protocol.json (file SHAs, configs, constants); refuses to overwrite
  run.py run      the registered job graph, detached-safe: preflights (pxc, pxm; each must reproduce S45's step 1)
                  -> formal pxc, pxm -> E1 at 5000 / 5500 / 6000, E2, E3, the S45-TR readouts, in-sample loops at the
                  late checkpoints -> main rows serially on GPU 4 -> decide.py.
Resources while S45-CT (outputs/s45ct/20261030, driver status_r3.json) is alive: GPUs 6-7 only, one training at a time,
no main row; after it ends, GPUs 2-7. A training starts only on idle cards (< 1 GiB used) with >= 160 GiB host memory
available. A failed job is preserved and never retried; its dependents are skipped. `touch <OUT>/STOP` stops the driver.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.s45px.common import (ARMS, CONFIGS, DEPENDENCIES, OUT, PHYSICAL_GPUS, PROTOCOL, S45_RUN, SOURCES,  # noqa: E402
                                config_path, now, read, require, save, sha)

TAG = "S45PX-PREREG-20261030"
ORDER = ("pxc", "pxm")
PAIRS = ((2, 3), (4, 5), (6, 7))
WINDOW_PAIRS = ((6, 7),)
WINDOW_GPUS = (6, 7)
ROW_GPU = 4
MIN_FREE_GIB = 160
CT_STATUS = ROOT / "outputs/s45ct/20261030/status_r3.json"
STATUS, STOP, LOGS = OUT / "status.json", OUT / "STOP", OUT / "logs"
PY = sys.executable
#: S45-TR readout tasks (tools/s45tr/readouts.py TASKS), the four mechanism readouts
TASKS = ("zl_e3", "tl_e3", "zgz_e1", "tl_e1", "zgz_e1_gtpose", "zgz_e1_gttransl", "tl_e1_gtpose", "tl_e1_gttransl",
         "zl_T10", "zl_R5", "zl_Fs", "tl_T10", "tl_R5", "tl_Fs", "zl_top", "zl_bottom", "tl_top", "tl_bottom",
         "zl_f1", "tl_f1")
AFTER = {"tl_f1": "tl_e1", "zl_f1": "zgz_e1"}


def tool(name, *args):
    return [PY, str(ROOT / "tools/s45px" / name), *args]


def jobs():
    """(key, kind, command, deps); kind: pair | gpu | row | cpu."""
    J = []
    for arm in ORDER:
        J.append((f"preflight:{arm}", "pair", tool("train.py", "--arm", arm, "--phase", "preflight",
                                                   "--config", str(config_path(arm, "preflight"))), []))
        J.append((f"accept:preflight:{arm}", "cpu", tool("accept.py", "--arm", arm, "--phase", "preflight"), [f"preflight:{arm}"]))
    pre = [f"accept:preflight:{a}" for a in ORDER]
    reads = []
    for arm in ORDER:
        J.append((f"formal:{arm}", "pair", tool("train.py", "--arm", arm, "--phase", "formal",
                                                "--config", str(config_path(arm, "formal"))), list(pre)))
        J.append((f"accept:formal:{arm}", "cpu", tool("accept.py", "--arm", arm, "--phase", "formal"), [f"formal:{arm}"]))
        done = f"accept:formal:{arm}"
        for step in ("5000", "5500", "6000"):
            J.append((f"eval:{arm}:e1:{step}", "gpu", tool("evaluate.py", "--net", arm, "--mode", "e1", "--step", step), [done]))
            reads.append(f"eval:{arm}:e1:{step}")
        for mode in ("e2", "e3"):
            J.append((f"eval:{arm}:{mode}", "gpu", tool("evaluate.py", "--net", arm, "--mode", mode), [done]))
            reads.append(f"eval:{arm}:{mode}")
        for task in TASKS:
            deps = [done] + ([f"ro:{arm}:{AFTER[task]}"] if task in AFTER else [])
            J.append((f"ro:{arm}:{task}", "gpu", tool("readouts.py", "run", "--net", arm, "--task", task), deps))
            reads.append(f"ro:{arm}:{task}")
        for step in ("5000", "5500"):
            J.append((f"ro:{arm}_{step}:tl_e1", "gpu", tool("readouts.py", "run", "--net", f"{arm}_{step}", "--task", "tl_e1"), [done]))
            reads.append(f"ro:{arm}_{step}:tl_e1")
        J.append((f"roa:{arm}", "cpu", tool("readouts.py", "analyze", "--net", arm), [f"ro:{arm}:{t}" for t in TASKS]))
        reads.append(f"roa:{arm}")
    J.append(("row:pxc", "row", tool("evaluate.py", "--net", "pxc", "--mode", "row"), list(reads)))
    J.append(("row:pxm", "row", tool("evaluate.py", "--net", "pxm", "--mode", "row"), ["row:pxc"]))
    J.append(("final", "cpu", tool("decide.py"), list(reads) + ["row:pxc", "row:pxm"]))
    return J


def gpu_used_mib(gpu):
    out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return int(out.splitlines()[0])


def mem_available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20
    raise RuntimeError("MemAvailable missing")


def ct_alive():
    """S45-CT's amendment-2 driver is still running (its evaluations and main rows need GPUs 2-7)."""
    try:
        s = read(CT_STATUS)
        os.kill(int(s["pid"]), 0)
        return s["status"] == "RUNNING"
    except (OSError, KeyError, ValueError):
        return False


def run():
    from tools.s45px.common import verify_protocol
    verify_protocol()
    require(not STATUS.exists(), "fresh-only driver; preserve the previous status")
    LOGS.mkdir(parents=True, exist_ok=True)
    graph = jobs()
    state = {k: {"status": "PENDING", "kind": kind, "deps": deps} for k, kind, _, deps in graph}
    commands = {k: cmd for k, _, cmd, _ in graph}
    running = {}                                       # key -> (Popen, gpus)
    status = {"status": "RUNNING", "pid": os.getpid(), "started_utc": now(), "jobs": state}
    save(STATUS, status)
    while True:
        if STOP.exists():
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
        shared = ct_alive()
        gpus_ok, pairs_ok = (WINDOW_GPUS, WINDOW_PAIRS) if shared else (PHYSICAL_GPUS, PAIRS)
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
                if free and mem_available_gib() >= MIN_FREE_GIB:
                    gpus = next((p for p in free if all(gpu_used_mib(g) < 1024 for g in p)), None)
            elif kind == "gpu" and not row_running:
                gpus = next(((g,) for g in gpus_ok if used.get(g, 0) < 2), None)
            elif kind == "row":
                gpus = (ROW_GPU,) if not running and not shared else None
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


def freeze():
    require(not PROTOCOL.exists(), "S45-PX protocol already frozen")
    text = (ROOT / "docs/EXPERIMENT_RECORDS.md").read_text()
    begin, end = f"<!-- EXPERIMENT_RECORD:{TAG}:BEGIN -->", f"<!-- EXPERIMENT_RECORD:{TAG}:END -->"
    require(text.count(begin) == 1 and text.count(end) == 1, "write the pre-registration section first")
    section = OUT / "prereg_section.md"
    OUT.mkdir(parents=True, exist_ok=True)
    with section.open("x") as stream:
        stream.write(text[text.index(begin):text.index(end) + len(end)])
    configs = sorted(str(p.relative_to(ROOT)) for p in CONFIGS.glob("*.yaml"))
    files = {name: sha(ROOT / name) for name in (*SOURCES, *DEPENDENCIES, *configs)}
    s45 = read(ROOT / "outputs/c1sf/20261029/receipts/formal_so3.json")
    from tools.s45px.decide import TH
    save(PROTOCOL, {
        "study": "S45PX_20261030", "status": "FROZEN", "frozen_utc": now(),
        "prereg": "docs/EXPERIMENT_RECORDS.md#s45px-prereg-20261030", "prereg_section_sha256": sha(section),
        "arms": ARMS, "order": ORDER, "thresholds": TH,
        "control": {"s45": {"run": str(S45_RUN.relative_to(ROOT)), "checkpoint_sha256_by_step": s45["checkpoint_sha256_by_step"]}},
        "jobs": [[k, kind, deps] for k, kind, _, deps in jobs()],
        "resources": {"physical_gpus": PHYSICAL_GPUS, "pairs": PAIRS, "while_s45ct_alive": {"pairs": WINDOW_PAIRS,
                      "gpus": WINDOW_GPUS, "one_training_at_a_time": True, "no_main_row": True},
                      "row_gpu": ROW_GPU, "min_free_host_gib_to_start_training": MIN_FREE_GIB},
        "file_sha256": files, "seed": 3407, "single_seed_descriptive": True,
    }, exclusive=True)
    print(PROTOCOL)


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    require(action in ("freeze", "run"), "usage: run.py freeze | run")
    freeze() if action == "freeze" else run()


if __name__ == "__main__":
    main()
