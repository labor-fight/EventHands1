#!/usr/bin/env python3
"""S45-CT driver (docs/EXPERIMENT_RECORDS.md#s45ct-prereg-20261030); physical GPUs 2-7 only.

  run.py freeze   after the ledger section is written: protocol.json (file SHAs, configs, constants); refuses to overwrite
  run.py run      the registered job graph, detached-safe: preflights (t1dreplay / sigma16 / shuffle, one pair each)
                  -> step-0 acceptance (audits, proxy reproduction, T1d re-evaluation, S45 / T1d late in-sample loops)
                  -> formal sigma16 (pair) and shuffle (pair) -> E1 at 5000 / 5500 / 6000, E2, E3, readouts
                  -> main rows serially on GPU 4 -> decide.py final.
A failed job is preserved and never retried; its dependents are skipped. `touch <OUT>/STOP` stops the driver.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.s45ct.common import (CONFIGS, DEPENDENCIES, FORMAL_ARMS, OUT, PHYSICAL_GPUS, PROTOCOL, S45_RUN, SOURCES,  # noqa: E402
                                T1D_RUN, config_path, now, read, require, save, sha)

TAG = "S45CT-PREREG-20261030"
PAIRS = ((2, 3), (4, 5), (6, 7))
ROW_GPU = 4
STATUS, STOP, LOGS = OUT / "status.json", OUT / "STOP", OUT / "logs"
PY = sys.executable


def tool(name, *args):
    return [PY, str(ROOT / "tools/s45ct" / name), *args]


def jobs():
    """(key, kind, command, deps); kind: pair | gpu | row | cpu."""
    from tools.s45ct.proxy import ylf_sequences
    J = []
    for arm in ("t1dreplay",) + FORMAL_ARMS:
        J.append((f"preflight:{arm}", "pair", tool("train.py", "--arm", arm, "--phase", "preflight",
                                                   "--config", str(config_path(arm, "preflight"))), []))
        J.append((f"accept:preflight:{arm}", "cpu", tool("accept.py", "--arm", arm, "--phase", "preflight"), [f"preflight:{arm}"]))
    step0 = [f"accept:preflight:{a}" for a in ("t1dreplay",) + FORMAL_ARMS]
    for mode in FORMAL_ARMS:
        J.append((f"audit:{mode}", "gpu", tool("audit.py", "--mode", mode), []))
        step0.append(f"audit:{mode}")
    seqs = ylf_sequences()
    for seq in seqs:
        J.append((f"proxy:{seq}", "gpu", tool("proxy.py", "run", "--seq", seq), []))
    J.append(("proxy:analyze", "cpu", tool("proxy.py", "analyze"), [f"proxy:{s}" for s in seqs]))
    J.append(("eval:t1d:e1:6000", "gpu", tool("evaluate.py", "--net", "t1d", "--mode", "e1"), []))
    for net in ("s45_5000", "s45_5500", "t1d_5000", "t1d_5500"):
        J.append((f"ro:{net}:tl_e1", "gpu", tool("readouts.py", "--net", net, "--task", "tl_e1"), []))
    step0 += ["proxy:analyze", "eval:t1d:e1:6000"] + [f"ro:{n}:tl_e1" for n in ("s45_5000", "s45_5500", "t1d_5000", "t1d_5500")]
    J.append(("accept:step0", "cpu", tool("decide.py", "acceptance"), step0))
    reads = []
    for arm in FORMAL_ARMS:
        J.append((f"formal:{arm}", "pair", tool("train.py", "--arm", arm, "--phase", "formal",
                                                "--config", str(config_path(arm, "formal"))), ["accept:step0"]))
        J.append((f"accept:formal:{arm}", "cpu", tool("accept.py", "--arm", arm, "--phase", "formal"), [f"formal:{arm}"]))
        done = f"accept:formal:{arm}"
        for step in ("5000", "5500", "6000"):
            J.append((f"eval:{arm}:e1:{step}", "gpu", tool("evaluate.py", "--net", arm, "--mode", "e1", "--step", step), [done]))
            net = arm if step == "6000" else f"{arm}_{step}"
            J.append((f"ro:{net}:tl_e1", "gpu", tool("readouts.py", "--net", net, "--task", "tl_e1"), [done]))
            reads += [f"eval:{arm}:e1:{step}", f"ro:{net}:tl_e1"]
        for mode in ("e2", "e3"):
            J.append((f"eval:{arm}:{mode}", "gpu", tool("evaluate.py", "--net", arm, "--mode", mode), [done]))
            reads.append(f"eval:{arm}:{mode}")
        J.append((f"ro:{arm}:tl_f1", "gpu", tool("readouts.py", "--net", arm, "--task", "tl_f1"), [f"ro:{arm}:tl_e1"]))
        for task in ("tl_e3", "zl_e3"):
            J.append((f"ro:{arm}:{task}", "gpu", tool("readouts.py", "--net", arm, "--task", task), [done]))
        reads += [f"ro:{arm}:tl_f1", f"ro:{arm}:tl_e3", f"ro:{arm}:zl_e3"]
    J.append(("row:sigma16", "row", tool("evaluate.py", "--net", "sigma16", "--mode", "row"), list(reads)))
    J.append(("row:shuffle", "row", tool("evaluate.py", "--net", "shuffle", "--mode", "row"), ["row:sigma16"]))
    J.append(("final", "cpu", tool("decide.py", "final"), list(reads) + ["row:sigma16", "row:shuffle"]))
    return J


def gpu_busy(gpu):
    used = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return int(used.splitlines()[0]) > 1000


def run():
    from tools.s45ct.common import verify_protocol
    verify_protocol()
    require(not STATUS.exists(), "fresh-only driver; preserve the previous status")
    busy = [g for g in PHYSICAL_GPUS if gpu_busy(g)]
    require(not busy, f"GPUs {busy} are in use; S45-CT waits for idle cards among 2-7")
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
        used = {}
        for _, gpus in running.values():
            for g in gpus:
                used[g] = used.get(g, 0) + (2 if len(gpus) == 2 else 1)
        row_running = any(state[k]["kind"] == "row" for k in running)
        for key, kind, _, deps in graph:
            s = state[key]
            if s["status"] != "PENDING" or not all(state[d]["status"] == "EXIT_0" for d in deps):
                continue
            gpus = None
            if kind == "cpu":
                gpus = ()
            elif kind == "pair":
                gpus = next((p for p in PAIRS if all(used.get(g, 0) == 0 for g in p)), None)
            elif kind == "gpu" and not row_running:
                gpus = next(((g,) for g in PHYSICAL_GPUS if used.get(g, 0) < 2), None)
            elif kind == "row":
                gpus = (ROW_GPU,) if not running else None
            if gpus is None:
                continue
            env = dict(os.environ)
            if gpus:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
            log = (LOGS / (key.replace(":", "_") + ".log")).open("w")
            running[key] = (subprocess.Popen(commands[key], cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT), gpus)
            s.update(status="RUNNING", gpus=list(gpus), started_utc=now())
            for g in gpus:
                used[g] = used.get(g, 0) + (2 if len(gpus) == 2 else 1)
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
    require(not PROTOCOL.exists(), "S45-CT protocol already frozen")
    text = (ROOT / "docs/EXPERIMENT_RECORDS.md").read_text()
    begin, end = f"<!-- EXPERIMENT_RECORD:{TAG}:BEGIN -->", f"<!-- EXPERIMENT_RECORD:{TAG}:END -->"
    require(text.count(begin) == 1 and text.count(end) == 1, "write the pre-registration section first")
    section = OUT / "prereg_section.md"
    OUT.mkdir(parents=True, exist_ok=True)
    with section.open("x") as stream:
        stream.write(text[text.index(begin):text.index(end) + len(end)])
    configs = sorted(str(p.relative_to(ROOT)) for p in CONFIGS.glob("*.yaml"))
    files = {name: sha(ROOT / name) for name in (*SOURCES, *DEPENDENCIES, *configs)}
    t1d = read(ROOT / "outputs/s45tr/20261030/receipts/formal_t1d.json")
    s45 = read(ROOT / "outputs/c1sf/20261029/receipts/formal_so3.json")
    save(PROTOCOL, {
        "study": "S45CT_20261030", "status": "FROZEN", "frozen_utc": now(),
        "prereg": "docs/EXPERIMENT_RECORDS.md#s45ct-prereg-20261030", "prereg_section_sha256": sha(section),
        "arms": {"t1dreplay": None, "sigma16": "sigma16", "shuffle": "shuffle"}, "formal_arms": FORMAL_ARMS,
        "controls": {"t1d": {"run": str(T1D_RUN.relative_to(ROOT)), "checkpoint_sha256_by_step": t1d["checkpoint_sha256_by_step"]},
                     "s45": {"run": str(S45_RUN.relative_to(ROOT)), "checkpoint_sha256_by_step": s45["checkpoint_sha256_by_step"]}},
        "jobs": [[k, kind, deps] for k, kind, _, deps in jobs()],
        "physical_gpus": PHYSICAL_GPUS, "pairs": PAIRS, "row_gpu": ROW_GPU,
        "file_sha256": files, "single_seed_descriptive": True,
    }, exclusive=True)
    print(PROTOCOL)


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    require(action in ("freeze", "run"), "usage: run.py freeze | run")
    freeze() if action == "freeze" else run()


if __name__ == "__main__":
    main()
