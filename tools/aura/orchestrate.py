#!/usr/bin/env python3
"""AURA GPU-stage supervisor (docs/EXPERIMENT_RECORDS.md#aura-prereg-20261024).

Waits until AGG778 R2 and MRF are both complete (their own completion markers, no live process; polled every 10 min)
and GPUs 4-7 are free, then runs, fail-closed and in order: the R2 precondition (`summarize.py r2gate`), the Stage C
queue, C2, `summarize.py c`, the Stage D selection (selection subjects only), the AURA wiring of model/model.py, D
training (preflight then formal, GPU pairs 4-5 / 6-7), D evaluation and `summarize.py final`. Every step's state goes to
outputs/aura/orchestrator.json; a failure stops the chain and leaves its evidence.

    nohup python tools/aura/orchestrate.py > outputs/aura/logs/orchestrator.log 2>&1 &
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.agg778 import common as C  # noqa: E402
from tools.aura import common as AC  # noqa: E402

PY = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"
STATE = AC.OUT / "orchestrator.json"


def note(**kw):
    s = json.loads(STATE.read_text()) if STATE.exists() else {"steps": []}
    s["steps"].append({"utc": datetime.now(timezone.utc).isoformat(), **kw})
    C.save_json(STATE, s, exclusive=False)
    print(json.dumps(kw), flush=True)


def run(step, args):
    note(step=step, status="start")
    r = subprocess.run([PY, *args], cwd=ROOT)
    note(step=step, status="done" if r.returncode == 0 else "FAILED", exit=r.returncode)
    if r.returncode != 0:
        raise SystemExit(f"{step} failed; the chain stops here")


def gpus_free():
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout
    busy = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.splitlines()
    used = {int(x.split(",")[0]): float(x.split(",")[1]) for x in busy if x.strip()}
    return all(used.get(g, 1e9) < 2000 for g in (4, 5, 6, 7)), q


def main():
    AC.verify_protocol("C")
    note(step="wait", status="start")
    while True:
        r2, mrf = AC.r2_complete(), AC.mrf_complete()
        free, _ = gpus_free()
        if r2["complete"] and mrf["complete"] and free:
            break
        time.sleep(600)
    note(step="wait", status="done", r2=r2, mrf=mrf)
    run("r2gate", ["tools/aura/summarize.py", "r2gate"])
    gate = json.loads((AC.OUT / "r2gate.json").read_text())
    if not gate.get("may_read_C"):
        note(step="stop", status="R2 A0/E3 validity failed: Stage C is not run; report to the user")
        return
    v6 = subprocess.run([PY, "tools/aura/v6_check.py", "--out", str(AC.OUT / "c" / "v6.json")], cwd=ROOT,
                        env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "4", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                             "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    note(step="v6", status="done" if v6.returncode == 0 else "FAILED", exit=v6.returncode)
    if v6.returncode != 0:
        raise SystemExit("V6 failed; the chain stops here")
    run("c_queue", ["tools/aura/gpu_queue.py", "c"])
    run("c2_queue", ["tools/aura/gpu_queue.py", "c2"])
    run("summarize_c", ["tools/aura/summarize.py", "c"])
    run("summarize_c2", ["tools/aura/summarize.py", "c2"])
    run("d_select", ["tools/aura/d_stage.py", "select"])
    run("d_wiring", ["tools/aura/d_stage.py", "apply-wiring"])
    run("d_train", ["tools/aura/d_stage.py", "train"])
    run("d_evaluate", ["tools/aura/d_stage.py", "evaluate"])
    run("final", ["tools/aura/summarize.py", "final"])
    note(step="complete", status="done")


if __name__ == "__main__":
    main()
