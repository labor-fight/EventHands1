#!/usr/bin/env python3
"""S45-LOC driver: stage 0 (two zgz baselines, bit-for-bit against the closed C1SF S45 arrays) -> stage 1 queue on
GPU 4-7, two processes per card -> token quantile map -> stage 2 -> analysis. Stops launching on the first failure;
running tasks finish and are kept. Status in <OUT>/status.json. Run detached:

    setsid nohup python tools/s45loc/chain.py > outputs/s45loc/20261029/chain.out 2>&1 &
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.s45loc import common as C  # noqa: E402

#: approximate packets per task, for longest-first scheduling
SIZE = {"train_base": 23000, "train_local": 11000, "zgz": 2600, "zgz_local": 1200}
B0_FIELDS = ("prediction", "incoming_used", "target", "gt_previous", "end_ms", "run_id", "mpjpe_ra_mm",
             "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")
CACHED = {"b0_zgz_e3": "e3_step6000_repeat1_none", "b0_zgz_e1": "e1_step6000_repeat1_none"}


def status(**fields):
    path = C.OUT / "status.json"
    state = C.read(path) if path.exists() else {"events": []}
    state.update({k: v for k, v in fields.items() if k != "event"})
    if "event" in fields:
        state["events"].append({"at_utc": C.now(), "event": fields["event"]})
    C.save(path, state)


def launch(task, gpu, limit):
    log = C.OUT / "logs" / f"{task}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [C.PYTHON, str(ROOT / "tools/s45loc/run.py"), "--task", task]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log.open("w"), stderr=subprocess.STDOUT)


def run_queue(tasks, limit, slots):
    """Longest first over (gpu, slot) pairs; returns the failed tasks (stops launching after the first failure)."""
    pending = sorted(tasks, key=lambda t: -SIZE[C.TASKS[t][0]])
    running, failed = {}, []
    while pending or running:
        free = [s for s in slots if s not in running.values()]
        while pending and free and not failed:
            task, slot = pending.pop(0), free.pop(0)
            running[launch(task, slot[0], limit)] = slot
            status(event=f"launch {task} on GPU {slot[0]}")
        for proc in list(running):
            code = proc.poll()
            if code is None:
                continue
            task = Path(proc.args[proc.args.index("--task") + 1]).name
            running.pop(proc)
            status(event=f"exit {task} code {code}")
            if code != 0:
                failed.append(task)
        if failed and not running:
            break
        time.sleep(5)
    return failed + ([] if not failed else [f"not launched: {t}" for t in pending])


def check_b0():
    """The zgz baselines must equal the closed C1SF S45 readings array for array."""
    out = {}
    for task, cached in CACHED.items():
        mine = np.load(C.OUT / "tasks" / task / "states.npz", allow_pickle=False)
        ref = np.load(C.S45_EVAL / cached / "states.npz", allow_pickle=False)
        rows = {}
        for seq in C.ZGZ:
            for field in B0_FIELDS:
                a, b = mine[f"{seq}/{field}"], ref[f"{seq}/{field}"]
                rows[f"{seq}/{field}"] = bool(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b))
        out[task] = {"cached": str((C.S45_EVAL / cached).relative_to(C.ROOT)), "fields": rows,
                     "pass": all(rows.values())}
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    C.save(C.OUT / "b0_check.json", out)
    return out["pass"]


def build_tokenmap():
    """Per timing column: zgz_local mid-rank CDF positions -> the training-local quantiles at those positions."""
    train = np.concatenate([np.load(C.OUT / "tasks/b0_train_e3/states.npz")[f"{s}/token_sample"] for s in C.TRAIN_LOCAL])
    zgz = np.load(C.OUT / "tasks/b0_zgz_e3/states.npz")["zgz_local/token_sample"]
    grid = np.linspace(0.0, 1.0, 2001)
    columns = {}
    for j, col in enumerate(C.TOKEN_TIMING_COLS):
        src = np.quantile(zgz[:, j].astype(np.float64), grid)
        dst = np.quantile(train[:, j].astype(np.float64), grid)
        columns[str(col)] = {"src": src.tolist(), "dst": dst.tolist()}
    C.save(C.OUT / "tokenmap.json", {"grid": "2001 equally spaced CDF levels", "columns": columns,
                                     "n_train": int(len(train)), "n_zgz": int(len(zgz))}, exclusive=True)


def main():
    limit = int(os.environ["S45LOC_LIMIT"]) if C.SMOKE and os.environ.get("S45LOC_LIMIT") else None
    proto = C.read(C.PROTOCOL)
    C.require(proto["status"] == ("SMOKE" if C.SMOKE else "FROZEN"), "protocol status / directory mismatch")
    slots = [(g, s) for g in C.PHYSICAL_GPUS for s in range(int(proto["processes_per_gpu"]))]
    status(stage="0", pid=os.getpid(), event="start")
    failed = run_queue(list(CACHED), limit, slots)
    C.require(not failed, f"stage 0 failed: {failed}")
    if limit is None:
        C.require(check_b0(), "stage 0 baselines differ from the closed C1SF S45 arrays: see b0_check.json")
    status(stage="1", event="stage 0 PASS")
    failed = run_queue([t for t in C.STAGE1 if t not in CACHED], limit, slots)
    C.require(not failed, f"stage 1 failed: {failed}")
    build_tokenmap()
    status(stage="2", event="stage 1 complete; tokenmap built")
    failed = run_queue(list(C.STAGE2), limit, slots)
    C.require(not failed, f"stage 2 failed: {failed}")
    status(stage="analysis", event="stage 2 complete")
    code = subprocess.call([C.PYTHON, str(ROOT / "tools/s45loc/analyze.py")], cwd=str(ROOT))
    C.require(code == 0, "analysis failed")
    status(stage="done", event="analysis complete")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        status(stage="FAILED", event=f"{type(exc).__name__}: {exc}")
        raise
