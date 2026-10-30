#!/usr/bin/env python3
"""AURA GPU queue (docs/EXPERIMENT_RECORDS.md#aura-prereg-20261024): the pre-registered Stage C processes (and C2) on
physical GPUs 4-7, at most two per GPU, a GPU taking a new job only with >= 20 GiB free (the MRF queue's rules). A job
that fails with a CUDA OOM is preserved under <stage>/failed/<name>_oom<k> and re-queued to run alone on a GPU (same
code, same config); any other failure is preserved and not retried. Writes <stage>/queue.json.

    python tools/aura/gpu_queue.py c | c2
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.agg778 import common as C  # noqa: E402

PY = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"
GPUS = (4, 5, 6, 7)
PER_GPU = 2
MIN_FREE_GIB = 20.0
OUT = ROOT / "outputs" / "aura"
#: heaviest first; the A/A pair last of each regime
C_ORDER = [("e3", "occ"), ("noise", "occ"), ("e3", "aura1"), ("noise", "aura1"), ("e3", "geo"), ("noise", "aura2"),
           ("e3", "aura2"), ("e3", "cal"), ("noise", "cal"), ("e3", "oracle"), ("noise", "oracle"), ("e3", "aa"),
           ("noise", "aa")]


def jobs(stage):
    out = []
    if stage == "c":
        for regime, group in C_ORDER:
            name = f"{regime}_{group}"
            out.append((name, ["tools/aura/c_task.py", "--regime", regime, "--group", group, "--out",
                               f"outputs/aura/c/probe/{name}"], OUT / "c" / "probe" / name))
    elif stage == "c2":
        out.append(("background", ["tools/aura/c_background.py", "--out", "outputs/aura/c/background/run"],
                    OUT / "c" / "background" / "run"))
    else:
        raise SystemExit("usage: gpu_queue.py c | c2")
    return out


def free_gib(gpu):
    q = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return float(q.stdout.strip()) / 1024.0


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    pending = jobs(stage)
    stage_dir = OUT / "c" if stage == "c" else OUT / "c" / "background"
    logs = OUT / "logs" / stage
    logs.mkdir(parents=True, exist_ok=True)
    running, done, failed, alone = {}, [], [], set()
    oom_count = {}
    env0 = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    while pending or running:
        for gpu in GPUS:
            on = [n for n, (_, g, _, _) in running.items() if g == gpu]
            if not pending or len(on) >= PER_GPU or any(n in alone for n in on):
                continue
            name, args, out = pending[0]
            if name in alone and on:
                continue
            if free_gib(gpu) < MIN_FREE_GIB:
                continue
            pending.pop(0)
            log = logs / f"{name}.log"
            if log.exists():
                log.rename(log.with_suffix(f".log.{int(time.time())}"))
            p = subprocess.Popen([PY, *args], cwd=ROOT, env=dict(env0, CUDA_VISIBLE_DEVICES=str(gpu)),
                                 stdout=log.open("w"), stderr=subprocess.STDOUT, start_new_session=True)
            running[name] = (p, gpu, args, out)
            print(datetime.now(timezone.utc).isoformat(), "start", name, "gpu", gpu, flush=True)
            time.sleep(60)                                                       # let the allocation settle
        for name, (p, gpu, args, out) in list(running.items()):
            code = p.poll()
            if code is None:
                continue
            del running[name]
            text = (logs / f"{name}.log").read_text(errors="replace")
            if code == 0 and "COMPLETE" in text:
                done.append(name)
            elif "OutOfMemoryError" in text and oom_count.get(name, 0) < 2:
                oom_count[name] = oom_count.get(name, 0) + 1
                dst = stage_dir / "failed" / f"{name}_oom{oom_count[name]}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if out.exists():
                    shutil.move(str(out), str(dst / "out"))
                shutil.copy(logs / f"{name}.log", dst.with_suffix(".log"))
                C.save_json(dst.with_suffix(".json"), {"status": "FAILED_PRESERVED", "reason": "CUDA OOM",
                                                       "gpu": gpu, "relaunch": "same code alone on a GPU"},
                            exclusive=False)
                alone.add(name)
                pending.insert(0, (name, args, out))
            else:
                failed.append({"name": name, "exit": code})
            print(datetime.now(timezone.utc).isoformat(), "end", name, code, flush=True)
            C.save_json(stage_dir / "queue.json", {"done": done, "failed": failed, "oom_relaunched": oom_count,
                                                   "pending": [j[0] for j in pending],
                                                   "running": {n: v[1] for n, v in running.items()}},
                        exclusive=False)
        time.sleep(30)
    C.save_json(stage_dir / "queue.json", {"done": done, "failed": failed, "oom_relaunched": oom_count,
                                           "complete": not failed}, exclusive=False)
    print("QUEUE", stage, "done", len(done), "failed", failed, flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
