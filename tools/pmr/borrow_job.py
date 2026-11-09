#!/usr/bin/env python3
"""Inner command of a PMR training on borrowed GPUs (docs/EXPERIMENT_RECORDS.md#pmr-amend4-20261109):

    gpu-borrow run --gpus 0,1 --state-dir <driver>/borrow/<key> -- \\
        python tools/pmr/borrow_job.py --key K --attempts <driver>/attempts/<key>.json --log-dir <driver>/logs -- <training>

gpu-borrow starts the same command again after a preemption, and the driver starts it on a direct pair when it re-homes
a job; every start appends to the job's attempt record. From the second start on, the training's output directory
(TRAIN.OUTPUT_DIR of the command's --config), if present, is the stopped attempt's partial run: it is moved to
outputs/pmr/20261109/preempted/<run>/attempt<k> before the fresh re-run (the training entries are fresh-only and bitwise
deterministic, so the re-run is the same training). An output directory present at the first start is left alone (the
entry then refuses it). The output goes to <log-dir>/<key>.attempt<k>.log; the training command is exec'd, so the DDP
rank-1 relaunch sees the training's own argv.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from tools.pmr.common import OUT, now, read, require, save  # noqa: E402

PREEMPTED = OUT / "preempted"


def safe(key):
    return key.replace(":", "_").replace("/", "_")


def output_dir(command):
    require("--config" in command, "a borrowed job is a training with --config")
    cfg = yaml.safe_load((ROOT / command[command.index("--config") + 1]).read_text())
    return Path(cfg["TRAIN"]["OUTPUT_DIR"])


def begin(key, attempts, command):
    """Record this start; archive the previous attempt's partial run. Returns the attempt number."""
    out = output_dir(command)
    rec = read(attempts) if attempts.exists() else {"key": key, "output_dir": str(out), "attempts": []}
    require(rec["key"] == key and rec["output_dir"] == str(out), "attempt record of another job")
    k = len(rec["attempts"]) + 1
    moved = None
    if k > 1 and out.exists():
        dest = PREEMPTED / out.name / f"attempt{k - 1}"
        require(not dest.exists(), "archive exists: " + str(dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), str(dest))
        moved = str(dest)
    rec["attempts"].append({"attempt": k, "started_utc": now(), "pid": os.getpid(),
                            "gpu_borrow_attempt": os.environ.get("GPU_BORROW_ATTEMPT"),
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "archived_previous": moved})
    attempts.parent.mkdir(parents=True, exist_ok=True)
    save(attempts, rec)
    return k


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    require("--" in argv, "usage: borrow_job.py --key K --attempts F --log-dir D -- <command>")
    i = argv.index("--")
    p = argparse.ArgumentParser()
    p.add_argument("--key", required=True)
    p.add_argument("--attempts", type=Path, required=True)
    p.add_argument("--log-dir", type=Path, required=True)
    a = p.parse_args(argv[:i])
    command = argv[i + 1:]
    k = begin(a.key, a.attempts, command)
    log = (a.log_dir / f"{safe(a.key)}.attempt{k}.log").open("ab")
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
