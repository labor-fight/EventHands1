#!/usr/bin/env python3
"""S38 (docs/S38_ROOT_TRACKING_PREREG.md): scheduler jobs for a set of arms x seeds.

Per run: training (`semkine/train.py`, one GPU), then on a GPU the evaluation of the *last* checkpoint
(`tools/tracking/evalx.py eval --ckpt last --controls --tf --perturb`; S38 reports no selected step), then
the closed-loop RA of every 500-step checkpoint (`tools/select_checkpoint.py`), whose grid median is
reported beside the last step as a robustness figure and never used to pick a checkpoint.

    python tools/s38/jobs.py --arms s38_spabs_2k s38_spdelta_2k --seeds 3407 3408 > /tmp/jobs.json
    SCHED_PROG=outputs/s38 python tools/tracking/sched.py submit /tmp/jobs.json
"""
from __future__ import annotations

import argparse
import json
import sys

PY = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"
REPO = "/data1/lyq/code/mesh/EventHands1"


def jobs_for(arm: str, seed: int, prio: int) -> list:
    run = f"{arm}_s{seed}"
    rd = f"{REPO}/outputs/semkine/{run}"
    train = {"id": run, "kind": "train", "cwd": REPO, "prio": prio,
             "cmd": [PY, "semkine/train.py", "--config", f"configs/s38/{arm}.yaml", "--seed", str(seed),
                     "--run-name", run, "--output-dir", rd]}
    last = {"id": f"{run}__evalx_last", "kind": "eval", "cwd": REPO, "after": [run], "prio": prio + 10,
            "cmd": [PY, "tools/tracking/evalx.py", "eval", "--run-dir", rd, "--ckpt", "last",
                    "--controls", "--tf", "--perturb"]}
    grid = {"id": f"{run}__grid", "kind": "select", "cwd": REPO, "after": [last["id"]], "prio": prio - 10,
            "cmd": [PY, "tools/select_checkpoint.py", "--run-dir", rd]}
    return [train, last, grid]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408])
    ap.add_argument("--prio", type=int, default=0)
    a = ap.parse_args()
    out = []
    for arm in a.arms:
        for s in a.seeds:
            out += jobs_for(arm, s, a.prio)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
