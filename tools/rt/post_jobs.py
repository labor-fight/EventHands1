#!/usr/bin/env python3
"""Root-tracking round: scheduler jobs that post-process finished training runs, on CPU.

Per run, chained after its training job: closed-loop selection over the 500-step grid
(`tools/select_checkpoint.py`), then `tools/tracking/evalx.py eval` on the last checkpoint with the
controls, the teacher-forced loop and root perturbation recovery, then evalx on the selected step.
The CPU slices are those of GPUs 0 / 1, which are not this program's to train on; four groups of 8
logical CPUs are handed out in turn.

    python tools/rt/post_jobs.py rt_g3_2k_s3407 [RUN ...] > /tmp/post.json
    SCHED_PROG=outputs/rt python tools/tracking/sched.py submit /tmp/post.json
"""
from __future__ import annotations

import json
import sys

PY = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"
REPO = "/data1/lyq/code/mesh/EventHands1"
GROUPS = [[0, 1, 2, 3, 72, 73, 74, 75], [4, 5, 6, 7, 76, 77, 78, 79],
          [9, 10, 11, 12, 81, 82, 83, 84], [13, 14, 15, 16, 85, 86, 87, 88]]


def jobs_for(run: str, k: int) -> list:
    rd = f"{REPO}/outputs/semkine/{run}"
    base = {"kind": "eval", "cwd": REPO, "device": "cpu", "cpus": GROUPS[k % len(GROUPS)],
            "env": {"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}, "prio": 20,
            "meta": {"run": run, "post": True}}
    sel = dict(base, id=f"{run}__select", kind="select", after=[run],
               cmd=[PY, "tools/select_checkpoint.py", "--run-dir", rd])
    last = dict(base, id=f"{run}__evalx_last", after=[sel["id"]],
                cmd=[PY, "tools/tracking/evalx.py", "eval", "--run-dir", rd, "--ckpt", "last",
                     "--controls", "--tf", "--perturb"])
    chosen = dict(base, id=f"{run}__evalx_sel", after=[sel["id"]],
                  cmd=[PY, "tools/tracking/evalx.py", "eval", "--run-dir", rd, "--ckpt", "selected"])
    return [sel, last, chosen]


if __name__ == "__main__":
    out = []
    for k, run in enumerate(sys.argv[1:]):
        out += jobs_for(run, k)
    json.dump(out, sys.stdout, indent=1)
