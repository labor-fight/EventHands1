#!/usr/bin/env python3
"""DT2 round: sched2 jobs for arms x seeds (train -> evalx last -> checkpoint grid), configs from configs/dt/ or --cfg-dir.

    python tools/dt/jobs2.py --arms dt_dz_l3_nfh --seeds 3407 3408 --cores 5 --prio 100 > /tmp/j.json
    SCHED_PROG=outputs/dt2 python tools/tracking/sched2.py submit /tmp/j.json
The train command gets `--num-workers {WORKERS}` (sched2 fills it from the cores it hands out).
"""
from __future__ import annotations

import argparse
import json
import sys

PY = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"
REPO = "/data1/lyq/code/mesh/EventHands1"


def jobs_for(arm, seed, prio, cores, mem, cfg_dir, grid, extra, suffix, env=None):
    run = f"{arm}_s{seed}{suffix}"
    rd = f"{REPO}/outputs/semkine/{run}"
    train = {"id": run, "kind": "train", "cwd": REPO, "prio": prio, "cores": cores, "mem_mib": mem,
             "cmd": [PY, "semkine/train.py", "--config", f"{cfg_dir}/{arm}.yaml", "--seed", str(seed),
                     "--run-name", run, "--output-dir", rd, "--num-workers", "{WORKERS}"] + extra}
    if env:
        train["env"] = dict(env)
    last = {"id": f"{run}__evalx_last", "kind": "eval", "cwd": REPO, "after": [run], "prio": prio + 10,
            "cores": 1, "mem_mib": 1500,
            "cmd": [PY, "tools/tracking/evalx.py", "eval", "--run-dir", rd, "--ckpt", "last",
                    "--controls", "--tf", "--perturb"]}
    out = [train, last]
    if grid:
        out.append({"id": f"{run}__grid", "kind": "select", "cwd": REPO, "after": [last["id"]], "prio": prio - 10,
                    "cores": 1, "mem_mib": 1500, "cmd": [PY, "tools/select_checkpoint.py", "--run-dir", rd]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408])
    ap.add_argument("--prio", type=int, default=0)
    ap.add_argument("--cores", type=int, default=5)
    ap.add_argument("--mem", type=int, default=11000)
    ap.add_argument("--cfg-dir", default="configs/dt")
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--fast", action="store_true", help="SEMKINE_FAST_PIPE=1 (bit-identical fast data path)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args()
    out = []
    for arm in a.arms:
        for s in a.seeds:
            out += jobs_for(arm, s, a.prio, a.cores, a.mem, a.cfg_dir, not a.no_grid, a.extra, a.suffix,
                            {"SEMKINE_FAST_PIPE": "1"} if a.fast else None)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
