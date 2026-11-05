#!/usr/bin/env python3
"""PFO phase-2 job lists for tools/pfo/chain.py (docs/EXPERIMENT_RECORDS.md#pfo-p2a-prereg-20261103 and later sections).

    python -m tools.pfo.jobs2 chains --name N --chain arm:seed,arm:seed,... [--chain ...] [--after KEY] --out FILE

Each chain trains its runs one after another on one GPU pair (so at most one training per chain); every distinct arm
gets a 40-step preflight first. After a training: acceptance (CPU), then the registered readings (one GPU each).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo.arms import ARMS, config_path, write_config  # noqa: E402
from tools.pfo.common import PY, ROOT as R, require  # noqa: E402

READINGS = (("e1", 6000), ("e3", 6000), ("e1", 5000), ("e1", 5500), ("tl_e1", 6000), ("pert", 6000), ("f1", 6000))
TRAIN_MEM = 130


def train_cmd(arm, seed, phase):
    return ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY, str(R / "tools/pfo/train.py"), "--config",
            str(write_config(arm, seed, phase))]


def chains(name, specs, after=None):
    jobs, pre_done = [], {}
    for chain in specs:
        prev = after
        for arm, seed in chain:
            require(arm in ARMS, "unregistered arm " + arm)
            if arm not in pre_done:
                k = f"{name}:pre:{arm}"
                jobs.append({"key": k, "kind": "pair", "command": train_cmd(arm, 3407, "preflight"),
                             "deps": [prev] if prev else [], "mem_gib": TRAIN_MEM, "note": "preflight"})
                pre_done[arm] = k
                prev = k
            k = f"{name}:train:{arm}:{seed}"
            jobs.append({"key": k, "kind": "pair", "command": train_cmd(arm, seed, "formal"),
                         "deps": [d for d in (prev, pre_done[arm]) if d], "mem_gib": TRAIN_MEM})
            acc = f"{name}:accept:{arm}:{seed}"
            jobs.append({"key": acc, "kind": "cpu", "deps": [k],
                         "command": [PY, "-m", "tools.pfo.run2", "accept", "--arm", arm, "--seed", str(seed)]})
            e1 = None
            for task, step in READINGS:
                rk = f"{name}:eval:{arm}:{seed}:{task}@{step}"
                deps = [acc] + ([e1] if task == "f1" else [])
                jobs.append({"key": rk, "kind": "gpu", "deps": deps,
                             "command": [PY, "-m", "tools.pfo.run2", "eval", "--arm", arm, "--seed", str(seed),
                                         "--task", task, "--step", str(step)]})
                if task == "e1" and step == 6000:
                    e1 = rk
            prev = k
    return jobs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("chains",))
    p.add_argument("--name", required=True)
    p.add_argument("--chain", action="append", required=True)
    p.add_argument("--after")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    specs = [[(x.split(":")[0], int(x.split(":")[1])) for x in c.split(",")] for c in a.chain]
    jobs = chains(a.name, specs, a.after)
    out = Path(a.out)
    require(not out.exists(), "job file exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
