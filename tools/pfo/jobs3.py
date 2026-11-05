#!/usr/bin/env python3
"""PFO phase-2b / router job lists for tools/pfo/chain.py (docs/EXPERIMENT_RECORDS.md#pfo-p2b-prereg-20261103,
#pfo-p2r-prereg-20261103).

    python -m tools.pfo.jobs3 chains --name N --chain arm:seed,... [--chain ...] [--after KEY] --out FILE
    python -m tools.pfo.jobs3 routers --name N --router R [--router R ...] [--swap-trunk T ...] --out FILE

Training chains as tools/pfo/jobs2.py (one training at a time per chain, 40-step preflight per arm, acceptance, the
registered readings) for the phase-2b arms (tools/pfo/train2.py, tools/pfo/run4.py). Router chains: preflight, 6000-step
training, router reading, zero-training swaps into the named trunks (E1, E3) and the reading of sr1 / t0p.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo.arms2 import ARMS2, write_config  # noqa: E402
from tools.pfo.common import PY, ROOT as R, require  # noqa: E402

READINGS = (("e1", 6000), ("e3", 6000), ("e1", 5000), ("e1", 5500), ("tl_e1", 6000), ("pert", 6000), ("f1", 6000))
TRAIN_MEM = 130
ROUTER_MEM = 150


def train_cmd(arm, seed, phase):
    return ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY, str(R / "tools/pfo/train2.py"), "--config",
            str(write_config(arm, seed, phase))]


def chains(name, specs, after=None):
    jobs, pre_done = [], {}
    for chain in specs:
        prev = after
        for arm, seed in chain:
            require(arm in ARMS2, "unregistered arm " + arm)
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
                         "command": [PY, "-m", "tools.pfo.run4", "accept", "--arm", arm, "--seed", str(seed)]})
            e1 = None
            for task, step in READINGS:
                rk = f"{name}:eval:{arm}:{seed}:{task}@{step}"
                jobs.append({"key": rk, "kind": "gpu", "deps": [acc] + ([e1] if task == "f1" else []),
                             "command": [PY, "-m", "tools.pfo.run4", "eval", "--arm", arm, "--seed", str(seed),
                                         "--task", task, "--step", str(step)]})
                if task == "e1" and step == 6000:
                    e1 = rk
            prev = k
    return jobs


def routers(name, names, trunks, after=None, read_refs=True):
    jobs = []
    if read_refs:
        for ref in ("sr1", "t0p"):
            jobs.append({"key": f"{name}:read:{ref}", "kind": "gpu", "deps": [],
                         "command": [PY, "-m", "tools.pfo.router_read", "--router", ref]})
    prev = after
    for r in names:
        pre = f"{name}:pre:{r}"
        jobs.append({"key": pre, "kind": "pair", "deps": [prev] if prev else [], "mem_gib": ROUTER_MEM,
                     "command": [PY, str(R / "tools/pfo/router_train.py"), "--router", r, "--phase", "preflight"]})
        tr = f"{name}:train:{r}"
        jobs.append({"key": tr, "kind": "pair", "deps": [pre], "mem_gib": ROUTER_MEM,
                     "command": [PY, str(R / "tools/pfo/router_train.py"), "--router", r, "--phase", "formal"]})
        jobs.append({"key": f"{name}:read:{r}", "kind": "gpu", "deps": [tr],
                     "command": [PY, "-m", "tools.pfo.router_read", "--router", r]})
        for t in trunks:
            for task in ("e1", "e3"):
                jobs.append({"key": f"{name}:swap:{t}+{r}:{task}", "kind": "gpu", "deps": [tr],
                             "command": [PY, "-m", "tools.pfo.run3", "swap", "--trunk", t, "--router", r, "--task", task]})
        prev = tr
    return jobs


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="what", required=True)
    c = sub.add_parser("chains")
    c.add_argument("--name", required=True)
    c.add_argument("--chain", action="append", required=True)
    c.add_argument("--after")
    c.add_argument("--out", required=True)
    r = sub.add_parser("routers")
    r.add_argument("--name", required=True)
    r.add_argument("--router", action="append", required=True)
    r.add_argument("--swap-trunk", action="append", default=[])
    r.add_argument("--after")
    r.add_argument("--no-refs", action="store_true")
    r.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.what == "chains":
        specs = [[(x.split(":")[0], int(x.split(":")[1])) for x in ch.split(",")] for ch in a.chain]
        jobs = chains(a.name, specs, a.after)
    else:
        jobs = routers(a.name, a.router, a.swap_trunk, a.after, not a.no_refs)
    out = Path(a.out)
    require(not out.exists(), "job file exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
