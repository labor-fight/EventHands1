#!/usr/bin/env python3
"""PFO training-chain job lists for any registered arm (tools/pfo/arms3.py): tools/pfo/jobs3.py `chains`, with
tools/pfo/train3.py and tools/pfo/run5.py as the entries.

    python -m tools.pfo.jobs4 chains --name N --chain arm:seed,... [--chain ...] [--after KEY] [--no-pre] --out FILE
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo import arms3 as A3  # noqa: E402
from tools.pfo import jobs3 as J3  # noqa: E402
from tools.pfo.common import PY, ROOT as R, require  # noqa: E402


def train_cmd(arm, seed, phase):
    return ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY, str(R / "tools/pfo/train3.py"), "--config",
            str(A3.write_config(arm, seed, phase))]


def chains(name, specs, after=None, preflight=True):
    J3.ARMS2, J3.train_cmd = A3.ARMS3, train_cmd
    jobs = J3.chains(name, specs, after)
    for j in jobs:
        j["command"] = [("tools.pfo.run5" if c == "tools.pfo.run4" else c) for c in j["command"]]
    if not preflight:
        pre = {j["key"] for j in jobs if ":pre:" in j["key"]}
        jobs = [j for j in jobs if j["key"] not in pre]
        for j in jobs:
            j["deps"] = [d for d in j["deps"] if d not in pre]
    return jobs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("chains",))
    p.add_argument("--name", required=True)
    p.add_argument("--chain", action="append", required=True)
    p.add_argument("--after")
    p.add_argument("--no-pre", action="store_true", help="arms whose preflight passed in an earlier chain")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    specs = [[(x.split(":")[0], int(x.split(":")[1])) for x in c.split(",")] for c in a.chain]
    jobs = chains(a.name, specs, a.after, not a.no_pre)
    out = Path(a.out)
    require(not out.exists(), "job file exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
