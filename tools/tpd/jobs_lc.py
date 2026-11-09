#!/usr/bin/env python3
"""TPD LC stage job list for tools/tpd/chain.py (docs/EXPERIMENT_RECORDS.md#tpd-lc-prereg-20261105), written once.

    python -m tools.tpd.jobs_lc lc --out outputs/tpd/20261105/jobs/lc.json

Freeze check; lc_init (r_s2d's weights in LC, history columns zero) and the checks K_LC (probe equal to r_s2d's) and
K_LC2 (L1 + lc_init e1 within 0.05 mm of L1 + r_s2d); the two preflights (pair, 40 steps) and their acceptance; the two
fine-tunes (pair, 3000 steps; 150 GiB declared, PCR's router declaration); acceptance; the probes; the written screen; the
swap readings of the routers that pass (tools/tpd/maybe_lc.py: no-op otherwise); the readout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd.common import PY, ROOT as R, ZGZ, require  # noqa: E402

ROUTERS = ("lc", "lc0")
MEM = 150
TASKS = ("e1", "e3", "pert")


def cpu(key, args, deps, note="", module="tools.tpd.run_lc"):
    return {"key": key, "kind": "cpu", "deps": list(deps), "note": note, "command": [PY, "-m", module, *args]}


def gpu(key, args, deps, note="", condition=None):
    cmd = [PY, "-m", "tools.tpd.run_lc", *args]
    if condition:
        cmd = [PY, "-m", "tools.tpd.maybe_lc", condition, "--", *cmd]
    return {"key": key, "kind": "gpu", "deps": list(deps), "note": note, "command": cmd}


def train(key, router, phase, deps):
    return {"key": key, "kind": "pair", "deps": list(deps), "mem_gib": MEM, "note": f"{router} {phase}",
            "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY, str(R / "tools/tpd/router_lc.py"),
                        "--router", router, "--phase", phase]}


def stage():
    J = [cpu("lc:freeze", ["verify", "--stage", "lc"], [], "manifest unchanged", module="tools.tpd.freeze"),
         cpu("lc:init", ["init"], ["lc:freeze"], "lc_init")]
    probes = []
    for s in ZGZ:
        J.append(gpu(f"lc:probe:r_s2d:{s}", ["probe", "--router", "r_s2d", "--seq", s], ["lc:freeze"], "probe"))
        probes.append(f"lc:probe:r_s2d:{s}")
    J.append(gpu("lc:probe:lc_init:zgz_local", ["probe", "--router", "lc_init", "--seq", "zgz_local"], ["lc:init"],
                 "probe"))
    probes.append("lc:probe:lc_init:zgz_local")
    J.append(cpu("lc:check", ["check"], ["lc:probe:r_s2d:zgz_local", "lc:probe:lc_init:zgz_local"], "K_LC"))
    J.append(gpu("lc:read:lc_init:e1", ["read", "--router", "lc_init", "--task", "e1"], ["lc:check"], "K_LC2 reading"))
    J.append(cpu("lc:check2", ["check2"], ["lc:read:lc_init:e1"], "K_LC2"))
    for r in ROUTERS:
        J.append(train(f"lc:pre:{r}", r, "preflight", ["lc:freeze"]))
        J.append(cpu(f"lc:pre:accept:{r}", ["accept", "--router", r, "--phase", "preflight"], [f"lc:pre:{r}"]))
    for r in ROUTERS:
        J.append(train(f"lc:train:{r}", r, "formal", [f"lc:pre:accept:{r}", "lc:check"]))
        J.append(cpu(f"lc:accept:{r}", ["accept", "--router", r, "--phase", "formal"], [f"lc:train:{r}"]))
        for s in ZGZ:
            J.append(gpu(f"lc:probe:{r}:{s}", ["probe", "--router", r, "--seq", s], [f"lc:accept:{r}"], "probe"))
            probes.append(f"lc:probe:{r}:{s}")
    J.append(cpu("lc:screen", ["screen"], probes, "written identity screen", module="tools.tpd.analyze_lc"))
    reads = ["lc:check2"]
    for r in ROUTERS:
        for t in TASKS:
            J.append(gpu(f"lc:read:{r}:{t}", ["read", "--router", r, "--task", t], ["lc:screen", "lc:check2"], "swap",
                         condition=f"swap:{r}"))
            reads.append(f"lc:read:{r}:{t}")
        J.append(gpu(f"lc:read:{r}:f1", ["read", "--router", r, "--task", "f1"], [f"lc:read:{r}:e1"], "swap",
                     condition=f"swap:{r}"))
        reads.append(f"lc:read:{r}:f1")
    J.append(cpu("lc:analyze", ["all"], reads, "readout", module="tools.tpd.analyze_lc"))
    keys = [j["key"] for j in J]
    require(len(keys) == len(set(keys)), "duplicate job keys")
    seen = set()
    for j in J:
        require(all(d in seen for d in j["deps"]), "dependency before its job: " + j["key"])
        seen.add(j["key"])
    return J


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("lc",))
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    require(not out.exists(), "written once: " + str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = stage()
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
