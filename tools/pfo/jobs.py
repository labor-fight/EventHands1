#!/usr/bin/env python3
"""PFO job lists for tools/pfo/chain.py (docs/EXPERIMENT_RECORDS.md#pfo-prereg-20261103).

    python -m tools.pfo.jobs phase1 --out FILE
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo.common import PY, require  # noqa: E402
from tools.pfo.engine import net_tag  # noqa: E402

D2_TRUNKS = {
    "pfasrdz": ("label", "t0p", "label_onhand", "label_part", "label_onhand_part", "label_loc", "label_shuffle",
                "router_shuffle"),
    "pfat0pdz_cos": ("sr1", "label", "label_onhand", "label_part", "label_onhand_part", "label_loc", "label_shuffle",
                     "router_shuffle"),
    "pfagtdz": ("sr1", "t0p", "label_onhand", "label_part", "label_onhand_part", "label_loc", "label_shuffle",
                "router_shuffle"),
}
TAUS = (0.3, 0.7, 0.9)
D3_ALL = ("dxyz_zero", "dxyz_shuffle", "zn_const", "tok_xy_mean", "tok_time_mean", "tok_pol_zero", "agg_mean_only",
          "agg_max_only", "agg_mass_zero", "gnn_skip", "gnn_rewire", "resid_zero", "prev_fing_mean")
D3_LABEL = ("dxyz_zero", "dxyz_shuffle", "tok_xy_mean", "gnn_skip", "gnn_rewire", "prev_fing_mean")
D1_NETS = ("pfasrdz", "xt0cos_srdwa", "pfat0pdz_cos", "pfagtdz", "s43_s3407")
COST = (("pfasrdz", "fast"), ("pfasrdz", "slow"), ("s43_s3407", "exec"), ("s43_s3407", "dense"),
        ("xt0cos_srdwa", "fast"))


def engine(net, task, route=None, tau=None, iv=()):
    cmd = [PY, "-m", "tools.pfo.engine", "run", "--net", net, "--task", task]
    if route:
        cmd += ["--route", route]
    if tau is not None:
        cmd += ["--tau", f"{tau:g}"]
    if iv:
        cmd += ["--intervene", ",".join(iv)]
    return cmd


def key(net, task, route=None, tau=None, iv=()):
    return f"p1:{net_tag(net, route, tau, iv)}:{task}"


def phase1():
    jobs, seen = [], set()

    def add(k, kind, cmd, deps=(), mem=None, note=""):
        if k in seen:
            return k
        seen.add(k)
        job = {"key": k, "kind": kind, "command": cmd, "deps": list(deps), "note": note}
        if mem:
            job["mem_gib"] = mem
        jobs.append(job)
        return k

    # C0: the cost block, one network x implementation x policy at a time on GPU 4 alone
    last = None
    for net, impl in COST:
        for policy in (("det",) if impl == "slow" else ("det", "default")):
            k = f"p1:cost:{net}:{impl}:{policy}"
            add(k, "exclusive", [PY, "-m", "tools.pfo.cost", "run", "--net", net, "--impl", impl, "--policy", policy],
                deps=[last] if last else [], note="C0 cost block")
            last = k
    after = [last]

    def ev(net, task, route=None, tau=None, iv=()):
        deps = list(after)
        if task in ("loop1", "f1"):
            deps.append(ev(net, "e1", route, tau, iv))
        if task == "cap_zgz_loop":
            deps.append(ev(net, "e1", route, tau))
        return add(key(net, task, route, tau, iv), "gpu", engine(net, task, route, tau, iv), deps)

    # D0 (+ G0) and D1
    g0_deps = [ev("pfasrdz", "e1")] + [ev(f"s43_s{s}", "e1") for s in (3407, 3408, 3409)]
    add("p1:check:g0", "cpu", [PY, "-m", "tools.pfo.check", "g0"], deps=g0_deps, note="G0 equivalence gate")
    for s in (3407, 3408, 3409):
        ev(f"s43_s{s}", "e3")
        ev(f"s43_s{s}", "pert")
    for net in D1_NETS:
        for task in ("e1", "e3", "pert", "loop1", "f1"):
            ev(net, task)
    for net in ("pfasrdz", "xt0cos_srdwa", "pfagtdz", "s43_s3407"):
        ev(net, "tl_e1")
        ev(net, "tl_e3")
    # D2
    for trunk, routes in D2_TRUNKS.items():
        for route in routes:
            ev(trunk, "e1", route)
            ev(trunk, "e3", route)
    for tau in TAUS:
        ev("pfasrdz", "e1", tau=tau)
        ev("pfasrdz", "e3", tau=tau)
    # D3
    for iv in D3_ALL:
        ev("pfasrdz", "e1", iv=(iv,))
        ev("pfasrdz", "e3", iv=(iv,))
    for iv in D3_LABEL:
        ev("pfagtdz", "e1", iv=(iv,))
        ev("pfagtdz", "e3", iv=(iv,))
    # D4
    for net, route in (("pfasrdz", None), ("pfasrdz", "label"), ("pfagtdz", None)):
        for task in ("cap_tl", "cap_zgz_e3", "cap_zgz_loop"):
            ev(net, task, route)
    return jobs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("phase1",))
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    jobs = phase1()
    out = Path(a.out)
    require(not out.exists(), "job file exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()
