#!/usr/bin/env python3
"""IDP readings, acceptance and engine checks (docs/EXPERIMENT_RECORDS.md#idp-prereg-20261105), through PFO's frozen
engine (tools/pfo/engine.py, tools/pfo/nets.py, unchanged) with the output root redirected to outputs/idp/20261105.

    python -m tools.idp.run b2 --task e1 --route router|label_part
        B2 (PFO2 fgs seed 3407, step 6000, the accepted checkpoint) read through this study's entry: engine check K1
        (route router, against PFO2's published e1) and K2 (route label_part, against B2UB's published reading); tags
        b2_s3407 / b2_s3407__label_part. The router of the mix is B2's deployed router r_s2d (tools/pfo/nets.py
        `_router_for` is re-bound in this process only, as B2UB did)
    python -m tools.idp.run read --arm A --task T [--step 6000] [--deployed]
        an arm's reading at the accepted checkpoint; the route is the arm's (l1 / l2: label_part, diagnostic; l3: the
        deployed router); --deployed reads a label_part arm with the deployed router instead (exposure, descriptive);
        tags idp_<arm>_s3407[__label_part]
    python -m tools.idp.run accept --arm A --phase preflight|formal
        training acceptance against B2 seed 3407 (same phase); receipts/<run>.json; exit 1 on failure
    python -m tools.idp.run check
        K1 / K2 (bitwise); receipts/k12.json; exit 1 on failure
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.idp import arms as A  # noqa: E402
from tools.idp.common import (CONFIGS, OUT, SEED, ZGZ, b2_reading, b2_run, b2_spec, b2ub_reading, check_b2, now, read,  # noqa: E402
                              require, save, sha)

TASKS = ("e1", "e3", "pert", "f1", "tl_e1")
GRID = {"preflight": (40,), "formal": tuple(range(500, 6001, 500))}
STEPS = (5000, 5500, 6000)


# --------------------------------------------------------------------------- networks
def deployed_router_for_mixes():
    """tools/pfo/nets.py builds its route mixes on sr1; B2's deployed router is the config's (r_s2d)."""
    from tools.pfo import nets as N
    from tools.pfo.common import T0P

    def router_for(route, cfg):
        if route == "t0p":
            return "t0p", str(T0P)
        s = cfg["S45SM"]
        return s["ROUTER_KIND"], s["ROUTER_CHECKPOINT"]
    N._router_for = router_for


def write_once(path, obj):
    from tools.pfo2.specs import write_once_text
    if str(path).endswith(".json"):
        return write_once_text(path, json.dumps(obj, indent=2, allow_nan=False) + "\n")
    import yaml
    return write_once_text(path, yaml.safe_dump(obj, sort_keys=False))


def eval_config(arm):
    from tools.pfo2.run import _eval_cfg
    name = A.run_name(arm, SEED)
    return write_once(CONFIGS / "eval" / f"{name}.yaml", _eval_cfg(A.build_config(arm, SEED), name))


def checkpoint(arm, step, phase="formal"):
    return A.run_dir(arm, SEED, phase) / f"{A.run_name(arm, SEED, phase)}-step={step}.ckpt"


def arm_spec(arm, step=6000):
    receipt = read(OUT / "receipts" / f"{A.run_name(arm, SEED)}.json")
    require(receipt["status"] == "PASS", "training not accepted: " + arm)
    ck = checkpoint(arm, step)
    require(sha(ck) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since acceptance")
    tag = A.run_name(arm, SEED) + ("" if step == 6000 else f"@{step}")
    sp = {"kind": "routed", "checkpoint": str(ck), "checkpoint_sha256": receipt["checkpoint_sha256_by_step"][str(step)],
          "config": str(eval_config(arm)), "trunk": "gshare", "module": "trunk4", "feat": A.ARMS[arm]["feat"],
          "train_routes": A.ARMS[arm]["train_routes"], "arm": arm}
    return tag, write_once(OUT / "specs" / f"{tag}.json", sp)


def prepare(sp):
    """Register the router kind and install the trunk the spec names (one network per process)."""
    import yaml
    s = read(sp)
    cfg = yaml.safe_load(Path(s["config"]).read_text())
    if cfg["S45SM"]["ROUTER_KIND"] == "pfr":
        from semkine import pfo_router as PR
        PR.register()
    if s.get("feat") == "fore":
        from semkine import idp_feat as IF
        IF.install()
    else:
        from tools.pfo2.train import install_trunk
        install_trunk(s["module"], s["trunk"])
    return s


def engine(tag, task, sp, route=None):
    from tools.pfo import engine as E
    prepare(sp)
    deployed_router_for_mixes()
    E.OUT = OUT                                                     # readings go under outputs/idp/20261105
    return E.run(tag, task, None if route in (None, "router") else route, None, (), str(sp))


# --------------------------------------------------------------------------- acceptance
def stream_rows(run_dir):
    return {r: [x["sha256"] for x in read(Path(run_dir) / f"pair_stream_rank{r}.json")[:2]] for r in (0, 1)}


def accept(arm, phase):
    require(arm in A.ARMS and phase in GRID, "arm / phase")
    name = A.run_name(arm, SEED, phase)
    out = OUT / "receipts" / f"{name}.json"
    require(not out.exists(), "accepted once")
    d = A.run_dir(arm, SEED, phase)
    launch, meta, det = read(d / "idp_launch.json"), read(d / "training_metadata.json"), read(d / "idp_det.json")
    cfg = A.build_config(arm, SEED, phase)
    s = A.ARMS[arm]
    mine, b2 = stream_rows(d), stream_rows(b2_run(phase))
    routing, trunk = launch["routing"], launch["trunk"]
    expect_sources = "tools.pfo.nets.mixed_sources" if s["train_routes"] == "label_part" else "semkine.s45sm.predicted_sources"
    ok = {"steps": all(checkpoint(arm, st, phase).exists() for st in GRID[phase]),
          "seed": meta["seed"] == SEED and launch["pfo"]["SEED"] == SEED,
          "max_steps": meta["max_steps"] == cfg["TRAIN"]["MAX_STEPS"],
          "lr_schedule": meta["lr_schedule"] == "cosine",
          "devices": meta["devices"] == 2 and meta["batch_size_per_gpu"] == 512,
          "config": launch["config_sha256"] == sha(launch["config"]) and Path(launch["config"]) == A.config_path(
              arm, SEED, phase).resolve(),
          "deterministic": bool(det["deterministic_algorithms"]) and det["trainer_deterministic"],
          "router": routing["router"] == "pfr" and routing["assoc"] == "pfa" and float(routing["onhand_tau"]) == 0.5
          and launch["router_checkpoint_sha256"] == sha(cfg["PFO"]["ROUTER_CHECKPOINT"]),
          "train_routes": routing["train_routes"] == s["train_routes"] and routing["sample_sources"] == expect_sources
          and routing.get("mix") == ("label_part" if s["train_routes"] == "label_part" else None),
          "sources": (launch["xag_source_root"] is not None) == (s["train_routes"] == "label_part"),
          "trunk": trunk["feat"] == s["feat"] and trunk["trunk"] == "gshare"
          and trunk["module"] == ("idp_feat" if s["feat"] == "fore" else "trunk4"),
          "data_stream_equals_b2": mine == b2}
    receipt = {"run": name, "arm": arm, "seed": SEED, "phase": phase, "checks": ok, "b2_run": str(b2_run(phase)),
               "status": "PASS" if all(ok.values()) else "FAIL", "accepted_at_utc": now(),
               "checkpoint_sha256_by_step": {str(st): sha(checkpoint(arm, st, phase)) for st in GRID[phase]
                                             if checkpoint(arm, st, phase).exists()}}
    save(out, receipt, exclusive=True)
    print(receipt["status"], ok, flush=True)
    require(receipt["status"] == "PASS", "acceptance failed")


# --------------------------------------------------------------------------- engine checks
def _states(d):
    return np.load(Path(d) / "states.npz", allow_pickle=False)


def _same(a, b, keys):
    return {k: bool(k in a.files and k in b.files and a[k].shape == b[k].shape and np.array_equal(a[k], b[k]))
            for k in keys}


def readings(tag, task):
    return OUT / "readings" / tag / task


def check():
    keys = [f"{s}/prediction" for s in ZGZ]
    rows = {"K1_router_e1_equals_pfo2_b2_e1": _same(_states(readings(f"b2_s{SEED}", "e1")), _states(b2_reading("e1")), keys),
            "K2_label_part_e1_equals_b2ub": _same(_states(readings(f"b2_s{SEED}__label_part", "e1")),
                                                  _states(b2ub_reading("label_part", "e1")), keys)}
    flat = [v for r in rows.values() for v in r.values()]
    receipt = {"check": "k12", "rows": rows, "status": "PASS" if flat and all(flat) else "FAIL", "checked_at_utc": now()}
    save(OUT / "receipts" / "k12.json", receipt, exclusive=True)
    print(receipt["status"], json.dumps(rows), flush=True)
    require(receipt["status"] == "PASS", "engine check failed")


# --------------------------------------------------------------------------- CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("b2")
    s.add_argument("--task", required=True, choices=TASKS)
    s.add_argument("--route", required=True, choices=("router", "label_part"))
    s = sub.add_parser("read")
    s.add_argument("--arm", required=True, choices=tuple(A.ARMS))
    s.add_argument("--task", required=True, choices=TASKS)
    s.add_argument("--step", type=int, default=6000, choices=STEPS)
    s.add_argument("--deployed", action="store_true")
    s = sub.add_parser("accept")
    s.add_argument("--arm", required=True, choices=tuple(A.ARMS))
    s.add_argument("--phase", required=True, choices=tuple(GRID))
    sub.add_parser("check")
    a = p.parse_args(argv)
    if a.verb == "b2":
        check_b2()
        sp = read(b2_spec())
        require(sha(sp["checkpoint"]) == read(b2_spec()).get("checkpoint_sha256", sha(sp["checkpoint"])), "B2 spec")
        return engine(f"b2_s{SEED}", a.task, b2_spec(), a.route)
    if a.verb == "read":
        tag, sp = arm_spec(a.arm, a.step)
        route = None if a.deployed else A.eval_route(a.arm)
        require(not (a.deployed and A.eval_route(a.arm) is None), "--deployed is for the label_part arms")
        return engine(tag, a.task, sp, route)
    if a.verb == "accept":
        return accept(a.arm, a.phase)
    return check()


if __name__ == "__main__":
    main()
