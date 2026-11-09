#!/usr/bin/env python3
"""TPD stage ROL readings, acceptance and checks (docs/EXPERIMENT_RECORDS.md#tpd-rol-prereg-20261105), through PFO's
frozen engine (tools/pfo/engine.py, tools/pfo/nets.py, unchanged) with the output root redirected to outputs/tpd/20261105.

    python -m tools.tpd.run_rol b2 --seed S --task T
        B2 (PFO2 fgs, step 6000, the accepted checkpoint) read through this study's entry; tag b2_s<seed>
    python -m tools.tpd.run_rol read --arm A --seed S --task T
        an arm's reading at its accepted step-1500 checkpoint with B2's deployed router r_s2d; tag tpd_<arm>_s<seed>
    python -m tools.tpd.run_rol accept --arm A --seed S --phase preflight|formal
        training acceptance; receipts/<run>.json; exit 1 on failure
    python -m tools.tpd.run_rol check
        K1: B2 3407's e1 through this entry equals PFO2's published e1 bit for bit; receipts/k1.json
    python -m tools.tpd.run_rol pair
        rol3 and rol3d (seed 3407) trained on the same rollout stream (first two batches, all packets, both ranks)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.tpd import arms_rol as A  # noqa: E402
from tools.tpd.common import (CONFIGS, OUT, PFO2_OUT, ZGZ, b2_checkpoint, b2_spec, b2_tag, now, read, require, save,  # noqa: E402
                              sha)

TASKS = ("e1", "e3", "pert", "f1", "tl_e1")
GRID = {"preflight": (40,), "formal": (500, 1000, 1500)}
MEM_MAX_GIB = 42.0


def write_once(path, obj):
    from tools.pfo2.specs import write_once_text
    if str(path).endswith(".json"):
        return write_once_text(path, json.dumps(obj, indent=2, allow_nan=False) + "\n")
    import yaml
    return write_once_text(path, yaml.safe_dump(obj, sort_keys=False))


def eval_config(arm, seed):
    from tools.pfo2.run import _eval_cfg
    name = A.run_name(arm, seed)
    return write_once(CONFIGS / "eval" / f"{name}.yaml", _eval_cfg(A.build_config(arm, seed), name))


def checkpoint(arm, seed, step, phase="formal"):
    return A.run_dir(arm, seed, phase) / f"{A.run_name(arm, seed, phase)}-step={step}.ckpt"


def arm_spec(arm, seed, step=1500):
    receipt = read(OUT / "receipts" / f"{A.run_name(arm, seed)}.json")
    require(receipt["status"] == "PASS", "training not accepted: " + arm)
    ck = checkpoint(arm, seed, step)
    require(sha(ck) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since acceptance")
    tag = A.run_name(arm, seed) + ("" if step == 1500 else f"@{step}")
    sp = {"kind": "routed", "checkpoint": str(ck), "checkpoint_sha256": receipt["checkpoint_sha256_by_step"][str(step)],
          "config": str(eval_config(arm, seed)), "trunk": "gshare", "module": "trunk4", "arm": arm, "seed": seed}
    return tag, write_once(OUT / "specs" / f"{tag}.json", sp)


def prepare(sp):
    """Register the router kind and install the trunk the spec names (one network per process)."""
    import yaml
    s = read(sp)
    cfg = yaml.safe_load(Path(s["config"]).read_text())
    if cfg["S45SM"]["ROUTER_KIND"] == "pfr":
        from semkine import pfo_router as PR
        PR.register()
    from tools.pfo2.train import install_trunk
    install_trunk(s["module"], s["trunk"])
    return s


def engine(tag, task, sp):
    from tools.pfo import engine as E
    prepare(sp)
    E.OUT = OUT                                                     # readings go under outputs/tpd/20261105
    return E.run(tag, task, None, None, (), str(sp))


# --------------------------------------------------------------------------- acceptance
def accept(arm, seed, phase):
    require(arm in A.ARMS and phase in GRID, "arm / phase")
    name = A.run_name(arm, seed, phase)
    out = OUT / "receipts" / f"{name}.json"
    require(not out.exists(), "accepted once")
    d = A.run_dir(arm, seed, phase)
    launch, meta, det = read(d / "tpd_launch.json"), read(d / "training_metadata.json"), read(d / "tpd_det.json")
    cfg = A.build_config(arm, seed, phase)
    starts = [read(d / f"rol_start_check_rank{r}.json") for r in (0, 1)]
    timing = [read(d / f"rol_timing_rank{r}.json") for r in (0, 1)]
    streams = [read(d / f"rollout_stream_rank{r}.json") for r in (0, 1)]
    s = A.ARMS[arm]
    ok = {"steps": all(checkpoint(arm, seed, st, phase).exists() for st in GRID[phase]),
          "seed": meta["seed"] == seed and launch["pfo"]["SEED"] == seed,
          "max_steps": meta["max_steps"] == cfg["TRAIN"]["MAX_STEPS"],
          "lr": meta["lr"] == cfg["TRAIN"]["LR"] and meta["lr_schedule"] == "cosine",
          "devices": meta["devices"] == 2 and meta["batch_size_per_gpu"] == 512,
          "config": launch["config_sha256"] == sha(launch["config"]) and Path(launch["config"]) == A.config_path(
              arm, seed, phase).resolve(),
          "deterministic": bool(det["deterministic_algorithms"]) and det["trainer_deterministic"],
          "router": launch["routing"]["router"] == "pfr" and launch["routing"]["assoc"] == "pfa"
          and float(launch["routing"]["onhand_tau"]) == 0.5
          and launch["router_checkpoint_sha256"] == sha(cfg["PFO"]["ROUTER_CHECKPOINT"]),
          "trunk": launch["trunk"] == "gshare" and launch["geometry"] == "edge_geometry_grad",
          "rollout": launch["rol"] == cfg["ROL"] and launch["rol"]["K"] == s["k"] and launch["rol"]["GRAD"] == s["grad"],
          "init_from_b2": launch["init_from"] == str(b2_checkpoint(seed)) and launch["init_from_sha256"] == sha(
              b2_checkpoint(seed)),
          "start_check": all(x["delta_bitwise_equal"] and x["init_equal"] and x["autocast"] and x["global_step"] == 0
                             for x in starts),
          "stream_recorded": all(len(x) == 2 and all(r["packets"] == s["k"] for r in x) for x in streams),
          "memory": max(x["max_reserved_gib"] for x in timing) <= MEM_MAX_GIB}
    receipt = {"run": name, "arm": arm, "seed": seed, "phase": phase, "checks": ok, "start_checks": starts,
               "peak_reserved_gib": [x["max_reserved_gib"] for x in timing],
               "status": "PASS" if all(ok.values()) else "FAIL", "accepted_at_utc": now(),
               "checkpoint_sha256_by_step": {str(st): sha(checkpoint(arm, seed, st, phase)) for st in GRID[phase]
                                             if checkpoint(arm, seed, st, phase).exists()}}
    save(out, receipt, exclusive=True)
    print(receipt["status"], ok, flush=True)
    require(receipt["status"] == "PASS", "acceptance failed")


def pair():
    rows = {r: (read(A.run_dir("rol3", 3407) / f"rollout_stream_rank{r}.json"),
                read(A.run_dir("rol3d", 3407) / f"rollout_stream_rank{r}.json")) for r in (0, 1)}
    same = {str(r): [x["sha256"] for x in a] == [x["sha256"] for x in b] for r, (a, b) in rows.items()}
    receipt = {"check": "rol3_rol3d_stream", "rows": same, "status": "PASS" if all(same.values()) else "FAIL",
               "checked_at_utc": now()}
    save(OUT / "receipts" / "pair_rol3_rol3d.json", receipt, exclusive=True)
    print(receipt["status"], same, flush=True)
    require(receipt["status"] == "PASS", "rol3 / rol3d streams differ")


def check():
    a = np.load(OUT / "readings" / "b2_s3407" / "e1" / "states.npz", allow_pickle=False)
    b = np.load(PFO2_OUT / "readings" / b2_tag(3407) / "e1" / "states.npz", allow_pickle=False)
    keys = [f"{s}/prediction" for s in ZGZ]
    same = {k: bool(k in a.files and k in b.files and np.array_equal(a[k], b[k])) for k in keys}
    receipt = {"check": "k1", "rows": same, "status": "PASS" if all(same.values()) else "FAIL", "checked_at_utc": now()}
    save(OUT / "receipts" / "k1.json", receipt, exclusive=True)
    print(receipt["status"], json.dumps(same), flush=True)
    require(receipt["status"] == "PASS", "engine check failed")


# --------------------------------------------------------------------------- CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("b2")
    s.add_argument("--seed", type=int, required=True, choices=(3407, 3408))
    s.add_argument("--task", required=True, choices=TASKS)
    s = sub.add_parser("read")
    s.add_argument("--arm", required=True, choices=tuple(A.ARMS))
    s.add_argument("--seed", type=int, required=True, choices=(3407, 3408))
    s.add_argument("--task", required=True, choices=TASKS)
    s = sub.add_parser("accept")
    s.add_argument("--arm", required=True, choices=tuple(A.ARMS))
    s.add_argument("--seed", type=int, required=True, choices=(3407, 3408))
    s.add_argument("--phase", required=True, choices=tuple(GRID))
    sub.add_parser("check")
    sub.add_parser("pair")
    a = p.parse_args(argv)
    if a.verb == "b2":
        b2_checkpoint(a.seed)
        return engine(f"b2_s{a.seed}", a.task, b2_spec(a.seed))
    if a.verb == "read":
        tag, sp = arm_spec(a.arm, a.seed)
        return engine(tag, a.task, sp)
    if a.verb == "accept":
        return accept(a.arm, a.seed, a.phase)
    if a.verb == "pair":
        return pair()
    return check()


if __name__ == "__main__":
    main()
