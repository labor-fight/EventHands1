#!/usr/bin/env python3
"""TPD synthesis / final-configuration readings and acceptance (docs/EXPERIMENT_RECORDS.md#tpd-syn-prereg-20261105),
through PFO's frozen engine with the output root redirected to outputs/tpd/20261105. An arm trained in the causal form is
read in the causal form (semkine/tpd_causal.py engine hooks); the deployed router is B2's r_s2d.

    python -m tools.tpd.run_final read --arm A --seed S --task T      tag tpd_<arm>_s<seed>
    python -m tools.tpd.run_final accept --arm A --seed S --phase P   receipts/<run>.json; exit 1 on failure
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd import arms_final as A  # noqa: E402
from tools.tpd.common import CONFIGS, OUT, now, read, require, save, sha  # noqa: E402

TASKS = ("e1", "e3", "pert", "f1")
GRID = {"preflight": (40,), "formal": (5000, 5500, 6000)}


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


def arm_spec(arm, seed, step=6000):
    receipt = read(OUT / "receipts" / f"{A.run_name(arm, seed)}.json")
    require(receipt["status"] == "PASS", "training not accepted: " + arm)
    ck = checkpoint(arm, seed, step)
    require(sha(ck) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since acceptance")
    tag = A.run_name(arm, seed) + ("" if step == 6000 else f"@{step}")
    sp = {"kind": "routed", "checkpoint": str(ck), "checkpoint_sha256": receipt["checkpoint_sha256_by_step"][str(step)],
          "config": str(eval_config(arm, seed)), "trunk": "gshare", "module": "trunk4", "arm": arm, "seed": seed,
          "causal": bool(A.ARMS[arm]["causal"])}
    return tag, write_once(OUT / "specs" / f"{tag}.json", sp)


def reading(arm, seed, task):
    from semkine import tpd_causal as TC
    from tools.pfo import engine as E
    from tools.tpd import run_rol as RR
    tag, sp = arm_spec(arm, seed)
    RR.prepare(sp)
    if A.ARMS[arm]["causal"]:
        TC.install()
        TC.engine_hooks()
    E.OUT = OUT
    out = E.run(tag, task, None, None, (), str(sp))
    save(Path(out) / "final_form.json", {"causal": bool(A.ARMS[arm]["causal"]), "written_utc": now()}, exclusive=True)


def accept(arm, seed, phase):
    name = A.run_name(arm, seed, phase)
    out = OUT / "receipts" / f"{name}.json"
    require(not out.exists(), "accepted once")
    d = A.run_dir(arm, seed, phase)
    launch, meta, det = read(d / "tpd_launch.json"), read(d / "training_metadata.json"), read(d / "tpd_det.json")
    cfg = A.build_config(arm, seed, phase)
    s = A.ARMS[arm]
    timing = [read(d / f"rol_timing_rank{r}.json") for r in (0, 1)]
    starts = [read(d / f"final_start_check_rank{r}.json") for r in (0, 1)] if s["k"] > 1 else []
    expect_src = "tools.pfo.nets.mixed_sources" if s["routes"] == "label_part" else "semkine.s45sm_fast.predicted_sources"
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
          "routes": launch["routing"]["train_routes"] == s["routes"] and launch["routing"]["sample_sources"] == expect_src
          and (launch["xag_source_root"] is not None) == (s["routes"] == "label_part"),
          "final_block": launch["final"] == cfg["FINAL"] and launch["rol"] == cfg["ROL"],
          "trunk": launch["trunk"] == "gshare" and launch["geometry"] == "edge_geometry_grad",
          "start_check": all(x["causal"] == bool(s["causal"]) and x["global_step"] == 0 for x in starts),
          "memory": max(x["max_reserved_gib"] for x in timing) <= 42.0}
    receipt = {"run": name, "arm": arm, "seed": seed, "phase": phase, "checks": ok, "start_checks": starts,
               "status": "PASS" if all(ok.values()) else "FAIL", "accepted_at_utc": now(),
               "checkpoint_sha256_by_step": {str(st): sha(checkpoint(arm, seed, st, phase)) for st in GRID[phase]
                                             if checkpoint(arm, seed, st, phase).exists()}}
    save(out, receipt, exclusive=True)
    print(receipt["status"], ok, flush=True)
    require(receipt["status"] == "PASS", "acceptance failed")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("read")
    s.add_argument("--arm", required=True, choices=tuple(A.ARMS))
    s.add_argument("--seed", type=int, required=True, choices=A.SEEDS)
    s.add_argument("--task", required=True, choices=TASKS)
    s = sub.add_parser("accept")
    s.add_argument("--arm", required=True, choices=tuple(A.ARMS))
    s.add_argument("--seed", type=int, required=True, choices=A.SEEDS)
    s.add_argument("--phase", required=True, choices=tuple(GRID))
    a = p.parse_args(argv)
    if a.verb == "read":
        return reading(a.arm, a.seed, a.task)
    return accept(a.arm, a.seed, a.phase)


if __name__ == "__main__":
    main()
