#!/usr/bin/env python3
"""PMR stage S3 rollout readings and acceptance (docs/EXPERIMENT_RECORDS.md#pmr-prereg-20261109), through PFO's frozen
engine with the output root redirected to outputs/pmr/20261109 (tools/pmr/run.py's procedure for the rollout arms).

    python -m tools.pmr.run_rol read --arm A --seed S --task T
        a rollout arm's reading at its accepted step-1500 checkpoint, deployed router r_s2d, the ordinary packet form;
        tag pmr_<arm>_s<seed>
    python -m tools.pmr.run_rol accept --arm A --seed S --phase preflight|formal
        training acceptance (TPD's rol5 checks with the PMR trunk: start check bitwise, warm start = the base arm's
        accepted step-6000 checkpoint, rollout stream recorded, peak memory); receipts/<run>.json; exit 1 on failure
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms as PA  # noqa: E402
from tools.pmr import arms_rol as A  # noqa: E402
from tools.pmr.common import CONFIGS, OUT, now, read, require, save, sha  # noqa: E402

TASKS = ("e1", "e3", "pert", "f1", "tl_e1")
GRID = {"preflight": (40,), "formal": (500, 1000, 1500)}
MEM_MAX_GIB = 42.0


def write_once(path, obj):
    from tools.pfo2.specs import write_once_text
    if str(path).endswith(".json"):
        return write_once_text(path, json.dumps(obj, indent=2, allow_nan=False) + "\n")
    import yaml
    return write_once_text(path, yaml.safe_dump(obj, sort_keys=False))


def eval_config(name, seed):
    from tools.pfo2.run import _eval_cfg
    rn = A.run_name(name, seed)
    cfg = A.build_config(name, seed)
    cfg["MODEL"].pop("INIT_FROM", None)                              # evaluation loads the fine-tune itself
    cfg.pop("ROL", None)
    return write_once(CONFIGS / "eval" / f"{rn}.yaml", _eval_cfg(cfg, rn))


def checkpoint(name, seed, step, phase="formal"):
    return A.run_dir(name, seed, phase) / f"{A.run_name(name, seed, phase)}-step={step}.ckpt"


def arm_spec(name, seed, step=1500):
    receipt = read(OUT / "receipts" / f"{A.run_name(name, seed)}.json")
    require(receipt["status"] == "PASS", "training not accepted: " + name)
    ck = checkpoint(name, seed, step)
    require(sha(ck) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since acceptance")
    base, _ = A.parse(name)
    design, variant = PA.parse(base)
    tag = A.run_name(name, seed) + ("" if step == 1500 else f"@{step}")
    sp = {"kind": "routed", "checkpoint": str(ck), "checkpoint_sha256": receipt["checkpoint_sha256_by_step"][str(step)],
          "config": str(eval_config(name, seed)), "arm": name, "design": design, "variant": variant, "messages": "on"}
    return tag, write_once(OUT / "specs" / f"{tag}.json", sp)


def engine(tag, task, sp):
    from tools.pfo import engine as E
    from tools.pmr import run as R
    R.prepare(sp)
    R.deployed_router_for_mixes()
    E.OUT = OUT
    return E.run(tag, task, None, None, (), str(sp))


def accept(name, seed, phase):
    base, rol = A.parse(name)
    require(phase in GRID, "phase")
    rn = A.run_name(name, seed, phase)
    out = OUT / "receipts" / f"{rn}.json"
    require(not out.exists(), "accepted once")
    d = A.run_dir(name, seed, phase)
    launch, meta, det = read(d / "pmr_launch.json"), read(d / "training_metadata.json"), read(d / "pmr_det.json")
    cfg = A.build_config(name, seed, phase)
    starts = [read(d / f"rol_start_check_rank{r}.json") for r in (0, 1)]
    timing = [read(d / f"rol_timing_rank{r}.json") for r in (0, 1)]
    streams = [read(d / f"rollout_stream_rank{r}.json") for r in (0, 1)]
    s = A.ROL[rol]
    design, variant = PA.parse(base)
    ck, ck_sha = A.init_checkpoint(base, seed), A.init_sha(base, seed)
    ok = {"steps": all(checkpoint(name, seed, st, phase).exists() for st in GRID[phase]),
          "seed": meta["seed"] == seed and launch["pfo"]["SEED"] == seed,
          "max_steps": meta["max_steps"] == cfg["TRAIN"]["MAX_STEPS"],
          "lr": meta["lr"] == cfg["TRAIN"]["LR"] and meta["lr_schedule"] == "cosine",
          "devices": meta["devices"] == 2 and meta["batch_size_per_gpu"] == 512,
          "config": launch["config_sha256"] == sha(launch["config"]) and Path(launch["config"]) == A.config_path(
              name, seed, phase).resolve(),
          "deterministic": bool(det["deterministic_algorithms"]) and det["trainer_deterministic"],
          "router": launch["routing"]["router"] == "pfr" and launch["routing"]["assoc"] == "pfa"
          and float(launch["routing"]["onhand_tau"]) == 0.5
          and launch["router_checkpoint_sha256"] == sha(cfg["PFO"]["ROUTER_CHECKPOINT"]),
          "trunk": launch["trunk"] == {"design": design, "variant": variant, "module": PA.DESIGNS[design][0]},
          "rollout": launch["rol"] == cfg["ROL"] and launch["rol"]["K"] == s["k"] and launch["rol"]["GRAD"] == s["grad"],
          "init_from_base": launch["init_from"] == ck and launch["init_from_sha256"] == ck_sha,
          "start_check": all(x["delta_bitwise_equal"] and x["init_equal"] and x["autocast"] and x["global_step"] == 0
                             for x in starts),
          "stream_recorded": all(len(x) == 2 and all(r["packets"] == s["k"] for r in x) for x in streams),
          "memory": max(x["max_reserved_gib"] for x in timing) <= MEM_MAX_GIB}
    receipt = {"run": rn, "arm": name, "seed": seed, "phase": phase, "checks": ok, "start_checks": starts,
               "peak_reserved_gib": [x["max_reserved_gib"] for x in timing],
               "status": "PASS" if all(ok.values()) else "FAIL", "accepted_at_utc": now(),
               "checkpoint_sha256_by_step": {str(st): sha(checkpoint(name, seed, st, phase)) for st in GRID[phase]
                                             if checkpoint(name, seed, st, phase).exists()}}
    save(out, receipt, exclusive=True)
    print(receipt["status"], ok, flush=True)
    require(receipt["status"] == "PASS", "acceptance failed")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("read")
    s.add_argument("--arm", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--task", required=True, choices=TASKS)
    s = sub.add_parser("accept")
    s.add_argument("--arm", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--phase", required=True, choices=tuple(GRID))
    a = p.parse_args(argv)
    if a.verb == "read":
        tag, sp = arm_spec(a.arm, a.seed)
        return engine(tag, a.task, sp)
    return accept(a.arm, a.seed, a.phase)


if __name__ == "__main__":
    main()
