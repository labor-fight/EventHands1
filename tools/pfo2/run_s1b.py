#!/usr/bin/env python3
"""PFO2 arm (b) acceptance and readings (docs/EXPERIMENT_RECORDS.md#pfo2-s1b-prereg-20261103): tools/pfo2/run_s1.py for
the arms of tools/pfo2/reg1b.py (trunk semkine/pfo2_trunk_b.py).

    python -m tools.pfo2.run_s1b accept --arm A --seed S
    python -m tools.pfo2.run_s1b eval --arm A --seed S --task T [--step 6000]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

import yaml  # noqa: E402

from tools.pfo2 import arms as A  # noqa: E402
from tools.pfo2 import reg1b as G  # noqa: E402
from tools.pfo2 import run as R  # noqa: E402
from tools.pfo2 import specs as SP  # noqa: E402
from tools.pfo2.common import CONFIGS, OUT, now, read, require, save, sha  # noqa: E402

STEPS = (5000, 5500, 6000)


def checkpoint(arm, seed, step):
    return A.run_dir(arm, seed) / f"{A.run_name(arm, seed)}-step={step}.ckpt"


def accept(arm, seed):
    out = OUT / "receipts" / f"{A.run_name(arm, seed)}.json"
    require(not out.exists(), "accepted once")
    d = A.run_dir(arm, seed)
    launch = read(d / "pfo2_launch.json")
    meta = read(d / "training_metadata.json")
    det = read(d / "pfo2_det.json")
    s = G.spec(arm)
    ok = {"steps": all(checkpoint(arm, seed, st).exists() for st in STEPS),
          "seed": meta["seed"] == seed and launch["pfo"]["SEED"] == seed,
          "max_steps": meta["max_steps"] == 6000, "lr_schedule": meta["lr_schedule"] == "cosine",
          "trunk": launch["trunk"] == s["trunk"] and launch["module"] == "pfo2", "devices": meta["devices"] == 2,
          "config": launch["config_sha256"] == sha(launch["config"]),
          "losses": launch["s1"]["LOSSES"] == dict(s.get("losses", {})),
          "dual": launch["s1"]["DUAL"] == dict(s.get("dual", {})),
          "deterministic": det["trainer_deterministic"] is True and det["deterministic_algorithms"] is True}
    receipt = {"run": A.run_name(arm, seed), "arm": arm, "seed": seed, "checks": ok,
               "status": "PASS" if all(ok.values()) else "FAIL", "accepted_at_utc": now(),
               "checkpoint_sha256_by_step": {str(st): sha(checkpoint(arm, seed, st)) for st in STEPS
                                             if checkpoint(arm, seed, st).exists()}}
    save(out, receipt, exclusive=True)
    print(receipt["status"], ok, flush=True)
    require(receipt["status"] == "PASS", "acceptance failed")


def eval_config(arm, seed):
    name = A.run_name(arm, seed)
    return SP.config(CONFIGS / "eval" / f"{name}.yaml", R._eval_cfg(G.build_config(arm, seed), name))


def arm_spec(arm, seed, step):
    receipt = read(OUT / "receipts" / f"{A.run_name(arm, seed)}.json")
    require(receipt["status"] == "PASS", "training not accepted")
    ck = checkpoint(arm, seed, step)
    require(sha(ck) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since acceptance")
    tag = A.run_name(arm, seed) + ("" if step == 6000 else f"@{step}")
    return tag, SP.spec(tag, {"kind": "routed", "checkpoint": str(ck), "config": str(eval_config(arm, seed)),
                              "trunk": G.spec(arm)["trunk"], "module": "pfo2"})


def prepare(sp):
    from tools.pfo2.train_s1b import install_arm
    s = read(sp)
    cfg = yaml.safe_load(Path(s["config"]).read_text())
    if cfg["S45SM"]["ROUTER_KIND"] == "pfr":
        from semkine import pfo_router as PR
        PR.register()
    install_arm({"TRUNK": s["trunk"]}, {})
    return s


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="verb", required=True)
    for verb in ("accept", "eval"):
        s = sub.add_parser(verb)
        s.add_argument("--arm", required=True)
        s.add_argument("--seed", required=True, type=int)
        if verb == "eval":
            s.add_argument("--task", required=True)
            s.add_argument("--step", type=int, default=6000, choices=STEPS)
    a = p.parse_args(argv)
    if a.verb == "accept":
        return accept(a.arm, a.seed)
    tag, sp = arm_spec(a.arm, a.seed, a.step)
    prepare(sp)
    R.engine_run(tag, a.task, sp)


if __name__ == "__main__":
    main()
