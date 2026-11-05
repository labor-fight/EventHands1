#!/usr/bin/env python3
"""PFO phase-2b entry (docs/EXPERIMENT_RECORDS.md#pfo-p2b-prereg-20261103): acceptance, readings and costs of the
phase-2b arms (tools/pfo/arms2.py) through the frozen phase-1 engine and cost harness; tools/pfo/run2.py's procedure
with the arm's trunk module (semkine/pfo_trunk.py or semkine/pfo_trunk2.py) and router (sr1 or a PFO router).

    python -m tools.pfo.run4 accept --arm A --seed S
    python -m tools.pfo.run4 eval --arm A --seed S --task T [--step 6000]
    python -m tools.pfo.run4 cost --arm A --seed S [--policy det|default]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

import yaml  # noqa: E402

from tools.pfo.arms import run_dir, run_name  # noqa: E402
from tools.pfo.arms2 import ARMS2, TRUNK2, build_config, spec  # noqa: E402
from tools.pfo.common import CONFIGS, OUT, now, read, require, save, sha  # noqa: E402

STEPS = (5000, 5500, 6000)


def checkpoint(arm, seed, step):
    return run_dir(arm, seed) / f"{run_name(arm, seed)}-step={step}.ckpt"


#: the protocol's segment-start noise (semkine/eval_track.py `sample_init_noise` reads these keys): every evaluation config
#: keeps them, so a training-noise arm never changes the evaluated initial states
PROTOCOL_INIT_NOISE = {"PREV_NOISE_T": 0.005, "PREV_NOISE_R": 0.05, "PREV_NOISE_POSE": 0.05}


def eval_config(arm, seed):
    cfg = build_config(arm, seed)
    cfg["TRACK"].update(PROTOCOL_INIT_NOISE)
    p = cfg["PFO"]
    e = "EVAL_" if "EVAL_ROUTER" in p else ""                       # a training-time router is never deployed
    cfg["S45SM"] = {"ARM": run_name(arm, seed), "ROUTER": p[e + "ROUTER"], "ROUTER_KIND": p[e + "ROUTER_KIND"],
                    "ROUTER_CHECKPOINT": p[e + "ROUTER_CHECKPOINT"], "ASSOC": p["ASSOC"], "ONHAND_TAU": p["ONHAND_TAU"],
                    "FAST": True}
    path = CONFIGS / "eval" / f"{run_name(arm, seed)}.yaml"
    text = yaml.safe_dump(cfg, sort_keys=False)
    if path.exists():
        require(path.read_text() == text, "eval config differs: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return path


def spec_file(arm, seed, step):
    receipt = read(OUT / "receipts" / f"{run_name(arm, seed)}.json")
    require(receipt["status"] == "PASS", "training not accepted")
    ck = checkpoint(arm, seed, step)
    require(sha(ck) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since acceptance")
    sp = {"kind": "routed", "checkpoint": str(ck), "config": str(eval_config(arm, seed)), "trunk": spec(arm)["trunk"]}
    path = OUT / "specs" / f"{run_name(arm, seed)}@{step}.json"
    if path.exists():
        require(read(path) == sp, "spec differs")
    else:
        save(path, sp)
    return path


def accept(arm, seed):
    out = OUT / "receipts" / f"{run_name(arm, seed)}.json"
    require(not out.exists(), "accepted once")
    d = run_dir(arm, seed)
    launch = read(d / "pfo_launch.json")
    meta = read(d / "training_metadata.json")
    ok = {"steps": all(checkpoint(arm, seed, s).exists() for s in STEPS),
          "seed": meta["seed"] == seed and launch["pfo"]["SEED"] == seed,
          "max_steps": meta["max_steps"] == 6000, "lr_schedule": meta["lr_schedule"] == "cosine",
          "trunk": launch["trunk"] == spec(arm)["trunk"], "devices": meta["devices"] == 2,
          "config": launch["config_sha256"] == sha(launch["config"]),
          "bank": (launch.get("bank") or None) == (spec(arm).get("bank") or None)}
    receipt = {"run": run_name(arm, seed), "arm": arm, "seed": seed, "checks": ok,
               "status": "PASS" if all(ok.values()) else "FAIL", "accepted_at_utc": now(),
               "checkpoint_sha256_by_step": {str(s): sha(checkpoint(arm, seed, s)) for s in STEPS
                                             if checkpoint(arm, seed, s).exists()}}
    save(out, receipt, exclusive=True)
    print(receipt["status"], ok, flush=True)
    require(receipt["status"] == "PASS", "acceptance failed")


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="verb", required=True)
    for verb in ("accept", "eval", "cost"):
        s = sub.add_parser(verb)
        s.add_argument("--arm", required=True, choices=tuple(ARMS2))
        s.add_argument("--seed", required=True, type=int)
        if verb == "eval":
            s.add_argument("--task", required=True)
            s.add_argument("--step", type=int, default=6000, choices=STEPS)
        if verb == "cost":
            s.add_argument("--policy", default="det", choices=("det", "default"))
    a = p.parse_args(argv)
    if a.verb == "accept":
        return accept(a.arm, a.seed)
    sp = spec_file(a.arm, a.seed, 6000 if a.verb == "cost" else a.step)
    s = spec(a.arm)
    if s.get("router") and not s.get("train_router"):
        from semkine import pfo_router as PR
        PR.register()
    if s["trunk"] in TRUNK2:
        from semkine import pfo_trunk2 as PT
    else:
        from semkine import pfo_trunk as PT
    PT.install(s["trunk"])
    name = run_name(a.arm, a.seed) + ("" if a.verb == "cost" or a.step == 6000 else f"@{a.step}")
    if a.verb == "eval":
        from tools.pfo.engine import run
        run(name, a.task, None, None, (), str(sp))
    else:
        from tools.pfo.cost import run
        run(name, "fast", str(sp), None, a.policy)


if __name__ == "__main__":
    main()
