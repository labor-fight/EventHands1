#!/usr/bin/env python3
"""PFO phase-2 entry (docs/EXPERIMENT_RECORDS.md#pfo-p2a-prereg-20261103): acceptance, readings and costs of the trained
arms through the frozen phase-1 tools (tools/pfo/engine.py, tools/pfo/cost.py, unchanged).

    python -m tools.pfo.run2 accept --arm A --seed S
    python -m tools.pfo.run2 eval --arm A --seed S --task T [--step 6000] [--route R] [--tau X] [--intervene a,b]
    python -m tools.pfo.run2 cost --arm A --seed S [--policy det|default]

A trained arm is read with its own trunk variant (semkine/pfo_trunk.py, installed before the model is built) and an
evaluation config = its training config + the routing block the frozen loader reads (S45SM: the same frozen router, hard
PFA, threshold), written once under configs/pfo_20261103/eval/. The reading is named `pfo_<arm>_s<seed>[@<step>]`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

import yaml  # noqa: E402

from tools.pfo.arms import ARMS, build_config, run_dir, run_name  # noqa: E402
from tools.pfo.common import CONFIGS, OUT, now, read, require, save, sha  # noqa: E402

STEPS = (5000, 5500, 6000)


def eval_config(arm, seed):
    cfg = build_config(arm, seed)
    p = cfg["PFO"]
    cfg["S45SM"] = {"ARM": run_name(arm, seed), "ROUTER": p["ROUTER"], "ROUTER_KIND": p["ROUTER_KIND"],
                    "ROUTER_CHECKPOINT": p["ROUTER_CHECKPOINT"], "ASSOC": p["ASSOC"], "ONHAND_TAU": p["ONHAND_TAU"],
                    "FAST": True}
    path = CONFIGS / "eval" / f"{run_name(arm, seed)}.yaml"
    text = yaml.safe_dump(cfg, sort_keys=False)
    if path.exists():
        require(path.read_text() == text, "eval config differs: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return path


def checkpoint(arm, seed, step):
    return run_dir(arm, seed) / f"{run_name(arm, seed)}-step={step}.ckpt"


def spec_file(arm, seed, step):
    receipt = read(OUT / "receipts" / f"{run_name(arm, seed)}.json")
    require(receipt["status"] == "PASS", "training not accepted")
    ck = checkpoint(arm, seed, step)
    require(sha(ck) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since acceptance")
    spec = {"kind": "routed", "checkpoint": str(ck), "config": str(eval_config(arm, seed)), "trunk": ARMS[arm]["trunk"]}
    path = OUT / "specs" / f"{run_name(arm, seed)}@{step}.json"
    if path.exists():
        require(read(path) == spec, "spec differs")
    else:
        save(path, spec)
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
          "trunk": launch["trunk"] == ARMS[arm]["trunk"], "devices": meta["devices"] == 2,
          "config": launch["config_sha256"] == sha(launch["config"])}
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
        s.add_argument("--arm", required=True, choices=tuple(ARMS))
        s.add_argument("--seed", required=True, type=int)
        if verb == "eval":
            s.add_argument("--task", required=True)
            s.add_argument("--step", type=int, default=6000, choices=STEPS)
            s.add_argument("--route")
            s.add_argument("--tau", type=float)
            s.add_argument("--intervene", default="")
        if verb == "cost":
            s.add_argument("--policy", default="det", choices=("det", "default"))
    a = p.parse_args(argv)
    if a.verb == "accept":
        return accept(a.arm, a.seed)
    from semkine import pfo_trunk as PT
    spec = spec_file(a.arm, a.seed, 6000 if a.verb == "cost" else a.step)
    PT.install(ARMS[a.arm]["trunk"])
    name = run_name(a.arm, a.seed) + ("" if a.verb == "cost" or a.step == 6000 else f"@{a.step}")
    if a.verb == "eval":
        from tools.pfo.engine import run
        run(name, a.task, a.route, a.tau, tuple(x for x in a.intervene.split(",") if x), str(spec))
    else:
        from tools.pfo.cost import run
        run(name, "fast", str(spec), None, a.policy)


if __name__ == "__main__":
    main()
