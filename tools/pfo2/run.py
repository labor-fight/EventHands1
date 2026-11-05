#!/usr/bin/env python3
"""PFO2 acceptance and readings (docs/EXPERIMENT_RECORDS.md#pfo2-prereg-20261103) through PFO's frozen engine
(tools/pfo/engine.py, unchanged) with its output root redirected to outputs/pfo2/20261103 (readings/<tag>/<task>/).

    python -m tools.pfo2.run accept --arm A --seed S
    python -m tools.pfo2.run eval --arm A --seed S --task T [--step 6000]
        a PFO2 arm (tools/pfo2/arms.py); tag pfo2_<arm>_s<seed>[@step]
    python -m tools.pfo2.run evalpfo --arm A --seed S --task T --step N
        a PFO run (outputs/pfo/20261103/runs, read-only; e.g. the terminated f1self / f1noise runs at their last
        checkpoints); tag pfo_<arm>_s<seed>[@step]; the checkpoint SHA is recorded, no PFO acceptance is required
Evaluation configs keep the protocol's segment-start noise and point the routing block at the deployed router
(tools/pfo/run4.py `eval_config`'s content), written once under configs/pfo2_20261103/eval*/.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

import yaml  # noqa: E402

from tools.pfo import arms3 as A3  # noqa: E402
from tools.pfo.run4 import PROTOCOL_INIT_NOISE  # noqa: E402
from tools.pfo2 import arms as A  # noqa: E402
from tools.pfo2.common import CONFIGS, OUT, now, pfo_run, read, require, save, sha  # noqa: E402

STEPS = (5000, 5500, 6000)


def _eval_cfg(cfg, name):
    cfg["TRACK"].update(PROTOCOL_INIT_NOISE)
    p = cfg["PFO"]
    e = "EVAL_" if "EVAL_ROUTER" in p else ""                       # a training-time router is never deployed
    cfg["S45SM"] = {"ARM": name, "ROUTER": p[e + "ROUTER"], "ROUTER_KIND": p[e + "ROUTER_KIND"],
                    "ROUTER_CHECKPOINT": p[e + "ROUTER_CHECKPOINT"], "ASSOC": p["ASSOC"], "ONHAND_TAU": p["ONHAND_TAU"],
                    "FAST": True}
    return cfg


def _write_once(path, cfg):
    text = yaml.safe_dump(cfg, sort_keys=False)
    if path.exists():
        require(path.read_text() == text, "eval config differs: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return path


def eval_config(arm, seed):
    name = A.run_name(arm, seed)
    return _write_once(CONFIGS / "eval" / f"{name}.yaml", _eval_cfg(A.build_config(arm, seed), name))


def pfo_eval_config(arm, seed):
    """PFO's run4 eval config of a PFO arm, as PFO would have written it (configs/pfo_20261103/eval/ is not touched)."""
    name = f"pfo_{arm}_s{seed}"
    return _write_once(CONFIGS / "eval_pfo" / f"{name}.yaml", _eval_cfg(A3.build_config(arm, seed), name))


def checkpoint(arm, seed, step):
    return A.run_dir(arm, seed) / f"{A.run_name(arm, seed)}-step={step}.ckpt"


def accept(arm, seed):
    out = OUT / "receipts" / f"{A.run_name(arm, seed)}.json"
    require(not out.exists(), "accepted once")
    d = A.run_dir(arm, seed)
    launch = read(d / "pfo2_launch.json")
    meta = read(d / "training_metadata.json")
    s = A.spec(arm)
    ok = {"steps": all(checkpoint(arm, seed, st).exists() for st in STEPS),
          "seed": meta["seed"] == seed and launch["pfo"]["SEED"] == seed,
          "max_steps": meta["max_steps"] == 6000, "lr_schedule": meta["lr_schedule"] == "cosine",
          "trunk": launch["pfo"]["TRUNK"] == A.recipe(arm)["trunk"], "module": launch["module"] == s.get("module", "trunk4"),
          "devices": meta["devices"] == 2, "config": launch["config_sha256"] == sha(launch["config"]),
          "bank": (launch.get("bank") or None) == (A.recipe(arm).get("bank") or None)}
    if s.get("reference"):
        for label in (f"{arm}_preflight", f"{arm}_s{seed}_head500"):
            rc = OUT / "receipts" / f"cmp_{label}.json"
            ok["bitwise_" + label] = rc.exists() and read(rc)["status"] == "PASS"
    receipt = {"run": A.run_name(arm, seed), "arm": arm, "seed": seed, "checks": ok,
               "status": "PASS" if all(ok.values()) else "FAIL", "accepted_at_utc": now(),
               "checkpoint_sha256_by_step": {str(st): sha(checkpoint(arm, seed, st)) for st in STEPS
                                             if checkpoint(arm, seed, st).exists()}}
    save(out, receipt, exclusive=True)
    print(receipt["status"], ok, flush=True)
    require(receipt["status"] == "PASS", "acceptance failed")


def _spec(tag, sp):
    path = OUT / "specs" / f"{tag}.json"
    if path.exists():
        require(read(path) == sp, "spec differs: " + str(path))
    else:
        save(path, sp)
    return path


def arm_spec(arm, seed, step):
    receipt = read(OUT / "receipts" / f"{A.run_name(arm, seed)}.json")
    require(receipt["status"] == "PASS", "training not accepted")
    ck = checkpoint(arm, seed, step)
    require(sha(ck) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since acceptance")
    tag = A.run_name(arm, seed) + ("" if step == 6000 else f"@{step}")
    return tag, _spec(tag, {"kind": "routed", "checkpoint": str(ck), "config": str(eval_config(arm, seed)),
                            "trunk": A.recipe(arm)["trunk"], "module": A.spec(arm).get("module", "trunk4")})


def pfo_spec(arm, seed, step):
    d = pfo_run(arm, seed)
    ck = d / f"{d.name}-step={step}.ckpt"
    require(ck.exists(), "missing PFO checkpoint " + str(ck))
    tag = d.name + ("" if step == 6000 else f"@{step}")
    return tag, _spec(tag, {"kind": "routed", "checkpoint": str(ck), "checkpoint_sha256": sha(ck),
                            "config": str(pfo_eval_config(arm, seed)), "trunk": A3.spec(arm)["trunk"],
                            "module": "trunk4"})


def prepare(sp):
    """Register the router kind and install the trunk the spec names (one network per process)."""
    from tools.pfo2.train import install_trunk
    s = read(sp)
    cfg = yaml.safe_load(Path(s["config"]).read_text())
    if cfg["S45SM"]["ROUTER_KIND"] == "pfr":
        from semkine import pfo_router as PR
        PR.register()
    install_trunk(s["module"], s["trunk"])
    return s


def engine_run(tag, task, sp, route=None):
    from tools.pfo import engine as E
    E.OUT = OUT                                                     # readings go under outputs/pfo2/20261103
    return E.run(tag, task, route, None, (), str(sp))


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="verb", required=True)
    for verb in ("accept", "eval", "evalpfo"):
        s = sub.add_parser(verb)
        s.add_argument("--arm", required=True)
        s.add_argument("--seed", required=True, type=int)
        if verb != "accept":
            s.add_argument("--task", required=True)
            s.add_argument("--step", type=int, default=6000)
    a = p.parse_args(argv)
    if a.verb == "accept":
        return accept(a.arm, a.seed)
    if a.verb == "eval":
        require(a.step in STEPS, "steps 5000 / 5500 / 6000")
        tag, sp = arm_spec(a.arm, a.seed, a.step)
    else:
        tag, sp = pfo_spec(a.arm, a.seed, a.step)
    prepare(sp)
    engine_run(tag, a.task, sp)


if __name__ == "__main__":
    main()
