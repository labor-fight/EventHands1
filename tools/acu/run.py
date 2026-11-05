#!/usr/bin/env python3
"""ACU stage-1 readings and acceptance (docs/EXPERIMENT_RECORDS.md#acu-s1-prereg-20261104), through PFO's frozen engine
(tools/pfo/engine.py, tools/pfo/nets.py, unchanged) with the output root redirected to outputs/acu/20261104.

    python -m tools.acu.run accept --arm A --phase preflight|formal
        the training is the base's procedure with the CA change only: same seed, same data stream as the base run (the
        pair-stream hashes of both ranks), the registered config, deterministic kernels, the CA trunk and hooks
    python -m tools.acu.run eval --arm A --task T [--route R] [--step 6000]
        reading of the accepted checkpoint; R = router (the evaluation config's router, r_s2d) | label (XAG labels as
        test-time identity, diagnostic); tag acu_<arm>_s3407[@step][__label]

In the reading process the CA trunk class is installed before the engine builds the model, and the CA association plus
the intrinsics record are installed right after the engine's routing (tools/pfo/nets.py `load` is wrapped in this process
only). The route mixes of tools/pfo/nets.py are re-bound to the configuration's router (as tools/b2ub/run.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.acu import arms as A  # noqa: E402
from tools.acu.common import CONFIGS, OUT, SCREEN_SEED, now, read, require, run_dir_base, save, sha  # noqa: E402

GRID = {"preflight": (40,), "formal": tuple(range(500, 6001, 500))}
TASKS = ("e1", "e3", "pert", "f1", "tl_e1")


def checkpoint(arm, step, phase="formal"):
    return A.run_dir(arm, SCREEN_SEED, phase) / f"{A.run_name(arm, SCREEN_SEED, phase)}-step={step}.ckpt"


def write_once(path, text):
    from tools.pfo2.specs import write_once_text
    return write_once_text(path, text)


def eval_config(arm):
    import yaml
    from tools.pfo2.run import _eval_cfg
    name = A.run_name(arm, SCREEN_SEED)
    cfg = _eval_cfg(A.build_config(arm, SCREEN_SEED), name)
    return write_once(CONFIGS / "eval" / f"{name}.yaml", yaml.safe_dump(cfg, sort_keys=False))


def spec(arm, step=6000):
    receipt = read(OUT / "receipts" / f"{A.run_name(arm, SCREEN_SEED)}.json")
    require(receipt["status"] == "PASS", "training not accepted")
    ck = checkpoint(arm, step)
    require(sha(ck) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since acceptance")
    name = A.run_name(arm, SCREEN_SEED) + ("" if step == 6000 else f"@{step}")
    sp = {"kind": "routed", "checkpoint": str(ck), "checkpoint_sha256": receipt["checkpoint_sha256_by_step"][str(step)],
          "config": str(eval_config(arm)), "trunk": A.ARMS[arm]["trunk"], "module": "acu_ca"}
    return name, write_once(OUT / "specs" / f"{name}.json", json.dumps(sp, indent=2) + "\n")


def install_ca(trunk):
    """CA trunk class now; CA association + intrinsics record right after the engine's routing install."""
    from semkine import acu_ca as CA
    from semkine import pfo_router as PR
    from tools.b2ub.run import deployed_router_for_mixes
    from tools.pfo import nets as N
    from model import MNISTModel
    PR.register()
    deployed_router_for_mixes()
    CA.install(trunk)
    native = N.load

    def load(*args, **kwargs):
        net = native(*args, **kwargs)
        CA.install_hooks()
        CA.install_intrinsics(MNISTModel)
        net.identity["acu"] = CA.installed()
        return net
    N.load = load


def evaluate(arm, task, route=None, step=6000):
    from tools.pfo import engine as E
    require(task in TASKS, "task")
    name, sp = spec(arm, step)
    install_ca(A.ARMS[arm]["trunk"])
    E.OUT = OUT
    return E.run(name, task, None if route in (None, "router") else route, None, (), str(sp))


# --------------------------------------------------------------------------- acceptance
def stream_rows(run_dir):
    return {r: [x["sha256"] for x in read(Path(run_dir) / f"pair_stream_rank{r}.json")[:2]] for r in (0, 1)}


def accept(arm, phase):
    require(arm in A.ARMS and phase in GRID, "arm / phase")
    name = A.run_name(arm, SCREEN_SEED, phase)
    out = OUT / "receipts" / f"{name}.json"
    require(not out.exists(), "accepted once")
    d = A.run_dir(arm, SCREEN_SEED, phase)
    launch, meta, det = read(d / "acu_launch.json"), read(d / "training_metadata.json"), read(d / "acu_det.json")
    cfg = A.build_config(arm, SCREEN_SEED, phase)
    s = A.ARMS[arm]
    base = run_dir_base(s["base"], phase)
    mine, ref = stream_rows(d), stream_rows(base)
    routing = launch["routing"]
    ok = {"steps": all(checkpoint(arm, st, phase).exists() for st in GRID[phase]),
          "seed": meta["seed"] == SCREEN_SEED and launch["pfo"]["SEED"] == SCREEN_SEED,
          "max_steps": meta["max_steps"] == cfg["TRAIN"]["MAX_STEPS"],
          "lr_schedule": meta["lr_schedule"] == "cosine",
          "devices": meta["devices"] == 2 and meta["batch_size_per_gpu"] == 512,
          "config": launch["config_sha256"] == sha(launch["config"]) and Path(launch["config"]) == A.config_path(
              arm, SCREEN_SEED, phase).resolve(),
          "deterministic": bool(det["deterministic_algorithms"]) and det["trainer_deterministic"],
          "routes": (routing.get("router") == "label") if s["train_routes"] == "label"
          else (routing.get("router") == cfg["PFO"]["ROUTER_KIND"] and routing.get("assoc") == "pfa"),
          "ca": launch["trunk"] == s["trunk"] and launch["hooks"]["route_association"] == "semkine.acu_ca.ca_association",
          "data_stream_equals_base": mine == ref}
    receipt = {"run": name, "arm": arm, "seed": SCREEN_SEED, "phase": phase, "checks": ok, "base_run": str(base),
               "status": "PASS" if all(ok.values()) else "FAIL", "accepted_at_utc": now(),
               "checkpoint_sha256_by_step": {str(st): sha(checkpoint(arm, st, phase)) for st in GRID[phase]
                                             if checkpoint(arm, st, phase).exists()}}
    save(out, receipt, exclusive=True)
    print(receipt["status"], ok, flush=True)
    require(receipt["status"] == "PASS", "acceptance failed")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    a = sub.add_parser("accept")
    a.add_argument("--arm", required=True)
    a.add_argument("--phase", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--arm", required=True)
    e.add_argument("--task", required=True)
    e.add_argument("--route", default=None)
    e.add_argument("--step", type=int, default=6000)
    x = p.parse_args(argv)
    if x.verb == "accept":
        accept(x.arm, x.phase)
    else:
        evaluate(x.arm, x.task, x.route, x.step)


if __name__ == "__main__":
    main()
