#!/usr/bin/env python3
"""PMR readings, acceptance and the engine check (docs/EXPERIMENT_RECORDS.md#pmr-prereg-20261109), through PFO's frozen
engine (tools/pfo/engine.py, tools/pfo/nets.py, unchanged) with the output root redirected to outputs/pmr/20261109.

    python -m tools.pmr.run b2 --task e1
        B2 (PFO2 fgs seed 3407, step 6000, the accepted checkpoint) read through this study's entry: engine check K1
        (against PFO2's published e1, bitwise); tag b2_s3407
    python -m tools.pmr.run read --arm A --task T [--seed S] [--step 6000] [--messages off]
        an arm's reading at its accepted checkpoint with the deployed router (r_s2d + PFA); tag pmr_<arm>_s<seed>;
        --messages off reads a P arm with every message zeroed at test time (frozen ablation, descriptive; tag
        suffix __nomsg)
    python -m tools.pmr.run accept --arm A --phase preflight|formal [--seed S]
        training acceptance against B2 (same seed and phase): the data stream (pair-stream hashes) equals B2's, the
        recipe / router / trunk identity; receipts/<run>.json; exit 1 on failure
    python -m tools.pmr.run check
        K1 (bitwise); receipts/k1.json; exit 1 on failure
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pmr import arms as A  # noqa: E402
from tools.pmr.common import (B2_TAG, CONFIGS, OUT, PFO2_OUT, SEED, ZGZ, b2_run, b2_spec, check_b2, now, read,  # noqa: E402
                              require, save, sha)

TASKS = ("e1", "e3", "pert", "f1", "tl_e1", "cap_zgz_e3", "cap_zgz_loop")
GRID = {"preflight": (40,), "formal": tuple(range(500, 6001, 500))}
STEPS = (5000, 5500, 6000)


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


def arm_spec(arm, seed, step=6000, messages="on"):
    receipt = read(OUT / "receipts" / f"{A.run_name(arm, seed)}.json")
    require(receipt["status"] == "PASS", "training not accepted: " + arm)
    ck = checkpoint(arm, seed, step)
    require(sha(ck) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since acceptance")
    design, variant = A.parse(arm)
    require(messages == "on" or variant == "r", "--messages off is for a P arm")
    tag = A.run_name(arm, seed) + ("" if step == 6000 else f"@{step}") + ("" if messages == "on" else "__nomsg")
    sp = {"kind": "routed", "checkpoint": str(ck), "checkpoint_sha256": receipt["checkpoint_sha256_by_step"][str(step)],
          "config": str(eval_config(arm, seed)), "arm": arm, "design": design, "variant": variant, "messages": messages}
    return tag, write_once(OUT / "specs" / f"{tag}.json", sp)


def prepare(sp):
    """Register the router kind and install the trunk the spec names (one network per process)."""
    import yaml
    s = read(sp)
    cfg = yaml.safe_load(Path(s["config"]).read_text())
    if cfg["S45SM"]["ROUTER_KIND"] == "pfr":
        from semkine import pfo_router as PR
        PR.register()
    if "PMR" in cfg:
        A.install_trunk(cfg)
        if s.get("messages", "on") == "off":
            from semkine import xyz_mesh as XM
            cls = XM.XYZMeshNet
            native = cls.forward
            cls.forward = lambda self, *a, **k: native(self, *a, **{**k, "messages": False})
    else:
        from tools.pfo2.train import install_trunk
        install_trunk(s.get("module", "trunk4"), s.get("trunk", "gshare"))
    return s


def deployed_router_for_mixes():
    """tools/pfo/nets.py builds route mixes on sr1; PMR reads only the deployed router (kept for parity with IDP)."""
    from tools.pfo import nets as N
    from tools.pfo.common import T0P

    def router_for(route, cfg):
        if route == "t0p":
            return "t0p", str(T0P)
        s = cfg["S45SM"]
        return s["ROUTER_KIND"], s["ROUTER_CHECKPOINT"]
    N._router_for = router_for


def engine(tag, task, sp):
    from tools.pfo import engine as E
    prepare(sp)
    deployed_router_for_mixes()
    E.OUT = OUT                                                     # readings go under outputs/pmr/20261109
    return E.run(tag, task, None, None, (), str(sp))


# --------------------------------------------------------------------------- acceptance
def stream_rows(run_dir):
    return {r: [x["sha256"] for x in read(Path(run_dir) / f"pair_stream_rank{r}.json")[:2]] for r in (0, 1)}


def accept(arm, phase, seed):
    design, variant = A.parse(arm)
    require(phase in GRID, "phase")
    name = A.run_name(arm, seed, phase)
    out = OUT / "receipts" / f"{name}.json"
    require(not out.exists(), "accepted once")
    d = A.run_dir(arm, seed, phase)
    launch, meta, det = read(d / "pmr_launch.json"), read(d / "training_metadata.json"), read(d / "pmr_det.json")
    cfg = A.build_config(arm, seed, phase)
    mine, b2 = stream_rows(d), stream_rows(b2_run(phase, seed))
    routing, trunk = launch["routing"], launch["trunk"]
    ok = {"steps": all(checkpoint(arm, seed, st, phase).exists() for st in GRID[phase]),
          "seed": meta["seed"] == seed and launch["pfo"]["SEED"] == seed,
          "max_steps": meta["max_steps"] == cfg["TRAIN"]["MAX_STEPS"],
          "lr_schedule": meta["lr_schedule"] == "cosine",
          "devices": meta["devices"] == 2 and meta["batch_size_per_gpu"] == 512,
          "config": launch["config_sha256"] == sha(launch["config"]) and Path(launch["config"]) == A.config_path(
              arm, seed, phase).resolve(),
          "deterministic": bool(det["deterministic_algorithms"]) and det["trainer_deterministic"],
          "router": routing["router"] == "pfr" and routing["assoc"] == "pfa" and float(routing["onhand_tau"]) == 0.5
          and launch["router_checkpoint_sha256"] == sha(cfg["PFO"]["ROUTER_CHECKPOINT"]),
          "trunk": trunk == {"design": design, "variant": variant, "module": A.DESIGNS[design][0]},
          "data_stream_equals_b2": mine == b2}
    receipt = {"run": name, "arm": arm, "seed": seed, "phase": phase, "checks": ok, "b2_run": str(b2_run(phase, seed)),
               "status": "PASS" if all(ok.values()) else "FAIL", "accepted_at_utc": now(),
               "checkpoint_sha256_by_step": {str(st): sha(checkpoint(arm, seed, st, phase)) for st in GRID[phase]
                                             if checkpoint(arm, seed, st, phase).exists()}}
    save(out, receipt, exclusive=True)
    print(receipt["status"], ok, flush=True)
    require(receipt["status"] == "PASS", "acceptance failed")


# --------------------------------------------------------------------------- engine check
def _states(d):
    return np.load(Path(d) / "states.npz", allow_pickle=False)


def check():
    keys = [f"{s}/prediction" for s in ZGZ]
    a, b = _states(OUT / "readings" / f"b2_s{SEED}" / "e1"), _states(PFO2_OUT / "readings" / B2_TAG / "e1")
    rows = {k: bool(k in a.files and k in b.files and a[k].shape == b[k].shape and np.array_equal(a[k], b[k]))
            for k in keys}
    receipt = {"check": "k1", "rows": rows, "status": "PASS" if rows and all(rows.values()) else "FAIL",
               "checked_at_utc": now()}
    save(OUT / "receipts" / "k1.json", receipt, exclusive=True)
    print(receipt["status"], json.dumps(rows), flush=True)
    require(receipt["status"] == "PASS", "engine check failed")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("b2")
    s.add_argument("--task", required=True, choices=TASKS)
    s = sub.add_parser("read")
    s.add_argument("--arm", required=True)
    s.add_argument("--task", required=True, choices=TASKS)
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--step", type=int, default=6000, choices=STEPS)
    s.add_argument("--messages", default="on", choices=("on", "off"))
    s = sub.add_parser("accept")
    s.add_argument("--arm", required=True)
    s.add_argument("--phase", required=True, choices=tuple(GRID))
    s.add_argument("--seed", type=int, default=SEED)
    sub.add_parser("check")
    a = p.parse_args(argv)
    if a.verb == "b2":
        check_b2()
        sp = read(b2_spec())
        tag = f"b2_s{SEED}"
        mine = write_once(OUT / "specs" / f"{tag}.json", {**sp, "checkpoint_sha256": sha(sp["checkpoint"])})
        return engine(tag, a.task, mine)
    if a.verb == "read":
        tag, sp = arm_spec(a.arm, a.seed, a.step, a.messages)
        return engine(tag, a.task, sp)
    if a.verb == "accept":
        return accept(a.arm, a.phase, a.seed)
    return check()


if __name__ == "__main__":
    main()
