#!/usr/bin/env python3
"""B2UB readings, acceptance and engine checks (docs/EXPERIMENT_RECORDS.md#b2ub-prereg-20261104), through PFO's frozen
engine (tools/pfo/engine.py, tools/pfo/nets.py, unchanged) with the output root redirected to outputs/b2ub/20261104.

    python -m tools.b2ub.run u1 --seed S --route R --task T
        B2 (PFO2 fgs, step 6000, the accepted checkpoint) with the test-time routes R: router (deployed r_s2d) | label |
        label_loc | label_part | label_onhand; tag b2_s<seed>[__<route>]. The router of every mix is B2's deployed
        router (r_s2d), not sr1 (tools/pfo/nets.py `_router_for` is re-bound in this process only)
    python -m tools.b2ub.run f1label
        engine check K2: F (PFO f1, seed 3407) with the plain label routes, e1; tag f1_s3407__label
    python -m tools.b2ub.run u2 --seed S --clamp C [--route label]
        B2's closed loop with the clamp set C fed back as GT (tools/b2ub/engine.py e1c); tag b2_s<seed>[__label]__clamp-C
    python -m tools.b2ub.run accept --phase preflight|formal
        U3 acceptance of B2-gt (seed 3407) against B2 seed 3407
    python -m tools.b2ub.run u3 --task T --route label|router [--step 6000]
        B2-gt readings; tag b2gt_s3407[@step][__label]
    python -m tools.b2ub.run u3c --clamp C
        B2-gt with the label routes and the clamp set C (e1c); tag b2gt_s3407__label__clamp-C
    python -m tools.b2ub.run check --what k123|k4|u3k4 [--seed S]
        the written engine checks K1-K4 (bitwise); receipts/<check>.json; exit 1 on failure
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.b2ub import arms as A  # noqa: E402
from tools.b2ub.common import (CONFIGS, F_LABEL_E1, OUT, SEEDS, U3_SEED, ZGZ, b2_reading, b2_run, b2_spec, check_b2,  # noqa: E402
                               f1_spec, now, read, require, save, sha)

U1_ROUTES = ("router", "label", "label_loc", "label_part", "label_onhand")
TASKS = ("e1", "e3", "pert", "f1", "tl_e1")
U3_STEPS = (5000, 5500, 6000)
GRID = {"preflight": (40,), "formal": tuple(range(500, 6001, 500))}


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


def b2gt_eval_config():
    from tools.pfo2.run import _eval_cfg
    name = A.run_name("b2gt", U3_SEED)
    return write_once(CONFIGS / "eval" / f"{name}.yaml", _eval_cfg(A.build_config("b2gt", U3_SEED), name))


def b2gt_checkpoint(step, phase="formal"):
    return A.run_dir("b2gt", U3_SEED, phase) / f"{A.run_name('b2gt', U3_SEED, phase)}-step={step}.ckpt"


def b2gt_spec(step):
    receipt = read(OUT / "receipts" / f"{A.run_name('b2gt', U3_SEED)}.json")
    require(receipt["status"] == "PASS", "B2-gt training not accepted")
    ck = b2gt_checkpoint(step)
    require(sha(ck) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since acceptance")
    name = f"b2gt_s{U3_SEED}" + ("" if step == 6000 else f"@{step}")
    sp = {"kind": "routed", "checkpoint": str(ck), "checkpoint_sha256": receipt["checkpoint_sha256_by_step"][str(step)],
          "config": str(b2gt_eval_config()), "trunk": "gshare", "module": "trunk4"}
    return name, write_once(OUT / "specs" / f"{name}.json", sp)


def engine(name, task, sp, route=None):
    from tools.pfo import engine as E
    from tools.pfo2.run import prepare
    prepare(sp)
    deployed_router_for_mixes()
    E.OUT = OUT                                                     # readings go under outputs/b2ub/20261104
    return E.run(name, task, None if route == "router" else route, None, (), str(sp))


def clamp_run(name, sp, clamp, route=None):
    from tools.b2ub import engine as BE
    from tools.pfo2.run import prepare
    prepare(sp)
    deployed_router_for_mixes()
    return BE.run(name, str(sp), clamp, route)


# --------------------------------------------------------------------------- U3 acceptance
def stream_rows(run_dir):
    return {r: [x["sha256"] for x in read(Path(run_dir) / f"pair_stream_rank{r}.json")[:2]] for r in (0, 1)}


def accept(phase):
    require(phase in GRID, "preflight | formal")
    name = A.run_name("b2gt", U3_SEED, phase)
    out = OUT / "receipts" / f"{name}.json"
    require(not out.exists(), "accepted once")
    d = A.run_dir("b2gt", U3_SEED, phase)
    launch, meta, det = read(d / "b2ub_launch.json"), read(d / "training_metadata.json"), read(d / "b2ub_det.json")
    cfg = A.build_config("b2gt", U3_SEED, phase)
    mine, b2 = stream_rows(d), stream_rows(b2_run(U3_SEED, phase))
    ok = {"steps": all(b2gt_checkpoint(st, phase).exists() for st in GRID[phase]),
          "seed": meta["seed"] == U3_SEED and launch["pfo"]["SEED"] == U3_SEED,
          "max_steps": meta["max_steps"] == cfg["TRAIN"]["MAX_STEPS"],
          "lr_schedule": meta["lr_schedule"] == "cosine",
          "devices": meta["devices"] == 2 and meta["batch_size_per_gpu"] == 512,
          "config": launch["config_sha256"] == sha(launch["config"]) and Path(launch["config"]) == A.config_path(
              "b2gt", U3_SEED, phase).resolve(),
          "deterministic": bool(det["deterministic_algorithms"]) and det["trainer_deterministic"],
          "label_routes": launch["routing"]["router"] == "label"
          and launch["routing"]["route_association"] == "semkine.s45sm.pfa_association"
          and launch["routing"]["associate_rays"] == "semkine.s45sm.empty_association",
          "trunk": launch["trunk"] == "gshare" and launch["module"] == "trunk4",
          "data_stream_equals_b2": mine == b2}
    receipt = {"run": name, "arm": "b2gt", "seed": U3_SEED, "phase": phase, "checks": ok, "b2_run": str(b2_run(U3_SEED, phase)),
               "status": "PASS" if all(ok.values()) else "FAIL", "accepted_at_utc": now(),
               "checkpoint_sha256_by_step": {str(st): sha(b2gt_checkpoint(st, phase)) for st in GRID[phase]
                                             if b2gt_checkpoint(st, phase).exists()}}
    save(out, receipt, exclusive=True)
    print(receipt["status"], ok, flush=True)
    require(receipt["status"] == "PASS", "acceptance failed")


# --------------------------------------------------------------------------- engine checks
def _states(d):
    return np.load(Path(d) / "states.npz", allow_pickle=False)


def _same(a, b, keys):
    out = {}
    for k in keys:
        out[k] = bool(k in a.files and k in b.files and a[k].shape == b[k].shape and np.array_equal(a[k], b[k]))
    return out


def readings(tag, task):
    return OUT / "readings" / tag / task


def first_of_segment(run_id):
    first = np.zeros(len(run_id), bool)
    first[0] = True
    first[1:] = run_id[1:] != run_id[:-1]
    return first


def check(what, seed=None):
    rows = {}
    if what == "k123":
        mine, ref = _states(readings(f"b2_s{U3_SEED}", "e1")), _states(b2_reading(U3_SEED, "e1"))
        rows["K1_router_e1_equals_pfo2_b2_e1"] = _same(mine, ref, [f"{s}/prediction" for s in ZGZ])
        mine, ref = _states(readings(f"f1_s{U3_SEED}__label", "e1")), _states(F_LABEL_E1)
        rows["K2_f1_label_e1_equals_p0b"] = _same(mine, ref, [f"{s}/prediction" for s in ZGZ])
        none = _states(readings(f"b2_s{U3_SEED}__clamp-none", "e1c"))
        rows["K3_clamp_none_equals_b2_e1"] = _same(none, _states(b2_reading(U3_SEED, "e1")),
                                                   [f"{s}/prediction" for s in ZGZ])
        trf, e3 = _states(readings(f"b2_s{U3_SEED}__clamp-TRF", "e1c")), _states(b2_reading(U3_SEED, "e3"))
        k3 = {}
        for s in ZGZ:
            keep = ~first_of_segment(trf[f"{s}/run_id"])
            k3[f"{s}/end_ms"] = bool(np.array_equal(trf[f"{s}/end_ms"], e3[f"{s}/end_ms"]))
            k3[f"{s}/prediction_after_segment_start"] = bool(np.array_equal(trf[f"{s}/prediction"][keep],
                                                                            e3[f"{s}/prediction"][keep]))
            k3[f"{s}/n_compared"] = int(keep.sum())
        rows["K3_clamp_TRF_equals_b2_e3"] = k3
        name = "k123"
    elif what == "k4":
        require(seed in SEEDS, "seed")
        none, e1 = _states(readings(f"b2_s{seed}__label__clamp-none", "e1c")), _states(readings(f"b2_s{seed}__label", "e1"))
        rows["K4_label_clamp_none_equals_label_e1"] = _same(none, e1, [f"{s}/prediction" for s in ZGZ])
        name = f"k4_s{seed}"
    elif what == "u3k4":
        none = _states(readings(f"b2gt_s{U3_SEED}__label__clamp-none", "e1c"))
        e1 = _states(readings(f"b2gt_s{U3_SEED}__label", "e1"))
        rows["K4_b2gt_label_clamp_none_equals_label_e1"] = _same(none, e1, [f"{s}/prediction" for s in ZGZ])
        name = "k4_b2gt"
    else:
        raise ValueError(what)
    flat = [v for r in rows.values() for k, v in r.items() if isinstance(v, bool)]
    receipt = {"check": name, "rows": rows, "status": "PASS" if flat and all(flat) else "FAIL", "checked_at_utc": now()}
    save(OUT / "receipts" / f"{name}.json", receipt, exclusive=True)
    print(receipt["status"], json.dumps(rows), flush=True)
    require(receipt["status"] == "PASS", "engine check failed: " + name)


# --------------------------------------------------------------------------- CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("u1")
    s.add_argument("--seed", type=int, required=True, choices=SEEDS)
    s.add_argument("--route", required=True, choices=U1_ROUTES)
    s.add_argument("--task", required=True, choices=TASKS)
    sub.add_parser("f1label")
    s = sub.add_parser("u2")
    s.add_argument("--seed", type=int, required=True, choices=SEEDS)
    s.add_argument("--clamp", required=True)
    s.add_argument("--route", choices=("label",))
    s = sub.add_parser("accept")
    s.add_argument("--phase", required=True, choices=tuple(GRID))
    s = sub.add_parser("u3")
    s.add_argument("--task", required=True, choices=TASKS)
    s.add_argument("--route", required=True, choices=("label", "router"))
    s.add_argument("--step", type=int, default=6000, choices=U3_STEPS)
    s = sub.add_parser("u3c")
    s.add_argument("--clamp", required=True)
    s = sub.add_parser("check")
    s.add_argument("--what", required=True, choices=("k123", "k4", "u3k4"))
    s.add_argument("--seed", type=int)
    a = p.parse_args(argv)
    if a.verb == "u1":
        check_b2(a.seed)
        return engine(f"b2_s{a.seed}", a.task, b2_spec(a.seed), a.route)
    if a.verb == "f1label":
        sp = read(f1_spec())
        require(sha(sp["checkpoint"]) == sp["checkpoint_sha256"], "F checkpoint changed")
        return engine(f"f1_s{U3_SEED}", "e1", f1_spec(), "label")
    if a.verb == "u2":
        check_b2(a.seed)
        return clamp_run(f"b2_s{a.seed}", b2_spec(a.seed), a.clamp, a.route)
    if a.verb == "accept":
        return accept(a.phase)
    if a.verb == "u3":
        name, sp = b2gt_spec(a.step)
        return engine(name, a.task, sp, a.route)
    if a.verb == "u3c":
        name, sp = b2gt_spec(6000)
        return clamp_run(name, sp, a.clamp, "label")
    return check(a.what, a.seed)


if __name__ == "__main__":
    main()
