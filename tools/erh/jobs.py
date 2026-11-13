#!/usr/bin/env python3
"""EvRealHands tasks in the real study's packer table (docs/EXPERIMENT_RECORDS.md#erh-prereg-20261111).

The packers (tools/real/pack.py, frozen) run any task of the shared table outputs/real/20261110/pack/state.json; this
file appends the EvRealHands tasks (prefix erh:, never editing an earlier task). Every command goes through
tools/erh/run.py, so the real-study tools run in the EvRealHands namespace:
  CMP arms   tools/real/pack.py's chain unchanged: smoke (seed 3407, 20 steps; epx after epxf, ehpe after epf) and
             30-endpoint smoke readings of eho / epx / ehpe -> erh:gate -> formal trainings x 3 seeds (epx after epxf's
             accept, ehpe after epf's accept), accept, E1 readings of eho / epx / ehpe / fme
  EvHandTrack  erh:ev:router (external: tools/real/evht_router.py on two cards, set DONE by `router-done`) -> d1r x 3 ->
             accept -> d1r_rol5 -> accept -> E1 (streaming form)
  erh:hold:<seed>  external holds of each seed's formal trainings, released by hand (`release --seed S`) as cards free
             up, so that the EvRealHands trainings fill the cards the real study's EvHandTrack work leaves free
  erh:table  tools/erh/table.py after every reading
The arm field of a training is the method's (eho / epxf / epx / epf / ehpe / fme, evht), so the per-card caps of
policy.json count both datasets' trainings of a method together; the memory estimates are the real study's.

    python tools/erh/jobs.py add [--dry-run]
    python tools/erh/jobs.py router-done
    python tools/erh/jobs.py release --seed S
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.real import pack as P  # noqa: E402  (first: the shared table is the real study's, bound at import)

os.environ["ERH_NAMESPACE"] = "1"
from tools.erh import ns as E  # noqa: E402

R = E.install()
from tools.real import evht as X  # noqa: E402

PY = P.PY
RUN = [PY, "tools/erh/run.py"]
CMP_ARMS = ("eho", "epxf", "epx", "epf", "ehpe", "fme")
DEP = {"epx": "epxf", "ehpe": "epf"}
READ_ARMS = ("eho", "epx", "ehpe", "fme")
SMOKE_READ = ("eho", "epx", "ehpe")
#: priorities (lower first when a card has room): readings, then the longest chains first
PRIO = {"fme": 1, "epf": 2, "ehpe": 2, "evht": 2, "epxf": 3, "epx": 3, "eho": 4}
EV_ARMS = ("d1r", "d1r_rol5")


def cfg(arm, seed, phase):
    return str(R.CONFIGS / (f"cmp_{arm}_s{seed}" + ("" if phase == "formal" else f"_{phase}") + ".yaml"))


def run_dir(arm, seed, phase):
    return R.OUT / "runs" / (f"cmp_{arm}_s{seed}" + ("" if phase == "formal" else f"_{phase}"))


def tasks():
    t = {}

    def add(key, kind, cmd, deps=(), arm=None, gpu=True, outputs=(), prio=9, est="e1", status="PENDING"):
        t[key] = {"kind": kind, "cmd": cmd, "deps": list(deps), "arm": arm, "gpu": gpu, "outputs": [str(o) for o in outputs],
                  "prio": prio, "est": est, "status": status, "added_utc": P.now(), "added_by": E.SECTION}
    s0 = 3407
    for arm in CMP_ARMS:
        deps = [f"erh:smoke:{DEP[arm]}"] if arm in DEP else []
        add(f"erh:smoke:{arm}", "smoke", RUN + ["tools.real.train", "--config", cfg(arm, s0, "smoke")], deps, arm,
            outputs=[run_dir(arm, s0, "smoke")], prio=0, est=arm)
    for arm in SMOKE_READ:
        add(f"erh:rsmoke:{arm}", "rsmoke", RUN + ["tools.real.run", "e1", "--arm", arm, "--seed", str(s0), "--phase",
                                                 "smoke", "--limit", "30"], [f"erh:smoke:{arm}"], arm,
            outputs=[R.OUT / "smoke" / "readings" / f"cmp_{arm}_s{s0}_smoke"], prio=0, est="e1smoke")
    add("erh:gate", "gate", None, [k for k in t], gpu=False)
    for seed in R.SEEDS:                       # released by hand (`release --seed S`) as cards free up
        add(f"erh:hold:{seed}", "external", None, (), gpu=False, status="EXTERNAL")
    for seed in R.SEEDS:
        for arm in CMP_ARMS:
            deps = ["erh:gate", f"erh:hold:{seed}"] + ([f"erh:accept:{DEP[arm]}:{seed}"] if arm in DEP else [])
            add(f"erh:train:{arm}:{seed}", "train", RUN + ["tools.real.train", "--config", cfg(arm, seed, "formal")],
                deps, arm, outputs=[run_dir(arm, seed, "formal")], prio=PRIO[arm], est=arm)
            add(f"erh:accept:{arm}:{seed}", "accept", RUN + ["tools.real.run", "accept", "--arm", arm, "--seed",
                                                             str(seed)],
                [f"erh:train:{arm}:{seed}"], arm, gpu=False, outputs=[R.OUT / "receipts" / f"cmp_{arm}_s{seed}.json"])
        for arm in READ_ARMS:
            add(f"erh:e1:{arm}:{seed}", "e1", RUN + ["tools.real.run", "e1", "--arm", arm, "--seed", str(seed)],
                [f"erh:accept:{arm}:{seed}"], arm, outputs=[R.OUT / "readings" / f"cmp_{arm}_s{seed}"], prio=0, est="e1")
    add("erh:ev:router", "external", None, (), "evht", gpu=False, outputs=[X.ROUTER_DIR], status="EXTERNAL")
    for seed in R.SEEDS:
        for arm in EV_ARMS:
            rol = X.kind(arm) == "rol"
            deps = [f"erh:ev:accept:{X.ROLS[arm]}:{seed}"] if rol else ["erh:ev:router", f"erh:hold:{seed}"]
            add(f"erh:ev:train:{arm}:{seed}", "train", RUN + ["tools.real.evht_train", "--config",
                                                              str(X.config_path(arm, seed))], deps, "evht",
                outputs=[X.run_dir(arm, seed)], prio=PRIO["evht"], est="evhtrol" if rol else "evht")
            add(f"erh:ev:accept:{arm}:{seed}", "accept", RUN + ["tools.real.evht", "accept", "--arm", arm, "--seed",
                                                                str(seed)],
                [f"erh:ev:train:{arm}:{seed}"], "evht", gpu=False,
                outputs=[X.OUT / "receipts" / f"{X.run_name(arm, seed)}.json"])
        add(f"erh:ev:e1:d1r_rol5:{seed}", "e1", RUN + ["tools.real.evht", "read", "--arm", "d1r_rol5", "--seed",
                                                       str(seed)],
            [f"erh:ev:accept:d1r_rol5:{seed}"], "evht", outputs=[X.report("d1r_rol5", seed).parents[1]], prio=0,
            est="e1ev")
    add("erh:table", "table", RUN + ["tools.erh.table"], [k for k in t if ":e1:" in k], gpu=False,
        outputs=[R.OUT / "table.md"])
    return t


def add_tasks(dry):
    new = tasks()
    with P.locked() as st:
        pol = json.loads(P.POLICY.read_text())
        missing = sorted({v["est"] for v in new.values()} - set(pol.get("est_gib", {})))
        X.require(not missing, f"policy.json lacks est_gib keys {missing} (run tools/real/evht_jobs.py add first)")
        T = st["tasks"]
        fresh = {k: v for k, v in new.items() if k not in T}
        if not dry:
            T.update(fresh)
    print(("would add" if dry else "added"), len(fresh), "tasks;", len(new) - len(fresh), "already present")


def router_done():
    rec = json.loads(X.ROUTER.with_suffix(".json").read_text())
    X.require(rec["status"] == "COMPLETE" and rec["phase"] == "formal" and rec["step"] == 6000, "router not complete")
    from tools.cmp.common import sha
    X.require(sha(X.ROUTER) == rec["checkpoint_sha256"], "router checkpoint changed")
    with P.locked() as st:
        t = st["tasks"]["erh:ev:router"]
        X.require(t["status"] == "EXTERNAL", "erh:ev:router is " + t["status"])
        t.update(status="DONE", ended=P.now(), checkpoint_sha256=rec["checkpoint_sha256"])
    print("erh:ev:router DONE", rec["checkpoint_sha256"])


def release(seed):
    with P.locked() as st:
        t = st["tasks"][f"erh:hold:{seed}"]
        X.require(t["status"] == "EXTERNAL", f"erh:hold:{seed} is " + t["status"])
        t.update(status="DONE", ended=P.now())
    print(f"erh:hold:{seed} released")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verb", choices=("add", "router-done", "release"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    if a.verb == "add":
        add_tasks(a.dry_run)
    elif a.verb == "release":
        release(a.seed)
    else:
        router_done()


if __name__ == "__main__":
    main()
