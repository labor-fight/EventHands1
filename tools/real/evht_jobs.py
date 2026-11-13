#!/usr/bin/env python3
"""REAL amendment 5 tasks in the real study's packer table (docs/EXPERIMENT_RECORDS.md#real-amend5-20261111).

The packers (tools/real/pack.py, frozen) run any task of the shared table outputs/real/20261110/pack/state.json; this
file appends the EvHandTrack / ablation tasks (never edits an earlier task) and the policy keys they need, under the
table lock:
  ev:router                 external (status EXTERNAL): the real r_s2d, trained by tools/real/evht_router.py outside
                            the packers (two-card DDP, the recipe's 2 x 512); `router-done` sets it DONE once its
                            completion record is there, and the router arms' trainings wait on it
  ev:train:<arm>:<seed>     tools/real/evht_train.py (one card); a rollout arm after its base arm's acceptance
  ev:accept:<arm>:<seed>    tools/real/evht.py accept (CPU)
  ev:e1:<arm>:<seed>        tools/real/evht.py read (streaming form, E1, test then dev) of the paper rows' arms
  ev:table                  tools/real/table2.py (CPU) after every reading
Policy keys (added only if absent): est_gib evht 7, evhtrol 6, e1ev 3.5; max_per_arm evht 1 (raised by hand as cards
free up, recorded in policy["edits"]).

    python tools/real/evht_jobs.py add [--dry-run]
    python tools/real/evht_jobs.py router-done
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from tools.real import evht as X  # noqa: E402
from tools.real import ns  # noqa: E402
from tools.real import pack as P  # noqa: E402

PY = P.PY
EST = {"evht": 7.0, "evhtrol": 6.0, "e1ev": 3.5}
READ = ("d1r_rol5", "d1r_rol1", "abgeo_rol5", "d1m_rol5", "d1g0_rol5")


def tasks():
    t = {}

    def add(key, kind, cmd, deps, gpu=True, outputs=(), prio=1, est="e1ev", status="PENDING"):
        t[key] = {"kind": kind, "cmd": cmd, "deps": list(deps), "arm": "evht", "gpu": gpu,
                  "outputs": [str(o) for o in outputs], "prio": prio, "est": est, "status": status,
                  "added_utc": P.now(), "added_by": X.SECTION}
    add("ev:router", "external", None, (), gpu=False, outputs=[X.ROUTER_DIR], status="EXTERNAL")
    for seed in ns.SEEDS:
        for arm in X.ARMS:
            rol = X.kind(arm) == "rol"
            if rol:
                deps = [f"ev:accept:{X.ROLS[arm]}:{seed}"]
            else:
                deps = [] if arm == "abgeo" else ["ev:router"]
            add(f"ev:train:{arm}:{seed}", "train", [PY, "tools/real/evht_train.py", "--config",
                                                   str(X.config_path(arm, seed))], deps,
                outputs=[X.run_dir(arm, seed)], prio=1, est="evhtrol" if rol else "evht")
            add(f"ev:accept:{arm}:{seed}", "accept", [PY, "tools/real/evht.py", "accept", "--arm", arm, "--seed",
                                                     str(seed)], [f"ev:train:{arm}:{seed}"], gpu=False,
                outputs=[X.OUT / "receipts" / f"{X.run_name(arm, seed)}.json"])
            if arm in READ:
                add(f"ev:e1:{arm}:{seed}", "e1", [PY, "tools/real/evht.py", "read", "--arm", arm, "--seed", str(seed)],
                    [f"ev:accept:{arm}:{seed}"], outputs=[X.report(arm, seed).parents[1]], prio=0, est="e1ev")
    add("ev:table", "table", [PY, "tools/real/table2.py"], [k for k in t if k.startswith("ev:e1:")], gpu=False,
        outputs=[ns.OUT / "table2.md"])
    return t


def add_tasks(dry):
    new = tasks()
    with P.locked() as st:
        pol = json.loads(P.POLICY.read_text())
        changed = False
        for k, v in EST.items():
            if k not in pol.setdefault("est_gib", {}):
                pol["est_gib"][k] = v
                changed = True
        if "evht" not in pol.setdefault("max_per_arm", {}):
            pol["max_per_arm"]["evht"] = 1
            changed = True
        if changed and not dry:
            pol.setdefault("edits", []).append({"at": P.now(), "change": f"est_gib += {EST}, max_per_arm.evht = 1",
                                                "why": "tools/real/evht_jobs.py add (#real-amend5-20261111)"})
            tmp = P.POLICY.with_suffix(".tmp")
            tmp.write_text(json.dumps(pol, indent=1) + "\n")
            os.replace(tmp, P.POLICY)          # before the tasks: a packer reads est_gib[t["est"]] of every task
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
        t = st["tasks"]["ev:router"]
        X.require(t["status"] == "EXTERNAL", "ev:router is " + t["status"])
        t.update(status="DONE", ended=P.now(), checkpoint_sha256=rec["checkpoint_sha256"])
    print("ev:router DONE", rec["checkpoint_sha256"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verb", choices=("add", "router-done"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.verb == "add":
        add_tasks(a.dry_run)
    else:
        router_done()


if __name__ == "__main__":
    main()
