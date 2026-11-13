#!/usr/bin/env python3
"""ABL amendment 1 (docs/EXPERIMENT_RECORDS.md#abl-amend1-20261111): move driver 7's pending jobs to driver 8 with the rerun
of ablation (a)'s rol5 fine-tunes in front.

    python -m tools.pmr.jobs21 move        (driver 7 must carry its DRAIN file; driver 8 must be fresh)

Queue of driver 8: the abl3 freeze check; tools/pmr/migrate.py's wait jobs for driver 7's running jobs; (a) abgeo_rol5 at
seed 3407 (preflight, formal, readings) and 3408 (formal, readings, the base reading its failed chain skipped); every job
still pending in driver 7 except the seed-3409 abgeo rol5 chain, unchanged; (a) at seed 3409 through the new entry. The
rollout trainings of (a) run through tools/pmr/train_rol7.py (semkine/abl_rol.py); acceptance and readings through the
frozen run_rol6 / run_forms6 / run11.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import migrate as M  # noqa: E402
from tools.pmr.common import OUT, now, read, require, save, sha  # noqa: E402
from tools.pmr.jobs19 import ROL_MEM, cpu, gpu, pair  # noqa: E402

ARM, S0 = "abgeo", 3407
DROP = ("a3:rtrain:abgeo_rol5:s3409", "a3:raccept:abgeo_rol5:s3409", "a3:fread:abgeo_rol5:s3409:str:e1",
        "a3:rread:abgeo_rol5:s3409:e1", "a3:read:abgeo:s3409:e1")
FREEZE = "g:freeze:abl3"


def chain(seed, deps, preflight):
    from tools.pmr import arms_any6  # noqa: F401
    from tools.pmr import arms_rol2 as R2
    name, J = f"{ARM}_rol5", []
    d = list(deps)
    if preflight:
        J += [pair(f"g:rpre:{name}:s{seed}", "tools/pmr/train_rol7.py", R2.write_config(name, seed, "preflight"), d, ROL_MEM,
                   "rol5 preflight, RayROLModel"),
              cpu(f"g:rpre:accept:{name}:s{seed}", ["accept", "--arm", name, "--seed", str(seed), "--phase", "preflight"],
                  [f"g:rpre:{name}:s{seed}"], module="tools.pmr.run_rol6")]
        d = [f"g:rpre:accept:{name}:s{seed}"]
    J += [pair(f"g:rtrain:{name}:s{seed}", "tools/pmr/train_rol7.py", R2.write_config(name, seed, "formal"), d, ROL_MEM,
               "rol5, RayROLModel"),
          cpu(f"g:raccept:{name}:s{seed}", ["accept", "--arm", name, "--seed", str(seed), "--phase", "formal"],
              [f"g:rtrain:{name}:s{seed}"], module="tools.pmr.run_rol6"),
          gpu(f"g:fread:{name}:s{seed}:str:e1", ["read", "--arm", name, "--seed", str(seed), "--form", "str", "--task", "e1"],
              [f"g:raccept:{name}:s{seed}"], "ablation table", module="tools.pmr.run_forms6"),
          gpu(f"g:rread:{name}:s{seed}:e1", ["read", "--arm", name, "--seed", str(seed), "--task", "e1"],
              [f"g:raccept:{name}:s{seed}"], "packet form", module="tools.pmr.run_rol6")]
    if seed != S0:
        J.append(gpu(f"g:read:{ARM}:s{seed}:e1", ["read", "--arm", ARM, "--seed", str(seed), "--task", "e1"],
                     [f"g:raccept:{name}:s{seed}"], "from scratch, packet form", module="tools.pmr.run11"))
    return J


def move():
    from tools.pmr import chain8 as C8
    src, target = OUT / "driver7", C8.C.D2
    require(target.name == "driver8" and not C8.C.QUEUE.exists(), "driver 8 must be fresh")
    require((src / "DRAIN").exists(), "drain driver 7 first")
    queue, status = read(src / "queue.json"), read(src / "status.json")
    waits, moved = M.plan(queue, status, "driver7")
    kept = [j for j in moved if j["key"] not in DROP]
    require(len(kept) == len(moved) - len(DROP), "the dropped chain must be pending in driver 7")
    require(not any(d in DROP for j in kept for d in j["deps"]), "a kept job depends on the dropped chain")
    head = [cpu(FREEZE, ["verify", "--stage", "abl3"], [], "manifest unchanged", module="tools.pmr.freeze")]
    jobs = (head + waits + chain(S0, [FREEZE], True) + chain(3408, [FREEZE, f"g:rpre:accept:{ARM}_rol5:s{S0}"], False)
            + kept + chain(3409, [FREEZE, "a3:accept:abgeo:s3409", f"g:rpre:accept:{ARM}_rol5:s{S0}"], False))
    keys = [j["key"] for j in jobs]
    require(len(keys) == len(set(keys)), "duplicate job keys")
    path = OUT / "jobs/driver8_abl_amend1.json"
    require(not path.exists(), "written once")
    save(path, jobs, exclusive=True)
    C8.C.add(jobs)
    save(target / "policy.json", {"train_pairs": [list(p) for p in C8.C.PAIRS],
                                  "note": "abl amend 1: direct (4,5),(6,7) first; borrowed pairs inside 0-3 only"}, exclusive=True)
    save(target / "migration.json", {"from": str(src), "waits": [w["key"] for w in waits], "moved": len(kept),
                                     "dropped": list(DROP), "source_queue_sha256": sha(src / "queue.json"),
                                     "source_status_sha256": sha(src / "status.json"), "jobs_file": str(path),
                                     "written_utc": now()}, exclusive=True)
    print(json.dumps({"waits": [w["key"] for w in waits], "moved": len(kept), "total": len(jobs)}))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("move",))
    p.parse_args(argv)
    move()


if __name__ == "__main__":
    main()
