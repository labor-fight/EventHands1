#!/usr/bin/env python3
"""ABL amendment 2 (docs/EXPERIMENT_RECORDS.md#abl-amend2-20261111): the ablation reduced to one variant per step, user
2026-11-11 ~02:00 UTC ("我只要证明每步是有效的，不一定非要遍历所有的方法吧"). Moves driver 8's jobs of the reduced set to
driver 9 and adds the no-propagation arm.

    python -m tools.pmr.jobs22 move        (driver 8 must carry its DRAIN file; driver 9 must be fresh)

The three steps and their one ablation each, every one with rol5 like the full model, seeds 3407 / 3408 / 3409:
  step 1  event-surface correspondence   (a) abgeo_rol5: S45's geometric ray association instead of the ESMNet router
  step 2  part propagation               no-propagation d1n_rol5: every message of P zeroed, per-node update and local
                                         readout kept (PMR's R-noprop; its from-scratch bases are S2's, three seeds)
  step 3  closed-loop update training    (f) d1r_rol1: the same 1500-step fine-tune with single-step updates (K = 1)
Kept from driver 8: the (a) chains (3407 running there, 3408, 3409) and (f)'s seed 3409; dropped: the seed-3409 chains of
(b) abnr (its training stopped), (c) d1m, (d) d1g0, (e) d1h0 (their finished seeds stay as supplementary rows).
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

FREEZE = "h:freeze:abl4"
KEEP_PREFIX = ("g:",)
KEEP_KEYS = ("a3:rtrain:d1r_rol1:s3409", "a3:raccept:d1r_rol1:s3409", "a3:fread:d1r_rol1:s3409:str:e1",
             "a3:rread:d1r_rol1:s3409:e1")


def noprop(seed):
    from tools.pmr import arms_any4  # noqa: F401
    from tools.pmr import arms_rol2 as R2
    name, J = "d1n_rol5", []
    if seed != 3407:
        J += [pair(f"h:rtrain:{name}:s{seed}", "tools/pmr/train_rol4.py", R2.write_config(name, seed, "formal"), [FREEZE],
                   ROL_MEM, "rol5 of the S2 no-propagation base"),
              cpu(f"h:raccept:{name}:s{seed}", ["accept", "--arm", name, "--seed", str(seed), "--phase", "formal"],
                  [f"h:rtrain:{name}:s{seed}"], module="tools.pmr.run_rol4")]
        deps = [f"h:raccept:{name}:s{seed}"]
        J.append(gpu(f"h:rread:{name}:s{seed}:e1", ["read", "--arm", name, "--seed", str(seed), "--task", "e1"], deps,
                     "packet form", module="tools.pmr.run_rol4"))
    else:
        deps = [FREEZE]
    J.append(gpu(f"h:fread:{name}:s{seed}:str:e1", ["read", "--arm", name, "--seed", str(seed), "--form", "str", "--task",
                                                     "e1"], deps, "ablation table", module="tools.pmr.run_forms"))
    return J


def move():
    from tools.pmr import chain9 as C9
    src, target = OUT / "driver8", C9.C.D2
    require(target.name == "driver9" and not C9.C.QUEUE.exists(), "driver 9 must be fresh")
    require((src / "DRAIN").exists(), "drain driver 8 first")
    queue, status = read(src / "queue.json"), read(src / "status.json")
    waits, moved = M.plan(queue, status, "driver8")
    kept = [j for j in moved if j["key"].startswith(KEEP_PREFIX) or j["key"] in KEEP_KEYS]
    dropped = [j["key"] for j in moved if j not in kept]
    keys = {j["key"] for j in kept} | {w["key"] for w in waits}
    require(all(d in keys for j in kept for d in j["deps"]), "a kept job depends on a dropped one")
    require(set(KEEP_KEYS) <= {j["key"] for j in kept}, "(f) seed 3409 must be pending in driver 8")
    by = {j["key"]: j for j in kept}
    g = lambda seed: [by[k] for k in by if k.startswith("g:") and k.endswith((f":s{seed}", f":s{seed}:e1", f":s{seed}:str:e1"))]
    f3409 = [by[k] for k in KEEP_KEYS]
    jobs = ([cpu(FREEZE, ["verify", "--stage", "abl4"], [], "manifest unchanged", module="tools.pmr.freeze")] + waits
            + noprop(3407) + g(3407) + noprop(3408) + g(3408) + noprop(3409) + g(3409) + f3409)
    jk = [j["key"] for j in jobs]
    require(len(jk) == len(set(jk)) and set(jk) >= {j["key"] for j in kept}, "every kept job exactly once")
    path = OUT / "jobs/driver9_abl_amend2.json"
    require(not path.exists(), "written once")
    save(path, jobs, exclusive=True)
    C9.C.add(jobs)
    save(target / "policy.json", {"train_pairs": [list(p) for p in C9.C.PAIRS],
                                  "note": "abl amend 2: direct (4,5),(6,7) first; borrowed pairs inside 0-3 only"}, exclusive=True)
    save(target / "migration.json", {"from": str(src), "waits": [w["key"] for w in waits], "kept": len(kept),
                                     "dropped": dropped, "source_queue_sha256": sha(src / "queue.json"),
                                     "source_status_sha256": sha(src / "status.json"), "jobs_file": str(path),
                                     "written_utc": now()}, exclusive=True)
    print(json.dumps({"waits": [w["key"] for w in waits], "kept": len(kept), "dropped": len(dropped), "total": len(jobs)}))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("move",))
    p.parse_args(argv)
    move()


if __name__ == "__main__":
    main()
