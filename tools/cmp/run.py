#!/usr/bin/env python3
"""CMP readings (docs/EXPERIMENT_RECORDS.md#cmp-prereg-20261110): PFO's frozen engine with the CMP adapters.

    python -m tools.cmp.run accept --arm A --seed S
    python -m tools.cmp.run e1 --arm A --seed S [--form main|rolling|w50|w100|w200]

accept  the training finished: the final-step checkpoint of the registered run exists, the launch receipt names the
        registered config (same SHA) and, for warm-started arms, the INIT_FROM checkpoint whose SHA the earlier stage's
        receipt recorded; writes receipts/<run>.json (checkpoint SHA) -- readings only read accepted checkpoints
e1      the engine's E1 on zgz (own closed loop on the 50 ms grid, GT + the protocol's initial noise at every segment
        start, the shared rng over the sequences) of the accepted final checkpoint; readings/<run>[__<form>]/e1/
The engine's output root is moved to outputs/cmp/20261110 (PFO's directory is not written); the engine requires one
physical GPU among 2-7 (CMP readings use 4-7).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.cmp.arms import ARMS, EXT, checkpoint, config_path  # noqa: E402
from tools.cmp.common import OUT, now, read, require, run_dir, save, sha  # noqa: E402


def final_step(arm):
    return int(ARMS[arm]["steps"])


def tag(arm, seed, form="main"):
    return f"cmp_{arm}_s{seed}" + ("" if form == "main" else f"__{form}")


def accept(arm, seed):
    run = run_dir(arm, seed)
    ck = checkpoint(arm, seed, final_step(arm))
    require(ck.exists(), "missing final checkpoint " + str(ck))
    launch = read(run / "cmp_launch.json")
    cfgp = config_path(arm, seed)
    require(Path(launch["config"]) == cfgp.resolve() and launch["config_sha256"] == sha(cfgp), "config changed")
    rec = {"arm": arm, "seed": seed, "run": str(run), "checkpoint": str(ck), "checkpoint_sha256": sha(ck),
           "config": str(cfgp), "config_sha256": sha(cfgp), "launch": launch, "status": "PASS", "accepted_utc": now()}
    init = ARMS[arm].get("init")
    if init:
        src = read(OUT / "receipts" / f"cmp_{init[0]}_s{seed}.json")
        require(src["status"] == "PASS", "earlier stage not accepted")
        src_ck = checkpoint(init[0], seed, init[1])
        require(launch["init_from"] == str(src_ck), "INIT_FROM is not the earlier stage's checkpoint")
        want = src["checkpoint_sha256"] if init[1] == final_step(init[0]) else sha(src_ck)
        require(launch["init_from_sha256"] == want, "INIT_FROM checkpoint changed")
        rec["init_from"] = {"checkpoint": str(src_ck), "sha256": launch["init_from_sha256"]}
    save(OUT / "receipts" / f"cmp_{arm}_s{seed}.json", rec, exclusive=True)
    print("PASS", arm, seed, ck.name)


def e1(arm, seed, form="main"):
    if arm in EXT and hasattr(EXT[arm], "e1"):                  # amendment 6: a reading of the extension's own
        return EXT[arm].e1(arm, seed, form)
    from tools.pfo import common as PC
    from tools.pfo import engine as E
    from tools.pfo import nets as PN
    from tools.cmp import nets as CN
    rec = read(OUT / "receipts" / f"cmp_{arm}_s{seed}.json")
    require(rec["status"] == "PASS" and sha(rec["checkpoint"]) == rec["checkpoint_sha256"], "unaccepted checkpoint")
    model = ARMS[arm]["model"]
    spec = {"kind": "cmp", "model": model, "checkpoint": rec["checkpoint"], "config": rec["config"], "form": form}
    sp = save(OUT / "specs" / f"{tag(arm, seed, form)}.json", spec)
    E.OUT = PC.OUT = OUT
    PN.load = CN.load
    E.run(tag(arm, seed, form), "e1", spec=str(sp))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("accept", "e1"))
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--form", default="main")
    a = p.parse_args(argv)
    if a.verb == "accept":
        accept(a.arm, a.seed)
    else:
        e1(a.arm, a.seed, a.form)


if __name__ == "__main__":
    main()
