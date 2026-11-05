#!/usr/bin/env python3
"""PFO2 frozen manifests (docs/EXPERIMENT_RECORDS.md#pfo2-prereg-20261103): SHA256 of every file a stage depends on.

    python -m tools.pfo2.freeze write --stage s0      outputs/pfo2/20261103/manifests/s0.json (written once)
    python -m tools.pfo2.freeze verify --stage s0     re-hash; exit non-zero on any difference
A stage's manifest covers its PFO2 code, configs and job file, the PFO tools / modules they import (read-only reuse) and
the frozen inputs (F's checkpoints, receipts and readings, S43's readings, the routers, the terminated PFO runs read here).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo2.common import CONFIGS, OUT, PFO_OUT, SEEDS, now, read, require, save, sha  # noqa: E402

PFO_TOOLS = ["tools/pfo/" + f for f in ("__init__.py", "common.py", "engine.py", "nets.py", "arms.py", "arms2.py",
                                        "arms3.py", "run4.py", "train.py", "cost.py", "router_train.py",
                                        "analyze.py")]
MODULES = ["semkine/" + f for f in ("pfo_trunk.py", "pfo_trunk2.py", "pfo_trunk3.py", "pfo_trunk4.py", "pfo_router.py",
                                    "s45sm.py", "s45sm_fast.py", "s45sm_oracle.py", "xyz_mesh.py", "train.py",
                                    "dataset.py", "eval_track.py", "s45rt_router.py", "xag.py")]
OTHER = ["model/model.py", "model/config.py", "tools/xcl/evaluate.py"]


def stage_files(stage):
    files = []
    if stage == "s0":
        files += ["tools/pfo2/" + f for f in ("__init__.py", "common.py", "chain.py", "arms.py", "arms_s0.json",
                                              "train.py", "run.py", "replay.py", "decide.py", "jobs.py", "freeze.py")]
        files += ["tests/test_pfo2.py"] + PFO_TOOLS + MODULES + OTHER
        files += sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tools/pfo").glob("arms_*.json"))
        files += sorted(str(p.relative_to(ROOT)) for p in CONFIGS.glob("*.yaml"))
        files += [str((OUT / "jobs/s0.json").relative_to(ROOT))]
        for s in SEEDS:
            d = PFO_OUT / "runs" / f"pfo_f1_s{s}"
            files += [str((d / f"pfo_f1_s{s}-step={st}.ckpt").relative_to(ROOT)) for st in (500, 5000, 5500, 6000)]
            files += [str((PFO_OUT / "receipts" / f"pfo_f1_s{s}.json").relative_to(ROOT))]
            for task in ("e1", "e3", "pert", "f1", "tl_e1"):
                files += [str((PFO_OUT / "readings" / f"pfo_f1_s{s}" / task / f).relative_to(ROOT))
                          for f in ("report.json", "states.npz")]
            files += [str((PFO_OUT / "readings" / f"s43_s{s}" / "e1" / f).relative_to(ROOT))
                      for f in ("report.json", "states.npz")]
        d = PFO_OUT / "runs" / "pfo_f1_s3407_preflight"
        files += [str((d / "pfo_f1_s3407_preflight-step=40.ckpt").relative_to(ROOT))]
        for r in ("r_s2d", "r_wide"):
            files += [str((PFO_OUT / "routers" / r / f"{r}_final.ckpt").relative_to(ROOT))]
        for arm, seed, step in (("f1self", 3407, 5000), ("f1self", 3407, 5500), ("f1self", 3408, 5000),
                                ("f1self", 3408, 5500), ("f1noise", 3407, 4000)):
            d = PFO_OUT / "runs" / f"pfo_{arm}_s{seed}"
            files += [str((d / f"pfo_{arm}_s{seed}-step={step}.ckpt").relative_to(ROOT))]
    else:
        raise ValueError("unknown stage " + stage)
    return files


def write(stage):
    dest = OUT / "manifests" / f"{stage}.json"
    require(not dest.exists(), "manifest written once: " + str(dest))
    files = stage_files(stage)
    missing = [f for f in files if not (ROOT / f).exists()]
    require(not missing, f"missing files: {missing[:5]}")
    save(dest, {"stage": stage, "written_at_utc": now(), "files": {f: sha(ROOT / f) for f in files}}, exclusive=True)
    print(len(files), "files ->", dest)


def verify(stage):
    m = read(OUT / "manifests" / f"{stage}.json")
    bad = [f for f, h in m["files"].items() if not (ROOT / f).exists() or sha(ROOT / f) != h]
    require(not bad, f"frozen files changed: {bad[:5]}")
    print("manifest", stage, "OK", len(m["files"]), "files")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("verb", choices=("write", "verify"))
    p.add_argument("--stage", required=True)
    a = p.parse_args(argv)
    (write if a.verb == "write" else verify)(a.stage)


if __name__ == "__main__":
    main()
