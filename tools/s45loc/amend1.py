#!/usr/bin/env python3
"""S45-LOC amendment 1 (docs/EXPERIMENT_RECORDS.md#s45loc-amend1-20261029): XCL E4-style GT clamps of S45's closed loop.

`freeze` writes amendment1.json once (status FROZEN, hashes of the files the amendment changed or added, the four
tasks); `run` executes them on GPU 4-7 through the main driver's queue and writes amend1_summary.json. Readouts,
fixed before running: per domain the E1 RA under each clamp, and each clamp's share of the loop gap
(E1 - E3), share = (E1 - E1_clamp) / (E1 - E3), labelled with the main protocol's thresholds.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.s45loc import common as C  # noqa: E402

AMENDMENT = C.OUT / "amendment1.json"
CHANGED = ("tools/s45loc/common.py", "tools/s45loc/run.py", "tools/s45loc/amend1.py")


def freeze():
    C.require(C.read(C.PROTOCOL)["status"] == ("SMOKE" if C.SMOKE else "FROZEN"), "main protocol missing")
    section = C.OUT / "amend1_section.md"
    C.require(C.SMOKE or section.exists(), "write the amendment section before freezing")
    C.save(AMENDMENT, {
        "amendment": 1, "status": "FROZEN", "frozen_utc": C.now(),
        "prereg": "docs/EXPERIMENT_RECORDS.md#s45loc-amend1-20261029",
        "section_sha256": C.sha(section) if section.exists() else None,
        "written_after_main_readings": True,
        "tasks": {k: list(v) for k, v in C.AMEND1.items()},
        "sources_sha256": {name: C.sha(C.ROOT / name) for name in CHANGED},
        "unchanged_paths": "e1 / e3 and every stage-1/2 intervention keep their code path; only the two clamp modes "
                           "and the amendment-aware hash check were added",
    }, exclusive=True)
    print(AMENDMENT)


def summarize():
    from tools.s45loc.analyze import arr, label, ra
    out = {"amendment_sha256": C.sha(AMENDMENT), "computed_utc": C.now()}
    rows = {}
    for seq, base_e1, base_e3, pose, transl in (
            [("zgz_local", "b0_zgz_e1", "b0_zgz_e3", "d1_zgz_e1_gtpose", "d1_zgz_e1_gttransl")]
            + [(r, "c1_train_e1", "b0_train_e3", "d1_train_local_e1_gtpose", "d1_train_local_e1_gttransl")
               for r in C.TRAIN_LOCAL]):
        e1, e3 = ra(base_e1, seq), ra(base_e3, seq)
        p, t = ra(pose, seq), ra(transl, seq)
        rows[seq] = {"e1": e1, "e3": e3, "e1_gtpose": p, "e1_gttransl": t,
                     "pose_clamp_share": (e1 - p) / (e1 - e3), "transl_clamp_share": (e1 - t) / (e1 - e3)}
        for name, task in (("gtpose", pose), ("gttransl", transl)):
            g = lambda k: arr(task, f"{seq}/stat_{k}")  # noqa: E731
            assoc = (g("n_sv_hit") + g("n_sv_near")).sum()
            bias = (arr(task, f"{seq}/incoming_used")[:, :3] - arr(task, f"{seq}/gt_previous")[:, :3]).mean(0) * 1000
            rows[seq][f"{name}_part_agreement"] = float(g("n_sv_agree").sum() / max(assoc, 1))
            rows[seq][f"{name}_point_error_mm"] = float(g("sv_point_err_mm_sum").sum() / max(assoc, 1))
            rows[seq][f"{name}_state_translation_bias_mm"] = [float(x) for x in bias]
    out["per_recording"] = rows
    tl = [rows[r] for r in C.TRAIN_LOCAL]
    for key in ("pose_clamp_share", "transl_clamp_share", "e1_gtpose", "e1_gttransl"):
        out[f"train_local_median_{key}"] = float(np.median([r[key] for r in tl]))
    out["labels"] = {
        "zgz_local_pose_clamp": label(rows["zgz_local"]["pose_clamp_share"]),
        "zgz_local_transl_clamp": label(rows["zgz_local"]["transl_clamp_share"]),
        "train_local_pose_clamp": label(out["train_local_median_pose_clamp_share"]),
        "train_local_transl_clamp": label(out["train_local_median_transl_clamp_share"]),
    }
    C.save(C.OUT / "amend1_summary.json", out)
    print(C.OUT / "amend1_summary.json")


def run():
    from tools.s45loc import chain
    C.require(C.read(AMENDMENT)["status"] == "FROZEN", "amendment not frozen")
    proto = C.read(C.PROTOCOL)
    slots = [(g, s) for g in C.PHYSICAL_GPUS for s in range(int(proto["processes_per_gpu"]))]
    chain.status(stage="amend1", event="amendment 1 start")
    failed = chain.run_queue(list(C.AMEND1), None, slots)
    C.require(not failed, f"amendment 1 failed: {failed}")
    summarize()
    chain.status(stage="done", event="amendment 1 complete")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("action", choices=("freeze", "run", "summarize"))
    args = p.parse_args(argv)
    {"freeze": freeze, "run": run, "summarize": summarize}[args.action]()


if __name__ == "__main__":
    main()
