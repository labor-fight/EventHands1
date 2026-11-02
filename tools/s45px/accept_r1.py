#!/usr/bin/env python3
"""S45-PX amendment 1 (docs/EXPERIMENT_RECORDS.md#s45px-amend1-20261030): engineering acceptance with the corrected
departure step; CPU only, never scores.

The frozen check (tools/s45px/common.py verify_run) expected the arm to leave S45 at step 2. S45's readout output layer
`xyz.fc2` is zero-initialised, so at step 1 every parameter upstream of it, the new map included, receives an exactly
zero gradient and only fc2 moves: steps 1 and 2 are both S45's and the arm can first differ at step 3. This acceptance
runs every frozen check of `verify_run` unchanged (each is a hard `require`) and replaces only that flag: steps 1 and 2
reproduce S45's recorded prediction, objective and every S45 gradient bit for bit (the new map's gradient is extra),
and step 3's prediction differs. Preflight receipts go to receipts_r1/ (the superseded receipts stay in receipts/);
formal receipts go to receipts/ as registered.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.frozen import accepts  # noqa: E402
from tools.s45px import common as C  # noqa: E402

MANIFEST = C.OUT / "manifests" / "amend1.json"


def steps_check(arm, phase):
    out = C.output_dir(arm, phase)
    ranks = []
    for rank in range(2):
        rows, ref = C.journal(out, rank, 3), C.journal(C.S45_RUN, rank, 3)
        same = {}
        for s in (1, 2):
            r, o = rows[s - 1], ref[s - 1]
            same[str(s)] = (r["step"] == o["step"] == s and r["prediction_sha256"] == o["prediction_sha256"]
                            and r["objective_sha256"] == o["objective_sha256"]
                            and set(r["gradients"]) - set(o["gradients"]) == {C.NEW_TENSOR}
                            and all(r["gradients"][k] == o["gradients"][k] for k in o["gradients"]))
        ranks.append({"rank": rank, "s45_steps_equal": same,
                      "departs_at_step3": rows[2]["prediction_sha256"] != ref[2]["prediction_sha256"]})
    return ranks


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--arm", required=True, choices=tuple(C.ARMS))
    p.add_argument("--phase", required=True, choices=("preflight", "formal"))
    a = p.parse_args()
    C.verify_protocol()
    C.require(accepts(__file__, C.read(MANIFEST)["file_sha256"]["tools/s45px/accept_r1.py"]), "amendment-1 file changed")
    base = C.verify_run(a.arm, a.phase)
    ranks = steps_check(a.arm, a.phase)
    ok = all(all(r["s45_steps_equal"].values()) and r["departs_at_step3"] for r in ranks)
    old_flag, old_status = base.pop("s45_step1_exact_and_departs_at_step2"), base.pop("status")
    receipt = {**base, "status": "PASS" if ok else "FAIL", "amendment": "s45px-amend1-20261030",
               "superseded": {"s45_step1_exact_and_departs_at_step2": old_flag, "status": old_status},
               "steps_check": ranks, "s45_steps_1_2_exact_and_departs_at_step3": ok,
               "protocol_sha256": C.sha(C.PROTOCOL), "wrapper_sha256": C.sha(ROOT / "tools/s45px/train.py"),
               "amendment_wrapper_sha256": C.sha(ROOT / "tools/s45px/train_r1.py"), "verified_at_utc": C.now(),
               "engineering_only_no_accuracy_result": True}
    target = C.OUT / ("receipts_r1" if a.phase == "preflight" else "receipts") / f"{a.phase}_{a.arm}.json"
    C.save(target, receipt, exclusive=True)
    C.require(ok, "acceptance failed: " + a.phase + " " + a.arm)
    print("PASS", a.phase, a.arm)


if __name__ == "__main__":
    main()
