#!/usr/bin/env python3
"""PMR stage S5 registration, run as the CPU job `s5:auto` once the S4 + S4b + S4c joint selection is written
(docs/EXPERIMENT_RECORDS.md#pmr-s5-prereg-20261109, a conditional pre-registration written before that selection).

  REFINE_<x>     the S5 job list of d1<x> (tools/pmr/jobs8.py, frozen with S5) is written to
                 outputs/pmr/20261109/jobs/s5.json and appended to driver 2; rollout fine-tunes are kept on the direct pairs
                 (4, 5) / (6, 7): a K = 5 rollout's batches are the largest of the study and a borrowed job's service is
                 capped at 192 GiB
  NO_REFINEMENT  no job; the next round is registered by hand
Either way decisions/s5_auto.json records the outcome (once).

    PMR_DRIVER_DIR=driver2 python -m tools.pmr.s5auto
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import chain2 as C2  # noqa: E402
from tools.pmr import jobs8 as J8  # noqa: E402
from tools.pmr.common import OUT, now, read, require, save  # noqa: E402

DIRECT = [list(p) for p in C2.DIRECT]


def plan(decision):
    """(arm, jobs) for a written joint selection; (None, []) when nothing was adopted."""
    v = decision["verdict"]
    if v == "NO_REFINEMENT":
        return None, []
    require(v.startswith("REFINE_"), "unknown verdict " + v)
    arm = "d1" + v[len("REFINE_"):]
    jobs = J8.s5(arm, "s5")
    for j in jobs:
        if "train_rol2.py" in " ".join(j["command"]):
            j["pairs"] = DIRECT
    return arm, jobs


def main():
    out = OUT / "jobs" / "s5.json"
    rec = OUT / "decisions" / "s5_auto.json"
    require(not rec.exists() and not out.exists(), "registered once")
    decision = read(OUT / "decisions" / "s4c_joint.json")
    arm, jobs = plan(decision)
    if jobs:
        out.write_text(json.dumps(jobs, indent=1) + "\n")
        C2.C.add(jobs)
    save(rec, {"verdict": decision["verdict"], "arm": arm, "jobs": len(jobs), "jobs_file": str(out) if jobs else None,
               "driver": str(C2.C.D2), "written_utc": now()}, exclusive=True)
    print(decision["verdict"], arm, len(jobs), "jobs")


if __name__ == "__main__":
    main()
