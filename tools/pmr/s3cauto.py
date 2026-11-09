#!/usr/bin/env python3
"""PMR stage S3c registration, run as the CPU job `s3c:auto` after `s5:auto` (docs/EXPERIMENT_RECORDS.md
#pmr-s3c-prereg-20261109): when S5 was registered for a refinement X (decisions/s5_auto.json), the final-form S3c list
of X (tools/pmr/jobs11.py `final`: X's readings, the cost block with S43 / B2 / D1 / X, the tables) is written to
outputs/pmr/20261109/jobs/s3c.json and appended to driver 2; otherwise nothing. decisions/s3c_auto.json records the
outcome (once).

    PMR_DRIVER_DIR=driver2 python -m tools.pmr.s3cauto
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import chain2 as C2  # noqa: E402
from tools.pmr import jobs11 as J11  # noqa: E402
from tools.pmr.common import OUT, now, read, require, save  # noqa: E402


def main():
    out = OUT / "jobs" / "s3c.json"
    rec = OUT / "decisions" / "s3c_auto.json"
    require(not rec.exists() and not out.exists(), "registered once")
    arm = read(OUT / "decisions" / "s5_auto.json")["arm"]
    jobs = J11.final(arm) if arm else []
    if jobs:
        out.write_text(json.dumps(jobs, indent=1) + "\n")
        C2.C.add(jobs)
    save(rec, {"arm": arm, "jobs": len(jobs), "jobs_file": str(out) if jobs else None, "written_utc": now()},
         exclusive=True)
    print(arm, len(jobs), "jobs")


if __name__ == "__main__":
    main()
