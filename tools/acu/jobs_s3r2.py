#!/usr/bin/env python3
"""ACU engineering amendment 2 (docs/EXPERIMENT_RECORDS.md#acu-amend2-20261104): the stage-3 planners of
tools/acu/jobs_s3.py (frozen, imported unchanged) with every planned job re-registered as in tools/acu/jobs_amend2.py:
key prefix "r2:", dependencies inside the list renamed, entry points that run PR through the deterministic window sums.
The lists go to jobs/<name>_r2.json (once each) and are appended to the driver.

    python -m tools.acu.jobs_s3r2 plan_a|plan_b
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.acu import jobs_s3 as J3  # noqa: E402
from tools.acu.common import OUT, PY, require  # noqa: E402
from tools.acu.jobs_amend2 import rename  # noqa: E402


def _write_and_add(name, jobs):
    out = OUT / "jobs" / f"{name}_r2.json"
    require(not out.exists(), "job list written once: " + str(out))
    out.write_text(json.dumps(rename(jobs), indent=1) + "\n")
    subprocess.run([PY, str(ROOT / "tools/acu/chain_r2.py"), "add", str(out)], check=True)
    print(len(jobs), "jobs added from", out, flush=True)


def main(argv=None):
    J3._write_and_add = _write_and_add
    J3.main(argv)


if __name__ == "__main__":
    main()
