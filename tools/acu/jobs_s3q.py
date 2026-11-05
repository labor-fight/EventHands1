#!/usr/bin/env python3
"""ACU PR revision (docs/EXPERIMENT_RECORDS.md#acu-pr2-prereg-20261104): the stage-3 planners of tools/acu/jobs_s3.py
(frozen, imported unchanged) under the namespace "q" (tools/acu/ns_q.py), every planned job re-registered as in
tools/acu/jobs_q.py: key prefix "q:", dependencies inside the list renamed, entry points under the namespace. The lists go
to outputs/acu/20261104q/jobs/<name>_q.json (once each) and are appended to the driver.

    python -m tools.acu.jobs_s3q plan_a|plan_b
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.acu import ns_q  # noqa: E402

ns_q.enter()

from tools.acu import jobs_s3 as J3  # noqa: E402
from tools.acu.common import OUT, PY, require  # noqa: E402
from tools.acu.jobs_q import rename  # noqa: E402


def _write_and_add(name, jobs):
    out = OUT / "jobs" / f"{name}_q.json"
    require(not out.exists(), "job list written once: " + str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rename(jobs), indent=1) + "\n")
    subprocess.run([PY, str(ROOT / "tools/acu/chain_r2.py"), "add", str(out)], check=True)
    print(len(jobs), "jobs added from", out, flush=True)


def main(argv=None):
    J3._write_and_add = _write_and_add
    J3.main(argv)


if __name__ == "__main__":
    main()
