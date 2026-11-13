#!/usr/bin/env python3
"""PMR driver 5 (docs/EXPERIMENT_RECORDS.md#pmr-amend9-20261110; user 2026-11-10 "你现在执行1试试，应该是有空卡"):
tools/pmr/chain4.py's scheduler unchanged, on the cards the comparison study (CMP) leaves free; state under
outputs/pmr/20261109/driver5.
  pairs    direct (5, 7); borrowed (0, 1) through gpu-borrow (from-scratch trainings only; a job's "pairs" narrows it).
           GPUs 3, 4, 6 carry CMP's FastMETRO trainings until about 12:10 UTC and are never used here; GPU 2 has no
           free partner of its own class.
  readings GPUs 5 / 7 only.
  peers    none (drivers 1-4 are stopped; their recorded pids are stale).
  CMP      nothing new starts while CMP's cost block (GPU 4, measured alone) runs or is about to start: a `cost:*` job
           RUNNING, or one PENDING with every other CMP job finished. A job already running here is not stopped.
Everything else (job kinds, DRAIN / STOP, memory admission, pair gap, fresh-only, failures preserved and never retried,
gpu-borrow wrapping, re-homing, reaping) is drivers 2-4's.

    PMR_START_BUSY_OK=7 setsid nohup python tools/pmr/chain5.py run     (add | drain | close | summary as before)
"""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["PMR_DRIVER_DIR"] = "driver5"
os.environ["PMR_PEER_DRIVERS"] = ""

from tools.pmr import chain4 as C4  # noqa: E402
from tools.pmr.common import now, read, require  # noqa: E402

C = C4.C
C2 = C4.C2
DIRECT = ((5, 7),)
BORROWED = ((0, 1),)
EVAL_GPUS = (5, 7)
C2.DIRECT, C2.BORROWED, C2.EVAL_GPUS = DIRECT, BORROWED, EVAL_GPUS
C.PAIRS = DIRECT + BORROWED
C.PHYSICAL_GPUS = EVAL_GPUS
CMP_STATUS = ROOT / "outputs/cmp/20261110/driver/status.json"
CMP_DONE = ("DONE", "FAILED", "SKIPPED")


def cmp_cost_block(jobs):
    """True while CMP's cost block runs or is the only thing CMP has left to start. jobs: {key: {"state": ...}}."""
    cost = [v.get("state") for k, v in jobs.items() if k.startswith("cost:")]
    if any(s == "RUNNING" for s in cost):
        return True
    rest = [v.get("state") for k, v in jobs.items() if not k.startswith(("cost:", "rows:"))]
    return any(s == "PENDING" for s in cost) and all(s in CMP_DONE for s in rest)


def cmp_jobs():
    try:
        return read(CMP_STATUS)["jobs"]
    except (OSError, ValueError, KeyError):
        return {}


class Chain(C4.Chain):
    def schedule(self, queue):
        if cmp_cost_block(cmp_jobs()):
            for job in queue["jobs"]:
                self.state["jobs"].setdefault(job["key"], {"status": "PENDING", "kind": job["kind"], "deps": job["deps"]})
            self.state["held_for_cmp_cost_utc"] = now()
            self.publish()
            return
        super().schedule(queue)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    require(C.D2.name == "driver5", "driver 5 writes only its own state")
    if argv != ["run"]:
        return C.main(argv)
    q = subprocess.run(["systemctl", "--user", "show", "--property=Version", "--value"], capture_output=True, text=True)
    require(q.returncode == 0, "gpu-borrow needs the user's systemd manager")

    def interrupted(number, frame):
        raise InterruptedError("PMR driver interrupted by signal " + str(number))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    Chain().run()


if __name__ == "__main__":
    main()
