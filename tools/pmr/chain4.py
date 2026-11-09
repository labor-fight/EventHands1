#!/usr/bin/env python3
"""PMR driver 4 (docs/EXPERIMENT_RECORDS.md#pmr-amend7-20261109): tools/pmr/chain3.py with exclusive jobs deferred; state
under outputs/pmr/20261109/$PMR_DRIVER_DIR (default driver4).
  deferred  a ready exclusive job (a cost reading on GPU 4) no longer stops the dispatch of other jobs. Driver 3 stopped
            dispatching as soon as S5's cost reading was ready and left GPUs idle for about an hour until every running
            training had ended. Here the cost reading starts when no job of this or a peer driver holds a GPU and no
            training or reading is ready; while it runs nothing else starts, so it is still measured alone.
Everything else is driver 3's (and driver 2's) scheduler unchanged.

    PMR_DRIVER_DIR=driver4 PMR_PEER_DRIVERS=driver3 python tools/pmr/chain4.py add|run|drain|close|summary
"""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PMR_DRIVER_DIR", "driver4")

from tools.pmr import chain3 as C3  # noqa: E402
from tools.pmr.common import now, require  # noqa: E402

C = C3.C
C2 = C3.C2
GPU_KINDS = ("pair", "gpu")


class Chain(C3.Chain):
    def schedule(self, queue):
        excl = [j for j in queue["jobs"] if j["kind"] == "exclusive"]
        rest = [j for j in queue["jobs"] if j["kind"] != "exclusive"]
        super().schedule({**queue, "jobs": rest})
        jobs = self.state["jobs"]
        ready = []
        for job in excl:
            st = jobs.setdefault(job["key"], {"status": "PENDING", "kind": job["kind"], "deps": job["deps"]})
            if st["status"] != "PENDING":
                continue
            deps = [jobs.get(d, {}).get("status") for d in job["deps"]]
            if any(s in ("FAILED_PRESERVED", "SKIPPED_DEP_FAILED") for s in deps):
                st.update(status="SKIPPED_DEP_FAILED", ended_at_utc=now())
            elif all(s == "EXIT_0" for s in deps):
                ready.append(job)
        gpu_ready = any(jobs[j["key"]]["status"] == "PENDING" and j["kind"] in GPU_KINDS
                        and all(jobs.get(d, {}).get("status") == "EXIT_0" for d in j["deps"]) for j in rest)
        if ready and not C.DRAIN.exists() and not self.children and not C2.peer_jobs() and not gpu_ready:
            self.spawn(ready[0], (C.EXCLUSIVE_GPU,))
        self.publish()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    require(C.D2.name not in ("driver1", "driver2", "driver3"), "driver 4 never writes an earlier driver's state")
    if argv != ["run"]:
        return C.main(argv)
    q = subprocess.run(["systemctl", "--user", "show", "--property=Version", "--value"], capture_output=True, text=True)
    require(q.returncode == 0, "gpu-borrow needs the user's systemd manager")
    held = {g for _, gpus in C2.peer_jobs() for g in gpus}
    ok = {g.strip() for g in os.environ.get("PMR_START_BUSY_OK", "").split(",") if g.strip()} | {str(g) for g in held}
    os.environ["PMR_START_BUSY_OK"] = ",".join(sorted(ok))

    def interrupted(number, frame):
        raise InterruptedError("PMR driver interrupted by signal " + str(number))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    Chain().run()


if __name__ == "__main__":
    main()
