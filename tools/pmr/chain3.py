#!/usr/bin/env python3
"""PMR driver 3 (docs/EXPERIMENT_RECORDS.md#pmr-amend5-20261109): tools/pmr/chain2.py's scheduler with three changes; state
under outputs/pmr/20261109/$PMR_DRIVER_DIR (default driver3).
  reaping   a stopped gpu-borrow wrapper is reaped (Popen.wait) before its process group is checked. Driver 2 polled the
            group of an exited but unreaped wrapper, read the zombie as alive, raised after 60 s and stopped every job.
            The emergency cleanup reaps before it waits as well.
  re-home   after REHOME_S = 900 s without its command running (was 1800 s)
  order     trainings take a free direct pair first, then a borrowed one: a direct pair starts at once, a borrowed one
            after gpu-borrow's 600 s idle period and its memory admission

    PMR_DRIVER_DIR=driver3 python tools/pmr/chain3.py add|run|drain|close|summary
"""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PMR_DRIVER_DIR", "driver3")

from tools.pmr import chain2 as C2  # noqa: E402
from tools.pmr.common import PY, now, require  # noqa: E402

C = C2.C
C.PAIRS = C2.DIRECT + C2.BORROWED
C2.REHOME_S = 900
STOP_WAIT_S = 60


def stop_group(p):
    """SIGTERM the child's process group, reap the child, SIGKILL what is left after STOP_WAIT_S."""
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        p.wait(timeout=STOP_WAIT_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()


class Chain(C2.Chain):
    def rehome(self, job, pair):
        key = job["key"]
        st = self.state["jobs"][key]
        p = self.children.pop(key)
        stop_group(p)
        require(not C.live_group(p.pid), "the gpu-borrow wrapper of " + key + " did not stop")
        env = dict(os.environ, PYTHONHASHSEED="3407", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   CUDA_VISIBLE_DEVICES=",".join(map(str, pair)), MASTER_PORT=str(29800 + pair[0]),
                   PL_INTER_BATCH_PARALLELISM="0")
        log = (C.LOGS / f"{C2.safe(key)}.rehome.log").open("x")
        q = subprocess.Popen(C2.inner(job), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        log.close()
        self.children[key] = q
        self.waiting.pop(key, None)
        self.last_pair_start = time.monotonic()
        st.update(pid=q.pid, gpus=list(pair), borrowed=False, rehomed_from=st["gpus"], rehomed_at_utc=now(),
                  borrow_exit_code=p.returncode)
        self.publish()

    def cleanup(self):
        for p in self.children.values():
            p.poll()
        super().cleanup()
        for p in self.children.values():
            p.poll()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    require(C.D2.name not in ("driver1", "driver2"), "driver 3 never writes an earlier driver's state")
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


__all__ = ["Chain", "PY", "stop_group"]

if __name__ == "__main__":
    main()
