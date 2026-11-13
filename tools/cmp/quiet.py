#!/usr/bin/env python3
"""Gate of the CMP cost block (docs/EXPERIMENT_RECORDS.md#cmp-amend8-20261110): wait until no compute process of this
user is on any GPU (the CMP driver only knows its own jobs; other studies' drivers run trainings on the same host), so
the exclusive cost readings that depend on this job start on a host that is quiet as far as this user controls it.

    python -m tools.cmp.quiet [--hold 60] [--poll 10]

Exits 0 once the condition has held for `hold` seconds; another user's processes are reported, not waited for.
"""
from __future__ import annotations

import argparse
import getpass
import subprocess
import time


def compute_pids():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                         text=True, check=True).stdout
    return [int(x) for x in out.split() if x.strip().isdigit()]


def owner(pid):
    return subprocess.run(["ps", "-o", "user=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hold", type=float, default=60.0)
    p.add_argument("--poll", type=float, default=10.0)
    a = p.parse_args(argv)
    me, since, last = getpass.getuser(), None, None
    while True:
        owners = {pid: owner(pid) for pid in compute_pids()}
        mine = sorted(pid for pid, u in owners.items() if u == me)
        others = sorted(pid for pid, u in owners.items() if u and u != me)
        state = (tuple(mine), tuple(others))
        if state != last:
            print(time.strftime("%H:%M:%S", time.gmtime()), "own compute pids", mine, "other users'", others, flush=True)
            last = state
        if mine:
            since = None
        else:
            since = since or time.time()
            if time.time() - since >= a.hold:
                print(time.strftime("%H:%M:%S", time.gmtime()), "quiet for", a.hold, "s", flush=True)
                return 0
        time.sleep(a.poll)


if __name__ == "__main__":
    raise SystemExit(main())
