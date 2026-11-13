#!/usr/bin/env python3
"""PMR driver 7 (docs/EXPERIMENT_RECORDS.md#abl-prereg-20261111): tools/pmr/chain6.py's scheduler with its own state
directory outputs/pmr/20261109/driver7 and the pool GPUs 0-7: direct pairs (4, 5), (6, 7); borrowed pairs only inside
GPUs 0-3 (readings land on 4-7, so a pair mixing a borrowed and a direct card would be yielded by gpu-borrow whenever a
reading starts beside it). Everything else is driver 6's.

Driver 6 (the text below): borrowed and direct cards may form one training pair.

The user authorized mixed pairs on 2026-11-10. The available pool is GPUs
0, 1, 2, 5 and 7; GPUs 3, 4 and 6 remain with CMP. The direct pair (5, 7)
starts first when available. Every pair containing GPU 0-3, including (2, 5),
is wrapped as a whole in gpu-borrow, so preemption stops every DDP rank.
The complete pair must meet gpu-borrow's idle criteria before dispatch.

Driver 4 supplies scheduling, fresh-only attempts, re-homing, resource
admission, and failure preservation. Readings stay on GPUs 5 and 7, and
the CMP cost block holds new dispatches as in driver 5. State is written
only under outputs/pmr/20261109/driver6. A job's optional pairs restriction
is still honored; queue migration records any explicitly authorized edits.
"""
from __future__ import annotations

from itertools import combinations
import json
import os
from pathlib import Path
import pwd
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["PMR_DRIVER_DIR"] = "driver7"
os.environ["PMR_PEER_DRIVERS"] = ""

from tools.pmr import chain4 as C4  # noqa: E402
from tools.pmr.common import now, read, require  # noqa: E402

C = C4.C
C2 = C4.C2
GPU_POOL = (0, 1, 2, 3, 4, 5, 6, 7)
DIRECT = ((4, 5), (6, 7))
BORROWED = tuple(combinations((0, 1, 2, 3), 2))
EVAL_GPUS = (4, 5, 6, 7)
C2.DIRECT, C2.BORROWED, C2.EVAL_GPUS = DIRECT, BORROWED, EVAL_GPUS
C2.BORROW_JOB = str(ROOT / "tools/pmr/borrow_job2.py")
C.PAIRS = DIRECT + BORROWED
C.PHYSICAL_GPUS = EVAL_GPUS
CMP_STATUS = ROOT / "outputs/cmp/20261110/driver/status.json"
CMP_DONE = ("DONE", "FAILED", "SKIPPED")


def probe():
    """Idle status for every candidate card; missing or failed data means busy.

    Both cards of a mixed pair enter gpu-borrow. Applying its strict idle
    criteria to direct candidates as well avoids dispatching a wrapper that
    cannot start because another process is already using its direct card.
    """
    try:
        q = subprocess.run(
            [C2.GPU_BORROW, "probe", "--gpus", ",".join(map(str, GPU_POOL))],
            capture_output=True, text=True, timeout=60, check=True,
        )
        gpus = json.loads(q.stdout)["gpus"]
        priority_uid = pwd.getpwnam(C2.PRIORITY_USER).pw_uid
        return {
            g["index"]: (
                g["used_mib"] <= C2.IDLE_MIB
                and g["utilization"] <= C2.IDLE_UTIL
                and not any(p["uid"] == priority_uid or p["kind"] != "G" for p in g["processes"])
            )
            for g in gpus
        }
    except (subprocess.SubprocessError, OSError, ValueError, KeyError, TypeError):
        return {}


C2.probe = probe


def cmp_cost_block(jobs):
    """Hold dispatch while CMP costs run or only costs and rows remain."""
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
    def rehome_mixed(self, queue):
        """A waiting mixed pair may reuse its own direct card when re-homed.

        The earlier drivers used disjoint borrowed/direct pools. Their free
        pair check excludes every card the current job holds, which would
        prevent (2, 5) from moving to (5, 7) even after GPU 7 becomes free.
        Only this job's cards may overlap; every other holder still blocks.
        """
        if C.DRAIN.exists() or (
            self.last_pair_start and time.monotonic() - self.last_pair_start < C.MIN_PAIR_GAP_S
        ):
            return
        pair_used, slots, exclusive = self.usage()
        peers = C2.peer_jobs()
        for kind, gpus in peers:
            if kind == "pair":
                pair_used |= set(gpus)
            elif kind == "gpu":
                slots[gpus[0]] = slots.get(gpus[0], 0) + 1
            elif kind == "exclusive":
                exclusive = True
        if exclusive:
            return
        by_key = {job["key"]: job for job in queue["jobs"]}
        for key in self.children:
            state = self.state["jobs"][key]
            held = set(state.get("gpus", ()))
            if not state.get("borrowed") or not any(g >= 4 for g in held):
                continue
            if C2.borrow_phase(key) == "RUNNING":
                self.waiting.pop(key, None)
                continue
            since = self.waiting.setdefault(key, time.monotonic())
            if time.monotonic() - since < C2.REHOME_S:
                continue
            job = by_key[key]
            # Pair usage has no counts because valid jobs never overlap.
            # Keep peer cards explicit even if a stale peer record overlaps.
            foreign = (pair_used - held) | set(slots)
            foreign |= {g for _, gpus in peers for g in gpus}
            for pair in C.train_pairs():
                if pair not in DIRECT or set(pair) & foreign:
                    continue
                if job.get("pairs") and list(pair) not in job["pairs"]:
                    continue
                if all(C.gpu_used_gib(g) <= C.FOREIGN_MAX_GIB for g in pair):
                    self.rehome(job, pair)
                    return

    def schedule(self, queue):
        if cmp_cost_block(cmp_jobs()):
            for job in queue["jobs"]:
                self.state["jobs"].setdefault(
                    job["key"], {"status": "PENDING", "kind": job["kind"], "deps": job["deps"]},
                )
            self.state["held_for_cmp_cost_utc"] = now()
            self.publish()
            return
        self.rehome_mixed(queue)
        super().schedule(queue)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    require(C.D2.name == "driver7", "driver 7 writes only its own state")
    if argv != ["run"]:
        return C.main(argv)
    q = subprocess.run(
        ["systemctl", "--user", "show", "--property=Version", "--value"],
        capture_output=True, text=True,
    )
    require(q.returncode == 0, "gpu-borrow needs the user's systemd manager")

    def interrupted(number, frame):
        raise InterruptedError("PMR driver interrupted by signal " + str(number))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    Chain().run()


if __name__ == "__main__":
    main()
