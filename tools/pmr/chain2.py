#!/usr/bin/env python3
"""PMR driver 2 (docs/EXPERIMENT_RECORDS.md#pmr-amend4-20261109; user 2026-11-09: every GPU may be used, GPUs 0-3 only
through `gpu-borrow run`): tools/pmr/chain.py's scheduler (verbs, queue / status / logs, job kinds, DRAIN / STOP,
host-memory admission, pair gap, fresh-only, failures preserved and never retried) on all eight GPUs, with its state under
outputs/pmr/20261109/$PMR_DRIVER_DIR (default driver2).
  borrowed  the pairs (0, 1), (2, 3) take trainings only, each started as
              gpu-borrow run --gpus i,j --state-dir <state>/borrow/<key> -- python tools/pmr/borrow_job.py ... -- <job>
            gpu-borrow starts it once both cards have been idle for 600 s, yields the whole job as soon as lmx (or any
            other compute context) appears on either card, and starts it again after another 600 s idle; borrow_job.py
            archives a stopped attempt's partial run before the fresh re-run. A borrowed pair is offered only while both
            cards look idle to gpu-borrow (no lmx or compute process, <= 256 MiB, <= 5 % utilisation) and no job of this
            or a peer driver holds them. Readings, cost blocks and CPU jobs never use GPUs 0-3 (the frozen reading tools
            take one GPU among 2-7 by index; a borrowed card holds one job).
  re-home   a borrowed job whose command has not been running for REHOME_S (lmx kept the cards) moves to a free direct pair:
            its wrapper is stopped (SIGTERM; gpu-borrow clears its own scope) and borrow_job.py starts the job there.
  direct    the pairs (4, 5), (6, 7) for trainings, evaluations two per GPU on 4-7, the exclusive cost block on GPU 4.
            Trainings take the first free pair in policy order, borrowed pairs first, so GPUs 4-7 stay open for readings
            while lmx is away.
  peers     the GPUs of jobs still RUNNING in a drained driver (PMR_PEER_DRIVERS, default driver1) count as occupied.
  budget    BUDGET_GIB 560 (four trainings declared at 130 GiB; ~55 GiB PSS measured each); MemAvailable >= declared + 20 GiB
            still gates every start.

    PMR_DRIVER_DIR=driver2 python tools/pmr/chain2.py add|run|drain|close|summary
"""
from __future__ import annotations

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
os.environ.setdefault("PMR_DRIVER_DIR", "driver2")

from tools.pmr import chain as C  # noqa: E402
from tools.pmr.common import OUT, PY, now, read, require  # noqa: E402

BORROWED = ((0, 1), (2, 3))
DIRECT = ((4, 5), (6, 7))
EVAL_GPUS = (4, 5, 6, 7)
GPU_BORROW = str(Path.home() / ".local/bin/gpu-borrow")
PRIORITY_USER = "lmx"
IDLE_MIB = 256
IDLE_UTIL = 5
REHOME_S = 1800
PEERS = [OUT / p for p in os.environ.get("PMR_PEER_DRIVERS", "driver1").split(",") if p.strip()]
BORROW_JOB = str(ROOT / "tools/pmr/borrow_job.py")

C.PAIRS = BORROWED + DIRECT
C.PHYSICAL_GPUS = EVAL_GPUS
C.BUDGET_GIB = 560


def safe(key):
    return key.replace(":", "_").replace("/", "_")


def probe():
    """gpu-borrow's view of GPUs 0-3, {index: idle}; a failed probe reads as busy."""
    try:
        q = subprocess.run([GPU_BORROW, "probe", "--gpus", "0,1,2,3"], capture_output=True, text=True, timeout=60,
                           check=True)
        gpus = json.loads(q.stdout)["gpus"]
    except (subprocess.SubprocessError, OSError, ValueError, KeyError):
        return {}
    lmx = pwd.getpwnam(PRIORITY_USER).pw_uid
    return {g["index"]: (g["used_mib"] <= IDLE_MIB and g["utilization"] <= IDLE_UTIL
                         and not any(p["uid"] == lmx or p["kind"] != "G" for p in g["processes"])) for g in gpus}


def peer_jobs():
    """(kind, gpus) of the jobs still running in a peer driver."""
    out = []
    for d in PEERS:
        if not (d / "status.json").exists():
            continue
        for v in read(d / "status.json")["jobs"].values():
            if v["status"] == "RUNNING" and not v.get("as_cpu") and C.live_group(v["pid"]):
                out.append((v["kind"], tuple(v.get("gpus", ()))))
    return out


def borrow_dir(key):
    return C.D2 / "borrow" / safe(key)


def attempts_file(key):
    return C.D2 / "attempts" / f"{safe(key)}.json"


def inner(job):
    """borrow_job.py around a training: the attempt record and the archive of a stopped attempt."""
    return [PY, BORROW_JOB, "--key", job["key"], "--attempts", str(attempts_file(job["key"])), "--log-dir", str(C.LOGS),
            "--", *job["command"]]


def borrow_phase(key):
    """gpu-borrow's phase for this job's wrapper (RUNNING while its command runs)."""
    runs = sorted((borrow_dir(key) / "runs").glob("*/state.json"))
    return read(runs[-1]).get("phase", "WAITING") if runs else "WAITING"


class Chain(C.Chain):
    def __init__(self):
        super().__init__()
        self.waiting = {}

    def spawn(self, job, gpus):
        if tuple(gpus) not in BORROWED:
            return super().spawn(job, gpus)
        require(job["kind"] == "pair", "a borrowed pair takes trainings only")
        d = borrow_dir(job["key"])
        d.mkdir(parents=True, exist_ok=False)
        d.chmod(0o700)
        wrapped = [GPU_BORROW, "run", "--gpus", ",".join(map(str, gpus)), "--state-dir", str(d), "--", *inner(job)]
        super().spawn({**job, "command": wrapped}, gpus)
        self.state["jobs"][job["key"]].update(borrowed=True, borrow_state_dir=str(d))
        self.waiting[job["key"]] = time.monotonic()
        self.publish()

    def rehome(self, job, pair):
        key = job["key"]
        st = self.state["jobs"][key]
        p = self.children.pop(key)
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 60
        while C.live_group(p.pid) and time.monotonic() < deadline:
            time.sleep(0.5)
        require(not C.live_group(p.pid), "the gpu-borrow wrapper of " + key + " did not stop")
        p.wait()
        env = dict(os.environ, PYTHONHASHSEED="3407", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   CUDA_VISIBLE_DEVICES=",".join(map(str, pair)), MASTER_PORT=str(29800 + pair[0]),
                   PL_INTER_BATCH_PARALLELISM="0")
        log = (C.LOGS / f"{safe(key)}.rehome.log").open("x")
        q = subprocess.Popen(inner(job), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        self.children[key] = q
        self.waiting.pop(key, None)
        self.last_pair_start = time.monotonic()
        st.update(pid=q.pid, gpus=list(pair), borrowed=False, rehomed_from=st["gpus"], rehomed_at_utc=now(),
                  borrow_exit_code=p.returncode)
        self.publish()

    def schedule(self, queue):
        jobs = self.state["jobs"]
        for job in queue["jobs"]:
            jobs.setdefault(job["key"], {"status": "PENDING", "kind": job["kind"], "deps": job["deps"]})
        ready = []
        for job in queue["jobs"]:
            st = jobs[job["key"]]
            if st["status"] != "PENDING":
                continue
            deps = [jobs.get(d, {}).get("status") for d in job["deps"]]
            if any(s in ("FAILED_PRESERVED", "SKIPPED_DEP_FAILED") for s in deps):
                st.update(status="SKIPPED_DEP_FAILED", ended_at_utc=now())
                continue
            if all(s == "EXIT_0" for s in deps):
                ready.append(job)
        if C.DRAIN.exists():
            self.publish()
            return
        pair_used, slots, exclusive = self.usage()
        peers = peer_jobs()
        for kind, gpus in peers:
            if kind == "pair":
                pair_used |= set(gpus)
            elif kind == "gpu":
                slots[gpus[0]] = slots.get(gpus[0], 0) + 1
            elif kind == "exclusive":
                exclusive = True
        n_cpu = sum(bool(jobs[k]["kind"] == "cpu" or jobs[k].get("as_cpu")) for k in self.children)
        gap = self.last_pair_start == 0.0 or time.monotonic() - self.last_pair_start >= C.MIN_PAIR_GAP_S
        allowed = C.train_pairs()
        reserved = set()
        idle = {}
        if any(j["kind"] == "pair" for j in ready) and any(not set(p) & pair_used for p in BORROWED if p in allowed):
            idle = probe()

        def free_pair(job, borrowed=True):
            for p in allowed:
                if job.get("pairs") and list(p) not in job["pairs"]:
                    continue
                if set(p) & (pair_used | set(slots) | reserved):
                    continue
                if p in BORROWED and borrowed and all(idle.get(g) for g in p):
                    return p
                if p in DIRECT and all(C.gpu_used_gib(g) <= C.FOREIGN_MAX_GIB for g in p):
                    return p
            return None

        by_key = {j["key"]: j for j in queue["jobs"]}
        for key in [k for k in self.children if jobs[k].get("borrowed")]:
            if borrow_phase(key) == "RUNNING":
                self.waiting.pop(key, None)
                continue
            since = self.waiting.setdefault(key, time.monotonic())
            if not gap or time.monotonic() - since < REHOME_S:
                continue
            job = by_key[key]
            pair = free_pair(job, borrowed=False)
            if pair is not None:
                old = set(jobs[key]["gpus"])
                self.rehome(job, pair)
                pair_used = (pair_used - old) | set(pair)
                gap = False
        waiting_exclusive = any(j["kind"] == "exclusive" for j in ready)
        for job in ready:
            kind = job["kind"]
            if kind == "cpu":
                if n_cpu < C.CPU_SLOTS and not exclusive and self.fits(job):
                    self.spawn(job, ())
                    n_cpu += 1
                continue
            if kind == "exclusive":
                if not self.children and not peers:
                    self.spawn(job, (C.EXCLUSIVE_GPU,))
                    return
                continue
            if waiting_exclusive or exclusive:
                continue
            if kind == "pair":
                pair = free_pair(job)
                if pair is not None and gap and self.fits(job):
                    self.spawn(job, pair)
                    pair_used |= set(pair)
                    gap = False
                elif pair is None or pair in DIRECT:
                    # keep the first direct pair without a training for this one: no new evaluation may take it
                    for p in allowed:
                        if p in DIRECT and (not job.get("pairs") or list(p) in job["pairs"]) and not set(p) & pair_used:
                            reserved |= set(p)
                            break
                continue
            if not self.fits(job):
                continue
            for gpu in EVAL_GPUS:
                if gpu in pair_used or gpu in reserved or slots.get(gpu, 0) >= C.EVAL_SLOTS_PER_GPU:
                    continue
                if C.gpu_used_gib(gpu) > C.EVAL_MAX_USED_GIB:
                    continue
                self.spawn(job, (gpu,))
                slots[gpu] = slots.get(gpu, 0) + 1
                break
        self.publish()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    require(C.D2.name != "driver1", "driver 2 never writes driver 1's state")
    if argv != ["run"]:
        return C.main(argv)
    q = subprocess.run(["systemctl", "--user", "show", "--property=Version", "--value"], capture_output=True, text=True)
    require(q.returncode == 0, "gpu-borrow needs the user's systemd manager")
    held = {g for _, gpus in peer_jobs() for g in gpus}
    ok = {g.strip() for g in os.environ.get("PMR_START_BUSY_OK", "").split(",") if g.strip()} | {str(g) for g in held}
    os.environ["PMR_START_BUSY_OK"] = ",".join(sorted(ok))

    def interrupted(number, frame):
        raise InterruptedError("PMR driver interrupted by signal " + str(number))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    Chain().run()


if __name__ == "__main__":
    main()
