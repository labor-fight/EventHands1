#!/usr/bin/env python3
"""ACU driver (docs/EXPERIMENT_RECORDS.md#acu-s1-prereg-20261104); physical GPUs 2-7 only, GPU 0-1 never.

B2UB's scheduler (tools/b2ub/chain.py = EVF's = PFO2's, logic unchanged) with this study's own queue / status /
logs (outputs/acu/20261104/{queue.json, status.json, logs/}): queue order, dependencies, job kinds pair (two GPUs, trainings) /
gpu (one GPU, at most EVAL_SLOTS_PER_GPU per GPU) / exclusive (GPU 4 with no other GPU job; cost blocks) / cpu; failures
preserved and never retried; host-memory admission by declared footprints (`mem_gib`); trainings start >=
MIN_PAIR_GAP_S apart.
  DRAIN   the file outputs/acu/20261104/DRAIN stops dispatching; running jobs run to their end; the driver then exits
          with status DRAINED (pending jobs stay pending). STOP exists for an emergency and terminates running jobs.
  policy  outputs/acu/20261104/policy.json {"train_pairs": [[2, 3], ...]}, re-read every loop: the pairs a training may
          start on (an operational knob, numerics do not depend on it); a job's optional "pairs" narrows it further;
          gpu jobs use any GPU without a training.
Started detached (setsid nohup), never inside a session task. Fresh-only: a second `run` refuses an existing status.
Jobs are appended stage by stage (never all seeds at once).

  chain.py add <jobs.json>  append jobs (never edited or removed)
  chain.py run              the driver; refuses to start unless GPUs 2-7 are idle
  chain.py drain            stop dispatching, let running jobs finish, exit DRAINED
  chain.py close            let the driver exit once every job is terminal
  chain.py summary          job counts by status, running jobs
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.acu.common import OUT, PAIRS, PHYSICAL_GPUS, now, read, require, save  # noqa: E402

QUEUE = OUT / "queue.json"
LOCK = OUT / "queue.lock"
STATUS = OUT / "status.json"
LOGS = OUT / "logs"
STOP = OUT / "STOP"
DRAIN = OUT / "DRAIN"
POLICY = OUT / "policy.json"
EVAL_SLOTS_PER_GPU = 2
CPU_SLOTS = 8
KINDS = ("pair", "gpu", "exclusive", "cpu")
TERMINAL = ("EXIT_0", "FAILED_PRESERVED", "SKIPPED_DEP_FAILED")
BUDGET_GIB = 440
MARGIN_GIB = 20
MIN_PAIR_GAP_S = 240
DEFAULT_MEM = {"pair": 130, "gpu": 15, "exclusive": 15, "cpu": 4}
EXCLUSIVE_GPU = 4


def locked(fn):
    OUT.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_queue():
    return locked(lambda: read(QUEUE) if QUEUE.exists() else {"closed": False, "jobs": []})


def add(jobs):
    def edit():
        q = read(QUEUE) if QUEUE.exists() else {"closed": False, "jobs": []}
        require(not q["closed"], "queue closed")
        keys = {j["key"] for j in q["jobs"]}
        for job in jobs:
            require(set(job) <= {"key", "kind", "command", "deps", "note", "mem_gib", "pairs"} and job["kind"] in KINDS,
                    "malformed job " + json.dumps(job))
            require(job["key"] not in keys, "duplicate job " + job["key"])
            require(all(d in keys for d in job.get("deps", [])), "unknown dependency in " + job["key"])
            keys.add(job["key"])
            q["jobs"].append({"deps": [], "note": "", "mem_gib": DEFAULT_MEM[job["kind"]], **job, "added_at_utc": now()})
        save(QUEUE, q)
    locked(edit)


def close():
    def edit():
        q = read(QUEUE)
        q["closed"] = True
        save(QUEUE, q)
    locked(edit)


def train_pairs():
    """The pairs a training may start on (policy.json; default every pair)."""
    if not POLICY.exists():
        return list(PAIRS)
    try:
        pairs = [tuple(p) for p in read(POLICY)["train_pairs"]]
    except (ValueError, KeyError, TypeError):
        return list(PAIRS)
    return [p for p in PAIRS if p in pairs]


def gpu_busy(gpu):
    q = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True, check=True)
    return bool(q.stdout.strip())


def mem_available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def live_group(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


class Chain:
    def __init__(self):
        self.state = {"status": "RUNNING", "started_at_utc": now(), "pid": os.getpid(), "jobs": {}}
        self.children = {}
        self.last_pair_start = 0.0

    def publish(self):
        save(STATUS, self.state)

    def spawn(self, job, gpus):
        env = dict(os.environ, PYTHONHASHSEED="3407", CUDA_DEVICE_ORDER="PCI_BUS_ID")
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus)) if gpus else ""
        if job["kind"] == "pair":
            env.update(MASTER_PORT=str(29800 + gpus[0]), PL_INTER_BATCH_PARALLELISM="0")
            self.last_pair_start = time.monotonic()
        log = (LOGS / (job["key"].replace(":", "_").replace("/", "_") + ".log")).open("x")
        p = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        log.close()
        self.children[job["key"]] = p
        self.state["jobs"][job["key"]].update(status="RUNNING", pid=p.pid, gpus=list(gpus), started_at_utc=now(),
                                              mem_gib=job["mem_gib"], mem_available_gib_at_start=round(mem_available_gib(), 1))
        self.publish()

    def reap(self):
        for key, p in list(self.children.items()):
            code = p.poll()
            if code is None or live_group(p.pid):
                continue
            del self.children[key]
            self.state["jobs"][key].update(status="EXIT_0" if code == 0 else "FAILED_PRESERVED", exit_code=code,
                                           ended_at_utc=now())
            self.publish()

    def usage(self):
        pairs, slots = set(), {}
        exclusive = False
        for key in self.children:
            job = self.state["jobs"][key]
            if job["kind"] == "pair":
                pairs |= set(job["gpus"])
            elif job["kind"] == "gpu":
                slots[job["gpus"][0]] = slots.get(job["gpus"][0], 0) + 1
            elif job["kind"] == "exclusive":
                exclusive = True
        return pairs, slots, exclusive

    def committed(self):
        return sum(self.state["jobs"][k].get("mem_gib", 0) for k in self.children)

    def fits(self, job):
        mem = job["mem_gib"]
        return self.committed() + mem <= BUDGET_GIB and mem_available_gib() >= mem + MARGIN_GIB

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
        if DRAIN.exists():
            self.publish()
            return
        pair_used, slots, exclusive = self.usage()
        n_cpu = sum(self.state["jobs"][k]["kind"] == "cpu" for k in self.children)
        waiting_exclusive = any(j["kind"] == "exclusive" for j in ready)
        allowed = train_pairs()
        reserved = set()
        for job in ready:
            kind = job["kind"]
            if kind == "cpu":
                if n_cpu < CPU_SLOTS and not exclusive and self.fits(job):
                    self.spawn(job, ())
                    n_cpu += 1
                continue
            if kind == "exclusive":
                if not self.children:
                    self.spawn(job, (EXCLUSIVE_GPU,))
                    return
                continue
            if waiting_exclusive or exclusive:
                continue
            if kind == "pair":
                mine = [p for p in allowed if not job.get("pairs") or list(p) in job["pairs"]]
                free = [p for p in mine if not set(p) & (pair_used | set(slots) | reserved)]
                gap = self.last_pair_start == 0.0 or time.monotonic() - self.last_pair_start >= MIN_PAIR_GAP_S
                if free and gap and self.fits(job):
                    self.spawn(job, free[0])
                    pair_used |= set(free[0])
                else:
                    # keep the first allowed pair without a training for this one: no new evaluation may take it
                    for p in mine:
                        if not set(p) & pair_used:
                            reserved |= set(p)
                            break
                continue
            if not self.fits(job):
                continue
            for gpu in PHYSICAL_GPUS:
                if gpu in pair_used or gpu in reserved or slots.get(gpu, 0) >= EVAL_SLOTS_PER_GPU:
                    continue
                self.spawn(job, (gpu,))
                slots[gpu] = slots.get(gpu, 0) + 1
                break
        self.publish()

    def run(self):
        require(not STATUS.exists(), "Fresh-only driver; preserve the previous status")
        require(not DRAIN.exists() and not STOP.exists(), "remove a stale DRAIN / STOP only on the user's order")
        LOGS.mkdir(parents=True, exist_ok=True)
        busy = [g for g in PHYSICAL_GPUS if gpu_busy(g)]
        require(not busy, f"GPUs {busy} are not idle")
        save(OUT / "chain.pid", {"pid": os.getpid(), "started_at_utc": now()})
        try:
            while True:
                if STOP.exists():
                    raise InterruptedError("STOP file present")
                self.reap()
                queue = load_queue()
                self.schedule(queue)
                if DRAIN.exists() and not self.children:
                    self.state.update(status="DRAINED", ended_at_utc=now())
                    break
                states = [self.state["jobs"][j["key"]]["status"] for j in queue["jobs"]]
                if queue["closed"] and all(s in TERMINAL for s in states) and not self.children:
                    failed = [k for k, v in self.state["jobs"].items() if v["status"] != "EXIT_0"]
                    self.state.update(status="COMPLETE" if not failed else "COMPLETE_WITH_PRESERVED_FAILURES",
                                      ended_at_utc=now())
                    break
                time.sleep(5)
        except BaseException as exc:
            self.cleanup()
            self.state.update(status="STOPPED_PRESERVED", error=f"{type(exc).__name__}: {exc}", ended_at_utc=now())
            raise
        finally:
            self.publish()

    def cleanup(self):
        for p in self.children.values():
            if live_group(p.pid):
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 60
        while any(live_group(p.pid) for p in self.children.values()) and time.monotonic() < deadline:
            time.sleep(1)
        for p in self.children.values():
            if live_group(p.pid):
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def summary():
    st = read(STATUS) if STATUS.exists() else {"status": "NOT_STARTED", "jobs": {}}
    counts = {}
    for v in st["jobs"].values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    print(st["status"], counts)
    for k, v in st["jobs"].items():
        if v["status"] == "RUNNING":
            print("  RUNNING", k, v.get("gpus"), v.get("started_at_utc"))
        elif v["status"] in ("FAILED_PRESERVED", "SKIPPED_DEP_FAILED"):
            print(" ", v["status"], k)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["add"] and len(argv) == 2:
        add(json.loads(Path(argv[1]).read_text()))
        return
    if argv == ["close"]:
        close()
        return
    if argv == ["drain"]:
        DRAIN.touch()
        return
    if argv == ["summary"]:
        summary()
        return
    require(argv == ["run"], "usage: chain.py run | add <jobs.json> | drain | close | summary")

    def interrupted(number, frame):
        raise InterruptedError("B2UB chain interrupted by signal " + str(number))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    Chain().run()


if __name__ == "__main__":
    main()
