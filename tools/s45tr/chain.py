#!/usr/bin/env python3
"""S45-TR detached driver (docs/EXPERIMENT_RECORDS.md#s45tr-prereg-20261030); physical GPUs 2-7 only.

The driver runs the jobs of `outputs/s45tr/20261030/queue.json` in queue order as their dependencies finish and
their resources free up. Jobs are appended with `chain.py add <jobs.json>` (never edited or removed), so arms whose
code is written while earlier arms train join the same driver. Job kinds:

  pair       a whole free GPU pair (2-3 / 4-5 / 6-7): one training; MASTER_PORT 29700 + first GPU
  gpu        one evaluation slot (two per GPU) on a GPU no training uses
  exclusive  physical GPU 4 with nothing else running on 2-7 (main rows with latency); starts before new GPU jobs
  cpu        no GPU (acceptance, decisions)

A job starts only when every dependency exited 0; if one failed, the job is SKIPPED_DEP_FAILED. A failed job is
preserved and never retried (pre-registered default). `chain.py run` refuses to start if GPUs 2-7 are not idle;
`chain.py close` lets the driver exit once every job is terminal; touching `STOP` in the output directory stops it.
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
from tools.s45tr.common import OUT, PAIRS, PHYSICAL_GPUS, now, read, require, save  # noqa: E402

PY = sys.executable
QUEUE = OUT / "queue.json"
LOCK = OUT / "queue.lock"
STATUS = OUT / "status.json"
LOGS = OUT / "logs"
STOP = OUT / "STOP"
EVAL_SLOTS_PER_GPU = 2
CPU_SLOTS = 4
KINDS = ("pair", "gpu", "exclusive", "cpu")
TERMINAL = ("EXIT_0", "FAILED_PRESERVED", "SKIPPED_DEP_FAILED")


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
            require(set(job) <= {"key", "kind", "command", "deps", "note"} and job["kind"] in KINDS,
                    "malformed job " + json.dumps(job))
            require(job["key"] not in keys, "duplicate job " + job["key"])
            require(all(d in keys for d in job.get("deps", [])), "unknown dependency in " + job["key"])
            keys.add(job["key"])
            q["jobs"].append({"deps": [], "note": "", **job, "added_at_utc": now()})
        save(QUEUE, q)
    locked(edit)


def close():
    def edit():
        q = read(QUEUE)
        q["closed"] = True
        save(QUEUE, q)
    locked(edit)


def gpu_busy(gpu):
    q = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True, check=True)
    return bool(q.stdout.strip())


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

    def publish(self):
        save(STATUS, self.state)

    def spawn(self, job, gpus):
        env = dict(os.environ, PYTHONHASHSEED="3407", CUDA_DEVICE_ORDER="PCI_BUS_ID")
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        if job["kind"] == "pair":
            env.update(MASTER_PORT=str(29700 + gpus[0]), PL_INTER_BATCH_PARALLELISM="0")
        log = (LOGS / (job["key"].replace(":", "_").replace("/", "_") + ".log")).open("x")
        p = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        log.close()
        self.children[job["key"]] = p
        self.state["jobs"][job["key"]].update(status="RUNNING", pid=p.pid, gpus=list(gpus), started_at_utc=now())
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
        pair_used, slots, exclusive = self.usage()
        busy = pair_used | set(slots) | ({4} if exclusive else set())
        n_cpu = sum(self.state["jobs"][k]["kind"] == "cpu" for k in self.children)
        waiting_exclusive = any(j["kind"] == "exclusive" for j in ready)
        reserved = set()
        for job in ready:
            kind = job["kind"]
            if kind == "cpu":
                if n_cpu < CPU_SLOTS:
                    self.spawn(job, ())
                    n_cpu += 1
                continue
            if kind == "exclusive":
                if not self.children or all(self.state["jobs"][k]["kind"] == "cpu" for k in self.children):
                    self.spawn(job, (4,))
                    return
                continue
            if waiting_exclusive or exclusive:
                continue
            if kind == "pair":
                free = [p for p in PAIRS if not set(p) & (busy | reserved)]
                if free:
                    self.spawn(job, free[0])
                    busy |= set(free[0])
                    pair_used |= set(free[0])
                else:
                    # keep the first pair that frees up for this training: no new evaluation may take it
                    for p in PAIRS:
                        if not set(p) & pair_used:
                            reserved |= set(p)
                            break
                continue
            for gpu in PHYSICAL_GPUS:
                if gpu in pair_used or gpu in reserved or slots.get(gpu, 0) >= EVAL_SLOTS_PER_GPU:
                    continue
                self.spawn(job, (gpu,))
                slots[gpu] = slots.get(gpu, 0) + 1
                busy.add(gpu)
                break
        self.publish()

    def run(self):
        require(not STATUS.exists(), "Fresh-only driver; preserve the previous status")
        LOGS.mkdir(parents=True, exist_ok=True)
        idle = [g for g in PHYSICAL_GPUS if gpu_busy(g)]
        require(not idle, f"GPUs {idle} are not idle; S45-TR does not share 2-7")
        save(OUT / "chain.pid", {"pid": os.getpid(), "started_at_utc": now()})
        try:
            while True:
                if STOP.exists():
                    raise InterruptedError("STOP file present")
                self.reap()
                queue = load_queue()
                self.schedule(queue)
                states = [self.state["jobs"][j["key"]]["status"] for j in queue["jobs"]]
                if queue["closed"] and all(s in TERMINAL for s in states) and not self.children:
                    failed = [k for k, v in self.state["jobs"].items() if v["status"] != "EXIT_0"]
                    self.state.update(status="COMPLETE" if not failed else "COMPLETE_WITH_PRESERVED_FAILURES",
                                      ended_at_utc=now())
                    break
                time.sleep(10)
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


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["add"] and len(argv) == 2:
        add(json.loads(Path(argv[1]).read_text()))
        return
    if argv == ["close"]:
        close()
        return
    require(argv == ["run"], "usage: chain.py run | add <jobs.json> | close")

    def interrupted(number, frame):
        raise InterruptedError("S45-TR chain interrupted by signal " + str(number))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    Chain().run()


if __name__ == "__main__":
    main()
