#!/usr/bin/env python3
"""S45-RT detached driver (docs/EXPERIMENT_RECORDS.md#s45rt-prereg-20261031); physical GPUs 2-7 only.

Same scheduler as tools/s45tr/chain.py (queue order, dependencies, job kinds pair / gpu / exclusive / cpu, failures
preserved and never retried, `STOP` file), with S45-CT's lesson on host memory: a pair (training) job starts only
when the host has >= MIN_FREE_GIB available and no other training started in the last MIN_PAIR_GAP_S seconds (a
training reaches its full footprint over its first minutes).

  chain.py freeze           protocol.json (pre-registration text, file SHAs, references); refuses to overwrite
  chain.py add <jobs.json>  append jobs (never edited or removed)
  chain.py run              the driver; refuses to start unless GPUs 2-7 are idle
  chain.py close            let the driver exit once every job is terminal
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
from tools.s45rt.common import (OUT, PAIRS, PHYSICAL_GPUS, PLAN1_FILES, PROTOCOL, S45_RUN, S45TR, SOURCES, TAG,  # noqa: E402
                                T0_CHECKPOINT, T0_CHECKPOINT_SHA256, now, read, require, save, sha)

QUEUE = OUT / "queue.json"
LOCK = OUT / "queue.lock"
STATUS = OUT / "status.json"
LOGS = OUT / "logs"
STOP = OUT / "STOP"
EVAL_SLOTS_PER_GPU = 2
CPU_SLOTS = 4
KINDS = ("pair", "gpu", "exclusive", "cpu")
TERMINAL = ("EXIT_0", "FAILED_PRESERVED", "SKIPPED_DEP_FAILED")
MIN_FREE_GIB = 280
MIN_PAIR_GAP_S = 1200


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
                                              mem_available_gib_at_start=round(mem_available_gib(), 1))
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

    def pair_allowed(self):
        return (time.monotonic() - self.last_pair_start >= MIN_PAIR_GAP_S or self.last_pair_start == 0.0) and \
            mem_available_gib() >= MIN_FREE_GIB

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
                if free and self.pair_allowed():
                    self.spawn(job, free[0])
                    busy |= set(free[0])
                    pair_used |= set(free[0])
                elif free:
                    reserved |= set(free[0])
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
        from tools.s45rt.common import verify_protocol
        verify_protocol()
        require(not STATUS.exists(), "Fresh-only driver; preserve the previous status")
        LOGS.mkdir(parents=True, exist_ok=True)
        busy = [g for g in PHYSICAL_GPUS if gpu_busy(g)]
        require(not busy, f"GPUs {busy} are not idle; S45-RT does not share 2-7")
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


def freeze():
    require(not PROTOCOL.exists(), "S45-RT protocol already frozen")
    text = (ROOT / "docs/EXPERIMENT_RECORDS.md").read_text()
    begin, end = f"<!-- EXPERIMENT_RECORD:{TAG}:BEGIN -->", f"<!-- EXPERIMENT_RECORD:{TAG}:END -->"
    require(text.count(begin) == 1 and text.count(end) == 1, "write the pre-registration section first")
    OUT.mkdir(parents=True, exist_ok=True)
    section = OUT / "prereg_section.md"
    with section.open("x") as stream:
        stream.write(text[text.index(begin):text.index(end) + len(end)])
    require(sha(T0_CHECKPOINT) == T0_CHECKPOINT_SHA256, "T0 checkpoint changed")
    references = {
        "s45_insample_tl_e1": {str(s): sha(p) for s, p in (
            (5000, ROOT / "outputs/s45ct/20261030/readouts/s45_5000/tl_e1/states.npz"),
            (5500, ROOT / "outputs/s45ct/20261030/readouts/s45_5500/tl_e1/states.npz"),
            (6000, S45TR / "readouts/s45/tl_e1/states.npz"))},
        "t1d_insample_tl_e1": {str(s): sha(p) for s, p in (
            (5000, ROOT / "outputs/s45ct/20261030/readouts/t1d_5000/tl_e1/states.npz"),
            (5500, ROOT / "outputs/s45ct/20261030/readouts/t1d_5500/tl_e1/states.npz"),
            (6000, S45TR / "readouts/t1d/tl_e1/states.npz"))},
        "s45_tl_f1": sha(S45TR / "readouts/s45/tl_f1/states.npz"),
        "t1d_tl_f1": sha(S45TR / "readouts/t1d/tl_f1/states.npz"),
        "s45_formal_receipt": read(ROOT / "outputs/c1sf/20261029/receipts/formal_so3.json")["checkpoint_sha256_by_step"],
        "t1d_formal_receipt": read(S45TR / "receipts/formal_t1d.json")["checkpoint_sha256_by_step"],
        "t0_gate": read(S45TR / "t0/gate.json"),
    }
    save(PROTOCOL, {
        "study": "S45RT_20261031", "status": "FROZEN", "frozen_utc": now(),
        "prereg": "docs/EXPERIMENT_RECORDS.md#s45rt-prereg-20261031", "prereg_section_sha256": sha(section),
        "plan1": {"arm": "t1b", "s45tr_registered": True, "t0_checkpoint_sha256": T0_CHECKPOINT_SHA256, "tau": 0.5},
        "plan2": {"router": "t0p", "arm": "t1p", "frozen_later_in": "manifests/plan2.json"},
        "physical_gpus": PHYSICAL_GPUS, "pairs": PAIRS, "min_free_gib": MIN_FREE_GIB, "min_pair_gap_s": MIN_PAIR_GAP_S,
        "s45_run": str(S45_RUN.relative_to(ROOT)), "references": references,
        "file_sha256": {name: sha(ROOT / name) for name in (*SOURCES, *PLAN1_FILES)},
        "single_seed_descriptive": True,
    }, exclusive=True)
    print(PROTOCOL)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["add"] and len(argv) == 2:
        add(json.loads(Path(argv[1]).read_text()))
        return
    if argv == ["close"]:
        close()
        return
    if argv == ["freeze"]:
        freeze()
        return
    require(argv == ["run"], "usage: chain.py freeze | run | add <jobs.json> | close")

    def interrupted(number, frame):
        raise InterruptedError("S45-RT chain interrupted by signal " + str(number))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    Chain().run()


if __name__ == "__main__":
    main()
