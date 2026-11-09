#!/usr/bin/env python3
"""TPD driver 2 (docs/EXPERIMENT_RECORDS.md#tpd-amend1-20261105), copy 3 (#tpd-amend3-20261105: `run` may start while
the GPUs named in TPD_START_BUSY_OK hold another TPD driver's training); physical GPUs 2-7 only, GPU 0-1 never.

tools/tpd/chain.py with two scheduler fixes and its own state directory outputs/tpd/20261105/$TPD_DRIVER_DIR (default
driver2; driver3 after the restart of #tpd-amend2-20261105):
  gap      the minimum time between two pair starts is evaluated once per scheduling pass (chain.py re-evaluated it per
           job while querying nvidia-smi, so the gap could expire mid-pass and a later job took the slot)
  closed   a pair / gpu job wrapped by a TPD conditional (tools/tpd/maybe.py, maybe_syn.py, maybe_lc.py) whose written rule
           is closed at dispatch time runs as a CPU job (the wrapper only writes its skip record), so it takes neither a
           GPU nor a pair start; an open or undecidable condition dispatches as registered
The jobs' commands are unchanged.

EVF's scheduler (tools/evf/chain.py = PFO2's tools/pfo2/chain.py, logic unchanged) with this study's own queue / status /
logs (outputs/tpd/20261105/{queue.json, status.json, logs/}): queue order, dependencies, job kinds pair (two GPUs, trainings) /
gpu (one GPU, at most EVAL_SLOTS_PER_GPU per GPU) / exclusive (GPU 4 with no other GPU job; cost blocks) / cpu; failures
preserved and never retried; host-memory admission by declared footprints (`mem_gib`); trainings start >=
MIN_PAIR_GAP_S apart.
  DRAIN   the file outputs/tpd/20261105/DRAIN stops dispatching; running jobs run to their end; the driver then exits
          with status DRAINED (pending jobs stay pending). STOP exists for an emergency and terminates running jobs.
  policy  outputs/tpd/20261105/policy.json {"train_pairs": [[2, 3], ...]}, re-read every loop: the pairs a training may
          start on (an operational knob, numerics do not depend on it); a job's optional "pairs" narrows it further;
          gpu jobs use any GPU without a training.
  foreign  other users' / other studies' processes may hold a little memory on GPUs 2-7 (2026-11-05: light labelling
          jobs on 3-6): `run` refuses to start if any of GPUs 2-7 has more than FOREIGN_MAX_GIB in use, a training starts
          only on two GPUs with at most FOREIGN_MAX_GIB in use, an evaluation only on a GPU with at most EVAL_MAX_USED_GIB
          (the only change against IDP's scheduler, which required idle GPUs)
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
from tools.tpd.common import OUT, PAIRS, PHYSICAL_GPUS, now, read, require, save  # noqa: E402

D2 = OUT / os.environ.get("TPD_DRIVER_DIR", "driver2")
QUEUE = D2 / "queue.json"
LOCK = D2 / "queue.lock"
STATUS = D2 / "status.json"
LOGS = D2 / "logs"
STOP = D2 / "STOP"
DRAIN = D2 / "DRAIN"
POLICY = D2 / "policy.json"
WRAPPERS = ("tools.tpd.maybe", "tools.tpd.maybe_syn", "tools.tpd.maybe_lc")
EVAL_SLOTS_PER_GPU = 2
CPU_SLOTS = 8
KINDS = ("pair", "gpu", "exclusive", "cpu")
TERMINAL = ("EXIT_0", "FAILED_PRESERVED", "SKIPPED_DEP_FAILED")
BUDGET_GIB = 440
MARGIN_GIB = 20
MIN_PAIR_GAP_S = 240
DEFAULT_MEM = {"pair": 130, "gpu": 15, "exclusive": 15, "cpu": 4}
EXCLUSIVE_GPU = 4
FOREIGN_MAX_GIB = 6.0
EVAL_MAX_USED_GIB = 30.0


def closed_condition(job):
    """True when the job is a TPD conditional whose written rule is closed now (False if open or undecidable)."""
    cmd = job["command"]
    try:
        i = cmd.index("-m")
    except ValueError:
        return False
    if i + 2 >= len(cmd) or cmd[i + 1] not in WRAPPERS:
        return False
    cond = cmd[i + 2]
    try:
        if cmd[i + 1] == "tools.tpd.maybe_lc":
            return not bool(read(OUT / "decisions" / "lc_screen.json")["screen"][cond[5:]]["swap"])
        import importlib
        return not importlib.import_module(cmd[i + 1]).is_open(cond)
    except Exception:
        return False


def locked(fn):
    D2.mkdir(parents=True, exist_ok=True)
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


def gpu_used_gib(gpu):
    q = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True)
    return float(q.stdout.strip()) / 1024


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
        if job["kind"] == "pair" and gpus:
            env.update(MASTER_PORT=str(29800 + gpus[0]), PL_INTER_BATCH_PARALLELISM="0")
            self.last_pair_start = time.monotonic()
        log = (LOGS / (job["key"].replace(":", "_").replace("/", "_") + ".log")).open("x")
        p = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        log.close()
        self.children[job["key"]] = p
        mem = 4 if self.state["jobs"][job["key"]].get("as_cpu") else job["mem_gib"]
        self.state["jobs"][job["key"]].update(status="RUNNING", pid=p.pid, gpus=list(gpus), started_at_utc=now(),
                                              mem_gib=mem, mem_available_gib_at_start=round(mem_available_gib(), 1))
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
            if job.get("as_cpu"):
                continue
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
        n_cpu = sum(bool(self.state["jobs"][k]["kind"] == "cpu" or self.state["jobs"][k].get("as_cpu"))
                    for k in self.children)
        gap = self.last_pair_start == 0.0 or time.monotonic() - self.last_pair_start >= MIN_PAIR_GAP_S
        waiting_exclusive = any(j["kind"] == "exclusive" for j in ready)
        allowed = train_pairs()
        reserved = set()
        for job in ready:
            kind = job["kind"]
            if kind in ("pair", "gpu") and closed_condition(job):
                if n_cpu < CPU_SLOTS:
                    self.state["jobs"][job["key"]]["as_cpu"] = True
                    self.spawn(job, ())
                    n_cpu += 1
                continue
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
                free = [p for p in mine if not set(p) & (pair_used | set(slots) | reserved)
                        and all(gpu_used_gib(g) <= FOREIGN_MAX_GIB for g in p)]
                if free and gap and self.fits(job):
                    self.spawn(job, free[0])
                    pair_used |= set(free[0])
                    gap = False
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
                if gpu_used_gib(gpu) > EVAL_MAX_USED_GIB:
                    continue
                self.spawn(job, (gpu,))
                slots[gpu] = slots.get(gpu, 0) + 1
                break
        self.publish()

    def run(self):
        require(not STATUS.exists(), "Fresh-only driver; preserve the previous status")
        require(not DRAIN.exists() and not STOP.exists(), "remove a stale DRAIN / STOP only on the user's order")
        LOGS.mkdir(parents=True, exist_ok=True)
        ok_busy = {int(g) for g in os.environ.get("TPD_START_BUSY_OK", "").split(",") if g.strip()}
        busy = [g for g in PHYSICAL_GPUS if g not in ok_busy and gpu_used_gib(g) > FOREIGN_MAX_GIB]
        require(not busy, f"GPUs {busy} hold more than {FOREIGN_MAX_GIB} GiB")
        save(D2 / "chain.pid", {"pid": os.getpid(), "started_at_utc": now()})
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
        raise InterruptedError("TPD driver 2 interrupted by signal " + str(number))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    Chain().run()


if __name__ == "__main__":
    main()
