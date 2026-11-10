#!/usr/bin/env python3
"""CMP driver (docs/EXPERIMENT_RECORDS.md#cmp-prereg-20261110): one queue, GPUs 4-7 direct, GPUs 0-3 through gpu-borrow.

    setsid nohup python tools/cmp/chain.py run > outputs/cmp/20261110/driver/driver.log 2>&1 &
    python tools/cmp/chain.py add <jobs.json> | summary | drain

Jobs (queue.json, appended with `add` only; keys unique, dependencies must exist):
  {"key", "kind": train | read | cpu | exclusive, "cmd": [...], "deps": [...], "borrow": bool}
  train      one GPU per training: a free direct GPU, else (borrow true) a free borrowed GPU through
             `gpu-borrow run --gpus g -- tools/cmp/borrow_job.py ... -- <cmd>` (600 s idle before every start, the whole
             job yields to lmx or to any other compute process; borrow_job.py archives a preempted attempt's output and
             re-runs fresh). A borrowed job still waiting for its first start after REHOME_S moves to a free direct GPU.
  read       a direct GPU only (frozen readers need a physical index 4-7): at most one reading per GPU, beside at most
             one training
  cpu        no GPU
  exclusive  the cost block: starts only when no other job runs, on GPU 4, and nothing starts beside it
Dispatch in queue order, skipping jobs whose resources are busy; a job whose dependency failed or was skipped is skipped.
Failures are preserved, never retried. `drain`: no new dispatch, exit once the running jobs end. HOLD file: keep polling
for jobs still to be added when the queue runs empty. STOP file (emergency only): terminate every running job's process
group, record STOPPED_PRESERVED, exit.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.cmp.common import BORROW_GPUS, DIRECT_GPUS, OUT, now, read, require, save  # noqa: E402

DRIVER = OUT / "driver"
QUEUE, STATUS, LOGS = DRIVER / "queue.json", DRIVER / "status.json", DRIVER / "logs"
STOP, DRAIN, HOLD = DRIVER / "STOP", DRIVER / "DRAIN", DRIVER / "HOLD"
GPU_BORROW = str(Path.home() / ".local/bin/gpu-borrow")
POLL_S = 10
REHOME_S = 900
KINDS = ("train", "read", "cpu", "exclusive")
ENV = {"PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "NCCL_P2P_DISABLE": "1",
       "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def safe(key):
    return key.replace(":", "_").replace("/", "_")


def load_queue():
    return read(QUEUE)["jobs"] if QUEUE.exists() else []


def add(path):
    jobs = load_queue()
    keys = {j["key"] for j in jobs}
    new = read(path)["jobs"]
    for j in new:
        require(j["kind"] in KINDS and isinstance(j["cmd"], list) and j["cmd"], "bad job " + json.dumps(j))
        require(j["key"] not in keys, "duplicate key " + j["key"])
        require(all(d in keys for d in j.get("deps", [])), "unknown dependency of " + j["key"])
        keys.add(j["key"])
        jobs.append({"key": j["key"], "kind": j["kind"], "cmd": j["cmd"], "deps": list(j.get("deps", [])),
                     "borrow": bool(j.get("borrow", False)), "added_utc": now()})
    DRIVER.mkdir(parents=True, exist_ok=True)
    save(QUEUE, {"jobs": jobs})
    print(f"added {len(new)} jobs ({len(jobs)} in the queue)")


class Driver:
    def __init__(self):
        self.status = read(STATUS) if STATUS.exists() else {"jobs": {}}
        self.procs = {}                       # key -> (Popen, gpu, borrowed)
        for key, st in self.status["jobs"].items():
            require(st["state"] not in ("RUNNING", "WAITING_BORROW"), "a previous driver left a running job: " + key)
        self.status.update(state="RUNNING", pid=os.getpid(), started_utc=now())
        self.save()

    def save(self):
        self.status["updated_utc"] = now()
        save(STATUS, self.status)

    def st(self, key):
        return self.status["jobs"].setdefault(key, {"state": "PENDING"})

    # ------------------------------------------------------------------ resources
    def usage(self):
        train, reads, excl = {}, {}, False
        for key, (p, gpu, borrowed) in self.procs.items():
            kind = self.job(key)["kind"]
            if kind == "exclusive":
                excl = True
            if gpu is None:
                continue
            if kind == "train":
                train[gpu] = key
            elif kind == "read":
                reads[gpu] = reads.get(gpu, 0) + 1
        return train, reads, excl

    def job(self, key):
        return self._jobs[key]

    def launch(self, job, gpu, borrowed):
        key = job["key"]
        LOGS.mkdir(parents=True, exist_ok=True)
        log = LOGS / f"{safe(key)}.log"
        env = {**os.environ, **ENV}
        cmd = list(job["cmd"])
        if borrowed:
            state = DRIVER / "borrow" / safe(key)
            state.mkdir(parents=True, exist_ok=True)
            os.chmod(state, 0o700)
            env.pop("CUDA_VISIBLE_DEVICES", None)
            cmd = [GPU_BORROW, "run", "--gpus", str(gpu), "--state-dir", str(state), "--",
                   cmd[0], str(ROOT / "tools/cmp/borrow_job.py"), "--key", key, "--attempts",
                   str(DRIVER / "attempts" / f"{safe(key)}.json"), "--log", str(log), "--", *cmd]
            out = (LOGS / f"{safe(key)}.borrow.log").open("ab")
        else:
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
            out = log.open("ab")
        p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        out.close()
        self.procs[key] = (p, gpu, borrowed)
        st = self.st(key)
        st.update(state="WAITING_BORROW" if borrowed else "RUNNING", gpu=gpu, borrowed=borrowed, pid=p.pid,
                  started_utc=now(), started_ts=time.time(), log=str(log))
        print(now(), "start", key, "gpu", gpu, "borrowed" if borrowed else "", flush=True)

    def kill(self, key):
        p, _, _ = self.procs[key]
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(120):
            if p.poll() is not None:
                break
            time.sleep(1)
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()

    # ------------------------------------------------------------------ loop
    def reap(self):
        for key in list(self.procs):
            p, gpu, borrowed = self.procs[key]
            st = self.st(key)
            if borrowed and st["state"] == "WAITING_BORROW":
                if (DRIVER / "attempts" / f"{safe(key)}.json").exists():
                    st["state"] = "RUNNING"
                    st["borrow_started_utc"] = now()
            rc = p.poll()
            if rc is None:
                continue
            del self.procs[key]
            st.update(state="DONE" if rc == 0 else "FAILED", rc=rc, finished_utc=now())
            print(now(), st["state"], key, "rc", rc, flush=True)

    def rehome(self):
        train, _, _ = self.usage()
        free = [g for g in DIRECT_GPUS if g not in train]
        for key in list(self.procs):
            p, gpu, borrowed = self.procs[key]
            st = self.st(key)
            if not (borrowed and st["state"] == "WAITING_BORROW" and free):
                continue
            if time.time() - st["started_ts"] < REHOME_S:
                continue
            self.kill(key)
            del self.procs[key]
            if (DRIVER / "attempts" / f"{safe(key)}.json").exists():       # started meanwhile: let it fail closed
                st.update(state="FAILED", rc=None, finished_utc=now(), note="borrowed start raced the re-home")
                continue
            st.setdefault("rehomed", []).append({"from_gpu": gpu, "at_utc": now()})
            self.launch(self.job(key), free.pop(0), False)

    def dispatch(self):
        if DRAIN.exists():
            return
        train, reads, excl = self.usage()
        if excl:
            return
        for job in self._order:
            key = job["key"]
            st = self.st(key)
            if st["state"] != "PENDING":
                continue
            deps = [self.st(d)["state"] for d in job["deps"]]
            if any(s in ("FAILED", "SKIPPED") for s in deps):
                st.update(state="SKIPPED", finished_utc=now())
                print(now(), "SKIPPED", key, flush=True)
                continue
            if not all(s == "DONE" for s in deps):
                continue
            kind = job["kind"]
            if kind == "exclusive":
                if not self.procs:
                    self.launch(job, DIRECT_GPUS[0], False)
                return
            if kind == "cpu":
                self.launch(job, None, False)
            elif kind == "read":
                gpu = next((g for g in DIRECT_GPUS if reads.get(g, 0) == 0), None)
                if gpu is not None:
                    self.launch(job, gpu, False)
                    reads[gpu] = 1
            else:
                gpu = next((g for g in DIRECT_GPUS if g not in train), None)
                if gpu is not None:
                    self.launch(job, gpu, False)
                    train[gpu] = key
                elif job["borrow"]:
                    gpu = next((g for g in BORROW_GPUS if g not in train), None)
                    if gpu is not None:
                        self.launch(job, gpu, True)
                        train[gpu] = key

    def run(self):
        stopping = {"flag": False}
        signal.signal(signal.SIGTERM, lambda *a: stopping.update(flag=True))
        while True:
            self._order = load_queue()
            self._jobs = {j["key"]: j for j in self._order}
            self.reap()
            if STOP.exists() or stopping["flag"]:
                for key in list(self.procs):
                    self.kill(key)
                    self.st(key).update(state="STOPPED", finished_utc=now())
                    del self.procs[key]
                self.status["state"] = "STOPPED_PRESERVED"
                self.save()
                return
            self.rehome()
            self.dispatch()
            pending = [k for k in self._jobs if self.st(k)["state"] == "PENDING"]
            self.save()
            if not self.procs and ((not pending and not HOLD.exists()) or DRAIN.exists()):
                self.status["state"] = "DRAINED" if pending else "COMPLETE"
                self.save()
                return
            time.sleep(POLL_S)


def summary():
    st = read(STATUS) if STATUS.exists() else {"jobs": {}}
    counts = {}
    for key, s in st["jobs"].items():
        counts[s["state"]] = counts.get(s["state"], 0) + 1
    print(st.get("state"), st.get("updated_utc"), counts)
    for j in load_queue():
        s = st["jobs"].get(j["key"], {"state": "PENDING"})
        if s["state"] != "DONE":
            print(f"  {j['key']:40s} {s['state']:15s} gpu={s.get('gpu')} {s.get('started_utc', '')[:19]}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    verb = argv[0] if argv else ""
    if verb == "add":
        add(Path(argv[1]))
    elif verb == "run":
        Driver().run()
    elif verb == "summary":
        summary()
    elif verb == "drain":
        DRAIN.touch()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
