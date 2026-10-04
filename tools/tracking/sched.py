#!/usr/bin/env python3
"""Semkine job scheduler: one job per GPU, NUMA-local CPU slices, resource accounting, run manifests.

Daemon:   python tools/tracking/sched.py daemon            (run under nohup; stops when sched/STOP exists)
Submit:   python tools/tracking/sched.py submit job.json   (a job object or a list of them)
Status:   python tools/tracking/sched.py status
Program:  SCHED_PROG=<dir> selects another program directory (default outputs/jobs).

A job is a JSON object:
  id        unique string (the run_id for training jobs)
  cmd       argv list, executed in `cwd` under `taskset -c <slice>` and GNU time
  cwd       working directory (a candidate's worktree)
  after     ids that must have finished with exit code 0 first (optional)
  gpu       pin to this GPU index (optional); otherwise the first free allowed GPU
  env       extra environment (optional)
  manifest  path of the run manifest to write (optional; default <sched>/manifests/<id>.json)
  log       stdout/stderr file (optional; default <prog>/logs/<id>.log)
  prio      launch priority (optional, default 0; higher first, then submission order)

Cancel:   python tools/tracking/sched.py cancel ID [ID ...]   (queued jobs only; dependents are skipped)
A restarted daemon re-attaches to jobs still running and reads their exit status from GNU time.

A GPU is free when it is in sched/allowed_gpus, no scheduler job holds it, and nvidia-smi shows no
compute process and < 1 GiB used on it -- so a GPU taken by another session is never shared.
CPU slice of GPU g: 9 physical cores of g's NUMA node and their hyper-thread siblings (18 logical
CPUs), so eight concurrent jobs partition the 72-core / 144-thread machine without overlap.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

#: program directory (queue, state, manifests, logs); `SCHED_PROG` points a second program, e.g.
#: the root-tracking round under outputs/rt, at its own queue with its own queue.
PROG = Path(os.environ.get("SCHED_PROG", str(Path(__file__).resolve().parents[2] / "outputs" / "jobs")))
SCHED = PROG / "sched"
QUEUE = SCHED / "queue.jsonl"
STATE = SCHED / "state.json"
ALLOWED = SCHED / "allowed_gpus"
STOP = SCHED / "STOP"
MANIFESTS = SCHED / "manifests"
CANCELLED = SCHED / "cancelled.txt"
SAMPLE_S = 30
PY = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"


def cpu_slice(g: int) -> list:
    node = 0 if g < 4 else 1
    base = 9 * (g % 4) + (36 if node else 0)
    phys = list(range(base, base + 9))
    return phys + [c + 72 for c in phys]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_write(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    os.replace(tmp, path)


def load_jobs() -> dict:
    jobs = {}
    if QUEUE.exists():
        for line in QUEUE.read_text().splitlines():
            if line.strip():
                j = json.loads(line)
                jobs[j["id"]] = j
    return jobs


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"jobs": {}}


def _allowed_entries() -> list:
    if not ALLOWED.exists():
        return []
    return [x.strip() for x in ALLOWED.read_text().replace("\n", ",").split(",") if x.strip()]


def allowed_gpus() -> list:
    return [int(x.rstrip("s")) for x in _allowed_entries()]


def shared_gpus() -> set:
    """GPUs listed with an `s` suffix (e.g. `2s`): another session's process may be on them, so only a
    job that pins the GPU is placed there, one scheduler job at a time, whatever the memory in use."""
    return {int(x[:-1]) for x in _allowed_entries() if x.endswith("s")}


def smi() -> tuple:
    """(per-GPU {util, mem}, set of GPU indices with a compute process)."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    gpus, uuid2idx = {}, {}
    for line in q.strip().splitlines():
        i, u, util, mem = [x.strip() for x in line.split(",")]
        gpus[int(i)] = {"util": float(util), "mem": float(mem)}
        uuid2idx[u] = int(i)
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    busy = {}
    for line in q.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[1] in uuid2idx:
            busy.setdefault(uuid2idx[parts[1]], []).append(int(parts[0]))
    return gpus, busy


def proc_tree(pid: int) -> list:
    out, todo = [], [pid]
    children = {}
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
                children.setdefault(ppid, []).append(int(d.name))
            except Exception:
                pass
    while todo:
        p = todo.pop()
        out.append(p)
        todo.extend(children.get(p, []))
    return out


def tree_cpu_io(pid: int) -> tuple:
    """CPU seconds (utime+stime) and read bytes summed over the live process tree."""
    tck = os.sysconf("SC_CLK_TCK")
    cpu, rd = 0.0, 0
    for p in proc_tree(pid):
        try:
            f = (Path("/proc") / str(p) / "stat").read_text().rsplit(")", 1)[1].split()
            cpu += (int(f[11]) + int(f[12])) / tck
            for line in (Path("/proc") / str(p) / "io").read_text().splitlines():
                if line.startswith("read_bytes:"):
                    rd += int(line.split()[1])
        except Exception:
            pass
    return cpu, rd


def git_info(cwd: str) -> dict:
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def sha256(path: str):
    import hashlib
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def config_of(cmd: list):
    return cmd[cmd.index("--config") + 1] if "--config" in cmd else None


def launch(job: dict, gpu, st: dict) -> None:
    cpus = list(job["cpus"]) if job.get("device") == "cpu" else cpu_slice(gpu)
    log = Path(job.get("log") or PROG / "logs" / f"{job['id']}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    timef = log.with_suffix(".time")
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu), "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"})
    env.update({k: str(v) for k, v in (job.get("env") or {}).items()})
    argv = ["/usr/bin/time", "-v", "-o", str(timef), "taskset", "-c", ",".join(map(str, cpus))] + job["cmd"]
    cfgp = config_of(job["cmd"])
    if cfgp and not os.path.isabs(cfgp):
        cfgp = os.path.join(job["cwd"], cfgp)
    manifest = {
        "id": job["id"], "kind": job.get("kind", "cmd"), "status": "running", "host": socket.gethostname(),
        "cwd": job["cwd"], "cmd": job["cmd"], "git": git_info(job["cwd"]),
        "config": cfgp, "config_sha256": sha256(cfgp) if cfgp else None,
        "gpu": gpu, "cpus": cpus, "numa_node": (0 if cpus[0] < 36 or 72 <= cpus[0] < 108 else 1),
        "env": {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                     "OPENBLAS_NUM_THREADS") if k in env} | (job.get("env") or {}),
        "log": str(log), "time_file": str(timef), "submitted": job.get("submitted"), "start": now(),
        "meta": job.get("meta", {}),
    }
    with open(log, "ab") as fo:
        fo.write(f"\n===== semkine launch {now()} gpu={gpu} cpus={','.join(map(str, cpus))}\n".encode())
        p = subprocess.Popen(argv, cwd=job["cwd"], env=env, stdout=fo, stderr=subprocess.STDOUT,
                             start_new_session=True)
    st["jobs"][job["id"]] = {"status": "running", "pid": p.pid, "gpu": gpu, "start": now(),
                             "t0": time.time(), "samples": 0, "util_sum": 0.0, "util_max": 0.0,
                             "mem_max": 0.0, "cpu_s": 0.0, "read_bytes": 0, "manifest": str(
                                 Path(job.get("manifest") or MANIFESTS / f"{job['id']}.json"))}
    atomic_write(Path(st["jobs"][job["id"]]["manifest"]), manifest)
    RUNNING[job["id"]] = p
    print(f"[{now()}] launch {job['id']} gpu={gpu} pid={p.pid}", flush=True)


def parse_time_file(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        for key, name in (("User time (seconds)", "user_s"), ("System time (seconds)", "sys_s"),
                          ("Maximum resident set size (kbytes)", "max_rss_kb"),
                          ("File system inputs", "fs_inputs_512b"), ("File system outputs", "fs_outputs_512b"),
                          ("Percent of CPU this job got", "cpu_percent")):
            if line.startswith(key):
                v = line.split(":", 1)[1].strip().rstrip("%")
                try:
                    out[name] = float(v)
                except ValueError:
                    out[name] = v
    return out


def rc_from_time_file(path: Path) -> int:
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("Exit status:"):
                return int(line.split(":", 1)[1])
            if line.startswith("Command terminated by signal"):
                return 128 + int(line.rsplit(" ", 1)[1])
    return -1


ATTACHED: dict = {}


def finish(jid: str, rc: int, st: dict) -> None:
    js = st["jobs"][jid]
    js.update({"status": "done" if rc == 0 else "failed", "rc": rc, "end": now()})
    wall = time.time() - js["t0"]
    mf = Path(js["manifest"])
    man = json.loads(mf.read_text()) if mf.exists() else {}
    tf = parse_time_file(Path(man.get("time_file", ""))) if man else {}
    man.update({"status": js["status"], "rc": rc, "end": js["end"], "wall_s": round(wall, 1),
                "gpu_hours": round(wall / 3600.0, 4),
                "gpu_util_mean": round(js["util_sum"] / max(js["samples"], 1), 1),
                "gpu_util_max": js["util_max"], "gpu_mem_max_mib": js["mem_max"],
                "cpu_seconds_sampled": round(js["cpu_s"], 1), "read_bytes_sampled": js["read_bytes"],
                "gnu_time": tf, "samples": js["samples"]})
    atomic_write(mf, man)
    RUNNING.pop(jid, None)
    ATTACHED.pop(jid, None)
    print(f"[{now()}] finish {jid} rc={rc} wall={wall/60:.1f} min", flush=True)


RUNNING: dict = {}


def daemon() -> None:
    SCHED.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    st = load_state()
    # A restarted daemon cannot wait() on jobs it did not start; mark them lost if dead.
    for jid, js in st["jobs"].items():
        if js.get("status") == "running":
            if Path(f"/proc/{js['pid']}").exists():
                ATTACHED[jid] = js["pid"]
                print(f"[{now()}] re-attached {jid} pid={js['pid']} gpu={js['gpu']}", flush=True)
            else:
                mf = Path(js["manifest"])
                tf = json.loads(mf.read_text()).get("time_file", "") if mf.exists() else ""
                finish(jid, rc_from_time_file(Path(tf)), st)
    atomic_write(STATE, st)
    last_sample = 0.0
    print(f"[{now()}] daemon up pid={os.getpid()}", flush=True)
    while not STOP.exists():
        jobs = load_jobs()
        # reap
        for jid, p in list(RUNNING.items()):
            rc = p.poll()
            if rc is not None:
                finish(jid, rc, st)
        for jid, pid in list(ATTACHED.items()):
            if not Path(f"/proc/{pid}").exists():
                mf = Path(st["jobs"][jid]["manifest"])
                tf = json.loads(mf.read_text()).get("time_file", "") if mf.exists() else ""
                finish(jid, rc_from_time_file(Path(tf)), st)
        gpus, busy = smi()
        # sample
        if time.time() - last_sample >= SAMPLE_S:
            last_sample = time.time()
            for jid, pid in [(j, q.pid) for j, q in RUNNING.items()] + list(ATTACHED.items()):
                js = st["jobs"][jid]
                g = gpus.get(js["gpu"], {"util": 0.0, "mem": 0.0}) if js["gpu"] is not None else {"util": 0.0, "mem": 0.0}
                js["samples"] += 1
                js["util_sum"] += g["util"]
                js["util_max"] = max(js["util_max"], g["util"])
                js["mem_max"] = max(js["mem_max"], g["mem"])
                cpu, rd = tree_cpu_io(pid)
                js["cpu_s"] = max(js["cpu_s"], cpu)
                js["read_bytes"] = max(js["read_bytes"], rd)
            with open(SCHED / "gpu_samples.csv", "a") as f:
                for i, g in sorted(gpus.items()):
                    f.write(f"{int(time.time())},{i},{g['util']},{g['mem']},{len(busy.get(i, []))}\n")
        # launch
        held = {js["gpu"] for js in st["jobs"].values() if js.get("status") == "running" and js.get("gpu") is not None}
        allow = allowed_gpus()
        cancelled = set(CANCELLED.read_text().split()) if CANCELLED.exists() else set()
        # post-processing (selection / evaluation) gates results and is short: it always goes before
        # any training launch, whatever the submitted priorities
        boost = lambda j: 1000 if j.get("kind") in ("select", "eval") else 0              # noqa: E731
        order = sorted(jobs.items(), key=lambda kv: (-(int(kv[1].get("prio", 0)) + boost(kv[1])),
                                                     list(jobs).index(kv[0])))
        for jid, job in order:
            if jid in st["jobs"]:
                continue
            if jid in cancelled:
                st["jobs"][jid] = {"status": "cancelled", "end": now()}
                print(f"[{now()}] cancel {jid}", flush=True)
                continue
            deps = job.get("after") or []
            dstat = [st["jobs"].get(d, {}).get("status") for d in deps]
            if any(s in ("failed", "lost", "skipped", "cancelled") for s in dstat):
                st["jobs"][jid] = {"status": "skipped", "reason": f"dependency {deps} failed", "end": now()}
                print(f"[{now()}] skip {jid}: dependency failed", flush=True)
                continue
            if not all(s == "done" for s in dstat):
                continue
            if job.get("device") == "cpu":
                launch(job, None, st)
                continue
            shared = shared_gpus()
            cands = [job["gpu"]] if job.get("gpu") is not None else [g for g in allow if g not in shared]
            free = [g for g in cands if g in allow and g not in held
                    and (g in shared or (not busy.get(g) and gpus.get(g, {"mem": 1e9})["mem"] < 1024))]
            if not free:
                continue
            launch(job, free[0], st)
            held.add(free[0])
        atomic_write(STATE, st)
        time.sleep(10)
    print(f"[{now()}] STOP file found; daemon exits (running jobs continue unsupervised)", flush=True)


def submit(path: str) -> None:
    obj = json.loads(Path(path).read_text())
    jobs = obj if isinstance(obj, list) else [obj]
    have = load_jobs()
    with open(QUEUE, "a") as f:
        for j in jobs:
            assert j["id"] not in have, f"duplicate job id {j['id']}"
            j.setdefault("submitted", now())
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
            print("submitted", j["id"])


def status() -> None:
    st, jobs = load_state(), load_jobs()
    for jid in jobs:
        js = st["jobs"].get(jid, {"status": "queued"})
        extra = ""
        if js.get("status") == "running":
            extra = f"gpu={js['gpu']} {(time.time() - js.get('t0', time.time())) / 60:.0f} min util~{js.get('util_sum', 0) / max(js.get('samples', 0), 1):.0f}%"
        elif js.get("status") in ("done", "failed"):
            extra = f"rc={js.get('rc')} end={js.get('end')}"
        print(f"{jid:44s} {js['status']:8s} {extra}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "daemon":
        daemon()
    elif cmd == "submit":
        submit(sys.argv[2])
    elif cmd == "cancel":
        with open(CANCELLED, "a") as f:
            for jid in sys.argv[2:]:
                f.write(jid + "\n")
                print("cancel requested", jid)
    else:
        status()
