#!/usr/bin/env python3
"""Semkine job scheduler, slot version (DT2 round): several jobs per GPU, CPU cores from a machine-wide pool.

Differences from `tools/tracking/sched.py` (one job per GPU, a fixed 9-core slice per GPU):
  * `sched/slots.json` (re-read every loop) gives, per GPU, how many *training* jobs it may hold (`cap`) and a
    memory ceiling (`mem_limit_mib`); evaluation / selection jobs do not count against `cap`, only against memory.
  * A job declares `cores` (physical cores; each comes with its hyper-thread sibling c+72) and `mem_mib` (its GPU
    memory estimate). Cores are taken from the free pool, NUMA-local to the GPU first, then from the other node.
    `slots.json: reserve` lists physical cores never handed out (interactive work).
  * A GPU accepts a job when (our training jobs on it) < cap and
    foreign memory (compute processes that are not ours) + the estimates of our jobs on it + the new estimate
    <= mem_limit_mib, so a GPU shared with another session is used only up to its free memory.
  * The command may contain the placeholder `{WORKERS}`, replaced by 2 * cores - 2 (min 1).
  * `slots.json: env` is added to every launched job's environment (a job's own `env` overrides it); the DT2 round
    uses it for SEMKINE_FAST_PIPE=1 (bit-identical fast data path).

Daemon:   SCHED_PROG=outputs/dt2 python tools/tracking/sched2.py daemon      (stops when sched/STOP exists)
Submit:   SCHED_PROG=outputs/dt2 python tools/tracking/sched2.py submit job.json
Status:   SCHED_PROG=outputs/dt2 python tools/tracking/sched2.py status
Cancel:   SCHED_PROG=outputs/dt2 python tools/tracking/sched2.py cancel ID [ID ...]   (queued jobs; dependents skip)
Kill:     SCHED_PROG=outputs/dt2 python tools/tracking/sched2.py kill ID [ID ...]     (running jobs; SIGTERM the tree)
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

PROG = Path(os.environ.get("SCHED_PROG", str(Path(__file__).resolve().parents[2] / "outputs" / "dt2")))
SCHED = PROG / "sched"
QUEUE = SCHED / "queue.jsonl"
STATE = SCHED / "state.json"
SLOTS = SCHED / "slots.json"
STOP = SCHED / "STOP"
MANIFESTS = SCHED / "manifests"
CANCELLED = SCHED / "cancelled.txt"
KILLS = SCHED / "kill.txt"
SAMPLE_S = 30
N_PHYS = 72
DEFAULTS = {"train": (5, 11000), "eval": (1, 1500), "select": (1, 1500), "probe": (2, 3000), "cmd": (1, 1500)}


def node_of(core: int) -> int:
    return 0 if (core % 72) < 36 else 1


def gpu_node(g: int) -> int:
    return 0 if g < 4 else 1


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


def load_slots() -> dict:
    if not SLOTS.exists():
        return {"gpus": {}, "reserve": []}
    s = json.loads(SLOTS.read_text())
    s.setdefault("gpus", {})
    s.setdefault("reserve", [])
    return s


def smi() -> tuple:
    """(per-GPU {util, mem}, {gpu: [(pid, mib), ...]} of compute processes)."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    gpus, uuid2idx = {}, {}
    for line in q.strip().splitlines():
        i, u, util, mem = [x.strip() for x in line.split(",")]
        gpus[int(i)] = {"util": float(util), "mem": float(mem)}
        uuid2idx[u] = int(i)
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    procs = {}
    for line in q.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3 and parts[1] in uuid2idx:
            try:
                procs.setdefault(uuid2idx[parts[1]], []).append((int(parts[0]), float(parts[2])))
            except ValueError:
                pass
    return gpus, procs


def children_map() -> dict:
    children = {}
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
                children.setdefault(ppid, []).append(int(d.name))
            except Exception:
                pass
    return children


def proc_tree(pid: int, children: dict) -> list:
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        todo.extend(children.get(p, []))
    return out


def tree_cpu(pids: list) -> float:
    tck = os.sysconf("SC_CLK_TCK")
    cpu = 0.0
    for p in pids:
        try:
            f = (Path("/proc") / str(p) / "stat").read_text().rsplit(")", 1)[1].split()
            cpu += (int(f[11]) + int(f[12])) / tck
        except Exception:
            pass
    return cpu


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


def job_kind(job: dict) -> str:
    return job.get("kind", "cmd")


def job_cores(job: dict) -> int:
    return int(job.get("cores", DEFAULTS.get(job_kind(job), (1, 1500))[0]))


def job_mem(job: dict) -> float:
    return float(job.get("mem_mib", DEFAULTS.get(job_kind(job), (1, 1500))[1]))


def pick_cores(n: int, node: int, used: set, reserve: set):
    free = [c for c in range(N_PHYS) if c not in used and c not in reserve]
    local = [c for c in free if node_of(c) == node]
    other = [c for c in free if node_of(c) != node]
    take = (local + other)[:n]
    return take if len(take) == n else None


RUNNING: dict = {}
ATTACHED: dict = {}


def launch(job: dict, gpu, phys: list, st: dict) -> None:
    cpus = sorted(phys) + sorted(c + 72 for c in phys)
    log = Path(job.get("log") or PROG / "logs" / f"{job['id']}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    timef = log.with_suffix(".time")
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu), "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"})
    env.update({k: str(v) for k, v in (load_slots().get("env") or {}).items()})   # global (slots.json "env")
    env.update({k: str(v) for k, v in (job.get("env") or {}).items()})             # the job's own wins
    workers = max(1, 2 * len(phys) - 2)
    cmd = [str(x).replace("{WORKERS}", str(workers)) for x in job["cmd"]]
    argv = ["/usr/bin/time", "-v", "-o", str(timef), "taskset", "-c", ",".join(map(str, cpus))] + cmd
    cfgp = config_of(cmd)
    if cfgp and not os.path.isabs(cfgp):
        cfgp = os.path.join(job["cwd"], cfgp)
    manifest = {
        "id": job["id"], "kind": job_kind(job), "status": "running", "host": socket.gethostname(),
        "cwd": job["cwd"], "cmd": cmd, "git": git_info(job["cwd"]),
        "config": cfgp, "config_sha256": sha256(cfgp) if cfgp else None,
        "gpu": gpu, "cpus": cpus, "phys_cores": sorted(phys), "workers": workers,
        "numa_nodes": sorted({node_of(c) for c in phys}), "mem_estimate_mib": job_mem(job),
        "env": {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "SEMKINE_FAST_PIPE") if k in env} | (job.get("env") or {}),
        "log": str(log), "time_file": str(timef), "submitted": job.get("submitted"), "start": now(),
        "meta": job.get("meta", {}),
    }
    with open(log, "ab") as fo:
        fo.write(f"\n===== sched2 launch {now()} gpu={gpu} cpus={','.join(map(str, cpus))} workers={workers}\n".encode())
        p = subprocess.Popen(argv, cwd=job["cwd"], env=env, stdout=fo, stderr=subprocess.STDOUT,
                             start_new_session=True)
    st["jobs"][job["id"]] = {"status": "running", "pid": p.pid, "gpu": gpu, "phys": sorted(phys),
                             "kind": job_kind(job), "mem_est": job_mem(job), "start": now(),
                             "t0": time.time(), "samples": 0, "util_sum": 0.0, "util_max": 0.0,
                             "mem_max": 0.0, "own_mem_max": 0.0, "cpu_s": 0.0,
                             "manifest": str(Path(job.get("manifest") or MANIFESTS / f"{job['id']}.json"))}
    atomic_write(Path(st["jobs"][job["id"]]["manifest"]), manifest)
    RUNNING[job["id"]] = p
    print(f"[{now()}] launch {job['id']} gpu={gpu} cores={sorted(phys)} pid={p.pid}", flush=True)


def parse_time_file(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        for key, name in (("User time (seconds)", "user_s"), ("System time (seconds)", "sys_s"),
                          ("Maximum resident set size (kbytes)", "max_rss_kb"),
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


def finish(jid: str, rc: int, st: dict) -> None:
    js = st["jobs"][jid]
    js.update({"status": "done" if rc == 0 else "failed", "rc": rc, "end": now()})
    wall = time.time() - js["t0"]
    mf = Path(js["manifest"])
    man = json.loads(mf.read_text()) if mf.exists() else {}
    tf = parse_time_file(Path(man.get("time_file", ""))) if man else {}
    man.update({"status": js["status"], "rc": rc, "end": js["end"], "wall_s": round(wall, 1),
                "gpu_util_mean": round(js["util_sum"] / max(js["samples"], 1), 1),
                "gpu_util_max": js["util_max"], "gpu_mem_max_mib": js["mem_max"],
                "own_gpu_mem_max_mib": js.get("own_mem_max", 0.0),
                "cpu_seconds_sampled": round(js["cpu_s"], 1), "gnu_time": tf, "samples": js["samples"]})
    atomic_write(mf, man)
    RUNNING.pop(jid, None)
    ATTACHED.pop(jid, None)
    print(f"[{now()}] finish {jid} rc={rc} wall={wall/60:.1f} min", flush=True)


def daemon() -> None:
    SCHED.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    st = load_state()
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
    print(f"[{now()}] sched2 daemon up pid={os.getpid()}", flush=True)
    while not STOP.exists():
        jobs = load_jobs()
        for jid, p in list(RUNNING.items()):
            rc = p.poll()
            if rc is not None:
                finish(jid, rc, st)
        for jid, pid in list(ATTACHED.items()):
            if not Path(f"/proc/{pid}").exists():
                mf = Path(st["jobs"][jid]["manifest"])
                tf = json.loads(mf.read_text()).get("time_file", "") if mf.exists() else ""
                finish(jid, rc_from_time_file(Path(tf)), st)
        # kill requests
        if KILLS.exists():
            for jid in KILLS.read_text().split():
                pid = st["jobs"].get(jid, {}).get("pid")
                if st["jobs"].get(jid, {}).get("status") == "running" and pid:
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGTERM)
                        print(f"[{now()}] SIGTERM {jid} pid={pid}", flush=True)
                    except Exception as e:
                        print(f"[{now()}] kill {jid} failed: {e}", flush=True)
            KILLS.unlink()
        gpus, procs = smi()
        children = children_map()
        running = {j: js for j, js in st["jobs"].items() if js.get("status") == "running"}
        ours = {}
        for jid, js in running.items():
            for p in proc_tree(js["pid"], children):
                ours[p] = jid
        own_mem = {}
        foreign = {}
        for g, lst in procs.items():
            for pid, mib in lst:
                if pid in ours:
                    own_mem[ours[pid]] = own_mem.get(ours[pid], 0.0) + mib
                else:
                    foreign[g] = foreign.get(g, 0.0) + mib
        if time.time() - last_sample >= SAMPLE_S:
            last_sample = time.time()
            for jid, js in running.items():
                g = gpus.get(js["gpu"], {"util": 0.0, "mem": 0.0}) if js["gpu"] is not None else {"util": 0.0, "mem": 0.0}
                js["samples"] += 1
                js["util_sum"] += g["util"]
                js["util_max"] = max(js["util_max"], g["util"])
                js["mem_max"] = max(js["mem_max"], g["mem"])
                js["own_mem_max"] = max(js.get("own_mem_max", 0.0), own_mem.get(jid, 0.0))
                js["cpu_s"] = max(js["cpu_s"], tree_cpu(proc_tree(js["pid"], children)))
            with open(SCHED / "gpu_samples.csv", "a") as f:
                for i, g in sorted(gpus.items()):
                    n_ours = sum(1 for js in running.values() if js.get("gpu") == i)
                    f.write(f"{int(time.time())},{i},{g['util']},{g['mem']},{n_ours},{foreign.get(i, 0.0)}\n")
        slots = load_slots()
        reserve = set(int(c) for c in slots["reserve"])
        used_cores = set()
        for js in running.values():
            used_cores.update(js.get("phys", []))
        ntrain = {}
        est = {}
        for rjid, js in running.items():
            if js.get("gpu") is None:
                continue
            g = js["gpu"]
            if js.get("kind") == "train":
                ntrain[g] = ntrain.get(g, 0) + 1
            # a job that has not allocated its memory yet still counts with its estimate
            est[g] = est.get(g, 0.0) + max(js.get("mem_est", 0.0), own_mem.get(rjid, 0.0))
        cancelled = set(CANCELLED.read_text().split()) if CANCELLED.exists() else set()
        boost = lambda j: 1000 if job_kind(j) in ("select", "eval") else 0              # noqa: E731
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
            ncores = job_cores(job)
            if job.get("device") == "cpu":
                phys = pick_cores(ncores, int(job.get("node", 0)), used_cores, reserve)
                if phys is None:
                    continue
                launch(job, None, phys, st)
                used_cores.update(phys)
                continue
            mem = job_mem(job)
            cands = [int(job["gpu"])] if job.get("gpu") is not None else [int(g) for g in slots["gpus"]]
            ok = []
            for g in cands:
                cfg = slots["gpus"].get(str(g))
                if not cfg:
                    continue
                if job_kind(job) == "train" and ntrain.get(g, 0) >= int(cfg.get("cap", 1)):
                    continue
                limit = float(cfg.get("mem_limit_mib", 44000))
                if foreign.get(g, 0.0) + est.get(g, 0.0) + mem > limit:
                    continue
                ok.append(g)
            if not ok:
                continue
            free_local = lambda g: sum(1 for c in range(N_PHYS) if c not in used_cores and c not in reserve and node_of(c) == gpu_node(g))  # noqa: E731
            ok.sort(key=lambda g: (ntrain.get(g, 0) if job_kind(job) == "train" else est.get(g, 0.0) / 1e4,
                                   -free_local(g), est.get(g, 0.0)))
            g = ok[0]
            phys = pick_cores(ncores, gpu_node(g), used_cores, reserve)
            if phys is None:
                continue
            launch(job, g, phys, st)
            used_cores.update(phys)
            if job_kind(job) == "train":
                ntrain[g] = ntrain.get(g, 0) + 1
            est[g] = est.get(g, 0.0) + mem
        atomic_write(STATE, st)
        time.sleep(10)
    print(f"[{now()}] STOP file found; daemon exits (running jobs continue unsupervised)", flush=True)


def submit(path: str) -> None:
    obj = json.loads(Path(path).read_text())
    jobs = obj if isinstance(obj, list) else [obj]
    have = load_jobs()
    SCHED.mkdir(parents=True, exist_ok=True)
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
            extra = (f"gpu={js['gpu']} cores={len(js.get('phys', []))} {(time.time() - js.get('t0', time.time())) / 60:.0f} min "
                     f"util~{js.get('util_sum', 0) / max(js.get('samples', 0), 1):.0f}% ownmem={js.get('own_mem_max', 0):.0f}")
        elif js.get("status") in ("done", "failed"):
            extra = f"rc={js.get('rc')} end={js.get('end')}"
        print(f"{jid:48s} {js['status']:9s} {extra}")


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
    elif cmd == "kill":
        with open(KILLS, "a") as f:
            for jid in sys.argv[2:]:
                f.write(jid + "\n")
                print("kill requested", jid)
    else:
        status()
