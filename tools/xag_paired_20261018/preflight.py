#!/usr/bin/env python3
"""Own only registered engineering preflight jobs; never select by scores."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "outputs/xag_paired/20261018"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def resources():
    mem = {k: int(v.strip().split()[0]) * 1024
           for k, v in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    vm = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    return mem["MemAvailable"], int(vm["oom_kill"])


def guard(protocol, path, expected):
    if sha(path) != expected:
        raise RuntimeError("Preflight protocol changed")
    for name, digest in protocol["file_sha256"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("Preflight active source changed: " + name)
    for name, identity in protocol["immutable_data_stat"].items():
        s = Path(name).stat()
        if s.st_size != identity["bytes"] or s.st_mtime_ns != identity["mtime_ns"]:
            raise RuntimeError("Preflight data identity changed: " + name)


def stop(processes):
    for p in processes.values():
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 15
    while any(p.poll() is None for p in processes.values()) and time.monotonic() < deadline:
        time.sleep(.2)
    for p in processes.values():
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        p.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    digest = sha(args.protocol)
    if protocol["status"] != "PREFLIGHT_REGISTERED":
        raise ValueError("Explicit new preflight registration required")
    out = Path(protocol["execution_output"])
    if not out.is_absolute():
        out = ROOT / out
    if not out.resolve().is_relative_to(STUDY / "preflight"):
        raise ValueError("Preflight output outside new namespace")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "status.json").exists():
        raise FileExistsError("Preserve previous execution")
    if any((ROOT / r["output"]).exists() for r in protocol["runs"]):
        raise FileExistsError("Preflight runs must start fresh")
    guard(protocol, args.protocol, digest)
    _, oom_start = resources()
    state = {"status": "RUNNING_ENGINEERING_PREFLIGHT", "started_at_utc": now(),
             "pid": os.getpid(), "protocol_sha256": digest, "oom_start": oom_start,
             "minimum_available_memory_bytes": 1 << 60, "runs": {}}
    processes, handles = {}, {}
    pending = list(protocol["runs"])
    low = 0
    def interrupted(number, frame):
        raise InterruptedError("Preflight interrupted: " + str(number))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while pending or any(p.poll() is None for p in processes.values()):
            guard(protocol, args.protocol, digest)
            available, oom = resources()
            low = low + 1 if available < 16 * 1024 ** 3 else 0
            state.update(updated_at_utc=now(), available_memory_bytes=available, oom_now=oom)
            state["minimum_available_memory_bytes"] = min(state["minimum_available_memory_bytes"], available)
            if low >= 2 or oom != oom_start:
                raise RuntimeError("Preflight host memory/OOM stop")
            busy = {g for r in protocol["runs"] if r["name"] in processes
                    and processes[r["name"]].poll() is None for g in r["physical_gpus"]}
            for run in list(pending):
                if busy.intersection(run["physical_gpus"]):
                    continue
                command = [sys.executable, "tools/xag_paired_20261018/train.py", "--cell", run["cell"],
                           "--phase", "preflight", "--sampler", "legacy_none", "--protocol",
                           str(args.protocol.resolve()), "--config", run["config"]]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, run["physical_gpus"])),
                           PYTHONHASHSEED="3407", PL_INTER_BATCH_PARALLELISM="0")
                handles[run["name"]] = (out / (run["name"] + ".log")).open("x")
                p = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handles[run["name"]],
                                     stderr=subprocess.STDOUT, start_new_session=True)
                processes[run["name"]] = p
                state["runs"][run["name"]] = {**run, "pid": p.pid, "status": "RUNNING", "command": command}
                busy.update(run["physical_gpus"])
                pending.remove(run)
            for name, p in processes.items():
                code = p.poll()
                if code is not None:
                    state["runs"][name].update(status="EXITED", exit_code=code)
                    if code:
                        raise RuntimeError(name + " failed with exit " + str(code))
                directory = ROOT / state["runs"][name]["output"]
                for rank in range(2):
                    f = directory / f"progress_rank{rank}.json"
                    if f.exists():
                        state["runs"][name][f"rank{rank}"] = json.loads(f.read_text())
            save(out / "status.json", state)
            time.sleep(2)
        guard(protocol, args.protocol, digest)
        state.update(status="TRAINING_COMPLETED_PENDING_INDEPENDENT_CHECKS", completed_at_utc=now())
        save(out / "status.json", state)
    except BaseException as error:
        stop(processes)
        state.update(status="FAILED_PRESERVED", error=str(error), error_type=type(error).__name__, ended_at_utc=now())
        save(out / "status.json", state)
        raise
    finally:
        for f in handles.values():
            f.close()


if __name__ == "__main__":
    main()
