#!/usr/bin/env python3
"""Supervise the fixed XCL queues: engineering preflight, fresh training, evaluation.

This is fresh-only. Failure preserves all evidence and stops only owned groups.
No launch of either historical coordinator is possible from this entry point.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.xcl.common import OUT, CELLS, now, path, read, require, save, sha, verify_protocol, verify_run

def resources():
    info = {k: int(v.split()[0]) * 1024 for k, v in
            (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    vm = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    return info["MemAvailable"], int(vm["oom_kill"])

def shared_memory_guard():
    mount = subprocess.run(["findmnt", "-n", "--first-only", "-o", "FSTYPE", "-T", "/dev/shm"],
                           text=True, stdout=subprocess.PIPE, check=True).stdout.strip()
    require(mount == "tmpfs" and stat.S_IMODE(Path("/dev/shm").stat().st_mode) == 0o1777,
            "/dev/shm tmpfs+1777 disappeared; preserve failure")

def live_group(group):
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            value = (p / "stat").read_text()
            fields = value[value.rfind(")") + 2:].split()
            if int(fields[2]) == group and fields[0] not in ("Z", "X"):
                return True
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return False

def depth_ready():
    # Completion is per sequence, not the presence of a partially written array.
    import json
    manifest = read("outputs/xag/source_manifest.json")
    sequences = manifest.get("sequences", [])
    if isinstance(sequences, dict):
        sequences = list(sequences)
    names = [s if isinstance(s, str) else s.get("name", s.get("sequence")) for s in sequences]
    if len(names) != 74:
        # The production manifest has an explicit sequence-directory mapping.
        names = sorted(p.name for p in (ROOT / "outputs/xag/sources").iterdir()
                       if p.is_dir() and (p / "metadata.json").exists())
    if len(names) != 74:
        return False
    for name in names:
        p = ROOT / "outputs/xag/sources" / name / "depth_metadata.json"
        if not p.is_file():
            return False
        m = json.loads(p.read_text())
        if not m.get("complete"):
            return False
    return True

class Executor:
    def __init__(self, protocol_path):
        self.protocol_path = Path(protocol_path).resolve()
        self.protocol = verify_protocol(self.protocol_path)
        self.digest = sha(self.protocol_path)
        self.children = {}
        self.handles = []
        self.state = {"status": "STARTING", "started_at_utc": now(), "pid": os.getpid(),
                      "protocol_sha256": self.digest, "runs": {}, "minimum_available_memory_bytes": 1 << 60}
        self.oom = resources()[1]
        self.low = 0
        self.last_guard = 0

    def guard(self):
        available, oom = resources()
        self.low = self.low + 1 if available < 16 * 1024**3 else 0
        require(self.low < 2 and oom == self.oom, "Host memory/OOM fail-closed guard")
        self.state["minimum_available_memory_bytes"] = min(available, self.state["minimum_available_memory_bytes"])
        if time.monotonic() - self.last_guard > 30:
            require(sha(self.protocol_path) == self.digest, "Protocol changed during execution")
            verify_protocol(self.protocol_path)
            shared_memory_guard()
            if not depth_ready() and (OUT / "depth_build_launch.json").is_file():
                builder_pid = read(OUT / "depth_build_launch.json")["pid"]
                status = Path(f"/proc/{builder_pid}/stat")
                require(status.is_file() and status.read_text().split(") ", 1)[1].split()[0] not in ("Z", "X"),
                        "Depth builder exited before all74 sidecars completed; inspect depth_build_full.log")
            self.last_guard = time.monotonic()
        self.state.update(updated_at_utc=now(), available_memory_bytes=available)

    def publish(self):
        save(OUT / "status.json", self.state)

    def cleanup(self):
        for p in self.children.values():
            if live_group(p.pid):
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 15
        while any(live_group(p.pid) for p in self.children.values()) and time.monotonic() < deadline:
            time.sleep(.2)
        for p in self.children.values():
            if live_group(p.pid):
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            p.wait()

    def training_stage(self, phase):
        runs = [r for r in self.protocol["runs"] if not r["reuse"]]
        pending = list(runs)
        active = {}
        finished = set()
        accepted = []
        self.state["status"] = "ENGINEERING_PREFLIGHT" if phase == "preflight" else "FORMAL_TRAINING"
        while pending or active:
            self.guard()
            busy = {g for run, p in active.values() for g in run["physical_gpus"]}
            for run in list(pending):
                if busy.intersection(run["physical_gpus"]):
                    continue
                # Fixed per-pair order even if one arm is waiting for its cache.
                prior = [r["cell"] for r in runs[:runs.index(run)] if r["physical_gpus"] == run["physical_gpus"]]
                if any(cell not in finished for cell in prior):
                    continue
                if run["cell"] == "a3f0" and not depth_ready():
                    self.state["depth_wait"] = "WAITING_FOR_ALL_74_DEPTH_SIDECARS"
                    continue
                if run["cell"] == "a3f0" and not (OUT / "depth_binding.json").is_file():
                    from semkine.xag_sources import depth_manifest
                    manifest = depth_manifest(ROOT / "outputs/xag/sources", self.protocol["depth_binding"]["sequences"])
                    save(OUT / "depth_manifest.json", manifest, exclusive=True)
                    save(OUT / "depth_binding.json", {"manifest_sha256": sha(OUT / "depth_manifest.json"),
                         "protocol_sha256": self.digest, "bound_before_any_A3_label_consumption": True}, exclusive=True)
                directory = path(run["preflight_output"] if phase == "preflight" else run["output"])
                require(not os.path.lexists(directory), "Preserve existing run: " + str(directory))
                key = phase + ":" + run["cell"]
                config = run["preflight_config"] if phase == "preflight" else run["config"]
                command = [sys.executable, "tools/xcl/train.py", "--cell", run["cell"], "--phase", phase,
                           "--protocol", str(self.protocol_path), "--config", config]
                if phase == "formal":
                    command += ["--acceptance", str(OUT / "preflight_acceptance.json")]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, run["physical_gpus"])),
                           PYTHONHASHSEED="3407", PL_INTER_BATCH_PARALLELISM="0",
                           CUDA_DEVICE_ORDER="PCI_BUS_ID")
                # Each DDP world has an exclusive rendezvous port.
                env["MASTER_PORT"] = str(29400 + min(run["physical_gpus"]))
                handle = (OUT / "logs" / (phase + "_" + run["cell"] + ".log")).open("x")
                self.handles.append(handle)
                p = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                     stderr=subprocess.STDOUT, start_new_session=True)
                self.children[key] = p
                active[key] = (run, p)
                busy.update(run["physical_gpus"])
                pending.remove(run)
                self.state["runs"][key] = {"status": "RUNNING", "pid": p.pid,
                    "physical_gpus": run["physical_gpus"], "output": str(directory), "command": command}
                self.publish()
            for key, (run, p) in list(active.items()):
                directory = path(run["preflight_output"] if phase == "preflight" else run["output"])
                for rank in range(2):
                    progress = directory / f"progress_rank{rank}.json"
                    if progress.is_file():
                        self.state["runs"][key][f"rank{rank}"] = read(progress)
                code = p.poll()
                if code is None:
                    continue
                require(code == 0, key + " failed; exit=" + str(code))
                if live_group(p.pid):
                    continue
                receipt = verify_run(run, phase=phase, directory=directory)
                save(OUT / "receipts" / (phase + "_" + run["cell"] + ".json"), receipt, exclusive=True)
                self.state["runs"][key].update(status="PASS", exit_code=code)
                finished.add(run["cell"])
                accepted.append(run["cell"])
                del active[key]
            self.publish()
            time.sleep(5)
        if phase == "preflight":
            # Both oracle feature controls share the same added projection initial values.
            a, b = [path(next(r for r in runs if r["cell"] == cell)["preflight_output"]) for cell in ("a0fod", "a0foo")]
            models = [read(p / "state_rank0_step000000.json")["model"] for p in (a, b)]
            require({k: v for k, v in models[0].items() if "xag_oracle_projection" in k}
                    == {k: v for k, v in models[1].items() if "xag_oracle_projection" in k}, "Fomega projection initial values differ")
            save(OUT / "preflight_acceptance.json", {"status": "PASS", "cells": accepted,
                 "protocol_sha256": self.digest, "wrapper_sha256": sha("tools/xcl/train.py"),
                 "depth_binding_sha256": sha(OUT / "depth_binding.json"),
                 "completed_at_utc": now(), "engineering_only_no_accuracy_result": True}, exclusive=True)

    def run(self):
        require(not (OUT / "status.json").exists(), "Fresh-only coordinator; preserve previous status")
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "logs").mkdir(exist_ok=True)
        with (OUT / "execution.lock").open("x") as stream:
            stream.write(str(os.getpid()) + "\n")
        try:
            shared_memory_guard()
            self.training_stage("preflight")
            self.training_stage("formal")
            self.state["status"] = "ALL_TRAINING_VERIFIED_EVALUATION_PENDING"
            self.publish()
            self.evaluate()
        except BaseException as error:
            self.cleanup()
            self.state.update(status="FAILED_PRESERVED", error_type=type(error).__name__, error=str(error), ended_at_utc=now())
            self.publish()
            raise
        finally:
            for handle in self.handles:
                handle.close()

    def evaluate(self):
        """Delegate the complete, fixed evaluation plan to the explicit evaluator."""
        # Evaluation CLI is validated before protocol publication.
        command = [sys.executable, "tools/xcl/evaluate_all.py", "--protocol", str(self.protocol_path)]
        handle = (OUT / "logs/evaluation.log").open("x")
        self.handles.append(handle)
        p = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        self.children["evaluation"] = p
        self.state.update(status="EVALUATION", evaluation_pid=p.pid)
        self.publish()
        while p.poll() is None:
            self.guard()
            self.publish()
            time.sleep(5)
        require(p.wait() == 0, "Unified evaluation failed; preserve outputs")
        self.state.update(status="COMPLETE_PENDING_SCIENTIFIC_REVIEW", completed_at_utc=now())
        self.publish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    def interrupted(number, frame):
        raise InterruptedError("XCL coordinator interrupted: " + str(number))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    Executor(args.protocol).run()

if __name__ == "__main__":
    main()
