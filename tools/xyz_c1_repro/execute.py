#!/usr/bin/env python3
"""Own the three predeclared C1 reproducibility jobs and verify their outputs."""
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
OUT = ROOT / "outputs/xyz_c1_repro/20261018"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def verify_sources(protocol):
    for name, digest in protocol["file_sha256"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("Frozen reproduction source changed: " + name)
    for name, expected in protocol.get("immutable_data_stat", {}).items():
        stat = Path(name).stat()
        if stat.st_size != expected["bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise RuntimeError("Frozen data identity changed: " + name)


def memory():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    vm = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    return values["MemAvailable"], int(vm["oom_kill"])


def stop_owned(processes):
    for proc in processes.values():
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 20
    while any(p.poll() is None for p in processes.values()) and time.monotonic() < deadline:
        time.sleep(0.25)
    for proc in processes.values():
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for proc in processes.values():
        proc.wait()


def checkpoint_agreement(canonical, replay, checked):
    for step in range(0, 6001, 500):
        if step in checked:
            continue
        paths = [directory / f"state_rank{rank}_step{step:06}.json"
                 for directory in (canonical, replay) for rank in range(2)]
        if not all(path.is_file() for path in paths):
            continue
        states = [read(p) for p in paths]
        if states[0] != states[2] or states[1] != states[3]:
            save(OUT / "trajectory_mismatch.json", {"step": step, "paths": [str(p) for p in paths],
                 "scope": "canonical/replay complete normalized model, Adam, schedule, RNG and input stream"})
            raise RuntimeError("Canonical/replay trajectories differ at step " + str(step))
        for a, b in ((states[0], states[1]), (states[2], states[3])):
            if any(a[field] != b[field] for field in ("model", "optimizers", "schedulers")):
                raise RuntimeError("Rank model/optimizer disagreement at " + str(step))
        checked.add(step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    args = parser.parse_args()
    protocol, acceptance = read(args.protocol), read(args.acceptance)
    protocol_sha256 = sha(args.protocol)
    acceptance_sha256 = sha(args.acceptance)
    if protocol["status"] != "FROZEN_READY" or acceptance["status"] != "PASS":
        raise RuntimeError("Frozen protocol and complete preflight acceptance required")
    if (acceptance["preflight_evidence_sha256"] != protocol["preflight_acceptance_sha256"]
            or acceptance["bindings"]["protocol_sha256"] != protocol_sha256
            or sha(ROOT / protocol["preflight_evidence_path"]) != protocol["preflight_acceptance_sha256"]
            or acceptance["bindings"]["wrapper_sha256"] != sha(ROOT / "tools/xyz_c1_repro/train.py")):
        raise RuntimeError("Preflight acceptance binding changed")
    verify_sources(protocol)
    runs = protocol["runs"]
    if len(runs) != 3 or [r["role"] for r in runs] != ["canonical_baseline", "exact_fresh_replay", "sampling_intervention"]:
        raise RuntimeError("Unexpected role inventory")
    if [r["sampler"] for r in runs] != ["legacy_none", "legacy_none", "dedicated"]:
        raise RuntimeError("Unexpected sampling intervention")
    if any(r["seed"] != 3407 or r["fresh_steps"] != 6000 for r in runs):
        raise RuntimeError("Unexpected seed or budget")
    if [r["physical_gpus"] for r in runs] != [[0, 1], [2, 3], [4, 5]]:
        raise RuntimeError("Unexpected or overlapping GPU allocation")
    if any((ROOT / r["output"]).exists() for r in runs):
        raise FileExistsError("Every formal run must start fresh in a new directory")
    for run in runs:
        key = str((ROOT / run["config"]).resolve())
        if (acceptance["bindings"]["config_sha256"][key] != sha(key)
                or acceptance["bindings"]["sampler_by_config"][key] != run["sampler"]):
            raise RuntimeError("Formal configuration or sampler binding changed")
    os.chdir(ROOT)
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    processes, handles, checked = {}, {}, set()
    _, oom_start = memory()
    state = {"status": "TRAINING", "baseline_ready": False, "started_at_utc": now(),
             "coordinator_pid": os.getpid(), "protocol_sha256": sha(args.protocol),
             "preflight_sha256": sha(args.acceptance), "runs": {}, "oom_start": oom_start}
    low_samples = 0
    def interrupted(number, frame):
        raise InterruptedError("Coordinator interrupted by signal " + str(number))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def frozen_guard():
        if sha(args.protocol) != protocol_sha256 or sha(args.acceptance) != acceptance_sha256:
            raise RuntimeError("Registered protocol or launch acceptance changed")
        verify_sources(protocol)
    def run_owned(tag, command, env=None):
        frozen_guard()
        handles[tag] = (logs / (tag + ".log")).open("x")
        processes[tag] = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handles[tag],
                                         stderr=subprocess.STDOUT, start_new_session=True)
        code = processes[tag].wait()
        if code:
            raise RuntimeError(tag + " failed with exit " + str(code))
        frozen_guard()
    try:
        for run in runs:
            name = run["name"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, run["physical_gpus"])),
                       PL_INTER_BATCH_PARALLELISM="0")
            command = [sys.executable, "tools/xyz_c1_repro/train.py", "--phase", "formal",
                       "--sampler", run["sampler"], "--protocol", str(args.protocol.resolve()),
                       "--acceptance", str(args.acceptance.resolve()), "--config", run["config"]]
            handles[name] = (logs / f"{name}.log").open("x")
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handles[name],
                                    stderr=subprocess.STDOUT, start_new_session=True)
            processes[name] = proc
            state["runs"][name] = {**run, "pid": proc.pid, "command": command,
                                  "started_at_utc": now(), "status": "TRAINING"}
        while True:
            if sha(args.protocol) != protocol_sha256:
                raise RuntimeError("Registered protocol changed during execution")
            if sha(args.acceptance) != acceptance_sha256:
                raise RuntimeError("Registered launch acceptance changed during execution")
            verify_sources(protocol)
            available, oom_now = memory()
            low_samples = low_samples + 1 if available < 16 * 1024 ** 3 else 0
            if low_samples >= 2 or oom_now != oom_start:
                raise RuntimeError("Host resource stop: sustained low memory or new oom_kill")
            state.update(updated_at_utc=now(), available_memory_bytes=available, oom_now=oom_now)
            for run in runs:
                name, directory = run["name"], ROOT / run["output"]
                proc = processes[name]
                for rank in range(2):
                    p = directory / f"progress_rank{rank}.json"
                    if p.exists():
                        state["runs"][name][f"rank{rank}"] = read(p)
                code = proc.poll()
                if code is not None:
                    state["runs"][name].update(exit_code=code, status="TRAINED" if code == 0 else "FAILED")
                    state["runs"][name].setdefault("finished_at_utc", now())
                    if code != 0:
                        raise RuntimeError(name + " failed with exit " + str(code))
            checkpoint_agreement(ROOT / runs[0]["output"], ROOT / runs[1]["output"], checked)
            state["matched_checkpoint_steps"] = sorted(checked)
            save(OUT / "status.json", state)
            if all(proc.poll() == 0 for proc in processes.values()):
                break
            time.sleep(5)
        verify_sources(protocol)
        for run in runs:
            directory = ROOT / run["output"]
            for rank in range(2):
                completion = read(directory / f"completion_rank{rank}.json")
                runtime = read(directory / f"effective_runtime_rank{rank}.json")
                actual_generator = runtime["actual_train_loader"]["underlying_generator"]
                if (completion["status"] != "COMPLETE" or completion["step"] != 6000
                        or completion["batches"] != 6000 or runtime["sampler_policy"] != run["sampler"]
                        or runtime["actual_train_loader"]["batch_size"] != 512
                        or runtime["world_size"] != 2
                        or ((actual_generator is None) != (run["sampler"] == "legacy_none"))):
                    raise RuntimeError("Incomplete or mismatched terminal run identity: " + run["name"])
                if (directory / f"indices_rank{rank}.i32").stat().st_size != 6000 * 512 * 4:
                    raise RuntimeError("Incomplete actual input tape: " + run["name"])
        state["status"] = "VERIFYING_FULL_REPRODUCTION"
        save(OUT / "status.json", state)
        command = [sys.executable, "tools/xyz_c1_repro/verify_final.py", runs[0]["output"],
                   runs[1]["output"], "--intervention", runs[2]["output"],
                   "--output", str(OUT / "full_reproduction.json")]
        run_owned("verify_full_reproduction", command)
        if checked != set(range(0, 6001, 500)):
            raise RuntimeError("Incomplete matched checkpoint grid")
        state["status"] = "TRAINING_REPRODUCED_PENDING_MAIN_ROWS"
        state["training_finished_at_utc"] = now()
        state["full_reproduction_sha256"] = sha(OUT / "full_reproduction.json")
        save(OUT / "status.json", state)
        # New namespace aliases let the original row tool resolve original data
        # and checkpoint metadata without changing any existing experiment.
        for run in runs:
            alias = ROOT / "outputs/semkine" / run["name"]
            if alias.exists() or alias.is_symlink():
                raise FileExistsError("Preserve existing namespace alias: " + str(alias))
            alias.symlink_to(ROOT / run["output"], target_is_directory=True)
        for i, run in enumerate(runs):
            prefix = run["name"].removesuffix("_s3407")
            command = [sys.executable, "tools/xyz_c1_repro/evaluate.py", "--output-dir",
                       str(OUT / "main_rows"), "--run", prefix, "--step", "6000", "--seeds", "3407"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
            run_owned(run["name"] + "_main_row", command, env)
            state["runs"][run["name"]]["main_row"] = str((OUT / "main_rows" / f"{prefix}_main_row_step6000.json").relative_to(ROOT))
            save(OUT / "status.json", state)
        rows = [read(ROOT / state["runs"][r["name"]]["main_row"]) for r in runs]
        if rows[0]["two_seed_mean"] != rows[1]["two_seed_mean"]:
            raise RuntimeError("Canonical/replay complete main-table metrics differ")
        for run in runs:
            directory = ROOT / run["output"]
            checkpoint = list(directory.glob("*step=6000.ckpt"))
            if len(checkpoint) != 1:
                raise RuntimeError("Ambiguous final checkpoint for prediction trace")
            command = [sys.executable, "tools/xyz_c1_repro/trace.py", "--config", run["config"],
                       "--checkpoint", str(checkpoint[0]), "--output-dir", str(directory / "strict_trace")]
            run_owned(run["name"] + "_strict_trace", command, dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        import numpy as np
        traces = [np.load(ROOT / run["output"] / "strict_trace/states.npz") for run in runs[:2]]
        if set(traces[0].files) != set(traces[1].files) or any(
                not np.array_equal(traces[0][key], traces[1][key]) for key in traces[0].files):
            raise RuntimeError("Canonical/replay full2590 predictions or incoming states differ")
        for run, row in zip(runs, rows):
            trace = read(ROOT / run["output"] / "strict_trace/report.json")
            for sequence in trace["per_sequence"]:
                for metric in ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm"):
                    if sequence[metric] != row["per_seed"]["3407"]["per_seq"][sequence["seq"]][metric]:
                        raise RuntimeError("Prediction trace does not reproduce its generated main row")
        frozen_guard()
        state.update(status="COMPLETE_REPRODUCIBLE_BASELINE_AND_ROWS", baseline_ready=True, completed_at_utc=now(),
                     canonical_baseline=runs[0]["name"], replay_is_not_averaged=True)
        save(OUT / "status.json", state)
    except BaseException as error:
        stop_owned(processes)
        state.update(status="FAILED_PRESERVED", baseline_ready=False, updated_at_utc=now(),
                     error_type=type(error).__name__, error=str(error))
        save(OUT / "status.json", state)
        raise
    finally:
        for handle in handles.values():
            handle.close()


if __name__ == "__main__":
    main()
