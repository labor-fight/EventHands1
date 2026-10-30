#!/usr/bin/env python3
"""Execute the five frozen single-seed arms paired to the completed C1."""
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
OUT = ROOT / "outputs/xag_paired/20261018"
CELLS = ("a0f1", "a1f0", "a1f1", "a2f0", "a2f1")
PAIRS = {"a0f1": [0, 1], "a1f1": [2, 3], "a2f1": [4, 5],
         "a1f0": [6, 7], "a2f0": [6, 7]}
BASELINE = ROOT / "outputs/xyz_c1_repro/20261018/runs/xyz_c1_stable_s3407"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def memory():
    values = {k: int(v.strip().split()[0]) * 1024
              for k, v in (s.split(":", 1) for s in Path("/proc/meminfo").read_text().splitlines())}
    vm = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    return values["MemAvailable"], int(vm["oom_kill"])


def live_owned_groups(processes):
    groups = {p.pid for p in processes.values()}
    alive = set()
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            value = (path / "stat").read_text()
            fields = value[value.rfind(")") + 2:].split()
            group = int(fields[2])
            if group in groups and fields[0] not in ("Z", "X"):
                alive.add(group)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return alive


def verify_sources(protocol):
    for name, value in protocol["file_sha256"].items():
        if sha(ROOT / name) != value:
            raise RuntimeError("Frozen paired source changed: " + name)
    for name, expected in protocol["immutable_data_stat"].items():
        s = Path(name).stat()
        if s.st_size != expected["bytes"] or s.st_mtime_ns != expected["mtime_ns"]:
            raise RuntimeError("Frozen paired data changed: " + name)


def stop_owned(processes):
    for p in processes.values():
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    while any(p.poll() is None for p in processes.values()) and time.monotonic() < deadline:
        time.sleep(.2)
    for p in processes.values():
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()


def checkpoint_pairing(run, checked, extra_reference):
    """Shared init + actual input/RNG contract, never cross-arm trained weights."""
    directory = ROOT / run["output"]
    for step in range(0, 6001, 500):
        if step in checked:
            continue
        paths = [directory / f"state_rank{r}_step{step:06}.json" for r in range(2)]
        if not all(p.exists() for p in paths):
            continue
        states = [read(p) for p in paths]
        for rank, state in enumerate(states):
            baseline = read(BASELINE / f"state_rank{rank}_step{step:06}.json")
            for field in ("step", "batches", "input_stream_sha256", "torch_cpu_rng"):
                if state[field] != baseline[field]:
                    raise RuntimeError(f"C1 paired input/CPU RNG mismatch: {run['name']} rank{rank} step{step} {field}")
            if step == 0:
                common = {k: v for k, v in state["model"].items() if not k.startswith("xyz.xag_global.")}
                extra = set(state["model"]) - set(common)
                if common != baseline["model"] or len(extra) != (39 if run["cell"].endswith("f1") else 0):
                    raise RuntimeError("Unpaired initial model/shared parameter inventory: " + run["name"])
                if extra:
                    extra_model = {k: state["model"][k] for k in extra}
                    if not extra_reference:
                        extra_reference.update(extra_model)
                    elif extra_reference != extra_model:
                        raise RuntimeError("Three F+ arms do not share identical initial added parameters")
        for field in ("model", "optimizers", "schedulers"):
            if states[0][field] != states[1][field]:
                raise RuntimeError("DDP ranks disagree in " + run["name"] + " " + field)
        checked.add(step)


def terminal_training_identity(run):
    directory = ROOT / run["output"]
    for rank in range(2):
        completed = read(directory / f"completion_rank{rank}.json")
        runtime = read(directory / f"effective_runtime_rank{rank}.json")
        loader = runtime["actual_train_loader"]
        if (completed["status"] != "COMPLETE" or completed["step"] != 6000
                or completed["batches"] != 6000 or runtime["cell"] != run["cell"]
                or runtime["sampler_policy"] != "legacy_none" or loader["underlying_generator"] is not None
                or loader["batch_size"] != 512 or runtime["world_size"] != 2):
            raise RuntimeError("Incomplete terminal paired identity: " + run["name"])
        if (directory / f"indices_rank{rank}.i32").stat().st_size != 6000 * 512 * 4:
            raise RuntimeError("Incomplete actual input tape: " + run["name"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    options = parser.parse_args()
    protocol, acceptance = read(options.protocol), read(options.acceptance)
    p_sha, a_sha = sha(options.protocol), sha(options.acceptance)
    if protocol["status"] != "FROZEN_READY" or acceptance["status"] != "PASS":
        raise ValueError("New frozen protocol and complete preflight acceptance required")
    bindings = acceptance["bindings"]
    if (bindings["protocol_sha256"] != p_sha
            or bindings["wrapper_sha256"] != sha(ROOT / "tools/xag_paired_20261018/train.py")
            or acceptance["preflight_evidence_sha256"] != protocol["preflight_acceptance_sha256"]
            or sha(protocol["preflight_evidence_path"]) != protocol["preflight_acceptance_sha256"]):
        raise ValueError("Paired E/P/A physical binding changed")
    for key, path in (("helper_sha256", ROOT / "tools/xyz_c1_repro/train.py"),
                      ("canonical_config_sha256", ROOT / "configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml"),
                      ("canonical_baseline_sha256", ROOT / "outputs/xyz_c1_repro/baseline.json"),
                      ("source_manifest_sha256", ROOT / "outputs/xag/source_manifest.json")):
        if bindings.get(key) != sha(path):
            raise ValueError("Accepted launch identity differs: " + key)
    verify_sources(protocol)
    runs = protocol["runs"]
    if len(runs) != 5 or set(r["cell"] for r in runs) != set(CELLS):
        raise ValueError("Exactly five paired arms required")
    if any(r["physical_gpus"] != PAIRS[r["cell"]] or r["seed"] != 3407
           or r["fresh_steps"] != 6000 or r["sampler"] != "legacy_none" for r in runs):
        raise ValueError("Unregistered pair/seed/budget/sampler")
    if ((OUT / "status.json").exists() or (OUT / "status.json").is_symlink()
            or any((ROOT / r["output"]).exists() or (ROOT / r["output"]).is_symlink() for r in runs)):
        raise FileExistsError("Preserve previous state: all new formal runs must be fresh")
    for run in runs:
        key = str((ROOT / run["config"]).resolve())
        if (bindings["config_sha256"][key] != sha(key)
                or bindings["sampler_by_config"][key] != "legacy_none"
                or bindings["cell_by_config"][key] != run["cell"]
                or bindings["modes_by_config"][key] != run["modes"]):
            raise ValueError("Actual config differs from accepted config")
    lock = os.open(OUT / "execution.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(lock, "w") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "protocol_sha256": p_sha}) + "\n")
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    _, oom_start = memory()
    processes, handles = {}, {}
    checked = {r["name"]: set() for r in runs}
    extra_reference = {}
    # Fixed order: three F+ pairs and F0 pair; a2f0 waits for a1f0 on6/7.
    order = ("a0f1", "a1f1", "a2f1", "a1f0", "a2f0")
    pending = [next(r for r in runs if r["cell"] == c) for c in order]
    state = {"status": "TRAINING_PAIRED_FIVE", "results_ready": False, "started_at_utc": now(),
             "pid": os.getpid(), "protocol_sha256": p_sha, "acceptance_sha256": a_sha,
             "oom_start": oom_start, "runs": {r["name"]: {**r, "status": "PENDING"} for r in runs}}
    low = 0
    def interrupted(number, frame):
        raise InterruptedError("Paired coordinator interrupted: " + str(number))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def guard():
        nonlocal low
        if (sha(options.protocol) != p_sha or sha(options.acceptance) != a_sha
                or sha(protocol["preflight_evidence_path"]) != protocol["preflight_acceptance_sha256"]):
            raise RuntimeError("Frozen E/P/A changed during execution")
        verify_sources(protocol)
        available, oom_now = memory()
        low = low + 1 if available < 16 * 1024 ** 3 else 0
        state.update(updated_at_utc=now(), available_memory_bytes=available, oom_now=oom_now)
        if low >= 2 or oom_now != oom_start:
            raise RuntimeError("Owned execution host memory/OOM stop")
    def run_owned(tag, command, env=None):
        guard()
        handles[tag] = (logs / (tag + ".log")).open("x")
        p = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handles[tag],
                             stderr=subprocess.STDOUT, start_new_session=True)
        processes[tag] = p
        while p.poll() is None:
            guard()
            time.sleep(5)
        if p.returncode:
            raise RuntimeError(tag + " failed with exit " + str(p.returncode))
        guard()
    try:
        while (pending or any(processes[r["name"]].poll() is None for r in runs if r["name"] in processes)
               or live_owned_groups(processes)):
            guard()
            live_groups = live_owned_groups(processes)
            busy = {g for r in runs if r["name"] in processes
                    and (processes[r["name"]].poll() is None or processes[r["name"]].pid in live_groups)
                    for g in r["physical_gpus"]}
            # Reap/validate existing jobs before allowing another fresh job onto that pair.
            for run in runs:
                name = run["name"]
                if name not in processes:
                    continue
                code = processes[name].poll()
                directory = ROOT / run["output"]
                for rank in range(2):
                    p = directory / f"progress_rank{rank}.json"
                    if p.exists():
                        state["runs"][name][f"rank{rank}"] = read(p)
                checkpoint_pairing(run, checked[name], extra_reference)
                state["runs"][name]["paired_checkpoint_steps"] = sorted(checked[name])
                if code is not None:
                    if code:
                        raise RuntimeError(name + " failed with exit " + str(code))
                    terminal_training_identity(run)
                    state["runs"][name].update(status="TRAINED", exit_code=0)
                    state["runs"][name].setdefault("finished_at_utc", now())
            for run in list(pending):
                if busy.intersection(run["physical_gpus"]):
                    continue
                if run["cell"] == "a2f0":
                    prior = next(r for r in runs if r["cell"] == "a1f0")
                    if state["runs"][prior["name"]]["status"] != "TRAINED":
                        continue
                name = run["name"]
                directory = ROOT / run["output"]
                if directory.exists() or directory.is_symlink():
                    raise FileExistsError("Fresh output appeared before launch: " + str(directory))
                command = [sys.executable, "tools/xag_paired_20261018/train.py", "--cell", run["cell"],
                           "--phase", "formal", "--sampler", "legacy_none", "--protocol",
                           str(options.protocol.resolve()), "--acceptance", str(options.acceptance.resolve()),
                           "--config", run["config"]]
                env = dict(os.environ, PYTHONHASHSEED="3407", PL_INTER_BATCH_PARALLELISM="0",
                           CUDA_VISIBLE_DEVICES=",".join(map(str, run["physical_gpus"])))
                handles[name] = (logs / (name + ".log")).open("x")
                p = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handles[name],
                                     stderr=subprocess.STDOUT, start_new_session=True)
                processes[name] = p
                state["runs"][name].update(status="TRAINING", pid=p.pid, command=command, started_at_utc=now())
                busy.update(run["physical_gpus"])
                pending.remove(run)
            save(OUT / "status.json", state)
            time.sleep(5)
        # Reap every final child even when all exit between polling intervals.
        for run in runs:
            code = processes[run["name"]].wait()
            if code:
                raise RuntimeError(run["name"] + " final exit " + str(code))
            terminal_training_identity(run)
            checkpoint_pairing(run, checked[run["name"]], extra_reference)
            state["runs"][run["name"]].update(status="TRAINED", exit_code=0)
            state["runs"][run["name"]].setdefault("finished_at_utc", now())
        if any(steps != set(range(0, 6001, 500)) for steps in checked.values()):
            raise RuntimeError("Incomplete C1-paired checkpoint grid")
        state["status"] = "VERIFYING_ALL_FIVE_TRAINING_CONTRACTS"
        save(OUT / "status.json", state)
        run_owned("verify_final", [sys.executable, "tools/xag_paired_20261018/verify_final.py",
                                  "--protocol", str(options.protocol.resolve()),
                                  "--output", str(OUT / "paired_training_audit.json")])
        if read(OUT / "paired_training_audit.json")["status"] != "PASS":
            raise RuntimeError("Full paired training audit failed")
        state["status"] = "GENERATING_FULL_SEQUENCE_ROWS_AND_TRACES"
        save(OUT / "status.json", state)
        for run in runs:
            name = run["name"]
            run_owned(name + "_main_row", [sys.executable, "tools/xag_paired_20261018/evaluate.py",
                       "--protocol", str(options.protocol.resolve()), "--run", run["prefix"],
                       "--step", "6000", "--seeds", "3407", "--output-dir", str(OUT / "main_rows")],
                      dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
            row = OUT / "main_rows" / (run["prefix"] + "_main_row_step6000.json")
            state["runs"][name]["main_row"] = str(row.relative_to(ROOT))
            for repeat in (1, 2):
                run_owned(name + f"_trace{repeat}", [sys.executable, "tools/xag_paired_20261018/trace.py",
                          "--protocol", str(options.protocol.resolve()), "--run", name,
                          "--output-dir", str(ROOT / run["output"] / f"strict_trace{repeat}")],
                          dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
            save(OUT / "status.json", state)
        state["status"] = "VERIFYING_ROWS_TRACES_AND_PUBLISHING"
        save(OUT / "status.json", state)
        run_owned("publish", [sys.executable, "tools/xag_paired_20261018/report.py",
                              "--protocol", str(options.protocol.resolve())])
        guard()
        state.update(status="COMPLETE_PAIRED_FIVE_RESULTS", results_ready=True, completed_at_utc=now())
        save(OUT / "status.json", state)
    except BaseException as error:
        stop_owned(processes)
        state.update(status="FAILED_PRESERVED", results_ready=False, error=str(error),
                     error_type=type(error).__name__, ended_at_utc=now())
        save(OUT / "status.json", state)
        raise
    finally:
        for handle in handles.values():
            handle.close()


if __name__ == "__main__":
    main()
