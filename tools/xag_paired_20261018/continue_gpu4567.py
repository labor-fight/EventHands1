#!/usr/bin/env python3
"""Append-only GPU4--7 execution override; adopt training, never restart it.

--check-only is read-only. A live handoff stops only the original coordinator
PID, commits a new owner, observes its remaining child's real OS exit status,
then uses the unchanged verifier/evaluator/trace/reporter. P/A/926 stay frozen.
"""
from __future__ import annotations
import argparse
import copy
import ctypes
import errno
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.xag_paired_20261018 import execute as E

OUT = E.OUT
SELF = ROOT / "tools/xag_paired_20261018/continue_gpu4567.py"
SCHEDULE = "serial_rows_then_four_gpu_traces"
CELLS = ("a0f1", "a1f0", "a1f1", "a2f0", "a2f1")
ROW_GPUS = (4, 5, 6, 7, 4)
ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONDONTWRITEBYTECODE")
KERNEL_PROOF = "outputs/xag_paired/20261018/audit/contract_review/postfreeze_handoff_dummy_v1/receipt.json"
KERNEL_PROOF_SHA = "bc131573c7b62ea8feffad97cf76f40f92aac6c58e7cc194fe960b417dca93b5"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def path(value):
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def proc(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        return {"pid": int(pid), "state": fields[0], "ppid": int(fields[1]), "pgid": int(fields[2]), "session": int(fields[3]),
                "starttime": int(fields[19]), "raw_wait_status": int(fields[49])}
    except FileNotFoundError:
        return None


def command(pid):
    return [item.decode() for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if item]


def argument(args, name):
    if name in args:
        return args[args.index(name) + 1]
    return next(item.split("=", 1)[1] for item in args if item.startswith(name + "="))


def environment(pid):
    values = dict(item.split(b"=", 1) for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0") if b"=" in item)
    return {key: values.get(key.encode(), b"").decode() for key in ENV_KEYS}


def pidfd_signal(fd, number):
    native = getattr(signal, "pidfd_send_signal", None)
    if native is not None:
        native(fd, number)
        return
    require(os.uname().machine in ("x86_64", "aarch64"), "Unsupported pidfd syscall architecture")
    # Python3.9 may expose pidfd_open without the signal wrapper. Linux assigns
    # pidfd_send_signal syscall424 on these two architectures. Errors stay fatal.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(424), ctypes.c_int(fd), ctypes.c_int(number),
                          ctypes.c_void_p(None), ctypes.c_uint(0))
    if result == -1:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    require(result == 0, "Unexpected pidfd_send_signal syscall result")


def pidfd_open(pid):
    native = getattr(os, "pidfd_open", None)
    if native is not None:
        return native(pid, 0)
    require(os.uname().machine in ("x86_64", "aarch64"), "Unsupported pidfd syscall architecture")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if result == -1:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    require(result >= 0, "Unexpected pidfd_open syscall result")
    return int(result)


def pidfd_backend():
    try:
        fd = pidfd_open(-1)
    except OSError as error:
        require(error.errno in (errno.EINVAL, errno.ESRCH), "pidfd_open backend unsupported: " + str(error))
    else:
        os.close(fd)
        raise RuntimeError("pidfd_open incorrectly accepted negative PID")
    # Invalid fd and signal0 cannot target a process. EBADF proves syscall/API
    # availability; ENOSYS or any other failure is an explicit rejection.
    try:
        pidfd_signal(-1, 0)
    except OSError as error:
        require(error.errno == errno.EBADF, "pidfd signaling backend unsupported: " + str(error))
    else:
        raise RuntimeError("pidfd signaling incorrectly accepted invalid descriptor")
    return {"open": "python_os_pidfd" if hasattr(os, "pidfd_open") else "linux_syscall434_ctypes",
            "signal": "python_signal_pidfd" if hasattr(signal, "pidfd_send_signal") else "linux_syscall424_ctypes"}


class PinnedPID:
    def __init__(self, pid, starttime):
        self.backend = pidfd_backend()
        self.pid, self.starttime = int(pid), int(starttime)
        self.fd = pidfd_open(self.pid)
        self.current()

    def current(self):
        value = proc(self.pid)
        require(value is not None and value["starttime"] == self.starttime, "Pinned process disappeared or PID was reused")
        return value

    def send(self, number):
        self.current()
        pidfd_signal(self.fd, number)

    def close(self):
        os.close(self.fd)


def live_groups(pids):
    return E.live_owned_groups({str(pid): argparse.Namespace(pid=pid) for pid in pids})


def group_members(group):
    members = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                info = proc(int(entry.name))
                if info and info["pgid"] == group and info["state"] not in ("Z", "X"):
                    members.append(info)
            except (PermissionError, ProcessLookupError):
                pass
    return members


def stop_training_group(train):
    """Also clean remaining rank1/workers when their root is a zombie or gone."""
    current = proc(train.pid)
    if current:
        require(current["starttime"] == train.starttime, "Refuse cleanup after training PID reuse")
    members = group_members(train.pid)
    if not members: return
    require(all(p["session"] == train.pid and p["starttime"] >= train.starttime for p in members),
            "Remaining group does not match the original private training session")
    # A process group keeps its original identity while any member survives.
    # The live-leader check is deliberately not required for orphan cleanup.
    for number, seconds in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        try: os.killpg(train.pid, number)
        except ProcessLookupError: return
        deadline = time.monotonic() + seconds
        while group_members(train.pid) and time.monotonic() < deadline:
            time.sleep(.2)
        if not group_members(train.pid): return
    raise RuntimeError("Owned training group still pending kernel exit after SIGKILL")


def fresh_products(protocol):
    products = [OUT / "paired_training_audit.json", OUT / "release"]
    for run in protocol["runs"]:
        row = OUT / "main_rows" / (run["prefix"] + "_main_row_step6000.json")
        products += [row, row.with_name(row.stem + "_receipt.json")]
        products += [path(run["output"]) / f"strict_trace{repeat}" for repeat in (1, 2)]
    require(not any(os.path.lexists(p) for p in products), "Existing postprocessing products must be preserved, never overwritten")


def state_identity(state, protocol, override):
    require(state["status"] == "TRAINING_PAIRED_FIVE" and state.get("results_ready") is False,
            "Original coordinator is not at the registered training handoff boundary")
    require(state["pid"] == override["coordinator_pid"] and state["protocol_sha256"] == override["protocol_sha256"]
            and state["acceptance_sha256"] == override["acceptance_sha256"], "Original state P/A/PID identity changed")
    require(set(state["runs"]) == {r["name"] for r in protocol["runs"]}, "Different training matrix")
    for run in protocol["runs"]:
        saved = state["runs"][run["name"]]
        require(all(saved[k] == run[k] for k in run), "Run registration changed")
        if run["name"] == override["training_run"]:
            require(saved["status"] == "TRAINING" and saved["pid"] == override["training_pid"]
                    and saved.get("exit_code") is None, "Remaining training identity changed")
        else:
            require(saved["status"] == "TRAINED" and saved["exit_code"] == 0, "Exactly four completed arms required")


def validate_override(filename):
    filename = path(filename).resolve()
    pidfd_backend()
    require(filename.is_relative_to(OUT), "Override must be an appended file in the new study namespace")
    value = E.read(filename)
    require(value.get("status") == "APPROVED_USER_GPU_OVERRIDE" and value.get("gpus") == [4, 5, 6, 7]
            and value.get("evaluation_schedule") == SCHEDULE, "Unapproved GPU override/schedule")
    require(value["coordinator_pid"] == 255504 and value["training_pid"] == 274970
            and value["training_run"] == "xag_paired_a2f0_s3407", "This one-time handoff owns only the registered original processes")
    for name in ("protocol", "acceptance", "state_snapshot"):
        require(E.sha(path(value[name + ("_path" if name == "state_snapshot" else "")])) == value[name + "_sha256"], "Override physical binding changed: " + name)
    require(path(value["protocol"]).resolve() == OUT / "protocol.json"
            and path(value["acceptance"]).resolve() == OUT / "launch_acceptance.json", "Use original sealed P/A")
    require(path(value["state_snapshot_path"]).resolve() != OUT / "status.json", "Reference snapshot must be an immutable copy, not live status")
    sources = value["source_sha256"]
    for name in (str(SELF.relative_to(ROOT)), "tools/xag_paired_20261018/execute.py"):
        require(sources.get(name) == E.sha(ROOT / name), "Override script/source not bound: " + name)
    for name, expected in sources.items():
        require(E.sha(path(name)) == expected, "Additional override source changed: " + name)
    proofs = value.get("external_evidence_sha256", {})
    require(proofs.get(KERNEL_PROOF) == KERNEL_PROOF_SHA, "Validated Linux handoff proof must be bound")
    for name, expected in proofs.items():
        require(E.sha(path(name)) == expected, "External handoff evidence changed: " + name)
    protocol, acceptance = E.read(path(value["protocol"])), E.read(path(value["acceptance"]))
    require(protocol["status"] == "FROZEN_READY" and acceptance["status"] == "PASS"
            and acceptance["bindings"]["protocol_sha256"] == value["protocol_sha256"]
            and acceptance["preflight_evidence_sha256"] == protocol["preflight_acceptance_sha256"]
            and E.sha(path(protocol["preflight_evidence_path"])) == protocol["preflight_acceptance_sha256"], "Original E/P/A changed")
    E.verify_sources(protocol)
    require(protocol["file_sha256"]["tools/xag_paired_20261018/execute.py"] == sources["tools/xag_paired_20261018/execute.py"], "Different original executor")
    directory = path(value["handoff_directory"]).resolve()
    require(directory.is_relative_to(OUT) and not os.path.lexists(directory), "Choose a fresh handoff evidence directory")
    state = E.read(OUT / "status.json")
    state_identity(E.read(path(value["state_snapshot_path"])), protocol, value)
    state_identity(state, protocol, value)  # progress is permitted to advance
    owner, train = proc(value["coordinator_pid"]), proc(value["training_pid"])
    require(owner and owner["starttime"] == value["coordinator_starttime"] and owner["state"] not in ("Z", "X", "T", "t"), "Original coordinator not live/identity changed")
    require(train and train["starttime"] == value["training_starttime"] and train["ppid"] == owner["pid"]
            and train["pgid"] == train["session"] == train["pid"] and train["state"] not in ("Z", "X"), "Training root PPID/PGID/starttime changed")
    owner_args = command(owner["pid"])
    require(any(item.endswith("tools/xag_paired_20261018/execute.py") for item in owner_args)
            and path(argument(owner_args, "--protocol")).resolve() == path(value["protocol"]).resolve()
            and path(argument(owner_args, "--acceptance")).resolve() == path(value["acceptance"]).resolve(),
            "Coordinator command is not original execute.py/P/A")
    require(Path(owner_args[0]).is_absolute() and Path(owner_args[0]).resolve() == Path(sys.executable).resolve(),
            "Continuation must use the original coordinator's Python environment")
    args = command(train["pid"])
    require(any(item.endswith("tools/xag_paired_20261018/train.py") for item in args)
            and "a2f0" in args and "formal" in args and environment(train["pid"])["CUDA_VISIBLE_DEVICES"] == "6,7", "Training command/GPU identity changed")
    fresh_products(protocol)
    return value, protocol, state


def gpu_inventory():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"],
                            check=True, text=True, capture_output=True)
    return {int(row[0]): {"uuid": row[1], "pci_bus_id": row[2]} for line in result.stdout.splitlines()
            if (row := [part.strip() for part in line.split(",")])}


def assert_idle(gpu, inventory):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                            check=True, text=True, capture_output=True)
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if fields[0] == inventory[gpu]["uuid"]:
            require(proc(int(fields[1])) is None, "GPU is not exclusive/idle for next job: " + str(gpu))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--override", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    override_path = path(args.override).resolve()
    override, protocol, state = validate_override(override_path)
    if args.check_only:
        print(json.dumps({"status": "PASS_READ_ONLY_OVERRIDE_PRECHECK", "gpus": override["gpus"], "schedule": SCHEDULE,
                          "pidfd_backend": pidfd_backend(), "invalid_fd_probe_errno": "EBADF",
                          "no_process_targeted_no_signals_no_outputs_no_training": True}))
        return
    directory = path(override["handoff_directory"])
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "jobs").mkdir()
    with (directory / "controller.lock").open("x") as f: f.write(json.dumps({"pid": os.getpid(), "override_sha256": E.sha(override_path)}) + "\n")
    owner = PinnedPID(override["coordinator_pid"], override["coordinator_starttime"])
    train = PinnedPID(override["training_pid"], override["training_starttime"])
    committed = stop_requested = owner_killed = False
    original_bytes = None
    children, handles, job_receipts = {}, {}, {}
    checked = {r["name"]: set() for r in protocol["runs"]}
    extra_reference, low = {}, 0
    override_sha = E.sha(override_path)
    low_last = -float("inf")
    original_oom = state["oom_start"]
    def interrupted(number, frame):
        raise InterruptedError("Continuation interrupted: " + str(number))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    def guard():
        nonlocal low, low_last
        require(E.sha(override_path) == override_sha, "Execution override changed")
        for name, expected in override["source_sha256"].items(): require(E.sha(path(name)) == expected, "Override source changed")
        for name, expected in override["external_evidence_sha256"].items(): require(E.sha(path(name)) == expected, "External handoff proof changed")
        require(E.sha(path(override["protocol"])) == override["protocol_sha256"]
                and E.sha(path(override["acceptance"])) == override["acceptance_sha256"]
                and E.sha(path(override["state_snapshot_path"])) == override["state_snapshot_sha256"]
                and E.sha(path(protocol["preflight_evidence_path"])) == protocol["preflight_acceptance_sha256"], "Original E/P/A changed")
        E.verify_sources(protocol)
        available, oom = E.memory()
        if time.monotonic() - low_last >= 4.5:
            low = low + 1 if available < 16 * 1024 ** 3 else 0
            low_last = time.monotonic()
        state.update(updated_at_utc=E.now(), available_memory_bytes=available, oom_now=oom)
        require(low < 2 and oom == original_oom, "Owned execution host memory/OOM stop")
        active_gpu = {tag: receipt for tag, receipt in job_receipts.items()
                      if receipt["status"] == "RUNNING" and receipt["physical_gpu"] is not None
                      and children[tag].poll() is None}
        if active_gpu:
            result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                                    check=True, text=True, capture_output=True)
            for line in result.stdout.splitlines():
                uuid, pid = [part.strip() for part in line.split(",")]; process = proc(int(pid))
                if process is None: continue
                for tag, receipt in active_gpu.items():
                    if uuid == receipt["gpu_identity"]["uuid"]:
                        require(process["pgid"] == children[tag].pid, "Another live process shares an active timing/trace GPU")
                        record = {"pid": int(pid), "gpu_uuid": uuid, "observed_at_utc": E.now()}
                        receipt.setdefault("NVML_compute_process_observations", []).append(record)
                        E.save(directory / "jobs" / (tag + ".json"), receipt)
    def start_job(tag, command_args, gpu=None, phase="CPU", run=None):
        guard()
        if gpu is not None: assert_idle(gpu, inventory)
        environment_values = dict(os.environ, **protocol["environment_policy"], CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu))
        handles[tag] = (directory / "jobs" / (tag + ".log")).open("x")
        process = subprocess.Popen(command_args, cwd=ROOT, env=environment_values, stdout=handles[tag],
                                   stderr=subprocess.STDOUT, start_new_session=True)
        children[tag] = process
        info = proc(process.pid)
        receipt = {"status": "RUNNING", "phase": phase, "run": run, "pid": process.pid, "process": info,
                   "started_at_utc": E.now(), "command": command_args, "physical_gpu": gpu,
                   "gpu_identity": inventory[gpu] if gpu is not None else None,
                   "requested_environment": {k: environment_values.get(k, "") for k in ENV_KEYS},
                   "actual_proc_environment": environment(process.pid), "override_sha256": override_sha,
                   "protocol_sha256": override["protocol_sha256"]}
        require(receipt["actual_proc_environment"]["CUDA_VISIBLE_DEVICES"] == environment_values["CUDA_VISIBLE_DEVICES"], "Actual job GPU environment differs")
        job_receipts[tag] = receipt; E.save(directory / "jobs" / (tag + ".json"), receipt)
        return process
    def finish_job(tag):
        process = children[tag]; code = process.wait()
        receipt = job_receipts[tag]
        receipt.update(status="COMPLETE" if code == 0 else "FAILED", OS_exit_code=code, finished_at_utc=E.now())
        E.save(directory / "jobs" / (tag + ".json"), receipt)
        if code == 0 and receipt["phase"] == "row":
            run = next(r for r in protocol["runs"] if r["name"] == receipt["run"])
            row_path = OUT / "main_rows" / (run["prefix"] + "_main_row_step6000.json")
            row = E.read(row_path)
            receipt["timing"] = {"source_path": str(row_path.relative_to(ROOT)), "source_sha256": E.sha(row_path),
                                 **{k: row[k] for k in ("latency_ms_raw", "anchor_full_raw_ms", "latency_ms_scaled_full1p75")}}
            state["runs"][run["name"]]["main_row"] = str(row_path.relative_to(ROOT))
        E.save(directory / "jobs" / (tag + ".json"), receipt)
        require(code == 0, tag + " OS exit " + str(code))
        guard()
    def serial_job(tag, command_args, gpu=None, phase="CPU", run=None):
        process = start_job(tag, command_args, gpu, phase, run)
        while process.poll() is None or process.pid in E.live_owned_groups({tag: process}):
            guard(); time.sleep(5)
        finish_job(tag)
    try:
        guard()
        stop_requested = True
        owner.send(signal.SIGSTOP)  # exact pidfd target, never coordinator process group
        deadline = time.monotonic() + 10
        while owner.current()["state"] not in ("T", "t"):
            require(time.monotonic() < deadline, "Coordinator failed to enter stopped state"); time.sleep(.05)
        original_bytes = (OUT / "status.json").read_bytes(); frozen_state = json.loads(original_bytes)
        state_identity(frozen_state, protocol, override)
        require(train.current()["ppid"] == owner.pid, "Remaining training parent changed")
        with (directory / "original_status_stopped.json").open("xb") as f: f.write(original_bytes)
        state = copy.deepcopy(frozen_state)
        state.update(pid=os.getpid(), original_coordinator_pid=owner.pid, execution_override=str(override_path.relative_to(ROOT)),
                     execution_override_sha256=override_sha, continuation_started_at_utc=E.now())
        E.save(OUT / "status.json", state)
        # From this point exceptions stop this task instead of waking GPU0 executor.
        committed = True
        E.save(directory / "handoff.json", {"status": "COMMITTED", "new_controller_pid": os.getpid(),
               "override_sha256": override_sha, "original_coordinator": owner.current(), "remaining_training": train.current(),
               "original_status_sha256": E.sha(directory / "original_status_stopped.json"), "at_utc": E.now()})
        train_groups = [r["pid"] for r in state["runs"].values()]
        exit_observed = None
        while True:
            guard()
            require(owner.current()["state"] in ("T", "t"), "Stopped original coordinator unexpectedly resumed")
            for run in protocol["runs"]:
                saved = state["runs"][run["name"]]
                for rank in (0, 1):
                    progress = path(run["output"]) / f"progress_rank{rank}.json"
                    if progress.exists(): saved[f"rank{rank}"] = E.read(progress)
                E.checkpoint_pairing(run, checked[run["name"]], extra_reference)
                saved["paired_checkpoint_steps"] = sorted(checked[run["name"]])
            observed = train.current()
            require(observed["ppid"] == owner.pid and observed["pgid"] == train.pid, "Adopted training ownership changed")
            if observed["state"] == "Z":
                exit_code = os.waitstatus_to_exitcode(observed["raw_wait_status"])
                require(exit_code == 0, "Adopted training OS exit " + str(exit_code))
                exit_observed = {**observed, "OS_exit_code": exit_code, "observation_time_utc": E.now()}
            if exit_observed is not None and not live_groups(train_groups):
                break
            E.save(OUT / "status.json", state); time.sleep(5)
        for run in protocol["runs"]:
            E.terminal_training_identity(run); E.checkpoint_pairing(run, checked[run["name"]], extra_reference)
            require(checked[run["name"]] == set(range(0, 6001, 500)), "Incomplete paired13 grid")
            state["runs"][run["name"]].update(status="TRAINED", exit_code=0)
            state["runs"][run["name"]].setdefault("finished_at_utc", E.now())
        E.save(directory / "adopted_training_exit.json", {"status": "PASS_OS0_ALL_TRAIN_GROUPS_QUIESCENT", "root_exit": exit_observed,
               "all_train_pgids": train_groups, "live_groups": [], "all13_step_grids": True})
        require(owner.current()["state"] in ("T", "t"), "Original coordinator must still be stopped before final PID kill")
        owner.send(signal.SIGKILL); owner_killed = True
        E.save(directory / "handoff.json", {"status": "TRAINING_COMPLETE_OLD_COORDINATOR_PID_KILLED", "new_controller_pid": os.getpid(),
               "override_sha256": override_sha, "original_coordinator_pid": owner.pid, "only_pid_signalled_never_group": True,
               "original_coordinator_deliberate_signal": "SIGKILL", "original_coordinator_expected_OS_exit_code": -9,
               "original_coordinator_shell_status_equivalent": 137, "original_coordinator_exit_is_not_training_failure": True,
               "adopted_training_OS_exit_code": 0, "at_utc": E.now()})
        inventory = gpu_inventory()
        require(all(gpu in inventory for gpu in override["gpus"]), "Required four physical GPUs unavailable")
        if "gpu_inventory" in override:
            require(all(override["gpu_inventory"][str(gpu)] == inventory[gpu] for gpu in override["gpus"]), "Physical GPU UUID/PCI inventory changed")
        state["status"] = "VERIFYING_ALL_FIVE_TRAINING_CONTRACTS"; E.save(OUT / "status.json", state)
        serial_job("verify_final", [sys.executable, "tools/xag_paired_20261018/verify_final.py", "--protocol", override["protocol"],
                   "--output", str(OUT / "paired_training_audit.json")])
        require(E.read(OUT / "paired_training_audit.json")["status"] == "PASS", "Full five-arm verification failed")
        state["status"] = "GENERATING_FULL_SEQUENCE_ROWS_AND_TRACES"; E.save(OUT / "status.json", state)
        for cell, gpu in zip(CELLS, ROW_GPUS):
            run = next(r for r in protocol["runs"] if r["cell"] == cell)
            serial_job(run["name"] + "_main_row", [sys.executable, "tools/xag_paired_20261018/evaluate.py", "--protocol", override["protocol"],
                       "--run", run["prefix"], "--step", "6000", "--seeds", "3407", "--output-dir", str(OUT / "main_rows")],
                       gpu, "row", run["name"])
            E.save(OUT / "status.json", state)
        queues = {4: [("a0f1", 1), ("a0f1", 2), ("a2f1", 1), ("a2f1", 2)],
                  5: [("a1f0", 1), ("a1f0", 2)], 6: [("a1f1", 1), ("a1f1", 2)], 7: [("a2f0", 1), ("a2f0", 2)]}
        active = {}
        while any(queues.values()) or active:
            guard()
            for gpu, tag in list(active.items()):
                process = children[tag]
                if process.poll() is not None and process.pid not in E.live_owned_groups({tag: process}):
                    finish_job(tag); del active[gpu]
            for gpu, queue in queues.items():
                if gpu in active or not queue: continue
                cell, repeat = queue.pop(0); run = next(r for r in protocol["runs"] if r["cell"] == cell)
                tag = run["name"] + f"_trace{repeat}"
                start_job(tag, [sys.executable, "tools/xag_paired_20261018/trace.py", "--protocol", override["protocol"],
                          "--run", run["name"], "--output-dir", str(path(run["output"]) / f"strict_trace{repeat}")], gpu, "trace", run["name"])
                active[gpu] = tag
            E.save(OUT / "status.json", state)
            if active: time.sleep(5)
        state["status"] = "VERIFYING_ROWS_TRACES_AND_PUBLISHING"; E.save(OUT / "status.json", state)
        serial_job("publish", [sys.executable, "tools/xag_paired_20261018/report.py", "--protocol", override["protocol"]])
        guard(); state.update(status="COMPLETE_PAIRED_FIVE_RESULTS", results_ready=True, completed_at_utc=E.now())
        E.save(OUT / "status.json", state)
        E.save(directory / "completion.json", {"status": "COMPLETE_GPU4567_OVERRIDE", "override_sha256": override_sha,
               "all_jobs": {tag: E.sha(directory / "jobs" / (tag + ".json")) for tag in job_receipts}, "no_training_restarted": True})
    except BaseException as error:
        if not committed:
            # Resume the original owner before giving up; it still owns its Popen.
            if original_bytes is not None:
                E.save(OUT / "status.json", json.loads(original_bytes))
            current = proc(train.pid)
            if current is not None:
                require(current["starttime"] == train.starttime, "Adopted PID reused during rollback")
            if stop_requested and current is not None and current["state"] not in ("Z", "X"):
                owner.send(signal.SIGCONT)
                abort_status = "ABORTED_ORIGINAL_COORDINATOR_RESUMED"
            elif stop_requested:
                # Waking the old owner after its final child exits would start
                # forbidden GPU0 evaluation. Stop any residual owned workers and
                # leave that stopped owner explicitly for supervised recovery.
                stop_training_group(train)
                abort_status = "ABORTED_TRAINING_ROOT_EXITED_OLD_COORDINATOR_HELD_STOPPED"
            else:
                abort_status = "ABORTED_ORIGINAL_COORDINATOR_UNCHANGED"
            E.save(directory / "handoff_aborted.json", {"status": abort_status, "error": str(error),
                   "no_unmonitored_live_training": True, "at_utc": E.now()})
        else:
            # All groups here belong to this task. Never signal another user's job.
            cleanup_errors = []
            try: E.stop_owned(children)
            except BaseException as cleanup: cleanup_errors.append(str(cleanup))
            try: stop_training_group(train)
            except BaseException as cleanup: cleanup_errors.append(str(cleanup))
            if not owner_killed:
                try: owner.send(signal.SIGKILL); owner_killed = True
                except BaseException as cleanup: cleanup_errors.append(str(cleanup))
            state.update(status="FAILED_PRESERVED", results_ready=False, error=str(error), error_type=type(error).__name__, ended_at_utc=E.now())
            E.save(OUT / "status.json", state)
            E.save(directory / "failure.json", {"status": "FAILED_PRESERVED", "error": str(error), "error_type": type(error).__name__,
                   "owned_only_cleanup": True, "cleanup_errors": cleanup_errors, "no_training_restart": True, "at_utc": E.now()})
        raise
    finally:
        for handle in handles.values(): handle.close()
        owner.close(); train.close()


if __name__ == "__main__":
    main()
