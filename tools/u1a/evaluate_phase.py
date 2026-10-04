#!/usr/bin/env python3
"""Evaluate one completed U1a phase, then measure costs and generate its table.

Only 2k/6k are accepted. This runner never trains or starts another phase.
Four explicit-last evaluations run on GPUs 0/2/4/6, each with 18 physical
cores and their SMT siblings. Their successful exits precede both sequential
GPU-0 cost measurements. With --wait-for-raw, costs also wait for the same
phase's complete successful raw probes. Standalone calls do not wait by default.
Existing evaluation/log/row/table artifacts are
never overwritten. A fixed S38 reference row can be reused only if identical.

  python tools/u1a/evaluate_phase.py 2k
  python tools/u1a/evaluate_phase.py 6k
  python tools/u1a/evaluate_phase.py 2k --plan   # commands only; no processes/files

The S37 row is the existing unified THREE-seed LAST row, not historical
selected checkpoints; the S38/U1a comparison uses the matched two seeds.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "outputs" / "u1a"
ROWS = REPO / "outputs" / "semkine"
MANIFEST = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51/splits_semkine.json")
SEEDS = (3407, 3408)
MODES = ("shared", "untied")
GPUS = (0, 2, 4, 6)
REFERENCE_ALIAS = "s38_u1a_fixed_reference"
S37_ALIAS = "rt_s37_3seed_tf_pert"
EVAL_TAG = "evalx_val_core_last_tf_pert"
sys.path.insert(0, str(REPO))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_state(path, state):
    state["updated_unix"] = time.time()
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(state, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def cpu_partition(index):
    physical = list(range(18 * index, 18 * (index + 1)))
    return physical + [cpu + 72 for cpu in physical]


def build_plan(phase, python, manifest=MANIFEST, threads=2):
    require(phase in ("2k", "6k"), "debug is not evaluated by this runner")
    budget = {"2k": 2000, "6k": 6000}[phase]
    jobs = []
    for index, (seed, mode) in enumerate((seed, mode) for seed in SEEDS for mode in MODES):
        run = f"u1a_{mode}_{phase}_s{seed}"
        cpus = cpu_partition(index)
        command = ["taskset", "-c", ",".join(map(str, cpus)), python, "-u",
                   "tools/tracking/evalx.py", "eval", "--run-dir", str(ROWS / run),
                   "--ckpt", "last", "--split", "val_core", "--manifest", str(manifest),
                   "--controls", "--tf", "--perturb"]
        jobs.append({"run": run, "seed": seed, "mode": mode, "gpu": GPUS[index],
                     "cpus": cpus, "physical_cores": 18, "smt_threads": 18,
                     "log": str(OUTPUT / f"{run}.eval.log"),
                     "pid_file": str(OUTPUT / f"{run}.eval.pid"),
                     "argv": command, "status": "pending", "threads_per_library": threads})
    return budget, jobs


def validate_cpu_partitions(jobs):
    allowed = os.sched_getaffinity(0)
    owners = set()
    for index, job in enumerate(jobs):
        cpus = job["cpus"]
        require(set(cpus) <= allowed, f"CPU partition {index} exceeds the runner's allowed affinity")
        require(not (owners & set(cpus)), "CPU partitions overlap")
        owners.update(cpus)
        for physical, sibling in zip(cpus[:18], cpus[18:]):
            root = Path("/sys/devices/system/cpu")
            identities = []
            for cpu in (physical, sibling):
                topology = root / f"cpu{cpu}" / "topology"
                identities.append(((topology / "physical_package_id").read_text().strip(),
                                   (topology / "core_id").read_text().strip()))
            require(identities[0] == identities[1], f"CPU {physical}/{sibling} are not SMT siblings")


def gpu_snapshot():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used",
                                   "--format=csv,noheader,nounits"], text=True)
    cards = {}
    for line in raw.strip().splitlines():
        index, uuid, memory = [part.strip() for part in line.split(",")]
        cards[int(index)] = {"uuid": uuid, "memory_mb": int(memory)}
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                                    "--format=csv,noheader,nounits"], text=True)
    for line in apps.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid, process = [part.strip() for part in line.split(",", 2)]
        for card in cards.values():
            if card["uuid"] == uuid:
                card.setdefault("compute_processes", []).append({"pid": int(pid), "process": process})
    return cards


def require_idle(gpus, limit):
    cards = gpu_snapshot()
    for gpu in gpus:
        require(gpu in cards, f"GPU {gpu} is unavailable")
        require(cards[gpu]["memory_mb"] < limit and not cards[gpu].get("compute_processes"),
                f"GPU {gpu} is occupied: {cards[gpu]}; no sharing or automatic retry")
    return {str(gpu): cards[gpu] for gpu in gpus}


def verify_phase(phase, budget, jobs):
    path = OUTPUT / f"{phase}_verification.json"
    result = read_json(path)
    require(set(result) == {job["run"] for job in jobs}, "phase verification must contain exactly the four expected runs")
    for job in jobs:
        entry = result[job["run"]]
        require(entry.get("last_step") == budget, f"{job['run']}: verification has the wrong last step")
        for key in ("frozen_encoder_bitwise", "all_state_finite", "paired_rank_streams_equal"):
            require(entry.get(key) is True, f"{job['run']}: verification did not pass {key}")
        run = ROWS / job["run"]
        last = run / "last.ckpt"
        require(last.is_file(), f"{job['run']}: last.ckpt is missing")
        numbered = []
        for checkpoint in run.glob("*step=*.ckpt"):
            match = re.search(r"step=(\d+)", checkpoint.name)
            if match:
                numbered.append((int(match.group(1)), checkpoint))
        require(numbered and max(step for step, _ in numbered) == budget,
                f"{job['run']}: evalx's actual latest numbered checkpoint is not {budget}")
        audited = [last, *[checkpoint for step, checkpoint in numbered if step == budget]]
        require(all(p.stat().st_mtime_ns <= path.stat().st_mtime_ns for p in audited),
                f"{job['run']}: checkpoint is newer than its phase verification; re-run verify.py")
    training_state = OUTPUT / f"{phase}_state.json"
    if training_state.is_file():
        state = read_json(training_state)
        require(state.get("phase") == phase and {j["run"] for j in state["jobs"]} == set(result),
                "training state does not match this phase")
        require(all(j.get("status") == "complete" and j.get("exit_code") == 0 for j in state["jobs"]),
                "training processes have not all exited successfully")
    return {"path": str(path), "sha256": sha256(path), "runs": result}


def untouched_outputs(phase, jobs):
    paths = [OUTPUT / f"{phase}_evaluation_state.json", OUTPUT / f"{phase}_table.md"]
    for job in jobs:
        paths.extend([Path(job["log"]), Path(job["pid_file"]),
                      ROWS / job["run"] / f"{EVAL_TAG}.json", ROWS / job["run"] / f"{EVAL_TAG}.npz"])
    for mode in MODES:
        arm = f"u1a_{mode}_{phase}"
        paths.extend([ROWS / f"{arm}_main_row.json", ROWS / f"{arm}_extended.json",
                      OUTPUT / f"{arm}_report.md", OUTPUT / f"{arm}.report.log", OUTPUT / f"{arm}.report.pid"])
    require(not any(path.exists() for path in paths),
            "old artifacts exist; inspect/archive them explicitly before retry: " + ", ".join(str(p) for p in paths if p.exists()))


def baseline_preflight():
    # Pure artifact validation: this import never allocates a model or calls CUDA.
    from tools.u1a import report
    runs = [str(ROWS / f"s38_spmeas_s{seed}") for seed in SEEDS]
    aggregate = report.aggregate_runs(runs, mode="baseline", budget=6000, variant="tf_pert")
    cost_path = ROWS / "s38_spmeas_tf_pert_main_row.json"
    cost = report.imported_cost(cost_path, arm="s38_spmeas", per_seed=aggregate[0])
    expected = report.make_row("s38_spmeas", "baseline", 6000, "tf_pert", aggregate, cost)
    target = ROWS / f"{REFERENCE_ALIAS}_main_row.json"
    require(not target.exists() or read_json(target) == expected,
            "the existing fixed S38 reference differs; it will not be overwritten")
    s37_path = ROWS / f"{S37_ALIAS}_main_row.json"
    s37 = read_json(s37_path)
    require(s37.get("ckpt_rule") == "last" and s37.get("n_seeds") == 3,
            "S37 table source must be the unified three-seed last artifact")
    require(set(s37["per_seed"]) == {"3407", "3408", "3409"} and
            all(seed["step"] == 6000 for seed in s37["per_seed"].values()),
            "S37 reference has unexpected seeds or steps")
    return expected, {"s38_cost_row": {"path": str(cost_path), "sha256": sha256(cost_path)},
                      "s37_row": {"path": str(s37_path), "sha256": sha256(s37_path)},
                      "s38_reference_per_seed": aggregate[2]}


def environment(gpu, threads):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads))
    return env


def launch(job, threads, state, path, children):
    # x-mode is a second no-overwrite guard after preflight, including races.
    with Path(job["log"]).open("x", encoding="utf-8") as log:
        process = subprocess.Popen(job["argv"], cwd=REPO, env=environment(job["gpu"], threads),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    children.append((job, process))
    job.update(pid=process.pid, status="running", start_unix=time.time())
    with Path(job["pid_file"]).open("x", encoding="utf-8") as pid_file:
        pid_file.write(str(process.pid) + "\n")
    atomic_state(path, state)
    print(f"launched {job.get('run', job.get('arm'))} pid={process.pid} gpu={job['gpu']}", flush=True)
    return process


def wait_children(children, state, path):
    while True:
        for job, process in children:
            exit_code = process.poll()
            if exit_code is not None and job["status"] == "running":
                job.update(status="complete" if exit_code == 0 else "failed", exit_code=exit_code, end_unix=time.time())
                print(f"{job.get('run', job.get('arm'))} {job['status']} rc={exit_code}", flush=True)
        atomic_state(path, state)
        if all(process.poll() is not None for _, process in children):
            break
        time.sleep(5)
    require(all(job["exit_code"] == 0 for job, _ in children), "one or more subprocesses failed; costs/table were not advanced")


def terminate_own_children(children):
    for job, process in children:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for job, process in children:
        if process.poll() is None:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        if job.get("status") == "running":
            job.update(status="interrupted", exit_code=process.returncode, end_unix=time.time())


def wait_for_raw(phase, verification, timeout_seconds):
    """Read-only barrier; all same-phase raw workers must have exited zero."""
    path = OUTPUT / f"{phase}_raw_state.json"
    expected = {f"u1a_{mode}_{phase}_s{seed}" for seed in SEEDS for mode in MODES}
    allowed = expected | {f"s38_spmeas_s{seed}" for seed in SEEDS}
    deadline = time.monotonic() + timeout_seconds
    print(f"waiting for complete {phase} raw probes before GPU-0 costs: {path}", flush=True)
    while True:
        if path.is_file():
            raw = read_json(path)
            require(raw.get("phase") == phase, "raw barrier state belongs to a different phase")
            require(raw.get("stage") != "failed", f"raw runner failed: {raw.get('error')}")
            jobs = raw.get("jobs", [])
            names = [job["run"] for job in jobs]
            require(len(names) == len(set(names)) and set(names) <= allowed,
                    "raw barrier has unexpected or duplicate jobs")
            require(not any(job.get("status") == "failed" for job in jobs), "a raw probe failed")
            if raw.get("stage") == "complete":
                require(expected <= set(names), "completed raw state is missing a preregistered candidate")
                require(all(job.get("status") == "complete" and job.get("exit_code") == 0 for job in jobs),
                        "completed raw state contains unfinished/failed workers")
                require(raw.get("verification", {}).get("sha256") == verification["sha256"],
                        "raw and recursive runners used different phase verification sources")
                print(f"{phase} raw barrier passed; all raw workers exited zero before costs", flush=True)
                return {"path": str(path), "sha256": sha256(path), "runner_pid": raw.get("runner_pid"),
                        "runs": names, "all_raw_workers_exit_zero": True}
        require(time.monotonic() < deadline, "timed out waiting for same-phase raw completion; no costs started")
        time.sleep(5)


def interrupt(signum, _frame):
    # Raising inside the main try triggers cleanup of only our own worker groups.
    raise SystemExit(f"evaluation runner received signal {signum}")


def report_job(mode, phase, budget, python):
    arm = f"u1a_{mode}_{phase}"
    cpus = cpu_partition(0)
    argv = ["taskset", "-c", ",".join(map(str, cpus)), python, "-u", "tools/u1a/report.py",
            "--arm", arm, "--mode", mode, "--budget", str(budget), "--runs",
            *[str(ROWS / f"{arm}_s{seed}") for seed in SEEDS], "--variant", "tf_pert", "--measure-cost",
            "--baseline-runs", *[str(ROWS / f"s38_spmeas_s{seed}") for seed in SEEDS],
            "--baseline-cost-row", str(ROWS / "s38_spmeas_tf_pert_main_row.json")]
    return {"arm": arm, "gpu": 0, "cpus": cpus, "argv": argv,
            "log": str(OUTPUT / f"{arm}.report.log"), "pid_file": str(OUTPUT / f"{arm}.report.pid"), "status": "pending"}


def write_reference(expected, phase):
    shared = read_json(ROWS / f"u1a_shared_{phase}_extended.json")["baseline"]["row"]
    untied = read_json(ROWS / f"u1a_untied_{phase}_extended.json")["baseline"]["row"]
    require(shared == untied == expected, "shared/untied report references differ from the preflight fixed S38 row")
    target = ROWS / f"{REFERENCE_ALIAS}_main_row.json"
    if target.exists():
        require(read_json(target) == expected, "existing S38 fixed reference differs; refusing overwrite")
    else:
        with target.open("x", encoding="utf-8") as stream:
            json.dump(shared, stream, indent=2, allow_nan=False)
    return {"path": str(target), "sha256": sha256(target)}


def generate_table(phase, python):
    arms = [S37_ALIAS, REFERENCE_ALIAS, f"u1a_shared_{phase}", f"u1a_untied_{phase}"]
    labels = [f"{S37_ALIAS}=S37（统一3种子 last 参照）",
              f"{REFERENCE_ALIAS}=S38 spmeas（匹配2种子 last）",
              f"u1a_shared_{phase}=U1a shared {phase}（冻结 S38）",
              f"u1a_untied_{phase}=U1a untied {phase}（冻结 S38）"]
    command = [python, "tools/report_table.py", *arms]
    for label in labels:
        command.extend(["--label", label])
    output = subprocess.check_output(command, cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), text=True)
    note = ("\nS37 为统一 3 种子 last 参照；S38 与 U1a 使用匹配的 3407/3408 两种子。"
            "本阶段的配对比较为共享/解共享 U1a；表中未使用历史 selected 结果。\n")
    target = OUTPUT / f"{phase}_table.md"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(output.rstrip() + "\n" + note)
    print(output, flush=True)
    return {"path": str(target), "sha256": sha256(target), "argv": command,
            "rows": {arm: {"path": str(ROWS / f"{arm}_main_row.json"), "sha256": sha256(ROWS / f"{arm}_main_row.json")} for arm in arms}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("2k", "6k"))
    parser.add_argument("--python", default=sys.executable, help="existing EventHandsTrain Python executable")
    parser.add_argument("--threads", type=int, default=2, help="threads per numerical library within each 18-core+SMT partition")
    parser.add_argument("--idle-memory-mb", type=int, default=1024, help="GPU must be below this memory usage and have no compute processes")
    parser.add_argument("--plan", action="store_true", help="print commands/partitions only; no validation, process, GPU query, or writes")
    parser.add_argument("--preflight", action="store_true", help="read-only source/output/resource checks; no locks, output, evaluation or GPU work")
    parser.add_argument("--wait-for-raw", action="store_true", help="wait for this phase's successful raw workers before cost profiling; used by finish_6k")
    parser.add_argument("--raw-wait-timeout-seconds", type=float, default=3600, help="bounded raw barrier wait; default one hour")
    args = parser.parse_args(argv)
    require(1 <= args.threads <= 18, "--threads must be between 1 and 18")
    require(args.idle_memory_mb > 0, "GPU idle memory threshold must be positive")
    require(0 < args.raw_wait_timeout_seconds < float("inf"), "raw wait timeout must be positive and finite")
    budget, jobs = build_plan(args.phase, args.python, threads=args.threads)
    reports = [report_job(mode, args.phase, budget, args.python) for mode in MODES]
    if args.plan:
        print(json.dumps({"phase": args.phase, "budget": budget, "evaluations": jobs, "reports_after_all_evaluations": reports}, indent=2))
        return
    verification = verify_phase(args.phase, budget, jobs)
    untouched_outputs(args.phase, jobs)
    validate_cpu_partitions(jobs)
    baseline, baseline_sources = baseline_preflight()
    require(Path(args.python).is_file(), f"Python executable is unavailable: {args.python}")
    if args.preflight:
        idle = require_idle(GPUS, args.idle_memory_mb)
        print(json.dumps({"preflight": "passed", "phase": args.phase, "budget": budget,
                          "verification": verification, "baseline_sources": baseline_sources,
                          "gpu_idle": idle, "wait_for_raw_before_cost": args.wait_for_raw}, indent=2))
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    state_path = OUTPUT / f"{args.phase}_evaluation_state.json"
    # Shared with run_phase.py: training and evaluation cannot own the phase concurrently.
    with (OUTPUT / "phase.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("phase runner lock is occupied; training/evaluation is still active")
        # Recheck after acquiring ownership; a second runner may have passed its
        # earlier preflight before another invocation created its artifacts.
        untouched_outputs(args.phase, jobs)
        state = {"phase": args.phase, "budget": budget, "runner_pid": os.getpid(), "start_unix": time.time(),
                 "stage": "validated", "verification": verification, "baseline_sources": baseline_sources,
                 "jobs": jobs, "reports": reports, "gpu_idle_before_eval": require_idle(GPUS, args.idle_memory_mb)}
        atomic_state(state_path, state)
        children = []
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        try:
            state["stage"] = "evaluating"
            for job in jobs:
                launch(job, args.threads, state, state_path, children)
            wait_children(children, state, state_path)
            state["all_evaluations_exit_zero_unix"] = time.time()
            if args.wait_for_raw:
                state["stage"] = "waiting_for_raw_before_cost"
                atomic_state(state_path, state)
                state["raw_barrier"] = wait_for_raw(args.phase, verification, args.raw_wait_timeout_seconds)
            # No evaluation process is alive. When requested, raw workers have
            # also exited, so their overlapping CPU partition cannot affect costs.
            for report in reports:
                state["stage"] = f"cost_{report['arm']}"
                state["gpu_idle_before_cost"] = require_idle([0], args.idle_memory_mb)
                process = launch(report, args.threads, state, state_path, children)
                wait_children([(report, process)], state, state_path)
            state["stage"] = "table"
            state["fixed_s38_reference"] = write_reference(baseline, args.phase)
            state["table"] = generate_table(args.phase, args.python)
            state.update(stage="complete", end_unix=time.time())
            atomic_state(state_path, state)
        except BaseException as error:
            terminate_own_children(children)
            state.update(stage="failed", error=str(error), traceback=traceback.format_exc(), end_unix=time.time())
            atomic_state(state_path, state)
            raise


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"U1a phase evaluation refused/failed: {error}")
