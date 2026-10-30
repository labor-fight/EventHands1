#!/usr/bin/env python3
"""Freshly replay the released canonical C1, with fail-closed identity checks.

No training is started until the original complete6000 engineering release is
ready.  This file is an additional entry point; the sealed trainer, evaluator,
controller, final verifier and prediction tracer remain unchanged.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ROOT = ROOT / "outputs/xyz_c1_repro/20261018"
CANONICAL_PROTOCOL_SHA256 = "c0d1e2fa7887342cfdfb4404c532bfdaab5c8aef7e9671495f3b2940850fd44d"
METRICS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")
IDENTITY_CONFIG_FIELDS = (("TRAIN", "RUN_NAME"), ("TRAIN", "OUTPUT_DIR"), ("EVAL", "OUTPUT_DIR"))


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def pair(text):
    if not re.fullmatch(r"\d+,\d+", text):
        raise ValueError("--gpus must name two ordered physical indices, e.g.0,1")
    selected = [int(value) for value in text.split(",")]
    if selected[0] == selected[1]:
        raise ValueError("Two distinct physical GPUs are required")
    return selected


def fresh_output(value, canonical_root=CANONICAL_ROOT):
    raw = Path(value)
    raw = raw if raw.is_absolute() else ROOT / raw
    allowed = canonical_root / "replays"
    if allowed.is_symlink() or raw.is_symlink() or raw.exists():
        raise FileExistsError("Replay requires a new directory; existing paths and symlinks are preserved")
    result = raw.resolve()
    if result.parent != allowed.resolve() or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,55}", result.name):
        raise ValueError("--output-dir must be outputs/xyz_c1_repro/20261018/replays/<new_alphanumeric_name>")
    return result


def require_ready(status):
    if status.get("status") != "COMPLETE_REPRODUCIBLE_BASELINE_AND_ROWS" or status.get("baseline_ready") is not True:
        raise RuntimeError("Canonical complete6000 model/Adam/tape/metrics/prediction verification is not released; no replay will start")


def verify_sources(protocol):
    for name, expected in protocol["file_sha256"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("Frozen source/config/reference changed: " + name)
    for name, expected in protocol["immutable_data_stat"].items():
        value = Path(name).stat()
        if value.st_size != expected["bytes"] or value.st_mtime_ns != expected["mtime_ns"]:
            raise RuntimeError("Frozen data identity changed: " + name)


def find_bound_json(directory, expected):
    matches = [p for p in directory.glob("*.json") if sha(p) == expected]
    if len(matches) != 1:
        raise RuntimeError("Expected one unchanged bound launch acceptance JSON")
    return matches[0]


def validate_acceptance(protocol, protocol_path, acceptance):
    if protocol.get("status") != "FROZEN_READY" or acceptance.get("status") != "PASS":
        raise RuntimeError("Frozen protocol and passed acceptance required")
    if (acceptance["bindings"]["protocol_sha256"] != sha(protocol_path)
            or acceptance["bindings"]["wrapper_sha256"] != sha(ROOT / "tools/xyz_c1_repro/train.py")
            or acceptance["preflight_evidence_sha256"] != protocol["preflight_acceptance_sha256"]
            or sha(ROOT / protocol["preflight_evidence_path"]) != protocol["preflight_acceptance_sha256"]):
        raise RuntimeError("E -> P -> A evidence binding changed")


def load_reference():
    status_path = CANONICAL_ROOT / "status.json"
    status = read(status_path)
    require_ready(status)  # Must run before imports, directory creation or any process launch.
    protocol_path = CANONICAL_ROOT / "protocol.json"
    if sha(protocol_path) != CANONICAL_PROTOCOL_SHA256 or status["protocol_sha256"] != CANONICAL_PROTOCOL_SHA256:
        raise RuntimeError("This entry point supports only the released frozen20261018 protocol")
    protocol = read(protocol_path)
    acceptance_path = find_bound_json(CANONICAL_ROOT, status["preflight_sha256"])
    validate_acceptance(protocol, protocol_path, read(acceptance_path))
    verify_sources(protocol)
    runtime_path = ROOT / protocol["runtime_identity_path"]
    if sha(runtime_path) != protocol["runtime_identity_sha256"]:
        raise RuntimeError("Frozen Python/software/GPU identity changed")
    canonical = next(run for run in protocol["runs"] if run["role"] == "canonical_baseline")
    replay = next(run for run in protocol["runs"] if run["role"] == "exact_fresh_replay")
    if (canonical["name"] != protocol["canonical_baseline"] or canonical["name"] != status["canonical_baseline"]
            or any(run["sampler"] != "legacy_none" for run in (canonical, replay))
            or any(run["seed"] != 3407 or run["fresh_steps"] != 6000 for run in (canonical, replay))):
        raise RuntimeError("Canonical identity, seed or sampler changed")
    acceptance = read(acceptance_path)
    for run in (canonical, replay):
        config_path = (ROOT / run["config"]).resolve()
        if (acceptance["bindings"]["config_sha256"].get(str(config_path)) != sha(config_path)
                or acceptance["bindings"]["sampler_by_config"].get(str(config_path)) != "legacy_none"):
            raise RuntimeError("Canonical config/sampler acceptance changed")
    full_path = CANONICAL_ROOT / "full_reproduction.json"
    full = read(full_path)
    if (sha(full_path) != status["full_reproduction_sha256"] or full["status"] != "PASS" or full["errors"]
            or full["source_sha256"] != sha(ROOT / "tools/xyz_c1_repro/verify_final.py")
            or full["fresh_optimizer_steps"] != 6000 or full["seed"] != 3407
            or full["actual_saved_checkpoint_grid_compared"] != list(range(500, 6001, 500))
            or full["checkpoint_grid_compared"] != list(range(0, 6001, 500))):
        raise RuntimeError("Original complete6000 reproduction evidence is incomplete or changed")
    if ((ROOT / full["canonical"]).resolve() != (ROOT / canonical["output"]).resolve()
            or (ROOT / full["replay"]).resolve() != (ROOT / replay["output"]).resolve()):
        raise RuntimeError("Full verification belongs to different run identities")
    return {"status": status, "status_path": status_path, "protocol": protocol, "protocol_path": protocol_path,
            "acceptance_path": acceptance_path, "runtime": read(runtime_path), "runtime_path": runtime_path,
            "canonical": canonical, "replay": replay, "full_path": full_path, "full": full}


def normalized_recipe(cfg):
    value = copy.deepcopy(cfg)
    value.pop("_config_path", None)
    for section, name in IDENTITY_CONFIG_FIELDS:
        value[section].pop(name, None)
    return value


def replay_config(original, name, output):
    value = copy.deepcopy(original)
    if (value["SEED"] != 3407 or value["TRAIN"]["MAX_STEPS"] != 6000
            or value["TRAIN"]["BATCH_SIZE_PER_GPU"] != 512 or value["TRAIN"]["DEVICES"] != 2
            or value["TRAIN"]["LR"] != 0.004 or value["MODEL"]["ENCODER"] != "xyz_mesh"
            or any(key.startswith("XAG_") for key in value["MODEL"]) or value["DATA"].get("XAG_SOURCE_ROOT")):
        raise RuntimeError("Original fixed3407 C1 recipe required")
    value["TRAIN"]["RUN_NAME"] = name
    value["TRAIN"]["OUTPUT_DIR"] = str(output.relative_to(ROOT))
    value["EVAL"]["OUTPUT_DIR"] = str((output / "eval").relative_to(ROOT))
    if normalized_recipe(original) != normalized_recipe(value):
        raise RuntimeError("Only run identity and output paths may change")
    return value


def gpu_inventory():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    keys = ("index", "uuid", "name", "driver_version", "memory_total_mib")
    return [dict(zip(keys, [v.strip() for v in row])) for row in csv.reader(result.stdout.splitlines())]


def compare_runtime(expected, actual):
    differences = [name for name in expected if name != "scope" and expected[name] != actual.get(name)]
    if differences:
        raise RuntimeError("Original environment/hardware changed; exact replay is unsupported: " + ", ".join(differences))


def collect_runtime(expected):
    # The exact accepted policy is enabled before any CUDA work. This process
    # never loads a model or performs a training/evaluation query.
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    math_flags = {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                  "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                  "cudnn_deterministic": torch.backends.cudnn.deterministic,
                  "cudnn_benchmark": torch.backends.cudnn.benchmark,
                  "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                  "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                  "matmul_precision": torch.get_float32_matmul_precision(),
                  "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                  "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}
    return {"python_version": platform.python_version(), "python_executable": str(Path(sys.executable).resolve()),
            "python_full": sys.version, "packages": {key: importlib.metadata.version(key) for key in expected["packages"]},
            "torch": torch.__version__, "lightning": importlib.metadata.version("pytorch-lightning"),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "nccl": list(torch.cuda.nccl.version()),
            "math_flags": math_flags, "environment": {key: os.environ.get(key) for key in expected["environment"]},
            "gpu_inventory": gpu_inventory(), "operating_system": platform.platform()}


def validate_reference_products(reference):
    import numpy as np
    for name, digest in reference["full"]["file_sha256"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("Canonical full6000 model/Adam/tape evidence changed: " + name)
    rows, traces = [], []
    for run in (reference["canonical"], reference["replay"]):
        directory = ROOT / run["output"]
        for rank in (0, 1):
            done = read(directory / f"completion_rank{rank}.json")
            runtime = read(directory / f"effective_runtime_rank{rank}.json")
            if (done["status"] != "COMPLETE" or done["step"] != 6000 or done["batches"] != 6000
                    or runtime["sampler_policy"] != "legacy_none" or runtime["effective_seed"] != 3407
                    or runtime["actual_train_loader"]["underlying_generator"] is not None
                    or runtime["actual_math_flags"] != reference["protocol"]["math_policy"]):
                raise RuntimeError("Canonical terminal runtime/budget is not certified")
        row_path = ROOT / reference["status"]["runs"][run["name"]]["main_row"]
        receipt = read(row_path.with_name(row_path.stem + "_receipt.json"))
        if (receipt["status"] != "COMPLETE" or receipt["row_sha256"] != sha(row_path)
                or receipt["wrapper_sha256"] != sha(ROOT / "tools/xyz_c1_repro/evaluate.py")):
            raise RuntimeError("Canonical main-row receipt changed")
        row = read(row_path)
        if set(row["per_seed"]) != {"3407"} or row["per_seed"]["3407"]["n_frames"] != 2590:
            raise RuntimeError("Canonical full2590 single-seed row required")
        for block in ("overall", "local", "global"):
            for metric in METRICS:
                if (not math.isfinite(row["two_seed_mean"][block][metric])
                        or row["two_seed_mean"][block][metric] != row["per_seed"]["3407"][block][metric]):
                    raise RuntimeError("Canonical single-seed metric schema changed")
        trace_path = directory / "strict_trace/report.json"
        trace = read(trace_path)
        if trace["status"] != "COMPLETE" or trace["n_endpoints"] != 2590 or trace["states_sha256"] != sha(directory / "strict_trace/states.npz"):
            raise RuntimeError("Canonical prediction trace changed")
        checkpoints = list(directory.glob("*step=6000.ckpt"))
        if (len(checkpoints) != 1 or trace["checkpoint_sha256"] != sha(checkpoints[0])
                or trace["config_sha256"] != sha(ROOT / run["config"])
                or trace["source_sha256"] != sha(ROOT / "tools/xyz_c1_repro/trace.py")):
            raise RuntimeError("Canonical trace weight/config/producer identity changed")
        for sequence in trace["per_sequence"]:
            for metric in METRICS:
                if sequence[metric] != row["per_seed"]["3407"]["per_seq"][sequence["seq"]][metric]:
                    raise RuntimeError("Canonical trace/row accuracy differs")
        rows.append(row)
        traces.append(np.load(directory / "strict_trace/states.npz"))
    if rows[0]["two_seed_mean"] != rows[1]["two_seed_mean"]:
        raise RuntimeError("Original canonical/replay metrics differ")
    compare_traces(traces[0], traces[1])
    return rows[0]


def compare_traces(a, b):
    import numpy as np
    expected = {f"{seq}/{name}" for seq in ("zgz_global", "zgz_local") for name in ("prediction", "incoming")}
    if set(a.files) != expected or set(b.files) != expected:
        raise RuntimeError("Full trace inventory changed")
    for key in expected:
        n = 1386 if key.startswith("zgz_global/") else 1204
        if (a[key].shape != (n, 51) or a[key].shape != b[key].shape or a[key].dtype != b[key].dtype
                or not np.isfinite(a[key]).all() or a[key].tobytes() != b[key].tobytes()):
            raise RuntimeError("Full2590 incoming/prediction replay differs: " + key)


def memory():
    values = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()}
    vm = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    return values["MemAvailable"], int(vm["oom_kill"])


def ensure_idle(gpus, inventory):
    selected = {row["uuid"] for row in inventory if int(row["index"]) in gpus}
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    occupied = [row for row in csv.reader(result.stdout.splitlines()) if row and row[0].strip() in selected and int(row[1]) != os.getpid()]
    if occupied:
        raise RuntimeError("Selected GPUs are busy; no fresh replay launched: " + repr(occupied))


def stop_owned(processes, timeout=20):
    """Stop our saved process groups even after a DDP launcher has exited."""
    def alive(pgid):
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    while any(alive(p.pid) for p in processes) and time.monotonic() < deadline:
        time.sleep(.1)
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def prepare_bundle(reference, output, gpus, cfg):
    import yaml
    run_name = "xyz_c1_user_" + output.name + "_s3407"
    alias = ROOT / "outputs/semkine" / run_name
    if alias.exists() or alias.is_symlink():
        raise FileExistsError("Existing run namespace is preserved: " + str(alias))
    run_output = output / "run"
    config = replay_config(cfg, run_name, run_output)
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    protocol = copy.deepcopy(reference["protocol"])
    run = {"name": run_name, "role": "exact_user_fresh_replay", "sampler": "legacy_none", "physical_gpus": gpus,
           "config": str(config_path.relative_to(ROOT)), "output": str(run_output.relative_to(ROOT)), "seed": 3407, "fresh_steps": 6000}
    protocol.update(created_at_utc=now(), study="XYZ_C1_REPRO_20261018_USER_FRESH_REPLAY", runs=[run])
    protocol["replay_reference"] = {"canonical": reference["canonical"], "protocol_sha256": sha(reference["protocol_path"]),
                                    "complete_release_sha256": sha(reference["status_path"]),
                                    "full_reproduction_sha256": sha(reference["full_path"]),
                                    "only_config_identity_and_output_paths_changed": True,
                                    "no_new_seed_or_score_selection": True, "not_an_accuracy_average": True}
    for path in (config_path, Path(__file__), reference["protocol_path"], reference["acceptance_path"],
                 reference["status_path"], reference["full_path"]):
        protocol["file_sha256"][str(path.relative_to(ROOT))] = sha(path)
    protocol_path = output / "protocol.json"
    save(protocol_path, protocol)
    acceptance = {"status": "PASS", "preflight_evidence_sha256": protocol["preflight_acceptance_sha256"],
                  "scope": "Inherited completed canonical release; unchanged accepted recipe/source/runtime, fresh identity paths only",
                  "parent_launch_acceptance_sha256": sha(reference["acceptance_path"]),
                  "canonical_complete6000_verification_sha256": sha(reference["full_path"]),
                  "bindings": {"wrapper_sha256": sha(ROOT / "tools/xyz_c1_repro/train.py"),
                               "protocol_sha256": sha(protocol_path), "config_sha256": {str(config_path): sha(config_path)},
                               "sampler_by_config": {str(config_path): "legacy_none"}}}
    acceptance_path = output / "launch_acceptance.json"
    save(acceptance_path, acceptance)
    validate_acceptance(protocol, protocol_path, acceptance)
    return run, protocol, protocol_path, acceptance_path, alias


def execute_replay(reference, output, gpus, cfg, canonical_row):
    run, protocol, protocol_path, acceptance_path, alias = prepare_bundle(reference, output, gpus, cfg)
    protocol_digest, acceptance_digest = sha(protocol_path), sha(acceptance_path)
    status = {"status": "TRAINING_FRESH_REPLAY", "started_at_utc": now(), "baseline_ready": False,
              "canonical_reference": reference["canonical"]["name"], "run": run,
              "protocol_sha256": protocol_digest, "launch_acceptance_sha256": acceptance_digest}
    status_path = output / "status.json"
    save(status_path, status)
    _, oom_start = memory()
    process = None
    processes = []
    def frozen_guard():
        if sha(protocol_path) != protocol_digest or sha(acceptance_path) != acceptance_digest:
            raise RuntimeError("Replay protocol or launch acceptance changed")
        verify_sources(protocol)
    def interrupted(number, frame):
        raise InterruptedError("Fresh replay interrupted by signal " + str(number))
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    def owned(tag, command, visible):
        nonlocal process
        frozen_guard()
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=visible)
        low = 0
        with (output / (tag + ".log")).open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            status.update(active_job=tag, pid=process.pid, command=command)
            while process.poll() is None:
                frozen_guard()
                available, oom = memory()
                low = low + 1 if available < 16 * 1024 ** 3 else 0
                if low >= 2 or oom != oom_start:
                    raise RuntimeError("Owned replay stopped: sustained low host memory or new oom_kill")
                status.update(updated_at_utc=now(), available_memory_bytes=available)
                progress = ROOT / run["output"] / "progress_rank0.json"
                if progress.is_file():
                    status["training_progress"] = read(progress)
                save(status_path, status)
                time.sleep(5)
            if process.returncode:
                raise RuntimeError(tag + " failed with exit " + str(process.returncode))
        if memory()[1] != oom_start:
            raise RuntimeError("Owned replay stopped: new oom_kill at process completion")
        frozen_guard()
    try:
        owned("train", [sys.executable, "tools/xyz_c1_repro/train.py", "--phase", "formal", "--sampler", "legacy_none",
                        "--protocol", str(protocol_path), "--acceptance", str(acceptance_path), "--config", run["config"]],
              ",".join(map(str, gpus)))
        status["status"] = "VERIFYING_COMPLETE6000"
        save(status_path, status)
        owned("verify", [sys.executable, "tools/xyz_c1_repro/verify_final.py", reference["canonical"]["output"], run["output"],
                         "--output", str(output / "full_reproduction.json")], str(gpus[0]))
        if read(output / "full_reproduction.json")["status"] != "PASS":
            raise RuntimeError("Complete model/Adam/tape/checkpoint verification did not pass")
        if alias.exists() or alias.is_symlink():
            raise FileExistsError("New namespace collision; preserve all outputs")
        alias.symlink_to(ROOT / run["output"], target_is_directory=True)
        prefix = run["name"].removesuffix("_s3407")
        owned("main_row", [sys.executable, "tools/xyz_c1_repro/evaluate.py", "--run", prefix, "--step", "6000", "--seeds", "3407",
                           "--output-dir", str(output / "main_rows")], str(gpus[0]))
        row_path = output / "main_rows" / (prefix + "_main_row_step6000.json")
        row = read(row_path)
        if row["two_seed_mean"] != canonical_row["two_seed_mean"]:
            raise RuntimeError("Fresh full2590 metrics differ from canonical; no result selection")
        checkpoints = list((ROOT / run["output"]).glob("*step=6000.ckpt"))
        if len(checkpoints) != 1:
            raise RuntimeError("Unique fixed6000 checkpoint required")
        owned("trace", [sys.executable, "tools/xyz_c1_repro/trace.py", "--config", run["config"], "--checkpoint", str(checkpoints[0]),
                        "--output-dir", str(ROOT / run["output"] / "strict_trace")], str(gpus[0]))
        import numpy as np
        a = np.load(ROOT / reference["canonical"]["output"] / "strict_trace/states.npz")
        b = np.load(ROOT / run["output"] / "strict_trace/states.npz")
        compare_traces(a, b)
        for sequence in read(ROOT / run["output"] / "strict_trace/report.json")["per_sequence"]:
            for metric in METRICS:
                if sequence[metric] != row["per_seed"]["3407"]["per_seq"][sequence["seq"]][metric]:
                    raise RuntimeError("Fresh trace does not reproduce fresh main row")
        validate_reference_products(reference)
        frozen_guard()
        compare_runtime(reference["runtime"], collect_runtime(reference["runtime"]))
        status.update(status="COMPLETE_EXACT_FRESH_REPLAY", baseline_ready=True, completed_at_utc=now(),
                      full_reproduction_sha256=sha(output / "full_reproduction.json"), main_row=str(row_path.relative_to(ROOT)),
                      main_row_sha256=sha(row_path), replay_is_not_averaged=True,
                      scope="Exact full6000 and full2590 engineering replay on the unchanged frozen host/software only")
        save(status_path, status)
        inventory = {str(path.relative_to(output)): sha(path) for path in output.rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts and path.name != "MANIFEST.json"}
        save(output / "MANIFEST.json", {"captured_at_utc": now(), "files": inventory})
        print(json.dumps({"status": status["status"], "main_row": status["main_row"], "status_file": str(status_path)}))
    except BaseException as error:
        stop_owned(processes)
        status.update(status="FAILED_PRESERVED", baseline_ready=False, updated_at_utc=now(), error=str(error), error_type=type(error).__name__)
        save(status_path, status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args(argv)
    gpus = pair(args.gpus)
    output = fresh_output(args.output_dir)
    reference = load_reference()
    if gpus not in reference["protocol"]["resource_policy"]["physical_pairs"]:
        raise ValueError("Choose an original ordered, preflighted physical GPU pair")
    for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "NODE_RANK"):
        if name in os.environ:
            raise RuntimeError("Run this fresh entry point from an ordinary non-DDP shell")
    expected = reference["runtime"]
    if str(Path(sys.executable).resolve()) != expected["python_executable"] or sys.version != expected["python_full"]:
        raise RuntimeError("Original Python interpreter/version required")
    policy = reference["protocol"]["environment_policy"]
    needs_exec = os.environ.get("PYTHONHASHSEED") != policy["PYTHONHASHSEED"]
    os.environ.update(policy)
    if needs_exec:
        os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ))
    actual = collect_runtime(expected)
    compare_runtime(expected, actual)
    ensure_idle(gpus, actual["gpu_inventory"])
    if memory()[0] < 16 * 1024 ** 3:
        raise RuntimeError("Insufficient available host memory; no replay launched")
    canonical_row = validate_reference_products(reference)
    import yaml
    cfg = yaml.safe_load((ROOT / reference["canonical"]["config"]).read_text())
    execute_replay(reference, output, gpus, cfg, canonical_row)


if __name__ == "__main__":
    main()
