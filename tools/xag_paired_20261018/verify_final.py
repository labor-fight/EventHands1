#!/usr/bin/env python3
"""Read-only contract verification for the five fresh, single-seed paired arms.

This module has no torch/CUDA/import-time policy effects. It validates one fresh
training trajectory against C1's actual input tape, never against its scores.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/xag_paired/20261018"
C1 = ROOT / "outputs/xyz_c1_repro/20261018"
CELLS = {"a0f1": ("base", "global_residual"), "a1f0": ("part_route", "legacy"),
         "a1f1": ("part_route", "global_residual"), "a2f0": ("face_route", "legacy"),
         "a2f1": ("face_route", "global_residual")}
GRID = tuple(range(0, 6001, 500))
SOURCE_FIELDS = ("xag_source_index", "xag_source_ms", "xag_part", "xag_face", "xag_bary", "xag_valid")
FIELDS = ("sequence_id", "t_start_us", "t_end_us", "target", "prev_state", "betas", "camera_K",
          "ptr", "delta_t_s", "is_sequence_start", "is_sequence_end")
RAW_FIELDS = ("events", *FIELDS, "lnes", "lnes_window_ms", *SOURCE_FIELDS)
C1_PROTOCOL_SHA = "c0d1e2fa7887342cfdfb4404c532bfdaab5c8aef7e9671495f3b2940850fd44d"
MATH = {"deterministic_algorithms": True, "warn_only": False, "cudnn_deterministic": True,
        "cudnn_benchmark": False, "matmul_allow_tf32": False, "cudnn_allow_tf32": False,
        "matmul_precision": "highest", "bf16_reduced_precision_reduction": False,
        "fp16_reduced_precision_reduction": False}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def path(value):
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def save_new(destination, value):
    destination = Path(destination)
    require(destination.resolve().is_relative_to(OUT), "Write only within the new paired study")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checked_file(value, expected, records=None):
    value = path(value)
    before = value.stat()
    actual = sha(value)
    after = value.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "File changed while hashing: " + str(value))
    require(actual == expected, "SHA mismatch: " + str(value))
    if records is not None:
        records[str(value)] = actual
    return value


def verify_protocol(protocol_path, *, full_sources=False):
    protocol_path = path(protocol_path).resolve()
    require(protocol_path.is_relative_to(OUT) and protocol_path.name == "protocol.json", "Explicit new paired protocol required")
    protocol = read(protocol_path)
    require(protocol.get("status") == "FROZEN_READY", "Formal paired protocol is not frozen/ready")
    require(protocol.get("canonical_baseline") == "xyz_c1_stable_s3407", "Predesignated C1 reference changed")
    require(protocol.get("math_policy") == MATH, "Strict math policy differs from C1")
    runs = protocol["runs"]
    require(len(runs) == 5 and {r["cell"] for r in runs} == set(CELLS), "Exactly the five approved arms are required")
    require(len({r["name"] for r in runs}) == 5, "Duplicate run identity")
    for run in runs:
        name = f"xag_paired_{run['cell']}_s3407"
        require(run["name"] == name and run["prefix"] == name.removesuffix("_s3407")
                and run["seed"] == 3407 and run["fresh_steps"] == 6000 and run["sampler"] == "legacy_none"
                and path(run["output"]).resolve() == OUT / "runs" / name, "Run identity/budget/sampler changed")
        require(sha(path(run["config"])) == protocol["file_sha256"][run["config"]], "New config is not frozen")
    for name, expected in protocol["file_sha256"].items():
        checked_file(name, expected)
    for name, expected in protocol["immutable_data_stat"].items():
        stat = path(name).stat()
        require((stat.st_size, stat.st_mtime_ns) == (expected["bytes"], expected["mtime_ns"]), "Data stat changed: " + name)
    checked_file(protocol["baseline_protocol_path"], protocol["baseline_protocol_sha256"])
    require(protocol["baseline_protocol_sha256"] == C1_PROTOCOL_SHA, "Different baseline protocol")
    baseline_protocol = read(path(protocol["baseline_protocol_path"]))
    require(len(baseline_protocol["file_sha256"]) == 213
            and all(protocol["file_sha256"].get(k) == v for k, v in baseline_protocol["file_sha256"].items()),
            "The complete original213 source/config contract must be inherited")
    require(len(protocol["immutable_data_stat"]) == 820
            and protocol["immutable_data_stat"] == baseline_protocol["immutable_data_stat"],
            "The original820 immutable data stat contract changed")
    require(protocol["environment_policy"] == baseline_protocol["environment_policy"], "Different strict training environment")
    checked_file(protocol["baseline_identity_path"], protocol["baseline_identity_sha256"])
    baseline = read(path(protocol["baseline_identity_path"]))
    require(baseline["status"] == "ACTIVE_REPRODUCIBLE_C1_BASELINE"
            and baseline["run"] == "xyz_c1_stable_s3407" and baseline["sampler"] == "legacy_none"
            and baseline["step"] == 6000 and baseline["model_seed"] == 3407, "Different active baseline identity")
    require(baseline["protocol_sha256"] == C1_PROTOCOL_SHA, "Active baseline protocol binding changed")
    checked_file(baseline["checkpoint"], baseline["checkpoint_sha256"])
    checked_file(baseline["main_row"], baseline["main_row_sha256"])
    checked_file(baseline["full_reproduction"], baseline["full_reproduction_sha256"])
    proof = read(path(baseline["full_reproduction"]))
    require(proof["status"] == "PASS" and proof["seed"] == 3407 and proof["fresh_optimizer_steps"] == 6000,
            "Canonical complete6000 replay proof is not PASS")
    require(path(protocol["reference_baseline_output"]).resolve() == C1 / "runs/xyz_c1_stable_s3407", "Different baseline input tape")
    checked_file(protocol["source_manifest_path"], protocol["source_manifest_sha256"])
    require(path(protocol["source_manifest_path"]).resolve() == ROOT / "outputs/xag/source_manifest.json", "Different GT source manifest")
    source = read(path(protocol["source_manifest_path"]))
    require(source["complete"] is True, "Incomplete GT source manifest")
    for name, expected in {**source["files"], **source["inputs"]}.items():
        source_path = path(name); stat = source_path.stat()
        require((stat.st_size, stat.st_mtime_ns) == (expected["bytes"], expected["mtime_ns"]), "Original GT source/input stat changed: " + name)
        if full_sources:
            checked_file(source_path, expected["sha256"])
    return protocol


def resolve_run(protocol, value):
    found = [r for r in protocol["runs"] if value in (r["name"], r["prefix"], r["cell"])]
    require(len(found) == 1, "One explicitly registered new paired run is required")
    return found[0]


def load_recipe(run):
    sys.path[:0] = [str(ROOT / "model"), str(ROOT)]
    from config import load_config
    actual = load_config(path(run["config"]))
    expected = load_config(ROOT / "configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml")
    expected["MODEL"].update(zip(("XAG_ASSOC", "XAG_GNN"), CELLS[run["cell"]]))
    expected["DATA"]["XAG_SOURCE_ROOT"] = str(ROOT / "outputs/xag/sources")
    expected["TRAIN"].update(RUN_NAME=run["name"], OUTPUT_DIR=str(path(run["output"]).resolve()))
    expected["EVAL"]["OUTPUT_DIR"] = str(path(run["output"]).resolve() / "eval")
    require({k: v for k, v in actual.items() if k != "_config_path"} == {k: v for k, v in expected.items() if k != "_config_path"},
            "Recipe differs outside approved A/F switches, source metadata and output identity")
    return actual


def normalized(value):
    import torch
    if torch.is_tensor(value):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return {"tensor_sha256": hashlib.sha256(raw).hexdigest(), "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): normalized(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalized(v) for v in value]
    return value


def check_bindings(protocol_path, protocol, run, records=None):
    directory = path(run["output"])
    source_old = read(directory / "xag_binding.json")
    expected_old = {"protocol_sha256": sha(ROOT / "outputs/xag/protocol.json"),
                    "frozen_sources_sha256": sha(ROOT / "outputs/xag/frozen_sources.json"),
                    "source_manifest_sha256": protocol["source_manifest_sha256"], "config_sha256": sha(path(run["config"]))}
    require(source_old == expected_old, "Original XAG source-implementation binding changed")
    bindings = []
    for rank in (0, 1):
        binding_path = directory / f"paired_binding_rank{rank}.json"
        binding = read(binding_path)
        require(binding["cell"] == run["cell"] and binding["sampler"] == "legacy_none" and binding["phase"] == "formal"
                and binding["protocol_sha256"] == sha(protocol_path) and binding["config_sha256"] == sha(path(run["config"]))
                and binding["source_manifest_sha256"] == protocol["source_manifest_sha256"]
                and binding["canonical_baseline_sha256"] == protocol["baseline_identity_sha256"]
                and binding["wrapper_sha256"] == sha(ROOT / "tools/xag_paired_20261018/train.py")
                and binding["reused_helper_sha256"] == sha(ROOT / "tools/xyz_c1_repro/train.py")
                and binding["frozen_sources_sha256"] == sha(ROOT / "outputs/xag/frozen_sources.json"), "New paired launch binding mismatch")
        acceptance_path = path(binding["acceptance"])
        acceptance_sha = sha(acceptance_path)
        require(binding.get("launch_acceptance_sha256", binding.get("acceptance_sha256")) == acceptance_sha, "Paired acceptance identity changed")
        acceptance = read(acceptance_path)
        require(acceptance["status"] == "PASS" and acceptance["bindings"]["protocol_sha256"] == sha(protocol_path)
                and acceptance["preflight_evidence_sha256"] == protocol["preflight_acceptance_sha256"], "New E -> P -> A mismatch")
        expected_acceptance = {"wrapper_sha256": binding["wrapper_sha256"], "helper_sha256": binding["reused_helper_sha256"],
                               "canonical_config_sha256": sha(ROOT / "configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml"),
                               "canonical_baseline_sha256": protocol["baseline_identity_sha256"],
                               "source_manifest_sha256": protocol["source_manifest_sha256"]}
        require(all(acceptance["bindings"].get(k) == v for k, v in expected_acceptance.items()), "Acceptance source/baseline binding changed")
        config_key = str(path(run["config"]).resolve())
        require(acceptance["bindings"]["config_sha256"][config_key] == sha(path(run["config"]))
                and acceptance["bindings"]["sampler_by_config"][config_key] == "legacy_none"
                and acceptance["bindings"]["cell_by_config"][config_key] == run["cell"]
                and acceptance["bindings"]["modes_by_config"][config_key] == list(CELLS[run["cell"]]), "Acceptance config/cell/mode mismatch")
        checked_file(protocol["preflight_evidence_path"], protocol["preflight_acceptance_sha256"], records)
        require(read(path(protocol["preflight_evidence_path"]))["status"] == "PASS", "Physical preflight evidence not PASS")
        require(binding.get("original_xag_source_binding_sha256") == sha(directory / "xag_binding.json"), "Old source provenance not bound")
        marker_path = directory / ".fresh_claim.json"
        marker = read(marker_path)
        require(binding["fresh_output_claim_sha256"] == sha(marker_path)
                and marker["schema"] == "xag-paired-fresh-output-v1"
                and Path(marker["output_dir"]).resolve() == directory.resolve()
                and marker["cell"] == run["cell"] and marker["phase"] == "formal" and marker["sampler"] == "legacy_none"
                and Path(marker["config_path"]).resolve() == path(run["config"]).resolve()
                and marker["config_sha256"] == sha(path(run["config"]))
                and Path(marker["protocol_path"]).resolve() == path(protocol_path).resolve()
                and marker["protocol_sha256"] == sha(protocol_path)
                and marker["wrapper_sha256"] == binding["wrapper_sha256"]
                and bool(re.fullmatch("[a-f0-9]{64}", marker["nonce"]))
                and isinstance(marker["parent_pid"], int) and marker["parent_pid"] > 1,
                "Fresh exclusive output ownership marker mismatch")
        if records is not None:
            records[str(binding_path)] = sha(binding_path); records[str(acceptance_path)] = acceptance_sha
            records[str(marker_path)] = sha(marker_path)
        bindings.append(binding)
    require(bindings[0] == bindings[1], "Paired launch identity differs between ranks")
    return bindings[0]


def check_checkpoint(protocol_path, protocol, run, checkpoint=None, *, final=False, records=None):
    import torch
    cfg = load_recipe(run); directory = path(run["output"])
    if checkpoint is None:
        candidates = list(directory.glob("*step=6000.ckpt"))
        require(len(candidates) == 1, "Unique fixed6000 checkpoint required")
        checkpoint = candidates[0]
    checkpoint = Path(checkpoint)
    require(checkpoint.parent.resolve() == directory.resolve(), "Checkpoint belongs to another run")
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(obj["global_step"] == 6000 and obj["epoch"] == (2 if final else 1), "Checkpoint step/terminal epoch changed")
    require({k: v for k, v in obj["hyper_parameters"].items() if k != "_config_path"}
            == {k: v for k, v in cfg.items() if k != "_config_path"}, "Checkpoint hyperparameters differ from registered recipe")
    require(all(not torch.is_tensor(v) or bool(torch.isfinite(v).all()) for v in obj["state_dict"].values()), "Nonfinite model state")
    require(len(obj["optimizer_states"]) == len(obj["lr_schedulers"]) == 1, "Single original Adam/scheduler required")
    optimizer = obj["optimizer_states"][0]
    require(set(optimizer["state"]) == {p for group in optimizer["param_groups"] for p in group["params"]} and bool(optimizer["state"]), "Incomplete Adam parameter state")
    require(all(group["lr"] == .004 for group in optimizer["param_groups"]) and obj["lr_schedulers"][0]["last_epoch"] == 6000, "Optimizer/schedule contract changed")
    for state in optimizer["state"].values():
        require(float(state["step"]) == 6000 and {"exp_avg", "exp_avg_sq"}.issubset(state), "Actual Adam step/state changed")
        require(all(not torch.is_tensor(v) or bool(torch.isfinite(v).all()) for v in state.values()), "Nonfinite optimizer state")
    meta = read(directory / "training_metadata.json")
    require(meta["run_name"] == run["name"] and meta["seed"] == 3407 and meta["resumed_from"] is None
            and meta["max_steps"] == 6000 and meta["effective_batch"] == 1024
            and Path(meta["config_path"]).resolve() == path(run["config"]).resolve(), "Training metadata identity differs")
    check_bindings(protocol_path, protocol, run, records)
    if records is not None:
        records[str(checkpoint)] = sha(checkpoint)
        for observed in (directory / "training_metadata.json", directory / "xag_binding.json"):
            records[str(observed)] = sha(observed)
    return checkpoint, obj


def initial_shared(new, baseline, cell):
    require(all(new.get(k) == v for k, v in baseline.items()), "Initial shared model tensors differ from canonical C1")
    extra = set(new) - set(baseline)
    shapes = {"norm1.weight": [128], "norm1.bias": [128], "identity.weight": [128, 16],
              "attention.in_proj_weight": [384, 128], "attention.in_proj_bias": [384],
              "attention.out_proj.weight": [128, 128], "attention.out_proj.bias": [128],
              "norm2.weight": [128], "norm2.bias": [128], "ff.0.weight": [256, 128],
              "ff.0.bias": [256], "ff.2.weight": [128, 256], "ff.2.bias": [128]}
    expected = {f"xyz.xag_global.{i}.{name}": shape for i in range(3) for name, shape in shapes.items()} if cell.endswith("f1") else {}
    require(extra == set(expected), "Unregistered extra initial tensors")
    require(all(new[k]["shape"] == v and new[k]["dtype"] == "torch.float32" for k, v in expected.items()), "Attention-block tensor shape changed")
    require(all(bool(re.fullmatch("[a-f0-9]{64}", new[k]["tensor_sha256"])) for k in expected), "Missing extra tensor fingerprint")


def check_gradient_record(row, reference, initial, baseline_initial, cell, selected):
    """Require actual predictions and all trainable gradients at native cadence."""
    if not selected:
        require("gradients" not in row and "prediction_sha256" not in row, "Unexpected gradient/prediction cadence")
        return
    expected = {name: {key: value[key] for key in ("shape", "dtype")}
                for name, value in reference["gradients"].items()}
    require(len(expected) == 25, "Canonical trainable inventory changed")
    extra = set(initial) - set(baseline_initial)
    require(len(extra) == (39 if cell.endswith("f1") else 0), "Added gradient inventory changed")
    expected.update({name: {key: initial[name][key] for key in ("shape", "dtype")} for name in extra})
    require(set(row["gradients"]) == set(expected), "Selected batch trainable gradient inventory incomplete")
    for name, metadata in expected.items():
        value = row["gradients"][name]
        require(isinstance(value, dict) and set(value) == {"tensor_sha256", "shape", "dtype"}
                and all(value[key] == item for key, item in metadata.items())
                and bool(re.fullmatch("[a-f0-9]{64}", value["tensor_sha256"])), "Gradient fingerprint/shape/dtype missing: " + name)
    require(bool(re.fullmatch("[a-f0-9]{64}", row["prediction_sha256"])), "Missing actual prediction fingerprint")


def verify_run(protocol_path, protocol, run):
    import numpy as np
    import torch
    records, checks, snapshots = {}, [], []
    directory = path(run["output"]); baseline = path(protocol["reference_baseline_output"])
    cfg = load_recipe(run)
    for rank in (0, 1):
        runtime_path = directory / f"effective_runtime_rank{rank}.json"; runtime = read(runtime_path)
        runtime_base = read(baseline / f"effective_runtime_rank{rank}.json")
        loader = runtime["actual_train_loader"]
        require(runtime["phase"] == "formal" and runtime["sampler_policy"] == "legacy_none" and runtime["actual_math_flags"] == MATH
                and runtime["effective_seed"] == 3407 and runtime["effective_max_steps"] == 6000
                and runtime["effective_accumulate_grad_batches"] == 1 and runtime["world_size"] == 2
                and runtime["precision"] == "bf16" and runtime["epoch_batches"] == 4437
                and runtime["global_rank"] == rank and runtime["local_rank"] == rank, "Effective runtime/strict budget mismatch")
        require(loader == runtime_base["actual_train_loader"] and loader["underlying_generator"] is None
                and loader["loader_generator"] is None and loader["batch_size"] == 512, "Actual canonical None loader/sampler differs")
        require(runtime["environment"] == runtime_base["environment"] == protocol["environment_policy"], "Strict environment changed")
        require(runtime["config_sha256"] == sha(path(run["config"])) and runtime["protocol_sha256"] == sha(protocol_path)
                and runtime["wrapper_sha256"] == sha(ROOT / "tools/xag_paired_20261018/train.py")
                and runtime["reused_helper_sha256"] == sha(ROOT / "tools/xyz_c1_repro/train.py")
                and runtime["original_entry_sha256"] == sha(ROOT / "semkine/train.py")
                and runtime["canonical_baseline_sha256"] == protocol["baseline_identity_sha256"]
                and runtime["source_manifest_sha256"] == protocol["source_manifest_sha256"]
                and runtime["cell"] == run["cell"] and runtime["xag_modes"] == list(CELLS[run["cell"]])
                and runtime["fresh_only"] is True and runtime["resume_supported"] is False, "Runtime provenance differs")
        physical = run["physical_gpus"]
        require(isinstance(physical, list) and len(physical) == len(set(physical)) == 2
                and runtime["cuda_visible_devices"] == ",".join(str(gpu) for gpu in physical), "Registered physical GPU pair differs")
        require(Path(runtime["fresh_output_claim_path"]).resolve() == directory.resolve() / ".fresh_claim.json"
                and runtime["fresh_output_claim_sha256"] == sha(directory / ".fresh_claim.json"), "Runtime fresh ownership marker mismatch")
        for key in ("torch", "lightning", "cuda", "cudnn", "nccl", "gpu"):
            require(runtime[key] == runtime_base[key], "Baseline software/GPU model identity differs: " + key)
        tape_path = directory / f"indices_rank{rank}.i32"; base_tape = baseline / f"indices_rank{rank}.i32"
        require(tape_path.stat().st_size == 6000 * 512 * 4 and tape_path.read_bytes() == base_tape.read_bytes(), "All6000 actual indices differ from canonical baseline")
        tape = np.fromfile(tape_path, dtype=np.int32).reshape(6000, 512)
        require(tape.min() >= 0 and tape.max() < 4544300, "Actual index outside official dataset")
        journal_path = directory / f"batches_rank{rank}.jsonl"; base_journal = baseline / f"batches_rank{rank}.jsonl"
        rows = [json.loads(v) for v in journal_path.read_text().splitlines()]; old = [json.loads(v) for v in base_journal.read_text().splitlines()]
        require(len(rows) == len(old) == 6000, "Complete6000 journals required")
        initial = read(directory / f"state_rank{rank}_step000000.json")["model"]
        baseline_initial = read(baseline / f"state_rank{rank}_step000000.json")["model"]
        initial_shared(initial, baseline_initial, run["cell"])
        h = hashlib.sha256(); prefix = {0: h.hexdigest()}; raw_count = 0
        for step, (row, reference) in enumerate(zip(rows, old), 1):
            require(row["step"] == step and row["batch_size"] == 512 and row["epoch"] == (step - 1) // 4437, "Journal step/epoch changed")
            require(row["indices_sha256"] == hashlib.sha256(tape[step - 1].tobytes()).hexdigest(), "Journal index hash differs from actual tape")
            for field in ("indices_sha256", "small_fields_sha256", "full_packet_sha256", "events"):
                require(row[field] == reference[field], f"Canonical public packet field differs rank{rank}/step{step}/{field}")
            raw = step <= 3 or step % 500 in (0, 1, 499) or step == 6000 or step % 4437 in (0, 1, 4436)
            for field in ("objective_sha256", "small_fields_sha256"):
                require(bool(re.fullmatch("[a-f0-9]{64}", row[field])), "Missing real objective/input fingerprint")
            for field in ("xag_full_packet_sha256", "xag_source_fields_sha256"):
                require((isinstance(row.get(field), str) and bool(re.fullmatch("[a-f0-9]{64}", row[field]))) if raw else row.get(field) is None,
                        "Actual GT-source fingerprint cadence missing")
            raw_count += int(raw)
            check_gradient_record(row, old[0], initial, baseline_initial, run["cell"], raw)
            identity = {key: row[key] for key in ("step", "indices_sha256", "small_fields_sha256", "full_packet_sha256")}
            h.update(json.dumps(identity, sort_keys=True).encode())
            if step in GRID: prefix[step] = h.hexdigest()
        done_path = directory / f"completion_rank{rank}.json"; done = read(done_path)
        require(done["status"] == "COMPLETE" and done["step"] == done["batches"] == 6000
                and done["input_stream_sha256"] == h.hexdigest() and done["indices_sha256"] == sha(tape_path)
                and done["final_checkpoint_sha256"] == sha(directory / "repro_final.ckpt"), "Terminal budget/actual tape binding differs")
        states = {}
        for step in GRID:
            state_path = directory / f"state_rank{rank}_step{step:06}.json"; value = read(state_path)
            require(value["step"] == value["batches"] == step and value["input_stream_sha256"] == prefix[step], "State/input-prefix grid incomplete")
            require(all(bool(re.fullmatch("[a-f0-9]{64}", value[k])) for k in ("torch_cpu_rng", "torch_cuda_rng")), "Missing recorded RNG grid")
            require(value["torch_cpu_rng"] == read(baseline / f"state_rank{rank}_step{step:06}.json")["torch_cpu_rng"],
                    "Canonical CPU RNG state grid differs")
            states[step] = value; records[str(state_path)] = sha(state_path)
        initial_shared(states[0]["model"], read(baseline / f"state_rank{rank}_step000000.json")["model"], run["cell"])
        snapshots.append(states)
        for value in (runtime_path, tape_path, base_tape, journal_path, base_journal, done_path): records[str(value)] = sha(value)
        checks.append({"rank": rank, "actual_batches": 6000, "actual_indices": 6000 * 512, "canonical_common_packet_hashes_equal": True,
                       "selected_source_fingerprint_batches": raw_count, "initial_shared_model_equal": True})
    require(all(snapshots[0][s][k] == snapshots[1][s][k] for s in GRID for k in ("model", "optimizers", "schedulers")), "Rank model/Adam/scheduler grid not synchronized")
    # Read every real periodic checkpoint. Model/Adam/scheduler must match the
    # independently preserved observation at that step; no expected score gate.
    for step in GRID[1:]:
        candidates = list(directory.glob(f"*step={step}.ckpt")); require(len(candidates) == 1, "Missing/ambiguous periodic checkpoint")
        obj = torch.load(candidates[0], map_location="cpu", weights_only=False)
        require(obj["global_step"] == step and obj["epoch"] == (step - 1) // 4437, "Periodic step/epoch differs")
        for actual, observed in (("state_dict", "model"), ("optimizer_states", "optimizers"), ("lr_schedulers", "schedulers")):
            require(normalized(obj[actual]) == snapshots[0][step][observed], "Actual checkpoint differs from recorded " + observed)
        require(all(not torch.is_tensor(v) or bool(torch.isfinite(v).all()) for v in obj["state_dict"].values()), "Nonfinite periodic model")
        for optimizer in obj["optimizer_states"]:
            require(set(optimizer["state"]) == {p for group in optimizer["param_groups"] for p in group["params"]}, "Incomplete periodic Adam")
            for value in optimizer["state"].values():
                require(float(value["step"]) == step and all(not torch.is_tensor(v) or bool(torch.isfinite(v).all()) for v in value.values()), "Periodic Adam nonfinite/wrong step")
        require(all(v["last_epoch"] == step for v in obj["lr_schedulers"]), "Periodic schedule step differs")
        records[str(candidates[0])] = sha(candidates[0])
    periodic, _ = check_checkpoint(protocol_path, protocol, run, records=records)
    _, final = check_checkpoint(protocol_path, protocol, run, directory / "repro_final.ckpt", final=True, records=records)
    for actual, observed in (("state_dict", "model"), ("optimizer_states", "optimizers"), ("lr_schedulers", "schedulers")):
        require(normalized(final[actual]) == snapshots[0][6000][observed], "Final state differs from full6000 grid")
    require(all(sha(name) == value for name, value in records.items()), "Read artifacts changed during verification")
    return {"status": "PASS", "name": run["name"], "cell": run["cell"], "seed": 3407, "fresh_optimizer_steps": 6000,
            "sampler": "legacy_none", "checks": checks, "state_grid": list(GRID), "periodic_checkpoints": list(GRID[1:]),
            "final_epoch": 2, "checkpoint": str(periodic), "checkpoint_sha256": sha(periodic),
            "config_sha256": sha(path(run["config"])), "file_sha256": records,
            "meaning": "Single fresh arm verified against the preselected C1 input contract; no second seed/replay or score-based acceptance.",
            "limits": "Learned parameters/objectives/metrics may differ. Canonical common fields and selected source hashes are observed; unavailable full raw packets are not reconstructed."}


def verify_matrix(protocol, runs):
    """Across-arm input/init checks only, never compare trained weights/scores."""
    require(len(runs) == 5 and {run["cell"] for run in runs} == set(CELLS), "Full five-arm matrix required")
    records = {}; counts = []
    baseline = path(protocol["reference_baseline_output"])
    for rank in (0, 1):
        baseline_keys = set(read(baseline / f"state_rank{rank}_step000000.json")["model"])
        added = []; streams = []
        for run in runs:
            directory = path(run["output"])
            snapshot_path = directory / f"state_rank{rank}_step000000.json"
            model = read(snapshot_path)["model"]
            if run["cell"].endswith("f1"):
                added.append({key: model[key] for key in set(model) - baseline_keys})
            journal_path = directory / f"batches_rank{rank}.jsonl"
            rows = [json.loads(line) for line in journal_path.read_text().splitlines()]
            require(len(rows) == 6000, "Full6000 matrix journals required")
            stream = [(row["step"], row["xag_full_packet_sha256"], row["xag_source_fields_sha256"])
                      for row in rows if row["full_packet_sha256"] is not None]
            require(all(bool(re.fullmatch("[a-f0-9]{64}", value)) for item in stream for value in item[1:]),
                    "Missing selected actual-source matrix fingerprints")
            streams.append(stream)
            for filename in (snapshot_path, journal_path): records[str(filename)] = sha(filename)
        require(len(added) == 3 and added[0] == added[1] == added[2] and len(added[0]) == 39,
                "Three F+ arms do not share identical added parameter initialization")
        require(len(streams[0]) > 0 and all(stream == streams[0] for stream in streams[1:]),
                "Selected actual full-XAG/source packet fingerprints differ across five arms")
        counts.append({"rank": rank, "selected_source_batches_per_arm": len(streams[0]), "same_added_initial_tensors": 39})
    require(all(sha(filename) == expected for filename, expected in records.items()), "Matrix artifacts changed during verification")
    return {"status": "PASS", "checks": counts, "file_sha256": records,
            "scope": "Five-arm source/public input and three F+ added initialization equality; no learned-state, score, seed or replay selection"}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--protocol", required=True); parser.add_argument("--run", action="append")
    parser.add_argument("--output", required=True); parser.add_argument("--full-sources", action="store_true")
    args = parser.parse_args(); result = {"status": "FAIL", "source_sha256": sha(__file__), "errors": []}
    try:
        protocol = verify_protocol(args.protocol, full_sources=args.full_sources)
        runs = [resolve_run(protocol, value) for value in args.run] if args.run else protocol["runs"]
        require(len({r["name"] for r in runs}) == len(runs), "Repeated verification run")
        results = [verify_run(path(args.protocol), protocol, run) for run in runs]
        matrix = verify_matrix(protocol, runs) if len(runs) == 5 else None
        verify_protocol(args.protocol)
        result.update(status="PASS", protocol_sha256=sha(path(args.protocol)), runs=results,
                      baseline_identity_sha256=protocol["baseline_identity_sha256"], source_manifest_sha256=protocol["source_manifest_sha256"],
                      matrix=matrix)
    except Exception as error:
        result["errors"].append(type(error).__name__ + ": " + str(error))
    save_new(path(args.output), result)
    print(json.dumps({"status": result["status"], "errors": result["errors"]}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
