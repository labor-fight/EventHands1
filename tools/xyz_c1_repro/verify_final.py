#!/usr/bin/env python3
"""Verify the two full fresh C1 trajectories without selecting an accuracy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def different(a, b, prefix=""):
    if torch.is_tensor(a) and torch.is_tensor(b):
        return [] if a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b) else [prefix]
    if type(a) is not type(b):
        return [prefix + ":type"]
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return [prefix + ":keys"]
        return [item for key in a for item in different(a[key], b[key], prefix + "/" + str(key))]
    if isinstance(a, (tuple, list)):
        if len(a) != len(b):
            return [prefix + ":length"]
        return [item for i, (x, y) in enumerate(zip(a, b)) for item in different(x, y, prefix + "/" + str(i))]
    return [] if a == b else [prefix]


def verify_run(directory, sampler):
    """Check the real batch tape against its journal, beyond pair equality."""
    errors, checks = [], []
    initial = []
    for rank in range(2):
        done = read(directory / f"completion_rank{rank}.json")
        runtime = read(directory / f"effective_runtime_rank{rank}.json")
        loader = runtime["actual_train_loader"]
        generator = loader["underlying_generator"]
        expected_flags = {
            "deterministic_algorithms": True, "warn_only": False,
            "cudnn_deterministic": True, "cudnn_benchmark": False,
            "matmul_allow_tf32": False, "cudnn_allow_tf32": False,
            "matmul_precision": "highest", "bf16_reduced_precision_reduction": False,
            "fp16_reduced_precision_reduction": False,
        }
        if (runtime["actual_math_flags"] != expected_flags or runtime["phase"] != "formal"
                or runtime["effective_seed"] != 3407 or runtime["effective_max_steps"] != 6000
                or runtime["effective_accumulate_grad_batches"] != 1 or runtime["precision"] != "bf16"
                or runtime["epoch_batches"] != 4437 or loader["sampler_seed"] != 3407
                or loader["sampler_rank"] != rank or loader["num_workers"] != 12
                or loader["prefetch_factor"] != 1 or not loader["drop_last"]):
            errors.append(f"{directory.name}/rank{rank}/effective_recipe_or_math_flags")
        if (done["status"] != "COMPLETE" or done["step"] != 6000 or done["batches"] != 6000
                or runtime["sampler_policy"] != sampler or runtime["world_size"] != 2
                or loader["batch_size"] != 512 or loader["dataset_samples"] != 4544300
                or ((generator is None) != (sampler == "legacy_none"))):
            errors.append(f"{directory.name}/rank{rank}/runtime_or_budget")
        if generator is not None and generator["initial_seed"] != 3407:
            errors.append(f"{directory.name}/rank{rank}/generator_seed")
        tape = np.fromfile(directory / f"indices_rank{rank}.i32", dtype=np.int32)
        if tape.shape != (6000 * 512,):
            errors.append(f"{directory.name}/rank{rank}/tape_shape")
            continue
        tape = tape.reshape(6000, 512)
        if tape.min() < 0 or tape.max() >= 4544300:
            errors.append(f"{directory.name}/rank{rank}/index_out_of_range")
        rows = [json.loads(line) for line in (directory / f"batches_rank{rank}.jsonl").read_text().splitlines()]
        if len(rows) != 6000:
            errors.append(f"{directory.name}/rank{rank}/journal_denominator")
        rolling = hashlib.sha256()
        for step, row in enumerate(rows, 1):
            expected = hashlib.sha256(tape[step - 1].tobytes()).hexdigest()
            if row["step"] != step or row["batch_size"] != 512 or row["indices_sha256"] != expected:
                errors.append(f"{directory.name}/rank{rank}/batch{step}")
            identity = {key: row[key] for key in ("step", "indices_sha256", "small_fields_sha256", "full_packet_sha256")}
            rolling.update(json.dumps(identity, sort_keys=True).encode())
        if rolling.hexdigest() != done["input_stream_sha256"]:
            errors.append(f"{directory.name}/rank{rank}/rolling_stream")
        if done["indices_sha256"] != sha(directory / f"indices_rank{rank}.i32"):
            errors.append(f"{directory.name}/rank{rank}/completion_tape_hash")
        initial.append(read(directory / f"state_rank{rank}_step000000.json")["model"])
        checks.append({"name": directory.name, "rank": rank, "sampler": sampler,
                       "batches_recomputed": len(rows), "rolling_stream_sha256": rolling.hexdigest()})
    if len(initial) == 2 and initial[0] != initial[1]:
        errors.append(directory.name + "/initial_rank_disagreement")
    checkpoint = torch.load(directory / "repro_final.ckpt", map_location="cpu", weights_only=False)
    # on_train_end runs after Lightning increments completed epochs, unlike
    # the periodic step6000 checkpoint saved inside the second epoch.
    expected_completed_epochs = (6000 + 4437 - 1) // 4437
    if checkpoint["global_step"] != 6000 or checkpoint["epoch"] != expected_completed_epochs:
        errors.append(directory.name + "/final_checkpoint_budget")
    for value in checkpoint["state_dict"].values():
        if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
            errors.append(directory.name + "/nonfinite_state")
    for optimizer in checkpoint["optimizer_states"]:
        for state in optimizer["state"].values():
            if float(state["step"]) != 6000:
                errors.append(directory.name + "/adam_step")
    for scheduler in checkpoint["lr_schedulers"]:
        if scheduler["last_epoch"] != 6000:
            errors.append(directory.name + "/scheduler_step")
    return errors, checks, initial[0] if initial else None


def verify(canonical, replay, output, intervention=None):
    canonical, replay = Path(canonical), Path(replay)
    errors, files, checks = [], {}, []
    for directory in (canonical, replay):
        failures, results, _ = verify_run(directory, "legacy_none")
        errors.extend(failures)
        checks.extend(results)
    if intervention is not None:
        intervention = Path(intervention)
        failures, results, initial = verify_run(intervention, "dedicated")
        errors.extend(failures)
        checks.extend(results)
        if initial != read(canonical / "state_rank0_step000000.json")["model"]:
            errors.append("sampling_intervention/initial_state_differs")
    for rank in range(2):
        for directory in (canonical, replay):
            receipt = read(directory / f"completion_rank{rank}.json")
            runtime = read(directory / f"effective_runtime_rank{rank}.json")
            if receipt["status"] != "COMPLETE" or receipt["step"] != 6000 or receipt["batches"] != 6000:
                errors.append(str(directory) + ":incomplete")
            if runtime["sampler_policy"] != "legacy_none" or runtime["actual_train_loader"]["underlying_generator"] is not None:
                errors.append(str(directory) + ":unexpected_sampler")
            if not runtime["actual_math_flags"]["deterministic_algorithms"] or runtime["actual_math_flags"]["warn_only"]:
                errors.append(str(directory) + ":unexpected_math_flags")
        name = f"indices_rank{rank}.i32"
        a, b = canonical / name, replay / name
        if a.stat().st_size != 6000 * 512 * 4 or a.read_bytes() != b.read_bytes():
            errors.append(name + ":index_tape_difference_or_missing_rows")
        checks.append({"rank": rank, "actual_indices": 6000 * 512, "indices_equal": sha(a) == sha(b)})
        name = f"batches_rank{rank}.jsonl"
        aa = [json.loads(line) for line in (canonical / name).read_text().splitlines()]
        bb = [json.loads(line) for line in (replay / name).read_text().splitlines()]
        if len(aa) != 6000 or len(bb) != 6000:
            errors.append(name + ":wrong_batch_denominator")
        for index, (a, b) in enumerate(zip(aa, bb), 1):
            if a["step"] != index or a["batch_size"] != 512 or a != b:
                errors.append(name + ":batch_difference_at_" + str(index))
        for step in range(0, 6001, 500):
            name = f"state_rank{rank}_step{step:06}.json"
            a, b = read(canonical / name), read(replay / name)
            errors.extend(different(a, b, f"rank{rank}/step{step}"))
        for directory in (canonical, replay):
            for p in directory.glob("*.json"):
                files[str(p)] = sha(p)
            for suffix in ("*.jsonl", "*.i32"):
                for p in directory.glob(suffix):
                    files[str(p)] = sha(p)
    for directory in (canonical, replay):
        for step in range(0, 6001, 500):
            a, b = [read(directory / f"state_rank{rank}_step{step:06}.json") for rank in range(2)]
            for field in ("model", "optimizers", "schedulers"):
                errors.extend(different(a[field], b[field], f"{directory.name}/rank_sync/step{step}/{field}"))
    actual_grid = []
    for step in range(500, 6001, 500):
        checkpoints = []
        for directory in (canonical, replay):
            candidates = list(directory.glob(f"*step={step}.ckpt"))
            if len(candidates) != 1:
                errors.append(directory.name + "/ambiguous_checkpoint/" + str(step))
                break
            p = candidates[0]
            checkpoint = torch.load(p, map_location="cpu", weights_only=False)
            if checkpoint["global_step"] != step:
                errors.append(directory.name + "/checkpoint_step/" + str(step))
            checkpoints.append(checkpoint)
            files[str(p)] = sha(p)
        if len(checkpoints) == 2:
            for field in ("state_dict", "optimizer_states", "lr_schedulers", "loops", "global_step", "epoch"):
                errors.extend(different(checkpoints[0][field], checkpoints[1][field], f"actual_checkpoint{step}/{field}"))
            actual_grid.append(step)
    ca, cb = [torch.load(p / "repro_final.ckpt", map_location="cpu", weights_only=False) for p in (canonical, replay)]
    for field in ("state_dict", "optimizer_states", "lr_schedulers", "loops", "global_step", "epoch"):
        errors.extend(different(ca[field], cb[field], "final_checkpoint/" + field))
    for directory in (canonical, replay):
        p = directory / "repro_final.ckpt"
        files[str(p)] = sha(p)
    report = {
        "status": "PASS" if not errors else "FAIL", "errors": errors,
        "canonical": str(canonical), "replay": str(replay),
        "checks": checks, "seed": 3407, "fresh_optimizer_steps": 6000,
        "checkpoint_grid_compared": list(range(0, 6001, 500)),
        "actual_saved_checkpoint_grid_compared": actual_grid,
        "final_checkpoint_tensor_and_loop_fields_compared": ["state_dict", "optimizer_states", "lr_schedulers", "loops", "global_step", "epoch"],
        "meaning": "Fresh full6000 reproduction of the predesignated baseline under this exact implementation/runtime. Replay is an engineering verification copy, not another seed or an accuracy average.",
        "limits": "No claim of hardware/software portability, equality to missing September weights, stable superiority across seeds, or exact mid-epoch checkpoint resume.",
        "source_sha256": sha(__file__), "file_sha256": files,
    }
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("canonical")
    parser.add_argument("replay")
    parser.add_argument("--output", required=True)
    parser.add_argument("--intervention")
    args = parser.parse_args()
    result = verify(args.canonical, args.replay, args.output, args.intervention)
    print(json.dumps({"status": result["status"], "errors": result["errors"][:20]}))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
