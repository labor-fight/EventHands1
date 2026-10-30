"""Small, CPU-only identity helpers for the XCL study."""
from __future__ import annotations
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/xcl/20261019"
CANONICAL = ROOT / "configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml"
REFERENCE = ROOT / "outputs/xyz_c1_repro/20261018/runs/xyz_c1_stable_s3407"
BASELINE = ROOT / "outputs/xyz_c1_repro/baseline.json"
SOURCE_MANIFEST = ROOT / "outputs/xag/source_manifest.json"
CELLS = {
    "a2f0": ("face_route", "legacy", True, (2, 3)),
    "c0": ("base", "legacy", False, (4, 5)),
    "a0fod": ("base", "oracle_dense", True, (6, 7)),
    "a3f0": ("full_oracle", "legacy", True, (2, 3)),
    "a0foo": ("base", "oracle_observed", True, (4, 5)),
}

def now():
    return datetime.now(timezone.utc).isoformat()

def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p

def sha(value):
    h = hashlib.sha256()
    with path(value).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def read(value):
    return json.loads(path(value).read_text())

def save(destination, value, *, exclusive=False):
    p = path(destination)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with p.open("x") as stream:
            stream.write(data)
    else:
        temp = p.with_name(p.name + f".{os.getpid()}.tmp")
        temp.write_text(data)
        temp.replace(p)

def require(value, message):
    if not value:
        raise RuntimeError(message)

def normalized(value):
    """Same tensor-byte fingerprint convention as the sealed C1 audit."""
    import torch
    if torch.is_tensor(value):
        a = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        return {"tensor_sha256": hashlib.sha256(memoryview(a).cast("B")).hexdigest(),
                "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): normalized(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalized(v) for v in value]
    return value

def verify_protocol(value, *, sources=True):
    p = read(value)
    require(p["study"] == "XCL_20261019" and p["seed"] == 3407, "Wrong XCL identity")
    require(len(p["runs"]) == 10 and len([r for r in p["runs"] if not r["reuse"]]) == 5,
            "XCL requires exactly five reused and five fresh models")
    if sources:
        for name, digest in p["file_sha256"].items():
            require(sha(name) == digest, "Active XCL source changed: " + name)
        for name, identity in p["immutable_data_stat"].items():
            s = path(name).stat()
            require([s.st_size, s.st_mtime_ns] == identity, "Data identity changed: " + name)
        binding_path = OUT / "depth_binding.json"
        if binding_path.is_file():
            from semkine.xag_sources import verify_depth_manifest
            binding = read(binding_path)
            require(sha(OUT / "depth_manifest.json") == binding["manifest_sha256"], "Accepted depth manifest changed")
            verify_depth_manifest(read(OUT / "depth_manifest.json"), p["depth_binding"]["sequences"])
    return p

def verify_run(run, *, phase="formal", directory=None):
    """Verify real per-rank streams and shared initial tensors; never score/select."""
    import numpy as np
    out = path(directory or run["output"])
    steps = 50 if phase == "preflight" else 6000
    results = []
    periodic_sha = {}
    checkpoint_sha = sha(out / "repro_final.ckpt")
    rolling_prefixes = {}
    for rank in range(2):
        done = read(out / f"completion_rank{rank}.json")
        require(done["status"] == "COMPLETE" and done["step"] == done["batches"] == steps,
                "Incomplete optimizer-step/rank completion")
        require(done["final_checkpoint_sha256"] == checkpoint_sha, "Final checkpoint SHA differs from completion")
        require(done["indices_sha256"] == sha(out / f"indices_rank{rank}.i32"), "Index tape SHA differs from completion")
        actual = np.fromfile(out / f"indices_rank{rank}.i32", dtype=np.int32)
        ref = np.fromfile(REFERENCE / f"indices_rank{rank}.i32", dtype=np.int32)[:steps * 512]
        require(actual.size == steps * 512 and np.array_equal(actual, ref), "Actual C1 indices differ")
        initial = read(out / f"state_rank{rank}_step000000.json")["model"]
        base = read(REFERENCE / f"state_rank{rank}_step000000.json")["model"]
        require(all(initial.get(k) == v for k, v in base.items()), "Shared C1 initial tensors differ")
        extra = set(initial) - set(base)
        expected = {"xyz.xag_oracle_projection.weight", "xyz.xag_oracle_projection.bias"} if run["cell"] in ("a0fod", "a0foo") else set()
        if run.get("reuse") and run["cell"].endswith("f1"):
            from tools.xag_paired_20261018.verify_final import initial_shared
            initial_shared(initial, base, run["cell"])
        else:
            require(extra == expected, "Unregistered added initial parameters")
        rows = [json.loads(s) for s in (out / f"batches_rank{rank}.jsonl").read_text().splitlines()]
        ref_rows = [json.loads(s) for s in (REFERENCE / f"batches_rank{rank}.jsonl").read_text().splitlines()][:steps]
        require(len(rows) == steps, "Input journal denominator differs")
        rolling = hashlib.sha256()
        prefixes = {0: rolling.hexdigest()}
        for row, old in zip(rows, ref_rows):
            for key in ("step", "indices_sha256", "small_fields_sha256"):
                require(row[key] == old[key], "Common C1 stream differs: " + key)
            if old.get("full_packet_sha256") is not None:
                require(row["full_packet_sha256"] == old["full_packet_sha256"], "Common C1 raw packet differs")
            identity = {k: row[k] for k in ("step", "indices_sha256", "small_fields_sha256", "full_packet_sha256")}
            rolling.update(json.dumps(identity, sort_keys=True).encode())
            prefixes[row["step"]] = rolling.hexdigest()
        rolling_prefixes[rank] = prefixes
        require(done["input_stream_sha256"] == rolling.hexdigest(), "Completion rolling stream differs")
        rt = read(out / f"effective_runtime_rank{rank}.json")
        loader = rt["actual_train_loader"]
        require(rt["world_size"] == 2 and rt["precision"] == "bf16" and rt["effective_seed"] == 3407,
                "Runtime world/precision/seed changed")
        require(loader["num_workers"] == 12 and loader["prefetch_factor"] == 1 and loader["batch_size"] == 512
                and loader["underlying_generator"] is None and loader["loader_generator"] is None,
                "Actual legacy_none/resource contract differs")
        require(rt["actual_math_flags"] == read(REFERENCE / f"effective_runtime_rank{rank}.json")["actual_math_flags"],
                "Strict C1 math flags differ")
        if phase == "preflight":
            for step in (25, 50):
                val = read(out / f"validation_rank{rank}_step{step:06}.json")
                require(val["status"] == "PASS" and val["batches"] == 126
                        and val["samples_with_original_distributed_padding"] == 64094
                        and val["model_state_unchanged"], "Full validation was not verified")
            require(rows[25]["step"] == 26, "Training did not return after validation")
        else:
            for step in range(500, 6001, 500):
                require((out / f"{run['name']}-step={step}.ckpt").is_file(), "Missing fixed checkpoint grid")
        results.append({"rank": rank, "steps": steps, "common_input_equal": True,
                        "shared_initial_equal": True, "added_initial_keys": sorted(extra)})
    require(read(out / "state_rank0_step000000.json")["model"] == read(out / "state_rank1_step000000.json")["model"],
            "DDP initial states differ")
    import torch
    obj = torch.load(out / "repro_final.ckpt", map_location="cpu")
    for rank in range(2):
        observed = read(out / f"state_rank{rank}_step{steps:06}.json")
        for actual, recorded in (("state_dict", "model"), ("optimizer_states", "optimizers"), ("lr_schedulers", "schedulers")):
            require(normalized(obj[actual]) == observed[recorded], "Final checkpoint differs from real rank snapshot: " + actual)
    require(all(bool(torch.isfinite(v).all()) for v in obj["state_dict"].values()), "Nonfinite final model")
    if phase == "formal":
        require(obj["global_step"] == 6000 and len(obj["optimizer_states"]) == len(obj["lr_schedulers"]) == 1,
                "Final checkpoint/Adam/scheduler identity differs")
        opt = obj["optimizer_states"][0]
        require(set(opt["state"]) == {k for group in opt["param_groups"] for k in group["params"]},
                "Incomplete trained Adam state")
        require(all(int(s["step"]) == 6000 for s in opt["state"].values())
                and obj["lr_schedulers"][0]["last_epoch"] == 6000, "Incomplete Adam/scheduler steps")
        for state in opt["state"].values():
            require(all(not torch.is_tensor(v) or bool(torch.isfinite(v).all()) for v in state.values()),
                    "Nonfinite Adam state")
        for step in range(0, 6001, 500):
            states = [read(out / f"state_rank{rank}_step{step:06}.json") for rank in range(2)]
            for rank, state in enumerate(states):
                require(state["step"] == state["batches"] == step
                        and state["input_stream_sha256"] == rolling_prefixes[rank][step], "Snapshot input prefix differs")
                require(state["torch_cpu_rng"] == read(REFERENCE / f"state_rank{rank}_step{step:06}.json")["torch_cpu_rng"],
                        "Formal C1 CPU RNG trajectory differs")
            require(states[0]["model"] == states[1]["model"] and states[0]["optimizers"] == states[1]["optimizers"],
                    "DDP model/Adam rank synchronization differs")
            if step:
                periodic_path = out / f"{run['name']}-step={step}.ckpt"
                periodic_sha[str(step)] = sha(periodic_path)
                periodic = torch.load(periodic_path, map_location="cpu")
                for actual, recorded in (("state_dict", "model"), ("optimizer_states", "optimizers"), ("lr_schedulers", "schedulers")):
                    require(normalized(periodic[actual]) == states[0][recorded], "Periodic checkpoint differs from audit snapshot")
    return {"status": "PASS", "phase": phase, "cell": run["cell"], "output": str(out), "ranks": results,
            "checkpoint": str(out / "repro_final.ckpt"), "checkpoint_sha256": checkpoint_sha,
            "checkpoint_sha256_by_step": periodic_sha}
