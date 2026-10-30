"""C1SF identity helpers (docs/EXPERIMENT_RECORDS.md#c1sf-prereg-20261029); CPU only, no scoring.

C1SF = three single-factor paired retrains of XYZ-C1 (`xyz_c1_stable_s3407`) plus a zero-training E3 reading of
the frozen C1 on the training subjects. Every arm is the canonical C1 recipe with exactly one registered change.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/c1sf/20261029"
#: amendment 1 (docs/EXPERIMENT_RECORDS.md#c1sf-amend1-20261029): GPU 2-7, data-loader resources, r2 configs /
#: preflights / protocol; the v1 protocol.json, configs and preflights stay as frozen records
CONFIGS = ROOT / "configs/c1sf_20261029/r2"
PROTOCOL = OUT / "protocol_r2.json"
PROTOCOL_V1 = OUT / "protocol.json"
#: data-loader resources only: every sample's augmentation is seeded by (seed, index) (semkine/dataset.py
#: __getitem__), so workers / prefetch change no input; the c1replay preflight checks that bit for bit
RESOURCES = {"NUM_WORKERS": 16, "PREFETCH_FACTOR": 2}
PHYSICAL_GPUS = (2, 3, 4, 5, 6, 7)
CANONICAL = ROOT / "configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml"
REFERENCE = ROOT / "outputs/xyz_c1_repro/20261018/runs/xyz_c1_stable_s3407"
C1_CHECKPOINT = ROOT / "outputs/semkine/xyz_c1_stable_s3407/xyz_c1_stable_s3407-step=6000.ckpt"
C1_CHECKPOINT_SHA256 = "41a8dc3feac3e7cffc5b19ec93491ec2c7a841c902ee583c39d8435eb5c08091"
C1_E1_TRACE = ROOT / "outputs/xyz_c1_repro/20261018/runs/xyz_c1_stable_s3407/strict_trace/states.npz"
C1_E1_TRACE_SHA256 = "dc961819cd76e381dca5037595e2dc2280569af02e2b290258d1263fee726c9b"
XCL_C1_EVALUATIONS = ROOT / "outputs/xcl/20261019/evaluations/xyz_c1_stable_s3407"

#: the 18 base training recordings (nine subjects x global / local; the _v2-_v4 augmented copies excluded)
TRAIN_SUBJECTS = ("ch", "lfz", "lpc", "lr", "ly", "lyh", "lyq", "ycy", "ylf")
TRAIN_BASE = tuple(f"{s}_{d}" for s in TRAIN_SUBJECTS for d in ("global", "local"))

#: arm -> the single registered change, as dotted-key -> value on the canonical C1 config
ARMS = {
    "c1replay": {},
    "dz": {"AUG.DOMRAND.SCALE_MODE": "depth"},
    "zr": {"C1SF.ZN": "root_relative"},
    "so3": {"LOSS.TYPE": "so3_trans_fk", "LOSS.ROT_WEIGHT": 1.0, "LOSS.TRANS_WEIGHT": 1.0, "LOSS.FK_WEIGHT": 2.0},
}
#: arms that get a formal 6000-step training (c1replay is a 50-step code-drift check only)
FORMAL_ARMS = ("dz", "zr", "so3")
#: arms whose training inputs must equal C1's packet for packet (dz changes labels / K by design)
SAME_INPUT_ARMS = ("c1replay", "zr", "so3")


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


def run_name(arm):
    return f"c1sf_{arm}_s3407"


def config_path(arm, phase):
    return CONFIGS / (run_name(arm) + ("_preflight" if phase == "preflight" else "") + ".yaml")


def output_dir(arm, phase):
    return OUT / ("preflight_r2" if phase == "preflight" else "runs") / run_name(arm)


def apply_change(cfg, change):
    for dotted, value in change.items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return cfg


def expected_raw(arm, phase, canonical_raw):
    """The arm's YAML content: canonical C1 + its one change + identity / budget / paths."""
    require(arm in ARMS, "unknown arm " + repr(arm))
    require(phase in ("preflight", "formal") and (phase == "preflight" or arm in FORMAL_ARMS), "unregistered phase")
    cfg = apply_change(copy.deepcopy(canonical_raw), ARMS[arm])
    cfg.setdefault("C1SF", {})["ARM"] = arm
    out = output_dir(arm, phase)
    cfg["TRAIN"]["RUN_NAME"] = run_name(arm) + ("_preflight" if phase == "preflight" else "")
    cfg["TRAIN"]["OUTPUT_DIR"] = str(out)
    cfg["EVAL"]["OUTPUT_DIR"] = str(out / "eval")
    cfg["TRAIN"].update(RESOURCES)
    if phase == "preflight":
        cfg["TRAIN"]["MAX_STEPS"] = 50
        cfg["TRAIN"]["VAL_CHECK_INTERVAL"] = 25
    return cfg


def zn_mode(cfg):
    return (cfg.get("C1SF") or {}).get("ZN", "absolute")


def verify_protocol(value=PROTOCOL):
    p = read(value)
    require(p["study"] == "C1SF_20261029" and p["seed"] == 3407, "Wrong C1SF identity")
    require(p.get("amends_sha256") == sha(PROTOCOL_V1), "Amended protocol does not bind the frozen v1 protocol")
    for name, digest in p["file_sha256"].items():
        require(sha(name) == digest, "Frozen C1SF file changed: " + name)
    return p


def verify_run(arm, phase):
    """Real per-rank input streams and shared initial tensors against canonical C1; never scores."""
    import numpy as np
    import torch
    out = output_dir(arm, phase)
    name = run_name(arm) + ("_preflight" if phase == "preflight" else "")
    steps = 50 if phase == "preflight" else 6000
    checkpoint_sha = sha(out / "repro_final.ckpt")
    result = {"status": "PASS", "arm": arm, "phase": phase, "output": str(out), "ranks": []}
    for rank in range(2):
        done = read(out / f"completion_rank{rank}.json")
        require(done["status"] == "COMPLETE" and done["step"] == done["batches"] == steps, "Incomplete rank")
        require(done["final_checkpoint_sha256"] == checkpoint_sha, "Final checkpoint SHA differs from completion")
        actual = np.fromfile(out / f"indices_rank{rank}.i32", dtype=np.int32)
        ref = np.fromfile(REFERENCE / f"indices_rank{rank}.i32", dtype=np.int32)[:steps * 512]
        require(actual.size == steps * 512 and np.array_equal(actual, ref), "Actual sample indices differ from C1")
        initial = read(out / f"state_rank{rank}_step000000.json")["model"]
        base = read(REFERENCE / f"state_rank{rank}_step000000.json")["model"]
        require(initial == base, "Initial model tensors differ from C1")
        rows = [json.loads(s) for s in (out / f"batches_rank{rank}.jsonl").read_text().splitlines()]
        ref_rows = [json.loads(s) for s in (REFERENCE / f"batches_rank{rank}.jsonl").read_text().splitlines()][:steps]
        require(len(rows) == steps, "Input journal denominator differs")
        differing = 0
        for row, old in zip(rows, ref_rows):
            require(row["step"] == old["step"] and row["indices_sha256"] == old["indices_sha256"], "Index stream differs")
            same = row["small_fields_sha256"] == old["small_fields_sha256"]
            if old.get("full_packet_sha256") is not None and row.get("full_packet_sha256") is not None:
                same = same and row["full_packet_sha256"] == old["full_packet_sha256"]
            differing += not same
        if arm in SAME_INPUT_ARMS:
            require(differing == 0, "Training packets differ from C1 for a same-input arm")
        else:
            require(differing >= 1, "Depth-scale arm did not change its training packets")
        replay = None
        if arm == "c1replay":
            # Steps 1-3 are recorded in both journals: same code + same inputs => same predictions / gradients.
            fields = ("prediction_sha256", "objective_sha256", "gradients")
            replay = {str(s): all(rows[s - 1].get(f) == ref_rows[s - 1].get(f) for f in fields) for s in (1, 2, 3)}
        rt = read(out / f"effective_runtime_rank{rank}.json")
        loader = rt["actual_train_loader"]
        require(rt["world_size"] == 2 and rt["precision"] == "bf16" and rt["effective_seed"] == 3407, "Runtime changed")
        require(loader["num_workers"] == RESOURCES["NUM_WORKERS"] and loader["prefetch_factor"] == RESOURCES["PREFETCH_FACTOR"]
                and loader["batch_size"] == 512
                and loader["underlying_generator"] is None and loader["loader_generator"] is None, "Loader contract")
        require(rt["actual_math_flags"] == read(REFERENCE / f"effective_runtime_rank{rank}.json")["actual_math_flags"],
                "Strict C1 math flags differ")
        if phase == "preflight":
            for step in (25, 50):
                val = read(out / f"validation_rank{rank}_step{step:06}.json")
                require(val["status"] == "PASS" and val["batches"] == 126 and val["model_state_unchanged"],
                        "Full validation was not verified")
            require(rows[25]["step"] == 26, "Training did not return after validation")
        else:
            for step in range(500, 6001, 500):
                require((out / f"{name}-step={step}.ckpt").is_file(), "Missing fixed checkpoint grid")
        result["ranks"].append({"rank": rank, "packets_differing_from_c1": differing, "c1_replay_steps_equal": replay})
    obj = torch.load(out / "repro_final.ckpt", map_location="cpu")
    require(all(bool(torch.isfinite(v).all()) for v in obj["state_dict"].values()), "Nonfinite final model")
    if phase == "formal":
        require(obj["global_step"] == 6000, "Final checkpoint step differs")
        result["checkpoint_sha256_by_step"] = {str(s): sha(out / f"{name}-step={s}.ckpt") for s in (5000, 5500, 6000)}
    result["checkpoint"] = str(out / "repro_final.ckpt")
    result["checkpoint_sha256"] = checkpoint_sha
    if arm == "c1replay":
        result["c1_code_replay_exact"] = all(all(r["c1_replay_steps_equal"].values()) for r in result["ranks"])
    return result
