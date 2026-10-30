"""S45-CT identity helpers (docs/EXPERIMENT_RECORDS.md#s45ct-prereg-20261030); CPU only, no scoring.

S45-CT = fresh retrains of S45-TR T1d (S45 + XAG GT face routing) that change only the oracle routes
(semkine/s45ct_route.py): `sigma16` (each routed face moved to a random face 8-24 mm away on the template) and
`shuffle` (the routed pairs permuted within each packet), plus `t1dreplay` (T1d's recipe through this entry, 50 steps,
must replay T1d's recorded steps 1-3 bit for bit). Matched controls are the existing S45 and T1d runs (same strict
contract, seed 3407, sample indices and initial tensors).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.s45tr.common import now, read, require, save, sha  # noqa: E402,F401

OUT = ROOT / "outputs/s45ct/20261030"
CONFIGS = ROOT / "configs/s45ct_20261030"
PROTOCOL = OUT / "protocol.json"
PHYSICAL_GPUS = (2, 3, 4, 5, 6, 7)
S45TR = ROOT / "outputs/s45tr/20261030"
T1D_CONFIG = ROOT / "configs/s45tr_20261030/s45tr_t1d_s3407.yaml"
T1D_RUN = S45TR / "runs/s45tr_t1d_s3407"
S45_RUN = ROOT / "outputs/c1sf/20261029/runs/c1sf_so3_s3407"
S45_CONFIG = ROOT / "configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml"
#: arm -> S45CT.MODE (None: T1d's routes unchanged)
ARMS = {"t1dreplay": None, "sigma16": "sigma16", "shuffle": "shuffle"}
FORMAL_ARMS = ("sigma16", "shuffle")
SOURCES = ("semkine/s45ct_route.py", "tools/s45ct/__init__.py", "tools/s45ct/common.py", "tools/s45ct/make_configs.py",
           "tools/s45ct/train.py", "tools/s45ct/accept.py", "tools/s45ct/evaluate.py", "tools/s45ct/readouts.py",
           "tools/s45ct/audit.py", "tools/s45ct/proxy.py", "tools/s45ct/decide.py", "tools/s45ct/run.py",
           "tests/test_s45ct.py")
DEPENDENCIES = ("configs/s45tr_20261030/s45tr_t1d_s3407.yaml", "configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml",
                "tools/s45tr/common.py", "tools/s45tr/train.py", "tools/s45tr/train_xag.py", "tools/s45tr/readouts.py",
                "tools/s45tr/evaluate.py", "tools/s45rf/p2.py", "tools/xyz_c1_repro/train.py", "tools/xcl/evaluate.py",
                "tools/make_s36_row.py", "semkine/xag.py", "semkine/xag_sources.py", "semkine/xyz_mesh.py",
                "semkine/train.py", "semkine/dataset.py", "semkine/eval_track.py", "model/model.py", "model/config.py")


def run_name(arm, phase="formal"):
    return f"s45ct_{arm}_s3407" + ("_preflight" if phase == "preflight" else "")


def config_path(arm, phase):
    return CONFIGS / (run_name(arm, phase) + ".yaml")


def output_dir(arm, phase):
    return OUT / ("preflight" if phase == "preflight" else "runs") / run_name(arm)


def expected_raw(arm, phase, t1d_raw):
    """The arm's YAML content: T1d's + `S45CT {ARM, MODE}` in place of `S45TR` + identity / budget / paths."""
    require(arm in ARMS and phase in ("preflight", "formal") and (phase == "preflight" or arm in FORMAL_ARMS),
            "unregistered S45-CT arm / phase")
    cfg = copy.deepcopy(t1d_raw)
    require(cfg.pop("S45TR") == {"ARM": "t1d"}, "T1d config identity changed")
    cfg["S45CT"] = {"ARM": arm, "MODE": ARMS[arm]}
    out = output_dir(arm, phase)
    cfg["TRAIN"]["RUN_NAME"] = run_name(arm, phase)
    cfg["TRAIN"]["OUTPUT_DIR"] = str(out)
    cfg["EVAL"]["OUTPUT_DIR"] = str(out / "eval")
    if phase == "preflight":
        cfg["TRAIN"]["MAX_STEPS"] = 50
        cfg["TRAIN"]["VAL_CHECK_INTERVAL"] = 25
    return cfg


def install_hook(cfg):
    mode = (cfg.get("S45CT") or {}).get("MODE")
    if mode is not None:
        from semkine import s45ct_route
        s45ct_route.install(mode, cfg["MANO"]["NPZ"])
    return mode


def frozen_files():
    return read(PROTOCOL)["file_sha256"]


def verify_protocol():
    p = read(PROTOCOL)
    require(p["study"] == "S45CT_20261030" and p["status"] == "FROZEN", "S45-CT protocol identity")
    for name, digest in p["file_sha256"].items():
        require(sha(ROOT / name) == digest, "frozen S45-CT file changed: " + name)
    return p


def config_frozen(config):
    rel = str(Path(config).resolve().relative_to(ROOT))
    files = frozen_files()
    require(rel in files and files[rel] == sha(config), "Config not frozen: " + rel)


def journal(directory, rank, steps=None):
    rows = [json.loads(s) for s in (Path(directory) / f"batches_rank{rank}.jsonl").read_text().splitlines()]
    return rows if steps is None else rows[:steps]


def verify_run(arm, phase):
    """Per-rank input streams, initial tensors and packets against T1d's (and S45's) recorded training; never scores."""
    import numpy as np
    import torch
    out = output_dir(arm, phase)
    steps = 50 if phase == "preflight" else 6000
    checkpoint_sha = sha(out / "repro_final.ckpt")
    result = {"status": "PASS", "arm": arm, "phase": phase, "output": str(out), "ranks": []}
    for rank in range(2):
        done = read(out / f"completion_rank{rank}.json")
        require(done["status"] == "COMPLETE" and done["step"] == done["batches"] == steps, "Incomplete rank")
        require(done["final_checkpoint_sha256"] == checkpoint_sha, "Final checkpoint SHA differs from completion")
        actual = np.fromfile(out / f"indices_rank{rank}.i32", dtype=np.int32)
        for ref_run in (T1D_RUN, S45_RUN):
            ref = np.fromfile(ref_run / f"indices_rank{rank}.i32", dtype=np.int32)[:steps * 512]
            require(actual.size == steps * 512 and np.array_equal(actual, ref), "Sample indices differ from " + ref_run.name)
        initial = read(out / f"state_rank{rank}_step000000.json")["model"]
        require(initial == read(T1D_RUN / f"state_rank{rank}_step000000.json")["model"], "Initial tensors differ from T1d")
        rows, ref_rows = journal(out, rank), journal(T1D_RUN, rank, steps)
        require(len(rows) == steps, "Input journal denominator differs")
        small = full = full_compared = 0
        for row, old in zip(rows, ref_rows):
            require(row["step"] == old["step"] and row["indices_sha256"] == old["indices_sha256"], "Index stream differs")
            small += row["small_fields_sha256"] != old["small_fields_sha256"]
            if old.get("full_packet_sha256") is not None and row.get("full_packet_sha256") is not None:
                full_compared += 1
                full += row["full_packet_sha256"] != old["full_packet_sha256"]
        require(small == 0 and full == 0 and full_compared > 0, "Training packets differ from T1d (routes change in-model only)")
        fields = ("prediction_sha256", "objective_sha256", "gradients")
        same = {str(s): all(rows[s - 1].get(f) == ref_rows[s - 1].get(f) for f in fields) for s in (1, 2, 3)}
        rt, ref_rt = read(out / f"effective_runtime_rank{rank}.json"), read(T1D_RUN / f"effective_runtime_rank{rank}.json")
        require(rt["world_size"] == 2 and rt["precision"] == "bf16" and rt["effective_seed"] == 3407, "Runtime changed")
        require(rt["actual_train_loader"] == ref_rt["actual_train_loader"], "Loader contract differs from T1d")
        require(rt["actual_math_flags"] == ref_rt["actual_math_flags"], "Strict math flags differ from T1d")
        require(rt["s45ct_mode"] == ARMS[arm], "Installed route corruption differs from the registered arm")
        if phase == "preflight":
            for step in (25, 50):
                val = read(out / f"validation_rank{rank}_step{step:06}.json")
                require(val["status"] == "PASS" and val["batches"] == 126 and val["model_state_unchanged"],
                        "Full validation was not verified")
            require(rows[25]["step"] == 26, "Training did not return after validation")
        else:
            for step in range(500, 6001, 500):
                require((out / f"{run_name(arm)}-step={step}.ckpt").is_file(), "Missing fixed checkpoint grid")
        result["ranks"].append({"rank": rank, "packets_full_compared": full_compared, "t1d_steps_1_3_equal": same})
    obj = torch.load(out / "repro_final.ckpt", map_location="cpu")
    require(all(bool(torch.isfinite(v).all()) for v in obj["state_dict"].values() if torch.is_tensor(v)), "Nonfinite model")
    if phase == "formal":
        require(obj["global_step"] == 6000, "Final checkpoint step differs")
        result["checkpoint_sha256_by_step"] = {str(s): sha(out / f"{run_name(arm)}-step={s}.ckpt") for s in (5000, 5500, 6000)}
    result["checkpoint_sha256"] = checkpoint_sha
    replay = all(all(r["t1d_steps_1_3_equal"].values()) for r in result["ranks"])
    if arm == "t1dreplay":
        result["t1d_code_replay_exact"] = replay
        if not replay:
            result["status"] = "FAIL"
    return result
