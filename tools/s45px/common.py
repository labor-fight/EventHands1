"""S45-PX identity helpers (docs/EXPERIMENT_RECORDS.md#s45px-prereg-20261030); CPU only, no scoring.

S45-PX = two fresh single-change retrains of S45 (`c1sf_so3_s3407`, outputs/s45/baseline.json) that add the associated
previous-mesh vertex position to every matched relation record (semkine/s45px.py): `pxc` in camera XYZ (the literal
request), `pxm` relative to the previous wrist (MANO's frame without its translation). The control is the existing
S45 run (same strict contract, seed 3407, sample indices, packets and every shared initial tensor); each preflight
must reproduce S45's recorded step 1 bit for bit (prediction, objective, every S45 gradient).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.frozen import accepts  # noqa: E402
from tools.s45tr.common import now, read, require, save, sha  # noqa: E402,F401

OUT = ROOT / "outputs/s45px/20261030"
CONFIGS = ROOT / "configs/s45px_20261030"
PROTOCOL = OUT / "protocol.json"
PHYSICAL_GPUS = (2, 3, 4, 5, 6, 7)
S45_CONFIG = ROOT / "configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml"
S45_CONFIG_SHA256 = "521d2d4aa2558d1eac33a77e23d972a1258fdc7733c4b35406e62215eaccd29f"
S45_RUN = ROOT / "outputs/c1sf/20261029/runs/c1sf_so3_s3407"
S45TR = ROOT / "outputs/s45tr/20261030"
#: arm -> frame of the added vertex position (semkine/s45px.py FRAMES)
ARMS = {"pxc": "camera", "pxm": "mano"}
NEW_TENSOR = "xyz.sign_in.weight"
SOURCES = ("semkine/s45px.py", "tools/s45px/__init__.py", "tools/s45px/common.py", "tools/s45px/make_configs.py",
           "tools/s45px/train.py", "tools/s45px/accept.py", "tools/s45px/evaluate.py", "tools/s45px/readouts.py",
           "tools/s45px/decide.py", "tools/s45px/run.py", "tests/test_s45px.py")
DEPENDENCIES = ("configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml", "tools/s45tr/common.py", "tools/s45tr/train.py",
                "tools/s45tr/readouts.py", "tools/s45tr/evaluate.py", "tools/xyz_c1_repro/train.py",
                "tools/xcl/evaluate.py", "tools/make_s36_row.py", "tools/report_table.py", "semkine/xyz_mesh.py",
                "semkine/train.py", "semkine/dataset.py", "semkine/eval_track.py", "model/model.py", "model/config.py")


def run_name(arm, phase="formal"):
    return f"s45px_{arm}_s3407" + ("_preflight" if phase == "preflight" else "")


def config_path(arm, phase):
    return CONFIGS / (run_name(arm, phase) + ".yaml")


def output_dir(arm, phase):
    return OUT / ("preflight" if phase == "preflight" else "runs") / run_name(arm)


def expected_raw(arm, phase, s45_raw):
    """The arm's YAML content: S45's + `S45PX {ARM, FRAME}` in place of `C1SF` + identity / budget / paths."""
    require(arm in ARMS and phase in ("preflight", "formal"), "unregistered S45-PX arm / phase")
    cfg = copy.deepcopy(s45_raw)
    require(cfg.pop("C1SF") == {"ARM": "so3"}, "S45 config identity changed")
    cfg["S45PX"] = {"ARM": arm, "FRAME": ARMS[arm]}
    out = output_dir(arm, phase)
    cfg["TRAIN"]["RUN_NAME"] = run_name(arm, phase)
    cfg["TRAIN"]["OUTPUT_DIR"] = str(out)
    cfg["EVAL"]["OUTPUT_DIR"] = str(out / "eval")
    if phase == "preflight":
        cfg["TRAIN"]["MAX_STEPS"] = 50
        cfg["TRAIN"]["VAL_CHECK_INTERVAL"] = 25
    return cfg


def install_hook(cfg):
    frame = (cfg.get("S45PX") or {}).get("FRAME")
    require(frame is not None, "not an S45-PX config")
    from semkine import s45px
    s45px.install(frame, cfg["MANO"]["NPZ"])
    return frame


def frozen_files():
    return read(PROTOCOL)["file_sha256"]


def verify_protocol():
    p = read(PROTOCOL)
    require(p["study"] == "S45PX_20261030" and p["status"] == "FROZEN", "S45-PX protocol identity")
    for name, digest in p["file_sha256"].items():
        require(accepts(ROOT / name, digest), "frozen S45-PX file changed: " + name)
    return p


def config_frozen(config):
    rel = str(Path(config).resolve().relative_to(ROOT))
    files = frozen_files()
    require(rel in files and accepts(config, files[rel]), "Config not frozen: " + rel)


def journal(directory, rank, steps=None):
    rows = [json.loads(s) for s in (Path(directory) / f"batches_rank{rank}.jsonl").read_text().splitlines()]
    return rows if steps is None else rows[:steps]


def verify_run(arm, phase):
    """Per-rank input streams, packets, initial tensors and step 1 against S45's recorded training; never scores."""
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
        ref = np.fromfile(S45_RUN / f"indices_rank{rank}.i32", dtype=np.int32)[:steps * 512]
        require(actual.size == steps * 512 and np.array_equal(actual, ref), "Sample indices differ from S45")
        initial = read(out / f"state_rank{rank}_step000000.json")["model"]
        base = read(S45_RUN / f"state_rank{rank}_step000000.json")["model"]
        require(set(initial) - set(base) == {NEW_TENSOR} and set(base) <= set(initial), "Unregistered tensor set")
        require(all(initial[k] == base[k] for k in base), "Initial S45 tensors differ")
        rows, ref_rows = journal(out, rank), journal(S45_RUN, rank, steps)
        require(len(rows) == steps, "Input journal denominator differs")
        small = full = full_compared = 0
        for row, old in zip(rows, ref_rows):
            require(row["step"] == old["step"] and row["indices_sha256"] == old["indices_sha256"], "Index stream differs")
            small += row["small_fields_sha256"] != old["small_fields_sha256"]
            if old.get("full_packet_sha256") is not None and row.get("full_packet_sha256") is not None:
                full_compared += 1
                full += row["full_packet_sha256"] != old["full_packet_sha256"]
        require(small == 0 and full == 0 and full_compared > 0, "Training packets differ from S45")
        # step 1: the new map is zero, so the function is S45's; step 2: the arm has left S45
        one, ref_one = rows[0], ref_rows[0]
        grads, ref_grads = one["gradients"], ref_one["gradients"]
        step1 = {"prediction": one["prediction_sha256"] == ref_one["prediction_sha256"],
                 "objective": one["objective_sha256"] == ref_one["objective_sha256"],
                 "s45_gradients": set(grads) - set(ref_grads) == {NEW_TENSOR}
                 and all(grads[k] == ref_grads[k] for k in ref_grads)}
        departs = rows[1]["prediction_sha256"] != ref_rows[1]["prediction_sha256"]
        rt, ref_rt = read(out / f"effective_runtime_rank{rank}.json"), read(S45_RUN / f"effective_runtime_rank{rank}.json")
        require(rt["world_size"] == 2 and rt["precision"] == "bf16" and rt["effective_seed"] == 3407, "Runtime changed")
        require(rt["actual_train_loader"] == ref_rt["actual_train_loader"], "Loader contract differs from S45")
        require(rt["actual_math_flags"] == ref_rt["actual_math_flags"], "Strict math flags differ from S45")
        require(rt["s45px_frame"] == ARMS[arm] and rt["s45px_new_map_zero_at_fit_start"] is True,
                "Installed frame / zero initial map differs from the registered arm")
        if phase == "preflight":
            for step in (25, 50):
                val = read(out / f"validation_rank{rank}_step{step:06}.json")
                require(val["status"] == "PASS" and val["batches"] == 126 and val["model_state_unchanged"],
                        "Full validation was not verified")
            require(rows[25]["step"] == 26, "Training did not return after validation")
        else:
            for step in range(500, 6001, 500):
                require((out / f"{run_name(arm)}-step={step}.ckpt").is_file(), "Missing fixed checkpoint grid")
        result["ranks"].append({"rank": rank, "packets_full_compared": full_compared, "s45_step1_equal": step1,
                                "departs_from_s45_at_step2": departs})
    obj = torch.load(out / "repro_final.ckpt", map_location="cpu")
    require(all(bool(torch.isfinite(v).all()) for v in obj["state_dict"].values() if torch.is_tensor(v)), "Nonfinite model")
    require(bool(obj["state_dict"][NEW_TENSOR].abs().sum() > 0), "The new map never learned")
    if phase == "formal":
        require(obj["global_step"] == 6000, "Final checkpoint step differs")
        result["checkpoint_sha256_by_step"] = {str(s): sha(out / f"{run_name(arm)}-step={s}.ckpt") for s in (5000, 5500, 6000)}
    result["checkpoint_sha256"] = checkpoint_sha
    exact = all(all(r["s45_step1_equal"].values()) and r["departs_from_s45_at_step2"] for r in result["ranks"])
    result["s45_step1_exact_and_departs_at_step2"] = exact
    if not exact:
        result["status"] = "FAIL"
    return result
