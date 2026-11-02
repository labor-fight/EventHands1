"""S45-RT plan-2 identity helpers (docs/EXPERIMENT_RECORDS.md#s45rt-prereg-20261031); CPU only, no scoring.

T0P = the surface-patch router (semkine/s45rt_router.py), trained like S45-TR T0 (S45's training data minus the
held-out subject ylf, XAG pseudo-sources). T1p = S45 + `MODEL.XAG_ASSOC: face_route` with the frozen T0P's routes
(one change from S45, as S45-TR T1b with T0); fresh 6000 steps under S45's strict contract; its training packets,
sample indices and initial tensors must equal S45's (the routes change inside the model only).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from tools.frozen import accepts
from tools.s45rt.common import OUT, ROOT, S45_CONFIG, S45_RUN, read, require, sha

CONFIGS = ROOT / "configs/s45rt_20261031"
ROUTER_DIR = OUT / "t0p"
ROUTER_PREFLIGHT_DIR = OUT / "t0p_preflight"
ROUTER_CHECKPOINT = ROUTER_DIR / "t0p_final.ckpt"
ARMS = ("t1p",)
TAUS = {"ONHAND_TAU": 0.5, "PATCH_TAU": 0.5}
HELDOUT = "ylf"
FILES = ("semkine/s45rt_router.py", "tools/s45rt/plan2.py", "tools/s45rt/make_configs.py", "tools/s45rt/router_train.py",
         "tools/s45rt/router_gate.py", "tools/s45rt/train.py", "tools/s45rt/accept.py", "tools/s45rt/evaluate.py",
         "tools/s45rt/barrier.py", "tools/s45rt/freeze_plan2.py", "tests/test_s45rt.py", "semkine/s45ct_route.py",
         "semkine/encoder.py", "semkine/events.py")


def run_name(arm="t1p", phase="formal"):
    require(arm in ARMS, "unregistered S45-RT arm " + repr(arm))
    return f"s45rt_{arm}_s3407" + ("_preflight" if phase == "preflight" else "")


def config_path(arm, phase):
    return CONFIGS / (run_name(arm, phase) + ".yaml")


def output_dir(arm, phase):
    return OUT / ("preflight" if phase == "preflight" else "runs") / run_name(arm)


def expected_raw(arm, phase, s45_raw):
    """The arm's YAML content: S45 + face_route with T0P's routes + identity / budget / paths."""
    require(arm in ARMS and phase in ("preflight", "formal"), "unregistered S45-RT arm / phase")
    cfg = copy.deepcopy(s45_raw)
    cfg.pop("C1SF", None)
    require("XAG_ASSOC" not in cfg["MODEL"] and "XAG_GNN" not in cfg["MODEL"], "S45 config changed")
    cfg["MODEL"]["XAG_ASSOC"] = "face_route"
    cfg["MODEL"]["XAG_GNN"] = "legacy"
    cfg["S45RT"] = {"ARM": arm, "ROUTER": str(ROUTER_CHECKPOINT), **TAUS}
    out = output_dir(arm, phase)
    cfg["TRAIN"]["RUN_NAME"] = run_name(arm, phase)
    cfg["TRAIN"]["OUTPUT_DIR"] = str(out)
    cfg["EVAL"]["OUTPUT_DIR"] = str(out / "eval")
    if phase == "preflight":
        cfg["TRAIN"]["MAX_STEPS"] = 50
        cfg["TRAIN"]["VAL_CHECK_INTERVAL"] = 25
    return cfg


def install_hook(cfg):
    s = cfg.get("S45RT") or {}
    if not s:
        return None
    from semkine import s45rt_router
    s45rt_router.install(s["ROUTER"], float(s["ONHAND_TAU"]), float(s["PATCH_TAU"]))
    return s45rt_router.installed()


def install_eval_hook(cfg):
    """Training hook + the evaluator must not attach ground-truth XAG sources (T1p reads none; as S45-TR T1b)."""
    state = install_hook(cfg)
    require(state is not None, "T1p routing not installed")
    from semkine import eval_track as ET
    ET.attach_xag_sources = lambda batch, *args, **kwargs: batch
    return state


def gate_passed():
    gate = ROUTER_DIR / "gate.json"
    return gate.is_file() and read(gate)["status"] == "PASS"


def manifest_files():
    path = OUT / "manifests" / "plan2.json"
    return read(path)["file_sha256"] if path.is_file() else {}


def config_frozen(config):
    rel = str(Path(config).resolve().relative_to(ROOT))
    files = manifest_files()
    require(rel in files and accepts(config, files[rel]), "Config not frozen in manifests/plan2.json: " + rel)


def resolve(arm, step):
    receipt = read(OUT / "receipts" / f"formal_{arm}.json")
    require(receipt["status"] == "PASS", "arm has no verified formal training")
    checkpoint = output_dir(arm, "formal") / f"{run_name(arm)}-step={step}.ckpt"
    require(sha(checkpoint) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since its receipt")
    return checkpoint, config_path(arm, "formal")


def journal(directory, rank, steps=None):
    rows = [json.loads(s) for s in (Path(directory) / f"batches_rank{rank}.jsonl").read_text().splitlines()]
    return rows if steps is None else rows[:steps]


def verify_run(arm, phase):
    """Per-rank input streams, initial tensors and packets against S45's recorded training; never scores."""
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
        require(initial == read(S45_RUN / f"state_rank{rank}_step000000.json")["model"], "Initial tensors differ from S45")
        rows, ref_rows = journal(out, rank), journal(S45_RUN, rank, steps)
        require(len(rows) == steps, "Input journal denominator differs")
        small = full = full_compared = 0
        for row, old in zip(rows, ref_rows):
            require(row["step"] == old["step"] and row["indices_sha256"] == old["indices_sha256"], "Index stream differs")
            small += row["small_fields_sha256"] != old["small_fields_sha256"]
            if old.get("full_packet_sha256") is not None and row.get("full_packet_sha256") is not None:
                full_compared += 1
                full += row["full_packet_sha256"] != old["full_packet_sha256"]
        require(small == 0 and full == 0 and full_compared > 0, "Training packets differ from S45 (routes change in-model only)")
        rt, ref_rt = read(out / f"effective_runtime_rank{rank}.json"), read(S45_RUN / f"effective_runtime_rank{rank}.json")
        require(rt["world_size"] == 2 and rt["precision"] == "bf16" and rt["effective_seed"] == 3407, "Runtime changed")
        require(rt["actual_train_loader"] == ref_rt["actual_train_loader"], "Loader contract differs from S45")
        require(rt["actual_math_flags"] == ref_rt["actual_math_flags"], "Strict math flags differ from S45")
        require(rt["s45rt_router"] == {"checkpoint": str(ROUTER_CHECKPOINT), "onhand_tau": TAUS["ONHAND_TAU"],
                                       "patch_tau": TAUS["PATCH_TAU"]}, "Installed routing differs from the registered arm")
        require(rt["router_checkpoint_sha256"] == sha(ROUTER_CHECKPOINT), "Router checkpoint differs")
        if phase == "preflight":
            for step in (25, 50):
                val = read(out / f"validation_rank{rank}_step{step:06}.json")
                require(val["status"] == "PASS" and val["batches"] == 126 and val["model_state_unchanged"],
                        "Full validation was not verified")
            require(rows[25]["step"] == 26, "Training did not return after validation")
        else:
            for step in range(500, 6001, 500):
                require((out / f"{run_name(arm)}-step={step}.ckpt").is_file(), "Missing fixed checkpoint grid")
        result["ranks"].append({"rank": rank, "packets_full_compared": full_compared})
    obj = torch.load(out / "repro_final.ckpt", map_location="cpu")
    require(all(bool(torch.isfinite(v).all()) for v in obj["state_dict"].values() if torch.is_tensor(v)), "Nonfinite model")
    require(not any(k.startswith(("router", "t0p")) for k in obj["state_dict"]), "Router weights leaked into the model")
    if phase == "formal":
        require(obj["global_step"] == 6000, "Final checkpoint step differs")
        result["checkpoint_sha256_by_step"] = {str(s): sha(out / f"{run_name(arm)}-step={s}.ckpt") for s in (5000, 5500, 6000)}
    result["checkpoint_sha256"] = checkpoint_sha
    return result


__all__ = ["ARMS", "CONFIGS", "FILES", "HELDOUT", "ROUTER_CHECKPOINT", "ROUTER_DIR", "ROUTER_PREFLIGHT_DIR", "TAUS",
           "config_frozen", "config_path", "expected_raw", "gate_passed", "install_eval_hook", "install_hook",
           "output_dir", "resolve", "run_name", "verify_run", "S45_CONFIG"]
