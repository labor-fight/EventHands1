"""S45-AT identity helpers (docs/EXPERIMENT_RECORDS.md#s45at-prereg-20261031); CPU only, no scoring.

S45-AT = which part of S45 is the problem (TRAIN.md section 9): five fresh 6000-step retrains, each one change from its
base under S45's strict contract (seed 3407):
  dz        S45 + depth scale augmentation (AUG.DOMRAND.SCALE_MODE: depth, as C1SF dz)
  t1ddz     S45-TR T1d (XAG GT face routes) + the same augmentation          (reads the XAG sources)
  t1dnoylf  S45-TR T1d trained without the subject ylf (8 sequences)          (reads the XAG sources)
  s45samp   S45 with the dedicated sampling stream (same seed / initial tensors, different data order): S45's own
            training-path noise for the paired comparisons (the strict contract fixes seed 3407)
  w200      S45 with a 200 ms evidence window and the previous state 50 ms before the window end
            (DATA.WINDOW_MIN = WINDOW_MAX = 200, DATA.PREV_LAG_MS 50, EVAL.WINDOW_MS 200), evaluated on the same
            50 ms endpoint grid with each window clipped at its run start
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

OUT = ROOT / "outputs/s45at/20261031"
CONFIGS = ROOT / "configs/s45at_20261031"
PROTOCOL = OUT / "protocol.json"
MANIFESTS = OUT / "manifests"
TAG = "S45AT-PREREG-20261031"
PHYSICAL_GPUS = (2, 3, 4, 5, 6, 7)
S45TR = ROOT / "outputs/s45tr/20261030"
S45CT = ROOT / "outputs/s45ct/20261030"
S45_CONFIG = ROOT / "configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml"
S45_RUN = ROOT / "outputs/c1sf/20261029/runs/c1sf_so3_s3407"
S45_EVAL = ROOT / "outputs/c1sf/20261029/evaluations/so3"
T1D_CONFIG = ROOT / "configs/s45tr_20261030/s45tr_t1d_s3407.yaml"
T1D_RUN = S45TR / "runs/s45tr_t1d_s3407"
WINDOW = 200
ARMS = {
    "dz": {"base": "s45", "change": {"AUG.DOMRAND.SCALE_MODE": "depth"}},
    "t1ddz": {"base": "t1d", "change": {"AUG.DOMRAND.SCALE_MODE": "depth"}},
    "t1dnoylf": {"base": "t1d", "change": {}, "exclude_subject": "ylf"},
    "s45samp": {"base": "s45", "change": {}, "sampler": "dedicated"},
    "w200": {"base": "s45", "change": {"DATA.WINDOW_MIN": WINDOW, "DATA.WINDOW_MAX": WINDOW, "DATA.PREV_LAG_MS": 50,
                                        "EVAL.WINDOW_MS": WINDOW}},
}
XAG_ARMS = ("t1ddz", "t1dnoylf")
#: arms whose sample-index stream must equal their base run's (same sampler and the same training dataset)
SAME_INDEX_ARMS = ("dz", "t1ddz")
SOURCES = ("tools/s45at/__init__.py", "tools/s45at/common.py", "tools/s45at/make_configs.py", "tools/s45at/train.py",
           "tools/s45at/accept.py", "tools/s45at/evaluate.py", "tools/s45at/readouts.py", "tools/s45at/decide.py",
           "tests/test_s45at.py")
DEPENDENCIES = ("configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml", "configs/s45tr_20261030/s45tr_t1d_s3407.yaml",
                "tools/xyz_c1_repro/train.py", "tools/xcl/evaluate.py", "tools/s45tr/readouts.py",
                "tools/s45tr/common.py", "tools/make_s36_row.py", "semkine/train.py", "semkine/dataset.py",
                "semkine/eval_track.py", "semkine/xag.py", "semkine/xag_sources.py", "semkine/xyz_mesh.py",
                "model/model.py", "tools/s45rt/barrier_r2.py", "tools/s45rt/chain_r2.py")


def run_name(arm, phase="formal"):
    require(arm in ARMS, "unregistered S45-AT arm " + repr(arm))
    return f"s45at_{arm}_s3407" + ("_preflight" if phase == "preflight" else "")


def config_path(arm, phase):
    return CONFIGS / (run_name(arm, phase) + ".yaml")


def output_dir(arm, phase):
    return OUT / ("preflight" if phase == "preflight" else "runs") / run_name(arm)


def base_config(arm):
    return S45_CONFIG if ARMS[arm]["base"] == "s45" else T1D_CONFIG


def base_run(arm):
    return S45_RUN if ARMS[arm]["base"] == "s45" else T1D_RUN


def apply_change(cfg, change):
    for dotted, value in change.items():
        node = cfg
        *path, leaf = dotted.split(".")
        for key in path:
            node = node[key]
        node[leaf] = value
    return cfg


def expected_raw(arm, phase, base_raw):
    """The arm's YAML content: its base (S45 or S45-TR T1d) + its one change + identity / budget / paths."""
    require(arm in ARMS and phase in ("preflight", "formal"), "unregistered S45-AT arm / phase")
    cfg = copy.deepcopy(base_raw)
    if ARMS[arm]["base"] == "s45":
        cfg.pop("C1SF", None)
    else:
        require(cfg.pop("S45TR") == {"ARM": "t1d"}, "T1d config identity changed")
    cfg = apply_change(cfg, ARMS[arm]["change"])
    cfg["S45AT"] = {"ARM": arm, "BASE": ARMS[arm]["base"], "EXCLUDE_SUBJECT": ARMS[arm].get("exclude_subject"),
                    "SAMPLER": ARMS[arm].get("sampler", "legacy_none")}
    out = output_dir(arm, phase)
    cfg["TRAIN"]["RUN_NAME"] = run_name(arm, phase)
    cfg["TRAIN"]["OUTPUT_DIR"] = str(out)
    cfg["EVAL"]["OUTPUT_DIR"] = str(out / "eval")
    if phase == "preflight":
        cfg["TRAIN"]["MAX_STEPS"] = 50
        cfg["TRAIN"]["VAL_CHECK_INTERVAL"] = 25
    return cfg


# --------------------------------------------------------------------------- opt-in hooks
def install_train_hook(cfg):
    """t1dnoylf: the training split's sequence list drops the subject's sequences (validation untouched)."""
    subject = (cfg.get("S45AT") or {}).get("EXCLUDE_SUBJECT")
    if not subject:
        return None
    import semkine.dataset as D
    native = D.sequences_for_split
    if getattr(native, "_s45at_excluded", None) == subject:
        return subject

    def sequences_for_split(root, split, manifest=None):
        seqs = native(root, split, manifest)
        return [s for s in seqs if not (split == "train" and s[0].startswith(subject + "_"))]
    sequences_for_split._s45at_excluded = subject
    D.sequences_for_split = sequences_for_split
    return subject


_W = {"runs": None, "window": None}


def install_window_patch(cfg):
    """w200 evaluation: keep the 50 ms endpoint grid; each window = min(200, end - run start + 1) ms (clipped at the
    run start, as the training windows); the packet duration is that clipped window. Others: nothing."""
    window = int(cfg["EVAL"].get("WINDOW_MS", 50))
    if window == 50:
        return None
    from semkine import eval_track as ET
    if getattr(ET._window_events, "_s45at_window", None) == window:
        return window
    load, events_of, packet_of = ET.load_sequence, ET._window_events, ET.make_eval_packet

    def load_sequence(root, legacy_dir, seq):
        result = load(root, legacy_dir, seq)
        _W["runs"] = [tuple(map(int, r)) for r in __import__("numpy").asarray(result[2]["valid_runs_ms"]).reshape(-1, 2)]
        return result

    def window_events(events, offsets, tsub, end, w):
        if int(w) == 50:
            starts = [a for a, b in _W["runs"] if a <= int(end) < b]
            require(len(starts) == 1, "endpoint outside every valid run")
            w = min(window, int(end) - starts[0] + 1)
        _W["window"] = int(w)
        return events_of(events, offsets, tsub, end, w)

    def make_eval_packet(ev5, prev, betas, camera_K, window_ms, device):
        if int(window_ms) == 50:
            window_ms = _W["window"]
        return packet_of(ev5, prev, betas, camera_K, window_ms, device)
    window_events._s45at_window = window
    ET.load_sequence, ET._window_events, ET.make_eval_packet = load_sequence, window_events, make_eval_packet
    return window


# --------------------------------------------------------------------------- protocol / receipts
def verify_protocol():
    p = read(PROTOCOL)
    require(p["study"] == "S45AT_20261031" and p["status"] == "FROZEN", "S45-AT protocol identity")
    for name, digest in p["file_sha256"].items():
        require(accepts(ROOT / name, digest), "frozen S45-AT file changed: " + name)
    return p


def config_frozen(config):
    rel = str(Path(config).resolve().relative_to(ROOT))
    files = read(PROTOCOL)["file_sha256"]
    require(rel in files and accepts(config, files[rel]), "Config not frozen: " + rel)


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
    """Completion, checkpoint grid, initial tensors (and, where defined, the index stream) against the base run."""
    import numpy as np
    import torch
    out = output_dir(arm, phase)
    steps = 50 if phase == "preflight" else 6000
    ref_run = base_run(arm)
    checkpoint_sha = sha(out / "repro_final.ckpt")
    result = {"status": "PASS", "arm": arm, "phase": phase, "output": str(out), "base_run": str(ref_run), "ranks": []}
    for rank in range(2):
        done = read(out / f"completion_rank{rank}.json")
        require(done["status"] == "COMPLETE" and done["step"] == done["batches"] == steps, "Incomplete rank")
        require(done["final_checkpoint_sha256"] == checkpoint_sha, "Final checkpoint SHA differs from completion")
        actual = np.fromfile(out / f"indices_rank{rank}.i32", dtype=np.int32)
        require(actual.size == steps * 512, "Index tape denominator differs")
        if arm in SAME_INDEX_ARMS:
            ref = np.fromfile(ref_run / f"indices_rank{rank}.i32", dtype=np.int32)[:steps * 512]
            require(np.array_equal(actual, ref), "Sample indices differ from the base run")
        initial = read(out / f"state_rank{rank}_step000000.json")["model"]
        require(initial == read(ref_run / f"state_rank{rank}_step000000.json")["model"], "Initial tensors differ from base")
        require(len(journal(out, rank)) == steps, "Input journal denominator differs")
        rt, ref_rt = read(out / f"effective_runtime_rank{rank}.json"), read(ref_run / f"effective_runtime_rank{rank}.json")
        require(rt["world_size"] == 2 and rt["precision"] == "bf16" and rt["effective_seed"] == 3407, "Runtime changed")
        require(rt["actual_math_flags"] == ref_rt["actual_math_flags"], "Strict math flags differ from the base run")
        require(rt["s45at_arm"] == arm and rt["s45at_excluded"] == ARMS[arm].get("exclude_subject"), "Arm hook differs")
        if phase == "preflight":
            for step in (25, 50):
                val = read(out / f"validation_rank{rank}_step{step:06}.json")
                require(val["status"] == "PASS" and val["model_state_unchanged"], "Full validation was not verified")
        else:
            for step in range(500, 6001, 500):
                require((out / f"{run_name(arm)}-step={step}.ckpt").is_file(), "Missing fixed checkpoint grid")
        result["ranks"].append({"rank": rank, "index_stream_equal_to_base": arm in SAME_INDEX_ARMS})
    obj = torch.load(out / "repro_final.ckpt", map_location="cpu")
    require(all(bool(torch.isfinite(v).all()) for v in obj["state_dict"].values() if torch.is_tensor(v)), "Nonfinite model")
    if phase == "formal":
        require(obj["global_step"] == 6000, "Final checkpoint step differs")
        result["checkpoint_sha256_by_step"] = {str(s): sha(out / f"{run_name(arm)}-step={s}.ckpt") for s in (5000, 5500, 6000)}
    result["checkpoint_sha256"] = checkpoint_sha
    return result
