"""S45-TR identity helpers (docs/EXPERIMENT_RECORDS.md#s45tr-prereg-20261030); CPU only, no scoring.

S45-TR = single-change fresh retrains of S45 (`c1sf_so3_s3407`, outputs/s45/baseline.json) plus the correspondence
predictor T0. Every S45 arm is the frozen S45 config with exactly its registered change and, where the change is a
mechanism rather than a key, an opt-in hook installed by `install_hooks` before the model / datasets are built.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from tools.frozen import accepts

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/s45tr/20261030"
CONFIGS = ROOT / "configs/s45tr_20261030"
PROTOCOL = OUT / "protocol.json"
MANIFESTS = OUT / "manifests"
PHYSICAL_GPUS = (2, 3, 4, 5, 6, 7)
PAIRS = ((2, 3), (4, 5), (6, 7))

S45_CONFIG = ROOT / "configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml"
S45_CONFIG_SHA256 = "521d2d4aa2558d1eac33a77e23d972a1258fdc7733c4b35406e62215eaccd29f"
S45_RUN = ROOT / "outputs/c1sf/20261029/runs/c1sf_so3_s3407"
S45_CHECKPOINT = S45_RUN / "c1sf_so3_s3407-step=6000.ckpt"
S45_CHECKPOINT_SHA256 = "d0164d55660c31184e8fd0b70f22f56cf9a7a064a216073862c9c56653d7dce0"
XAG_SOURCES = ROOT / "outputs/xag/sources"
T0_DIR = OUT / "t0"
T0_CHECKPOINT = T0_DIR / "t0_final.ckpt"
N4_BANK = OUT / "n4_bank"

#: T1a event-graph node features (semkine/event_gnn.py defaults, no readout)
T1A_GNN = {"HIDDEN": 96, "K": 8, "WINDOW": 32, "LAYERS": 3, "T_SCALE": 1.0}

#: arm -> the registered change, as dotted-key -> value on the S45 config
ARMS = {
    "s45replay": {},
    "n1": {"S45TR.HOOK": "n1"},
    "n2": {"S45TR.HOOK": "n2", "S45TR.N2_LAMBDA": 1.0},
    "n3": {"S45TR.HOOK": "n3", "S45TR.N3_OFFSET_MS": [100, 1000]},
    "n4": {"S45TR.HOOK": "n4", "S45TR.N4_BANK": str(N4_BANK), "S45TR.N4_P": 0.5},
    "t1a": {"S45TR.HOOK": "t1a", "S45TR.T1A_GNN": T1A_GNN},
    "t1b": {"S45TR.HOOK": "t1b", "MODEL.XAG_ASSOC": "face_route", "MODEL.XAG_GNN": "legacy",
            "S45TR.T0_CHECKPOINT": str(T0_CHECKPOINT), "S45TR.T1B_TAU": 0.5},
    "t1d": {"MODEL.XAG_ASSOC": "face_route", "MODEL.XAG_GNN": "legacy", "DATA.XAG_SOURCE_ROOT": str(XAG_SOURCES)},
}
#: arms that get a formal 6000-step training (s45replay is a 50-step code-drift check only)
FORMAL_ARMS = ("n1", "n2", "n3", "n4", "t1a", "t1b", "t1d")
#: arms whose training packets must equal S45's packet for packet (n3 / n4 change the prior by design;
#: t1d adds the XAG source fields, so only its small fields are compared)
SAME_PACKET_ARMS = ("s45replay", "n1", "n2", "t1a", "t1b")
SAME_SMALL_FIELD_ARMS = SAME_PACKET_ARMS + ("t1d",)
PRIOR_ARMS = ("n3", "n4")
#: arms whose initial model tensors differ from S45's by registered keys: prefix -> reason
INIT_EXCEPTIONS = {"t1a": ("xyz.rel.0.weight", "xyz.evgnn.")}


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


def run_name(arm, phase="formal"):
    return f"s45tr_{arm}_s3407" + ("_preflight" if phase == "preflight" else "")


def config_path(arm, phase):
    return CONFIGS / (run_name(arm, phase) + ".yaml")


def output_dir(arm, phase):
    return OUT / ("preflight" if phase == "preflight" else "runs") / run_name(arm)


def apply_change(cfg, change):
    for dotted, value in change.items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = copy.deepcopy(value)
    return cfg


def expected_raw(arm, phase, s45_raw):
    """The arm's YAML content: S45 + its one change + identity / budget / paths."""
    require(arm in ARMS, "unknown arm " + repr(arm))
    require(phase in ("preflight", "formal") and (phase == "preflight" or arm in FORMAL_ARMS), "unregistered phase")
    cfg = copy.deepcopy(s45_raw)
    cfg.pop("C1SF", None)
    cfg = apply_change(cfg, ARMS[arm])
    cfg.setdefault("S45TR", {})["ARM"] = arm
    out = output_dir(arm, phase)
    cfg["TRAIN"]["RUN_NAME"] = run_name(arm, phase)
    cfg["TRAIN"]["OUTPUT_DIR"] = str(out)
    cfg["EVAL"]["OUTPUT_DIR"] = str(out / "eval")
    if phase == "preflight":
        cfg["TRAIN"]["MAX_STEPS"] = 50
        cfg["TRAIN"]["VAL_CHECK_INTERVAL"] = 25
    return cfg


def hook_of(cfg):
    return (cfg.get("S45TR") or {}).get("HOOK")


def install_hooks(cfg):
    """Install the arm's opt-in hook for this process (before the model / datasets are built); returns its name."""
    s = cfg.get("S45TR") or {}
    hook = s.get("HOOK")
    if hook is None:
        return None
    if hook == "n1":
        from semkine import s45tr_n1
        s45tr_n1.install()
    elif hook == "n2":
        from semkine import s45tr_n2
        s45tr_n2.install(float(s["N2_LAMBDA"]))
    elif hook == "n3":
        from semkine import s45tr_prior
        lo, hi = s["N3_OFFSET_MS"]
        s45tr_prior.install_n3(int(lo), int(hi))
    elif hook == "n4":
        from semkine import s45tr_prior
        s45tr_prior.install_n4(s["N4_BANK"], float(s["N4_P"]))
    elif hook == "t1a":
        from semkine import s45tr_t1a
        s45tr_t1a.install(s["T1A_GNN"])
    elif hook == "t1b":
        from semkine import s45tr_t1b
        s45tr_t1b.install(s["T0_CHECKPOINT"], float(s["T1B_TAU"]))
    else:
        raise ValueError("unregistered S45-TR hook " + repr(hook))
    return hook


def frozen_files():
    """{relative path: sha256} over the protocol and every manifest frozen so far."""
    p = read(PROTOCOL)
    files = dict(p["file_sha256"])
    for m in sorted(MANIFESTS.glob("*.json")):
        files.update(read(m)["file_sha256"])
    return files


def verify_protocol():
    p = read(PROTOCOL)
    require(p["study"] == "S45TR_20261030" and p["seed"] == 3407, "Wrong S45-TR identity")
    for name, digest in frozen_files().items():
        require(accepts(name, digest), "Frozen S45-TR file changed: " + name)
    return p


def config_frozen(config):
    rel = str(Path(config).resolve().relative_to(ROOT))
    files = frozen_files()
    require(rel in files and accepts(config, files[rel]), "Config not frozen: " + rel)


def _init_matches(arm, initial, base):
    allowed = INIT_EXCEPTIONS.get(arm, ())

    def exempt(key):
        return any(key == a or (a.endswith(".") and key.startswith(a)) for a in allowed)
    shared = {k for k in base if not exempt(k)}
    require(shared <= set(initial), "S45 tensors missing from the arm's initial state")
    require(all(initial[k] == base[k] for k in shared), "Initial S45 tensors differ")
    extra = sorted(k for k in initial if k not in base)
    require(all(exempt(k) for k in extra), "Unregistered new tensors: " + ", ".join(extra[:5]))
    return extra


def verify_run(arm, phase):
    """Real per-rank input streams and shared initial tensors against S45's recorded training; never scores."""
    import numpy as np
    import torch
    out = output_dir(arm, phase)
    name = run_name(arm, phase)
    steps = 50 if phase == "preflight" else 6000
    checkpoint_sha = sha(out / "repro_final.ckpt")
    result = {"status": "PASS", "arm": arm, "phase": phase, "output": str(out), "ranks": []}
    for rank in range(2):
        done = read(out / f"completion_rank{rank}.json")
        require(done["status"] == "COMPLETE" and done["step"] == done["batches"] == steps, "Incomplete rank")
        require(done["final_checkpoint_sha256"] == checkpoint_sha, "Final checkpoint SHA differs from completion")
        actual = np.fromfile(out / f"indices_rank{rank}.i32", dtype=np.int32)
        ref = np.fromfile(S45_RUN / f"indices_rank{rank}.i32", dtype=np.int32)[:steps * 512]
        require(actual.size == steps * 512 and np.array_equal(actual, ref), "Actual sample indices differ from S45")
        initial = read(out / f"state_rank{rank}_step000000.json")["model"]
        base = read(S45_RUN / f"state_rank{rank}_step000000.json")["model"]
        new_tensors = _init_matches(arm, initial, base)
        rows = [json.loads(s) for s in (out / f"batches_rank{rank}.jsonl").read_text().splitlines()]
        ref_rows = [json.loads(s) for s in (S45_RUN / f"batches_rank{rank}.jsonl").read_text().splitlines()][:steps]
        require(len(rows) == steps, "Input journal denominator differs")
        small_diff = full_diff = 0
        for row, old in zip(rows, ref_rows):
            require(row["step"] == old["step"] and row["indices_sha256"] == old["indices_sha256"], "Index stream differs")
            small_diff += row["small_fields_sha256"] != old["small_fields_sha256"]
            if old.get("full_packet_sha256") is not None and row.get("full_packet_sha256") is not None:
                full_diff += row["full_packet_sha256"] != old["full_packet_sha256"]
        if arm in SAME_SMALL_FIELD_ARMS:
            require(small_diff == 0, "Training packets (small fields) differ from S45 for a same-input arm")
        if arm in SAME_PACKET_ARMS:
            require(full_diff == 0, "Full training packets differ from S45 for a same-input arm")
        if arm in PRIOR_ARMS:
            require(small_diff >= 1, "Prior arm did not change its training priors")
        replay = None
        if arm == "s45replay":
            # Steps 1-3 are recorded in both journals: same code + same inputs => same predictions / gradients.
            fields = ("prediction_sha256", "objective_sha256", "gradients")
            replay = {str(s): all(rows[s - 1].get(f) == ref_rows[s - 1].get(f) for f in fields) for s in (1, 2, 3)}
        rt = read(out / f"effective_runtime_rank{rank}.json")
        loader = rt["actual_train_loader"]
        ref_rt = read(S45_RUN / f"effective_runtime_rank{rank}.json")
        require(rt["world_size"] == 2 and rt["precision"] == "bf16" and rt["effective_seed"] == 3407, "Runtime changed")
        require(loader == {**ref_rt["actual_train_loader"]}, "Loader contract differs from S45")
        require(rt["actual_math_flags"] == ref_rt["actual_math_flags"], "Strict math flags differ from S45")
        if phase == "preflight":
            for step in (25, 50):
                val = read(out / f"validation_rank{rank}_step{step:06}.json")
                require(val["status"] == "PASS" and val["batches"] == 126 and val["model_state_unchanged"],
                        "Full validation was not verified")
            require(rows[25]["step"] == 26, "Training did not return after validation")
        else:
            for step in range(500, 6001, 500):
                require((out / f"{name}-step={step}.ckpt").is_file(), "Missing fixed checkpoint grid")
        result["ranks"].append({"rank": rank, "packets_small_fields_differing": small_diff,
                                "packets_full_differing": full_diff, "new_initial_tensors": new_tensors,
                                "s45_replay_steps_equal": replay})
    obj = torch.load(out / "repro_final.ckpt", map_location="cpu")
    require(all(bool(torch.isfinite(v).all()) for v in obj["state_dict"].values() if torch.is_tensor(v)),
            "Nonfinite final model")
    if phase == "formal":
        require(obj["global_step"] == 6000, "Final checkpoint step differs")
        result["checkpoint_sha256_by_step"] = {str(s): sha(out / f"{name}-step={s}.ckpt") for s in (5000, 5500, 6000)}
    result["checkpoint"] = str(out / "repro_final.ckpt")
    result["checkpoint_sha256"] = checkpoint_sha
    if arm == "s45replay":
        result["s45_code_replay_exact"] = all(all(r["s45_replay_steps_equal"].values()) for r in result["ranks"])
    return result
