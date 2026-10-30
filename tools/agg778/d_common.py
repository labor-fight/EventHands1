"""AGG778 Stage D registry and CPU-only identity helpers (docs/EXPERIMENT_RECORDS.md#agg778-prereg-20261024).

An arm = one aggregation variant (`MODEL.XYZ_AGG`: matpt / mass / fanout / vin) on one recipe:

    a0  the C1 recipe (configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml), paired with C1 xyz_c1_stable_s3407
    a3  the XCL A3F0 recipe (configs/xcl_20261019/xcl_a3f0_s3407.yaml; oracle), paired with xcl_a3f0_s3407

Every arm config is its control's config with only MODEL.XYZ_AGG, the run name / output paths (and, on a3, the XCL
cell label the native training audit records) changed. All eight formal and eight preflight configs are generated and
frozen before any reading; Stage B selects which are trained.

    python tools/agg778/d_common.py write-configs
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_D = ROOT / "outputs" / "agg778" / "d"
CONFIG_DIR = ROOT / "configs" / "agg778"
VARIANTS = ("matpt", "mass", "fanout", "vin")
CONTROLS = {
    "a0": {"config": "configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml",
           "run": "outputs/xyz_c1_repro/20261018/runs/xyz_c1_stable_s3407", "name": "xyz_c1_stable_s3407",
           "checkpoint": "outputs/semkine/xyz_c1_stable_s3407/xyz_c1_stable_s3407-step=6000.ckpt",
           "main_row": "outputs/semkine/xyz_c1_baseline_main_row.json",
           "xcl_eval": "outputs/xcl/20261019/evaluations/xyz_c1_stable_s3407"},
    "a3": {"config": "configs/xcl_20261019/xcl_a3f0_s3407.yaml",
           "run": "outputs/xcl/20261019/runs/xcl_a3f0_s3407", "name": "xcl_a3f0_s3407",
           "checkpoint": "outputs/xcl/20261019/runs/xcl_a3f0_s3407/repro_final.ckpt",
           "main_row": None,
           "xcl_eval": "outputs/xcl/20261019/evaluations/xcl_a3f0_s3407"},
}
GPU_GROUPS = ((4, 5), (6, 7))


def arm_name(variant: str, cond: str) -> str:
    return f"agg778_{variant}_{cond}_s3407"


ARMS = {arm_name(v, c): {"variant": v, "cond": c} for v in VARIANTS for c in CONTROLS}


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value, exclusive=False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with p.open("x") as f:
            f.write(text)
    else:
        tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
        tmp.write_text(text)
        tmp.replace(p)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def paths(arm: str) -> dict:
    return {"config": CONFIG_DIR / f"{arm}.yaml", "preflight_config": CONFIG_DIR / "preflight" / f"{arm}_preflight.yaml",
            "output": OUT_D / "runs" / arm, "preflight_output": OUT_D / "preflight" / f"{arm}_preflight"}


def expected_config(arm: str, phase: str) -> dict:
    """The only configuration an arm may train with (its control's, with the registered differences)."""
    sys.path[:0] = [str(ROOT), str(ROOT / "model")]
    from config import load_config
    a = ARMS[arm]
    cfg = load_config(str(ROOT / CONTROLS[a["cond"]]["config"]))
    cfg.pop("_config_path")
    cfg["MODEL"]["XYZ_AGG"] = a["variant"]
    p = paths(arm)
    if phase == "preflight":
        out = p["preflight_output"]
        cfg["TRAIN"].update(RUN_NAME=arm + "_preflight", OUTPUT_DIR=str(out), MAX_STEPS=50, VAL_CHECK_INTERVAL=25)
    else:
        out = p["output"]
        cfg["TRAIN"].update(RUN_NAME=arm, OUTPUT_DIR=str(out))
    cfg["EVAL"]["OUTPUT_DIR"] = str(out / "eval")
    if a["cond"] == "a3":
        # semkine/train.py records XAG-study provenance through the XCL audit when this flag is on
        cfg["XCL"] = {"ENABLED": True, "CELL": arm}
    return cfg


def write_configs():
    import yaml
    written = []
    for arm in ARMS:
        for phase, key in (("formal", "config"), ("preflight", "preflight_config")):
            dst = paths(arm)[key]
            cfg = expected_config(arm, phase)
            dst.parent.mkdir(parents=True, exist_ok=True)
            text = yaml.safe_dump(cfg, sort_keys=True)
            if dst.exists():
                require(dst.read_text() == text, f"registered config differs: {dst}")
            else:
                dst.write_text(text)
            written.append(str(dst.relative_to(ROOT)))
    return written


def shared_initial_state(run_dir: Path, rank: int):
    return read(Path(run_dir) / f"state_rank{rank}_step000000.json")["model"]


def verify_pairing(arm: str, directory=None) -> dict:
    """Shared C1 parameters start from the control's values; per-rank actual index streams and per-step small-field
    packet fingerprints equal the control's."""
    a = ARMS[arm]
    ctrl = ROOT / CONTROLS[a["cond"]]["run"]
    run = Path(directory) if directory else paths(arm)["output"]
    out = {"arm": arm, "control": str(ctrl.relative_to(ROOT))}
    for rank in (0, 1):
        mine, theirs = shared_initial_state(run, rank), shared_initial_state(ctrl, rank)
        shared = sorted(set(theirs) & set(mine))
        extra = sorted(set(mine) - set(theirs))
        same = all(mine[k] == theirs[k] for k in shared)
        import numpy as np
        im = np.fromfile(run / f"indices_rank{rank}.i32", dtype=np.int32)
        ic = np.fromfile(ctrl / f"indices_rank{rank}.i32", dtype=np.int32)
        idx_same = len(im) > 0 and len(im) <= len(ic) and bool(np.array_equal(im, ic[:len(im)]))
        rows_m = [json.loads(x) for x in (run / f"batches_rank{rank}.jsonl").read_text().splitlines()]
        rows_c = [json.loads(x) for x in (ctrl / f"batches_rank{rank}.jsonl").read_text().splitlines()]
        n = min(len(rows_m), len(rows_c))
        small = all(rows_m[i]["small_fields_sha256"] == rows_c[i]["small_fields_sha256"] for i in range(n))
        out[f"rank{rank}"] = {"shared_tensors": len(shared), "shared_initial_equal": same, "new_tensors": extra,
                              "indices_equal": idx_same, "small_fields_equal_steps": n, "small_fields_equal": small}
    out["status"] = "PASS" if all(out[f"rank{r}"]["shared_initial_equal"] and out[f"rank{r}"]["indices_equal"]
                                  and out[f"rank{r}"]["small_fields_equal"] for r in (0, 1)) else "FAIL"
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["write-configs"]:
        print("\n".join(write_configs()))
    else:
        raise SystemExit("usage: d_common.py write-configs")
