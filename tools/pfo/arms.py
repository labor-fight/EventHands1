"""PFO training arms (docs/EXPERIMENT_RECORDS.md#pfo-prereg-20261103 and the phase-2 sections): name -> one change against
the base arm B, and the generated configs.

Base B = pfasrdz's recipe (configs/s45sm_20261031/s45sm_pfasrdz_s3407.yaml: S45 trunk, frozen sr1 + hard PFA, depth
augmentation, S45 loss / noise contract / 2 x 512 / Adam 0.004 / 500 warm-up / BF16 / 6000 steps) with the cosine
learning-rate decay written by S45-SM round 14. The seed sets `SEED` (initial values and the data stream).
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .common import CONFIGS, OUT, SM_CONFIGS, SR1, require

BASE_CONFIG = SM_CONFIGS / "s45sm_pfasrdz_s3407.yaml"
SEEDS = (3407, 3408, 3409)
PREFLIGHT_STEPS = 40

#: arm -> {"trunk": variant of semkine/pfo_trunk.py, "set": {dotted config key: value}, "phase": registration section}
ARMS = {
    "b": {"trunk": "base", "set": {}, "section": "pfo-p2a-prereg-20261103"},
    "nomp": {"trunk": "nomp", "set": {}, "section": "pfo-p2a-prereg-20261103"},
    "self": {"trunk": "self", "set": {}, "section": "pfo-p2a-prereg-20261103"},
    "rewire": {"trunk": "rewire", "set": {}, "section": "pfo-p2a-prereg-20261103"},
}


def run_name(arm, seed, phase="formal"):
    return f"pfo_{arm}_s{seed}" + ("_preflight" if phase == "preflight" else "")


def run_dir(arm, seed, phase="formal"):
    return OUT / "runs" / run_name(arm, seed, phase)


def config_path(arm, seed, phase="formal"):
    return CONFIGS / f"{run_name(arm, seed, phase)}.yaml"


def _set(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def build_config(arm, seed, phase="formal"):
    require(arm in ARMS and seed in SEEDS and phase in ("preflight", "formal"), "unregistered arm / seed / phase")
    spec = ARMS[arm]
    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    cfg = copy.deepcopy(cfg)
    name = run_name(arm, seed, phase)
    cfg["SEED"] = int(seed)
    cfg["TRAIN"].update(LR_SCHEDULE="cosine", OUTPUT_DIR=str(run_dir(arm, seed, phase)), RUN_NAME=name)
    cfg["EVAL"]["OUTPUT_DIR"] = str(run_dir(arm, seed, phase) / "eval")
    if phase == "preflight":
        cfg["TRAIN"].update(MAX_STEPS=PREFLIGHT_STEPS, SAVE_EVERY_N_STEPS=PREFLIGHT_STEPS,
                            VAL_CHECK_INTERVAL=PREFLIGHT_STEPS, WARMUP_STEPS=10, LIMIT_VAL_BATCHES=2)
    cfg.pop("S45SM", None)
    cfg["PFO"] = {"ARM": arm, "SEED": int(seed), "PHASE": phase, "TRUNK": spec["trunk"], "ROUTER": "sr1",
                  "ROUTER_KIND": "sr", "ROUTER_CHECKPOINT": str(SR1), "ASSOC": "pfa", "ONHAND_TAU": 0.5,
                  "SECTION": spec["section"]}
    for k, v in spec["set"].items():
        _set(cfg, k, v)
    return cfg


def write_config(arm, seed, phase="formal"):
    p = config_path(arm, seed, phase)
    text = yaml.safe_dump(build_config(arm, seed, phase), sort_keys=False)
    if p.exists():
        require(p.read_text() == text, "config on disk differs from the registry: " + str(p))
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


__all__ = ["ARMS", "SEEDS", "build_config", "config_path", "run_dir", "run_name", "write_config"]
