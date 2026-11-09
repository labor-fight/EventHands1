"""TPD synthesis / final-configuration arm registry (docs/EXPERIMENT_RECORDS.md#tpd-syn-prereg-20261105): B2's recipe from
scratch (PFO2 `fgs` config of the seed: 2 x 512, BF16, Adam 4e-3, 500-step warm-up, cosine to 0.02, 6000 steps; 40-step
preflight as PFO2's) with the registered FINAL block and nothing else changed except names / directories / identity keys.

An arm spec:
  routes   label_part (IDP L1's training routes: router on-hand set and template point, the XAG label's part; the batches
           carry the XAG pseudo-sources) | router (B2's: the deployed router r_s2d)
  causal   true: trained in the causal deployment form (semkine/tpd_causal.py nodes and tokens, n_prev per packet)
  k, grad  rollout length (1 = B2's variable-window samples and single-packet step) and gradient through the prior
  rol      the ROL arm whose training method the arm carries (tools/tpd/arms_rol.py)
The rollout arms use 12 data workers per rank (B2: 16) to fit two XAG-reading trainings in host memory.
"""
from __future__ import annotations

import copy

from tools.pfo2 import arms as P2

from .common import CONFIGS, OUT, STUDY, require
from tools.pfo.common import XAG_SOURCE_ROOT

SECTION = "tpd-syn-prereg-20261105"
SEEDS = (3407, 3408, 3409)
PHASES = ("preflight", "formal")
ARMS = {
    "c0": {"routes": "label_part", "causal": True, "k": 1, "grad": False},
    "c3d": {"routes": "label_part", "causal": True, "k": 3, "grad": False, "rol": "rol3d"},
    "c3": {"routes": "label_part", "causal": True, "k": 3, "grad": True, "rol": "rol3"},
    "c5": {"routes": "label_part", "causal": True, "k": 5, "grad": True, "rol": "rol5"},
}
#: data workers of the rollout arms (B2's 16 for c0): every sample's augmentation is a function of (seed, index), so the
#: worker count changes the host memory, not the data
ROLLOUT_WORKERS = 12
#: the synthesis arm that carries ROL arm X's training method
OF_ROL = {s["rol"]: a for a, s in ARMS.items() if "rol" in s}


def run_name(arm, seed, phase="formal"):
    return f"tpd_{arm}_s{seed}" + ("" if phase == "formal" else f"_{phase}")


def run_dir(arm, seed, phase="formal"):
    return OUT / "runs" / run_name(arm, seed, phase)


def config_path(arm, seed, phase="formal"):
    return CONFIGS / f"{run_name(arm, seed, phase)}.yaml"


def build_config(arm, seed, phase="formal"):
    require(arm in ARMS and seed in SEEDS and phase in PHASES, "unregistered arm / seed / phase")
    s = ARMS[arm]
    cfg = copy.deepcopy(P2.build_config("fgs", seed, phase))
    name = run_name(arm, seed, phase)
    cfg["TRAIN"].update(OUTPUT_DIR=str(run_dir(arm, seed, phase)), RUN_NAME=name)
    cfg["EVAL"]["OUTPUT_DIR"] = str(run_dir(arm, seed, phase) / "eval")
    if s["routes"] == "label_part":
        cfg["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCE_ROOT)
    k = int(s["k"])
    if k > 1:
        cfg["TRAIN"]["NUM_WORKERS"] = ROLLOUT_WORKERS              # data-loading resource only (per-sample streams)
    cfg["ROL"] = {"K": k, "GRAD": bool(s["grad"]), "CHECKPOINT": k > 1, "WEIGHTS": [1.0] * k, "PACKET_MS": 50}
    cfg["FINAL"] = {"CAUSAL": bool(s["causal"]), "ROUTES": s["routes"], "ROLLOUT": k > 1}
    cfg["PFO"].update(ARM=arm, BASE_ARM="fgs", SECTION=SECTION, STUDY=STUDY)
    return cfg


def write_config(arm, seed, phase="formal"):
    import yaml
    p = config_path(arm, seed, phase)
    text = yaml.safe_dump(build_config(arm, seed, phase), sort_keys=False)
    if p.exists():
        require(p.read_text() == text, "config on disk differs from the registry: " + str(p))
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


__all__ = ["ARMS", "OF_ROL", "PHASES", "SECTION", "SEEDS", "build_config", "config_path", "run_dir", "run_name",
           "write_config"]
