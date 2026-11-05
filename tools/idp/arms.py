"""IDP arm registry (docs/EXPERIMENT_RECORDS.md#idp-prereg-20261105): three single-seed arms on B2's recipe (PFO2 `fgs`,
tools/pfo2/arms.py `build_config`, unchanged), each differing from B2 only in the registered keys.

  l1  training and test routes `label_part`: the deployed router r_s2d's on-hand set and template point, the part
      replaced by the XAG label's part (the nearest face of that part), i.e. perfect identity with the router's real
      within-part point error (tools/pfo/nets.py `mixed_sources`, B2UB's U1 definition); no new feature
  l2  l1 + the record feature of semkine/idp_feat.py (the finger's foreshortening and flexion at the prior pose,
      zero-initialised map into the relation encoder)
  l3  B2's routes (r_s2d + PFA, training and test) + the same record feature: the deployable form

The only changes against B2 at the same seed and phase: DATA.XAG_SOURCE_ROOT (l1 / l2: the batches carry the XAG
pseudo-sources the mix needs), PFO.TRAIN_ROUTES, PFO.FEAT, and the names / output directories / identity keys. The
router block stays B2's (r_s2d), so the evaluation config (tools/pfo2/run.py `_eval_cfg`) names B2's router; the
`label_part` arms are read with the engine route `label_part` (diagnostic, not deployable).
"""
from __future__ import annotations

import copy

from tools.pfo2 import arms as P2

from .common import CONFIGS, OUT, SECTION, SEED, STUDY, XAG_SOURCE_ROOT, require

ARMS = {"l1": {"base": "fgs", "train_routes": "label_part", "feat": "none"},
        "l2": {"base": "fgs", "train_routes": "label_part", "feat": "fore"},
        "l3": {"base": "fgs", "train_routes": "router", "feat": "fore"}}
PHASES = ("preflight", "formal")
TRAIN_ROUTES = ("router", "label_part")
FEATS = ("none", "fore")


def eval_route(arm):
    """The engine route the arm is read with (None = the deployed router)."""
    return "label_part" if ARMS[arm]["train_routes"] == "label_part" else None


def run_name(arm, seed=SEED, phase="formal"):
    return f"idp_{arm}_s{seed}" + ("" if phase == "formal" else f"_{phase}")


def run_dir(arm, seed=SEED, phase="formal"):
    return OUT / "runs" / run_name(arm, seed, phase)


def config_path(arm, seed=SEED, phase="formal"):
    return CONFIGS / f"{run_name(arm, seed, phase)}.yaml"


def build_config(arm, seed=SEED, phase="formal"):
    require(arm in ARMS and seed == SEED and phase in PHASES, "unregistered arm / seed / phase")
    s = ARMS[arm]
    cfg = copy.deepcopy(P2.build_config(s["base"], seed, phase))
    name = run_name(arm, seed, phase)
    cfg["TRAIN"].update(OUTPUT_DIR=str(run_dir(arm, seed, phase)), RUN_NAME=name)
    cfg["EVAL"]["OUTPUT_DIR"] = str(run_dir(arm, seed, phase) / "eval")
    if s["train_routes"] == "label_part":
        cfg["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCE_ROOT)
    cfg["PFO"].update(ARM=arm, BASE_ARM=s["base"], TRAIN_ROUTES=s["train_routes"], FEAT=s["feat"], SECTION=SECTION,
                      STUDY=STUDY)
    return cfg


def write_config(arm, seed=SEED, phase="formal"):
    import yaml
    p = config_path(arm, seed, phase)
    text = yaml.safe_dump(build_config(arm, seed, phase), sort_keys=False)
    if p.exists():
        require(p.read_text() == text, "config on disk differs from the registry: " + str(p))
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


__all__ = ["ARMS", "FEATS", "PHASES", "TRAIN_ROUTES", "build_config", "config_path", "eval_route", "run_dir", "run_name",
           "write_config"]
