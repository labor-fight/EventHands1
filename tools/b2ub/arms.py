"""B2UB U3 arm registry (docs/EXPERIMENT_RECORDS.md#b2ub-prereg-20261104): B2-gt = B2's training config (PFO2 `fgs`,
tools/pfo2/arms.py `build_config`, unchanged) with the label routes in training.

The only two changes against B2 at the same seed and phase:
  DATA.XAG_SOURCE_ROOT  the batches carry the XAG per-event pseudo-sources (S45-SM pfagtdz's source root)
  PFO routing block     ROUTER "label" (training reads the batch's pseudo-sources through the native
                        `semkine.xag.sample_sources`; the association hooks stay B2's PFA + empty ray association,
                        tools/b2ub/train.py); the deployed / evaluation router is kept as EVAL_ROUTER* (r_s2d), so the
                        evaluation config (tools/pfo2/run.py `_eval_cfg`) names B2's router
plus names, output directories and the B2UB identity keys. Phases: preflight (PFO's 40-step preflight) and formal.
"""
from __future__ import annotations

import copy

from tools.pfo2 import arms as P2

from .common import CONFIGS, OUT, SECTION, STUDY, U3_SEED, XAG_SOURCE_ROOT, require

ARMS = {"b2gt": {"base": "fgs", "train_routes": "label", "section": SECTION}}
PHASES = ("preflight", "formal")


def run_name(arm, seed, phase="formal"):
    return f"b2ub_{arm}_s{seed}" + ("" if phase == "formal" else f"_{phase}")


def run_dir(arm, seed, phase="formal"):
    return OUT / "runs" / run_name(arm, seed, phase)


def config_path(arm, seed, phase="formal"):
    return CONFIGS / f"{run_name(arm, seed, phase)}.yaml"


def build_config(arm, seed, phase="formal"):
    require(arm in ARMS and seed == U3_SEED and phase in PHASES, "unregistered arm / seed / phase")
    s = ARMS[arm]
    cfg = copy.deepcopy(P2.build_config(s["base"], seed, phase))
    name = run_name(arm, seed, phase)
    cfg["TRAIN"].update(OUTPUT_DIR=str(run_dir(arm, seed, phase)), RUN_NAME=name)
    cfg["EVAL"]["OUTPUT_DIR"] = str(run_dir(arm, seed, phase) / "eval")
    cfg["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCE_ROOT)
    p = cfg["PFO"]
    p.update(EVAL_ROUTER=p["ROUTER"], EVAL_ROUTER_KIND=p["ROUTER_KIND"], EVAL_ROUTER_CHECKPOINT=p["ROUTER_CHECKPOINT"],
             ROUTER="label", ROUTER_KIND="label", ROUTER_CHECKPOINT=None)
    p.update(ARM=arm, BASE_ARM=s["base"], TRAIN_ROUTES=s["train_routes"], SECTION=s["section"], STUDY=STUDY)
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


__all__ = ["ARMS", "PHASES", "build_config", "config_path", "run_dir", "run_name", "write_config"]
