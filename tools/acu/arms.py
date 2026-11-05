"""ACU stage-1 arm registry (docs/EXPERIMENT_RECORDS.md#acu-s1-prereg-20261104). Every arm is its base's training config
(B2 = PFO2 `fgs`, tools/pfo2/arms.py; B2-gt = B2UB `b2gt`, tools/b2ub/arms.py, both unchanged) with one change:

  a1   B2-gt + the CA interface (semkine/acu_ca.py): label identity in training; the interface's information ceiling
  a2   B2    + the CA interface: r_s2d identity in training (the deployed router), isolates the interface

plus names, output directories and the ACU identity block. Phases: preflight (40 steps) and formal (6000 steps).
"""
from __future__ import annotations

import copy

from .common import CONFIGS, OUT, SCREEN_SEED, SECTION, STUDY, require

ARMS = {
    "a1": {"base": "b2gt", "iface": "ca", "trunk": "gshare", "train_routes": "label"},
    "a2": {"base": "fgs", "iface": "ca", "trunk": "gshare", "train_routes": "router"},
}
PHASES = ("preflight", "formal")


def run_name(arm, seed, phase="formal"):
    return f"acu_{arm}_s{seed}" + ("" if phase == "formal" else f"_{phase}")


def run_dir(arm, seed, phase="formal"):
    return OUT / "runs" / run_name(arm, seed, phase)


def config_path(arm, seed, phase="formal"):
    return CONFIGS / f"{run_name(arm, seed, phase)}.yaml"


def base_config(base, seed, phase):
    if base == "fgs":
        from tools.pfo2 import arms as P2
        return P2.build_config("fgs", seed, phase)
    from tools.b2ub import arms as BA
    return BA.build_config("b2gt", seed, phase)


def build_config(arm, seed, phase="formal"):
    require(arm in ARMS and seed == SCREEN_SEED and phase in PHASES, "unregistered arm / seed / phase")
    s = ARMS[arm]
    cfg = copy.deepcopy(base_config(s["base"], seed, phase))
    name = run_name(arm, seed, phase)
    cfg["TRAIN"].update(OUTPUT_DIR=str(run_dir(arm, seed, phase)), RUN_NAME=name)
    cfg["EVAL"]["OUTPUT_DIR"] = str(run_dir(arm, seed, phase) / "eval")
    require(cfg["PFO"]["TRUNK"] == s["trunk"] and cfg["PFO"]["MODULE"] == "trunk4", "base trunk changed")
    cfg["ACU"] = {"ARM": arm, "BASE_ARM": s["base"], "IFACE": s["iface"], "TRUNK": s["trunk"],
                  "TRAIN_ROUTES": s["train_routes"], "SECTION": SECTION, "STUDY": STUDY}
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


__all__ = ["ARMS", "PHASES", "base_config", "build_config", "config_path", "run_dir", "run_name", "write_config"]
