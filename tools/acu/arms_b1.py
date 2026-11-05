"""ACU stage-1b arm registry (docs/EXPERIMENT_RECORDS.md#acu-s1b-prereg-20261104): b1 = B2's training config (PFO2 `fgs`,
tools/pfo2/arms.py, unchanged) with the B1 consistency training of semkine/acu_b1.py (prior time inside the window, the
trunk's token times measured from the prior, the zero-initialised age term); names, directories and the ACU block.
"""
from __future__ import annotations

import copy

from .common import CONFIGS, OUT, SCREEN_SEED, STUDY, require

SECTION_B1 = "acu-s1b-prereg-20261104"
ARMS = {"b1": {"base": "fgs", "iface": "pfa", "update": "b1", "trunk": "gshare", "train_routes": "router"}}
PHASES = ("preflight", "formal")


def run_name(arm, seed, phase="formal"):
    return f"acu_{arm}_s{seed}" + ("" if phase == "formal" else f"_{phase}")


def run_dir(arm, seed, phase="formal"):
    return OUT / "runs" / run_name(arm, seed, phase)


def config_path(arm, seed, phase="formal"):
    return CONFIGS / f"{run_name(arm, seed, phase)}.yaml"


def build_config(arm, seed, phase="formal"):
    from tools.pfo2 import arms as P2
    require(arm in ARMS and seed == SCREEN_SEED and phase in PHASES, "unregistered arm / seed / phase")
    s = ARMS[arm]
    cfg = copy.deepcopy(P2.build_config(s["base"], seed, phase))
    name = run_name(arm, seed, phase)
    cfg["TRAIN"].update(OUTPUT_DIR=str(run_dir(arm, seed, phase)), RUN_NAME=name)
    cfg["EVAL"]["OUTPUT_DIR"] = str(run_dir(arm, seed, phase) / "eval")
    require(cfg["PFO"]["TRUNK"] == s["trunk"] and cfg["PFO"]["MODULE"] == "trunk4", "base trunk changed")
    cfg["ACU"] = {"ARM": arm, "BASE_ARM": s["base"], "IFACE": s["iface"], "UPDATE": s["update"], "TRUNK": s["trunk"],
                  "TRAIN_ROUTES": s["train_routes"], "SECTION": SECTION_B1, "STUDY": STUDY}
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


__all__ = ["ARMS", "PHASES", "SECTION_B1", "build_config", "config_path", "run_dir", "run_name", "write_config"]
