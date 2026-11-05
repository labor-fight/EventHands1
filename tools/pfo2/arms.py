"""PFO2 arm registry, data-driven (docs/EXPERIMENT_RECORDS.md#pfo2-prereg-20261103 and later PFO2 sections).

Every arm file tools/pfo2/arms_<stage>.json ({arm: spec}) is written once, before any reading of its arms, and frozen
with its section; later stages add files, never edit old ones. A spec:
  pfo_arm   the PFO arm whose recipe the arm starts from (tools/pfo/arms3.py, frozen), e.g. "f1" = F
  trunk     optional override of the PFO trunk variant string (semkine/pfo_trunk4.py syntax: "full", "l1", ...)
  module    the trunk module that builds it: "trunk4" (semkine/pfo_trunk4.py, default) or "trunk3" (semkine/pfo_trunk3.py,
            the single-change l1 / gshare module, used for the reference runs of the bitwise checks)
  section   the registration anchor; other keys are bookkeeping (replay_of, reference, reference_for)
The config is PFO's generated config of that recipe (tools/pfo/arms2.py `build_config`, unchanged) with PFO2's run name,
output directories and PFO block entries. Phases: preflight (PFO's 40-step preflight), formal (6000 steps), head500
(the formal config stopped at step 500 with the constant schedule: during the 500 warm-up steps both schedules give the
same factor, so its step-500 checkpoint is the formal run's step-500 checkpoint when the code is equivalent).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from tools.pfo import arms2 as A2
from tools.pfo import arms3 as A3

from .common import CONFIGS, OUT, SEEDS, STUDY, require

HERE = Path(__file__).resolve().parent
PHASES = ("preflight", "formal", "head500")
MODULES = ("trunk4", "trunk3")


def _load():
    arms = {}
    for f in sorted(HERE.glob("arms_*.json")):
        for name, s in json.loads(f.read_text()).items():
            require(name not in arms, f"arm {name} registered twice ({f.name})")
            require(s.get("module", "trunk4") in MODULES and s["pfo_arm"] in A3.ARMS3, f"bad PFO2 arm {name}")
            arms[name] = s
    return arms


ARMS = _load()


def spec(arm):
    require(arm in ARMS, "unregistered PFO2 arm " + arm)
    return ARMS[arm]


def recipe(arm):
    """The PFO arm spec (trunk / set / router / bank / train_router) of this arm."""
    s = spec(arm)
    r = copy.deepcopy(A3.spec(s["pfo_arm"]))
    if "trunk" in s:
        r["trunk"] = s["trunk"]
    return r


def run_name(arm, seed, phase="formal"):
    return f"pfo2_{arm}_s{seed}" + ("" if phase == "formal" else f"_{phase}")


def run_dir(arm, seed, phase="formal"):
    return OUT / "runs" / run_name(arm, seed, phase)


def config_path(arm, seed, phase="formal"):
    return CONFIGS / f"{run_name(arm, seed, phase)}.yaml"


def _pfo_config(r, seed, phase):
    tmp = "__pfo2__"
    saved = dict(A2.ARMS2)
    try:
        A2.ARMS2[tmp] = r
        return A2.build_config(tmp, seed, phase)
    finally:
        A2.ARMS2.clear()
        A2.ARMS2.update(saved)


def build_config(arm, seed, phase="formal"):
    require(seed in SEEDS and phase in PHASES, "unregistered seed / phase")
    s = spec(arm)
    cfg = _pfo_config(recipe(arm), seed, "formal" if phase == "head500" else phase)
    name = run_name(arm, seed, phase)
    cfg["TRAIN"].update(OUTPUT_DIR=str(run_dir(arm, seed, phase)), RUN_NAME=name)
    cfg["EVAL"]["OUTPUT_DIR"] = str(run_dir(arm, seed, phase) / "eval")
    if phase == "head500":
        cfg["TRAIN"].update(MAX_STEPS=500, LR_SCHEDULE="constant")
    cfg["PFO"].update(ARM=arm, PHASE=phase, SECTION=s["section"], STUDY=STUDY, MODULE=s.get("module", "trunk4"),
                      PFO_ARM=s["pfo_arm"])
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


__all__ = ["ARMS", "MODULES", "PHASES", "build_config", "config_path", "recipe", "run_dir", "run_name", "spec",
           "write_config"]
