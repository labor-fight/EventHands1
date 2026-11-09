"""PMR arm registry (docs/EXPERIMENT_RECORDS.md#pmr-prereg-20261109 and its design sections).

A design is a trunk module (semkine/pmr_trunk*.py) with the three screening variants r (P), noprop (messages removed),
mlp (concat + MLP readout over the same node inputs); an arm = design x variant. Every arm is B2's recipe (PFO2 `fgs`,
tools/pfo2/arms.py `build_config`, unchanged: r_s2d + PFA, so3_trans_fk 1 / 1 / 2, 2 x 512, Adam 4e-3, warm-up 500, cosine,
6000 steps, bf16, B2's data / augmentation / prior-noise contract) with only the trunk replaced; the differences against
B2's config at the same seed and phase are PFO.TRUNK ("pmr"), the PMR block (design, variant, module) and the names /
output directories / identity keys. Designs are added by appending to DESIGNS (never editing an earlier entry) before
their registration is frozen.
"""
from __future__ import annotations

import copy

from tools.pfo2 import arms as P2

from .common import CONFIGS, OUT, SEEDS, STUDY, require

#: design -> (trunk module, registration anchor)
DESIGNS = {"d1": ("semkine.pmr_trunk", "pmr-d1-prereg-20261109")}
VARIANTS = {"r": "r", "n": "noprop", "m": "mlp"}
PHASES = ("preflight", "formal")
#: engineering smoke of a new entry (2-GPU DDP, deterministic, 20 steps; never read): B2's preflight config shortened
SMOKE = "smoke"


def arms():
    return {f"{d}{c}": (d, v) for d in DESIGNS for c, v in VARIANTS.items()}


def parse(arm):
    a = arms()
    require(arm in a, "unregistered PMR arm " + arm)
    return a[arm]


def run_name(arm, seed, phase="formal"):
    return f"pmr_{arm}_s{seed}" + ("" if phase == "formal" else f"_{phase}")


def run_dir(arm, seed, phase="formal"):
    return OUT / "runs" / run_name(arm, seed, phase)


def config_path(arm, seed, phase="formal"):
    return CONFIGS / f"{run_name(arm, seed, phase)}.yaml"


def build_config(arm, seed, phase="formal"):
    design, variant = parse(arm)
    require(seed in SEEDS and phase in PHASES + (SMOKE,), "unregistered seed / phase")
    module, section = DESIGNS[design]
    cfg = copy.deepcopy(P2.build_config("fgs", seed, "preflight" if phase == SMOKE else phase))
    if phase == SMOKE:
        cfg["TRAIN"].update(MAX_STEPS=20, SAVE_EVERY_N_STEPS=20, VAL_CHECK_INTERVAL=20)
        cfg["PFO"]["PHASE"] = SMOKE
    name = run_name(arm, seed, phase)
    cfg["TRAIN"].update(OUTPUT_DIR=str(run_dir(arm, seed, phase)), RUN_NAME=name)
    cfg["EVAL"]["OUTPUT_DIR"] = str(run_dir(arm, seed, phase) / "eval")
    cfg["PFO"].update(ARM=arm, TRUNK="pmr", SECTION=section, STUDY=STUDY)
    cfg["PMR"] = {"DESIGN": design, "VARIANT": variant, "MODULE": module}
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


def install_trunk(cfg):
    """Install the arm's trunk (one process = one network); returns the identity record."""
    import importlib
    m = cfg["PMR"]
    mod = importlib.import_module(m["MODULE"])
    variant = mod.install(m["VARIANT"])
    return {"design": m["DESIGN"], "variant": variant, "module": m["MODULE"]}


__all__ = ["DESIGNS", "PHASES", "SMOKE", "VARIANTS", "arms", "build_config", "config_path", "install_trunk", "parse", "run_dir",
           "run_name", "write_config"]
