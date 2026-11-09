"""PMR stage S3b rounds rungs after engineering amendment 3 (docs/EXPERIMENT_RECORDS.md#pmr-amend3-20261109): the
interface of tools/pmr/arms2.py for two rungs that replace d1l1 / d1l2 (their configs changed MODEL.ENCODER_LAYERS, which
semkine/xag.py refuses with the PFA routes on):

  d1l1b  rounds 1   R built as registered, first round kept    (semkine/pmr_trunk3.py "r1")
  d1l2b  rounds 2   R built as registered, first two kept      (semkine/pmr_trunk3.py "r2")
Each config is D1's R config of the same seed and phase with only the PMR block and names / directories / identity keys
changed (MODEL.ENCODER_LAYERS stays 3).
"""
from __future__ import annotations

import copy

from tools.pmr import arms as A

from .common import CONFIGS, OUT, SEEDS, require

SECTION = "pmr-amend3-20261109"
LADDER = {"l1b": ("semkine.pmr_trunk3", "r1"), "l2b": ("semkine.pmr_trunk3", "r2")}
DESIGNS = {d: (m, SECTION) for d, (m, _) in LADDER.items()}
PHASES = ("preflight", "formal")
BASE = "d1r"


def arms():
    return {f"d1{d}": (d, v) for d, (_, v) in LADDER.items()}


def parse(arm):
    a = arms()
    require(arm in a, "unregistered PMR rounds rung " + arm)
    return a[arm]


def run_name(arm, seed, phase="formal"):
    return f"pmr_{arm}_s{seed}" + ("" if phase == "formal" else f"_{phase}")


def run_dir(arm, seed, phase="formal"):
    return OUT / "runs" / run_name(arm, seed, phase)


def config_path(arm, seed, phase="formal"):
    return CONFIGS / f"{run_name(arm, seed, phase)}.yaml"


def build_config(arm, seed, phase="formal"):
    design, variant = parse(arm)
    require(seed in SEEDS and phase in PHASES, "unregistered seed / phase")
    module, _ = LADDER[design]
    cfg = copy.deepcopy(A.build_config(BASE, seed, phase))
    name = run_name(arm, seed, phase)
    cfg["TRAIN"].update(OUTPUT_DIR=str(run_dir(arm, seed, phase)), RUN_NAME=name)
    cfg["EVAL"]["OUTPUT_DIR"] = str(run_dir(arm, seed, phase) / "eval")
    cfg["PFO"].update(ARM=arm, SECTION=SECTION)
    cfg["PMR"] = {"DESIGN": design, "VARIANT": variant, "MODULE": module, "BASE_ARM": BASE}
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


install_trunk = A.install_trunk
VARIANTS = A.VARIANTS

__all__ = ["BASE", "DESIGNS", "LADDER", "PHASES", "SECTION", "VARIANTS", "arms", "build_config", "config_path",
           "install_trunk", "parse", "run_dir", "run_name", "write_config"]
