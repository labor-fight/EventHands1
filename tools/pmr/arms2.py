"""PMR stage S3b ladder registry (docs/EXPERIMENT_RECORDS.md#pmr-s3b-prereg-20261109): one-change variants of D1's R,
from scratch on B2's recipe (seed 3407, 6000 steps, no ROL), with the same interface as tools/pmr/arms.py so the frozen
entries (tools/pmr/train.py, run.py) run them through thin wrappers (tools/pmr/train2.py, run2.py) that rebind their
registry to this module.

  d1l1  rounds 1      R with MODEL.ENCODER_LAYERS 1                                   (semkine/pmr_trunk.py "r")
  d1l2  rounds 2      R with MODEL.ENCODER_LAYERS 2                                   (semkine/pmr_trunk.py "r")
  d1g0  gate ablation the gates without geometry (learned vector per edge type)       (semkine/pmr_trunk2.py "rconst")
  d1h0  hub ablation  messages on the tree edges only                                  (semkine/pmr_trunk2.py "rnohub")
  d1a0  no mesh association  events on their rays at the wrist depth, no residual      (semkine/pmr_trunk2.py "rnomesh")
Every config is D1's R config of the same seed and phase (tools/pmr/arms.py `build_config("d1r", ...)`) with only the
PMR block (design / variant / module), MODEL.ENCODER_LAYERS for the rounds rungs, names / output directories / identity
keys changed.
"""
from __future__ import annotations

import copy

from tools.pmr import arms as A

from .common import CONFIGS, OUT, SEEDS, require

SECTION = "pmr-s3b-prereg-20261109"
#: ladder design -> (trunk module, variant, MODEL.ENCODER_LAYERS)
LADDER = {"l1": ("semkine.pmr_trunk", "r", 1), "l2": ("semkine.pmr_trunk", "r", 2),
          "g0": ("semkine.pmr_trunk2", "rconst", 3), "h0": ("semkine.pmr_trunk2", "rnohub", 3),
          "a0": ("semkine.pmr_trunk2", "rnomesh", 3)}
DESIGNS = {d: (m, SECTION) for d, (m, _, _) in LADDER.items()}
PHASES = ("preflight", "formal")
BASE = "d1r"


def arms():
    return {f"d1{d}": (d, v) for d, (_, v, _) in LADDER.items()}


def parse(arm):
    a = arms()
    require(arm in a, "unregistered PMR ladder arm " + arm)
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
    module, _, layers = LADDER[design]
    cfg = copy.deepcopy(A.build_config(BASE, seed, phase))
    name = run_name(arm, seed, phase)
    cfg["TRAIN"].update(OUTPUT_DIR=str(run_dir(arm, seed, phase)), RUN_NAME=name)
    cfg["EVAL"]["OUTPUT_DIR"] = str(run_dir(arm, seed, phase) / "eval")
    cfg["MODEL"]["ENCODER_LAYERS"] = layers
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
