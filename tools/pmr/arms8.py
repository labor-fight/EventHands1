"""PMR stage S7 registry (docs/EXPERIMENT_RECORDS.md#pmr-s7-prereg-20261109): ghand combined with a hand-node refinement, with
tools/pmr/arms.py's interface so the frozen entries run them through thin wrappers (tools/pmr/train8.py, run8.py).

  d1ghandhp    ghand + hp (semkine/pmr_trunk8.py "ghandhp")
  d1ghandhub   ghand + hub (semkine/pmr_trunk8.py "ghandhub")
Their node inputs and readout are D1's, so D1's R-noprop / R-mlp are their controls.
Every config is D1's R config of the same seed and phase with only the PMR block and names / directories / identity keys
changed (width and rounds stay D1's: semkine/xag.py pins them).
"""
from __future__ import annotations

import copy

from tools.pmr import arms as A

from .common import CONFIGS, OUT, SEEDS, require

SECTION = "pmr-s7-prereg-20261109"
REFINE = {"ghandhp": "ghandhp", "ghandhub": "ghandhub"}
MODULE = "semkine.pmr_trunk8"
DESIGNS = {d: (MODULE, SECTION) for d in REFINE}
PHASES = ("preflight", "formal")
BASE = "d1r"


def arms():
    return {f"d1{d}": (d, v) for d, v in REFINE.items()}


def parse(arm):
    a = arms()
    require(arm in a, "unregistered PMR S7 arm " + arm)
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
    cfg = copy.deepcopy(A.build_config(BASE, seed, phase))
    name = run_name(arm, seed, phase)
    cfg["TRAIN"].update(OUTPUT_DIR=str(run_dir(arm, seed, phase)), RUN_NAME=name)
    cfg["EVAL"]["OUTPUT_DIR"] = str(run_dir(arm, seed, phase) / "eval")
    cfg["PFO"].update(ARM=arm, SECTION=SECTION)
    cfg["PMR"] = {"DESIGN": design, "VARIANT": variant, "MODULE": MODULE, "BASE_ARM": BASE}
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

__all__ = ["BASE", "DESIGNS", "MODULE", "PHASES", "REFINE", "SECTION", "VARIANTS", "arms", "build_config", "config_path",
           "install_trunk", "parse", "run_dir", "run_name", "write_config"]
