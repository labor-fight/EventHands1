"""PMR stage S4 registry (docs/EXPERIMENT_RECORDS.md#pmr-s4-prereg-20261109): refinements of D1 for the global columns,
with tools/pmr/arms.py's interface so the frozen entries run them through thin wrappers (tools/pmr/train4.py, run4.py).

  d1dual / d1dualn / d1dualm   R / R-noprop / R-mlp with each part's moments also in the camera frame
                               (semkine/pmr_trunk4.py "dual" / "dualn" / "dualm")
  d1hp                         R with the palm <-> hand edges typed apart (semkine/pmr_trunk4.py "hp"); its controls are
                               D1's R-noprop / R-mlp (same node inputs)
Every config is D1's R config of the same seed and phase with only the PMR block and names / directories / identity keys
changed (width and rounds stay D1's: semkine/xag.py pins them).
"""
from __future__ import annotations

import copy

from tools.pmr import arms as A

from .common import CONFIGS, OUT, SEEDS, require

SECTION = "pmr-s4-prereg-20261109"
REFINE = {"dual": "dual", "dualn": "dualn", "dualm": "dualm", "hp": "hp"}
MODULE = "semkine.pmr_trunk4"
DESIGNS = {d: (MODULE, SECTION) for d in REFINE}
PHASES = ("preflight", "formal")
BASE = "d1r"


def arms():
    return {f"d1{d}": (d, v) for d, v in REFINE.items()}


def parse(arm):
    a = arms()
    require(arm in a, "unregistered PMR S4 arm " + arm)
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
