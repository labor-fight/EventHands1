"""ACU stage-2 registry (docs/EXPERIMENT_RECORDS.md#acu-s2-prereg-20261104): the asynchronous 2x2. Every cell is B2's training
config (PFO2 `fgs`, tools/pfo2/arms.py, unchanged: data, augmentation, loss, optimiser, schedule, 6000 steps, gshare
trunk) run on the causal per-event pipeline (semkine/acu_causal.py: 50 ms history, causal-stride nodes, history-aware
tokens, PR identities from tools/acu/router_pr.py) with

  observation  pfa  PR's part + rest-template point placed exactly (B2's PFA records)
               ca   PR's finger group + the previous state's contour, normal residual (semkine/acu_ca.py)
  update       orig B2's readout
               b1   B1 consistency training (prior time inside the window + age term, semkine/acu_b1.py)

Cells: c00 = pfa/orig, c10 = ca/orig, c01 = pfa/b1, c11 = ca/b1; seeds 3407 (screen), more only by the written rules.
"""
from __future__ import annotations

import copy

from .common import CONFIGS, OUT, STUDY, require

SECTION_S2 = "acu-s2-prereg-20261104"
CELLS = {"c00": ("pfa", "orig"), "c10": ("ca", "orig"), "c01": ("pfa", "b1"), "c11": ("ca", "b1")}
TRUNKS = ("gshare", "nomp")
SEEDS = (3407, 3408, 3409)
PHASES = ("preflight", "formal")
PR_CHECKPOINT = OUT / "routers" / "pr" / "pr_final.ckpt"


def arm_name(cell, trunk="gshare"):
    return cell if trunk == "gshare" else f"{cell}{trunk}"


def run_name(cell, seed, phase="formal", trunk="gshare"):
    return f"acu_s2_{arm_name(cell, trunk)}_s{seed}" + ("" if phase == "formal" else f"_{phase}")


def run_dir(cell, seed, phase="formal", trunk="gshare"):
    return OUT / "runs" / run_name(cell, seed, phase, trunk)


def config_path(cell, seed, phase="formal", trunk="gshare"):
    return CONFIGS / f"{run_name(cell, seed, phase, trunk)}.yaml"


def build_config(cell, seed, phase="formal", trunk="gshare"):
    from tools.pfo2 import arms as P2
    require(cell in CELLS and seed in SEEDS and phase in PHASES and trunk in TRUNKS, "unregistered cell / seed / phase")
    obs, upd = CELLS[cell]
    cfg = copy.deepcopy(P2.build_config("fgs", seed, phase))
    name = run_name(cell, seed, phase, trunk)
    cfg["TRAIN"].update(OUTPUT_DIR=str(run_dir(cell, seed, phase, trunk)), RUN_NAME=name)
    cfg["EVAL"]["OUTPUT_DIR"] = str(run_dir(cell, seed, phase, trunk) / "eval")
    require(cfg["PFO"]["TRUNK"] == "gshare" and cfg["PFO"]["MODULE"] == "trunk4", "base trunk changed")
    cfg["PFO"]["TRUNK"] = trunk
    cfg["ACU"] = {"ARM": arm_name(cell, trunk), "CELL": cell, "IFACE": obs, "UPDATE": upd, "TRUNK": trunk,
                  "PIPELINE": "causal", "IDENTITY": "pr", "PR_CHECKPOINT": str(PR_CHECKPOINT), "ONHAND_TAU": 0.5,
                  "SECTION": SECTION_S2, "STUDY": STUDY}
    return cfg


def write_config(cell, seed, phase="formal", trunk="gshare"):
    import yaml
    p = config_path(cell, seed, phase, trunk)
    text = yaml.safe_dump(build_config(cell, seed, phase, trunk), sort_keys=False)
    if p.exists():
        require(p.read_text() == text, "config on disk differs from the registry: " + str(p))
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


__all__ = ["CELLS", "PHASES", "PR_CHECKPOINT", "SECTION_S2", "SEEDS", "TRUNKS", "arm_name", "build_config", "config_path",
           "run_dir", "run_name", "write_config"]
