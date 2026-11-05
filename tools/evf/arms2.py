"""EVF arm registry, second model registry (semkine/evf_model2.py; docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104 and
the rounds that register these arms). tools/evf/arms.py's logic (frozen with round 1) over tools/evf/arms2_<round>.json.

Every arm file tools/evf/arms2_<round>.json ({arm: spec}) is written once, before any reading of its arms, and frozen with
its round; later rounds add files, never edit old ones. A spec:
  evf       the EVF block entries that differ from semkine/evf_model.py EVF_KEYS (the network), e.g. {"VF": false}
  track     optional TRACK overrides (arms on the training prior-state distribution only)
  base      the arm this one changes in one place (bookkeeping; "" for the starting points)
  section   the registration anchor
The config is B2's training config of the same seed (configs/pfo2_20261103/pfo2_fgs_s<seed>.yaml, read-only: data, nine
training subjects, depth augmentation, no hot pixels, 30-300 ms windows, prior-noise contract 0.5 / 0.3 / 0.2, the loss,
Adam 0.004, 500 warm-up steps then cosine over 6000 steps, 2 x 512, BF16) with the MODEL block reduced to the keys the
EVF model and the dataset read, the PFO block removed and the EVF block added. MODEL.XAG_ASSOC / XAG_GNN stay:
semkine/train.py seeds the DataLoader generator for configs that carry them, which is how B2 was trained, so an EVF arm
sees B2's data stream at the same seed (checked at acceptance from the first batches' hashes).
Phases, one recipe (the schedule is always the 6000-step one; `STOP` steps are run):
  smoke3 / smoke3b   3 steps, a checkpoint every step (the two runs must be bitwise identical)
  smoke50            50 steps (GPU smoke)
  screen             stopped at step 2000: under the deterministic policy its checkpoints are the formal run's
  formal             6000 steps
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .common import B2_CONFIG, CONFIGS, OUT, SEEDS, STUDY, require

HERE = Path(__file__).resolve().parent
PHASES = {"smoke3": (3, 1), "smoke3b": (3, 1), "smoke50": (50, 50), "screen": (2000, 500), "formal": (6000, 500)}
MODEL_KEEP = ("MANO_NCOMPS", "OUTPUT_DIM", "POSE_REPR", "PREDICT_BETA", "PREDICT_DELTA", "RENDER_H", "RENDER_W",
              "RENDER_SCALE", "RENDER_CHUNK", "XAG_ASSOC", "XAG_GNN")
SPEC_KEYS = {"evf", "track", "base", "section", "note"}


def _load():
    from semkine.evf_model2 import EVF2_KEYS as EVF_KEYS
    arms = {}
    for f in sorted(HERE.glob("arms2_*.json")):
        for name, s in json.loads(f.read_text()).items():
            require(name not in arms, f"arm {name} registered twice ({f.name})")
            require(set(s) <= SPEC_KEYS and "section" in s, f"bad EVF arm {name}")
            require(set(s.get("evf", {})) <= set(EVF_KEYS) - {"ARM", "SEED", "PHASE", "STUDY", "SECTION"},
                    f"bad EVF keys in {name}")
            arms[name] = {**s, "file": f.name}
    return arms


ARMS = _load()


def spec(arm):
    require(arm in ARMS, "unregistered EVF arm " + arm)
    return ARMS[arm]


def run_name(arm, seed, phase="formal"):
    return f"evf_{arm}_s{seed}" + ("" if phase == "formal" else f"_{phase}")


def run_dir(arm, seed, phase="formal"):
    return OUT / "runs" / run_name(arm, seed, phase)


def config_path(arm, seed, phase="formal"):
    return CONFIGS / f"{run_name(arm, seed, phase)}.yaml"


def stop_step(phase):
    return PHASES[phase][0]


def build_config(arm, seed, phase="formal"):
    import yaml
    require(seed in SEEDS and phase in PHASES, "unregistered seed / phase")
    s = spec(arm)
    b2 = yaml.safe_load(Path(str(B2_CONFIG).format(seed=seed)).read_text())
    require(b2["PFO"]["ARM"] == "fgs" and b2["SEED"] == seed, "not B2's config")
    cfg = copy.deepcopy(b2)
    cfg.pop("PFO")
    cfg["MODEL"] = {k: b2["MODEL"][k] for k in MODEL_KEEP}
    name = run_name(arm, seed, phase)
    cfg["TRAIN"].update(OUTPUT_DIR=str(run_dir(arm, seed, phase)), RUN_NAME=name, SAVE_EVERY_N_STEPS=PHASES[phase][1])
    cfg["EVAL"]["OUTPUT_DIR"] = str(run_dir(arm, seed, phase) / "eval")
    if s.get("track"):
        require(set(s["track"]) <= set(cfg["TRACK"]), "TRACK overrides must name existing keys")
        cfg["TRACK"].update(s["track"])
    cfg["EVF"] = {**s.get("evf", {}), "ARM": arm, "SEED": seed, "PHASE": phase, "STUDY": STUDY,
                  "SECTION": s["section"]}
    return cfg


def write_config(arm, seed, phase="formal"):
    import yaml
    from tools.pfo2.specs import write_once_text
    return write_once_text(config_path(arm, seed, phase), yaml.safe_dump(build_config(arm, seed, phase),
                                                                          sort_keys=False))


__all__ = ["ARMS", "MODEL_KEEP", "PHASES", "build_config", "config_path", "run_dir", "run_name", "spec", "stop_step",
           "write_config"]
