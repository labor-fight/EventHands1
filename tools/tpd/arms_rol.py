"""TPD stage ROL arm registry (docs/EXPERIMENT_RECORDS.md#tpd-rol-prereg-20261105): fine-tunes of B2 (PFO2 `fgs`) from its
own accepted step-6000 checkpoint of the same seed, on K consecutive 50 ms packets (semkine/tpd_rol.py).

  rol1   K = 1                                   the fine-tune itself (50 ms packets, schedule, float32 output): control
  rol3d  K = 3, prior detached between packets   exposure to the network's own errors only (DAgger-like)
  rol3   K = 3, gradient through the prior       the closed-loop objective
  rol5   K = 5, gradient through the prior       longer horizon (formal run only if its preflight step time allows)

Every config is PFO2's `fgs` config of the same seed and phase (tools/pfo2/arms.py `build_config`, unchanged) with only:
TRAIN.MAX_STEPS / LR / WARMUP_STEPS / SAVE_EVERY_N_STEPS (formal 1500 / 1e-3 / 100 / 500; preflight 40 / 1e-3 / 10 / 40),
MODEL.INIT_FROM (B2's accepted step-6000 checkpoint of the seed), the ROL block, names / output directories and PFO
identity keys. The router block stays B2's (r_s2d, PFA, tau 0.5), so the evaluation config (tools/pfo2/run.py `_eval_cfg`)
reads the arm with B2's deployed router.
"""
from __future__ import annotations

import copy

from tools.pfo2 import arms as P2

from .common import CONFIGS, OUT, SEEDS, STUDY, b2_spec, read, require

SECTION = "tpd-rol-prereg-20261105"
ARMS = {"rol1": {"k": 1, "grad": False}, "rol3d": {"k": 3, "grad": False}, "rol3": {"k": 3, "grad": True},
        "rol5": {"k": 5, "grad": True}}
PHASES = ("preflight", "formal")
FORMAL = {"MAX_STEPS": 1500, "LR": 0.001, "WARMUP_STEPS": 100, "SAVE_EVERY_N_STEPS": 500, "VAL_CHECK_INTERVAL": 500}
PREFLIGHT = {"MAX_STEPS": 40, "LR": 0.001, "WARMUP_STEPS": 10, "SAVE_EVERY_N_STEPS": 40, "VAL_CHECK_INTERVAL": 40}
PACKET_MS = 50


def run_name(arm, seed, phase="formal"):
    return f"tpd_{arm}_s{seed}" + ("" if phase == "formal" else f"_{phase}")


def run_dir(arm, seed, phase="formal"):
    return OUT / "runs" / run_name(arm, seed, phase)


def config_path(arm, seed, phase="formal"):
    return CONFIGS / f"{run_name(arm, seed, phase)}.yaml"


def build_config(arm, seed, phase="formal"):
    require(arm in ARMS and seed in SEEDS and phase in PHASES, "unregistered arm / seed / phase")
    s = ARMS[arm]
    cfg = copy.deepcopy(P2.build_config("fgs", seed, phase))
    name = run_name(arm, seed, phase)
    cfg["TRAIN"].update(**(FORMAL if phase == "formal" else PREFLIGHT), OUTPUT_DIR=str(run_dir(arm, seed, phase)),
                        RUN_NAME=name)
    require(cfg["TRAIN"]["LR_SCHEDULE"] == "cosine", "B2's schedule is cosine")
    cfg["EVAL"]["OUTPUT_DIR"] = str(run_dir(arm, seed, phase) / "eval")
    cfg["MODEL"]["INIT_FROM"] = read(b2_spec(seed))["checkpoint"]
    cfg["ROL"] = {"K": s["k"], "GRAD": s["grad"], "CHECKPOINT": s["k"] > 1, "WEIGHTS": [1.0] * s["k"],
                  "PACKET_MS": PACKET_MS}
    cfg["PFO"].update(ARM=arm, BASE_ARM="fgs", SECTION=SECTION, STUDY=STUDY)
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


__all__ = ["ARMS", "FORMAL", "PACKET_MS", "PHASES", "PREFLIGHT", "SECTION", "build_config", "config_path", "run_dir",
           "run_name", "write_config"]
