"""PMR stage S3 rollout fine-tunes (docs/EXPERIMENT_RECORDS.md#pmr-prereg-20261109, "过筛之后"): TPD's rol5 recipe
(semkine/tpd_rol.py, tools/tpd/arms_rol.py `rol5`) applied to an accepted PMR arm from its own step-6000 checkpoint.

  <arm>_rol5   K = 5 consecutive 50 ms packets, gradient through the prior, per-packet checkpointing, equal weights;
               formal 1500 steps, peak LR 1e-3, 100 warm-up steps, cosine to 0, checkpoints every 500 (preflight 40 / 10 / 40)
Every config is the PMR arm's config of the same seed and phase (tools/pmr/arms.py `build_config`, unchanged: B2's recipe
with the PMR trunk) with only TRAIN.MAX_STEPS / LR / WARMUP_STEPS / SAVE_EVERY_N_STEPS / VAL_CHECK_INTERVAL,
MODEL.INIT_FROM (the arm's accepted step-6000 checkpoint of the seed), the ROL block, names / output directories and
identity keys, i.e. exactly the keys TPD's rol5 changed on B2's config, plus TRAIN.PREFETCH_FACTOR 1 (TPD amendment 3:
K = 5 rollout batches exhaust /dev/shm with prefetch 2; a loader setting, the sample stream is unchanged). The router block
stays B2's, so the evaluation config reads the fine-tune with the deployed router r_s2d (the ordinary packet form).
"""
from __future__ import annotations

import copy

from tools.pmr import arms as A

from .common import CONFIGS, OUT, SEEDS, read, require

ROL = {"rol5": {"k": 5, "grad": True}}
PHASES = ("preflight", "formal")
FORMAL = {"MAX_STEPS": 1500, "LR": 0.001, "WARMUP_STEPS": 100, "SAVE_EVERY_N_STEPS": 500, "VAL_CHECK_INTERVAL": 500}
PREFLIGHT = {"MAX_STEPS": 40, "LR": 0.001, "WARMUP_STEPS": 10, "SAVE_EVERY_N_STEPS": 40, "VAL_CHECK_INTERVAL": 40}
#: engineering smoke of the entry (2-GPU DDP, deterministic; never read): the preflight shortened to 10 steps
SMOKE = "smoke"
SMOKE_TRAIN = {"MAX_STEPS": 10, "LR": 0.001, "WARMUP_STEPS": 5, "SAVE_EVERY_N_STEPS": 10, "VAL_CHECK_INTERVAL": 10}
PACKET_MS = 50


def parse(name):
    base, _, rol = name.rpartition("_")
    require(rol in ROL, "unregistered rollout arm " + name)
    A.parse(base)
    return base, rol


def run_name(name, seed, phase="formal"):
    return f"pmr_{name}_s{seed}" + ("" if phase == "formal" else f"_{phase}")


def run_dir(name, seed, phase="formal"):
    return OUT / "runs" / run_name(name, seed, phase)


def config_path(name, seed, phase="formal"):
    return CONFIGS / f"{run_name(name, seed, phase)}.yaml"


def init_checkpoint(base, seed):
    """The base arm's step-6000 checkpoint path (written into the config before the base run exists)."""
    return str(A.run_dir(base, seed) / f"{A.run_name(base, seed)}-step=6000.ckpt")


def init_sha(base, seed):
    """The SHA-256 the base arm's acceptance recorded for its step-6000 checkpoint (the warm start must be that file)."""
    receipt = read(OUT / "receipts" / f"{A.run_name(base, seed)}.json")
    require(receipt["status"] == "PASS", "base arm not accepted: " + base)
    return receipt["checkpoint_sha256_by_step"]["6000"]


def build_config(name, seed, phase="formal"):
    base, rol = parse(name)
    require(seed in SEEDS and phase in PHASES + (SMOKE,), "unregistered seed / phase")
    s = ROL[rol]
    cfg = copy.deepcopy(A.build_config(base, seed, "preflight" if phase == SMOKE else phase))
    rn = run_name(name, seed, phase)
    sched = FORMAL if phase == "formal" else PREFLIGHT if phase == "preflight" else SMOKE_TRAIN
    cfg["TRAIN"].update(**sched, OUTPUT_DIR=str(run_dir(name, seed, phase)), RUN_NAME=rn, PREFETCH_FACTOR=1)
    require(cfg["TRAIN"]["LR_SCHEDULE"] == "cosine", "B2's schedule is cosine")
    cfg["EVAL"]["OUTPUT_DIR"] = str(run_dir(name, seed, phase) / "eval")
    cfg["MODEL"]["INIT_FROM"] = init_checkpoint(base, seed)
    cfg["ROL"] = {"K": s["k"], "GRAD": s["grad"], "CHECKPOINT": s["k"] > 1, "WEIGHTS": [1.0] * s["k"],
                  "PACKET_MS": PACKET_MS}
    cfg["PFO"].update(ARM=name, BASE_ARM=base)
    if phase == SMOKE:
        cfg["PFO"]["PHASE"] = SMOKE
    return cfg


def write_config(name, seed, phase="formal"):
    import yaml
    p = config_path(name, seed, phase)
    text = yaml.safe_dump(build_config(name, seed, phase), sort_keys=False)
    if p.exists():
        require(p.read_text() == text, "config on disk differs from the registry: " + str(p))
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


__all__ = ["FORMAL", "PACKET_MS", "PHASES", "PREFLIGHT", "ROL", "build_config", "config_path", "init_checkpoint", "init_sha",
           "parse",
           "run_dir", "run_name", "write_config"]
