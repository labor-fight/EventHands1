"""CMP arms (docs/EXPERIMENT_RECORDS.md#cmp-prereg-20261110): one registry, one config builder.

Every arm inherits B2's data / augmentation / prior-noise / MANO / loss blocks of the same seed (configs/pfo2_20261103/
pfo2_fgs_s<seed>.yaml: nine training subjects of splits_semkine.json, domrand roll / scale / shift / keep without hot
pixels, speed augmentation 30-300 ms, polarity flip and per-pixel swap, mixed prior noise, so3_trans_fk 1 / 1 / 2) and
changes only the network, the input face and the schedule. All arms train on one GPU (batch x accumulation = B2's
effective batch for the frame stages), Adam, bf16, warm-up then cosine, deterministic Trainer, fresh-only.

  eh      EventHands (Rudnev et al., ICCV 2021): LNES -> ResNet-18 -> absolute 51-D state (full MANO pose instead of
          the paper's 6 PCA coefficients; sequence GT betas like every method); 6000 steps
  em      event encoder + previous-state MLP: the same LNES ResNet-18 (512-d pooled feature), MLP([feature; prev]) ->
          51-D delta (zero-initialised output layer), zero-event gate, prev + delta; 6000 steps, B2's prior contract
  emrol5  em + EvHandTrack's rol5 stage: K = 5 consecutive 50 ms packets from GT + prior noise, gradient through the
          previous state, 1500 steps from em's step 6000 of the seed, peak 1e-3, warm-up 100, cosine to 0
  eg      event encoder + GRU: eh's encoder and head at step 6000 of the seed, GRU(512) over the per-packet features from a
          zero hidden state, output = head(feature) + zero-initialised head(hidden); K = 16 packets of 50 ms, 1500
          steps, peak 1e-3, warm-up 100, cosine to 0
  epf     EvHandPose (Jiang et al., TPAMI 2024) stage 1: the official FlowNet (MultiResUNet on a 5-bin voxel grid)
          trained on mesh-flow pseudo ground truth of the GT MANO surface (the official squared-error flow loss at the
          four scales, flow / 32 px); deterministic "warn" (the official U-Net's upsampling has no deterministic
          CUDA backward)
  ep      EvHandPose stage 2: official Backbone (LNES + finest shape flow -> fusion ResNet-34 layers -> ConvGRU ->
          predictor), FlowNet frozen from epf; sequences of 12 packets of 50 ms from a zero ConvGRU state, the last 3
          supervised; full frame (no GT-joint crop); Adam 5e-4, weight decay 1e-4; 32 sequences per step (official batch
          size), 7500 steps
  eho     (amendment 4) EventHands as published, the official recipe (semkine/cmp_eho.py): 12-D output [6 MANO PCA
          coefficients, translation, root axis-angle], the official weighted MSE in log10, batch 64, Adam 1e-3 constant,
          fp32, fixed 100 ms LNES windows, no augmentation (no domrand, speed, polarity flip or swap); 96000 steps = the
          6000 x 1024 samples of eh; B2's DATA / MANO / TRACK blocks (the TRACK block only sets the protocol's
          segment-start noise, the network never reads the previous state)
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .common import B2_CONFIG, CONFIGS, ROOT, SECTION, SEEDS, STUDY, require, run_dir

FRAME = {"steps": 6000, "lr": 4.0e-3, "warmup": 500, "batch": 512, "accumulate": 2, "workers": 16}
ARMS = {
    "eh": {"model": "eh", "stage": "frame", **FRAME},
    "em": {"model": "em", "stage": "frame", **FRAME},
    "emrol5": {"model": "em", "stage": "seq", "init": ("em", 6000), "steps": 1500, "lr": 1.0e-3, "warmup": 100,
               "batch": 256, "accumulate": 4, "workers": 16, "seq": {"K": 5, "GRAD": True}},
    "eg": {"model": "eg", "stage": "seq", "init": ("eh", 6000), "steps": 1500, "lr": 1.0e-3, "warmup": 100,
           "batch": 64, "accumulate": 4, "workers": 16, "seq": {"K": 16}},
    "epf": {"model": "epf", "stage": "seq", "steps": 3000, "lr": 5.0e-4, "warmup": 200, "batch": 32, "accumulate": 1,
            "workers": 16, "seq": {"K": 1, "VOXEL_BINS": 5}, "deterministic": "warn"},
    "ep": {"model": "ep", "stage": "seq", "init": ("epf", 3000), "steps": 7500, "lr": 5.0e-4, "warmup": 500,
           "batch": 16, "accumulate": 2, "workers": 16, "seq": {"K": 12, "VOXEL_BINS": 5, "ANNOT_LEN": 3}},
    # amendment 1 (#cmp-amend1-20261110): EvHandPose with its hand crop, online (box from the state at each packet's start)
    "epcf": {"model": "epcf", "stage": "seq", "steps": 3000, "lr": 5.0e-4, "warmup": 200, "batch": 32, "accumulate": 1,
             "workers": 16, "seq": {"K": 1, "VOXEL_BINS": 5}, "deterministic": "warn", "crop": True},
    "epc": {"model": "epc", "stage": "seq", "init": ("epcf", 3000), "steps": 7500, "lr": 5.0e-4, "warmup": 500,
            "batch": 16, "accumulate": 2, "workers": 16, "seq": {"K": 12, "VOXEL_BINS": 5, "ANNOT_LEN": 3}, "crop": True},
    # amendment 2 (#cmp-amend2-20261110): epc with 3x the steps (robustness check of EvHandPose's schedule)
    "epcl": {"model": "epc", "stage": "seq", "init": ("epcf", 3000), "steps": 22500, "lr": 5.0e-4, "warmup": 500,
             "batch": 16, "accumulate": 2, "workers": 16, "seq": {"K": 12, "VOXEL_BINS": 5, "ANNOT_LEN": 3},
             "crop": True},
    # amendment 4 (#cmp-amend4-20261110): EventHands as published (the official ICCV 2021 recipe, semkine/cmp_eho.py):
    # batch 64, Adam 1e-3 constant, fp32, fixed 100 ms LNES, no augmentation; 96000 steps = eh's 6000 x 1024 samples
    "eho": {"model": "eho", "stage": "frame", "steps": 96000, "lr": 1.0e-3, "warmup": 0, "batch": 64, "accumulate": 1,
            "workers": 16, "original": True},
}
#: amendment 4: the official EventHands training (model/train.py, model/model.py at abfca72) in the CMP config schema
ORIGINAL = {
    "AUG": {"DOMRAND": {"ENABLED": False}, "SPEED_AUG": False, "POLARITY_FLIP": False, "PIXEL_POLARITY_SWAP": False},
    "WINDOW_MS": 100,
    "LOSS": {"TYPE": "mse_51d", "LAMBDA_POSE": 60.0, "LAMBDA_T": 30000.0, "LAMBDA_R": 60.0, "NORMALIZER": 12,
             "LOG10": True},
    "MODEL": {"POSE_REPR": "mano_pca6", "OUTPUT_DIM": 12, "MANO_NCOMPS": 6},
    "TRAIN": {"PRECISION": 32, "LR_SCHEDULE": "constant", "SAVE_EVERY_N_STEPS": 6000},
}
#: EvHandPose's official crop (configs/train_supervision.yml `preprocess.bbox`: size 128, joints_to_bbox_rate 2, shift_var 0.15)
CROP = {"SIZE": 128, "RATE": 2.0, "SHIFT": 0.15}
#: evaluation form of every model kind (the PFO engine's E1 on the 50 ms grid)
EVAL = {"eh": {"WINDOW_MS": 100}, "em": {"WINDOW_MS": 50}, "eg": {"WINDOW_MS": 50, "HISTORY": 16},
        "ep": {"WINDOW_MS": 50, "HISTORY": 12}, "epc": {"WINDOW_MS": 50, "HISTORY": 12}, "eho": {"WINDOW_MS": 100}}
PHASES = ("formal", "smoke")
SMOKE_STEPS = 20


def config_path(arm, seed, phase="formal"):
    return CONFIGS / (f"cmp_{arm}_s{seed}" + ("" if phase == "formal" else f"_{phase}") + ".yaml")


def checkpoint(arm, seed, step, phase="formal"):
    return run_dir(arm, seed, phase) / f"cmp_{arm}_s{seed}{'' if phase == 'formal' else '_' + phase}-step={step}.ckpt"


def build_config(arm, seed, phase="formal"):
    require(arm in ARMS and seed in SEEDS and phase in PHASES, "unknown arm / seed / phase")
    a = ARMS[arm]
    base = yaml.safe_load(Path(str(B2_CONFIG).format(seed=seed)).read_text())
    cfg = {k: copy.deepcopy(base[k]) for k in ("AUG", "DATA", "LOSS", "MANO", "TRACK")}
    cfg["SEED"] = int(seed)
    cfg["DATA"]["INPUT_MODE"] = "legacy_lnes"
    if a["stage"] == "frame":
        cfg["DATA"]["FAST_PIPE"] = True
    model = a["model"]
    cfg["MODEL"] = {
        "BACKBONE": "resnet18", "POSE_REPR": "mano_full_axis_angle", "OUTPUT_DIM": 51, "MANO_NCOMPS": 45,
        "PREDICT_BETA": False, "PREDICT_DELTA": model == "em", "PREVPOS_EMBED": False, "PREV_RENDER": False,
        "ZERO_EVENT_GATE": model == "em", "RENDER_H": 180, "RENDER_W": 240, "RENDER_SCALE": 0.375, "RENDER_CHUNK": 256,
    }
    run = run_dir(arm, seed, phase)
    name = run.name
    smoke = phase == "smoke"
    if "init" in a:
        src, step = a["init"]
        cfg["MODEL"]["INIT_FROM"] = str(checkpoint(src, seed, SMOKE_STEPS if smoke else step, phase))
    cfg["TRAIN"] = {
        "DEVICES": 1, "STRATEGY": "ddp", "PRECISION": "bf16", "BATCH_SIZE_PER_GPU": 8 if smoke else a["batch"],
        "ACCUMULATE_GRAD_BATCHES": a["accumulate"], "NUM_WORKERS": 4 if smoke else a["workers"],
        "PERSISTENT_WORKERS": True, "PREFETCH_FACTOR": 2, "PIN_MEMORY": True, "OPTIMIZER": "adam", "LR": a["lr"],
        "WARMUP_STEPS": 5 if smoke else a["warmup"], "MAX_STEPS": SMOKE_STEPS if smoke else a["steps"],
        "VAL_CHECK_INTERVAL": 2000, "SAVE_EVERY_N_STEPS": 10 if smoke else 500, "SYNC_BATCHNORM": False,
        "OUTPUT_DIR": str(run), "RUN_NAME": name, "LR_SCHEDULE": "cosine",
        "LIMIT_VAL_BATCHES": 8 if a["stage"] == "frame" else 0,
    }
    if model in ("ep", "epc"):
        cfg["TRAIN"]["WEIGHT_DECAY_L2"] = 1.0e-4
    if a.get("original"):                                     # amendment 4: the official EventHands recipe
        cfg["AUG"] = copy.deepcopy(ORIGINAL["AUG"])
        cfg["DATA"]["WINDOW_MIN"] = cfg["DATA"]["WINDOW_MAX"] = ORIGINAL["WINDOW_MS"]
        cfg["LOSS"] = dict(ORIGINAL["LOSS"])
        cfg["MODEL"].update(ORIGINAL["MODEL"])
        cfg["TRAIN"].update(ORIGINAL["TRAIN"])
        if smoke:
            cfg["TRAIN"]["SAVE_EVERY_N_STEPS"] = 10
    cfg["EVAL"] = {"WINDOW_MS": 50, "OUTPUT_DIR": str(run / "eval")}
    cmp_ = {"ARM": arm, "SEED": int(seed), "PHASE": phase, "MODEL": model, "STAGE": a["stage"], "STUDY": STUDY,
            "SECTION": SECTION}
    if "seq" in a:
        cmp_["SEQ"] = {"PACKET_MS": 50, **a["seq"]}
    if model in EVAL:
        cmp_["EVAL"] = dict(EVAL[model])
    if "init" in a:
        cmp_["INIT"] = {"ARM": a["init"][0], "STEP": a["init"][1]}
    if a.get("deterministic", True) is not True:
        cmp_["DETERMINISTIC"] = a["deterministic"]
    if a.get("crop"):
        cmp_["CROP"] = dict(CROP)
    cfg["CMP"] = cmp_
    return cfg


def write_configs(phase="formal", arms=None):
    out = []
    for arm in arms or ARMS:
        for seed in SEEDS:
            p = config_path(arm, seed, phase)
            p.parent.mkdir(parents=True, exist_ok=True)
            text = yaml.safe_dump(build_config(arm, seed, phase), sort_keys=False)
            if p.exists():
                require(p.read_text() == text, "registered config changed: " + str(p))
            else:
                p.write_text(text)
            out.append(p)
    return out


if __name__ == "__main__":
    import sys
    for p in write_configs(*(sys.argv[1:2] or ["formal"])):
        print(p.relative_to(ROOT))
