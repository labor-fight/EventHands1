#!/usr/bin/env python3
"""DT round (docs/DT_RENDER_TRACK_PREREG.md): write every arm's config from its parent.

Same contract as `tools/rt/make_configs.py` and `tools/s38/make_configs.py`: an arm is its parent plus the
listed overrides and nothing else, so an arm differs from its control exactly where the table below says.

Parent: `configs/rt/rt_cnntrack.yaml`, the ResNet18 / LNES + rendered previous state `prev + delta` tracker
under the unified recipe (1 GPU x 512 x accumulate 2, Adam 4e-3, warmup 500, cosine -> 2 %, bf16, 6000 steps,
500-step checkpoint grid, fixed protocol split `splits_semkine.json`).

`dt_base` adds the objective of the historical EventHands-Track (render + SO3 + FK + DomRand), i.e. the LOSS
block of `configs/eventhands_track_render51_so3fk.yaml`: `so3_trans_fk` with rotation 1, translation 1,
root-relative FK 2. That is the architecture under study, re-run under the recipe every other arm of the
project uses. `rt_cnntrack` (the same network under the MSE objective) is the second reference.
Every candidate is `dt_base` plus ONE factor. `_2k` files are the screening budget (the cosine compressed
to 2000 steps; the schedule reads TRAIN.MAX_STEPS, so a `--max-steps` cut would leave the LR high).

    python tools/dt/make_configs.py            # (re)writes configs/dt/*.yaml
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PARENT = REPO / "configs/rt/rt_cnntrack.yaml"
OUT = REPO / "configs/dt"
SCREEN_STEPS = 2000

#: the historical so3fk objective, verbatim (the LAMBDA_* / NORMALIZER / LOG10 keys of the parent stay)
SO3FK = {"TYPE": "so3_trans_fk", "ROT_WEIGHT": 1.0, "TRANS_WEIGHT": 1.0, "FK_WEIGHT": 2.0}

#: name -> (overrides on top of dt_base, why, code needed beyond the committed model). Order = table order.
ARMS = {
    "dt_base": ({}, "EventHands-Track (render + SO3 + FK + DomRand) under the unified recipe", ""),
    # P1: translation is barely supervised (SmoothL1 beta 1 m puts every centimetre error in the quadratic
    # region; the FK term is root-relative). Both remedies exist in the loss and were never trained on this arm.
    "dt_tr": ({"LOSS": {"TRANS_BETA": 0.01, "ABS_FK_WEIGHT": 1.0}},
              "C1: translation supervised at its error scale (SmoothL1 beta 0.01 m) + absolute FK term", ""),
    # P3: the scale augmentation changes the pixel size of the hand while the 3D label stays, so the
    # pixel-size <-> depth cue the network could read is broken. The control for dt_cam.
    "dt_nos": ({"AUG": {"DOMRAND": {"SCALE_MIN": 1.0, "SCALE_MAX": 1.0}}},
               "C2a: no scale augmentation (pixel size keeps meaning depth); control for dt_cam", ""),
    # F4: zgz_local lies at 699-742 mm, outside the training depths (max 710 mm); every recorded model shrinks its
    # depth toward the training range there. The scale augmentation with the label depth scaled consistently
    # (image x s, depth / s, K unchanged) is a depth-range augmentation; same 0.8-1.25 range, nothing tuned.
    "dt_dz": ({"AUG": {"DOMRAND": {"SCALE_MODE": "depth"}}},
              "C2c: depth-consistent scale augmentation (image x s, label depth / s, K unchanged)", "domrand.py, dataset.py"),
    # 2 x 2 factorial of the two zero-code factors (C1 x C2a): the interaction, no gate of its own
    "dt_trnos": ({"LOSS": {"TRANS_BETA": 0.01, "ABS_FK_WEIGHT": 1.0}, "AUG": {"DOMRAND": {"SCALE_MIN": 1.0, "SCALE_MAX": 1.0}}},
                 "C1 + C2a (factorial arm): translation at its error scale, no scale augmentation", ""),
    "dt_cam": ({"MODEL": {"CAM_PLANES": True}},
               "C2b: two camera-ray planes (u - cx) / fx, (v - cy) / fy of the augmented K as extra input", "model.py"),
    # P4: root update is a sum of axis-angles; the training roots sit at a median |aa| of 132 deg
    "dt_so3c": ({"MODEL": {"ROOT_COMPOSE": "so3"}},
                "C3: the root rotation is composed on SO(3), R = Exp(delta) R_prev (left multiplication)", "model.py"),
    # P8 / P2: no temporal term in the loss; jitter is second-difference of the error
    "dt_acc": ({"LOSS": {"ACCEL_WEIGHT": 1.0}, "DATA": {"TRIPLET": True}},
               "C4: acceleration loss on three consecutive packets (teacher forced), UmeTrack-style", "model.py, dataset.py"),
    # P7: 11.2 M parameters for a 180 x 240 input, layer4 is 8.4 M of them
    "dt_w05": ({"MODEL": {"CNN_BACKBONE": "resnet18_w0.5"}},
               "C5: ResNet18 at half width (32-64-128-256), ~4x fewer parameters", "model.py, backbones.py"),
    "dt_l3": ({"MODEL": {"CNN_BACKBONE": "resnet18_l3"}},
              "C6: ResNet18 without layer4 (fc from 256), -75 % parameters", "model.py, backbones.py"),
}


def merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if v is None:
            dst.pop(k, None)
        elif isinstance(v, dict):
            merge(dst.setdefault(k, {}), v)
        else:
            dst[k] = v
    return dst


def arm_overrides(name: str) -> dict:
    """The full override of an arm over the parent: the so3fk objective plus the arm's own factor."""
    own = ARMS[name][0]
    return merge(merge({}, {"LOSS": copy.deepcopy(SO3FK)}), copy.deepcopy(own))


def write(name: str, steps=None) -> Path:
    over = arm_overrides(name)
    cfg = merge(copy.deepcopy(yaml.safe_load(PARENT.read_text())), copy.deepcopy(over))
    run = name if steps is None else f"{name}_2k"
    if steps is not None:
        cfg["TRAIN"]["MAX_STEPS"] = steps
    cfg["TRAIN"]["OUTPUT_DIR"] = f"outputs/semkine/{run}"
    cfg["TRAIN"]["RUN_NAME"] = run
    cfg["EVAL"]["OUTPUT_DIR"] = f"outputs/semkine/{run}/eval"
    why, needs = ARMS[name][1], ARMS[name][2]
    head = (f"# {run}: {why}.\n# Generated by tools/dt/make_configs.py from {PARENT.relative_to(REPO)} with {over}"
            + (f", TRAIN.MAX_STEPS {steps} (screening budget)" if steps else "") + ".\n"
            + (f"# Needs code beyond the committed model: {needs} (unknown MODEL keys raise, so it cannot run without).\n"
               if needs else "")
            + "# Seeds and output paths are set on the command line.\n")
    p = OUT / f"{run}.yaml"
    p.write_text(head + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return p


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ARMS:
        print(write(name).relative_to(REPO))
        print(write(name, SCREEN_STEPS).relative_to(REPO))


if __name__ == "__main__":
    main()
