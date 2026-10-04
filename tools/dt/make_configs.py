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
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.config_utils import merge  # noqa: E402
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
    # 2 x 2 factorial of C1 and C2c (with dt_base, dt_tr, dt_dz): the interaction of the two factors the
    # in-sample depth diagnostic (F14) points at, no gate of its own
    "dt_trdz": ({"LOSS": {"TRANS_BETA": 0.01, "ABS_FK_WEIGHT": 1.0}, "AUG": {"DOMRAND": {"SCALE_MODE": "depth"}}},
                "C1 + C2c (factorial arm): translation at its error scale, depth-consistent scale augmentation", "domrand.py, dataset.py"),
    # F4 mechanism: the event-blind prev_mlp (51 -> 64 -> 51) reads the raw previous state and may have learned a
    # pull of the depth toward the training mean (the training prev noise makes the previous depth unreliable).
    # dt_nopm removes it (zero code: PREVPOS_EMBED is an existing key); dt_pmt keeps it but masks its three
    # translation outputs.
    "dt_nopm": ({"MODEL": {"PREVPOS_EMBED": False}},
                "C9a: no event-blind prev_mlp (the previous state enters through the render only)", ""),
    "dt_pmt": ({"MODEL": {"PREV_MLP_TRANSL": False}},
               "C9b: prev_mlp keeps its rotation / finger outputs, its translation outputs are masked to 0", "model.py"),
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
    # DT2 round (docs/DT2_PREREG.md): single factors, combined onto dt_dz_l3 with --pack.
    # Noise floor: training LNES never has fewer than ~551 occupied slots (2e-4 hot pixels per slot-ms = 864 per
    # 50 ms) while zgz_local has a median of 326 and a real background of ~32 events per 50 ms.
    "dt_nfh": ({"AUG": {"DOMRAND": {"HOT_PIXEL_RATE": 2.0e-5}}},
               "DT2-A3a: hot-pixel rate / 10 (training noise floor below the zgz_local occupancy)", ""),
    "dt_nfk": ({"AUG": {"DOMRAND": {"HOT_PIXEL_RATE": 2.0e-5, "KEEP_MIN": 0.10}}},
               "DT2-A3b: hot-pixel rate / 10 and event keep down to 0.10 (sparse packets in training)", ""),
    # Window: 78 % of training windows exceed the 50 ms evaluation step, and the window also sets the training step
    "dt_w50": ({"AUG": {"SPEED_AUG": False}, "DATA": {"WINDOW_MIN": 50, "WINDOW_MAX": 50}},
               "DT2-A4a: fixed 50 ms window = evaluation step", ""),
    "dt_w100": ({"DATA": {"WINDOW_MAX": 100}},
                "DT2-A4b: log-uniform window 30-100 ms", ""),
    # Event time: a second event plane with the earliest timestamp of each slot (events.py `first`)
    "dt_lf": ({"DATA": {"EVENT_CHANNELS": ["last", "first"]}},
              "DT2-A5: event planes [last, first] (direction of motion inside the window)", ""),
    # Polarity: the per-pixel polarity swap at training time only erases the sign of the brightness change
    "dt_nosw": ({"AUG": {"PIXEL_POLARITY_SWAP": False}},
                "DT2-A8: no per-pixel polarity swap (polarity kept as evaluated)", ""),
    # Model-side arms (docs/DT2_PREREG.md section 2; model/model.py, all default-off). Root rotation is 63 % of RA and its
    # error is a slow systematic bias that is already there under teacher forcing (80 %); training has no regulariser.
    "dt_ema": ({"TRAIN": {"EMA_DECAY": 0.999}},
               "DT2-B1: weight EMA 0.999 (every checkpoint's state_dict is the EMA model, raw weights in raw_state_dict)",
               "model.py"),
    "dt_wd": ({"TRAIN": {"WEIGHT_DECAY": 0.01}},
              "DT2-B2: AdamW weight decay 0.01 (Conv / Linear weights only; BatchNorm and biases not decayed)", "model.py"),
    "dt_reg": ({"TRAIN": {"EMA_DECAY": 0.999, "WEIGHT_DECAY": 0.01}},
               "DT2-B3: EMA 0.999 + AdamW weight decay 0.01", "model.py"),
    "dt_rootw4": ({"LOSS": {"ROOT_ROT_WEIGHT": 4.0}},
                  "DT2-B4: root rotation weighted 4 among the 16 rotations of L_rot (was 1)", "model.py"),
    "dt_rooth": ({"MODEL": {"ROOT_HEAD": "spatial"}},
                 "DT2-B5: spatial root readout on the layer3 map (zero-initialised, added to the fc's root row)",
                 "model.py, backbones.py"),
    "dt_rootanc": ({"MODEL": {"ROOT_HEAD": "anchor"}},
                   "DT2-B6: anchored root readout (layer2 features sampled at the previous state's 21 projected joints)",
                   "model.py, backbones.py"),
    "dt_r32": ({"MODEL": {"RENDER_FP32": True}},
               "DT2-B7: previous-state render in fp32 under bf16 training (hygiene arm)", "model.py"),
}


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


def pack_overrides(factors: list) -> dict:
    """Overrides of a combination arm: the so3fk objective plus the keys of every listed single-factor arm, in order
    (a later factor overrides an earlier one on the same key, which the registered rule of section 6 never needs:
    the factors it combines touch different keys)."""
    out = merge({}, {"LOSS": copy.deepcopy(SO3FK)})
    for f in factors:
        merge(out, copy.deepcopy(ARMS[f][0]))
    return out


def write_pack(name: str, factors: list, steps=None) -> Path:
    over = pack_overrides(factors)
    cfg = merge(copy.deepcopy(yaml.safe_load(PARENT.read_text())), copy.deepcopy(over))
    run = name if steps is None else f"{name}_2k"
    if steps is not None:
        cfg["TRAIN"]["MAX_STEPS"] = steps
    cfg["TRAIN"]["OUTPUT_DIR"] = f"outputs/semkine/{run}"
    cfg["TRAIN"]["RUN_NAME"] = run
    cfg["EVAL"]["OUTPUT_DIR"] = f"outputs/semkine/{run}/eval"
    needs = sorted({ARMS[f][2] for f in factors if ARMS[f][2]})
    joined = ", ".join(factors)
    head = (f"# {run}: combination of {joined} (docs/DT_RENDER_TRACK_PREREG.md section 6).\n"
            f"# Generated by tools/dt/make_configs.py --pack from {PARENT.relative_to(REPO)} with {over}"
            + (f", TRAIN.MAX_STEPS {steps} (screening budget)" if steps else "") + ".\n"
            + (f"# Needs code beyond the committed model: {', '.join(needs)}.\n" if needs else "")
            + "# Seeds and output paths are set on the command line.\n")
    p = OUT / f"{run}.yaml"
    p.write_text(head + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return p


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", nargs="+", metavar=("NAME", "FACTOR"),
                    help="write one combination arm: NAME followed by the single-factor arms to merge")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.pack:
        name, factors = a.pack[0], a.pack[1:]
        unknown = [f for f in factors if f not in ARMS]
        if not factors or unknown:
            raise SystemExit(f"--pack NAME needs known factor arms, got {factors} (unknown: {unknown})")
        print(write_pack(name, factors).relative_to(REPO))
        print(write_pack(name, factors, SCREEN_STEPS).relative_to(REPO))
        return
    for name in ARMS:
        print(write(name).relative_to(REPO))
        print(write(name, SCREEN_STEPS).relative_to(REPO))


if __name__ == "__main__":
    main()
