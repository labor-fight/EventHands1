#!/usr/bin/env python3
"""XRF configs from configs/xkf/xkf_s3407.yaml (XDL Q1, state96, depth-scale augmentation, small prev noise):
per-packet training (no self burn), the XYZ path as a refiner of the direct measurement. `debug` writes the round-1
variants; `formal` writes the full-training arms once a debug round meets its expectations."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / 'configs/xkf/xkf_s3407.yaml'
S39_CKPT = str(REPO / 'outputs/semkine/dt_dz_l3_s3407/dt_dz_l3_s3407-step=6000.ckpt')
DEBUG = {
    'xrf_dbg_a': {},
    'xrf_dbg_b': {'XRF_D_INIT': S39_CKPT},
    'xrf_dbg_c': {'XRF_REFINE_ITERS': 2},
}


def make(name, model_overrides, seed=3407, output_dir=None):
    base = yaml.safe_load(BASE.read_text())
    cfg = copy.deepcopy(base)
    cfg['SEED'] = seed
    for k in ('XKF', 'XKF_KMODE', 'XKF_EMA', 'XKF_KP_WEIGHT'):
        cfg['MODEL'].pop(k, None)
    cfg['MODEL'].update(XDL_SELF=False, XKF_KP_WEIGHT=300.0, XRF=True, XRF_KMODE='fused', XRF_REFINE_ITERS=1,
                        XRF_HYP_NOISE=[0.01, 0.05, 0.05], XRF_HYP_GT_END=0.25, XRF_HYP_RAMP=2000)
    cfg['MODEL'].update(model_overrides)
    cfg['XKF_DATA'] = {'WINDOW_MIN': 30, 'WINDOW_MAX': 300, 'EVAL_WINDOW_MS': 200, 'PREFIX_LNES': False}
    cfg['TRAIN'].update(RUN_NAME=name, OUTPUT_DIR=f'outputs/semkine/{name}')
    cfg['EVAL'].update(OUTPUT_DIR=f'outputs/semkine/{name}/eval')
    out = Path(output_dir or REPO / 'configs/xrf') / f'{name}.yaml'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('which', choices=['debug'])
    args = ap.parse_args()
    for name, ov in DEBUG.items():
        print(make(name, ov))
