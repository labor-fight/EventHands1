#!/usr/bin/env python3
"""XKF configs: the XDG q1t0 config (XDL Q1 with state96) plus depth-scale augmentation, small previous-state noise
(the XYZ branch is the motion model; correction is the direct branch's job) and the XKF keys."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / 'configs/xdg/xdg_q1t0_s3407.yaml'
SEEDS = (3407, 3408, 3409)


def make_configs(output_dir=None):
    output_dir = Path(output_dir or REPO / 'configs/xkf')
    output_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(BASE.read_text())
    paths = []
    for seed in SEEDS:
        cfg = copy.deepcopy(base)
        run = f'xkf_s{seed}'
        cfg['SEED'] = seed
        cfg['AUG']['DOMRAND']['SCALE_MODE'] = 'depth'
        cfg['MODEL'].update(XKF=True, XKF_KMODE='fused', XKF_EMA=0.99, XKF_KP_WEIGHT=300.0,
                            XKF_HM_WEIGHT=30.0, XKF_FK_WEIGHT=30000.0)
        cfg['TRACK'].update(PREV_NOISE_MODE='gaussian', PREV_NOISE_T=0.005, PREV_NOISE_R=0.05, PREV_NOISE_POSE=0.05)
        cfg['XKF_DATA'] = {'WINDOW_MIN': 30, 'WINDOW_MAX': 300, 'EVAL_WINDOW_MS': 200}
        cfg['TRAIN'].update(RUN_NAME=run, OUTPUT_DIR=f'outputs/semkine/{run}')
        cfg['EVAL'].update(OUTPUT_DIR=f'outputs/semkine/{run}/eval')
        path = output_dir / f'{run}.yaml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        paths.append(path)
    return paths


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir')
    for p in make_configs(ap.parse_args().output_dir):
        print(p)
