#!/usr/bin/env python3
"""XDG-1: three fresh seed3407 runs derived from configs/xdl/xdl_q1_s3407.yaml, one change each.

  xdg_q1t0  MODEL.XDL_STATE_T=false   (state99 -> state96: absolute translation t/.1 removed)
  xdg_q1dz  AUG.DOMRAND.SCALE_MODE=depth (scale moves the labelled root depth instead of the focal length)
  xdg_q1r   unchanged Q1 (fresh same-seed repeat: engineering noise floor)
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / 'configs/xdl/xdl_q1_s3407.yaml'
ARMS = {
    'q1t0': lambda c: c['MODEL'].update(XDL_STATE_T=False),
    'q1dz': lambda c: c['AUG']['DOMRAND'].update(SCALE_MODE='depth'),
    'q1r': lambda c: None,
}


def make_configs(output_dir=None):
    output_dir = Path(output_dir or REPO / 'configs/xdg')
    output_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(BASE.read_text())
    paths = []
    for arm, change in ARMS.items():
        cfg = copy.deepcopy(base)
        change(cfg)
        run = f'xdg_{arm}_s3407'
        cfg['TRAIN'].update(RUN_NAME=run, OUTPUT_DIR=f'outputs/semkine/{run}')
        cfg['EVAL'].update(OUTPUT_DIR=f'outputs/semkine/{run}/eval')
        path = output_dir / f'{run}.yaml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        paths.append(path)
    return paths


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir')
    for path in make_configs(ap.parse_args().output_dir):
        print(path)
