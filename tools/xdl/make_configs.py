#!/usr/bin/env python3
"""Emit the five matched, fresh-init, seed3407 XD-V2 first-stage configs."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[2]
CONDITIONS = {
    'f0': ('fixed', False, False),
    'q0': ('mano', False, False),
    'f1': ('fixed', True, False),
    'q1': ('mano', True, False),
    'n1': ('mano', True, True),
}


def make_configs(output_dir=None):
    output_dir = Path(output_dir or REPO / 'configs/xdl')
    output_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((REPO / 'configs/xa/xa_c0_s3407.yaml').read_text())
    base['SEED'] = 3407
    base['DATA'].update(WINDOW_MIN=50, WINDOW_MAX=50, INPUT_MODE='raw_packed')
    base['DATA']['SPLITS_MANIFEST'] = str((REPO / 'data/hand_data51/splits_semkine.json').resolve())
    base['AUG']['SPEED_AUG'] = False
    base['MODEL'].update(XDL_STUDY=True, XYZ_GEOM=False, XDL_BURN_MODE='full')
    # Existing conditioning mechanisms remain disabled; raw XDL is implemented
    # in the isolated study class and dataset rather than those LNES switches.
    for key in ('SELF_P', 'UNROLL_P', 'ROLLOUT_P'):
        base.setdefault('TRACK', {})[key] = 0.0
    base['TRAIN'].update(DEVICES=2, BATCH_SIZE_PER_GPU=512, ACCUMULATE_GRAD_BATCHES=1,
                         NUM_WORKERS=20, PREFETCH_FACTOR=1, MAX_STEPS=6000,
                         LR=0.004, LR_SCHEDULE='constant', WARMUP_STEPS=500,
                         SAVE_EVERY_N_STEPS=500, SYNC_BATCHNORM=False)
    paths = []
    for arm, (query, self_state, null) in CONDITIONS.items():
        cfg = copy.deepcopy(base)
        run = f'xdl_{arm}_s3407'
        cfg['MODEL'].update(XDL_QUERY=query, XDL_SELF=self_state,
                             XDL_OBSERVATION_NULL=null)
        cfg['TRAIN'].update(RUN_NAME=run, OUTPUT_DIR=f'outputs/semkine/{run}')
        cfg.setdefault('EVAL', {}).update(WINDOW_MS=50, OUTPUT_DIR=f'outputs/semkine/{run}/eval')
        path = output_dir / f'{run}.yaml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        paths.append(path)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir')
    args = ap.parse_args()
    for path in make_configs(args.output_dir):
        print(path)


if __name__ == '__main__':
    main()
