#!/usr/bin/env python3
"""Compact read-only phase status and TensorBoard training progress."""
import argparse
import json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
R = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument('phase')
args = ap.parse_args()
state = json.loads((R/f'outputs/u1a/{args.phase}_state.json').read_text())
for job in state['jobs']:
    row = {k:job[k] for k in ['run','status','gpu']}
    files = list((R/'outputs/semkine'/job['run']/'logs').rglob('events.out.tfevents.*'))
    if files:
        acc = EventAccumulator(str(files[0]), size_guidance={'scalars':0}).Reload()
        for tag in ['train_loss','train_rot_loss','train_mano_loss','train_pos_loss',
                    'train_samples_per_sec','cuda_max_mem_gb']:
            if tag in acc.Tags()['scalars']:
                event = acc.Scalars(tag)[-1]
                row[tag] = round(event.value, 4)
                row['step'] = event.step
    print(json.dumps(row))
