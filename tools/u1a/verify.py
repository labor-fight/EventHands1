#!/usr/bin/env python3
"""Audit last checkpoints and paired streams without selecting any checkpoint."""
import argparse
import json
from pathlib import Path
import torch
R = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument('phase', choices=['debug','2k','6k'])
args = ap.parse_args()
budget = {'debug':20,'2k':2000,'6k':6000}[args.phase]
seeds = [3407] if args.phase == 'debug' else [3407,3408]
torch.set_num_threads(2)
result = {}
for seed in seeds:
    dirs = [R/'outputs/semkine'/f'u1a_{m}_{args.phase}_s{seed}' for m in ['shared','untied']]
    for rank in range(2):
        assert json.loads((dirs[0]/f'pair_stream_rank{rank}.json').read_text()) == json.loads(
            (dirs[1]/f'pair_stream_rank{rank}.json').read_text()), f'pair stream mismatch seed={seed} rank={rank}'
    anchor = torch.load(R/f'outputs/semkine/s38_spmeas_s{seed}/last.ckpt', map_location='cpu')['state_dict']
    for mode, directory in zip(['shared','untied'], dirs):
        ckpt = torch.load(directory/'last.ckpt', map_location='cpu')
        assert ckpt['global_step'] == budget, (directory, ckpt['global_step'])
        state = ckpt['state_dict']
        for key, val in state.items():
            assert torch.isfinite(val).all(), key
            if key.startswith('event_encoder.'):
                assert torch.equal(val, anchor[key]), f'frozen encoder changed: {directory} {key}'
        result[directory.name] = {'last_step': budget, 'frozen_encoder_bitwise': True,
                                  'all_state_finite': True, 'paired_rank_streams_equal':True}
(R/f'outputs/u1a/{args.phase}_verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
