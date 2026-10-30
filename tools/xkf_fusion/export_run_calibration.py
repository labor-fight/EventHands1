#!/usr/bin/env python3
"""Export fixed leave-whole-run-out affine calibration and a separate dev refit."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.xkf_fusion.calibration_probe import fit, embargo_mask, sha, write, array_hash


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=ROOT / 'outputs/xkf/fusion_20261015_r3/calibration')
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs/xkf/fusion_20261015_r3/run_calibration')
    args = ap.parse_args()
    if (args.output / 'complete.json').exists():
        raise FileExistsError(args.output)
    protocol = {'kind': 'leave_whole_valid_run_out_translation_affine', 'formal': False,
                'fit': 'pooled available local/global rows, excluding every scored row of current run and <200ms neighbors',
                'no_fit_target_from_scored_run': True, 'fixed_through_complete_run': True,
                'sequence_specific_runtime_deployment_parameters': False,
                'global_caveat': 'one global valid run: its leaveout calibrator is fitted on local only; retain outcome',
                'full_development_refit_is_separate': True, 'source_sha256': sha(__file__),
                'inputs': {str(s): sha(args.input / f'seed{s}_crossfit.npz') for s in (3407, 3408, 3409)}}
    write(args.output / 'protocol.json', protocol)
    for seed in (3407, 3408, 3409):
        with np.load(args.input / f'seed{seed}_crossfit.npz', allow_pickle=False) as raw:
            p, j, y = [raw[k].copy() for k in ('input_direct_state', 'identity__joints0', 'truth')]
            seq, ids, end = [raw[k].copy() for k in ('sequence', 'packet_id', 'end_ms')]
        runs = np.array([int(x.split(':')[1]) for x in ids])
        def model(mask):
            value = fit('translation_affine', p[mask], j[mask], y[mask], np.zeros((mask.sum(), 6)), False)
            value.pop('fit_rows')
            return value
        out = {'seed': seed, 'protocol_sha256': sha(args.output / 'protocol.json'),
               'run_specs': {}, 'audit': {}, 'full_development_refit_spec': model(np.ones(len(y), bool))}
        for name in np.unique(seq):
            out['run_specs'][name], out['audit'][name] = {}, {}
            for run in np.unique(runs[seq == name]):
                test = (seq == name) & (runs == run)
                train = embargo_mask(seq, end, ~test, test, 200)
                assert not train[test].any() and train.any()
                out['run_specs'][name][str(run)] = model(train)
                out['audit'][name][str(run)] = {'n_train': int(train.sum()), 'n_scored': int(test.sum()),
                     'train_packet_ids': ids[train].tolist(), 'scored_packet_ids': ids[test].tolist(),
                     'train_mask_sha256': array_hash(train), 'minimum_same_sequence_distance_ms':
                     int(np.abs(end[train & (seq == name), None] - end[test][None]).min())
                     if (train & (seq == name)).any() else None}
        write(args.output / f'seed{seed}_runtime.json', out)
    write(args.output / 'complete.json', {'seeds': [3407, 3408, 3409], 'protocol_sha256': sha(args.output / 'protocol.json')})


if __name__ == '__main__':
    main()
