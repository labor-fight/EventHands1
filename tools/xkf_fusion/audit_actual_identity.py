#!/usr/bin/env python3
"""Independent checkpoint/source binding for the three actual-forward audits.

Raw trajectory/metric validation is deliberately a separate independent audit.
Runtime buffer equality is identified as a producer assertion checked in its
source, while checkpoint parameter identity is recomputed here from tensors.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT / 'outputs/xkf/fusion_20261015_r3/direct_finetune')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs, results = {}, {}

    def digest(path):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        value = sha(path)
        inputs[str(path.resolve())] = value
        return value

    def read(path):
        digest(path)
        return json.loads(Path(path).read_text())

    training = read(args.study / 'independent_training_audit.json')
    assessment = read(args.study / 'assessment_raw.json')
    require(training['status'] == 'PASS', 'training audit is not complete')
    require(assessment['status'] == 'ASSESSED' and not assessment['validation_errors'] and not assessment['missing'],
            'independent raw assessment is not complete')
    for seed in (3407, 3408, 3409):
        directory = args.study / f'actual_forward_raw_s{seed}'
        summary = read(directory / 'summary.json')
        protocol = read(directory / 'protocol.json')
        require(summary['protocol_sha256'] == digest(directory / 'protocol.json'), 'actual protocol changed')
        require(summary['identity'] == protocol['identity'], 'actual identity differs between summary/protocol')
        identity = summary['identity']
        require(identity['seed'] == seed and identity['global_step'] == 2400
                and identity['clock_scope'] == {'I': 2200, 'D_and_global': 2400}, 'wrong mixed clock identity')
        require(not summary['partial_smoke'] and summary['clip_count'] == 112 and len(protocol['clips']) == 112,
                'incomplete actual-forward audit')
        require(summary['modes'] == ['reference27', 'off_forward1', 'off_forward9', 'stable_forward9', 'hold'],
                'actual modes incomplete')
        require(summary['candidate_id'] == protocol['candidate_id'] == 'uncalibrated_centroid_t05_r025_f05_ramp',
                'candidate changed')
        require(protocol['independent_assessment_sha256'] == digest(args.study / 'assessment_raw.json'),
                'assessment binding changed')
        for path, expected in protocol['assessed_reference_input_sha256'].items():
            require(digest(path) == expected == assessment['input_sha256'][path], 'reference hash changed')
        require(len(protocol['assessed_reference_input_sha256']) == 114, 'reference source denominator changed')
        for path, expected in protocol['source_sha256'].items():
            require(digest(path) == expected, 'producer source changed: ' + path)
        checkpoint = Path(identity['checkpoint'])
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        expected = training['seeds'][str(seed)]['raw']
        require(digest(checkpoint) == identity['checkpoint_sha256'] == expected['checkpoint_sha256'],
                'checkpoint differs from independently audited endpoint')
        require(digest(identity['config']) == identity['config_sha256'] == expected['config_sha256'], 'config changed')
        order = read(checkpoint.parent / 'resume_audit_rank0.json')['parameter_order']
        require(order == read(checkpoint.parent / 'resume_audit_rank1.json')['parameter_order']
                and len(order) == len(set(order)) == 96, 'parameter registration order differs')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(saved['global_step'] == 2400, 'actual checkpoint clock changed')
        state = saved['state_dict']
        parameter_hash = hashlib.sha256()
        for name in order:
            tensor = state[name]
            require(bool(torch.isfinite(tensor).all()), 'nonfinite checkpoint parameter: ' + name)
            parameter_hash.update(name.encode())
            parameter_hash.update(tensor.float().contiguous().numpy().tobytes())
        require(parameter_hash.hexdigest() == summary['unchanged_parameters_sha256'],
                'actual runtime parameter digest differs from independently read checkpoint tensors')
        require(summary['all_buffers_unchanged'] is True, 'runtime producer buffer equality assertion failed')
        parent = ROOT / f'outputs/xkf/repair_20261015_r2/repair_s{seed}/xkf_repair_s{seed}-step=2200.ckpt'
        digest(parent)
        old = torch.load(parent, map_location='cpu', weights_only=False)['state_dict']
        frozen_names = [name for name in state if not name.startswith('direct.')
                        or name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
        require(all(torch.equal(state[name], old[name]) for name in frozen_names), 'saved I/BN/EMA differs from parent')
        results[str(seed)] = {
            'checkpoint_sha256': identity['checkpoint_sha256'], 'config_sha256': identity['config_sha256'],
            'independently_recomputed_parameter_sha256': parameter_hash.hexdigest(), 'parameter_count': len(order),
            'saved_I_BN_EMA_bitwise_parent': True, 'saved_frozen_tensor_count': len(frozen_names),
            'runtime_all_buffers_unchanged': {'producer_assertion': True,
                'method': 'source checks every named buffer with torch.equal against its pre-inference CPU clone',
                'independent_runtime_tensor_snapshot_available': False},
            'reference_assessment_bound_files': 114, 'complete_clips': 112,
            'formal': False, 'effect_or_readiness_pass_implied': False}
        del saved, state, old
    result = {'kind': 'XKF_ACTUAL_FORWARD_IDENTITY_AUDIT', 'status': 'PASS', 'formal': False,
              'scope': 'checkpoint/source/provenance only; raw trajectory and metric assessment separate',
              'seeds': results, 'input_sha256': inputs, 'validator_sha256': sha(__file__)}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'seeds': list(results)}), flush=True)


if __name__ == '__main__':
    main()
