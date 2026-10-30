#!/usr/bin/env python3
"""Independent saved-state and complete sample-stream audit of paired D tuning."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import torch
from config import load_config
from semkine.xkf_repair import XKFRepairModel
from semkine.dataset import sequences_for_split, splits_manifest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, description):
    if not value:
        raise ValueError(description)


def read(path):
    return json.loads(Path(path).read_text())


def validate_arm(directory, seed, steps):
    metadata = read(directory / 'training_metadata.json')
    config = directory / 'config_resolved.yaml'
    cfg = load_config(str(config))
    arm = metadata['arm']
    end = 2200 + steps
    require(metadata['updates'] == steps and metadata['max_steps'] == end, 'declared endpoint differs')
    require(metadata['seed'] == seed and metadata['formal'] is False, 'seed/formal identity differs')
    require(metadata['effective_batch'] == 1024 and metadata['batch_size_per_gpu'] == 512, 'batch differs')
    # The producer's historical key identifies the input parent YAML, not the
    # emitted effective YAML. Check both identities explicitly and separately.
    parent_config = ROOT / f'configs/xkf_repair/repair_s{seed}.yaml'
    require(metadata['lr'] == .0004 and sha(parent_config) == metadata['config_sha256'], 'LR/parent config differs')
    expected_cfg = load_config(str(parent_config))
    expected_cfg['XKF_DATA']['PREFIX_LNES'] = True
    if arm == 'raw':
        expected_cfg['AUG']['DOMRAND']['ENABLED'] = False
        expected_cfg['AUG']['POLARITY_FLIP'] = False
        expected_cfg['AUG']['PIXEL_POLARITY_SWAP'] = False
    expected_cfg['TRAIN'].update(MAX_STEPS=end, LR=.0004, OUTPUT_DIR=str(directory.resolve()),
                                 NUM_WORKERS=cfg['TRAIN']['NUM_WORKERS'], RUN_NAME=f'xkf_donly_{arm}_s{seed}')
    normalized = lambda c: {k: v for k, v in c.items() if k != '_config_path'}
    require(normalized(cfg) == normalized(expected_cfg), 'effective configuration differs from declared treatment')
    canonical = sorted(s for s, _ in sequences_for_split(Path(cfg['DATA']['ROOT']), 'train', splits_manifest(cfg)))
    require(canonical == sorted(metadata['train_sequences']) and len(canonical) == 72, 'canonical train set differs')
    require({s.split('_')[0] for s in canonical} == {'ch','lfz','lpc','lr','ly','lyh','lyq','ycy','ylf'}, 'train identities differ')
    for name, expected in metadata['source_sha256'].items():
        require(sha(ROOT / name) == expected, f'producer changed: {name}')
    parent_path = Path(metadata['resumed_from'])
    require(sha(parent_path) == metadata['parent_checkpoint_sha256'], 'parent changed')
    checkpoint_path = directory / f'xkf_donly_{arm}_s{seed}-step={end}.ckpt'
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    require(parent['global_step'] == 2200 and saved['global_step'] == end, 'checkpoint clock differs')
    model = XKFRepairModel(cfg=cfg)
    names = [name for name, _ in model.named_parameters()]
    direct = {name for name in names if name.startswith('direct.')}
    require(set(saved['state_dict']) == set(parent['state_dict']), 'state dictionary structure changed')
    changed = []
    for name, value in saved['state_dict'].items():
        require(bool(torch.isfinite(value).all()), f'nonfinite saved tensor {name}')
        if name not in direct:
            require(torch.equal(value, parent['state_dict'][name]), f'I/BN/EMA/frozen buffer changed: {name}')
        elif not torch.equal(value, parent['state_dict'][name]):
            changed.append(name)
    require(len(changed) == len(direct), 'not every declared D parameter updated')
    old_opt, new_opt = parent['optimizer_states'][0], saved['optimizer_states'][0]
    old_ids, new_ids = [o['param_groups'][0]['params'] for o in (old_opt, new_opt)]
    require(old_ids == new_ids and len(names) == len(old_ids) == 96, 'Adam slots/order changed')
    require(new_opt['param_groups'][0]['lr'] == .0004, 'saved actual Adam LR differs')
    for name, old_id, new_id in zip(names, old_ids, new_ids):
        a, b = old_opt['state'][old_id], new_opt['state'][new_id]
        require(int(b['step']) == (end if name in direct else 2200), f'Adam clock differs: {name}')
        for key in ('exp_avg', 'exp_avg_sq'):
            require(b[key].shape == saved['state_dict'][name].shape and bool(torch.isfinite(b[key]).all()), f'Adam shape/finite: {name}')
            if name not in direct:
                require(torch.equal(a[key], b[key]), f'frozen I optimizer changed: {name}')
    require(saved['lr_schedulers'][0]['last_epoch'] == end, 'scheduler clock differs')
    require(saved['lr_schedulers'][0]['base_lrs'] == [.0004], 'scheduler base LR differs')
    initial = directory / 'initial_state.ckpt'
    require(sha(initial) == metadata['initial_checkpoint_sha256'] == sha(metadata['original_initial_checkpoint']), 'initial lineage differs')
    require(torch.load(initial, map_location='cpu', weights_only=False)['global_step'] == 0, 'initial is not real random step0')
    for rank in (0, 1):
        audit = read(directory / f'freeze_audit_rank{rank}.json')
        require(audit['status'] == 'PASS' and audit['global_step'] == end, 'rank freeze audit failed')
        require(audit['main_supervised_rows'] == steps * 512 and audit['i_forward_rows'] == 0 and audit['burn_model_rows'] == 0, 'training exposure differs')
        stream = read(directory / f'pair_stream_rank{rank}.json')
        require([v['optimizer_step'] for v in stream] == list(range(2200, end)), 'sample counters missing or duplicated')
        require(all(v['rank'] == rank and v['rows'] == 512 for v in stream), 'rank exposure differs')
    del model, saved, parent
    return {'checkpoint': str(checkpoint_path), 'checkpoint_sha256': sha(checkpoint_path),
            'metadata_sha256': sha(directory / 'training_metadata.json'), 'config_sha256': sha(config),
            'named_adam_slots': len(names), 'changed_direct_parameters': changed, 'rows': steps * 1024,
            'i_clock': 2200, 'd_clock': end, 'status': 'PASS'}


def validate_pair(aug, raw, steps):
    counts, differences = 0, 0
    for rank in (0, 1):
        astream = read(aug / f'pair_stream_rank{rank}.json')
        rstream = read(raw / f'pair_stream_rank{rank}.json')
        for a, r in zip(astream, rstream):
            for key in ('optimizer_step', 'rank', 'rows', 'sampling_identity_sha256', 'lnes_window_sha256'):
                require(a[key] == r[key], f'paired stream differs: {key}')
            differences += a['main_lnes_sha256'] != r['main_lnes_sha256']
        with (aug / f'sample_identities_rank{rank}.jsonl').open() as af, (raw / f'sample_identities_rank{rank}.jsonl').open() as rf:
            for step, (aa, rr) in enumerate(itertools.zip_longest(af, rf)):
                require(aa is not None and rr is not None, 'paired stream length differs')
                a, r = json.loads(aa), json.loads(rr)
                require(a == r and a['optimizer_step'] == 2200 + step, 'actual complete paired sampling identities differ')
                require(len(a['samples']) == 512, 'wrong sample count')
                require([s['global_row'] for s in a['samples']] == list(range(rank, 1024, 2)), 'global rows differ')
                for sample in a['samples']:
                    require(sample['sample_index'] == a['optimizer_step'] * 1024 + sample['global_row'], 'sample identity/counter differs')
                counts += len(a['samples'])
    require(counts == steps * 1024 and differences == steps * 2, 'missing paired evidence or identical augmentation')
    return {'paired_rows': counts, 'different_lnes_batches': differences, 'status': 'PASS'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT / 'outputs/xkf/fusion_20261015_r3/direct_finetune')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(4)
    seeds, steps, prefix = ([3407], 2, 'smoke_') if args.smoke else ([3407,3408,3409], 200, '')
    result = {'status': 'PASS', 'scope': 'implementation/provenance only, no effect pass implied',
              'seeds': {}, 'validator_sha256': sha(__file__), 'formal': False}
    for seed in seeds:
        directories = {arm: args.root / f'{prefix}{arm}_s{seed}' for arm in ('aug', 'raw')}
        result['seeds'][str(seed)] = {arm: validate_arm(d, seed, steps) for arm, d in directories.items()}
        result['seeds'][str(seed)]['pair'] = validate_pair(directories['aug'], directories['raw'], steps)
    output = args.root / ('independent_smoke_audit.json' if args.smoke else 'independent_training_audit.json')
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'status': result['status']}))


if __name__ == '__main__':
    main()
