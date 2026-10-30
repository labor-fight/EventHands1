"""Read-only identity checks for the fixed I-only debug continuation endpoint."""
from pathlib import Path
import hashlib
import json
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_incremental_endpoint(model, saved, checkpoint_path):
    path = Path(checkpoint_path).resolve()
    metadata_path, finish_path = path.parent / 'training_metadata.json', path.parent / 'train_finish.json'
    metadata = json.loads(metadata_path.read_text())
    finish = json.loads(finish_path.read_text())
    require(saved['global_step'] == 2600 and metadata['start_step'] == 2400
            and metadata['max_steps'] == 2600 and metadata['updates'] == 200,
            'expected exact 200-update I-only continuation, global2600')
    require(metadata['smoke_only'] is False and metadata['formal'] is False
            and finish['status'] == 'DEBUG200_COMPLETE' and finish['global_step'] == 2600,
            'partial/smoke/formal endpoint is not this paired debug experiment')
    require(finish['checkpoint_sha256'] == sha(path), 'completed checkpoint changed')
    seed = saved['hyper_parameters']['SEED']
    require(seed == metadata['seed'] == model.cfg['SEED'], 'checkpoint seed mismatch')
    parent_path = Path(metadata['resumed_from']).resolve()
    require(sha(parent_path) == metadata['parent_checkpoint_sha256'] == finish['parent_checkpoint_sha256'],
            'raw D-only parent identity mismatch')
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    require(parent['global_step'] == 2400 and parent['hyper_parameters']['SEED'] == seed,
            'wrong parent step/seed')
    prior = parent.get('xkf_direct_continuation', {})
    clock = saved.get('xkf_incremental_continuation', {})
    require(prior.get('i_adam_step') == 2200 and prior.get('d_adam_step') == 2400,
            'wrong parent parameter clocks')
    require(clock.get('formal') is False and clock.get('start_step') == 2400
            and clock.get('i_adam_step') == 2400 and clock.get('d_adam_step') == 2400
            and clock.get('all_bn_and_xkf_statistics_frozen') is True
            and clock.get('parent_direct_continuation') == prior,
            'I-only parameter clock declaration differs from parent')
    require(not saved.get('ema') and 'raw_state_dict' not in saved, 'unexpected weight EMA')
    names = list(model.named_parameters())
    require(len(names) == 96 and sum(name.startswith('direct.') for name, _ in names) == 60,
            'original 96-parameter model identity changed')
    require(set(saved['state_dict']) == set(parent['state_dict']) == set(model.state_dict()),
            'checkpoint tensor keys changed')
    buffers = {name for name, _ in model.named_buffers()}
    frozen = buffers | {name for name, _ in names if name.startswith('direct.')}
    changed_i = []
    for name, value in saved['state_dict'].items():
        require(value.shape == parent['state_dict'][name].shape, 'tensor shape changed: ' + name)
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()),
                'nonfinite checkpoint tensor: ' + name)
        same = torch.equal(value, parent['state_dict'][name])
        if name in frozen:
            require(same, 'frozen D parameter or buffer changed: ' + name)
        elif not same:
            changed_i.append(name)
    require(bool(changed_i), 'no I parameter changed')
    require(len(saved['optimizer_states']) == len(parent['optimizer_states']) == 1,
            'expected one complete original Adam optimizer')
    current_opt, parent_opt = saved['optimizer_states'][0], parent['optimizer_states'][0]
    require(len(current_opt['param_groups']) == len(parent_opt['param_groups']) == 1,
            'original single Adam group changed')
    ids, old_ids = current_opt['param_groups'][0]['params'], parent_opt['param_groups'][0]['params']
    require(ids == old_ids and len(ids) == len(current_opt['state']) == len(parent_opt['state']) == 96,
            'original Adam slot order or coverage changed')
    for (name, parameter), key in zip(names, ids):
        current, old = current_opt['state'][key], parent_opt['state'][key]
        require(set(current) == set(old) and int(current['step']) == 2400,
                'wrong endpoint Adam fields/clock: ' + name)
        require(int(old['step']) == (2400 if name.startswith('direct.') else 2200),
                'wrong parent Adam clock: ' + name)
        for field in ('exp_avg', 'exp_avg_sq'):
            require(current[field].shape == parameter.shape and bool(torch.isfinite(current[field]).all()),
                    'Adam tensor shape/numeric failure: ' + name + '/' + field)
        if name.startswith('direct.'):
            for field, value in current.items():
                require(torch.equal(value, old[field]) if torch.is_tensor(value) else value == old[field],
                        'frozen D Adam slot changed: ' + name + '/' + field)
    for rank in (0, 1):
        audit = json.loads((path.parent / f'freeze_audit_rank{rank}.json').read_text())
        exposure = json.loads((path.parent / f'actual_exposure_rank{rank}.json').read_text())
        require(audit['status'] == 'PASS' and audit['global_step'] == 2600
                and audit['i_adam_step'] == audit['d_adam_step'] == 2400,
                'missing matching per-rank endpoint audit')
        require(exposure['final_global_step'] == 2600 and exposure['main_supervised_rows'] == 102400
                and exposure['self_main_rows'] > 0 and exposure['burn_model_rows'] > 0,
                'missing actual per-rank main/self/burn exposure')
    return dict(clock, global_step=2600, all_parameters_at_global_step=False,
                original_named_adam_slots_verified=96, changed_i_parameters=len(changed_i),
                frozen_d_parameters=60, all_buffers_and_d_adam_bitwise_unchanged=True,
                parent_checkpoint=str(parent_path), parent_checkpoint_sha256=sha(parent_path),
                training_metadata_sha256=sha(metadata_path), train_finish_sha256=sha(finish_path))
