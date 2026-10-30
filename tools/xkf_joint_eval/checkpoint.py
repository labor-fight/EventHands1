"""Verify real joint-continuation clocks without relabelling the old protocol."""
from pathlib import Path
import hashlib
import json
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_joint_endpoint(model, saved, checkpoint_path):
    path = Path(checkpoint_path).resolve()
    metadata_path, finish_path = path.parent/'training_metadata.json', path.parent/'train_finish.json'
    metadata, finish = json.loads(metadata_path.read_text()), json.loads(finish_path.read_text())
    step = int(saved['global_step'])
    require(step in (2800, 6200), 'only declared joint debug2800/full6200 endpoints are supported')
    updates, parameter_step = step-2600, step-200
    mode = 'debug' if step == 2800 else 'full'
    expected = 'JOINT_DEBUG200_COMPLETE' if mode == 'debug' else 'JOINT_GLOBAL6200_PARAMETER6000_COMPLETE'
    require(metadata['kind'] == 'XKF_JOINT_FROZEN_STATISTICS_CONTINUATION'
            and metadata['mode'] == mode and metadata['start_step'] == 2600
            and metadata['max_steps'] == step and metadata['updates'] == updates,
            'joint endpoint declaration differs')
    require(metadata['formal'] is False and metadata['original_global6000_protocol'] is False
            and finish['status'] == expected and finish['global_step'] == step
            and finish['checkpoint_sha256'] == sha(path), 'incomplete or mislabelled joint endpoint')
    require(metadata['origin_i_adam_step'] == metadata['origin_d_adam_step'] == 2400
            and metadata['end_i_adam_step'] == metadata['end_d_adam_step'] == parameter_step,
            'joint metadata parameter clocks differ')
    seed = saved['hyper_parameters']['SEED']
    require(seed == model.cfg['SEED'] == metadata['seed'], 'joint seed differs')
    require(sha(path.parent/'config_resolved.yaml') == metadata['config_sha256'],
            'joint resolved training config changed')
    current_config = {k: v for k, v in model.cfg.items() if k != '_config_path'}
    saved_config = {k: v for k, v in saved['hyper_parameters'].items() if k != '_config_path'}
    require(current_config == saved_config, 'joint inference configuration differs from saved training configuration')
    parent_path = Path(metadata['resumed_from']).resolve()
    require(parent_path.name == f'xkf_ionly_raw_s{seed}-step=2600.ckpt'
            and sha(parent_path) == metadata['parent_checkpoint_sha256'] == finish['parent_checkpoint_sha256'],
            'joint parent differs from declared raw I2600 endpoint')
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    from tools.xkf_fusion.incremental_checkpoint import validate_incremental_endpoint
    validate_incremental_endpoint(model, parent, parent_path)
    clock = saved.get('xkf_joint_continuation', {})
    require(clock.get('origin_global_step') == 2600 and clock.get('origin_parameter_step') == 2400
            and clock.get('global_step') == step and clock.get('i_adam_step') == clock.get('d_adam_step') == parameter_step
            and clock.get('is_original_global6000_protocol') is False
            and clock.get('all_bn_and_xkf_statistics_frozen') is True
            and clock.get('parent_incremental_continuation') == parent['xkf_incremental_continuation'],
            'joint checkpoint clocks or lineage differ')
    require(finish['i_adam_step'] == finish['d_adam_step'] == parameter_step, 'finish parameter clocks differ')
    require(not saved.get('ema') and 'raw_state_dict' not in saved, 'unexpected weight EMA')
    names = list(model.named_parameters())
    require(len(names) == 96 and sum(name.startswith('direct.') for name, _ in names) == 60,
            'original parameter identity differs')
    state, previous = saved['state_dict'], parent['state_dict']
    require(set(state) == set(previous) == set(model.state_dict()), 'joint state structure changed')
    buffers = {name for name, _ in model.named_buffers()}
    changed = []
    for name, value in state.items():
        require(value.shape == previous[name].shape and
                (not value.is_floating_point() or bool(torch.isfinite(value).all())), 'invalid joint state '+name)
        equal = torch.equal(value, previous[name])
        if name in buffers:
            require(equal, 'frozen joint buffer changed '+name)
        elif not equal:
            changed.append(name)
    require(any(name.startswith('direct.') for name in changed)
            and any(not name.startswith('direct.') for name in changed), 'both branches must actually update')
    require(len(saved['optimizer_states']) == len(parent['optimizer_states']) == 1, 'wrong joint optimizer count')
    optimizer, prior = saved['optimizer_states'][0], parent['optimizer_states'][0]
    require(len(optimizer['param_groups']) == len(prior['param_groups']) == 1, 'wrong joint optimizer groups')
    ids = optimizer['param_groups'][0]['params']
    require(ids == prior['param_groups'][0]['params'] and len(ids) == len(optimizer['state']) == 96,
            'joint Adam slot identity/order differs')
    require(optimizer['param_groups'][0]['lr'] == .0004, 'actual joint learning rate differs')
    for (name, parameter), key in zip(names, ids):
        current, old = optimizer['state'][key], prior['state'][key]
        require(set(current) == set(old) and int(current['step']) == parameter_step and int(old['step']) == 2400,
                'joint Adam clock/fields differ '+name)
        for field in ('exp_avg', 'exp_avg_sq'):
            require(current[field].shape == parameter.shape and bool(torch.isfinite(current[field]).all()),
                    'joint Adam tensor invalid '+name+'/'+field)
    require(len(saved['lr_schedulers']) == 1 and saved['lr_schedulers'][0]['last_epoch'] == step
            and saved['lr_schedulers'][0]['base_lrs'] == saved['lr_schedulers'][0]['_last_lr'] == [.0004],
            'joint scheduler clock/rate differs')
    for rank in (0, 1):
        audit = json.loads((path.parent/f'freeze_audit_rank{rank}.json').read_text())
        exposure = json.loads((path.parent/f'actual_exposure_rank{rank}.json').read_text())
        require(audit['status'] == 'PASS' and audit['global_step'] == step
                and audit['i_adam_step'] == audit['d_adam_step'] == parameter_step
                and audit['all_named_buffers_bitwise_unchanged'] is True, 'rank joint audit differs')
        require(exposure['final_global_step'] == step and exposure['main_supervised_rows'] == updates*512
                and exposure['burn_model_rows'] == updates*512*3 and exposure['self_main_rows'] > 0,
                'rank joint exposure incomplete')
    source = metadata['source_sha256']
    root = Path(__file__).resolve().parents[2]
    require(source == finish['source_sha256'], 'joint source records differ')
    for name, digest in source.items():
        require(sha(root/name) == digest, 'joint training source changed '+name)
    if mode == 'full':
        release = metadata.get('release')
        require(release is not None and release == finish['release'], 'full continuation lacks matching release')
        from tools.xkf_joint.release import validate_release
        require(validate_release(release['release'], release['manifest'], root,
                                 path.parent/'config_resolved.yaml', parent_path, seed) == release,
                'full continuation release no longer validates')
    return dict(global_step=step, i_adam_step=parameter_step, d_adam_step=parameter_step,
                origin_global_step=2600, origin_i_d_steps=2400, updates=updates,
                original_global6000_protocol=False, all_parameters_at_global_step=False,
                original_named_adam_slots_verified=96, frozen_buffers_bitwise_parent=True,
                changed_parameters=changed, parent_checkpoint=str(parent_path), parent_checkpoint_sha256=sha(parent_path),
                training_metadata_sha256=sha(metadata_path), train_finish_sha256=sha(finish_path))
