#!/usr/bin/env python3
"""Nonformal paired XKF repair continuation; original trainer files stay frozen.

Full Lightning resume preserves Adam moments, scheduler/global counter clocks and
gain statistics. Both paired arms explicitly use the configured reduced learning
rate AFTER checkpoint restoration; YAML alone cannot change a resumed optimizer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'model')]

import torch


def override_learning_rate(trainer, rate):
    """Preserve optimizer state and scheduler clock, replace their rate bases."""
    if not 0 < rate < float('inf'):
        raise ValueError('finite positive continuation learning rate required')
    before = [[float(group['lr']) for group in optimizer.param_groups]
              for optimizer in trainer.optimizers]
    if not before:
        raise RuntimeError('optimizer is not initialized')
    for optimizer in trainer.optimizers:
        for group in optimizer.param_groups:
            group['lr'] = rate
            group['initial_lr'] = rate
    schedulers = []
    for configuration in trainer.lr_scheduler_configs:
        scheduler = configuration.scheduler
        if not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
            raise TypeError('repair continuation requires the original LambdaLR scheduler')
        scheduler.base_lrs = [rate for _ in scheduler.optimizer.param_groups]
        scheduler._last_lr = list(scheduler.base_lrs)
        schedulers.append({'class': type(scheduler).__name__, 'last_epoch': scheduler.last_epoch,
                           'base_lrs': list(scheduler.base_lrs),
                           'last_lr': list(scheduler.get_last_lr())})
    after = [[float(group['lr']) for group in optimizer.param_groups]
             for optimizer in trainer.optimizers]
    return {'configured_lr': rate, 'actual_before': before, 'actual_after': after,
            'schedulers': schedulers, 'adam_state_and_scheduler_clock_preserved': True}


def find_original_initial(resumed_from):
    """Follow real checkpoint lineage; never relabel the warm-start weights step0."""
    directory = Path(resumed_from).resolve().parent
    visited = set()
    while directory not in visited:
        visited.add(directory)
        initial = directory / 'initial_state.ckpt'
        if initial.is_file():
            saved = torch.load(initial, map_location='cpu', weights_only=False)
            if int(saved.get('global_step', -1)) != 0:
                raise ValueError(f'initial checkpoint is not actual step0: {initial}')
            return initial
        metadata = directory / 'training_metadata.json'
        if not metadata.is_file():
            break
        earlier = json.loads(metadata.read_text()).get('resumed_from')
        if not earlier:
            break
        directory = Path(earlier).resolve().parent
    raise FileNotFoundError('cannot resolve actual original random initial checkpoint')


def main():
    # Keep importing this module for unit tests free of the canonical trainer's
    # module rebinding. The bindings below exist only in the executable wrapper.
    from config import load_config
    # Load the repair class first; its disabled path also follows the stable
    # class MRO rather than the legacy wrapper's rebound module export.
    from semkine.xkf_repair import XKFRepairModel
    import tools.xkf.train as original
    import tools.xdl.train as T
    import semkine.xyz_recovery_study as S
    from tools.xkf_repair.data import build_train_dataset, collate_motion
    inheritance = XKFRepairModel.__mro__
    if (inheritance[1] is not original.XKFModel
            or inheritance[2] is not original.XKFModel.__mro__[1]
            or inheritance[2].__name__ != 'RecoveryXYZModel'):
        raise RuntimeError('repair inheritance differs from the original stable XYZ class chain')

    check = argparse.ArgumentParser(add_help=False)
    check.add_argument('--config', required=True)
    check.add_argument('--resume')
    check.add_argument('--preflight', action='store_true')
    check.add_argument('--inspect-data', action='store_true')
    check.add_argument('--frozen-manifest')
    args, _ = check.parse_known_args()
    if not args.preflight:
        raise ValueError('XKF repair wrapper prohibits formal training; --preflight required')
    if not args.inspect_data and (not args.resume or not args.frozen_manifest):
        raise ValueError('paired repair continuation requires --resume and --frozen-manifest')
    cfg = load_config(args.config)
    train = cfg['TRAIN']
    if train.get('LR_SCHEDULE', 'constant') != 'constant':
        raise ValueError('paired repair continuation currently requires constant LR')
    initial_source = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        start_step = int(checkpoint.get('global_step', -1))
        if start_step < 2000 or start_step < int(train.get('WARMUP_STEPS', 0)):
            raise ValueError('repair continuation must begin at/after step2000 and completed warmup')
        if int(checkpoint.get('hyper_parameters', {}).get('SEED', -1)) != int(cfg['SEED']):
            raise ValueError('resume checkpoint seed differs from config')
        del checkpoint
        initial_source = find_original_initial(args.resume)

    class RepairProvenance(original.XKFProvenance):
        def on_fit_start(self, trainer, pl_module):
            super().on_fit_start(trainer, pl_module)
            if trainer.global_rank != 0:
                return
            metadata_path = self.output_dir / 'training_metadata.json'
            metadata = json.loads(metadata_path.read_text())
            if initial_source is None:
                raise RuntimeError('missing original initial checkpoint')
            target = self.output_dir / 'initial_state.ckpt'
            digest = T.sha(initial_source)
            if target.exists():
                if T.sha(target) != digest:
                    raise RuntimeError('output initial checkpoint differs from actual lineage')
            else:
                temporary = target.with_suffix('.ckpt.tmp')
                shutil.copyfile(initial_source, temporary)
                if T.sha(temporary) != digest:
                    raise RuntimeError('initial checkpoint copy failed hash verification')
                temporary.replace(target)
            metadata.update(
                study='XKF-REPAIR-20261015-r2-paired-continuation',
                selection_policy='paired nonformal fixed endpoint; no bestseed or perseed checkpoint selection',
                formal_training_prohibited=True,
                original_initial_checkpoint=str(initial_source),
                initial_checkpoint_sha256=digest,
                optimizer_restore_contract='full resume; preserve Adam moments and clocks; override rate after restoration',
                learning_rate_override_pending=True,
                learning_rate_override_hook='on_train_start (Lightning1.9 restores optimizer AFTER on_fit_start)',
                motion_target_contract='training-only dict: transformed current GT minus transformed previous GT, additive51',
                loss_contract='control: original XKF loss; repair: declared opt-in motion/auxiliary losses; rank-local512 then DDPmean',
            )
            T.atomic_json(metadata_path, metadata)

        def on_train_start(self, trainer, pl_module):
            # Lightning1.9 on_fit_start precedes restore_training_state. This hook
            # runs after restoration and before the first resumed update.
            rate = float(pl_module.cfg['TRAIN']['LR'])
            audit = override_learning_rate(trainer, rate)
            audit.update(global_step=int(trainer.global_step), rank=int(trainer.global_rank),
                         phase='after_restore_before_first_update')
            T.atomic_json(self.output_dir / f'learning_rate_override_rank{trainer.global_rank}.json', audit)
            if trainer.global_rank == 0:
                path = self.output_dir / 'training_metadata.json'
                metadata = json.loads(path.read_text())
                metadata['learning_rate_override_pending'] = False
                metadata['learning_rate_override'] = audit
                T.atomic_json(path, metadata)

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            rate = float(pl_module.cfg['TRAIN']['LR'])
            if any(float(group['lr']) != rate for optimizer in trainer.optimizers
                   for group in optimizer.param_groups):
                raise RuntimeError('restored optimizer/scheduler lost the configured paired learning rate')
            super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    T.build_train_dataset = build_train_dataset
    T.collate_recovery = collate_motion
    T.RecoveryProvenance = RepairProvenance
    T.SOURCE_FILES = tuple(T.SOURCE_FILES) + (
        'tools/xkf_repair/data.py', 'tools/xkf_repair/train.py',
        'semkine/xkf_repair.py', 'semkine/xkf_repair_losses.py')
    S.RecoveryXYZModel = XKFRepairModel
    T.main()


if __name__ == '__main__':
    main()
