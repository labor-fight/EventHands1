#!/usr/bin/env python3
"""Prepared joint continuation: default prepare-only / 200-update debug.

--execute --mode debug runs exactly global2600->2800. Full execution requires
a new three-seed release+manifest and runs global2600->6200 (I/D2400->6000).
Neither this file nor preparing its config starts a job automatically.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import yaml
from config import load_config
from tools.xdl.data import packet_hash, rank_sampler
from tools.xkf_fusion.motion_data import MotionDataset, arm_config, collate_incremental
from tools.xkf_fusion.incremental_checkpoint import validate_incremental_endpoint
from tools.xkf_fusion.finetune_motion import atomic_json, tensor_hash, optimizer_named_state, source_hashes as previous_sources
from tools.xkf_repair.train import override_learning_rate, find_original_initial
from tools.xkf_joint.model import JointFrozenStatisticsModel, ORIGIN_GLOBAL, ORIGIN_PARAMETER, FULL_GLOBAL, RATE, POLICY
from tools.xkf_joint.release import validate_release, sha, require

NEW_SOURCES = ('tools/xkf_joint/model.py', 'tools/xkf_joint/train.py', 'tools/xkf_joint/release.py',
               'tools/xkf_joint/verify.py', 'tools/xkf_fusion/incremental_checkpoint.py', 'tests/test_xkf_joint.py')


def sources():
    return previous_sources() | {name: sha(ROOT / name) for name in NEW_SOURCES}


def all_buffer_hashes(model):
    # Includes six nonpersistent topology/atlas buffers as well as saved BN/EMA.
    return {name: tensor_hash(value) for name, value in model.named_buffers()}


def parent_audit(model, saved, path):
    clocks = validate_incremental_endpoint(model, saved, path)
    metadata = json.loads((path.parent / 'training_metadata.json').read_text())
    require(metadata['arm'] == 'raw', 'joint origin must be the declared raw I-only endpoint')
    require(path.name == f'xkf_ionly_raw_s{model.cfg["SEED"]}-step=2600.ckpt', 'wrong joint parent filename')
    model.load_state_dict(saved['state_dict'], strict=True)
    model.parent_incremental_continuation = copy.deepcopy(saved['xkf_incremental_continuation'])
    optimizer = model.configure_optimizers()['optimizer']
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer_states'][0]))
    order = [name for name, _ in model.named_parameters()]
    require(len(order) == 96 and all(p.requires_grad for p in model.parameters()), 'all original96 parameters must train')
    for name, parameter in model.named_parameters():
        require(int(optimizer.state[parameter]['step']) == ORIGIN_PARAMETER, 'joint parent Adam clock differs: ' + name)
    return dict(parent_clocks=clocks, parameter_order=order,
                optimizer_state=optimizer_named_state(model, optimizer),
                all_buffers_sha256=all_buffer_hashes(model))


class JointAudit(pl.Callback):
    def __init__(self, output, metadata, restored, initial, release):
        self.output, self.metadata, self.restored = output, metadata, restored
        self.initial, self.release = initial, release
        self.records, self.losses = [], []
        self.started = time.monotonic()

    def on_fit_start(self, trainer, model):
        if trainer.global_rank == 0:
            target = self.output / 'initial_state.ckpt'
            if target.exists():
                require(sha(target) == sha(self.initial), 'true initial copy changed')
            else:
                shutil.copyfile(self.initial, target)
            self.metadata.update(original_initial_checkpoint=str(self.initial), initial_checkpoint_sha256=sha(target))
            atomic_json(self.output / 'training_metadata.json', self.metadata)

    def on_train_start(self, trainer, model):
        require(trainer.global_step == ORIGIN_GLOBAL and trainer.world_size == 2, 'wrong actual resume/world size')
        lr = override_learning_rate(trainer, RATE)
        require(all(c.scheduler.last_epoch == ORIGIN_GLOBAL for c in trainer.lr_scheduler_configs), 'wrong resumed scheduler clock')
        require(optimizer_named_state(model, trainer.optimizers[0]) == self.restored['optimizer_state'], 'actual named Adam restore differs')
        self.buffers = all_buffer_hashes(model)
        require(self.buffers == self.restored['all_buffers_sha256'], 'actual resumed buffers differ')
        self.before = {name: tensor_hash(p) for name, p in model.named_parameters()}
        atomic_json(self.output / f'resume_audit_rank{trainer.global_rank}.json',
                    dict(self.restored, learning_rate=lr, actual_global_step=int(trainer.global_step)))

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        rank, step = int(trainer.global_rank), int(trainer.global_step)
        require(bool((batch['optimizer_step'] == step).all()), 'sampling/optimizer clock differs')
        require(batch['main'].batch_size == 512 and torch.equal(batch['global_row'].cpu(), torch.arange(rank, 1024, 2)), 'rank512 global rows differ')
        require(all(g['lr'] == RATE for g in trainer.optimizers[0].param_groups), 'actual LR differs')
        require(not any(m.training for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)), 'BN entered train mode')
        rows = batch['sampling_identity']
        self.records.append(dict(optimizer_step=step, rank=rank, rows=len(rows), main_packet_sha256=packet_hash(batch['main']),
                                 main_lnes_sha256=tensor_hash(batch['main'].lnes), lnes_window_sha256=tensor_hash(batch['main'].lnes_window_ms)))
        with (self.output / f'sample_identities_rank{rank}.jsonl').open('a') as f:
            f.write(json.dumps(dict(optimizer_step=step, samples=rows), separators=(',', ':')) + '\n')

    def on_before_optimizer_step(self, trainer, model, optimizer, opt_idx):
        for name, parameter in model.named_parameters():
            require(parameter.requires_grad and parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
                    'missing or nonfinite joint gradient: ' + name)

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        value = float(outputs['loss'].detach().float().cpu())
        require(bool(torch.isfinite(torch.tensor(value))), 'nonfinite joint batch loss')
        self.losses.append(dict(global_step=int(trainer.global_step), rank_local_log10_I_plus_D_loss=value))
        if trainer.global_rank == 0:
            atomic_json(self.output / 'progress.json', dict(self.losses[-1], max_steps=trainer.max_steps,
                        completed_updates=int(trainer.global_step) - ORIGIN_GLOBAL, elapsed_s=time.monotonic() - self.started))
            if int(trainer.global_step) % 20 == 0:
                print(json.dumps(dict(step=int(trainer.global_step), max_steps=trainer.max_steps, loss=value)), flush=True)

    def on_fit_end(self, trainer, model):
        step, rank = int(trainer.global_step), int(trainer.global_rank)
        updates = step - ORIGIN_GLOBAL
        require(all_buffer_hashes(model) == self.buffers, 'BN/residual/static buffers changed')
        require(len(self.records) == updates and model.xdl_self_rows > 0 and model.xdl_burn_rows == 3 * 512 * updates, 'main/self/burn exposure incomplete')
        require(sources() == self.metadata['source_sha256'] and sha(self.metadata['resumed_from']) == self.metadata['parent_checkpoint_sha256'], 'training source/parent changed')
        changed = [name for name, p in model.named_parameters() if tensor_hash(p) != self.before[name]]
        require(any(n.startswith('direct.') for n in changed) and any(not n.startswith('direct.') for n in changed), 'both branches must update')
        for name, parameter in model.named_parameters():
            require(int(trainer.optimizers[0].state[parameter]['step']) == ORIGIN_PARAMETER + updates, 'joint Adam clock differs: ' + name)
        require(all(c.scheduler.last_epoch == step for c in trainer.lr_scheduler_configs), 'final scheduler clock differs')
        if self.release is not None:
            for name, digest in self.release['evidence_sha256'].items():
                require(sha(name) == digest, 'release evidence changed during full run')
            require(sha(self.release['release']) == self.release['release_sha256']
                    and sha(self.release['manifest']) == self.release['manifest_sha256'], 'release/manifest changed')
        atomic_json(self.output / f'freeze_audit_rank{rank}.json', dict(status='PASS', global_step=step,
                    i_adam_step=ORIGIN_PARAMETER + updates, d_adam_step=ORIGIN_PARAMETER + updates,
                    changed_parameters=changed, all_named_buffers_bitwise_unchanged=True,
                    all_named_buffer_sha256=self.buffers, source_sha256=self.metadata['source_sha256']))
        atomic_json(self.output / f'actual_exposure_rank{rank}.json', dict(final_global_step=step,
                    main_supervised_rows=updates * 512, self_main_rows=int(model.xdl_self_rows),
                    burn_model_rows=int(model.xdl_burn_rows), burn_count_attribute_present=True))
        atomic_json(self.output / f'pair_stream_rank{rank}.json', self.records)
        atomic_json(self.output / f'losses_rank{rank}.json', self.losses)
        trainer.strategy.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True, choices=(3407, 3408, 3409))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('debug', 'full'), default='debug')
    parser.add_argument('--execute', action='store_true', help='otherwise prepare configs and identity only')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--frozen-manifest', type=Path)
    parser.add_argument('--num-workers', type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = args.output_dir.resolve()
    endpoint = ORIGIN_GLOBAL + 200 if args.mode == 'debug' else FULL_GLOBAL
    parent = ROOT / f'outputs/xkf/fusion_20261015_r3/direct_finetune/incremental_finetune/raw_s{args.seed}/xkf_ionly_raw_s{args.seed}-step=2600.ckpt'
    cfg = arm_config(load_config(str(ROOT / f'configs/xkf_repair/repair_s{args.seed}.yaml')), 'raw')
    cfg['TRAIN'].update(MAX_STEPS=endpoint, LR=RATE, OUTPUT_DIR=str(output), NUM_WORKERS=args.num_workers,
                        RUN_NAME=f'xkf_joint_{args.mode}_s{args.seed}')
    config_text = yaml.safe_dump({k: v for k, v in cfg.items() if k != '_config_path'}, sort_keys=False)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'config_resolved.yaml'
    if config_path.exists():
        require(config_path.read_text() == config_text, 'existing prepared config differs')
    elif not args.execute or int(os.environ.get('LOCAL_RANK', 0)) == 0:
        config_path.write_text(config_text)
    release = None
    if args.execute and args.mode == 'full':
        release = validate_release(args.release, args.frozen_manifest, ROOT, config_path, parent, args.seed)
    source = sources()
    if release is not None:
        require(all(release['manifest_source_sha256'].get(name) == digest for name, digest in source.items()),
                'full release manifest does not cover the actual complete training source closure')
    pl.seed_everything(args.seed, workers=True)
    model = JointFrozenStatisticsModel(cfg)
    saved = torch.load(parent, map_location='cpu', weights_only=False)
    restored = parent_audit(model, saved, parent)
    del saved
    initial = find_original_initial(parent)
    metadata = dict(kind='XKF_JOINT_FROZEN_STATISTICS_CONTINUATION', seed=args.seed, mode=args.mode,
                    formal=False, original_global6000_protocol=False, automatic_full_training=False,
                    start_step=ORIGIN_GLOBAL, max_steps=endpoint, updates=endpoint - ORIGIN_GLOBAL,
                    origin_i_adam_step=ORIGIN_PARAMETER, origin_d_adam_step=ORIGIN_PARAMETER,
                    end_i_adam_step=ORIGIN_PARAMETER + endpoint - ORIGIN_GLOBAL,
                    end_d_adam_step=ORIGIN_PARAMETER + endpoint - ORIGIN_GLOBAL,
                    resumed_from=str(parent), parent_checkpoint_sha256=sha(parent), source_sha256=source,
                    config_sha256=sha(config_path), effective_batch=1024, devices=2, batch_size_per_gpu=512,
                    lr=RATE, precision='bf16', freeze='ALL named buffers including nonpersistent topology/atlas; BN and residual moments fixed',
                    training='raw canonical9/72 K4; original Gaussian+self-burn; joint log10(LI)+log10(LD), motion target/FP32/Daux .5',
                    self_burn_policy='original fused; no external deployment policy', deployment_policy=POLICY,
                    selection='unique declared endpoint per seed; no best checkpoint selection', release=release)
    if not args.execute:
        atomic_json(output / 'prepare.json', dict(metadata, status='PREPARED_NO_TRAINING',
                    full_execution_blocked_without_new_release=True, parent_named_Adam_restored=True))
        print(json.dumps(dict(status='PREPARED_NO_TRAINING', output=str(output), target_global=endpoint,
                              target_I_D=ORIGIN_PARAMETER + endpoint - ORIGIN_GLOBAL)), flush=True)
        return
    cards = [int(v) for v in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if v]
    require(len(cards) == 2 and len(set(cards)) == 2 and set(cards) <= set(range(2, 8)), 'requires exactly two authorized GPU2-7 cards')
    rank = int(os.environ.get('LOCAL_RANK', 0))
    require(not (output / 'train_finish.json').exists() and not (output / f'sample_identities_rank{rank}.jsonl').exists(), 'refusing to overwrite/append an executed run')
    dataset = MotionDataset(cfg, start_step=ORIGIN_GLOBAL, max_steps=endpoint)
    metadata.update(train_sequences=[name for name, _ in dataset.sequences], coverage=dataset.coverage)
    loader = DataLoader(dataset, batch_size=512, sampler=rank_sampler(dataset, replicas=2, rank=rank),
                        shuffle=False, drop_last=True, num_workers=args.num_workers,
                        pin_memory=args.num_workers > 0, persistent_workers=args.num_workers > 0,
                        prefetch_factor=1 if args.num_workers > 0 else None, collate_fn=collate_incremental,
                        generator=torch.Generator().manual_seed(args.seed))
    callback = JointAudit(output, metadata, restored, initial, release)
    saver = pl.callbacks.ModelCheckpoint(dirpath=str(output), filename=cfg['TRAIN']['RUN_NAME'] + '-{step}',
                                        monitor=None, save_top_k=-1, save_last=False,
                                        every_n_train_steps=endpoint, save_on_train_epoch_end=False)
    trainer = pl.Trainer(accelerator='gpu', devices=2, strategy='ddp_find_unused_parameters_false',
                         precision='bf16', max_steps=endpoint, accumulate_grad_batches=1,
                         callbacks=[callback, saver], logger=False, enable_progress_bar=False,
                         num_sanity_val_steps=0, limit_val_batches=0, log_every_n_steps=1,
                         default_root_dir=str(output), sync_batchnorm=False, enable_model_summary=False)
    trainer.fit(model, loader, ckpt_path=str(parent))
    if trainer.global_rank == 0:
        final = output / f'{cfg["TRAIN"]["RUN_NAME"]}-step={endpoint}.ckpt'
        saved = torch.load(final, map_location='cpu', weights_only=False)
        require(saved['global_step'] == trainer.global_step == endpoint, 'saved final global step differs')
        require({k: tensor_hash(v) for k, v in saved['state_dict'].items()} == {k: tensor_hash(v) for k, v in model.state_dict().items()}, 'saved state differs from audited final model')
        require(all(not value.is_floating_point() or bool(torch.isfinite(value).all()) for value in saved['state_dict'].values()), 'saved state is nonfinite')
        saved_optimizer = saved['optimizer_states'][0]
        identifiers = saved_optimizer['param_groups'][0]['params']
        require(len(identifiers) == len(saved_optimizer['state']) == 96, 'saved joint Adam coverage differs')
        expected_clock = ORIGIN_PARAMETER + endpoint - ORIGIN_GLOBAL
        require(saved_optimizer['param_groups'][0]['lr'] == RATE, 'saved actual joint LR differs')
        for (name, parameter), identifier in zip(model.named_parameters(), identifiers):
            slot = saved_optimizer['state'][identifier]
            require(int(slot['step']) == expected_clock, 'saved joint Adam clock differs: ' + name)
            for field in ('exp_avg', 'exp_avg_sq'):
                require(slot[field].shape == parameter.shape and bool(torch.isfinite(slot[field]).all()),
                        'saved joint Adam shape/numeric differs: ' + name)
                require(torch.equal(slot[field], trainer.optimizers[0].state[parameter][field].detach().cpu()),
                        'saved joint Adam differs from actual audited optimizer: ' + name)
        require(len(saved['lr_schedulers']) == 1 and saved['lr_schedulers'][0]['last_epoch'] == endpoint
                and saved['lr_schedulers'][0]['base_lrs'] == saved['lr_schedulers'][0]['_last_lr'] == [RATE],
                'saved joint scheduler clock/LR differs')
        for rank in (0, 1):
            require((output / f'freeze_audit_rank{rank}.json').exists(), 'missing rank final audit')
        atomic_json(output / 'train_finish.json', dict(status='JOINT_DEBUG200_COMPLETE' if args.mode == 'debug' else 'JOINT_GLOBAL6200_PARAMETER6000_COMPLETE',
                    formal=False, original_global6000_protocol=False, global_step=endpoint,
                    i_adam_step=ORIGIN_PARAMETER + endpoint - ORIGIN_GLOBAL, d_adam_step=ORIGIN_PARAMETER + endpoint - ORIGIN_GLOBAL,
                    checkpoint=str(final), checkpoint_sha256=sha(final), parent_checkpoint_sha256=sha(parent),
                    source_sha256=source, release=release))


if __name__ == '__main__':
    main()
