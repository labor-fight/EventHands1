#!/usr/bin/env python3
"""Nonformal paired200 I-only continuation from raw D2400 / I2200.

The original complete Adam parameter order/moments are retained. D parameters, all BN buffers and XKF moments remain bitwise frozen.
Only I receives gradients; original three-packet fused self-burn is retained.
The sole objective is the unchanged R2 actual-motion delta loss.
An explicit --smoke performs two updates in a separate directory, never part of
the fixed 200-step experiment. This executable cannot launch full training.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
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
from semkine.xkf_repair import XKFRepairModel
from tools.xdl.data import packet_hash, rank_sampler
from tools.xkf_fusion.motion_data import MotionDataset, arm_config, collate_incremental
from tools.xkf_repair.train import find_original_initial, override_learning_rate

R2 = ROOT / 'outputs/xkf/repair_20261015_r2'
START, I_START, D_START, STEPS, RATE = 2400, 2200, 2400, 200, .0004
NEW_SOURCES = ('tools/xkf_fusion/direct_data.py', 'tools/xkf_fusion/motion_data.py', 'tools/xkf_fusion/finetune_motion.py', 'tests/test_xkf_motion_finetune.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def source_hashes():
    frozen = json.loads((R2 / 'frozen_manifest.json').read_text())['source_sha256']
    for name, expected in frozen.items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f'R2 frozen source changed: {name}')
    return {**frozen, **{name: sha(ROOT / name) for name in NEW_SOURCES}}


def tensor_hash(tensor):
    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(t.shape), str(t.dtype))).encode())
    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def state_hashes(model, *, frozen_only=False):
    # Every buffer is frozen, as are all D parameters.
    frozen = {name for name, _ in model.named_buffers()}
    frozen.update(name for name, _ in model.named_parameters() if name.startswith('direct.'))
    return {name: tensor_hash(value) for name, value in model.state_dict().items()
            if not frozen_only or name in frozen}


def optimizer_named_state(model, optimizer):
    order = [(name, p) for name, p in model.named_parameters()]
    actual = [p for group in optimizer.param_groups for p in group['params']]
    if len(actual) != len(order) or any(a is not b for a, (_, b) in zip(actual, order)):
        raise RuntimeError('Adam parameter identity/order differs from complete original named_parameters')
    return {name: {key: tensor_hash(value) if torch.is_tensor(value) else value
                   for key, value in optimizer.state[p].items()}
            for name, p in order}


def validate_parent(model, checkpoint):
    if int(checkpoint.get('global_step', -1)) != START:
        raise ValueError('I-only continuation requires the exact raw D2400 / I2200 parent')
    if int(checkpoint['hyper_parameters']['SEED']) != int(model.cfg['SEED']):
        raise ValueError('parent seed differs')
    if 'raw_state_dict' in checkpoint or checkpoint.get('ema'):
        raise ValueError('this continuation requires the EMA-disabled R2 parent')
    if checkpoint['hyper_parameters']['MODEL'].get('XKF_REPAIR_D_AUX_2D_WEIGHT') != .5 or checkpoint['hyper_parameters']['MODEL'].get('XKF_REPAIR_D_AUX_3D_WEIGHT') != .5:
        raise ValueError('parent must be the repair arm with both D auxiliaries .5')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    marker = checkpoint.get('xkf_direct_continuation')
    if not marker or marker.get('i_adam_step') != I_START or marker.get('d_adam_step') != D_START:
        raise ValueError('parent D-only lineage marker missing or invalid')
    model.parent_direct_continuation = copy.deepcopy(marker)
    optimizer = model.configure_optimizers()['optimizer']
    saved = checkpoint['optimizer_states']
    if len(saved) != 1 or len(saved[0]['param_groups']) != 1:
        raise ValueError('exact original single Adam group required')
    optimizer.load_state_dict(copy.deepcopy(saved[0]))
    names = [name for name, _ in model.named_parameters()]
    ids = saved[0]['param_groups'][0]['params']
    if len(ids) != len(names) or len(saved[0]['state']) != len(names):
        raise ValueError('parent must have a populated Adam slot for every original parameter')
    for (name, parameter), key in zip(model.named_parameters(), ids):
        old, restored = saved[0]['state'][key], optimizer.state[parameter]
        if int(old['step']) != (D_START if name.startswith('direct.') else I_START):
            raise ValueError(f'wrong parent Adam clock for {name}')
        for field in ('step', 'exp_avg', 'exp_avg_sq'):
            if not torch.equal(old[field], restored[field]):
                raise ValueError(f'Adam migration changed {name}/{field}')
        if old['exp_avg'].shape != parameter.shape:
            raise ValueError(f'Adam slot shape differs for {name}')
    return {'parameter_order': names, 'optimizer_state': optimizer_named_state(model, optimizer),
            'parameter_count': len(names), 'direct_parameter_count': sum(n.startswith('direct.') for n in names),
            'parent_I_adam_step': I_START, 'parent_D_adam_step': D_START, 'all_named_moments_restored_exactly': True}


class IncrementalFinetuneModel(XKFRepairModel):
    def __init__(self, cfg=None, num_classes=None):
        super().__init__(cfg=cfg, num_classes=num_classes)
        if self.ema_decay != 0 or self.weight_decay != 0:
            raise ValueError('I-only continuation requires original Adam without weight decay or model EMA')
        if self.repair_d_aux_2d_weight != .5 or self.repair_d_aux_3d_weight != .5:
            raise ValueError('both paired arms must retain the .5/.5 R2 direct auxiliaries')
        if not self.repair_motion_target or not self.repair_fp32_add or not self.xdl_self or self.xdl_burn_mode != 'full':
            raise ValueError('I-only stage preserves actual-motion loss, FP32 sum and original full self-burn')
        track=self.cfg['TRACK']
        if track.get('PREV_NOISE_MODE')!='gaussian' or tuple(track.get(k) for k in ('PREV_NOISE_T','PREV_NOISE_R','PREV_NOISE_POSE'))!=(.005,.05,.05):
            raise ValueError('paired I stage keeps original small Gaussian; no mixed-noise confound')
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(not name.startswith('direct.'))
        self.parent_direct_continuation = None
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        # train() is called again by Lightning; freeze BN at every invocation.
        for module in self.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def configure_optimizers(self):
        # Deliberately includes frozen D, in its original order. Do not invoke
        # Base.configure_optimizers: it filters requires_grad and moves the slots.
        optimizer = torch.optim.Adam(list(self.parameters()), lr=RATE)
        warmup = int(self.cfg['TRAIN'].get('WARMUP_STEPS', 500))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: float(step + 1) / warmup if step < warmup else 1.)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def training_step(self, batch, batch_idx):
        main, take = self._burn_main(batch)
        prediction = self.incremental_prediction(main)
        loss, parts = self.incremental_loss(prediction, main, batch)
        loss = loss.float().clamp_min(1e-12).log10()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('nonfinite I-only motion loss')
        self.log('train_loss', loss, sync_dist=True, batch_size=main.batch_size)
        for name, value in parts.items():
            self.log(f'train_{name}', value, sync_dist=True, batch_size=main.batch_size)
        return loss

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint['xkf_incremental_continuation'] = {
            'formal': False, 'start_step': START, 'i_adam_step': I_START + int(self.global_step) - START,
            'd_adam_step': D_START, 'all_bn_and_xkf_statistics_frozen': True,
            'loss': 'actual clean GT motion delta, never endpoint correction',
            'burn': 'unchanged original fused self-burn with frozen D and residual moments',
            'parent_direct_continuation': self.parent_direct_continuation}


class IncrementalAudit(pl.Callback):
    def __init__(self, output, metadata, parent_audit, initial):
        self.output, self.metadata = Path(output), metadata
        self.parent_audit, self.initial = parent_audit, Path(initial)
        self.records, self.losses = [], []
        self.started = time.monotonic()

    def on_fit_start(self, trainer, model):
        if trainer.global_rank == 0:
            target = self.output / 'initial_state.ckpt'
            if target.exists() and sha(target) != sha(self.initial):
                raise RuntimeError('initial lineage changed')
            if not target.exists():
                shutil.copyfile(self.initial, target)
            self.metadata.update(original_initial_checkpoint=str(self.initial),
                                 initial_checkpoint_sha256=sha(self.initial))
            atomic_json(self.output / 'training_metadata.json', self.metadata)

    def on_train_start(self, trainer, model):
        lr = override_learning_rate(trainer, RATE)
        if int(trainer.global_step) != START or trainer.world_size != 2:
            raise RuntimeError('resume global clock/world size differs from paired protocol')
        self.frozen = state_hashes(model, frozen_only=True)
        self.optimizer_before = optimizer_named_state(model, trainer.optimizers[0])
        if self.optimizer_before != self.parent_audit['optimizer_state']:
            raise RuntimeError('actual Lightning-restored named Adam state differs from parent')
        if any(configuration.scheduler.last_epoch != START for configuration in trainer.lr_scheduler_configs):
            raise RuntimeError('scheduler failed to resume at step2400')
        self.all_before = state_hashes(model)
        self.main_events = self.prefix_events = 0
        atomic_json(self.output / f'resume_audit_rank{trainer.global_rank}.json', {
            **self.parent_audit, 'learning_rate': lr, 'frozen_state_sha256': self.frozen,
            'actual_restored_state_matches_parent': True, 'all_buffers_and_d_frozen': True})

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        step, rank = int(trainer.global_step), int(trainer.global_rank)
        if not bool((batch['optimizer_step'] == step).all()):
            raise RuntimeError('sample counter does not match resumed optimizer clock')
        rows = batch['global_row']
        if batch['main'].batch_size != 512 or not torch.equal(rows.cpu(), torch.arange(rank, 1024, 2)):
            raise RuntimeError('each rank must consume the exact original 512 rank-strided rows')
        if any(group['lr'] != RATE for group in trainer.optimizers[0].param_groups):
            raise RuntimeError('actual LR differs from .0004')
        if any(module.training for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)):
            raise RuntimeError('BN unexpectedly entered train mode')
        self.main_events += int(batch['main'].events.shape[0])
        self.prefix_events += sum(int(p.events.shape[0]) for p in batch['prefix'])
        identities = batch['sampling_identity']
        serialized = json.dumps(identities, sort_keys=True, separators=(',', ':')).encode()
        self.records.append({'optimizer_step': step, 'rank': rank, 'rows': len(identities),
                             'sampling_identity_sha256': hashlib.sha256(serialized).hexdigest(),
                             'main_packet_sha256': packet_hash(batch['main']),
                             'main_lnes_sha256': tensor_hash(batch['main'].lnes),
                             'lnes_window_sha256': tensor_hash(batch['main'].lnes_window_ms)})
        with (self.output / f'sample_identities_rank{rank}.jsonl').open('a') as f:
            f.write(json.dumps({'optimizer_step': step, 'samples': identities}, separators=(',', ':')) + '\n')

    def on_before_optimizer_step(self, trainer, model, optimizer, opt_idx):
        for name, parameter in model.named_parameters():
            if not name.startswith('direct.'):
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(f'missing/nonfinite I gradient: {name}')
            elif parameter.grad is not None:
                raise RuntimeError(f'frozen D unexpectedly has a gradient: {name}')

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        loss = float(outputs['loss'].detach().float().cpu())
        if not torch.isfinite(torch.tensor(loss)):
            raise FloatingPointError('nonfinite saved loss')
        self.losses.append({'global_step': int(trainer.global_step), 'rank_local_log10_I_motion_loss': loss})
        if trainer.global_rank == 0:
            progress = {**self.losses[-1], 'max_steps': trainer.max_steps,
                        'elapsed_s': time.monotonic() - self.started,
                        'completed_updates': int(trainer.global_step) - START,
                        'preflight_not_formal': True}
            atomic_json(self.output / 'progress.json', progress)
            print(f'I-only {self.metadata["arm"]} seed{self.metadata["seed"]}: {trainer.global_step}/{trainer.max_steps}, loss={loss:.5f}', flush=True)

    def on_fit_end(self, trainer, model):
        frozen_after = state_hashes(model, frozen_only=True)
        if frozen_after != self.frozen:
            raise RuntimeError('D, BN or another frozen buffer changed')
        optimizer = trainer.optimizers[0]
        after = optimizer_named_state(model, optimizer)
        for name, parameter in model.named_parameters():
            expected = D_START if name.startswith('direct.') else I_START + int(trainer.global_step) - START
            if int(optimizer.state[parameter]['step']) != expected:
                raise RuntimeError(f'wrong final Adam clock for {name}')
            if name.startswith('direct.') and after[name] != self.optimizer_before[name]:
                raise RuntimeError(f'D optimizer state changed: {name}')
        all_after = state_hashes(model)
        changed = [name for name, value in all_after.items() if value != self.all_before[name]]
        if not changed or any(name.startswith('direct.') for name in changed):
            raise RuntimeError('updates must change only I parameters')
        if model.xdl_self_rows <= 0 or model.xdl_burn_rows != 3 * 512 * len(self.records):
            raise RuntimeError('required original full self-burn exposure missing')
        if source_hashes() != self.metadata['source_sha256'] or sha(self.metadata['resumed_from']) != self.metadata['parent_checkpoint_sha256']:
            raise RuntimeError('training sources or parent changed during run')
        rank = int(trainer.global_rank)
        atomic_json(self.output / f'freeze_audit_rank{rank}.json', {
            'status': 'PASS', 'global_step': int(trainer.global_step),
            'frozen_bitwise_equal': True, 'd_adam_moments_bitwise_equal': True,
            'i_adam_step': I_START + int(trainer.global_step) - START, 'd_adam_step': D_START,
            'all_bn_buffers_bitwise_equal': True, 'xkf_moments_bitwise_equal': True,
            'changed_parameters': changed, 'burn_model_rows': int(model.xdl_burn_rows), 'i_forward_rows': len(self.records) * 512 + int(model.xdl_burn_rows),
            'main_supervised_rows': len(self.records) * 512,
            'source_sha256': self.metadata['source_sha256']})
        atomic_json(self.output / f'actual_exposure_rank{rank}.json', {
            'main_supervised_rows': len(self.records) * 512, 'main_events': self.main_events,
            'prefix_data_events': self.prefix_events, 'burn_model_rows': int(model.xdl_burn_rows),
            'self_main_rows': int(model.xdl_self_rows), 'burn_count_attribute_present': True,
            'final_global_step': int(trainer.global_step), 'preflight_not_formal': True})
        atomic_json(self.output / f'pair_stream_rank{rank}.json', self.records)
        atomic_json(self.output / f'losses_rank{rank}.json', self.losses)
        if any(configuration.scheduler.last_epoch != int(trainer.global_step)
               for configuration in trainer.lr_scheduler_configs):
            raise RuntimeError('final scheduler clock differs from global clock')
        trainer.strategy.barrier()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, required=True, choices=(3407, 3408, 3409))
    ap.add_argument('--arm', required=True, choices=('aug', 'raw'))
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--smoke', action='store_true', help='exactly two isolated updates; never counts toward paired200')
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--execute-debug', action='store_true', help='explicit debug execution after paired preflight')
    ap.add_argument('--num-workers', type=int, default=12)
    args = ap.parse_args()
    endpoint = START + (2 if args.smoke else STEPS)
    output = Path(args.output_dir).resolve()
    if args.smoke and 'smoke' not in str(output):
        raise ValueError('--smoke output path must explicitly contain smoke')
    parent = ROOT / f'outputs/xkf/fusion_20261015_r3/direct_finetune/raw_s{args.seed}/xkf_donly_raw_s{args.seed}-step=2400.ckpt'
    cfg_path = ROOT / f'configs/xkf_repair/repair_s{args.seed}.yaml'
    cfg = arm_config(load_config(str(cfg_path)), args.arm)
    cfg['TRAIN'].update(MAX_STEPS=endpoint, LR=RATE, OUTPUT_DIR=str(output), NUM_WORKERS=args.num_workers)
    cfg['TRAIN']['RUN_NAME'] = f'xkf_ionly_{args.arm}_s{args.seed}'
    config_text=yaml.safe_dump({k:v for k,v in cfg.items() if k!='_config_path'},sort_keys=False)
    if cfg['XKF_DATA']['WINDOW_MIN'] != 30 or cfg['XKF_DATA']['WINDOW_MAX'] != 300:
        raise ValueError('canonical W30–300 law must be preserved')
    manifest = source_hashes()
    pl.seed_everything(args.seed, workers=True)
    model = IncrementalFinetuneModel(cfg)
    checkpoint = torch.load(parent, map_location='cpu', weights_only=False)
    parent_audit = validate_parent(model, checkpoint)
    del checkpoint
    initial = find_original_initial(parent)
    metadata = {'study': 'XKF-R3-IONLY-PAIRED200', 'formal': False, 'preflight_not_formal': True,
                'smoke_only': args.smoke, 'automatic_full_training': False,
                'seed': args.seed, 'arm': args.arm, 'start_step': START, 'max_steps': endpoint,
                'updates': endpoint - START, 'resumed_from': str(parent),
                'parent_checkpoint_sha256': sha(parent), 'source_sha256': manifest,
                'source_config_sha256': sha(cfg_path), 'config_sha256': hashlib.sha256(config_text.encode()).hexdigest(), 'effective_batch': 1024, 'devices': 2,
                'batch_size_per_gpu': 512, 'precision': 'bf16', 'lr': RATE,
                'loss_contract': 'log10(original pure actual GT motion-delta weighted MSE); rank512, DDPmean2; no endpoint correction',
                'freeze_contract': 'D parameters and Adam slots, ALL buffers incl BN and XKF moments bitwise fixed',
                'data_contract': 'canonical K4 raw W50 and original small Gaussian/selfburn; all prefix LNES retained; original GT endpoints paired; transformed labels follow shared augmentation',
                'selection_policy': 'fixed endpoint all3 seeds both arms; no best checkpoint or seed selection',
                'optimizer_contract': 'original96 params/order; I2200->2400, D2400 fixed, global/scheduler2400->2600',
                'augmentation_difference': 'raw disables DOMRAND (including hot/dropout), POLARITY_FLIP, PIXEL_POLARITY_SWAP only',
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                'created_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    output.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        atomic_json(output / 'prepare.json', {**metadata, 'parent_audit': parent_audit})
        print(json.dumps({'status': 'PREPARED', 'output': str(output), 'parent_parameters': parent_audit['parameter_count']}))
        return
    if not args.execute_debug:
        raise ValueError('execution requires explicit --execute-debug; use --prepare-only for CPU preflight')
    cards = [int(s) for s in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if s]
    if len(cards) != 2 or len(set(cards)) != 2 or not set(cards) <= set(range(2, 8)):
        raise ValueError('requires exactly two authorized physical GPUs2–7')
    if (output / 'train_finish.json').exists() or (output / f'sample_identities_rank{int(os.environ.get("LOCAL_RANK", 0))}.jsonl').exists():
        raise FileExistsError('refusing to append/overwrite an existing run')
    ds = MotionDataset(cfg, start_step=START, max_steps=endpoint)
    metadata.update(train_sequences=[s for s, _ in ds.sequences], coverage=ds.coverage,
                    split_sha256=sha(Path(cfg['DATA']['ROOT']) / 'splits_semkine.json'))
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        (output / 'config_resolved.yaml').write_text(config_text)
    sampler = rank_sampler(ds, replicas=2, rank=int(os.environ.get('LOCAL_RANK', 0)))
    loader = DataLoader(ds, batch_size=512, sampler=sampler, shuffle=False, drop_last=True,
                        num_workers=args.num_workers, pin_memory=args.num_workers > 0,
                        persistent_workers=args.num_workers > 0,
                        prefetch_factor=1 if args.num_workers > 0 else None,
                        collate_fn=collate_incremental, generator=torch.Generator().manual_seed(args.seed))
    callback = IncrementalAudit(output, metadata, parent_audit, initial)
    saver = pl.callbacks.ModelCheckpoint(dirpath=str(output), filename=cfg['TRAIN']['RUN_NAME'] + '-{step}',
                                         monitor=None, save_last=False, save_top_k=-1,
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
        if int(saved['global_step']) != endpoint or int(trainer.global_step) != endpoint:
            raise RuntimeError('saved endpoint differs')
        if any(t.is_floating_point() and not bool(torch.isfinite(t).all()) for t in saved['state_dict'].values()):
            raise FloatingPointError('saved checkpoint contains nonfinite tensors')
        if {name: tensor_hash(value) for name, value in saved['state_dict'].items()} != state_hashes(model):
            raise RuntimeError('saved state differs from audited final model')
        ids = saved['optimizer_states'][0]['param_groups'][0]['params']
        for (name, _), key in zip(model.named_parameters(), ids):
            if int(saved['optimizer_states'][0]['state'][key]['step']) != (D_START if name.startswith('direct.') else I_START + endpoint - START):
                raise RuntimeError('saved Adam clock differs from audited final optimizer')
        if any(not (output / f'freeze_audit_rank{rank}.json').exists() for rank in (0, 1)):
            raise RuntimeError('missing per-rank final audit')
        atomic_json(output / 'train_finish.json', {'status': 'SMOKE_COMPLETE' if args.smoke else 'DEBUG200_COMPLETE',
                    'formal': False, 'automatic_full_training': False, 'global_step': endpoint,
                    'checkpoint': str(final), 'checkpoint_sha256': sha(final),
                    'parent_checkpoint_sha256': sha(parent), 'source_sha256': manifest,
                    'i_adam_step': I_START + endpoint - START, 'd_adam_step': D_START, 'freeze_audit': 'PASS'})


if __name__ == '__main__':
    main()
