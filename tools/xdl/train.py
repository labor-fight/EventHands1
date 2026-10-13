#!/usr/bin/env python3
"""Isolated exact-last XDL trainer with deterministic raw K4 packet sampling.

Formal launches preserve two ranks x 512 samples, without microbatch log10
averaging. The preflight flag allows short explicitly nonformal health runs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'model'))

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from config import load_config
from semkine import train as canonical
from tools.xdl.data import build_train_dataset, collate_recovery, packet_hash, rank_sampler

SOURCE_FILES = ('tools/xdl/data.py', 'tools/xdl/train.py', 'tools/xdl/make_configs.py',
                'semkine/xyz_observation.py', 'semkine/xyz_recovery_study.py',
                'model/model.py', 'semkine/xyz_mesh.py', 'semkine/event_gnn.py',
                'semkine/dataset.py', 'semkine/events.py', 'model/mano_layer.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_frozen_manifest(path):
    """Root creates a mapping of relative/absolute paths to SHA256 before launch."""
    values = json.loads(Path(path).read_text())
    entries = values.get('source_sha256', values.get('sha256', values))
    if not isinstance(entries, dict) or not entries:
        raise ValueError('freeze manifest must contain a nonempty path-to-SHA256 mapping')
    for name, expected in entries.items():
        p = Path(name)
        if not p.is_absolute():
            p = REPO / p
        if not isinstance(expected, str) or sha(p) != expected:
            raise RuntimeError(f'frozen source/config/split identity mismatch: {name}')
    return sha(path)


def make_loader(cfg, *, start_step=0, max_steps=None, batch_size=None, workers=None, distributed=False):
    ds = build_train_dataset(cfg, start_step=start_step, max_steps=max_steps)
    train = cfg['TRAIN']
    batch_size = int(batch_size if batch_size is not None else train['BATCH_SIZE_PER_GPU'])
    workers = int(workers if workers is not None else train['NUM_WORKERS'])
    sampler = (rank_sampler(ds, replicas=int(train['DEVICES']),
                            rank=int(os.environ.get('LOCAL_RANK', '0')))
               if distributed else None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, sampler=sampler, drop_last=True,
                        num_workers=workers, pin_memory=workers > 0,
                        persistent_workers=workers > 0,
                        prefetch_factor=int(train.get('PREFETCH_FACTOR', 1)) if workers > 0 else None,
                        collate_fn=collate_recovery,
                        generator=torch.Generator().manual_seed(int(cfg['SEED'])))
    return ds, loader


class RecoveryProvenance(canonical.pl.Callback):
    def __init__(self, output_dir, *, preflight=False):
        self.output_dir = Path(output_dir)
        self.preflight = preflight
        self.stream_rows = []
        self.started = None
        self.last_progress = -1
        self.main_rows = 0
        self.main_events = 0
        self.prefix_events = 0
        self.nonfinite = 0

    def on_fit_start(self, trainer, pl_module):
        self.started = time.monotonic()
        if trainer.global_rank == 0:
            metadata = json.loads((self.output_dir / 'training_metadata.json').read_text())
            metadata['source_sha256'] = {p: sha(REPO / p) for p in SOURCE_FILES if (REPO / p).is_file()}
            metadata['model_class'] = type(pl_module).__name__
            metadata['cuda_devices_on_rank0'] = torch.cuda.device_count()
            metadata['started_time'] = dt.datetime.now(dt.timezone.utc).isoformat()
            atomic_json(self.output_dir / 'training_metadata.json', metadata)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        expected = int(trainer.global_step)
        if not bool((batch['optimizer_step'] == expected).all()):
            raise RuntimeError(f'counter step differs from optimizer step {expected}: {batch["optimizer_step"].tolist()}')
        rows = batch['global_row']
        world, rank = int(trainer.world_size), int(trainer.global_rank)
        if len(rows.unique()) != rows.numel() or not bool((rows % world == rank).all()):
            raise RuntimeError('DDP counter row mapping does not equal rank-strided global rows')
        self.main_rows += batch['main'].batch_size
        self.main_events += int(batch['main'].events.shape[0])
        self.prefix_events += sum(int(p.events.shape[0]) for p in batch['prefix'])
        if len(self.stream_rows) < 3:
            self.stream_rows.append({'optimizer_step': expected, 'rank': rank,
                                     'global_rows': rows.detach().cpu().tolist(),
                                     'sample_index_sha256': hashlib.sha256(batch['sample_index'].detach().cpu().numpy().tobytes()).hexdigest(),
                                     'main_sha256': packet_hash(batch['main']),
                                     'prefix_sha256': [packet_hash(p) for p in batch['prefix']],
                                     'main_rows': batch['main'].batch_size,
                                     'main_events': int(batch['main'].events.shape[0])})
            atomic_json(self.output_dir / f'pair_stream_rank{rank}.json', self.stream_rows)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
        finite = bool(torch.isfinite(loss).all()) if torch.is_tensor(loss) else False
        if not finite:
            self.nonfinite += 1
            atomic_json(self.output_dir / f'nonfinite_rank{trainer.global_rank}.json',
                        {'status': 'FAIL', 'optimizer_step': int(trainer.global_step),
                         'nonfinite_batches': self.nonfinite})
            raise FloatingPointError('nonfinite XDL batch loss; run remains failed')
        step = int(trainer.global_step)
        if trainer.global_rank != 0 or step == self.last_progress or (step % 100 and not self.preflight):
            return
        self.last_progress = step
        elapsed = time.monotonic() - self.started
        progress = {'time_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                    'global_step': step, 'max_steps': int(trainer.max_steps),
                    'elapsed_s': elapsed, 'optimizer_steps_per_second': step / max(elapsed, 1e-9),
                    'latest_batch_log10_loss': float(loss.detach().float().cpu()),
                    'main_rows_rank0': self.main_rows, 'main_events_rank0': self.main_events,
                    'prefix_data_events_rank0': self.prefix_events,
                    'max_memory_allocated_gb_rank0': torch.cuda.max_memory_allocated() / 2**30,
                    'preflight_not_formal': self.preflight}
        atomic_json(self.output_dir / 'progress.json', progress)
        print(f'XDL {self.output_dir.name}: {step}/{trainer.max_steps}, elapsed={elapsed:.1f}s, loss={progress["latest_batch_log10_loss"]}', flush=True)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx):
        # A finite loss can still have invalid derivatives. In particular the
        # final update must not save invalid weights and claim step6000 success.
        gradients = [(name, parameter.grad) for name, parameter in pl_module.named_parameters()
                     if parameter.requires_grad and parameter.grad is not None]
        if gradients and not bool(torch.stack([torch.isfinite(grad).all() for _, grad in gradients]).all()):
            bad = [name for name, grad in gradients if not bool(torch.isfinite(grad).all())]
            atomic_json(self.output_dir / f'nonfinite_gradient_rank{trainer.global_rank}.json',
                        {'status': 'FAIL', 'optimizer_step': int(trainer.global_step),
                         'parameters': bad})
            raise FloatingPointError('nonfinite XDL gradient; optimizer update and completion rejected')

    def on_fit_end(self, trainer, pl_module):
        rank = int(trainer.global_rank)
        atomic_json(self.output_dir / f'actual_exposure_rank{rank}.json', {
            'main_supervised_rows': self.main_rows, 'main_events': self.main_events,
            'prefix_data_events': self.prefix_events,
            'burn_model_rows': int(getattr(pl_module, 'xdl_burn_rows', 0)),
            'self_main_rows': int(getattr(pl_module, 'xdl_self_rows', 0)),
            'burn_count_attribute_present': hasattr(pl_module, 'xdl_burn_rows'),
            'elapsed_s': time.monotonic() - self.started,
            'final_global_step': int(trainer.global_step),
            'preflight_not_formal': self.preflight})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--devices', type=int, default=None)
    ap.add_argument('--num-workers', type=int, default=None)
    ap.add_argument('--output-dir')
    ap.add_argument('--run-name')
    ap.add_argument('--max-steps', type=int)
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--inspect-data', action='store_true')
    ap.add_argument('--resume')
    ap.add_argument('--frozen-manifest')
    ap.add_argument('--no-logger', action='store_true')
    args = ap.parse_args()
    cfg = load_config(args.config)
    train = cfg['TRAIN']
    if args.output_dir:
        train['OUTPUT_DIR'] = str(Path(args.output_dir).resolve())
    if args.run_name:
        train['RUN_NAME'] = args.run_name
    if args.devices:
        train['DEVICES'] = int(args.devices)
    if args.num_workers is not None:
        train['NUM_WORKERS'] = int(args.num_workers)
    if args.max_steps is not None:
        if not args.preflight and args.max_steps != 6000:
            raise ValueError('short max_steps is only allowed for --preflight, never a formal result')
        train['MAX_STEPS'] = int(args.max_steps)
    max_steps = int(train['MAX_STEPS'])
    devices = int(train['DEVICES'])
    if devices != 2 or int(train['BATCH_SIZE_PER_GPU']) != 512 or int(train.get('ACCUMULATE_GRAD_BATCHES', 1)) != 1:
        raise ValueError('XDL preserves two GPU ranks x512 with no accumulation because log10 loss is batch nonlinear')
    if not args.preflight and max_steps != 6000:
        raise ValueError('formal XDL max_steps must be 6000')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not args.inspect_data:
        cards = [int(x) for x in visible.split(',') if x.strip()]
        if len(cards) != devices or len(set(cards)) != devices or not set(cards) <= set(range(2, 8)):
            raise RuntimeError(f'XDL requires two authorized physical GPU2-7 cards: {visible!r}')
    freeze_hash = verify_frozen_manifest(args.frozen_manifest) if args.frozen_manifest else None
    if not args.preflight and not args.inspect_data and not args.frozen_manifest:
        raise ValueError('formal XDL launch requires --frozen-manifest from the root execution review')
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        start_step = int(checkpoint['global_step'])
        if not 0 <= start_step < max_steps:
            raise ValueError('resume checkpoint global_step is outside unfinished run')
    canonical.pl.seed_everything(int(cfg['SEED']), workers=True)
    ds, loader = make_loader(cfg, start_step=start_step, max_steps=max_steps,
                             batch_size=4 if args.inspect_data else None,
                             workers=0 if args.inspect_data else None,
                             distributed=not args.inspect_data)
    output = Path(train['OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        'study': 'XD-V2-RAW-RECOVERY-20261013', 'run_name': train['RUN_NAME'],
        'seed': cfg['SEED'], 'config_path': str(Path(args.config).resolve()),
        'config_sha256': sha(args.config), 'preflight_not_formal': args.preflight,
        'selection_policy': 'unique exact step6000 checkpoint only; no intermediate selection',
        'train_sequences': [s for s, _ in ds.sequences],
        'train_subjects': sorted({s.split('_')[0] for s, _ in ds.sequences}),
        'split_manifest': str(Path(cfg['DATA']['ROOT']) / 'splits_semkine.json'),
        'split_sha256': sha(Path(cfg['DATA']['ROOT']) / 'splits_semkine.json'),
        'data_counter_contract': '(seed, optimizer_step_before_update, global_row, stream); DDP shuffle=False rank-stride',
        'paired_main_stream': True, 'coverage': ds.coverage,
        'effective_batch': 1024, 'devices': devices, 'batch_size_per_gpu': 512,
        'accumulate_grad_batches': 1, 'max_steps': max_steps, 'start_step': start_step,
        'resumed_from': args.resume, 'cuda_visible_devices': visible,
        'freeze_manifest_sha256': freeze_hash,
        'lr': train['LR'], 'precision': train['PRECISION'],
        'main_samples': len(ds), 'argv': sys.argv,
        'loss_contract': 'canonical rank-local 512-sample weighted MSE then log10; DDP gradient mean over 2 ranks',
    }
    if args.inspect_data:
        batch = next(iter(loader))
        metadata['inspect_first_four_main_sha256'] = packet_hash(batch['main'])
        atomic_json(output / 'data_inspection.json', metadata)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return
    atomic_json(output / 'training_metadata.json', metadata)
    (output / 'config_resolved.yaml').write_text(yaml.safe_dump(
        {k: v for k, v in cfg.items() if k != '_config_path'}, sort_keys=False))
    from semkine.xyz_recovery_study import RecoveryXYZModel
    model = RecoveryXYZModel(cfg)
    callback = RecoveryProvenance(output, preflight=args.preflight)
    ckpt = canonical.ModelCheckpoint(dirpath=str(output), filename=train['RUN_NAME'] + '-{step}',
                                     monitor=None, save_last=False, save_top_k=-1,
                                     every_n_train_steps=500 if not args.preflight else max_steps,
                                     save_on_train_epoch_end=False)
    logger = False if args.no_logger else canonical.TensorBoardLogger(
        save_dir=str(output), name='logs', version=train['RUN_NAME'])
    trainer = canonical.pl.Trainer(
        accelerator='gpu', devices=devices, strategy='ddp_find_unused_parameters_false',
        precision=train['PRECISION'], max_steps=max_steps, accumulate_grad_batches=1,
        callbacks=[ckpt, callback, canonical.ThroughputCallback()],
        logger=logger, enable_progress_bar=False, num_sanity_val_steps=0,
        limit_val_batches=0, log_every_n_steps=20, default_root_dir=str(output),
        sync_batchnorm=False, enable_model_summary=False)
    trainer.fit(model, loader, ckpt_path=args.resume)
    if trainer.global_rank == 0:
        finals = list(output.glob(f'{train["RUN_NAME"]}-step={max_steps}.ckpt'))
        if int(trainer.global_step) != max_steps or len(finals) != 1:
            raise RuntimeError(f'final identity failure: step={trainer.global_step}, files={finals}')
        loaded = torch.load(finals[0], map_location='cpu')
        if int(loaded['global_step']) != max_steps:
            raise RuntimeError('saved checkpoint step is not the required exact endpoint')
        invalid = [name for name, tensor in loaded['state_dict'].items()
                   if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())]
        if invalid:
            raise FloatingPointError(f'exact endpoint has nonfinite saved tensors: {invalid}')
        atomic_json(output / 'train_finish.json', {
            'status': 'preflight_complete' if args.preflight else 'completed_exact_step6000',
            'global_step': int(trainer.global_step), 'checkpoint': str(finals[0]),
            'checkpoint_sha256': sha(finals[0]), 'preflight_not_formal': args.preflight,
            'config_sha256': sha(output / 'config_resolved.yaml'),
            'completed_utc': dt.datetime.now(dt.timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
