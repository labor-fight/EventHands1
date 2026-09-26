"""Complete, committed-step recovery for this repository's deterministic-index datasets.

The sampler cursor counts batches actually consumed by training, never worker prefetch.
Lightning still owns model/optimizer/scheduler/precision/loop serialization. Grid saves
are deferred until the next batch starts, after the previous loop progress is committed.
This mode deliberately rejects old checkpoints and incompatible execution contracts.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import uuid
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import RandomSampler, Sampler
from lightning_fabric.utilities.distributed import DistributedSamplerWrapper


LEGACY_BRANCH_MODE = 'NONEXACT_LEGACY_OPTIMIZER_BRANCH'


def legacy_branch_provenance(path, checkpoint, seed):
    """Identify an explicitly new trajectory; the legacy source is only ever read."""
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('branch-seed must be an integer in [0, 2**32)')
    if CompleteCheckpoint.KEY in checkpoint:
        raise ValueError('branch-from requires a legacy checkpoint; use --resume for complete checkpoints')
    required = ('state_dict', 'optimizer_states', 'lr_schedulers', 'global_step', 'epoch', 'loops')
    if any(key not in checkpoint for key in required) or len(checkpoint['optimizer_states']) != 1:
        raise ValueError('Legacy optimizer branch requires one optimizer and complete Lightning training state')
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(mode=LEGACY_BRANCH_MODE, source_path=str(Path(path).resolve()),
                source_sha256=digest.hexdigest(), source_global_step=int(checkpoint['global_step']),
                source_epoch=int(checkpoint['epoch']), branch_seed=seed, data_epoch_origin=0,
                data_batch_origin=0, total_batch_origin=int(checkpoint['global_step']),
                rng_policy='same branch_seed on every rank; dataset SEED unchanged')


def reserve_branch_directory(directory, provenance):
    """Reserve a fresh directory once; Lightning's child scripts inherit its token."""
    directory = Path(directory)
    marker = directory / 'legacy_branch_origin.json'
    env_key = 'SEMKINE_LEGACY_BRANCH_LAUNCH'
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        directory.mkdir(parents=True, exist_ok=False)
        token = uuid.uuid4().hex
        marker.write_text(json.dumps(dict(token=token, provenance=provenance), indent=2))
        os.environ[env_key] = token
    else:
        token = os.environ.get(env_key)
        expected = dict(token=token, provenance=provenance)
        if token is None or not marker.exists() or json.loads(marker.read_text()) != expected:
            raise ValueError('DDP branch child does not own this fresh output directory')


class CommittedRandomSampler(Sampler):
    """Native PyTorch shuffled order with a cursor advanced only by the training callback."""

    def __init__(self, dataset, seed: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.rank, self.world_size, self.epoch = 0, 1, 0
        self.consumed = 0
        self.order = None

    def configure(self, rank, world_size):
        if self.order is not None and (rank, world_size) != (self.rank, self.world_size):
            raise ValueError('Cannot change sampler rank/world size after initialization')
        self.rank, self.world_size = int(rank), int(world_size)

    def __len__(self):
        return (len(self.dataset) + self.world_size - 1) // self.world_size

    def set_epoch(self, epoch):
        if int(epoch) != self.epoch:
            self.epoch, self.consumed, self.order = int(epoch), 0, None

    def __iter__(self):
        if self.order is None:
            if self.world_size == 1:
                # Match the ordinary DataLoader(shuffle=True) sampler, including its
                # one draw from the main torch RNG to seed its private generator.
                native = RandomSampler(self.dataset)
            else:
                # Lightning 1.9 wraps the loader's RandomSampler before sharding;
                # preserve that original two-stage order and its main-RNG draw.
                native = DistributedSamplerWrapper(RandomSampler(self.dataset),
                            num_replicas=self.world_size, rank=self.rank, seed=self.seed, shuffle=True)
                native.set_epoch(self.epoch)
            self.order = torch.tensor(list(native), dtype=torch.int64)
        # Freeze start for this iterator. Prefetch may consume this iterator far ahead;
        # only commit() changes the checkpoint's cursor.
        return iter(self.order[self.consumed:].tolist())

    def commit(self, count):
        if self.order is None or self.consumed + int(count) > len(self.order):
            raise RuntimeError('Consumed batch exceeds the frozen sampler order')
        self.consumed += int(count)

    def state_dict(self):
        return dict(epoch=self.epoch, consumed=self.consumed, order=self.order,
                    rank=self.rank, world_size=self.world_size, seed=self.seed,
                    dataset_length=len(self.dataset))

    def load_state_dict(self, state):
        expected = (self.rank, self.world_size, self.seed, len(self.dataset))
        actual = tuple(state[k] for k in ('rank', 'world_size', 'seed', 'dataset_length'))
        if actual != expected:
            raise ValueError('Sampler recovery contract changed')
        order = state['order']
        if order is None or order.dtype != torch.int64 or order.numel() != len(self):
            raise ValueError('Missing or malformed sampler order')
        if order.numel() and (order.min() < 0 or order.max() >= len(self.dataset)):
            raise ValueError('Sampler order contains out-of-range sample IDs')
        if not 0 <= state['consumed'] <= len(self):
            raise ValueError('Invalid committed sampler cursor')
        self.epoch, self.consumed = int(state['epoch']), int(state['consumed'])
        self.order = order.cpu().clone()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if not torch.cuda.is_initialized():
            raise ValueError('CUDA RNG checkpoint cannot resume on a CPU execution')
        torch.cuda.set_rng_state(state['cuda'].cpu())


class CompleteCheckpoint(pl.Callback):
    """Fixed optimizer-step grid, per-rank RNG/sampler, fail-closed resume contracts.

    Scope: deterministic(seed,index) map datasets, one train loader, fixed world size,
    automatic optimization, completed optimizer steps. Unfinished accumulated gradients
    are not serialized, and arbitrary exception checkpoints are not certified complete.
    """

    KEY = 'semkine_complete_resume'
    VERSION = 1

    def __init__(self, sampler, directory, run_name, every_n_steps, batch_size, contract,
                 legacy_branch=None):
        self.sampler = sampler
        self.directory = Path(directory)
        self.run_name = run_name
        self.every = int(every_n_steps)
        self.batch_size = int(batch_size)
        self.contract = copy.deepcopy(contract)
        self.legacy_branch = copy.deepcopy(legacy_branch)
        self._start_legacy_branch = False
        if legacy_branch is not None and self.contract.get('legacy_branch') != legacy_branch:
            raise ValueError('Legacy branch provenance must be pinned in the recovery contract')
        self.last_model_path = ''
        self._last_saved = -1
        self._pending = False
        self._last_step = 0
        self._boundary = True
        self._restore = None
        self._restore_epoch = None
        self._saving = False
        self._stop_validation_rng = None
        if self.every <= 0 or self.batch_size <= 0:
            raise ValueError('Positive checkpoint period and batch size required')
        if pl.__version__ != '1.9.5':
            raise RuntimeError('Complete loop-progress recovery is validated for Lightning 1.9.5 only')

    def setup(self, trainer, pl_module, stage):
        if stage != 'fit':
            return
        self.sampler.configure(trainer.global_rank, trainer.world_size)
        if trainer.world_size != self.contract['world_size']:
            raise ValueError('Recovery world_size differs from execution contract')
        if not pl_module.automatic_optimization:
            raise ValueError('Complete recovery requires automatic optimization')
        if self.contract.get('legacy_branch') is not None and trainer.accumulate_grad_batches != 1:
            raise ValueError('Legacy optimizer branches require accumulate_grad_batches=1')
        if trainer.state._fault_tolerant_mode.is_enabled:
            raise ValueError('Do not combine committed sampling with Lightning experimental FT')

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        state = checkpoint.get(self.KEY)
        if self.legacy_branch is not None:
            if state is not None:
                raise ValueError('branch-from cannot load a complete checkpoint; use --resume')
            observed = legacy_branch_provenance(self.legacy_branch['source_path'], checkpoint,
                                                self.legacy_branch['branch_seed'])
            if observed != self.legacy_branch:
                raise ValueError('Legacy branch source changed after its provenance was registered')
            self._load_legacy_optimizer_branch(trainer, checkpoint)
            return
        if state is None or state.get('version') != self.VERSION:
            raise ValueError('Checkpoint lacks complete RNG/sampler recovery; use explicit legacy resume or warm start')
        if state['contract'] != self.contract:
            changed = sorted(k for k in set(state['contract']) | set(self.contract)
                             if state['contract'].get(k) != self.contract.get(k))
            raise ValueError('Complete recovery contract changed: '+', '.join(changed))
        if state.get('legacy_branch') != self.contract.get('legacy_branch'):
            raise ValueError('Complete checkpoint legacy branch provenance changed')
        if not state['optimizer_boundary'] or len(state['ranks']) != trainer.world_size:
            raise ValueError('Checkpoint is not a complete optimizer-boundary state')
        rank_state = state['ranks'][trainer.global_rank]
        self.sampler.load_state_dict(rank_state['sampler'])
        self._restore = rank_state['rng']
        self._restore_epoch = self.sampler.epoch
        self._last_saved = int(checkpoint['global_step'])
        self._last_step = int(checkpoint['global_step'])

    def _load_legacy_optimizer_branch(self, trainer, checkpoint):
        # Begin a new data epoch while retaining the old optimizer's cumulative
        # step count. Fresh loop state also drops stale validation/batch markers.
        source = checkpoint['loops']['fit_loop']
        key = 'epoch_loop.batch_loop.optimizer_loop.optim_progress'
        optimization = copy.deepcopy(source[key])
        step = optimization['optimizer']['step']['total']
        if step['completed'] != checkpoint['global_step'] or step['ready'] != step['completed']:
            raise ValueError('Legacy source is not at a completed optimizer boundary')
        if source['epoch_loop.batch_progress']['total']['processed'] != checkpoint['global_step']:
            raise ValueError('Legacy optimizer branch requires one completed optimizer update per consumed batch')
        for progress in optimization['optimizer'].values():
            completed = progress['total']['completed']
            progress['total'] = {name: completed for name in progress['total']}
            progress['current'] = {name: 0 for name in progress['current']}
        optimization['optimizer_position'] = 0
        fresh = trainer.fit_loop.state_dict()
        fresh[key] = optimization
        total = fresh['epoch_loop.batch_progress']['total']
        for name in total:
            total[name] = int(checkpoint['global_step'])
        fresh['epoch_loop.state_dict']['_batches_that_stepped'] = int(checkpoint['global_step'])
        checkpoint['loops']['fit_loop'] = fresh
        checkpoint['epoch'] = 0
        self._last_step = int(checkpoint['global_step'])
        self._last_saved = self._last_step
        self._start_legacy_branch = True

    def on_train_start(self, trainer, pl_module):
        self._last_step = trainer.global_step
        if self._start_legacy_branch:
            # Re-seed after model/optimizer/device initialization and before loader
            # iteration. Equal per-rank seeds preserve the shared DDP permutation.
            pl.seed_everything(self.legacy_branch['branch_seed'], workers=True)
            self._start_legacy_branch = False

    def on_train_epoch_start(self, trainer, pl_module):
        if self._restore is not None and self.sampler.epoch != self._restore_epoch:
            # A completed-epoch last.ckpt first closes that epoch without updates.
            # Restore before generating the next epoch's native shuffle order.
            restore_rng(self._restore)
            self._restore = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Construction, iterator seeding and worker startup may consume RNG. Restore
        # after these and before this model's next stochastic training operation.
        if self._restore is not None:
            restore_rng(self._restore)
            self._restore = None
        if self._pending:
            self._save(trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.sampler.commit(self.batch_size)  # this entry point uses drop_last=True
        self._boundary = trainer.global_step > self._last_step
        self._last_step = trainer.global_step
        if self._boundary and trainer.global_step % self.every == 0:
            self._pending = True

    def on_train_end(self, trainer, pl_module):
        if self._boundary and trainer.global_step > 0:
            self._save(trainer, final=True)

    def on_validation_start(self, trainer, pl_module):
        # should_stop forces an extra validation outside the scheduled cadence.
        # It must not perturb the training RNG of an otherwise identical continuation.
        loop = trainer.fit_loop.epoch_loop
        iteration = loop.total_batch_idx if trainer.check_val_every_n_epoch is None else loop.batch_idx
        if trainer.should_stop and (iteration + 1) % trainer.val_check_batch != 0:
            self._stop_validation_rng = rng_state()

    def on_validation_end(self, trainer, pl_module):
        if self._stop_validation_rng is not None:
            restore_rng(self._stop_validation_rng)
            self._stop_validation_rng = None

    def _save(self, trainer, final=False):
        if not self._boundary:
            raise RuntimeError('Refusing to certify unfinished accumulated gradients')
        self.directory.mkdir(parents=True, exist_ok=True)
        self._saving = True
        try:
            if trainer.global_step != self._last_saved and trainer.global_step % self.every == 0:
                path = self.directory / f'{self.run_name}-step={trainer.global_step}.ckpt'
                trainer.save_checkpoint(str(path))
                self._last_saved = trainer.global_step
            self.last_model_path = str(self.directory / 'last.ckpt')
            trainer.save_checkpoint(self.last_model_path)
        finally:
            self._saving = False
        self._pending = False

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if not self._saving:
            raise RuntimeError('Use the complete fixed-grid callback to save at a certified boundary')
        rank_state = dict(sampler=self.sampler.state_dict(), rng=rng_state())
        states = [None] * trainer.world_size
        if trainer.world_size > 1:
            torch.distributed.all_gather_object(states, rank_state)
        else:
            states[0] = rank_state
        checkpoint[self.KEY] = dict(version=self.VERSION, contract=self.contract,
                                    optimizer_boundary=True, ranks=states)
        if self.contract.get('legacy_branch') is not None:
            checkpoint[self.KEY]['legacy_branch'] = copy.deepcopy(self.contract['legacy_branch'])
        # Lightning has already fetched (but not trained) the next batch when a
        # deferred grid is saved. Its ready/is_last_batch markers describe that
        # uncommitted batch and must not make restore skip it or shift validation.
        fit = checkpoint['loops']['fit_loop']
        progress = fit['epoch_loop.batch_progress']
        for scope in ('total', 'current'):
            completed = progress[scope]['completed']
            for key in ('ready', 'started', 'processed'):
                progress[scope][key] = completed
        full_epoch = self.sampler.consumed >= (len(self.sampler)//self.batch_size)*self.batch_size
        progress['is_last_batch'] = full_epoch
        # Resume through the actual data epoch. If fully consumed, Lightning will
        # close it without updates once, then generate the following permutation.
        for scope in ('total', 'current'):
            fit['epoch_progress'][scope].update(ready=self.sampler.epoch+1,
                started=self.sampler.epoch+1, processed=self.sampler.epoch, completed=self.sampler.epoch)
        checkpoint['epoch'] = self.sampler.epoch
