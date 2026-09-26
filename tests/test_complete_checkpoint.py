"""Interrupted-vs-continuous optimization, including shuffled IDs and model RNG."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import random
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from semkine.checkpointing import CommittedRandomSampler, CompleteCheckpoint
from semkine.checkpointing import legacy_branch_provenance, reserve_branch_directory, LEGACY_BRANCH_MODE

torch.set_num_threads(4)


class IndexedData(Dataset):
    def __len__(self):
        return 24

    def __getitem__(self, index):
        rng = np.random.default_rng(391 + index)
        return torch.tensor(index), torch.from_numpy(rng.standard_normal(5).astype('float32'))


class StochasticModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Dropout(.25), torch.nn.Linear(8, 1))
        self.trace = []
        self.val_trace = []

    def training_step(self, batch, batch_idx):
        ids, x = batch
        r = [random.random(), float(np.random.random()), float(torch.rand(()))]
        pred = self.net(x + sum(r)*.01)
        loss = (pred[:, 0] - ids.float()/24).square().mean()
        self.trace.append(dict(epoch=self.current_epoch, batch_idx=batch_idx, step=self.global_step,
                               ids=ids.tolist(), rng=r, loss=loss.detach().item()))
        return loss

    def validation_step(self, batch, batch_idx):
        self.val_trace.append((self.global_step, batch_idx, batch[0].tolist()))
        return self.net(batch[1]).sum()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=.003)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.StepLR(opt, 2, .9), 'interval': 'step'}}


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a == b


def execute(directory, workers, every, accumulate=1, resume=None, stop_at=None):
    pl.seed_everything(191, workers=True)
    ds = IndexedData()
    sampler = CommittedRandomSampler(ds, seed=191)
    contract = dict(world_size=1, batch_size=4, workers=workers, accumulation=accumulate,
                    max_steps=13, dataset='IndexedData-v1', precision='32', seed=191)
    cb = CompleteCheckpoint(sampler, directory, 'fixture', every, 4, contract)
    train = DataLoader(ds, batch_size=4, sampler=sampler, num_workers=workers, drop_last=True,
                       persistent_workers=workers>0, prefetch_factor=2 if workers else None)
    val = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    model = StochasticModel()
    class StopAt(pl.Callback):
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if stop_at is not None and trainer.global_step >= stop_at:
                trainer.should_stop = True
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_steps=13, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                         callbacks=[cb, StopAt()], num_sanity_val_steps=0, val_check_interval=4,
                         check_val_every_n_epoch=None, limit_val_batches=2,
                         accumulate_grad_batches=accumulate, replace_sampler_ddp=False)
    trainer.fit(model, train, val, ckpt_path=resume)
    checkpoint = torch.load(Path(directory)/'last.ckpt', map_location='cpu')
    return model.trace, checkpoint, model.val_trace


@pytest.mark.parametrize('workers,every,accumulate', [(0,5,1),(2,5,1),(0,6,1),(2,4,2)])
def test_full_resume_preserves_optimization_and_samples(tmp_path,workers,every,accumulate):
    trace, final, validation = execute(tmp_path/'reference',workers,every,accumulate)
    checkpoint = tmp_path/'reference'/f'fixture-step={every}.ckpt'
    assert checkpoint.exists()
    resumed, after, resumed_validation = execute(tmp_path/'resume',workers,every,accumulate,str(checkpoint))
    expected = [row for row in trace if row['step'] >= every]
    assert resumed == expected
    assert resumed_validation == [row for row in validation if row[0] > every]
    for key in ('state_dict','optimizer_states','lr_schedulers','epoch','global_step'):
        assert tree_equal(final[key],after[key]), key
    assert tree_equal(final[CompleteCheckpoint.KEY],after[CompleteCheckpoint.KEY])


@pytest.mark.parametrize('stop_at', [5,6])
def test_last_checkpoint_resumes_partial_or_completed_epoch(tmp_path,stop_at):
    trace, final, _ = execute(tmp_path/'reference',2,5)
    execute(tmp_path/'interrupted',2,5,stop_at=stop_at)
    resumed, after, _ = execute(tmp_path/'resume',2,5,resume=str(tmp_path/'interrupted'/'last.ckpt'))
    assert resumed == [row for row in trace if row['step'] >= stop_at]
    for key in ('state_dict','optimizer_states','lr_schedulers','epoch','global_step',CompleteCheckpoint.KEY):
        assert tree_equal(final[key],after[key]),key


def test_prefetch_does_not_advance_committed_cursor():
    sampler = CommittedRandomSampler(IndexedData(),seed=7)
    it = iter(sampler)
    first = [next(it) for _ in range(12)]
    sampler.commit(4)
    state = sampler.state_dict()
    replay = CommittedRandomSampler(IndexedData(),seed=7)
    replay.load_state_dict(state)
    assert list(replay)[:8] == first[4:]


def test_mismatched_sampler_and_legacy_checkpoint_are_rejected():
    sampler=CommittedRandomSampler(IndexedData(),seed=7)
    list(sampler)
    different=CommittedRandomSampler(IndexedData(),seed=8)
    with pytest.raises(ValueError, match='contract changed'):
        different.load_state_dict(sampler.state_dict())
    cb=CompleteCheckpoint(sampler,'.','unused',5,4,dict(world_size=1))
    with pytest.raises(ValueError, match='lacks complete'):
        cb.on_load_checkpoint(None,None,{'state_dict':{}})


def make_legacy_checkpoint(directory):
    """Use the real old grid-save timing, not a hand-built approximation."""
    pl.seed_everything(191, workers=True)
    callback = pl.callbacks.ModelCheckpoint(dirpath=directory, filename='legacy-{step}',
                    every_n_train_steps=3, save_top_k=-1, monitor=None)
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_steps=5, logger=False,
                        callbacks=[callback], enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(StochasticModel(), DataLoader(IndexedData(), batch_size=4, shuffle=True, drop_last=True))
    return Path(directory)/'legacy-step=3.ckpt'


def execute_legacy_branch(directory, source, resume=None, workers=0):
    pl.seed_everything(191, workers=True)
    saved = torch.load(resume or source, map_location='cpu')
    origin = (saved[CompleteCheckpoint.KEY]['legacy_branch'] if resume else
              legacy_branch_provenance(source, saved, 881))
    if not resume:
        reserve_branch_directory(directory, origin)
    sampler = CommittedRandomSampler(IndexedData(), origin['branch_seed'])
    contract = dict(world_size=1, batch_size=4, workers=workers, max_steps=13,
                    accumulate_grad_batches=1, seed=191, legacy_branch=origin)
    callback = CompleteCheckpoint(sampler, directory, 'branch', 5, 4, contract,
                                  legacy_branch=None if resume else origin)
    class AuditFirstStep(pl.Callback):
        first = None
        def on_train_batch_start(self, trainer, model, batch, batch_idx):
            if self.first is None:
                import copy
                self.first = copy.deepcopy(dict(state_dict=model.state_dict(),
                    optimizer_states=[o.state_dict() for o in trainer.optimizers],
                    lr_schedulers=[s.scheduler.state_dict() for s in trainer.lr_scheduler_configs],
                    global_step=trainer.global_step, epoch=trainer.current_epoch, batch_idx=batch_idx))
    audit = AuditFirstStep()
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_steps=13, logger=False,
                        callbacks=[callback, audit], enable_checkpointing=False,
                        enable_progress_bar=False, enable_model_summary=False, replace_sampler_ddp=False,
                        num_sanity_val_steps=0, val_check_interval=4, check_val_every_n_epoch=None,
                        limit_val_batches=2, deterministic=True)
    train = DataLoader(IndexedData(), batch_size=4, sampler=sampler, num_workers=workers,
                       drop_last=True, persistent_workers=workers > 0,
                       prefetch_factor=2 if workers else None)
    model = StochasticModel()
    trainer.fit(model, train, DataLoader(IndexedData(), batch_size=4), ckpt_path=str(resume or source))
    return model, torch.load(Path(directory)/'last.ckpt', map_location='cpu'), audit.first


@pytest.mark.parametrize('workers', [0, 2])
def test_legacy_branch_preserves_optimizer_then_resumes_exactly(tmp_path, workers):
    import hashlib
    source = make_legacy_checkpoint(tmp_path/'legacy')
    original_bytes = source.read_bytes()
    original = torch.load(source, map_location='cpu')
    model, final, first = execute_legacy_branch(tmp_path/'branch', source, workers=workers)
    for key in ('state_dict', 'optimizer_states', 'lr_schedulers', 'global_step'):
        assert tree_equal(first[key], original[key]), key
    assert first['epoch'] == first['batch_idx'] == 0
    assert first['global_step'] == 3
    assert model.trace[0]['step'] == 3
    # Preserve the original absolute validation cadence despite starting data epoch 0.
    assert sorted(set(row[0] for row in model.val_trace)) == [4, 8, 12]
    provenance = final[CompleteCheckpoint.KEY]['legacy_branch']
    assert provenance['mode'] == LEGACY_BRANCH_MODE
    assert provenance['source_sha256'] == hashlib.sha256(original_bytes).hexdigest()
    assert provenance['source_global_step'] == 3
    assert provenance['source_epoch'] == original['epoch']
    assert provenance['branch_seed'] == 881
    assert provenance['total_batch_origin'] == 3
    assert source.read_bytes() == original_bytes
    resumed, after, _ = execute_legacy_branch(tmp_path/'resumed', source,
                            tmp_path/'branch'/'branch-step=5.ckpt', workers=workers)
    assert resumed.trace == [row for row in model.trace if row['step'] >= 5]
    assert resumed.val_trace == [row for row in model.val_trace if row[0] > 5]
    for key in ('state_dict','optimizer_states','lr_schedulers','epoch','global_step',CompleteCheckpoint.KEY):
        assert tree_equal(final[key], after[key]), key
    import copy
    tampered = copy.deepcopy(final)
    tampered[CompleteCheckpoint.KEY]['legacy_branch']['branch_seed'] += 1
    callback = CompleteCheckpoint(CommittedRandomSampler(IndexedData(), 881), tmp_path/'unused',
                                  'unused', 5, 4, final[CompleteCheckpoint.KEY]['contract'])
    with pytest.raises(ValueError, match='provenance changed'):
        callback.on_load_checkpoint(None, None, tampered)


def test_legacy_branch_rejects_wrong_modes_and_directory_reuse(tmp_path, monkeypatch):
    from types import SimpleNamespace
    source = make_legacy_checkpoint(tmp_path/'legacy')
    original = torch.load(source, map_location='cpu')
    origin = legacy_branch_provenance(source, original, 881)
    with pytest.raises(ValueError, match='branch-seed'):
        legacy_branch_provenance(source, original, -1)
    with pytest.raises(ValueError, match='use --resume'):
        legacy_branch_provenance(source, {CompleteCheckpoint.KEY: {}}, 881)
    callback = CompleteCheckpoint(CommittedRandomSampler(IndexedData(), 881), tmp_path/'unused',
                    'unused', 5, 4, dict(world_size=1, legacy_branch=origin), legacy_branch=origin)
    trainer = SimpleNamespace(global_rank=0, world_size=1, accumulate_grad_batches=2)
    with pytest.raises(ValueError, match='accumulate_grad_batches=1'):
        callback.setup(trainer, SimpleNamespace(automatic_optimization=True), 'fit')
    directory = tmp_path/'reserved'
    reserve_branch_directory(directory, origin)
    with pytest.raises(FileExistsError):
        reserve_branch_directory(directory, origin)
    monkeypatch.setenv('LOCAL_RANK', '1')
    reserve_branch_directory(directory, origin)  # PL child inherits the launch ownership token.
    with pytest.raises(ValueError, match='does not own'):
        reserve_branch_directory(directory, dict(origin, branch_seed=882))


@pytest.mark.parametrize('arguments', [
    ['--resume', 'old.ckpt', '--branch-from', 'old.ckpt', '--branch-seed', '8'],
    ['--branch-from', 'old.ckpt'], ['--branch-seed', '8'],
    ['--branch-from', 'old.ckpt', '--branch-seed', '8', '--seed', '9'],
])
def test_legacy_branch_cli_rejects_ambiguous_modes(monkeypatch, arguments):
    import sys
    from semkine.train import main
    monkeypatch.setattr(sys, 'argv', ['train.py', '--config', 'must-not-read.yaml', *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
