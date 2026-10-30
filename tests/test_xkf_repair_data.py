"""Repair labels preserve packets and align to the physical pre-main timestamp."""
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'model')]

from semkine import domrand as DR
from tools.xdl.data import packet_hash
from tools.xkf.data import LongWindowRecoveryDataset, collate_long
from tools.xkf_repair.data import MotionTargetDataset, collate_motion
from tools.xkf_repair.train import override_learning_rate, find_original_initial


@pytest.fixture
def stream(tmp_path):
    count = 700
    base = tmp_path / 'train' / 'fixture_global'
    base.parent.mkdir()
    ms = np.arange(count)
    events = np.column_stack([ms % 37, ms % 29, ms % 2]).astype(np.uint8)
    np.save(str(base) + '_events.npy', events)
    np.save(str(base) + '_offsets.npy', np.arange(count + 1, dtype=np.int64))
    np.save(str(base) + '_tsub.npy', np.zeros(count, dtype=np.uint16))
    np.savez(str(base) + '_aux.npz', valid_runs_ms=np.array([[10, 360], [400, 700]]),
             category='global', betas=np.zeros(10, np.float32),
             camera_K=np.array([[400., 0., 120.], [0., 400., 90.], [0., 0., 1.]], np.float32))
    dtype = np.dtype([('data', np.float64, 51), ('m0', np.uint8), ('m1', np.uint8)])
    records = np.zeros(count, dtype=dtype)
    records['data'][:, 0] = ms * .00001
    records['data'][:, 2] = .5 + ms * .00002
    records['data'][:, 3] = ms * .0001
    records['data'][:, 6] = ms * .0003
    records['m0'], records['m1'] = 4, 13
    with open(str(base) + '.meta', 'wb') as handle:
        handle.write(struct.pack('<i', 51))
        records.tofile(handle)
    return tmp_path


def dataset(root, cls, augmented):
    return cls(root=root, sequences=[('fixture_global', 'train')],
               components=np.eye(45, dtype=np.float32), mano_npz=str(REPO / 'assets/mano_right.npz'),
               seed=3407, max_steps=2002, start_step=2000,
               domrand=DR.DomRandConfig(enabled=augmented, scale_mode='depth'),
               polarity_flip=True, pixel_polarity_swap=True,
               prev_noise=(.005, .05, .05))


@pytest.mark.parametrize('augmented', [False, True])
def test_motion_targets_keep_original_packet_stream_and_correct_clock(stream, augmented):
    old = dataset(stream, LongWindowRecoveryDataset, augmented)
    new = dataset(stream, MotionTargetDataset, augmented)
    old_items = [old[i] for i in (0, 7, 1025)]
    new_items = [new[i] for i in (0, 7, 1025)]
    a, b = collate_long(old_items), collate_motion(new_items)
    for packet_a, packet_b in zip([a['main'], *a['prefix']], [b['main'], *b['prefix']]):
        assert packet_hash(packet_a) == packet_hash(packet_b)
        assert torch.equal(packet_a.lnes, packet_b.lnes)
        assert torch.equal(packet_a.lnes_window_ms, packet_b.lnes_window_ms)
        assert not hasattr(packet_b, 'previous_gt')
        assert not hasattr(packet_b, 'motion_delta')
    assert all(not bool(p.target.any()) for p in b['prefix'])
    for i, item in enumerate(new_items):
        main = item['main']
        handle = new._handle(main.sequence_id)
        previous_ms = int(main.meta['incoming_ms'])
        assert previous_ms == main.meta['end_idx'] - 50
        expected = handle.pos51[previous_ms].copy()
        if augmented:
            expected, _ = DR.transform_labels(expected, main.meta['domrand'], handle.camera_K,
                                               handle.j0, new.render_scale)
        assert np.array_equal(item['previous_gt'], expected)
        assert np.array_equal(item['motion_delta'], main.target - expected)
        assert not np.array_equal(item['previous_gt'], main.prev_state)
        assert np.array_equal(b['previous_gt'][i].numpy(), expected)
        assert np.array_equal(b['motion_delta'][i].numpy(), item['motion_delta'])
    # Modifying the auxiliary labels cannot mutate inference packet tensors.
    before = packet_hash(b['main'])
    b['previous_gt'].add_(10)
    b['motion_delta'].zero_()
    assert packet_hash(b['main']) == before


def test_explicit_rate_override_survives_next_scheduler_step_without_resetting_adam():
    weight = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.Adam([weight], lr=.004)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    weight.square().sum().backward()
    optimizer.step()
    scheduler.step()
    adam_step = optimizer.state[weight]['step'].clone()
    moment = optimizer.state[weight]['exp_avg'].clone()
    clock = scheduler.last_epoch
    trainer = SimpleNamespace(optimizers=[optimizer],
                              lr_scheduler_configs=[SimpleNamespace(scheduler=scheduler)])
    audit = override_learning_rate(trainer, .0004)
    assert audit['actual_before'] == [[.004]]
    assert audit['actual_after'] == [[.0004]]
    assert scheduler.last_epoch == clock
    assert torch.equal(optimizer.state[weight]['step'], adam_step)
    assert torch.equal(optimizer.state[weight]['exp_avg'], moment)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == .0004
    assert scheduler.base_lrs == [.0004]


def test_initial_lineage_rejects_warmstart_mislabeled_as_step_zero(tmp_path):
    checkpoint = tmp_path / 'run-step=2000.ckpt'
    initial = tmp_path / 'initial_state.ckpt'
    torch.save({'global_step': 2000}, initial)
    with pytest.raises(ValueError, match='actual step0'):
        find_original_initial(checkpoint)
    torch.save({'global_step': 0}, initial)
    assert find_original_initial(checkpoint) == initial
