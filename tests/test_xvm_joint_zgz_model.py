"""Synthetic formal-packet tests; no source recording or checkpoint is read."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from tools.xvm_joint.data import advance_field16
from tools.xvm_joint.train import native_lnes_ms
from tools.xvm_joint.zgz_model import (PacketFieldState, XJPacketModel,
                                      packet_lnes, relative_packet_events)


class Packet:
    def __init__(self, relative, window=50, previous=None):
        relative = np.asarray(relative, np.int64).reshape(-1, 4)
        e = np.zeros((len(relative), 5), np.float32)
        e[:, 1:3], e[:, 3], e[:, 4] = relative[:, :2], relative[:, 2].astype(np.float32)*np.float32(1e-6), relative[:, 3]
        self.events = torch.from_numpy(e)
        self.ptr = torch.tensor([0, len(e)])
        self.t_start_us, self.t_end_us = torch.tensor([0]), torch.tensor([window*1000])
        self.prev_state = (torch.tensor([[0., 0., .6]+[0.]*48]) if previous is None else previous)
        self.camera_K = torch.eye(3)[None]

    @property
    def target(self):
        raise AssertionError('target must never be read')

    @property
    def betas(self):
        raise AssertionError('actual beta must never be read')


def test_relative_float32_seconds_roundtrip_every_us_through_300ms():
    stamps = np.arange(300000)
    events = np.column_stack((np.zeros(len(stamps)), np.zeros(len(stamps)), stamps, stamps % 2)).astype(np.int64)
    recovered, duration = relative_packet_events(Packet(events, window=300))
    assert duration == 300
    np.testing.assert_array_equal(recovered, events)


def test_malformed_times_and_noninteger_coordinates_are_rejected():
    packet = Packet([[1, 2, 49999, 0]])
    packet.events[0, 1] = 1.5
    with pytest.raises(ValueError):
        relative_packet_events(packet)
    packet = Packet([[1, 2, 50000, 0]])
    with pytest.raises(ValueError):
        relative_packet_events(packet)


def test_original_W50_lnes_is_bitwise_training_builder():
    e = np.array([[3, 4, 1, 0], [3, 4, 999, 0], [3, 4, 1000, 0],
                  [3, 4, 49999, 0], [5, 6, 39999, 1]], np.int64)
    np.testing.assert_array_equal(packet_lnes(e, 50), native_lnes_ms(e, 0, 50))


def test_W50_snapshots_are_exact_frozen_field_builder_and_reset_isolated():
    cache = PacketFieldState()
    latest = np.full((2, 180, 240), -1, np.int64)
    previous = np.zeros((16, 180, 240), np.float32)
    for index in range(3):
        relative = np.array([[10, 20, 1, 0], [10, 20, 10000, 0], [30, 40, 49999, 1]], np.int64)
        native = relative.copy(); native[:, 2] += index*50000
        expected, latest, _ = advance_field16(latest, native, start_ms=index*50,
                                               cutoff_ms=(index+1)*50, run_start_ms=0)
        observed = cache.advance(relative, 50)
        np.testing.assert_array_equal(observed['field0'], previous)
        np.testing.assert_array_equal(observed['field1'], expected)
        assert observed['both_exposed'] == bool(index)
        previous = expected
    cache.reset()
    reset = cache.advance(relative, 50)
    assert not reset['both_exposed']
    assert not reset['field0'].any()
    assert cache.cursor_ms == 50 and cache.queries == 1


def test_clipped_W200_overlap_consumes_each_new_50ms_once():
    # The official clipped-run-start W200 schedule is 50,100,150,200,200...
    complete = np.array([[i % 240, i % 180, i*10000+17, i % 2] for i in range(30)], np.int64)
    cache = PacketFieldState()
    latest = np.full((2, 180, 240), -1, np.int64)
    previous = np.zeros((16, 180, 240), np.float32)
    for cutoff in range(50, 301, 50):
        window = min(200, cutoff); start = cutoff-window
        packet = complete[(complete[:, 2] >= start*1000) & (complete[:, 2] < cutoff*1000)].copy()
        packet[:, 2] -= start*1000
        new = complete[(complete[:, 2] >= (cutoff-50)*1000) & (complete[:, 2] < cutoff*1000)]
        expected, latest, _ = advance_field16(latest, new, start_ms=cutoff-50,
                                               cutoff_ms=cutoff, run_start_ms=0)
        observed = cache.advance(packet, window)
        np.testing.assert_array_equal(observed['current_events'], new)
        np.testing.assert_array_equal(observed['field0'], previous)
        np.testing.assert_array_equal(observed['field1'], expected)
        assert observed['event_count'] == 5
        previous = expected


def test_first_long_window_primes_history_but_local_exposure_remains_false():
    cache = PacketFieldState()
    packet = np.array([[1, 2, 17, 0], [1, 2, 199999, 1]], np.int64)
    observed = cache.advance(packet, 200)
    assert cache.cursor_ms == 200 and observed['event_count'] == 1
    assert not observed['both_exposed']
    assert observed['field1'][:6].sum() > 0
    assert observed['field1'][6:].sum() == 1


class FakeBase(nn.Module):
    encoder_name = ''
    root_compose = 'add'
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.ones(()))
    def forward(self, image, previous, betas, camera_K):
        assert torch.count_nonzero(betas) == 0
        output = previous.clone(); output[:, 0] += .002
        return output
    def _fk(self, states, beta):
        return states[:, None, :3].expand(-1, 778, -1), states[:, None, :3].expand(-1, 21, -1)


class FakeLocal(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.ones(()))
    def forward(self, both):
        delta = self.weight.new_zeros((len(both), 51))
        delta[:, 0] = both.float()*.001
        return delta*0, delta, {'active': both[:, None]}


class FakeGeometry:
    def prepare(self, field0, field1, previous, absolute, K, both_exposed):
        return SimpleNamespace(model_kwargs={'both': both_exposed})


def test_model_never_reads_target_or_beta_and_local_starts_on_second_query():
    model = XJPacketModel(FakeBase(), FakeLocal(), FakeGeometry())
    first = model.forward_packet(Packet([[1, 2, 10000, 0]]))
    assert first[0, 0] == pytest.approx(.002)
    second = model.forward_packet(Packet([[1, 2, 10000, 0]], previous=first))
    assert second[0, 0] == pytest.approx(.005)
    assert model.last_audit['both_snapshots_valid']
    assert not model.last_audit['actual_beta_accessed'] and not model.last_audit['target_accessed']
    assert set(dict(model.named_parameters())) == {'base.weight', 'local.weight'}
    assert not any(p.requires_grad for p in model.parameters())


def test_empty_current_packet_holds_even_with_nonempty_age_history():
    model = XJPacketModel(FakeBase(), FakeLocal(), FakeGeometry())
    first = model.forward_packet(Packet([[1, 2, 10000, 0]]))
    held = model.forward_packet(Packet([], previous=first))
    torch.testing.assert_close(held, first, rtol=0, atol=0)
    assert model.field_state.field[:6].sum() > 0
    assert model.last_audit['empty_fallback'] == 1


def test_snapshots_are_deep_and_repeat_query_is_exact_after_restore():
    model = XJPacketModel(FakeBase(), FakeLocal(), FakeGeometry())
    first = model.forward_packet(Packet([[1, 2, 10000, 0]]))
    snapshot = model.snapshot_state()
    original_field = snapshot['field']['field'].copy()
    second = model.forward_packet(Packet([[3, 4, 10000, 1]], previous=first))
    np.testing.assert_array_equal(snapshot['field']['field'], original_field)
    model.restore_state(snapshot)
    repeated = model.forward_packet(Packet([[3, 4, 10000, 1]], previous=first))
    torch.testing.assert_close(repeated, second, rtol=0, atol=0)
    model.reset_state(object())
    assert model.field_state.queries == 0 and model.previous_absolute is None
    assert not hasattr(model, 'get_state')
