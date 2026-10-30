"""Synthetic S39 postfilter parity, feedback and single-ingest contracts."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from semkine.anchored import AdaptiveFilter
from tools.xvm_joint.zgz_filter import S39PacketFilter


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.calls = self.resets = 0
        self.last_lnes = None
        self.last_current_count = 0
        self.last_audit = {}
        self.previous_inputs = []
        self.context = None

    def set_hand_context(self, beta, camera_K):
        self.context = beta, camera_K

    def reset_state(self, prev):
        self.resets += 1
        self.last_lnes, self.last_current_count = None, 0
        self.previous_inputs = []

    def forward_packet(self, batch):
        self.calls += 1
        self.previous_inputs.append(batch.prev_state.clone())
        self.last_lnes = batch.synthetic_lnes
        self.last_current_count = batch.synthetic_count
        return batch.prev_state + batch.synthetic_delta

    def snapshot_state(self):
        return copy.deepcopy(dict(last_lnes=self.last_lnes, last_current_count=self.last_current_count,
                                  previous_inputs=self.previous_inputs, calls=self.calls))

    def restore_state(self, state):
        for name, value in copy.deepcopy(state).items():
            setattr(self, name, value)


class FixedProposal(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = None

    def forward(self, x, prevpos, betas=None, camera_K=None):
        return self.value


class Packet:
    def __init__(self, previous, count, *, empty_image=False):
        self.prev_state = previous
        self.synthetic_count = count
        self.synthetic_lnes = torch.zeros(1, 2, 3, 2) if empty_image else torch.ones(1, 2, 3, 2)
        self.synthetic_delta = torch.linspace(-.02, .03, 51).reshape(1, 51)
        # A longer overlapping evidence window can contain many old events
        # while its latest50ms count is zero. This count must not be used.
        self.events = torch.ones(100, 5)

    @property
    def betas(self):
        raise AssertionError('Wrapper must not read score-only packet betas')

    @property
    def target(self):
        raise AssertionError('Wrapper must not read score-only packet target')


def initial():
    state = torch.zeros(1, 51)
    state[:, 2] = .6
    return state


def ramp_spec():
    path = Path(__file__).resolve().parents[1] / 'outputs/dt3/gate_specs/ramp1000.json'
    return json.loads(path.read_text())


def test_ramp_filter_matches_original_exactly_with_one_core_call_and_filtered_feedback():
    core, reference_proxy = Core(), FixedProposal()
    actual = S39PacketFilter(core, ramp_spec())
    expected = AdaptiveFilter.from_spec(reference_proxy, ramp_spec())
    previous = initial()
    actual.reset_state(previous); expected.reset_state(previous)
    for index, count in enumerate((40, 90, 650, 1500)):
        batch = Packet(previous, count)
        reference_proxy.value = previous + batch.synthetic_delta
        wanted = expected(batch.synthetic_lnes, previous, n_events=count)
        got = actual.forward_packet(batch)
        assert torch.equal(got, wanted)
        assert core.calls == index + 1
        assert torch.equal(core.previous_inputs[-1], previous)
        assert actual._proposal_proxy.proposal is None
        previous = got
    assert not torch.equal(core.previous_inputs[1], initial() + Packet(initial(), 40).synthetic_delta)


@pytest.mark.parametrize('count,empty_image', [(0, False), (1000, True), (0, True)])
def test_latest50ms_count_and_full_window_lnes_both_have_original_empty_hold(count, empty_image):
    core = Core(); actual = S39PacketFilter(core, ramp_spec())
    previous = initial(); actual.reset_state(previous)
    batch = Packet(previous, count, empty_image=empty_image)
    assert len(batch.events) == 100
    assert torch.equal(actual.forward_packet(batch), previous)
    assert core.calls == 1


def test_reset_clears_core_and_native_filter_velocity_and_context_never_reads_beta():
    class PoisonBeta:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Score-only beta converted')

    core = Core(); wrapped = S39PacketFilter(core, dict(a_root=.5, a_rest=.7, a_trans=.4,
                                                       beta_root=.2, beta_trans=.3))
    previous = initial(); wrapped.reset_state(previous)
    wrapped.forward_packet(Packet(previous, 100))
    assert torch.count_nonzero(wrapped.filter.get_state()['v_trans']) > 0
    camera_K = torch.eye(3).unsqueeze(0)
    wrapped.set_hand_context(PoisonBeta(), camera_K)
    assert core.context[0] is None and core.context[1] is camera_K
    wrapped.reset_state(previous)
    assert core.resets == 2 and core.last_lnes is None
    assert all(torch.count_nonzero(v) == 0 for v in wrapped.filter.get_state().values())
    assert wrapped._proposal_proxy.proposal is None


def test_snapshot_restore_repeats_native_stateful_filter_without_double_counting_core_parameters():
    core = Core(); actual = S39PacketFilter(core, dict(a_root=.5, a_rest=.7, a_trans=.4,
                                                     beta_root=.2, beta_trans=.3))
    previous = initial(); actual.reset_state(previous)
    previous = actual.forward_packet(Packet(previous, 100))
    saved = actual.snapshot_state()
    batch = Packet(previous, 100)
    first = actual.forward_packet(batch)
    actual.restore_state(saved)
    second = actual.forward_packet(batch)
    assert torch.equal(first, second)
    assert actual.encoder_name and not hasattr(actual, 'get_state')
    assert sum(p.numel() for p in actual.parameters()) == sum(p.numel() for p in core.parameters())
