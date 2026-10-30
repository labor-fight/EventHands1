"""Synthetic cost-contract checks; no recording, label or GPU reads."""
from types import SimpleNamespace
import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from tools.xvm_joint.zgz_resources import (
    PacketCase, benchmark_packet_replay, parameter_counts, profile_packet_layers,
    synthetic_profile_cases, measure_resources,
)


class ToyStateful(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.base = nn.Linear(n, n, bias=False)
        self.local = nn.Linear(n, n, bias=False)
        with torch.no_grad():
            self.base.weight.copy_(torch.eye(n))
            self.local.weight.zero_()
        self.counter = 5
        self.observed_previous = []
        self.resets = []
        self.fail = False

    def snapshot_state(self):
        return {"counter": self.counter}

    def restore_state(self, state):
        self.counter = state["counter"]

    def reset_run(self, start_ms=0, sequence_id=None):
        self.counter = 0
        self.resets.append((start_ms, sequence_id))

    def forward_packet(self, batch):
        self.counter += 1
        if self.fail:
            raise RuntimeError("synthetic failure")
        self.observed_previous.append(batch.prev_state.detach().clone())
        return self.base(batch.prev_state)+self.local(batch.prev_state)+self.counter


def cases(n=2):
    return [PacketCase(SimpleNamespace(events=torch.zeros(i+1, 5),
                        prev_state=torch.full((1, n), 1. if i == 0 else 99.)),
                        reset=i == 0, run_start_ms=100, sequence_id="test")
            for i in range(3)]


def test_latency_replays_own_state_from_same_snapshot_and_restores_caller():
    model = ToyStateful()
    packet_cases = cases()
    original = [row.batch.prev_state.clone() for row in packet_cases]
    result = benchmark_packet_replay(model, packet_cases, device="cpu", anchor_raw_ms=2.,
                                    passes=3, warmup_packets=2)
    assert model.counter == 5
    expected = [1., 2., 1., 2., 4., 1., 2., 4., 1., 2., 4.]
    assert [float(x[0, 0]) for x in model.observed_previous] == expected
    assert result["latency_pass_terminal_states_equal"]
    digest = hashlib.sha256(np.full((1, 2), 7., np.float32).tobytes()).hexdigest()
    assert result["latency_pass_terminal_state_sha256"] == [digest]*3
    assert result["latency_ms_raw"] > 0
    assert result["latency_ms_scaled_full1p75"] == pytest.approx(result["latency_ms_raw"]*.875)
    assert result["latency_packets"] == 3
    assert result["latency_mean_events_per_packet"] == 2
    assert all(torch.equal(row.batch.prev_state, value) for row, value in zip(packet_cases, original))


def test_run_boundary_resets_history_and_restarts_declared_previous():
    model = ToyStateful()
    rows = cases()
    rows[2] = PacketCase(rows[2].batch, reset=True, run_start_ms=700, sequence_id="next")
    benchmark_packet_replay(model, rows, device="cpu", anchor_raw_ms=1., passes=1, warmup_packets=0)
    assert [float(x[0, 0]) for x in model.observed_previous] == [1., 2., 99.]
    assert model.resets == [(100, "test"), (700, "next")]
    assert model.counter == 5


def test_failure_also_restores_snapshot_and_invalid_anchor_rejected():
    model = ToyStateful();model.fail = True
    with pytest.raises(RuntimeError, match="synthetic failure"):
        benchmark_packet_replay(model, cases(), device="cpu", anchor_raw_ms=1., warmup_packets=0)
    assert model.counter == 5
    with pytest.raises(ValueError, match="anchor"):
        benchmark_packet_replay(model, cases(), device="cpu", anchor_raw_ms=float("nan"))


def test_profile_counts_base_and_local_standard_layers():
    pytest.importorskip("thop")
    model = ToyStateful()
    rows = cases()
    result = profile_packet_layers(model, rows[-1], priming_cases=rows[:-1])
    assert result["macs_forward_packet"] == 8
    assert result["params_total"] == 8
    assert result["params_canonical"] == 4
    assert result["params_local"] == 4
    assert "finite-difference" in result["macs_forward_packet_note"]
    assert "lower bound" in result["macs_forward_packet_note"]


def test_wrapper_and_global_none_parameter_counts_are_actual_registered_weights():
    model = ToyStateful()
    wrapper = nn.Module();wrapper.core = model
    assert parameter_counts(wrapper) == parameter_counts(model)
    model.local = None
    assert parameter_counts(wrapper) == {"params_total": 4, "params_local": 0, "params_canonical": 4}


def test_state_filter_wrapper_uses_reset_state_with_declared_previous():
    class ResetStateOnly(nn.Module):
        def __init__(self):
            super().__init__();self.core = ToyStateful();self.received = []
        def snapshot_state(self):return self.core.snapshot_state()
        def restore_state(self, value):self.core.restore_state(value)
        def reset_state(self, previous):
            self.received.append(previous.clone());self.core.reset_run()
        def forward_packet(self, batch):return self.core.forward_packet(batch)
    model = ResetStateOnly()
    result = benchmark_packet_replay(model, cases(), device="cpu", anchor_raw_ms=1.,
                                    passes=2, warmup_packets=0)
    assert len(model.received) == 2
    assert all(torch.equal(value, torch.ones(1, 2)) for value in model.received)
    assert result["latency_pass_terminal_states_equal"]
    assert model.core.counter == 5


@pytest.mark.parametrize("window", [50, 200])
def test_saturated_fixture_obeys_exact_formal_integer_us_packet_contract(window):
    from tools.xvm_joint.zgz_model import relative_packet_events
    rows = synthetic_profile_cases(window_ms=window)
    assert len(rows) == 3 and rows[0].reset and not rows[-1].reset
    assert rows[-1].batch.events.shape == (4096, 5)
    for row in rows:
        native, duration = relative_packet_events(row.batch)
        assert duration == window
        assert native.shape == (4096, 4)
        assert np.all((native[:, 2] >= 0) & (native[:, 2] < window*1000))
    assert torch.equal(rows[0].batch.prev_state[:, 2], torch.tensor([.45]))


def test_factory_uses_distinct_timing_and_cost_instances():
    pytest.importorskip("thop")
    created = []
    def factory(device):
        instance = ToyStateful(n=51).to(device);created.append(instance);return instance
    result = measure_resources(factory, cases(n=51), device="cpu", anchor_raw_ms=1.,
                               passes=1, warmup_packets=0, max_packets=2)
    assert len(created) == 2 and created[0] is not created[1]
    assert created[0].counter == 5
    assert result["latency_packets"] == 2
    assert result["params_total"] == 2*51*51
    assert result["macs_forward_packet"] == 2*51*51
