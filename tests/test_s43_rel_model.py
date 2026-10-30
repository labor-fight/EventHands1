"""Meaningful adapter and continuous-history contracts, without data reads."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn

from tools.s43_rel.model import S43RelationModel, TypedSupportAdapter
from tools.s43_rel.data import EMATrace, enhanced_channels


class TinyS43(nn.Module):
    sparse_trunk = True
    active_head = False
    transl_head = None
    log10_loss = False

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 3, 3, padding=1)
        self.rn = nn.Module()
        self.rn.fc = nn.Linear(3, 51)

    def forward(self, image, previous, betas=None, camera_K=None):
        return self.rn.fc(self.conv1(image.permute(0, 3, 1, 2)).mean((2, 3))) + previous


def reps():
    rng = np.random.default_rng(3)
    return [{"lnes": rng.normal(size=(8, 8, 2)).astype(np.float32),
             "sensor": rng.normal(size=(4, 4, 20)).astype(np.float32),
             "f": rng.normal(size=(51, 24)).astype(np.float32),
             "a": rng.normal(size=(4, 51)).astype(np.float32),
             "lam": np.ones((4, 24), np.float32)} for _ in range(2)]


@pytest.mark.parametrize("arm", ("s43", "enhanced", "no_order", "full"))
def test_zero_adapter_exact_and_shared_base_trainable(arm):
    torch.manual_seed(3407)
    base = TinyS43()
    model = S43RelationModel(base, arm).eval()
    inputs, previous = reps(), torch.ones((2, 51))
    expected = base(torch.tensor(np.stack([r["lnes"] for r in inputs])), previous)
    actual = model(inputs, previous)
    assert torch.equal(actual, expected)
    assert all(p.requires_grad for p in base.parameters())
    actual.square().sum().backward()
    if arm != "s43":
        assert model.sensor_adapter.weight.grad.abs().sum() > 0
    else:
        assert model.sensor_adapter.weight.grad is None
    if arm in ("full", "no_order"):
        assert model.typed_adapter.output.weight.grad.abs().sum() > 0


def test_no_support_hard_gate_survives_learned_bias():
    adapter = TypedSupportAdapter()
    with torch.no_grad():
        adapter.output.weight.fill_(.2)
        adapter.output.bias.fill_(4.)
    result = adapter(torch.zeros(2, 51, 24), torch.ones(2, 4, 51),
                     torch.zeros(2, 4, 24), torch.ones(2, 3))
    assert torch.count_nonzero(result) == 0
    with pytest.raises(ValueError, match="nonnegative"):
        adapter(torch.zeros(2, 51, 24), torch.ones(2, 4, 51),
                -torch.ones(2, 4, 24), torch.ones(2, 3))


def test_forward_exception_releases_temporary_hooks():
    model = S43RelationModel(TinyS43(), "full")
    inputs = reps()
    inputs[0]["f"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        model(inputs, torch.zeros(2, 51))
    assert model._active_sensor is None and model._active_typed is None
    assert torch.isfinite(model(reps(), torch.zeros(2, 51))).all()


def test_ema_prefix_partition_parity_and_checkpoint():
    first = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [8, 9, 49, 1]])
    second = np.array([[1, 2, 51, 1], [8, 9, 71, 0]])
    together, split = EMATrace(0), EMATrace(0)
    expected = together.ingest(np.concatenate((first, second)), 100)
    split.ingest(first, 50)
    restored = EMATrace.from_state(split.state_dict())
    actual = restored.ingest(second, 100)
    np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=1e-7)
    assert actual.sum() > 0
    # First 50ms history is decayed, not erased at the next window boundary.
    cold = EMATrace(50).ingest(second, 100)
    assert np.any(actual > cold)


def test_enhanced_has_bins_counts_and_continuous_ema():
    events = np.array([[1, 2, 0, 0], [1, 2, 10, 0], [1, 2, 10, 0], [4, 5, 49, 1]])
    trace = EMATrace(0)
    A = trace.ingest(events, 50)
    x = enhanced_channels(events, 0, 50, A)
    assert x.shape == (180, 240, 20)
    assert x[2, 1, 0] == np.float32(.2)
    assert x[2, 1, 4] == np.log1p(np.float32(3))
    assert x[2, 1, 6] == np.log1p(np.float32(3))
    assert np.isfinite(x).all()


def test_run_lanes_never_read_gt_and_consume_short_tail(monkeypatch):
    from tools.s43_rel import data
    class Reader:
        target_reads = 0
        def __init__(self, name, root):
            self.runs = np.array([[0, 123]])
            self.offsets = np.arange(124)
            self.events = np.tile(np.array([[1, 2, 0]], np.uint8), (123, 1))
            self.betas, self.K_cam = np.zeros(10, np.float32), np.eye(3, dtype=np.float32)
        def target(self, end):
            self.target_reads += 1
            return np.full(51, end, np.float32)
    monkeypatch.setattr(data, "SequenceReader", Reader)
    monkeypatch.setattr(data, "recording_dirs", lambda root: {"ch_local": "unused"})
    lanes = data.ContinuousRunLanes("/unused", subjects=("ch",), batch_size=1)
    first, second, third = lanes.next_batch()[0], lanes.next_batch()[0], lanes.next_batch()[0]
    assert first["reset"] and not second["reset"] and third["reset"]
    assert first["seq_id"] == second["seq_id"] != third["seq_id"]
    assert not any(k in first for k in ("target", "targets", "previous_GT"))
    assert lanes.readers["ch_local"].target_reads == 0
    assert third["released_tail"]["source_range"] == (100, 123)
    assert len(third["released_tail"]["events"]) == 23
    assert lanes.tail_ms == lanes.tail_events == 23
    assert lanes.events == 173 and lanes.outputs == 3
    targets = lanes.targets([first])
    assert targets[0, 0] == 49
    assert lanes.readers["ch_local"].target_reads == 1
    restored = data.ContinuousRunLanes("/unused", subjects=("ch",), batch_size=1)
    restored.load_state_dict(lanes.state_dict())
    a, b = lanes.next_batch()[0], restored.next_batch()[0]
    assert a["seq_id"] == b["seq_id"]
    np.testing.assert_array_equal(a["events"], b["events"])
