"""Synthetic S37 packet identities and legacy nonempty arithmetic across AMP dtypes.

CPU is the default. Opt into CUDA with S37_PRECISION_DEVICE=cuda:0; this module never
chooses an available GPU automatically. No training data or checkpoint is loaded.
"""
from __future__ import annotations

import copy
import dataclasses
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from model import MNISTModel  # noqa: E402
from semkine.events import EventPacketBatch  # noqa: E402


def _config():
    return {
        "MODEL": {
            "POSE_REPR": "mano_full_axis_angle", "OUTPUT_DIM": 51, "MANO_NCOMPS": 45,
            "PREDICT_DELTA": True, "PREVPOS_EMBED": True, "PREV_RENDER": False,
            "ROUTED_READOUT": True, "ZERO_EVENT_GATE": True,
            "ROUTE_BAND_PX": 16.0, "ROUTE_FRONT_K": 8,
            "RENDER_H": 180, "RENDER_W": 240, "RENDER_SCALE": 0.375,
            "ENCODER": "event_gnn", "ENCODER_HIDDEN": 16, "ENCODER_FEAT": 32,
            "ENCODER_LAYERS": 1, "ENCODER_K": 4, "ENCODER_MAX_NODES": 16,
            "ENCODER_WINDOW": 8, "ACTIVE_HEAD": True, "ACTIVE_HIDDEN": 8,
        },
        "LOSS": {"LAMBDA_POSE": 450.0, "LAMBDA_T": 30000.0, "LAMBDA_R": 60.0,
                 "NORMALIZER": 51, "LOG10": True},
        "TRAIN": {"LR": 1e-3, "WARMUP_STEPS": 0},
        "DATA": {"HEIGHT": 180, "WIDTH": 240},
        "MANO": {"NPZ": str(ROOT / "assets/mano_right.npz")},
    }


@pytest.fixture(scope="module", autouse=True)
def _execution_contract():
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    torch.set_num_threads(min(old_threads, 4))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    yield
    torch.set_num_threads(old_threads)
    torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
    if old_workspace is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


@pytest.fixture(scope="module")
def device():
    selected = torch.device(os.environ.get("S37_PRECISION_DEVICE", "cpu"))
    if selected.type not in ("cpu", "cuda"):
        pytest.fail("S37_PRECISION_DEVICE must select cpu or an explicitly authorized CUDA device")
    if selected.type == "cuda" and not torch.cuda.is_available():
        pytest.fail("Explicit CUDA precision test requested but CUDA is unavailable")
    return selected


@pytest.fixture(params=["fp32", "bf16", "fp16"])
def precision(request, device):
    if request.param == "fp16" and device.type == "cpu":
        pytest.skip("PyTorch 2.1 CPU autocast does not implement the CUDA FP16 execution contract")
    return request.param


@pytest.fixture
def model(device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(892)
        result = MNISTModel(_config()).to(device).eval()
    return result


def _amp(device, precision):
    return torch.autocast(device.type,
                         dtype=torch.float16 if precision == "fp16" else torch.bfloat16,
                         enabled=precision != "fp32")


def _packet(model, kind, state_dtype=torch.float32):
    device = next(model.parameters()).device
    prev = torch.linspace(-0.07123, 0.09137, 3 * 51, device=device).reshape(3, 51)
    prev[:, :3] = torch.tensor([[.010123, -.014567, .50123],
                               [-.008731, .017321, .54789],
                               [.011357, .003719, .61237]], device=device)
    prev = prev.to(state_dtype)
    if state_dtype == torch.float32:
        assert not torch.equal(prev, prev.to(torch.bfloat16).float()), "vacuous precision witness"
    active = [] if kind == "empty" else [1] if kind == "mixed" else [0, 1, 2]
    rows = []
    with torch.no_grad():
        betas, camera = model._resolve_betas_K(prev.float(), None, None)
        uv, _ = model._project_prev(prev.float(), betas, camera)
        for b in active:
            in_frame = ((uv[b, :, 0] >= 1) & (uv[b, :, 0] < 239)
                        & (uv[b, :, 1] >= 1) & (uv[b, :, 1] < 179))
            selected = in_frame.nonzero().flatten()[:12]
            assert selected.numel() >= 8, "synthetic events must reach the projected MANO hand"
            for t, vertex in enumerate(selected.tolist()):
                xy = uv[b, vertex].round().tolist()
                rows.append([b, xy[0], xy[1], .001 * (t + 1), t % 2])
    events = torch.tensor(rows, dtype=torch.float32, device=device).reshape(-1, 5)
    counts = torch.bincount(events[:, 0].long(), minlength=3)
    packet = EventPacketBatch(
        events=events, ptr=torch.cat([counts.new_zeros(1), counts.cumsum(0)]),
        sequence_id=torch.arange(3, device=device),
        t_start_us=torch.zeros(3, dtype=torch.long, device=device),
        t_end_us=torch.full((3,), 50000, dtype=torch.long, device=device),
        delta_t_s=torch.full((3,), .05, device=device),
        is_sequence_start=torch.zeros(3, dtype=torch.bool, device=device),
        is_sequence_end=torch.zeros(3, dtype=torch.bool, device=device),
        target=torch.zeros(3, 51, device=device), prev_state=prev.detach().requires_grad_(),
        betas=torch.zeros(3, 10, device=device), camera_K=None,
    )
    packet.validate()
    return packet


def _old_output(model, packet):
    """Run the real encoder/heads, then reconstruct only the old packet update boundary."""
    delta_mode = model.predict_delta
    model.predict_delta = False
    try:
        raw = model.forward_packet(packet)
    finally:
        model.predict_delta = delta_mode
    if not delta_mode:
        return raw
    delta = torch.where((packet.counts <= 0).unsqueeze(-1), torch.zeros_like(raw), raw) \
        if model.zero_event_gate else raw
    return delta + packet.prev_state.to(raw.dtype)


def _parameters(model):
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


def _assert_same_gradients(model, prev, out, reference, old_prev, old_out, selected):
    weights = torch.linspace(-.75, .625, 51, device=out.device)
    actual_parameters, old_parameters = _parameters(model), _parameters(reference)
    assert [name for name, _ in actual_parameters] == [name for name, _ in old_parameters]
    gradients = torch.autograd.grad((out[selected].float() * weights).sum(),
        [prev] + [parameter for _, parameter in actual_parameters], allow_unused=True)
    expected = torch.autograd.grad((old_out[selected].float() * weights).sum(),
        [old_prev] + [parameter for _, parameter in old_parameters], allow_unused=True)
    for name, actual, old in zip(["prev_state"] + [name for name, _ in actual_parameters],
                                 gradients, expected):
        assert (actual is None) == (old is None), name
        if actual is not None:
            assert torch.equal(actual, old), name
    assert any(g is not None and bool(g.any()) for g in gradients[1:]), "vacuous parameter gradient comparison"


@pytest.mark.parametrize("kind", ["empty", "mixed"])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16], ids=["prev_fp32", "prev_bf16"])
def test_empty_rows_are_exact_state_and_identity_gradient(model, device, precision, kind, state_dtype):
    packet = _packet(model, kind, state_dtype)
    empty = packet.counts == 0
    with _amp(device, precision):
        out = model.forward_packet(packet)
    head_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    assert out.dtype == torch.promote_types(state_dtype, head_dtype)
    assert torch.equal(out[empty].to(state_dtype), packet.prev_state[empty])
    # Comparing after a round-trip alone would hide FP32 -> BF16 rounding; also
    # compare directly in the output dtype and use a nonrepresentable FP32 witness.
    assert torch.equal(out[empty], packet.prev_state[empty].to(out.dtype))
    weights = torch.linspace(-.75, .625, 51, device=device)
    gradients = torch.autograd.grad((out[empty].float() * weights).sum(),
        [packet.prev_state] + [parameter for _, parameter in _parameters(model)], allow_unused=True)
    # Explicit outer product avoids PyTorch 2.1's deterministic CUDA indexed
    # assignment failure when a 1-D value broadcasts across boolean row indices.
    expected = empty.to(state_dtype).unsqueeze(1) * weights.to(state_dtype).unsqueeze(0)
    assert torch.equal(gradients[0], expected)
    assert all(g is None or not bool(g.any()) for g in gradients[1:])


def test_repeated_empty_packets_preserve_fp32_bits_including_signed_zero(model, device, precision):
    packet = _packet(model, "empty")
    previous = packet.prev_state.detach().clone()
    previous[:, 6] = -0.0
    previous[:, 7] = 0.0
    expected_bits = previous.view(torch.int32).clone()
    assert bool(torch.signbit(previous[:, 6]).all())
    assert not bool(torch.signbit(previous[:, 7]).any())
    with torch.no_grad(), _amp(device, precision):
        for _ in range(5):
            packet = dataclasses.replace(packet, prev_state=previous)
            previous = model.forward_packet(packet)
            assert previous.dtype == torch.float32
            assert torch.equal(previous.view(torch.int32), expected_bits)


@pytest.mark.parametrize("kind", ["mixed", "nonempty"])
def test_nonempty_values_and_gradients_match_legacy_arithmetic(model, device, precision, kind):
    packet = _packet(model, kind)
    reference = copy.deepcopy(model)
    old_packet = dataclasses.replace(packet, prev_state=packet.prev_state.detach().clone().requires_grad_())
    with _amp(device, precision):
        out = model.forward_packet(packet)
        old = _old_output(reference, old_packet)
    selected = packet.counts > 0
    assert out.dtype == torch.promote_types(packet.prev_state.dtype, old.dtype)
    assert torch.equal(out[selected].float(), old[selected].float())
    assert float(model.route_stats["route_frac_routed"]) > 0
    _assert_same_gradients(model, packet.prev_state, out, reference, old_packet.prev_state, old, selected)


@pytest.mark.parametrize("delta,gate", [(True, False), (False, True), (False, False)],
                         ids=["gate_disabled", "absolute_with_gate", "absolute_without_gate"])
def test_disabled_gate_or_absolute_prediction_preserves_old_contract(model, device, precision, delta, gate):
    model.predict_delta, model.zero_event_gate = delta, gate
    packet = _packet(model, "mixed")
    reference = copy.deepcopy(model)
    old_packet = dataclasses.replace(packet, prev_state=packet.prev_state.detach().clone().requires_grad_())
    with _amp(device, precision):
        out = model.forward_packet(packet)
        old = _old_output(reference, old_packet)
    assert out.dtype == old.dtype
    assert torch.equal(out, old)
    assert not torch.equal(out[packet.counts == 0].float(), packet.prev_state[packet.counts == 0])
    _assert_same_gradients(model, packet.prev_state, out, reference, old_packet.prev_state, old,
                          torch.ones(packet.batch_size, dtype=torch.bool, device=device))
