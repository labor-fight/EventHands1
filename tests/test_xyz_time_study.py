"""CPU operator tests for the isolated XYZ geometry/time study."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from config import load_config  # noqa: E402
from model import MNISTModel  # noqa: E402
from semkine.events import EventPacketBatch  # noqa: E402
from semkine.lie import so3_exp  # noqa: E402
from semkine.xyz_mesh import L_REL, associate_rays, vertex_face_table  # noqa: E402
from semkine.xyz_time_study import (TimeXYZModel, associate_time_slices,  # noqa: E402
                                    extrapolate_state, physical_motion_delta)


def cfg(geom=True, mode="static"):
    result = load_config(str(ROOT / "configs/xa/xa_c1_s3407.yaml"))
    result["MANO"]["NPZ"] = str(ROOT / "assets/mano_right.npz")
    result["MODEL"].update(XYZ_GEOM=geom, XYZ_CAUSAL_STUDY=True,
                          XYZ_CAUSAL_TIME_MODE=mode, ENCODER_HIDDEN=16,
                          ENCODER_LAYERS=1, XYZ_REL_HIDDEN=8, XYZ_SLOT_DIM=4,
                          XYZ_READOUT_HIDDEN=16, ENCODER_MAX_NODES=128)
    result["TRACK"].update(UNROLL_PAIR=True, UNROLL_P=0.0)
    return result


def model(geom=True, mode="static", seed=3):
    torch.manual_seed(seed)
    result = TimeXYZModel(cfg(geom, mode)).float().eval()
    # A nonzero head is essential: initial fc2 is zero and would hide all bugs.
    generator = torch.Generator().manual_seed(71)
    with torch.no_grad():
        result.xyz.fc2.weight.copy_(torch.randn(result.xyz.fc2.weight.shape, generator=generator) * 0.002)
        result.xyz.fc2.bias.copy_(torch.randn(result.xyz.fc2.bias.shape, generator=generator) * 0.0002)
    result.xyz_keep_aux = True
    return result


def packet(batch_size=1, n=48, start_us=50000, duration_us=50000):
    generator = torch.Generator().manual_seed(37)
    events = torch.zeros(batch_size * n, 5)
    for b in range(batch_size):
        events[b * n:(b + 1) * n, 0] = b
        events[b * n:(b + 1) * n, 3] = torch.linspace(0.001, max(duration_us * 1e-6 - 0.001, 0.001), n)
    events[:, 1] = (122 + torch.randn(batch_size * n, generator=generator) * 18).round()
    events[:, 2] = (91 + torch.randn(batch_size * n, generator=generator) * 18).round()
    events[:, 4] = torch.randint(2, (batch_size * n,), generator=generator).float()
    prev = torch.zeros(batch_size, 51)
    prev[:, 2] = 0.45
    prev[:, 3:6] = torch.tensor([0.2, -0.15, 0.1])
    prev[:, 6:] = torch.randn(batch_size, 45, generator=generator) * 0.12
    camera = torch.tensor([[603.4507, 0, 325.09183], [0, 602.95654, 242.09796], [0, 0, 1.]])
    return EventPacketBatch(events=events, ptr=torch.arange(batch_size + 1) * n,
                            sequence_id=torch.zeros(batch_size, dtype=torch.long),
                            t_start_us=torch.full((batch_size,), start_us),
                            t_end_us=torch.full((batch_size,), start_us + duration_us),
                            delta_t_s=torch.full((batch_size,), duration_us * 1e-6),
                            is_sequence_start=torch.zeros(batch_size, dtype=torch.bool),
                            is_sequence_end=torch.zeros(batch_size, dtype=torch.bool),
                            target=torch.zeros_like(prev), prev_state=prev,
                            betas=torch.zeros(batch_size, 10), camera_K=camera.expand(batch_size, -1, -1).clone())


def test_physical_motion_crosses_root_and_mean_shifted_local_pi_seams():
    mean = torch.linspace(-0.2, 0.2, 45, dtype=torch.float64)
    older, newer = torch.zeros(1, 51, dtype=torch.float64), torch.zeros(1, 51, dtype=torch.float64)
    older[:, 3] = math.pi - 0.02
    newer[:, 3] = -math.pi + 0.02
    older[:, 6:] = -mean
    newer[:, 6:] = -mean
    older[:, 6] += math.pi - 0.03
    newer[:, 6] += -math.pi + 0.03
    newer[:, :3] = torch.tensor([0.01, -0.02, 0.03])
    delta = physical_motion_delta(older, newer, mean)
    assert delta[0, 3] == pytest.approx(0.04, abs=1e-12)
    assert delta[0, 6] == pytest.approx(0.06, abs=1e-12)
    assert torch.allclose(delta[0, :3], newer[0, :3])
    assert torch.max(delta[0, 9:].abs()) < 1e-12


def test_extrapolation_uses_physical_composition_and_mean_once():
    mean = torch.linspace(-0.3, 0.3, 45, dtype=torch.float64)
    anchor = torch.randn(2, 51, generator=torch.Generator().manual_seed(23), dtype=torch.float64) * 0.2
    velocity = torch.randn(2, 51, generator=torch.Generator().manual_seed(24), dtype=torch.float64) * 0.5
    elapsed = torch.tensor([0.049, 0.050], dtype=torch.float64)
    got = extrapolate_state(anchor, velocity, elapsed, mean)
    root = so3_exp(velocity[:, 3:6] * elapsed[:, None]) @ so3_exp(anchor[:, 3:6])
    local = so3_exp((velocity[:, 6:] * elapsed[:, None]).reshape(2, 15, 3)) @ \
        so3_exp((anchor[:, 6:] + mean).reshape(2, 15, 3))
    assert torch.allclose(so3_exp(got[:, 3:6]), root, atol=1e-12)
    assert torch.allclose(so3_exp((got[:, 6:] + mean).reshape(2, 15, 3)), local, atol=1e-12)
    assert torch.allclose(got[:, :3], anchor[:, :3] + velocity[:, :3] * elapsed[:, None])
    assert torch.equal(extrapolate_state(anchor, torch.zeros_like(velocity), elapsed, mean), anchor)
    assert torch.equal(extrapolate_state(anchor, velocity, 0.0, mean), anchor)


def test_time_association_refills_nonmonotonic_slots_and_padding_exactly():
    base = torch.tensor([[-0.03, -0.03, 0.5], [0.03, -0.03, 0.5],
                         [0., 0.04, 0.5], [0.3, 0.3, 0.5]])
    vertices = torch.stack([base + torch.tensor([i * 0.015, 0., 0.]) for i in range(4)])[None]
    faces = torch.tensor([[0, 1, 2]])
    table = vertex_face_table(faces, 4)
    bins = torch.tensor([[0, 3, 1, 2, 3, 0, 2, 1, 0]])
    rays = torch.stack([bins.float() * 0.03, torch.zeros_like(bins, dtype=torch.float32),
                        torch.ones_like(bins, dtype=torch.float32)], -1)
    mask = torch.ones_like(bins, dtype=torch.bool)
    mask[:, -1] = False
    kwargs = dict(band=torch.tensor([0.1]), front_tol=torch.tensor([0.02]), z_tol=.01, k=4)
    actual = associate_time_slices(rays, mask, vertices, bins, faces, table, **kwargs)
    for i in range(8):
        reference = associate_rays(rays[:, i:i + 1], mask[:, i:i + 1], vertices[:, bins[0, i]], faces, table, **kwargs)
        for key in actual:
            assert torch.equal(actual[key][:, i:i + 1], reference[key]), key
    assert all(not bool(value[:, -1].any()) for value in actual.values())
    mass = actual['q'][..., None] * actual['A']
    assert torch.allclose(mass.sum(-1) + (1 - actual['q']) * mask, mask.float())


def test_all_four_arms_have_identical_parameters_and_common_initialization():
    arms = [model(g, t) for g in (False, True) for t in ("static", "extrapolate4")]
    counts = [sum(p.numel() for p in arm.parameters()) for arm in arms]
    assert len(set(counts)) == 1
    initial = arms[0].state_dict()
    for arm in arms[1:]:
        assert set(initial) == set(arm.state_dict())
        assert all(torch.equal(value, arm.state_dict()[key]) for key, value in initial.items())
    assert torch.count_nonzero(arms[0].xyz.fc1.weight[:, -51:]) == 0


@pytest.mark.parametrize('geom', [False, True])
def test_zero_velocity_temporal_branch_is_exact_static_with_nonzero_head(geom):
    static, temporal = model(geom, "static"), model(geom, "extrapolate4")
    batch = packet(batch_size=2)
    temporal.set_time_history(batch.prev_state, .049, query_start_offset_s=.001)
    static.set_time_history(batch.prev_state, .049, query_start_offset_s=.001)
    with torch.no_grad():
        expected, actual = static.forward_packet(batch), temporal.forward_packet(batch)
    assert torch.equal(actual, expected)
    for key in ("assoc",):
        assert all(torch.equal(value, static.xyz_last[key][name]) for name, value in temporal.xyz_last[key].items())
    assert torch.equal(temporal.xyz_last['motion_delta'], torch.zeros(2, 51))
    assert 'verts_by_time' not in temporal.xyz_last


def test_dynamic_geometry_matches_own_record_time_and_edge_anchor():
    temporal = model(True, "extrapolate4")
    batch = packet()
    old = batch.prev_state.clone()
    old[:, 0] -= 0.009
    old[:, 3] -= .06
    temporal.set_time_history(old, .049, query_start_offset_s=.001)
    seen = []
    edge_input = []
    handle = temporal.xyz.rel[0].register_forward_pre_hook(lambda _, args: seen.append(args[0].detach().clone()))
    edge_handle = temporal.xyz.edge.register_forward_pre_hook(lambda _, args: edge_input.append(args[0].detach().clone()))
    with torch.no_grad():
        output = temporal.forward_packet(batch)
    handle.remove()
    edge_handle.remove()
    aux = temporal.xyz_last
    rb, rn, rr = (aux['w'] > 0).nonzero(as_tuple=True)
    record_vertex = aux['verts_by_time'][rb, aux['time_bin'][rb, rn], aux['assoc']['vid'][rb, rn, rr]]
    z = aux['assoc']['z'][rb, rn]
    expected_dxyz = (z[:, None] * aux['d'][rb, rn] - record_vertex) / L_REL
    assert len(seen) == 2 and rb.numel() > 0
    assert torch.equal(seen[0][:, 7:10], expected_dxyz)
    assert torch.count_nonzero(seen[1][:, 7:]) == 0
    assert torch.equal(edge_input[0], temporal.xyz.edge_geometry(aux['verts']))
    assert torch.isfinite(output).all()
    assert torch.allclose(aux['query_horizon_s'], torch.tensor([.050]))
    assert torch.allclose(aux['motion_delta'], aux['physical_velocity'] * .050)


def test_history_is_explicit_and_uses_actual_interval_not_packet_clock():
    temporal = model(False, 'extrapolate4')
    batch = packet()
    older = batch.prev_state.clone()
    older[:, 0] -= .01
    temporal.set_time_history(older, .049, query_start_offset_s=.001)
    velocity = temporal.prepare_history(batch.prev_state)
    assert velocity[0, 0] == pytest.approx(.01 / .049, abs=1e-6)
    temporal.set_time_history(older, .1)
    assert temporal.prepare_history(batch.prev_state)[0, 0] == pytest.approx(.1, abs=1e-6)
    temporal.set_time_history(older, 0.)
    assert torch.count_nonzero(temporal.prepare_history(batch.prev_state)) == 0
    with pytest.raises(ValueError, match='nonnegative'):
        temporal.set_time_history(older, -.01)
    temporal.reset_time_history()
    assert torch.count_nonzero(temporal.prepare_history(batch.prev_state)) == 0


def test_runtime_clock_shift_changes_only_geometry_bin_binding():
    temporal = model(True, 'extrapolate4')
    batch = packet()
    older = batch.prev_state.clone()
    older[:, 0] -= .009
    temporal.set_time_history(older, .049, query_start_offset_s=.001)
    with torch.no_grad():
        temporal.forward_packet(batch)
        zero = temporal.xyz_last
        temporal.query_bin_shift = 2
        temporal.forward_packet(batch)
        shifted = temporal.xyz_last
    assert torch.equal(shifted['time_bin'], (zero['time_bin'] + 2) % 4)
    for key in ('tok', 'mask', 'd', 'motion_delta', 'physical_velocity', 'verts',
                'verts_by_time', 'query_horizon_s', 'query_start_offset_s'):
        assert torch.equal(shifted[key], zero[key]), key
    # Per-record D/Z must be gathered using the shifted destination bin.
    rb, rn, rr = (shifted['w'] > 0).nonzero(as_tuple=True)
    record_vertex = shifted['verts_by_time'][rb, shifted['time_bin'][rb, rn], shifted['assoc']['vid'][rb, rn, rr]]
    z = shifted['assoc']['z'][rb, rn]
    assert torch.equal(shifted['record_geometry'][:, :3],
                       (z[:, None] * shifted['d'][rb, rn] - record_vertex) / L_REL)
    temporal.query_bin_shift = .5
    with pytest.raises(ValueError, match='integer'):
        temporal.forward_packet(batch)


def test_main_and_leading_targets_do_not_enter_motion_or_prediction():
    temporal = model(True, 'extrapolate4')
    lead, main = packet(start_us=0), packet(start_us=50000)
    main.prev_state[:, 0] += .003
    saved = main.prev_state.clone()
    with torch.no_grad():
        first, *_ = temporal._predict_batch((lead, main))
        first_motion = temporal.xyz_last['motion_delta'].clone()
        lead.target.fill_(123.)
        main.target.fill_(-456.)
        second, *_ = temporal._predict_batch((lead, main))
    assert torch.equal(first, second)
    assert torch.equal(first_motion, temporal.xyz_last['motion_delta'])
    assert torch.equal(main.prev_state, saved)
    assert temporal._time_history is None


def test_leading_no_grad_main_grad_and_context_cleanup():
    temporal = model(True, 'extrapolate4').train()
    lead, main = packet(start_us=0), packet(start_us=50000)
    lead.prev_state.requires_grad_()
    main.prev_state.requires_grad_()
    predicted, *_ = temporal._predict_batch((lead, main))
    predicted.square().mean().backward()
    assert lead.prev_state.grad is None
    assert main.prev_state.grad is not None and torch.isfinite(main.prev_state.grad).all()
    assert temporal.xyz.fc1.weight.grad is not None and torch.isfinite(temporal.xyz.fc1.weight.grad).all()
    assert torch.count_nonzero(temporal.xyz.fc1.weight.grad[:, -51:]) > 0
    assert temporal._time_history is None and temporal._motion_delta_context is None
    assert temporal._relation_geometry_context is None


@pytest.mark.parametrize('geom', [False, True])
@pytest.mark.parametrize('mode', ['static', 'extrapolate4'])
def test_bf16_leading_does_not_cache_detached_main_parameters(geom, mode):
    temporal = model(geom, mode).train()
    lead, main = packet(start_us=0), packet(start_us=50000)
    # Incoming states deliberately do not require grad: input gradients must
    # not hide detached model weights in a shared outer autocast region.
    assert not main.prev_state.requires_grad and not lead.prev_state.requires_grad
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        predicted, *_ = temporal._predict_batch((lead, main))
        objective = predicted.float().square().mean()
    assert objective.requires_grad
    objective.backward()
    for layer in (temporal.xyz.fc1, temporal.xyz.fc2, temporal.xyz.rel[0]):
        assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
        assert torch.count_nonzero(layer.weight.grad) > 0
    assert temporal._time_history is None


def test_empty_packet_preserves_state_without_advancing_history():
    temporal = model(True, 'extrapolate4')
    batch = packet(n=0)
    older = batch.prev_state.clone()
    older[:, 0] -= .01
    temporal.set_time_history(older, .049, query_start_offset_s=.001)
    with torch.no_grad():
        got = temporal.forward_packet(batch)
    assert torch.equal(got, batch.prev_state)
    assert torch.equal(temporal._time_history[0], older)


def test_default_mnist_model_is_not_patched_by_constructing_study_subclass():
    baseline_cfg = cfg()
    for key in ('XYZ_CAUSAL_STUDY', 'XYZ_CAUSAL_TIME_MODE'):
        baseline_cfg['MODEL'].pop(key)
    torch.manual_seed(101)
    baseline = MNISTModel(copy.deepcopy(baseline_cfg)).eval()
    batch = packet()
    snapshot = {k: v.clone() for k, v in baseline.state_dict().items()}
    with torch.no_grad():
        before = baseline.forward_packet(batch)
        _ = model(True, 'extrapolate4')
        after = baseline.forward_packet(batch)
    assert torch.equal(before, after)
    assert all(torch.equal(value, baseline.state_dict()[key]) for key, value in snapshot.items())
    assert baseline.xyz.fc1.in_features + 51 == model().xyz.fc1.in_features

