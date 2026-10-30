"""XKF contracts (docs/EXPERIMENT_RECORDS.md#xkf-20261014): the direct branch never reads prev, empty packets hold,
the fusion limits, the DGP solve, the long-window LNES codes, and the XDL default path / stream left unchanged."""
from __future__ import annotations

import copy
import importlib.util
import math
import subprocess
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'model')]

from semkine.events import splat_event_image  # noqa: E402
from semkine.lie import so3_exp, so3_log  # noqa: E402
from semkine.xkf import N_GROUPS, XKFModel, fuse_states, nearest_aa  # noqa: E402
from semkine.xkf_direct import DirectBranch, HM_STRIDE, dgp_translation  # noqa: E402
from tools.xkf.data import lnes_codes  # noqa: E402


def _cfg(name='configs/xkf/xkf_s3407.yaml'):
    from config import load_config
    return load_config(str(REPO / name))


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(0)
    m = XKFModel(_cfg()).eval()
    with torch.no_grad():
        for p in m.direct.parameters():
            p.add_(0.01 * torch.randn_like(p))
    return m


def _packet(m, prev, n=300, lnes_W=200, empty=False, seed=0):
    from semkine.events import EventPacketBatch
    g = torch.Generator().manual_seed(seed)
    B = prev.shape[0]
    n = 0 if empty else n
    ev = torch.zeros(B * n, 5)
    if n:
        ev[:, 0] = torch.arange(B).repeat_interleave(n).float()
        ev[:, 1] = torch.randint(80, 160, (B * n,), generator=g).float()
        ev[:, 2] = torch.randint(50, 130, (B * n,), generator=g).float()
        ev[:, 3] = torch.sort(torch.rand(B * n, generator=g) * 0.05)[0]
        ev[:, 4] = torch.randint(0, 2, (B * n,), generator=g).float()
    lnes = torch.zeros(B, m.render_h, m.render_w, 2, dtype=torch.int16)
    if not empty:
        lnes[:, 60:120, 90:150, 0] = torch.randint(1, lnes_W, (B, 60, 60), generator=g).to(torch.int16)
    K = torch.tensor([[603.4507, 0.0, 325.09183], [0.0, 602.95654, 242.09796], [0.0, 0.0, 1.0]]).expand(B, 3, 3)
    return EventPacketBatch(
        events=ev, ptr=torch.arange(B + 1) * n, sequence_id=torch.zeros(B, dtype=torch.long),
        t_start_us=torch.zeros(B, dtype=torch.long), t_end_us=torch.full((B,), 50000, dtype=torch.long),
        delta_t_s=torch.full((B,), 0.05), is_sequence_start=torch.zeros(B, dtype=torch.bool),
        is_sequence_end=torch.zeros(B, dtype=torch.bool), target=torch.zeros(B, 51), prev_state=prev,
        betas=torch.zeros(B, 10), camera_K=K.clone(), lnes=lnes, lnes_window_ms=torch.full((B,), lnes_W))


def _prev(B=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    p = torch.zeros(B, 51)
    p[:, 2] = 0.6
    p[:, 3:6] = 0.3 * torch.randn(B, 3, generator=g)
    p[:, 6:] = 0.1 * torch.randn(B, 45, generator=g)
    return p


def test_direct_never_reads_prev(model):
    a, b = _prev(seed=1), _prev(seed=2)
    pa, pb = _packet(model, a), _packet(model, b)
    with torch.no_grad():
        za, _ = model.direct_measurement(pa)
        zb, _ = model.direct_measurement(pb)
    assert torch.equal(za, zb)
    prev = _prev(seed=3).requires_grad_(True)
    z, _ = model.direct_measurement(_packet(model, prev))
    z.sum().backward()
    assert prev.grad is None or float(prev.grad.abs().max()) == 0.0


def test_empty_packet_holds_prev(model):
    prev = _prev()
    with torch.no_grad():
        out = model.forward_packet(_packet(model, prev, empty=True))
    assert torch.equal(out, prev.float())


def test_pure_direct_has_no_root_innovation_clip(model, monkeypatch):
    prev, direct = _prev(), _prev(seed=2)
    prev[:, 3:6] = 0.0
    direct[:, 3:6] = torch.tensor([0.0, 0.0, 2.5])
    monkeypatch.setattr(model, 'kmode', 'd')
    monkeypatch.setattr(model, 'branches', lambda batch: (batch.prev_state, direct, {}))
    assert torch.equal(model.forward_packet(_packet(model, prev)), direct)
    changed_prev = prev + 0.2
    assert torch.equal(model.forward_packet(_packet(model, changed_prev)), direct)
    assert torch.equal(model.forward_packet(_packet(model, prev, empty=True)), prev)


def test_moment_rows_do_not_fall_back_after_self_curriculum(model, monkeypatch):
    packet = _packet(model, _prev())
    monkeypatch.setattr(model, 'self_probability', lambda step: 0.0)
    assert model._moment_rows(packet, []) == [0, 1]
    monkeypatch.setattr(model, 'self_probability', lambda step: 0.5)
    assert model._moment_rows(packet, []) == []
    assert model._moment_rows(packet, [1]) == [1]
    before = model.xkf_moments.clone()
    updates = model.xkf_updates.clone()
    model._update_moments(packet.prev_state, packet.prev_state, packet.prev_state, packet.target, [])
    assert torch.equal(model.xkf_moments, before)
    assert torch.equal(model.xkf_updates, updates)


def test_fusion_limits():
    a, b = _prev(seed=4), _prev(seed=5)
    b[:, 3:6] = so3_log(so3_exp(torch.tensor([0.0, 0.0, 0.2])) @ so3_exp(a[:, 3:6]))
    assert torch.allclose(fuse_states(a, b, torch.zeros(N_GROUPS)), a, atol=1e-6)
    f1 = fuse_states(a, b, torch.ones(N_GROUPS))
    assert torch.allclose(f1[:, :3], b[:, :3]) and torch.allclose(f1[:, 6:], b[:, 6:], atol=1e-6)
    Rf, Rb = so3_exp(f1[:, 3:6]), so3_exp(b[:, 3:6])
    assert float((Rf - Rb).abs().max()) < 1e-5


def test_nearest_aa_seam():
    ref = torch.tensor([[0.0, 0.0, math.pi - 0.01]])
    aa = torch.tensor([[0.0, 0.0, -(math.pi - 0.02)]])          # same rotation family across the seam
    out = nearest_aa(aa, ref)
    assert float((so3_exp(out) - so3_exp(aa)).abs().max()) < 1e-5
    assert float((out - ref).norm()) < 0.1


def test_dgp_exact():
    g = torch.Generator().manual_seed(0)
    J = 0.05 * torch.randn(3, 21, 3, generator=g)
    t = torch.tensor([[0.02, -0.01, 0.6], [0.0, 0.03, 0.75], [-0.05, 0.0, 0.45]])
    X = J + t[:, None]
    uv = X[..., :2] / X[..., 2:3]
    est = dgp_translation(J, uv, torch.ones(3, 21))
    assert float((est - t).abs().max()) < 1e-5


def test_dgp_covariance_and_gradients():
    g = torch.Generator().manual_seed(10)
    joints = (0.04 * torch.randn(2, 21, 3, generator=g)).requires_grad_(True)
    target_t = torch.tensor([[0.02, -0.01, 0.6], [0.0, 0.03, 0.75]])
    xyz = joints.detach() + target_t[:, None]
    uv = (xyz[..., :2] / xyz[..., 2:3] + 0.002 * torch.randn(2, 21, 2, generator=g)).requires_grad_(True)
    weights = torch.ones(2, 21, requires_grad=True)
    estimate, covariance = dgp_translation(joints, uv, weights, return_covariance=True)
    assert covariance.shape == (2, 3, 3)
    assert torch.isfinite(covariance).all()
    assert torch.allclose(covariance, covariance.transpose(-1, -2), atol=1e-7)
    assert (torch.linalg.eigvalsh(covariance) >= -1e-9).all()
    estimate.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in (joints, uv, weights))


def test_heatmap_loss_constrains_spread_not_only_mean():
    direct = DirectBranch(height=36, width=40)
    h, w = 9, 10
    xs = (torch.arange(w) + 0.5) * HM_STRIDE - 0.5
    ys = (torch.arange(h) + 0.5) * HM_STRIDE - 0.5
    target = torch.tensor([[[xs.mean(), ys.mean()]]])
    gaussian = (-0.5 * ((xs[None, :] - target[0, 0, 0]) / 6.0).square()
                - 0.5 * ((ys[:, None] - target[0, 0, 1]) / 6.0).square()).view(1, 1, h, w)
    heat = gaussian / direct.log_beta.detach().exp()
    visible = torch.ones(1, 1, dtype=torch.bool)
    uniform = torch.zeros_like(heat)
    assert torch.allclose(direct.soft_argmax(heat), direct.soft_argmax(uniform), atol=1e-5)
    assert direct.heatmap_loss(heat, target, visible) < 1e-6
    assert direct.heatmap_loss(uniform, target, visible) > 0.1
    assert direct.heatmap_loss(heat, target, ~visible) == 0.0


def test_direct_loss_gradients_reach_only_direct_branch(model):
    model.zero_grad(set_to_none=True)
    packet = _packet(model, _prev())
    packet.target = _prev(seed=4)
    z, aux = model.direct_measurement(packet)
    loss, parts = model._direct_loss(z, aux, packet)
    assert torch.isfinite(loss) and loss > 0
    assert parts['d_hm'] > 0 and parts['d_fk'] > 0
    loss.log10().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.direct.parameters())
    assert all(p.grad is None for p in model.xyz.parameters())
    model.zero_grad(set_to_none=True)


def test_lnes_codes_decode_to_splat():
    g = np.random.default_rng(0)
    n, W = 4000, 173
    xs, ys = g.integers(0, 240, n), g.integers(0, 180, n)
    ps, ms = g.integers(0, 2, n), np.sort(g.integers(0, W, n))
    code = lnes_codes(xs, ys, ps, ms, 180, 240).astype(np.float32)
    dec = np.where(code > 0, (code - 1) / np.float32(W), 0).astype(np.float32)
    ref = splat_event_image(xs, ys, ps, ms, W, 180, 240)
    assert np.array_equal(dec, ref)


def test_xkf_dataset_keeps_xdl_stream():
    from tools.xdl.data import build_train_dataset as xdl_build, collate_recovery, packet_hash
    from tools.xkf.data import build_train_dataset as xkf_build
    cfg = _cfg()
    base = copy.deepcopy(cfg)
    xdl = xdl_build(base, max_steps=2)
    xkf = xkf_build(cfg, max_steps=2)
    for i in (0, 7, 1500):
        a, b = collate_recovery([xdl[i]]), collate_recovery([xkf[i]])
        assert packet_hash(a['main']) == packet_hash(b['main'])
        assert all(packet_hash(p) == packet_hash(q) for p, q in zip(a['prefix'], b['prefix']))
        assert b['main'].lnes.dtype == torch.int16 and int(b['main'].lnes_window_ms[0]) >= 30


def test_xdl_default_path_bitwise():
    """RecoveryXYZModel (refactored burn, new optional batch field) vs the committed HEAD module on real packets."""
    head = subprocess.run(['git', 'show', 'HEAD:semkine/xyz_recovery_study.py'], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout
    path = Path(__file__).with_name('_head_xyz_recovery_study.py')
    path.write_text(head)
    try:
        spec = importlib.util.spec_from_file_location('semkine._head_xyz_recovery_study', path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        from semkine.xyz_recovery_study import RecoveryXYZModel
        cfg = _cfg('configs/xdl/xdl_q1_s3407.yaml')
        torch.manual_seed(0)
        new_m = RecoveryXYZModel(copy.deepcopy(cfg)).eval()
        torch.manual_seed(0)
        old_m = old.RecoveryXYZModel(copy.deepcopy(cfg)).eval()
        old_m.load_state_dict(new_m.state_dict())
        with torch.no_grad():
            for p in new_m.parameters():
                p.add_(0.01)
            old_m.load_state_dict(new_m.state_dict())
            from tools.xdl.data import build_train_dataset, collate_recovery
            data = build_train_dataset(cfg, max_steps=1)
            for start in range(0, 20, 2):
                pk = collate_recovery([data[start], data[start + 1]])['main']
                assert torch.equal(new_m.forward_packet(pk), old_m.forward_packet(pk))
    finally:
        path.unlink(missing_ok=True)
