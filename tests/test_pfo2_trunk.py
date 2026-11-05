"""PFO2 stage-1 trunk (semkine/pfo2_trunk.py) and training-side changes (semkine/pfo2_train.py): CPU tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.test_pfo import ROOT, _net_and_inputs


def _mano():
    with np.load(ROOT / "assets/mano_right.npz") as z:
        return torch.tensor(np.asarray(z["weights"], np.float32)), torch.tensor(np.asarray(z["f"], np.int64))


def _build(module, variant):
    from semkine import xyz_mesh as XM
    w, f = _mano()
    module.install(variant)
    try:
        torch.manual_seed(0)
        net = XM.XYZMeshNet(w, f).eval()
    finally:
        module.uninstall()
    torch.manual_seed(1)
    for p in net.fc2.parameters():
        torch.nn.init.normal_(p, std=0.01)
    return net


def _fk():
    from mano_layer import ManoLayer
    from pose_repr import decode_to_mano_inputs
    mano = ManoLayer(str(ROOT / "assets/mano_right.npz"), add_mean=False).eval()

    def fk(p, b):
        dec = decode_to_mano_inputs(p, "mano_full_axis_angle", mano.hands_components, mano.hands_mean)
        return mano(b, dec["global_orient"], dec["local_full_aa"], dec["transl"])[0]
    return fk


@pytest.mark.parametrize("form", ["l1", "gshare"])
def test_no_change_equals_trunk4(form):
    from semkine import pfo2_trunk as P2, pfo_trunk4 as T4
    _, args = _net_and_inputs()
    a, b = _build(T4, form), _build(P2, form)
    assert {k: v.shape for k, v in a.state_dict().items()} == {k: v.shape for k, v in b.state_dict().items()}
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())       # same initial values
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        assert torch.equal(a(*args)[0], b(*args)[0])


def test_controls_keep_parameters_and_change_the_graph():
    from semkine import pfo2_trunk as P2
    base = _build(P2, "l1")
    for v in ("l1+self", "l1+rewire", "l1+diff", "l1+self+diff", "l1+rewire+diff"):
        net = _build(P2, v)
        assert {k: x.shape for k, x in net.state_dict().items()} == {k: x.shape for k, x in base.state_dict().items()}, v
        assert all(torch.equal(net.state_dict()[k], base.state_dict()[k]) for k in base.state_dict()), v
    assert not torch.equal(_build(P2, "l1+rewire").ring_idx, base.ring_idx)
    with pytest.raises(ValueError):
        P2.parse("l1+self+rewire")


def test_diff_messages_vanish_on_the_self_graph():
    from semkine import pfo2_trunk as P2
    _, args = _net_and_inputs()
    a = _build(P2, "l1+self+diff")
    b = _build(P2, "l1+self+diff")
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        b.layers[0].msg.weight.mul_(0.0)
        assert torch.allclose(a(*args)[0], b(*args)[0], atol=1e-6)           # W_m (H_j - H_j) = 0: msg is unused
        c = _build(P2, "l1+diff")
        c.load_state_dict(a.state_dict())
        assert not torch.allclose(a(*args)[0], c(*args)[0], atol=1e-6)        # on the real graph it is used


def test_jacobian_matches_autograd_and_least_squares_recovers():
    from semkine import pfo2_trunk as P2
    fk = _fk()
    g = torch.Generator().manual_seed(0)
    prev = torch.zeros(2, 51)
    prev[:, 2] = 0.45
    prev[:, 3:] = torch.randn(2, 48, generator=g) * 0.2
    betas = torch.randn(2, 10, generator=g) * 0.5
    J = P2.jacobian(fk, prev, betas)
    Ja = torch.autograd.functional.jacobian(lambda p: fk(p.view(1, 51), betas[:1]).reshape(-1), prev[0].clone())
    assert torch.allclose(J[0].reshape(-1, 51), Ja, atol=2e-4, rtol=1e-3)
    x = torch.randn(2, 51, generator=g) * torch.tensor(P2.LS_SCALE) * 0.1
    u = torch.einsum("bvkl,bl->bvk", J, x)
    got = P2.least_squares(J, u, torch.tensor(P2.LS_SCALE), damp=1e-12)
    assert torch.allclose(got, x, atol=1e-4)


def test_disp_starts_as_the_base_function():
    from semkine import pfo2_trunk as P2
    _, args = _net_and_inputs()
    base, net = _build(P2, "l1"), _build(P2, "l1+disp")
    assert set(net.state_dict()) - set(base.state_dict()) == {"disp_head.weight", "disp_head.bias", "disp_in.weight"}
    assert all(torch.equal(net.state_dict()[k], base.state_dict()[k]) for k in base.state_dict())
    fk = _fk()
    B = args[3].shape[0]
    prev = torch.zeros(B, 51)
    prev[:, 2] = 0.45
    prev[:, 3:] = args[5]
    from mano_layer import ManoLayer
    net.pfo2_ctx = {"prev": prev, "betas": torch.zeros(B, 10), "fk": fk,
                    "mano": ManoLayer(str(ROOT / "assets/mano_right.npz"), add_mean=False).eval()}
    with torch.no_grad():
        assert torch.allclose(net(*args)[0], base(*args)[0], atol=0)
    assert net.pfo2_aux["u"].shape == (B, 778, 3)


def test_dual_injection_kinds_and_magnitudes():
    import dataclasses
    import math
    from semkine import pfo2_dual as D
    from semkine.events import EventPacketBatch
    assert dataclasses.is_dataclass(EventPacketBatch)
    g = torch.Generator().manual_seed(0)
    prev = torch.zeros(256, 51)
    prev[:, 3:51] = torch.randn(256, 48, generator=torch.Generator().manual_seed(1)) * 0.3
    out, kind = D.inject(prev, g)
    R0 = D.axis_angle_to_matrix(prev[:, 3:6].double())
    R1 = D.axis_angle_to_matrix(out[:, 3:6].double())
    ang = torch.rad2deg(torch.acos(((R1 @ R0.transpose(1, 2)).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2))
    r15, f15 = kind == 0, kind == 1
    assert torch.allclose(ang[r15], torch.full_like(ang[r15], 15.0), atol=1e-3)
    assert torch.equal(out[r15, 6:51], prev[r15, 6:51]) and torch.equal(out[f15, 3:6], prev[f15, 3:6])
    assert abs(float((out[f15, 6:51] - prev[f15, 6:51]).std()) - 0.15) < 0.01
    assert 0.35 < float(r15.float().mean()) < 0.65 and math.isfinite(float(out.abs().max()))


def test_lbs_jacobian_matches_skinning_finite_differences():
    from mano_layer import ManoLayer, _transform_mat, batch_rodrigues
    from semkine import pfo2_trunk as P2
    mano = ManoLayer(str(ROOT / "assets/mano_right.npz"), add_mean=False).eval().double()
    g = torch.Generator().manual_seed(3)
    prev = torch.zeros(2, 51, dtype=torch.float64)
    prev[:, 0:3] = torch.randn(2, 3, generator=g, dtype=torch.float64) * 0.05
    prev[:, 3:] = torch.randn(2, 48, generator=g, dtype=torch.float64) * 0.3
    betas = torch.randn(2, 10, generator=g, dtype=torch.float64) * 0.5

    def skin(p, frozen_posed, beta):
        B = p.shape[0]
        pose = torch.cat([p[:, 3:6], p[:, 6:51] + mano.hands_mean.view(1, -1)], 1)
        v_shaped = mano.v_template + torch.einsum("bl,mkl->bmk", beta, mano.shapedirs)
        J = torch.einsum("bik,ji->bjk", v_shaped, mano.J_regressor)
        rot = batch_rodrigues(pose.reshape(-1, 3)).view(B, 16, 3, 3)
        rel = J.clone()
        rel[:, 1:] -= J[:, mano._parents_py[1:]]
        mats = _transform_mat(rot.reshape(-1, 3, 3), rel.reshape(-1, 3, 1)).view(B, 16, 4, 4)
        chain = [mats[:, 0]]
        for i in range(1, 16):
            chain.append(chain[mano._parents_py[i]] @ mats[:, i])
        G = torch.stack(chain, 1)
        Jh = torch.cat([J, torch.zeros(B, 16, 1, dtype=p.dtype)], 2)
        T = G.clone()
        T[:, :, :3, 3] = G[:, :, :3, 3] - (G @ Jh.unsqueeze(-1))[:, :, :3, 0]
        Tv = torch.einsum("vj,bjkl->bvkl", mano.weights, T)
        hom = torch.cat([frozen_posed, torch.ones(B, frozen_posed.shape[1], 1, dtype=p.dtype)], 2)
        return (Tv @ hom.unsqueeze(-1)).squeeze(-1)[:, :, :3] + p[:, 0:3].unsqueeze(1)

    for b in range(2):
        pose = torch.cat([prev[b:b + 1, 3:6], prev[b:b + 1, 6:51] + mano.hands_mean.view(1, -1)], 1)
        v_shaped = mano.v_template + torch.einsum("bl,mkl->bmk", betas[b:b + 1], mano.shapedirs)
        rot = batch_rodrigues(pose.reshape(-1, 3)).view(1, 16, 3, 3)
        frozen = v_shaped + torch.einsum("bl,mkl->bmk", (rot[:, 1:] - torch.eye(3, dtype=torch.float64)).reshape(1, -1),
                                         mano.posedirs)
        ref = torch.autograd.functional.jacobian(lambda q: skin(q.view(1, 51), frozen, betas[b:b + 1]).reshape(-1),
                                                 prev[b].clone())
        got = P2.lbs_jacobian(mano, prev[b:b + 1], betas[b:b + 1])[0].reshape(-1, 51).double()
        assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4), float((got - ref).abs().max())
