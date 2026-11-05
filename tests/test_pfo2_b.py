"""PFO2 arm (b), first-moment slot pooling (semkine/pfo2_trunk_b.py): CPU tests."""
from __future__ import annotations

import torch

from tests.test_pfo import _net_and_inputs
from tests.test_pfo2_trunk import _build


def test_first_moments_match_the_p0d_reference():
    from semkine.pfo2_trunk_b import first_moments
    from tools.pfo2.p0d import moments
    g = torch.Generator().manual_seed(0)
    B, V, J, C = 3, 9, 4, 5
    H, S = torch.randn(B, V, C, generator=g), (torch.rand(B, V, generator=g) > 0.3).float()
    W = torch.rand(V, J, generator=g)
    verts, joints = torch.randn(B, V, 3, generator=g), torch.randn(B, J, 3, generator=g)
    R = torch.linalg.qr(torch.randn(B, 3, 3, generator=g))[0]
    _, ref = moments(H, S, W, verts, joints, R)
    assert torch.allclose(first_moments(H, S, W, verts, joints, R), ref, atol=1e-5)


def test_mom_starts_as_the_base_function():
    from semkine import pfo2_trunk as P2, pfo2_trunk_b as PB
    _, args = _net_and_inputs()
    base, net = _build(P2, "gshare"), _build(PB, "gshare+mom")
    assert set(net.state_dict()) - set(base.state_dict()) == {"mom_proj.weight"}
    assert all(torch.equal(net.state_dict()[k], base.state_dict()[k]) for k in base.state_dict())
    with torch.no_grad():
        assert torch.equal(net(*args)[0], base(*args)[0])
        torch.nn.init.normal_(net.mom_proj.weight, std=0.01)
        assert not torch.allclose(net(*args)[0], base(*args)[0])
    from semkine import xyz_mesh as XM
    assert XM.lbs_slot_pool.__module__ == "semkine.xyz_mesh" and "forward" not in vars(net.slot_proj)
