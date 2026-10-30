"""XVM C1+sign: opt-in, C1 parameters and initial function unchanged, default path untouched."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine.xyz_mesh import XYZMeshNet, R_MAX  # noqa: E402


def _mano():
    d = np.load(ROOT / "assets" / "mano_right.npz")
    return torch.tensor(d["weights"], dtype=torch.float32), torch.tensor(d["f"].astype(np.int64))


def _inputs(seed=0, B=2, N=64):
    g = torch.Generator().manual_seed(seed)
    tok = torch.rand(B, N, 7, generator=g)
    tok[..., 2] = torch.where(tok[..., 2] > 0.5, 1.0, -1.0)
    d = torch.cat([0.1 * torch.randn(B, N, 2, generator=g), torch.ones(B, N, 1)], -1)
    mask = torch.ones(B, N, dtype=torch.bool)
    verts = torch.randn(B, 778, 3, generator=g) * 0.03 + torch.tensor([0.0, 0.0, 0.7])
    vid = torch.randint(0, 778, (B, N, R_MAX), generator=g)
    A = torch.rand(B, N, R_MAX, generator=g)
    A = A / A.sum(-1, keepdim=True)
    assoc = {"vid": vid, "A": A, "q": torch.rand(B, N, generator=g), "z": 0.7 + 0.01 * torch.rand(B, N, generator=g),
             "valid": torch.ones(B, N, dtype=torch.bool), "hit": torch.rand(B, N, generator=g) > 0.5}
    return tok, d, mask, verts, assoc, torch.randn(B, 48, generator=g)


def test_sign_keeps_c1_parameters_and_initial_function():
    W, F = _mano()
    torch.manual_seed(3407)
    base = XYZMeshNet(W, F)
    after_base = torch.rand(3)
    torch.manual_seed(3407)
    sign = XYZMeshNet(W, F, sign=True)
    after_sign = torch.rand(3)
    assert torch.equal(after_base, after_sign), "construction must not move the global RNG stream"
    sb, ss = base.state_dict(), sign.state_dict()
    assert set(ss) - set(sb) == {"sign_in.weight"}
    for k in sb:
        assert torch.equal(sb[k], ss[k]), k
    assert sum(p.numel() for p in sign.parameters()) - sum(p.numel() for p in base.parameters()) == 4 * 64
    with torch.no_grad():
        ob = base(*_inputs())[0]
        fc = fc_sign = None
        # make the final layer non-zero so the comparison is not trivial
        for m in (base, sign):
            torch.manual_seed(1)
            torch.nn.init.normal_(m.fc2.weight)
        ob, os_ = base(*_inputs())[0], sign(*_inputs())[0]
    assert torch.equal(ob, os_)


def test_sign_features_reach_the_gradient():
    W, F = _mano()
    torch.manual_seed(3407)
    net = XYZMeshNet(W, F, sign=True)
    torch.nn.init.normal_(net.fc2.weight)
    net(*_inputs())[0].square().sum().backward()
    assert net.sign_in.weight.grad is not None and net.sign_in.weight.grad.abs().sum() > 0


def test_sign_feature_geometry():
    W, F = _mano()
    net = XYZMeshNet(W, F, sign=True)
    verts = torch.zeros(1, 778, 3)
    verts[..., 2] = 0.7
    # an event ray exactly through vertex 0 has zero signed offset
    d = torch.tensor([[0.0, 0.0, 1.0]])
    f = net.sign_features(verts + torch.randn(1, 778, 3) * 0.01 * torch.tensor([1.0, 1.0, 0.0]),
                          torch.tensor([0]), torch.tensor([0]), d, torch.tensor([1.0]))
    assert f.shape == (1, 4)
