"""XVM operator: vertex-frame features are invariant to in-plane rotation and to depth scaling; transport solves the
weighted least squares exactly; image-frame conversion is consistent."""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import xvm as XV  # noqa: E402


def _frames(theta):
    c, s = math.cos(theta), math.sin(theta)
    e1 = torch.tensor([[c, s]])
    e2 = torch.stack([-e1[..., 1], e1[..., 0]], -1)
    return torch.stack([e1, e2], -1).unsqueeze(0)                       # (1, 1, 2, 2)


def _setup(theta=0.3, scale=1.0, rot=0.0):
    g = torch.Generator().manual_seed(0)
    N = 50
    uv = torch.tensor([[[120.0, 90.0]]])
    off = torch.randn(1, N, 2, generator=g) * 3.0
    c, s = math.cos(rot), math.sin(rot)
    Rr = torch.tensor([[c, -s], [s, c]])
    ev = {"xy": uv + scale * (off @ Rr.T), "age_ms": torch.rand(1, N, generator=g) * 100,
          "pol": torch.where(torch.rand(1, N, generator=g) > 0.5, 1.0, -1.0)}
    R = Rr @ _frames(theta)[0, 0]
    R = R.view(1, 1, 2, 2)
    rho = torch.tensor([[6.0 * scale]])
    return uv, R, rho, ev


def _features(uv, R, rho, ev):
    idx, ok, _ = XV.gather(uv, torch.ones(1, 1, dtype=torch.bool), rho, ev["xy"], torch.ones(1, 50, dtype=torch.bool), 16)
    f = XV.pair_features(uv, R, rho, torch.tensor([[0.2]]), ev, idx, ok, 200.0, torch.zeros(1, 50))
    return f, idx, ok


def test_features_invariant_to_image_rotation_and_depth_scale():
    f0, i0, o0 = _features(*_setup())
    f1, i1, o1 = _features(*_setup(rot=1.1))
    f2, i2, o2 = _features(*_setup(scale=2.5))
    assert torch.equal(i0, i1) and torch.equal(i0, i2)
    assert torch.allclose(f0, f1, atol=1e-5) and torch.allclose(f0, f2, atol=1e-5)


def test_transport_solves_weighted_least_squares():
    g = torch.Generator().manual_seed(1)
    B, V, D = 2, 30, 7
    Bv = torch.randn(B, V, 2, D, generator=g, dtype=torch.float64)
    x = torch.randn(B, D, generator=g, dtype=torch.float64)
    mt = (Bv @ x.view(B, 1, D, 1)).squeeze(-1)
    A = torch.randn(B, V, 2, 2, generator=g, dtype=torch.float64)
    Lt = A @ A.transpose(-1, -2) + 0.1 * torch.eye(2, dtype=torch.float64)
    w = torch.ones(B, V, dtype=torch.float64)
    sol = XV.transport(Bv, mt, Lt, w, torch.eye(D, dtype=torch.float64), lam=1e-9)
    assert torch.allclose(sol, x, atol=1e-6)


def test_to_image_roundtrip():
    R = _frames(0.7)
    m = torch.tensor([[[0.4, -0.2]]])
    L = torch.tensor([[[[2.0, 0.0], [0.5, 1.0]]]])
    rho = torch.tensor([[5.0]])
    mt, Lt = XV.to_image(m, L, R, rho)
    back = (R.transpose(-1, -2) @ mt.unsqueeze(-1)).squeeze(-1) / rho.unsqueeze(-1)
    assert torch.allclose(back, m, atol=1e-6)
    lam = L @ L.transpose(-1, -2)
    assert torch.allclose(R.transpose(-1, -2) @ Lt @ R * 25.0, lam, atol=1e-5)
