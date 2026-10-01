"""semkine.anchored: the blend's endpoints and geometry, the event-free hold, and both composites."""
import math

import torch
from torch import nn

from semkine.anchored import AnchoredTracker, CausalFilter, anchor_blend


class _Const(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, x, prevpos, betas=None, camera_K=None):
        return self.v.expand(prevpos.shape[0], -1).clone()


def _pose(rot, t=(0.0, 0.0, 0.4), fing=0.1):
    p = torch.full((1, 51), fing)
    p[0, :3] = torch.tensor(t)
    p[0, 3:6] = torch.tensor(rot)
    return p


def test_blend_endpoints_and_geodesic_midpoint():
    a, b = _pose([0.0, 0.0, 0.0], fing=0.0), _pose([0.0, 0.0, math.radians(30)], t=(0.1, 0.0, 0.5), fing=0.2)
    assert torch.allclose(anchor_blend(a, b, 0.0, 0.0), a, atol=1e-6)
    assert torch.allclose(anchor_blend(a, b, 1.0, 1.0), b, atol=1e-6)
    m = anchor_blend(a, b, 0.5, 0.5)
    assert abs(float(m[0, 5]) - math.radians(15)) < 1e-5 and abs(float(m[0, 10]) - 0.1) < 1e-6
    t = anchor_blend(a, b, 0.5, 0.5, a_trans=0.0)
    assert torch.allclose(t[:, :3], a[:, :3]) and torch.allclose(t[:, 3:], m[:, 3:])


def test_composites_hold_on_empty_and_blend_otherwise():
    meas, trk = _pose([0.0, 0.2, 0.0], fing=0.3), _pose([0.0, 0.0, 0.0], fing=0.1)
    prev = _pose([0.1, 0.0, 0.0], fing=-0.1)
    x = torch.zeros(1, 180, 240, 2)
    x[0, 10, 10, 0] = 0.5
    f = CausalFilter(_Const(meas), 0.5, 0.5).eval()
    assert torch.allclose(f(x, prev), anchor_blend(prev, meas, 0.5, 0.5, 1.0), atol=1e-6)
    assert torch.equal(f(torch.zeros_like(x), prev), prev)
    g = AnchoredTracker(_Const(meas), _Const(trk), 0.4, 0.6).eval()
    assert torch.allclose(g(x, prev), anchor_blend(trk, meas, 0.4, 0.6), atol=1e-6)
    assert torch.equal(g(torch.zeros_like(x), prev), prev)
