"""Root tracking R3 (ABS_TRACK): the blend reduces to the delta head at a = 0 and to the absolute head at
a = 1, an event-free packet holds prev, and both heads receive gradient from the training loss."""
import copy
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model"))
from config import load_config                     # noqa: E402
from model import MNISTModel                       # noqa: E402


def _model(a_r, a_f):
    cfg = copy.deepcopy(load_config(REPO / "configs/rt/rt_cnnar_2k.yaml"))
    cfg["MODEL"]["ABS_TRACK_ALPHA_ROOT"], cfg["MODEL"]["ABS_TRACK_ALPHA_REST"] = a_r, a_f
    torch.manual_seed(0)
    return MNISTModel(cfg).eval()


def _inputs():
    g = torch.Generator().manual_seed(0)
    x = (torch.rand(2, 180, 240, 2, generator=g) > 0.97).float() * torch.rand(2, 180, 240, 2, generator=g)
    prev = torch.zeros(2, 51)
    prev[:, 2] = 0.4
    prev[:, 3:6] = torch.tensor([[0.3, -0.2, 0.1], [-1.0, 0.5, 2.0]])
    return x, prev


def test_blend_endpoints_and_empty_packet():
    x, prev = _inputs()
    for a, which in ((0.0, 1), (1.0, 0)):
        m = _model(a, a)
        with torch.no_grad():
            out = m(x, prev)
        x_abs, x_trk = m._abs_track_parts
        ref = (x_abs, x_trk)[which]
        assert torch.allclose(out[:, :3], ref[:, :3], atol=1e-6) and torch.allclose(out[:, 6:], ref[:, 6:], atol=1e-6)
        assert torch.allclose(out[:, 3:6], ref[:, 3:6], atol=1e-4)
    m = _model(0.5, 0.5)
    with torch.no_grad():
        out = m(torch.zeros_like(x), prev)
    assert torch.equal(out, prev)


def test_both_heads_train():
    m = _model(0.5, 0.5).train()
    x, prev = _inputs()
    m(x, prev)
    x_abs, x_trk = m._abs_track_parts
    y = torch.zeros(2, 51)
    loss = m._compute_loss(x_abs, y)[0] + m._compute_loss(x_trk, y)[0]
    loss.backward()
    g = m.rn.fc.weight.grad
    assert g is not None and float(g[:51].abs().sum()) > 0 and float(g[51:].abs().sum()) > 0
    assert m.prev_mlp[2].weight.grad is not None
