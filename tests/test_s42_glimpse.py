"""S42 (trk6) MODEL.GLIMPSE: hand-centric crop of the dense tracker (docs/EXPERIMENT_RECORDS.md#s42-prereg). CPU only."""
import copy
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "model"))
from config import load_config                     # noqa: E402
from model import MNISTModel                       # noqa: E402

BASE = REPO / "configs/dt/dt_dz_l3.yaml"


def _cfg(**model):
    c = load_config(str(BASE))
    c = copy.deepcopy(c)
    c["MODEL"].update(model)
    return c


def _prev(B=3):
    p = torch.zeros(B, 51)
    p[:, 2] = torch.tensor([0.45, 0.55, 0.7])[:B]          # in front of the camera
    p[:, 0] = torch.tensor([0.02, -0.05, 0.08])[:B]
    return p


def _K(B=3):
    K = torch.tensor([[570.0, 0, 320.0], [0, 570.0, 240.0], [0, 0, 1.0]])
    return K.expand(B, 3, 3).clone()


def test_default_has_no_glimpse_and_glimpse_keeps_params():
    a = MNISTModel(_cfg())
    b = MNISTModel(_cfg(GLIMPSE=112))
    assert a.glimpse == 0 and b.glimpse == 112
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_forward_shapes_and_crop_size():
    torch.manual_seed(0)
    m = MNISTModel(_cfg(GLIMPSE=112)).eval()
    x = (torch.rand(3, 180, 240, 2) > 0.9).float() * torch.rand(3, 180, 240, 2)
    betas = torch.zeros(3, 10)
    with torch.no_grad():
        out = m(x, _prev(), betas=betas, camera_K=_K())
        rend, grid = m._glimpse_inputs(_prev(), betas, _K())
    assert out.shape == (3, 51) and torch.isfinite(out).all()
    assert rend.shape == (3, 112, 112, 2) and grid.shape == (3, 224, 224, 2)
    assert m.glimpse_lnes(x, grid).shape == (3, 2, 112, 112)


def test_empty_window_holds_the_previous_state_exactly():
    m = MNISTModel(_cfg(GLIMPSE=112)).eval()
    with torch.no_grad():
        out = m(torch.zeros(3, 180, 240, 2), _prev(), betas=torch.zeros(3, 10), camera_K=_K())
    assert torch.equal(out, _prev())


def test_crop_centre_and_side_follow_the_projection():
    m = MNISTModel(_cfg(GLIMPSE=96, GLIMPSE_KAPPA=1.5, GLIMPSE_SPAN_M=0.2)).eval()
    p, K = _prev(), _K()
    with torch.no_grad():
        verts, _ = m._fk(p, torch.zeros(3, 10))
        g = m.glimpse_geometry(verts, K)
    fx = K[:, 0, 0] * m.render_scale
    z = verts[..., 2].mean(1)
    assert torch.allclose(g["side"], 1.5 * fx * 0.2 / z, rtol=1e-5)
    # crop intrinsics map the centre to the crop centre
    fx2, fy2, cx2, cy2 = g["K"]
    u = fx[:, None] * verts[..., 0] / verts[..., 2] + K[:, 0, 2, None] * m.render_scale
    uc = 0.5 * (u.amin(1) + u.amax(1))
    assert torch.allclose((uc - (K[:, 0, 2] * m.render_scale)) * g["sc"] + cx2, torch.full_like(uc, 48.0), atol=1e-3)


@pytest.mark.parametrize("bad", [dict(GLIMPSE=16), dict(GLIMPSE=112, CAM_PLANES=True), dict(GLIMPSE=112, PREV_RENDER=False),
                                 dict(GLIMPSE_KAPPA=1.5), dict(GLIMPSE=112, GLIMPSE_SPAN_M=0.0)])
def test_bad_configs_raise(bad):
    with pytest.raises(ValueError):
        MNISTModel(_cfg(**bad))


def test_backward_reaches_the_trunk():
    torch.manual_seed(0)
    m = MNISTModel(_cfg(GLIMPSE=64)).train()
    x = (torch.rand(3, 180, 240, 2) > 0.8).float() * torch.rand(3, 180, 240, 2)
    out = m(x, _prev(), betas=torch.zeros(3, 10), camera_K=_K())
    out.pow(2).sum().backward()
    assert m.conv1.weight.grad is not None and m.conv1.weight.grad.abs().sum() > 0
