"""Test the repair auxiliaries' geometry, denominators and gradient isolation."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'model')]

from semkine.xkf_direct import dgp_translation  # noqa: E402
from semkine.xkf_repair_losses import direct_aux_losses  # noqa: E402


class GeometryModel:
    """Translation-equivariant FK stand-in; the MANO graph has this contract."""
    def __init__(self):
        generator = torch.Generator().manual_seed(18)
        self.joints0 = torch.randn(2, 21, 3, generator=generator) * 0.035
        self.joints0 += torch.tensor([0.09, 0.006, 0.006])

    def _resolve_betas_K(self, values, betas, camera):
        return betas, camera

    def _fk(self, state, betas):
        joints = self.joints0[:len(state)] + state[:, None, :3]
        return joints, joints

    def _intrinsics(self, camera):
        return camera[:, 0, 0], camera[:, 1, 1], camera[:, 0, 2], camera[:, 1, 2]


def inputs(*, perturb=True, principal_x=120.):
    model = GeometryModel()
    target = torch.zeros(2, 51)
    target[:, :3] = torch.tensor([[.01, -.02, .62], [-.02, .03, .75]])
    target.requires_grad_(True)
    camera = torch.tensor([[225., 0., principal_x], [0., 224., 90.], [0., 0., 1.]]).repeat(2, 1, 1)
    j = model.joints0 + target.detach()[:, None, :3]
    uv = j[..., :2] / j[..., 2:3]
    kp = uv * torch.tensor([225., 224.]) + torch.tensor([principal_x, 90.])
    predj = model.joints0.clone()
    if perturb:
        kp += torch.sin(torch.arange(kp.numel()).reshape_as(kp)) * 1.7
        predj += torch.cos(torch.arange(predj.numel()).reshape_as(predj)) * .004
    kp.requires_grad_(True)
    predj.requires_grad_(True)
    weights = torch.linspace(.2, 1.2, 21).repeat(2, 1).requires_grad_(True)
    batch = SimpleNamespace(target=target, betas=torch.zeros(2, 10), camera_K=camera)
    aux = {'kp_px': kp, 'joints0': predj, 'w': weights}
    return model, target.detach().clone(), aux, batch


@pytest.mark.parametrize('key,expected_index', [('d_aux_2d', 0), ('d_aux_3d', 1)])
def test_auxiliary_gradient_reaches_only_its_predicted_geometry(key, expected_index):
    model, state, aux, batch = inputs()
    parts = direct_aux_losses(model, state, aux, batch)
    leaves = [aux['kp_px'], aux['joints0'], aux['w'], batch.target]
    grads = torch.autograd.grad(parts[key], leaves, allow_unused=True)
    for i, grad in enumerate(grads):
        if i == expected_index:
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        else:
            assert grad is None


@pytest.mark.parametrize('principal_x', [120., 1000.])
def test_exact_geometry_has_near_zero_loss_even_for_image_outside_joints(principal_x):
    model, state, aux, batch = inputs(perturb=False, principal_x=principal_x)
    if principal_x == 1000.:
        assert (aux['kp_px'][..., 0] > 240).all()
    parts = direct_aux_losses(model, state, aux, batch)
    assert set(parts) == {'d_aux_2d', 'd_aux_3d'}
    assert all(torch.isfinite(loss) and loss < 1e-10 for loss in parts.values())


def test_auxiliary_uses_all_b_times_three_coordinates():
    model, state, aux, batch = inputs()
    parts = direct_aux_losses(model, state, aux, batch)
    expected_uv = ((aux['kp_px'] - torch.tensor([120., 90.])) / torch.tensor([225., 224.]))
    expected = dgp_translation(model.joints0, expected_uv, aux['w'].detach())
    assert torch.allclose(parts['d_aux_2d'], (expected - batch.target.detach()[:, :3]).square().sum() / 6)


@pytest.mark.parametrize('invalid', ['behind', 'nonfinite', 'camera'])
def test_invalid_training_geometry_is_explicit_failure(invalid):
    model, state, aux, batch = inputs()
    with torch.no_grad():
        if invalid == 'behind':
            batch.target[0, 2] = -.5
        elif invalid == 'nonfinite':
            batch.target[0, 5] = float('nan')
        else:
            batch.camera_K[0, 0, 0] = 0
    with pytest.raises(ValueError):
        direct_aux_losses(model, state, aux, batch)
