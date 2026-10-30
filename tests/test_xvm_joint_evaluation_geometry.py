"""Synthetic/public-asset parity of eval vectorization; no recording access."""
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.xvm_joint.evaluation_geometry import (
    FastJointGeometry, fast_sensor_supports, fast_vertex_candidates)
from tools.xvm_joint.geometry import GeometryConfig, JointGeometry, sensor_supports


def assert_tensor_parity(actual, expected, *, atol=0.):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if actual.is_floating_point():
        torch.testing.assert_close(actual, expected, atol=atol, rtol=0)
    else:
        assert torch.equal(actual, expected)


def assert_support_parity(field, config):
    expected = sensor_supports(field, config)
    actual = fast_sensor_supports(field, config)
    assert actual.keys() == expected.keys()
    for name in expected:
        assert_tensor_parity(actual[name], expected[name])


@pytest.mark.parametrize('kind', ['integer', 'empty', 'flat', 'fractional', 'large'])
def test_sensor_supports_partial_tiles_ties_centroids_and_fallback(kind):
    generator = torch.Generator().manual_seed(2007)
    c = GeometryConfig(max_supports=20, height=35, width=51)
    f = torch.rand((3, 16, c.height, c.width), generator=generator)
    f[:, 6:] = torch.randint(0, 5, (3, 10, c.height, c.width), generator=generator).float()
    if kind == 'empty':
        f[:, 6:] = 0
    elif kind == 'flat':
        f[:, 6:] = 2
    elif kind == 'fractional':
        f[:, 6:] *= .2
    elif kind == 'large':
        f[:, 6:] *= 2**20
    assert_support_parity(f, c)


def test_sensor_supports_180_by_240_raw_count_contract():
    generator = torch.Generator().manual_seed(17)
    f = torch.rand((2, 16, 180, 240), generator=generator)
    f[:, 6:] = torch.randint(0, 4, (2, 10, 180, 240), generator=generator).float()
    assert_support_parity(f, GeometryConfig())


def test_candidate_equal_distance_uses_vertex_id_and_first_vertex_per_part():
    c = GeometryConfig(max_supports=2, candidates_per_support=3)
    uv = torch.full((1, 778, 2), 100.)
    uv[0, :, 1] = 90.
    part = torch.arange(778) % 16
    # IDs 0..15 tie exactly; the first 3 IDs must win across body parts.
    support = dict(xy=torch.tensor([[[100., 90.], [100., 90.]]]),
                   normal_valid=torch.tensor([[True, False]]))
    K = torch.tensor([[[100., 0., 120.], [0., 100., 90.], [0., 0., 1.]]])
    ids, valid = fast_vertex_candidates(support, uv, uv, torch.ones(1, 778, dtype=torch.bool),
        torch.ones(1, 778, dtype=torch.bool), torch.full((1, 778), 10.), part, K, c)
    assert torch.equal(ids, torch.tensor([[0, 1, 2, 0, 0, 0]]))
    assert torch.equal(valid, torch.tensor([[True, True, True, False, False, False]]))


class SyntheticBase:
    render_scale = .375

    def __init__(self):
        x = torch.linspace(-.05, .05, 30)
        y = torch.linspace(-.05, .05, 26)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        self.template = torch.stack((xx.flatten()[:778], yy.flatten()[:778], torch.zeros(778)), -1)
        self.template[:4, :2] = torch.tensor([[-.05, -.05], [.05, -.05], [.05, .05], [-.05, .05]])
        weights = torch.zeros(778, 16)
        weights[torch.arange(778), torch.arange(778) % 16] = 1
        self.mano = SimpleNamespace(f=torch.tensor([[0, 1, 2], [0, 2, 3]]), weights=weights)

    def _intrinsics(self, K):
        return tuple(K[:, a, b]*self.render_scale for a, b in ((0, 0), (1, 1), (0, 2), (1, 2)))

    def _fk(self, q, beta):
        vertices = self.template.to(q)[None].expand(len(q), -1, -1).clone()+q[:, None, :3]
        vertices[..., 0] += .01*q[:, None, 6]
        vertices[..., 1] += .02*q[:, None, 7]
        return vertices, vertices[:, :21]


def prepared_fixtures():
    generator = torch.Generator().manual_seed(3407)
    q0 = torch.zeros(2, 51)
    q0[:, 2] = torch.tensor([.6, .7])
    q0[:, 3:] = torch.randn((2, 48), generator=generator)*.03
    q1 = q0+torch.randn((2, 51), generator=generator)*.002
    K = torch.tensor([[[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]]]).repeat(2, 1, 1)
    f0 = torch.rand((2, 16, 180, 240), generator=generator)
    f1 = torch.rand((2, 16, 180, 240), generator=generator)
    for f in (f0, f1):
        f[:, 6:] = 0
        f[:, 6:, 65:115, 90:150] = torch.randint(0, 4, (2, 10, 50, 60), generator=generator).float()
    return f0, f1, q0, q1, K


def assert_prepared_parity(base, *, invalid=False):
    c = GeometryConfig(max_supports=6, candidates_per_support=3)
    args = list(prepared_fixtures())
    if invalid:
        args[2][:, 2] = -.6
        args[3][:, 2] = -.6
    expected = JointGeometry(base, c).prepare(*args)
    actual = FastJointGeometry(base, c).prepare(*args)
    for item in fields(expected):
        a, e = getattr(actual, item.name), getattr(expected, item.name)
        if isinstance(e, torch.Tensor):
            assert_tensor_parity(a, e)
        else:
            assert a == e
    if not invalid:
        assert actual.observed.any(), 'parity must exercise observed candidates'
    # The trained local readout and joint solver must see identical inputs too.
    from tools.xvm_joint.model import JointLocalOperator
    torch.manual_seed(901)
    local = JointLocalOperator().eval()
    with torch.no_grad():
        local.mean_head.weight.normal_(std=.003)
        local.association_head[-1].weight.normal_(std=.003)
        local.mean_head.bias.fill_(.001)
        ea, eb, eaux = local(**expected.model_kwargs)
        aa, ab, aaux = local(**actual.model_kwargs)
    for a, e in ((aa, ea), (ab, eb), (aaux['weights'], eaux['weights'])):
        assert_tensor_parity(a, e)


def test_full_prepare_and_local_update_synthetic_exact_parity():
    assert_prepared_parity(SyntheticBase())


def test_full_prepare_invalid_reference_exact_parity():
    assert_prepared_parity(SyntheticBase(), invalid=True)


def test_full_prepare_and_local_update_public_mano_exact_parity():
    from model.mano_layer import ManoLayer

    class RealBase(SyntheticBase):
        def __init__(self):
            asset = Path(__file__).resolve().parents[1]/'assets/mano_right.npz'
            self.mano = ManoLayer(asset, add_mean=True).cpu()
            self.mano.fast_chain = True

        def _fk(self, q, beta):
            return self.mano(beta, q[:, 3:6], q[:, 6:], q[:, :3])

    assert_prepared_parity(RealBase())
