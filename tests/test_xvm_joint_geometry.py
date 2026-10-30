"""Synthetic source/units/Jacobian checks; no recorded data or GPU needed."""
from types import SimpleNamespace

import torch

from tools.xvm_joint.geometry import GeometryConfig, JointGeometry, sensor_supports, triangle_silhouette


class SyntheticBase:
    """Exact known linear geometry with 778 proxy vertices and a square mesh."""
    render_scale = .375

    def __init__(self):
        x = torch.linspace(-.05, .05, 30)
        y = torch.linspace(-.05, .05, 26)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        self.template = torch.stack((xx.flatten()[:778], yy.flatten()[:778],
                                     torch.zeros(778)), -1)
        self.template[:4, :2] = torch.tensor([[-.05, -.05], [.05, -.05],
                                             [.05, .05], [-.05, .05]])
        weights = torch.zeros(778, 16)
        weights[torch.arange(778), torch.arange(778) % 16] = 1
        self.mano = SimpleNamespace(f=torch.tensor([[0, 1, 2], [0, 2, 3]]), weights=weights)

    def _intrinsics(self, K):
        return tuple(K[:, a, b]*self.render_scale for a, b in ((0, 0), (1, 1), (0, 2), (1, 2)))

    def _fk(self, q, beta):
        vertices = self.template.to(q)[None].expand(len(q), -1, -1).clone()+q[:, None, :3]
        # Two explicit rotation-slot proxy effects allow independent known derivatives.
        vertices[..., 0] += .01*q[:, None, 6]
        vertices[..., 1] += .02*q[:, None, 7]
        return vertices, vertices[:, :21]


def fixtures():
    q = torch.zeros(1, 51)
    q[:, 2] = .6
    K = torch.tensor([[[200., 0., 320.], [0., 200., 240.], [0., 0., 1.]]])
    field = torch.zeros(1, 16, 180, 240)
    # A finite bright stripe has a reliable horizontal edge normal.
    field[:, 6, 87:94, 116:121] = 4
    return q, K, field


def test_supports_ignore_history_age_and_have_sensor_normals():
    config = GeometryConfig(max_supports=3, candidates_per_support=2)
    q, K, field = fixtures()
    source = sensor_supports(field, config)
    assert source['normal_valid'].any()
    n = source['normal'][source['normal_valid']]
    assert torch.allclose(n.norm(dim=-1), torch.ones(len(n)), atol=1e-6)
    empty = torch.zeros_like(field)
    empty[:, :6] = .8
    old_only = sensor_supports(empty, config)
    assert not old_only['normal_valid'].any()
    assert torch.equal(old_only['source_ids'], torch.full_like(old_only['source_ids'], -1))


def test_camera_scaled_once_and_physical_jacobian_units():
    q, K, _ = fixtures()
    geo = JointGeometry(SyntheticBase(), GeometryConfig(max_supports=1, candidates_per_support=1))
    event_K = geo.event_intrinsics(K)
    assert torch.equal(event_K[0], torch.tensor([[75., 0., 120.], [0., 75., 90.], [0., 0., 1.]]))
    ids = torch.tensor([[6, 100]])
    J = geo.projected_jacobian(q, event_K, ids)
    assert J.shape == (1, 2, 2, 51)
    assert torch.allclose(J[..., 0, 0], torch.full((1, 2), 125.), atol=.1)
    assert torch.allclose(J[..., 1, 1], torch.full((1, 2), 125.), atol=.1)
    assert torch.allclose(J[..., 0, 6], torch.full((1, 2), 1.25), atol=.02)
    assert torch.allclose(J[..., 1, 7], torch.full((1, 2), 2.5), atol=.02)
    assert torch.equal(J[..., 0, 8:], torch.zeros_like(J[..., 0, 8:]))


def test_true_triangle_distance_and_same_vertex_target_semantics():
    q, K, field = fixtures()
    geo = JointGeometry(SyntheticBase(), GeometryConfig(max_supports=2, candidates_per_support=3))
    prepared = geo.prepare(field.clone(), field, q, q.clone(), K)
    assert prepared.patches.shape == (1, 6, 37, 16, 16)
    assert prepared.observed.any()
    assert torch.isfinite(prepared.patches).all()
    assert prepared.model_kwargs['source_ids'].shape == (1, 6)
    ids = prepared.vertex_ids[prepared.observed]
    for source in prepared.source_ids[prepared.observed].unique():
        use = prepared.observed & (prepared.source_ids == source)
        parts = geo.part[prepared.vertex_ids[use]]
        assert len(parts.unique()) == len(parts)
    vertices, _ = geo._fk(q)
    sdf, valid = geo.silhouette_distance(vertices, prepared.K_event)
    assert valid.all()
    assert sdf[0, 90, 120] < 0
    assert sdf[0, 0, 0] > 0
    gt0 = q.clone();gt0[:, 0] += .002
    gt1 = gt0.clone();gt1[:, 0] += .003
    target = geo.targets(prepared, gt0, gt1)
    mask = target['supervised']
    assert torch.allclose(target['target_a_px'][mask], prepared.normal[..., 0][mask]*.25, atol=.002)
    assert torch.allclose(target['target_d_px'][mask], prepared.normal[..., 0][mask]*.375, atol=.002)
    assert target['target_kind'].startswith('beta0')


def test_invalid_camera_reference_produces_no_local_records():
    q, K, field = fixtures()
    geo = JointGeometry(SyntheticBase(), GeometryConfig(max_supports=1, candidates_per_support=1))
    bad = q.clone();bad[:, 2] = -.6
    prepared = geo.prepare(field, field, bad, bad, K)
    assert not prepared.observed.any()
    assert not prepared.candidate_mask.any()
    assert torch.count_nonzero(prepared.patches) == 0
    assert torch.isfinite(prepared.B0).all()
    assert torch.equal(prepared.B0, torch.zeros_like(prepared.B0))


def test_real_mano_asset_jacobian_matches_autograd():
    from pathlib import Path
    from model.mano_layer import ManoLayer

    class RealBase:
        render_scale = .375

        def __init__(self):
            asset = Path(__file__).resolve().parents[1]/'assets/mano_right.npz'
            self.mano = ManoLayer(asset, add_mean=True).cpu()
            self.mano.fast_chain = True

        _intrinsics = SyntheticBase._intrinsics

        def _fk(self, q, beta):
            return self.mano(beta, q[:, 3:6], q[:, 6:], q[:, :3])

    q, K, _ = fixtures()
    q[:, 3:6] = torch.tensor([.03, -.02, .04])
    base = RealBase()
    geo = JointGeometry(base, GeometryConfig(max_supports=1, candidates_per_support=1))
    ids = torch.tensor([[12, 320]])
    event_K = geo.event_intrinsics(K)
    finite = geo.projected_jacobian(q, event_K, ids)
    q_grad = q.clone().requires_grad_()
    vertices, _ = base._fk(q_grad, torch.zeros(1, 10))
    z = vertices[..., 2]
    uv = torch.stack((75*vertices[..., 0]/z+120, 75*vertices[..., 1]/z+90), -1)
    rows = []
    for vid in ids[0]:
        rows.append(torch.stack([torch.autograd.grad(uv[0, vid, axis], q_grad,
                                 retain_graph=True)[0][0] for axis in range(2)]))
    actual = torch.stack(rows)[None]
    assert torch.isfinite(actual).all()
    # Float32 pixel differencing is the largest source of error at these steps.
    assert torch.allclose(finite, actual, atol=.10, rtol=.025)


def test_near_plane_crossing_invalidates_whole_fd_record():
    q, K, _ = fixtures()
    q[:, 2] = .03005
    geo = JointGeometry(SyntheticBase(), GeometryConfig(max_supports=1, candidates_per_support=1))
    J, valid = geo.projected_jacobian(q, geo.event_intrinsics(K), torch.tensor([[4]]), return_valid=True)
    assert not valid.any()
    assert torch.equal(J, torch.zeros_like(J))


def test_local_binary_triangle_mask_matches_original_constant_albedo_raster():
    import numpy as np
    from tools.s43_br.geometry import fk_geometry, _mano, rasterize

    # Only the public MANO asset and declared synthetic states are used.
    states = np.zeros((3, 51), np.float32)
    states[:, 2] = [.6, .45, .8]
    states[1, :2] = [.025, -.012]
    states[1, 3:6] = [.15, -.1, .07]
    states[2, 6] = .2
    vertices, _ = fk_geometry(states, np.zeros(10, np.float32))
    faces = _mano().f.cpu().numpy()
    camera = np.asarray([[200., 0., 120.], [0., 200., 90.], [0., 0., 1.]])
    for v in vertices:
        original = rasterize(v, faces, camera, (180, 240),
                             np.full(778, .8), background=.1, near=.03) > .45
        local = triangle_silhouette(v, faces, camera, (180, 240), near=.03)
        np.testing.assert_array_equal(local, original)


def test_silhouette_rendering_survives_legacy_nonpackage_model_namespace(monkeypatch):
    import sys
    from types import ModuleType

    q, K, _ = fixtures()
    geo = JointGeometry(SyntheticBase(), GeometryConfig(max_supports=1, candidates_per_support=1))
    vertices, _ = geo._fk(q)
    # The legacy Lightning loader can occupy this top-level name with a module.
    # Production silhouette construction must never import model.mano_layer.
    monkeypatch.setitem(sys.modules, 'model', ModuleType('model'))
    assert not hasattr(sys.modules['model'], '__path__')
    sdf, valid = geo.silhouette_distance(vertices, geo.event_intrinsics(K))
    assert valid.all()
    assert sdf[0, 90, 120] < 0
    assert torch.isfinite(sdf).all()
