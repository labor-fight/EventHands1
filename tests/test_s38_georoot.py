#!/usr/bin/env python3
"""S38 (`semkine.geo_root`, docs/S38_GEOROOT_PREREG.md): contract tests of the geometric root solve.

What must hold:

  * topology: every interior edge has two distinct faces; boundary edges (the wrist opening) are dropped;
  * the analytic Jacobian rows agree with central differences of the projection, for all six DOFs;
  * no events, events far from the hand, or all-zero residuals give exactly the identity twist --
    the geometry multiplies the evidence and never produces a correction on its own;
  * `apply_twist` is a rotation about the wrist plus a translation and touches the six root entries only;
  * the contour normals point out of the silhouette;
  * from events sampled on the true contour of a real pose, a root error injected into the anchor is
    reduced: the contour residual collapses and the depth error shrinks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from mano_layer import ManoLayer                                    # noqa: E402
from pose_repr import decode_to_mano_inputs                         # noqa: E402
from semkine import geo_root as GR                                  # noqa: E402
from semkine.dataset import _read_meta51                            # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALE = 0.375
SEQ = REPO / "data/hand_data51/train/ch_local"


@pytest.fixture(scope="module")
def mano():
    return ManoLayer(REPO / "assets/mano_right.npz", add_mean=False).to(DEV).eval()


@pytest.fixture(scope="module")
def scene(mano):
    """Real poses of a training sequence with its betas and render-frame intrinsics."""
    pos = _read_meta51(str(SEQ) + ".meta")
    aux = np.load(str(SEQ) + "_aux.npz", allow_pickle=True)
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=DEV).view(1, -1)
    Km = np.asarray(aux["camera_K"], np.float64)
    K = GR.Intrinsics(Km[0, 0] * SCALE, Km[1, 1] * SCALE, Km[0, 2] * SCALE, Km[1, 2] * SCALE)
    runs = np.asarray(aux["valid_runs_ms"]).reshape(-1, 2)
    t = np.linspace(runs[0][0] + 2000, runs[0][1] - 2000, 3).astype(int)
    states = torch.from_numpy(np.asarray(pos[t], np.float32)).to(DEV)

    def fk(p51):
        p = p51.float().view(-1, 51)
        d = decode_to_mano_inputs(p, "mano_full_axis_angle", mano.hands_components, mano.hands_mean)
        v, j = mano(betas.expand(p.shape[0], -1), d["global_orient"], d["local_full_aa"], d["transl"])
        return v.float(), j.float()

    faces = mano.f.long().to(DEV)
    return {"states": states, "fk": fk, "K": K, "topo": (faces, *GR.edge_topology(faces))}


def _contour_events(scene, state, per_seg=4, noise=0.3, seed=0):
    V, _ = scene["fk"](state)
    faces, edges, adj = scene["topo"]
    C = GR.occluding_contour(V[0], faces, edges, adj, scene["K"], 0.005)
    g = torch.Generator(device=DEV).manual_seed(seed)
    lam = torch.rand(C.pa.shape[0], per_seg, device=DEV, generator=g)
    pts = (C.pa[:, None] + lam[..., None] * (C.pb - C.pa)[:, None]).reshape(-1, 2)
    pts = pts + noise * torch.randn(pts.shape, device=DEV, generator=g)
    return pts[:, 0], pts[:, 1]


def _perturb(state, rot_deg, t_mm):
    R = GR.so3_exp(torch.tensor(np.radians(rot_deg), dtype=torch.float64, device=DEV))
    return GR.apply_twist(state.view(1, -1), R, torch.tensor(np.asarray(t_mm) * 1e-3, device=DEV))


# ----------------------------------------------------------------------------- geometry
def test_edge_topology_keeps_interior_edges_with_two_faces(mano):
    faces = mano.f.long()
    edges, adj = GR.edge_topology(faces)
    assert (adj[:, 0] != adj[:, 1]).all()
    for (a, b), (f0, f1) in zip(edges.tolist()[::97], adj.tolist()[::97]):
        assert {a, b} <= set(faces[f0].tolist()) and {a, b} <= set(faces[f1].tolist())
    all_e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).sort(1).values
    n_boundary = int((torch.unique(all_e, dim=0, return_counts=True)[1] == 1).sum())
    assert n_boundary > 0, "MANO's wrist opening should leave boundary edges"
    assert 2 * edges.shape[0] + n_boundary == 3 * faces.shape[0]


def test_jacobian_rows_match_central_differences():
    g = torch.Generator().manual_seed(0)
    K = GR.Intrinsics(226.3, 226.1, 121.9, 90.8)
    X = torch.rand(64, 3, generator=g, dtype=torch.float64) * torch.tensor([0.2, 0.2, 0.3], dtype=torch.float64) \
        + torch.tensor([-0.1, -0.1, 0.4], dtype=torch.float64)
    n = torch.nn.functional.normalize(torch.randn(64, 2, generator=g, dtype=torch.float64), dim=1)
    c = torch.tensor([0.02, 0.05, 0.55], dtype=torch.float64)
    A = GR.jacobian_rows(X, n, c, K)
    for k in range(6):
        h = 1e-6
        xi = torch.zeros(6, dtype=torch.float64)
        xi[k] = h

        def moved(sign):
            R = GR.so3_exp(sign * xi[:3])
            return (R @ (X - c).T).T + c + sign * xi[3:]

        du = ((GR.project(moved(1.0), K) - GR.project(moved(-1.0), K)) * n).sum(-1) / (2 * h)
        assert torch.allclose(du, A[:, k], rtol=1e-5, atol=1e-6), GR.DOF[k]


def test_contour_normals_point_out_of_the_silhouette(scene):
    faces, edges, adj = scene["topo"]
    K = scene["K"]
    for s in scene["states"]:
        V, _ = scene["fk"](s)
        C = GR.occluding_contour(V[0], faces, edges, adj, K, 0.005)
        assert C.pa.shape[0] > 50
        pm, zm = 0.5 * (C.pa + C.pb), 0.5 * (C.xa[:, 2] + C.xb[:, 2])
        z_out = GR.raycast_depth(V[0], faces, (pm + 1.5 * C.n)[:, 0], (pm + 1.5 * C.n)[:, 1], K)
        z_in = GR.raycast_depth(V[0], faces, (pm - 1.5 * C.n)[:, 0], (pm - 1.5 * C.n)[:, 1], K)
        assert float(((z_out > zm + 0.005) | torch.isinf(z_out)).float().mean()) > 0.75
        assert float((z_in <= zm + 0.005).float().mean()) > 0.85


# ----------------------------------------------------------------------------- contracts
def test_zero_evidence_gives_exactly_no_correction(scene):
    P = GR.GeoRootParams()
    A = torch.randn(200, 6, dtype=torch.float64, device=DEV)
    assert torch.equal(GR.solve_twist(A, torch.zeros(200, dtype=torch.float64, device=DEV), P),
                       torch.zeros(6, dtype=torch.float64, device=DEV))
    assert torch.equal(GR.solve_twist(A[:10], torch.randn(10, dtype=torch.float64, device=DEV), P),
                       torch.zeros(6, dtype=torch.float64, device=DEV))
    s = scene["states"][0].view(1, -1)
    far = torch.full((500,), -500.0, device=DEV)
    for px, py in ((far, far), (far[:0], far[:0])):
        R, v, n = GR.root_correction(scene["fk"], s, [torch.arange(px.shape[0], device=DEV)], px, py,
                                     scene["K"], scene["topo"], P)
        assert n == 0
        assert torch.equal(R, torch.eye(3, dtype=torch.float64, device=DEV))
        assert torch.equal(v, torch.zeros(3, dtype=torch.float64, device=DEV))


def test_apply_twist_rotates_about_the_wrist_and_touches_the_root_only(scene):
    s = scene["states"][1].view(1, -1)
    V, J = scene["fk"](s)
    R = GR.so3_exp(torch.tensor(np.radians([7.0, -5.0, 11.0]), dtype=torch.float64, device=DEV))
    p = GR.apply_twist(s, R, torch.zeros(3, dtype=torch.float64, device=DEV))
    V2, J2 = scene["fk"](p)
    assert float((J2[0, 0] - J[0, 0]).norm()) < 1e-6
    expect = (R.float() @ (V[0] - J[0, 0]).T).T + J[0, 0]
    assert float((V2[0] - expect).norm(dim=-1).max()) < 2e-6
    assert torch.equal(p[:, 6:], s[:, 6:])
    v = torch.tensor([0.004, -0.003, 0.02], dtype=torch.float64, device=DEV)
    V3, _ = scene["fk"](GR.apply_twist(s, torch.eye(3, dtype=torch.float64, device=DEV), v))
    assert float((V3[0] - V[0] - v.float()).norm(dim=-1).max()) < 1e-6


def test_interpolate_state_hits_both_ends(scene):
    a, b = scene["states"][0], scene["states"][2]
    assert torch.allclose(GR.interpolate_state(a, b, 0.0), a, atol=1e-5)
    e = GR.interpolate_state(a, b, 1.0)
    assert torch.allclose(e[6:], b[6:], atol=1e-6) and torch.allclose(e[:3], b[:3], atol=1e-6)
    Rb, Re = GR.so3_exp(b[3:6].double()), GR.so3_exp(e[3:6].double())
    assert float((Rb - Re).abs().max()) < 1e-5


# ----------------------------------------------------------------------------- recovery
def test_injected_root_error_is_reduced_from_true_contour_events(scene):
    faces, edges, adj = scene["topo"]
    K = scene["K"]
    for i, truth in enumerate(scene["states"]):
        px, py = _contour_events(scene, truth, seed=i)
        anchor = _perturb(truth, [4.0, -4.0, 3.0], [5.0, -4.0, 25.0])
        R, v, n = GR.root_correction(scene["fk"], anchor, [torch.arange(px.shape[0], device=DEV)], px, py,
                                     K, scene["topo"])
        fixed = GR.apply_twist(anchor, R, v)

        def median_residual(state):
            V, _ = scene["fk"](state)
            keep, s, _, _ = GR.associate(px, py, GR.occluding_contour(V[0], faces, edges, adj, K, 0.005), 10.0)
            return float(s[keep].abs().median())

        assert n >= GR.GeoRootParams().min_events
        assert median_residual(fixed) < 0.35 * median_residual(anchor)
        assert abs(float(fixed[0, 2] - truth[2])) < 0.5 * abs(float(anchor[0, 2] - truth[2]))
