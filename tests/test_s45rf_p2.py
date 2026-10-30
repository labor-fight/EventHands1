"""S45-RF P2 (docs/EXPERIMENT_RECORDS.md#s45rf-p2-prereg-20261030): the innovation geometry, the per-part pooling, the
centred informativeness (exact innovations explain everything, noise and constants explain nothing, uncovered pairs
explain nothing) and the template geodesic rings behave as registered."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from tools.s45rf import p2 as P  # noqa: E402


def test_points_projection_and_pooling():
    V = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 2.0], [1.0, 1.0, 2.0]])
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]])
    p = P.face_point(V, faces, torch.tensor([0]), torch.tensor([[1.0, 1.0, 2.0]]))
    assert torch.allclose(p, torch.tensor([[0.25, 0.5, 1.5]]))
    assert torch.allclose(P.project(p), torch.tensor([[0.25 / 1.5, 0.5 / 1.5]]))
    q = P.assoc_point(V, torch.tensor([[0, 1, 0, 0]]), torch.tensor([[1.0, 3.0, 0.0, -1.0]]))
    assert torch.allclose(q, torch.tensor([[0.75, 0.0, 1.0]]))
    lbs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert P.assoc_part(lbs, torch.tensor([[0, 1, 0, 0]]), torch.tensor([[1.0, 3.0, 0.0, 0.0]])).item() == 1
    cnt, tot = P.pool(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), torch.tensor([0, 2, 0]), n_parts=3)
    assert cnt.tolist() == [2, 0, 1] and torch.allclose(tot, torch.tensor([[6.0, 8.0], [0, 0], [3.0, 4.0]], dtype=torch.float64))


def test_part_truth_is_the_mean_projected_displacement():
    Vp = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    Ve = Vp + torch.tensor([0.1, -0.2, 0.0])
    D = P.part_truth(Vp, Ve, torch.tensor([0, 0, 1]), n_parts=2)
    assert torch.allclose(D, torch.tensor([[0.1, -0.2], [0.1, -0.2]], dtype=torch.float64))


def synthetic(rng, W=400, parts=4, groups=2):
    D = rng.normal(size=(W, parts, 2)) + rng.normal(size=(1, parts, 2)) * 3   # per-part offsets on top
    group = np.repeat(np.arange(groups), W // groups)
    vis = np.ones((W, parts), bool)
    return D, group, vis


def test_exact_innovation_explains_all_and_noise_or_constants_explain_nothing():
    rng = np.random.default_rng(0)
    D, group, vis = synthetic(rng)
    cnt = np.full(vis.shape, 5)
    I_exact, c = P.informativeness(*P.window_terms(D, cnt, D * 5 * 0.5, vis, group)[:3])
    assert np.isclose(I_exact, 1.0) and np.isclose(c, 2.0)            # nu = D / 2 -> gain 2, everything explained
    noise = rng.normal(size=D.shape) * 5
    assert P.informativeness(*P.window_terms(D, cnt, noise, vis, group)[:3])[0] < 0.02
    const = np.broadcast_to(rng.normal(size=(1,) + D.shape[1:]) * 5, D.shape).copy()
    assert P.informativeness(*P.window_terms(D, cnt, const, vis, group)[:3])[0] < 1e-12


def test_uncovered_pairs_explain_nothing():
    rng = np.random.default_rng(1)
    D, group, vis = synthetic(rng)
    cnt = np.full(vis.shape, 5)
    cnt[:, :2] = P.M_MIN - 1                                             # half the parts not covered
    a, b, d, cover = P.window_terms(D, cnt, D * 5, vis, group)
    I, _ = P.informativeness(a, b, d)
    dc = P.centred(D, vis, group)
    share = (dc[:, 2:] ** 2).sum() / (dc ** 2).sum()
    assert not cover[:, :2].any() and np.isclose(I, share, rtol=0.05)


def test_geodesic_rings_on_a_flat_strip():
    xs = np.arange(6, dtype=float) * 0.004
    tmpl = np.array([[x, y, 0.0] for x in xs for y in (0.0, 0.004)])
    faces = []
    for i in range(5):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        faces += [[a, c, b], [b, c, d]]
    dist = P.face_geodesic(tmpl, np.array(faces))
    assert dist.shape == (10, 10) and np.allclose(np.diag(dist), 0) and np.allclose(dist, dist.T)
    assert dist[0, 9] > dist[0, 2] > 0
    table, count = P.ring_table(dist, 0.008)
    for f in range(10):
        picked = table[f, :count[f]]
        inside = (dist[f, picked] >= 0.004) & (dist[f, picked] <= 0.012)
        assert inside.all() or count[f] == 1


def test_partswap_table_draws_only_other_parts():
    fparts = np.array([0, 0, 1, 2, 2, 2] + [3] * 10 + list(range(4, 16)))
    table, count = P.other_part_table(fparts)
    for p in range(P.N_PARTS):
        picked = table[p, :count[p]]
        assert (fparts[picked] != p).all() and count[p] == (fparts != p).sum()
    assert P.SOURCES[-2:] == ("partswap40", "shuffled") and P.NOISE_MM == (4, 8, 16, 32, 64)
