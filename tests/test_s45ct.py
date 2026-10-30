"""S45-CT (docs/EXPERIMENT_RECORDS.md#s45ct-prereg-20261030): the corrupted-route hook changes only the (face,
barycentrics) of the events the oracle face route replaces; sigma16 draws exactly S45-RF P2's noise16 ring, shuffle
is a derangement inside each packet that keeps its routed face multiset; both are deterministic per source row and
blind to event order; every other routing argument reaches the native function untouched."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from semkine import s45ct_route as H  # noqa: E402
from semkine import xag as XAG  # noqa: E402

NPZ = ROOT / "assets" / "mano_right.npz"


@pytest.fixture(scope="module")
def ring():
    with np.load(NPZ, allow_pickle=False) as z:
        dist = H.face_geodesic(z["v_template"], z["f"])
    return dist, H.ring_table(dist, H.RING_M)


@pytest.fixture
def clean_state():
    saved = dict(H._STATE)
    native = (XAG.sample_sources, XAG.route_association)
    H._STATE.update(mode=None, native_route=None, native_sample=None, rows=None, capture=None)
    yield
    XAG.sample_sources, XAG.route_association = native
    H._STATE.clear()
    H._STATE.update(saved)


def packet(rng, B=3, N=64, F=1538):
    face = torch.from_numpy(rng.integers(0, F, (B, N)))
    bary = torch.from_numpy(rng.dirichlet(np.ones(3), (B, N)).astype(np.float32))
    eligible = torch.from_numpy(rng.random((B, N)) < 0.7)
    rows = torch.from_numpy(rng.choice(10 ** 7, (B, N), replace=False))
    return face, bary, eligible, rows


def test_hash_is_uniform_deterministic_and_salted():
    rows = torch.arange(100000)
    u = H.hash01(rows, 1)
    assert torch.equal(u, H.hash01(rows, 1)) and not torch.equal(u, H.hash01(rows, 2))
    assert float(u.min()) >= 0 and float(u.max()) < 1 and abs(float(u.mean()) - 0.5) < 0.01


def test_ring_equals_s45rf_p2_noise16(ring):
    from tools.s45rf import p2 as P2
    dist, (table, count) = ring
    t2, c2 = P2.ring_table(dist, 16 / 1000.0)
    assert np.array_equal(table, t2) and np.array_equal(count, c2)


def test_sigma16_moves_only_eligible_events_into_the_ring(ring):
    dist, (table, count) = ring
    face, bary, eligible, rows = packet(np.random.default_rng(0))
    nf, nb = H.corrupt("sigma16", face, bary, eligible, rows, NPZ)
    assert torch.equal(nf[~eligible], face[~eligible]) and torch.equal(nb[~eligible], bary[~eligible])
    assert torch.allclose(nb[eligible], torch.full_like(nb[eligible], 1 / 3))
    for f0, f1 in zip(face[eligible].tolist(), nf[eligible].tolist()):
        assert f1 in set(table[f0, :count[f0]].tolist())
    d = dist[face[eligible].numpy(), nf[eligible].numpy()]
    assert np.mean((d >= 0.008) & (d <= 0.024)) > 0.99


def test_shuffle_is_a_derangement_keeping_each_packets_multiset():
    face, bary, eligible, rows = packet(np.random.default_rng(1))
    nf, nb = H.corrupt("shuffle", face, bary, eligible, rows, NPZ)
    assert torch.equal(nf[~eligible], face[~eligible])
    for b in range(face.shape[0]):
        e = eligible[b]
        assert sorted(face[b][e].tolist()) == sorted(nf[b][e].tolist())
        assert sorted(map(tuple, bary[b][e].tolist())) == sorted(map(tuple, nb[b][e].tolist()))
        assert bool((nf[b][e] != face[b][e]).any()) and int((nb[b][e] == bary[b][e]).all(-1).sum()) == 0


@pytest.mark.parametrize("mode", H.MODES)
def test_corruption_is_a_function_of_source_rows_not_event_order(mode):
    face, bary, eligible, rows = packet(np.random.default_rng(2))
    nf, nb = H.corrupt(mode, face, bary, eligible, rows, NPZ)
    perm = torch.from_numpy(np.random.default_rng(3).permutation(face.shape[1]))
    pf, pb = H.corrupt(mode, face[:, perm], bary[:, perm], eligible[:, perm], rows[:, perm], NPZ)
    assert torch.equal(pf, nf[:, perm]) and torch.equal(pb, nb[:, perm])
    again = H.corrupt(mode, face, bary, eligible, rows, NPZ)
    assert torch.equal(again[0], nf) and torch.equal(again[1], nb)


def test_install_passes_corrupted_face_route_and_everything_else_untouched(clean_state):
    calls = []

    def native_route(mode, assoc, *args, **kw):
        calls.append((mode, assoc, args, kw))
        return assoc, {}

    def native_sample(batch, src, mask, **kw):
        return {"xag_source_index": batch["rows"]}
    XAG.route_association, XAG.sample_sources = native_route, native_sample
    H.install("sigma16", NPZ)
    rng = np.random.default_rng(4)
    face, bary, _, rows = packet(rng)
    mask = torch.ones(face.shape, dtype=torch.bool)
    assoc = {"q": torch.from_numpy(rng.random(face.shape)).float(), "valid": torch.from_numpy(rng.random(face.shape) < 0.8)}
    valid = torch.ones_like(mask)
    faces = torch.zeros((1538, 3), dtype=torch.long)
    for mode in ("face_route", "part_route"):
        XAG.sample_sources({"rows": rows}, None, mask)
        XAG.route_association(mode, assoc, "d", mask, "verts", faces, "vf", "lbs", 0.1, 0.2, 0.3, source_part="part",
                              source_face=face, source_bary=bary, source_valid=valid, k=16)
    (m1, _, a1, k1), (m2, _, a2, k2) = calls
    eligible = mask & (assoc["q"] > 0) & assoc["valid"] & valid
    ef, eb = H.corrupt("sigma16", face, bary, eligible, rows, NPZ)
    assert m1 == "face_route" and torch.equal(k1["source_face"], ef) and torch.equal(k1["source_bary"], eb)
    assert m2 == "part_route" and torch.equal(k2["source_face"], face) and torch.equal(k2["source_bary"], bary)
    assert a1 == a2 == ("d", mask, "verts", faces, "vf", "lbs", 0.1, 0.2, 0.3) and k1["source_part"] == "part"
    with pytest.raises(RuntimeError):
        XAG.route_association("face_route", assoc, "d", mask, "verts", faces, "vf", "lbs", 0.1, 0.2, 0.3,
                              source_part="part", source_face=face, source_bary=bary, source_valid=valid)
