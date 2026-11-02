"""CPU tests for S45-SM (docs/EXPERIMENT_RECORDS.md#s45sm-prereg-20261031): the lean router, the hooks, and PFA
(semkine/s45sm.py; the model-side half of the study's tests)."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import s45sm as SM  # noqa: E402
from semkine import xag as XAG  # noqa: E402
from semkine import xyz_mesh as XM  # noqa: E402
from semkine.events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y  # noqa: E402

NPZ = ROOT / "assets/mano_right.npz"
S45_PARAMS = 459059


@pytest.fixture(scope="module")
def mano():
    with np.load(NPZ, allow_pickle=False) as z:
        return (torch.tensor(np.asarray(z["v_template"], np.float32)), torch.tensor(np.asarray(z["f"], np.int64)),
                np.asarray(z["weights"]))


def packet(n_per=(300, 0, 50), seed=0):
    g = torch.Generator().manual_seed(seed)
    rows, ptr = [], [0]
    for b, n in enumerate(n_per):
        ev = torch.zeros(n, 5)
        ev[:, EV_BATCH] = b
        ev[:, EV_X] = torch.randint(0, 240, (n,), generator=g).float()
        ev[:, EV_Y] = torch.randint(0, 180, (n,), generator=g).float()
        ev[:, EV_T] = torch.sort(torch.rand(n, generator=g) * 0.05).values
        ev[:, EV_P] = (torch.rand(n, generator=g) > 0.5).float()
        rows.append(ev)
        ptr.append(ptr[-1] + n)
    return torch.cat(rows), torch.tensor(ptr), torch.full((len(n_per),), 0.05)


def test_budget_and_parts():
    net = SM.build(NPZ)
    assert SM.n_params(net) == 137028
    assert SM.n_params(net) + S45_PARAMS <= 600000          # pre-registered budget: <= +30 % parameters
    counts = torch.bincount(net.face_part, minlength=SM.N_PARTS)
    assert int(counts.sum()) == net.faces.shape[0] and bool((counts > 0).all())


def test_forward_and_route_shapes():
    torch.manual_seed(0)
    net = SM.build(NPZ).eval()
    events, ptr, dt = packet()
    with torch.no_grad():
        out = net(events, ptr, dt)
        B, N = out["mask"].shape
        assert out["part"].shape == (B, N, SM.N_PARTS) and out["coord"].shape == (B, N, 3)
        assert out["mask"].sum(1).tolist() == [300, 0, 50]
        sel = out["mask"].clone()
        face, bary = net.route(out, sel)
    part = out["part"].argmax(-1)
    assert bool((net.face_part[face[sel]] == part[sel]).all())       # routed face lies in the predicted part
    assert torch.allclose(bary[sel].sum(-1), torch.ones(int(sel.sum())), atol=1e-5)
    assert bool((bary[sel] >= 0).all())


def test_nearest_face_in_part(mano):
    """Feeding a face's own centroid as the predicted point (and its part) returns that face."""
    net = SM.build(NPZ).eval()
    faces = torch.arange(0, net.faces.shape[0], 97)
    n = len(faces)
    part = torch.full((1, n, SM.N_PARTS), -10.0)
    part[0, torch.arange(n), net.face_part[faces]] = 10.0
    out = {"part": part, "coord": (net.centroid[faces] / 0.01).unsqueeze(0)}
    face, bary = net.route(out, torch.ones(1, n, dtype=torch.bool))
    assert torch.equal(face[0], faces)
    assert torch.allclose(bary[0], torch.full((n, 3), 1 / 3), atol=1e-4)


def make_assoc_inputs(mano, B=2, N=64, seed=1):
    verts, faces, weights = mano
    g = torch.Generator().manual_seed(seed)
    V = verts.unsqueeze(0).repeat(B, 1, 1).clone()
    V[..., 2] += 0.5                                                   # half a metre in front of the camera
    mask = torch.rand(B, N, generator=g) > 0.2
    f = torch.randint(0, faces.shape[0], (B, N), generator=g)
    bary = torch.rand(B, N, 3, generator=g)
    bary = bary / bary.sum(-1, keepdim=True)
    valid = torch.rand(B, N, generator=g) > 0.3
    d = torch.cat([torch.rand(B, N, 2, generator=g) * 0.2 - 0.1, torch.ones(B, N, 1)], -1)
    return V, faces, mask, f, bary, valid, d


def test_empty_association_matches_native_shapes(mano):
    V, faces, mask, *_ , d = make_assoc_inputs(mano)
    vf = XM.vertex_face_table(faces, V.shape[1])
    band = torch.full((V.shape[0],), 0.02)
    native = XM.associate_rays(d, mask, V, faces, vf, band, band / 2, 0.01)
    empty = SM.empty_association(d, mask, V, faces, vf, band, band / 2, 0.01)
    assert set(native) == set(empty)
    for k in native:
        assert native[k].shape == empty[k].shape and native[k].dtype == empty[k].dtype
        assert not bool(empty[k].any())


def test_pfa_association(mano):
    V, faces, mask, f, bary, valid, d = make_assoc_inputs(mano)
    vf = XM.vertex_face_table(faces, V.shape[1])
    band = torch.full((V.shape[0],), 0.02)
    base = SM.empty_association(d, mask, V, faces, vf, band, band / 2, 0.01)
    lbs = torch.tensor(mano[2], dtype=torch.float32)
    out, audit = SM.pfa_association("face_route", base, d, mask, V, faces, vf, lbs, band, band / 2, 0.01,
                                    source_part=torch.zeros_like(f), source_face=f, source_bary=bary, source_valid=valid)
    elig = mask & valid
    assert torch.equal(audit["xag_eligible"], elig)
    assert torch.equal(out["valid"], elig) and torch.equal(out["hit"], elig)
    assert torch.equal(out["q"], elig.float())
    b, n = elig.nonzero(as_tuple=True)
    fv = faces[f[b, n]]
    assert torch.equal(out["vid"][b, n, :3], fv) and torch.equal(out["vid"][b, n, 3], fv[:, 0])
    assert torch.allclose(out["A"][b, n, :3], bary[b, n], atol=1e-6) and not bool(out["A"][b, n, 3].any())
    point = (bary[b, n].unsqueeze(-1) * V[b.unsqueeze(1), fv]).sum(1)
    assert torch.allclose(out["z"][b, n], point[:, 2], atol=1e-6)
    off = ~elig
    for k in ("vid", "A", "q", "z"):
        assert not bool(out[k][off].any())


def test_install_uninstall(tmp_path):
    torch.manual_seed(0)
    net = SM.build(NPZ)
    ckpt = tmp_path / "sr.ckpt"
    SM.save_checkpoint(net, ckpt, step=0)
    natives = (XAG.sample_sources, XAG.route_association, XM.associate_rays)
    try:
        state = SM.install("sr", ckpt, "pfa", 0.5)
        assert state["router_params"] == 137028 and state["assoc"] == "pfa"
        assert XAG.sample_sources is SM.predicted_sources and XAG.route_association is SM.pfa_association
        assert XM.associate_rays is SM.empty_association
        with pytest.raises(RuntimeError):
            SM.install("sr", ckpt, "pfa", 0.5)
    finally:
        SM.uninstall()
    assert (XAG.sample_sources, XAG.route_association, XM.associate_rays) == natives
    try:
        SM.install("sr", ckpt, "face_route", 0.5)
        assert XAG.route_association is natives[1] and XM.associate_rays is natives[2]
    finally:
        SM.uninstall()
    assert SM.installed() is None


def test_predicted_sources_follow_router(tmp_path):
    torch.manual_seed(0)
    net = SM.build(NPZ)
    ckpt = tmp_path / "sr.ckpt"
    SM.save_checkpoint(net, ckpt, step=0)
    events, ptr, dt = packet()
    src, mask = XM.stride_sample(ptr, SM.MAX_NODES)

    class Batch:
        pass
    batch = Batch()
    batch.events, batch.ptr, batch.delta_t_s = events, ptr, dt
    try:
        SM.install("sr", ckpt, "face_route", 0.0)                   # tau 0: every live node routed
        s = XAG.sample_sources(batch, src, mask)
    finally:
        SM.uninstall()
    assert torch.equal(s["xag_valid"], mask)
    loaded, _ = SM.load(ckpt)
    with torch.no_grad():
        out = loaded(events, ptr, dt)
        face, bary = loaded.route(out, mask)
    assert torch.equal(s["xag_face"][mask], face[mask]) and torch.allclose(s["xag_bary"][mask], bary[mask])
