#!/usr/bin/env python3
"""[XYZ-ASSOC] opt-in association variants (docs/EXPERIMENT_RECORDS.md#xyz-assoc-20261009). CPU only.

  1. `associate_rays(mode="hit_first")` is the pre-change function bit for bit (`_assoc_ref` below is a
     verbatim copy of the pre-change source), also with `normals` / `gap` passed;
  2. `mode="contour"` conserves every live event's unit mass (sum w + r = 1), reduces to `hit_first` where no
     gap triggers (and where no candidate is culled), and does trigger on real hand meshes;
  3. the minimal occluded-silhouette scene: a ray 1 px outside a foreground triangle that sits in front of a
     background triangle is the background's under `hit_first` and split with the foreground (ctr = 1)
     under `contour`; a back-facing foreground leaves it alone;
  4. `XYZMeshNet`: the default is the pre-change network (459,059 parameters, measured on the pre-change
     source), `contour=True` adds the 64 weights of one more relation input, shapes and gradients are sane;
  5. `forward(verts_geo=...)`: None is the default path bit for bit, a geometry mesh equal to the association
     mesh reproduces it to float rounding, and the recomputed dXYZ / z_prior follow the stated formulas;
  6. the model keys (validation, oracle `XYZ_ASSOC_PREV=target`, `route_stats`), `track_sequence`'s per-step
     ground-truth hand-over, `make_s36_row.py --step` and `report_table.py --tag`.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from mano_layer import ManoLayer                                   # noqa: E402
from model import MNISTModel                                       # noqa: E402
from pose_repr import decode_to_mano_inputs                        # noqa: E402
from semkine.events import EventPacketBatch                        # noqa: E402
from semkine.xyz_mesh import (ASSOC_CHUNK, L_REL, R_MAX, REL_IN, XYZMeshNet,   # noqa: E402,F401
                              associate_rays, ray_triangle, vertex_face_table, vertex_normals)

NPZ = str(ROOT / "assets" / "mano_right.npz")
#: parameters of the pre-change XYZMeshNet / MNISTModel (C1 and C0), counted on the pre-change source
N_PARAMS_DEFAULT = 459_059
F_PX = 0.375 * 603.0                      # focal length in event-frame pixels (the K of the default context)
BAND, FRONT, ZTOL = 16.0, 3.0, 0.01


# --------------------------------------------------------------------------- the pre-change function, verbatim
@torch.no_grad()
def _assoc_ref(d: torch.Tensor, mask: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor,
                   vert_faces: torch.Tensor, band: torch.Tensor, front_tol: torch.Tensor,
                   z_tol: float, k: int = 16, chunk: int = ASSOC_CHUNK) -> Dict[str, torch.Tensor]:
    """Event rays against the posed mesh, in camera space.

    d: `(B, N, 3)` rays with `d_z = 1`; mask: `(B, N)` live events; verts: `(B, V, 3)` metres;
    faces: `(F, 3)`; vert_faces: `(V, M)` from `vertex_face_table`; band / front_tol: `(B,)` angular
    radii in ray-direction units (pixels / focal length); z_tol: metres.

    Candidates are the `k` vertices nearest in ray direction and the faces incident to them. A
    surface event takes the nearest positive hit among those faces (front and back faces alike, so
    an occluding part in front wins); a near event, with no hit, takes the front depth layer among
    the in-band candidates (reference = the angularly nearest one, the layer = within `front_tol`
    of it in angle and `z_tol` of the front-most of those in depth), at most `R_MAX` of them, with
    Gaussian angular weights (sigma = band / 2). `z_prior` is the hit depth, or the kernel mean of
    the candidates' closest-point depths on the ray. Everything here is a hypothesis conditioned
    on `prev`, not a measurement.

    Returns a dict of `(B, N, R_MAX)` `vid` / `A` and `(B, N)` `q`, `z`, `valid`, `hit`.
    """
    B, N, _ = d.shape
    V = verts.shape[1]
    dev = d.device
    vid = torch.zeros(B, N, R_MAX, dtype=torch.long, device=dev)
    A = torch.zeros(B, N, R_MAX, dtype=torch.float32, device=dev)
    q = torch.zeros(B, N, dtype=torch.float32, device=dev)
    z = torch.zeros(B, N, dtype=torch.float32, device=dev)
    valid = torch.zeros(B, N, dtype=torch.bool, device=dev)
    hit = torch.zeros(B, N, dtype=torch.bool, device=dev)
    out = {"vid": vid, "A": A, "q": q, "z": z, "valid": valid, "hit": hit}
    flat = mask.reshape(-1).nonzero().squeeze(1)
    if flat.numel() == 0:
        return out
    Vf = verts.float()
    zv = Vf[..., 2]
    bad = zv <= 1e-6
    vu = (Vf[..., 0] / zv.clamp_min(1e-6)).masked_fill(bad, 1e6)                   # (B, V)
    vv = (Vf[..., 1] / zv.clamp_min(1e-6)).masked_fill(bad, 1e6)
    faces = faces.long()
    kk = max(1, min(int(k), V))
    col = torch.arange(kk, device=dev)
    parts = {key: [] for key in out}
    for i0 in range(0, int(flat.numel()), int(chunk)):
        rows = flat[i0:i0 + chunk]
        b = rows // N
        dd = d.reshape(-1, 3)[rows].float()                                          # (m, 3)
        m = rows.numel()
        d2 = (vu[b] - dd[:, 0:1]).square() + (vv[b] - dd[:, 1:2]).square()           # (m, V)
        near = d2.topk(kk, dim=-1, largest=False)
        nid = near.indices                                                          # (m, kk)
        nang = near.values.clamp_min(0.0).sqrt()                                     # ascending
        # ---- surface: exact hits against the faces around the angular neighbours
        cf = vert_faces[nid].reshape(m, -1)                                          # (m, kk*M)
        cfs = cf.clamp_min(0)
        tri = faces[cfs]                                                             # (m, C, 3)
        P = Vf[b.view(m, 1, 1), tri]                                                 # (m, C, 3, 3)
        t, u, v = ray_triangle(dd.view(m, 1, 3), P[:, :, 0], P[:, :, 1], P[:, :, 2])
        t = t.masked_fill(cf < 0, float("inf"))
        tmin, fsel = t.min(dim=-1)
        is_hit = torch.isfinite(tmin)
        fh = cfs.gather(1, fsel[:, None]).squeeze(1)
        uh = u.gather(1, fsel[:, None]).squeeze(1)
        vh = v.gather(1, fsel[:, None]).squeeze(1)
        wb = torch.stack([(1.0 - uh - vh).clamp_min(0), uh.clamp_min(0), vh.clamp_min(0),
                          torch.zeros_like(uh)], dim=-1)
        A_hit = wb / wb.sum(-1, keepdim=True).clamp_min(1e-12)
        fv = faces[fh]                                                               # (m, 3)
        vid_hit = torch.cat([fv, fv[:, :1]], dim=1)
        # ---- near: the front depth layer among the in-band angular neighbours
        bnd = band[b].float().unsqueeze(1)
        ftl = front_tol[b].float().unsqueeze(1)
        Vn = Vf[b.view(m, 1), nid]                                                   # (m, kk, 3)
        zj = (Vn * dd.unsqueeze(1)).sum(-1) / (dd * dd).sum(-1, keepdim=True)        # closest-point depth
        inband = (nang <= bnd) & (zj > 1e-4)
        same = inband & (nang <= nang[:, :1] + ftl)
        zfront = torch.where(same, zj, torch.full_like(zj, float("inf"))).min(-1).values
        layer = inband & (zj <= zfront.unsqueeze(1) + float(z_tol))
        rank = torch.where(layer, col.unsqueeze(0), col.unsqueeze(0) + kk)
        cols = rank.argsort(dim=-1)[:, :R_MAX]                                       # first R_MAX in layer
        kept = layer.gather(1, cols)
        ang = nang.gather(1, cols)
        sig = bnd / 2.0
        kw = torch.exp(-0.5 * (ang / sig).square()) * kept
        is_near = (~is_hit) & kept.any(-1)
        A_near = kw / kw.sum(-1, keepdim=True).clamp_min(1e-12)
        q_near = torch.exp(-0.5 * (ang[:, 0] / sig[:, 0]).square())
        z_near = (A_near * zj.gather(1, cols)).sum(-1)
        vid_near = nid.gather(1, cols)
        ok = is_hit | is_near
        parts["vid"].append(torch.where(is_hit[:, None], vid_hit, vid_near) * ok[:, None])
        parts["A"].append(torch.where(is_hit[:, None], A_hit, A_near) * ok[:, None])
        parts["q"].append(torch.where(is_hit, torch.ones_like(q_near), q_near) * ok)
        parts["z"].append(torch.where(is_hit, tmin, z_near) * ok)
        parts["valid"].append(ok)
        parts["hit"].append(is_hit)
    for key, dst in out.items():
        dst.view(-1, *dst.shape[2:])[flat] = torch.cat(parts[key]).to(dst.dtype)
    return out


# --------------------------------------------------------------------------- scenes
@pytest.fixture(scope="module")
def mano():
    return ManoLayer(NPZ, add_mean=False).eval()


def _hand_verts(mano, seed: int, B: int = 2, flex: float = 0.9, tz: float = 0.45) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    params = torch.zeros(B, 51)
    params[:, 0:2] = torch.randn(B, 2, generator=g) * 0.01
    params[:, 2] = tz
    params[:, 3:6] = torch.randn(B, 3, generator=g) * 0.7
    params[:, 6:] = torch.randn(B, 45, generator=g) * flex
    dec = decode_to_mano_inputs(params, "mano_full_axis_angle", mano.hands_components, mano.hands_mean)
    with torch.no_grad():
        v, _ = mano(torch.zeros(B, 10), dec["global_orient"], dec["local_full_aa"], dec["transl"])
    return v


def _rays(verts: torch.Tensor, seed: int, N: int = 400, spread: float = 0.03):
    """Rays through random vertex projections plus noise (ray units; 0.03 ~ 7 px), d_z = 1; ~10 % dead slots."""
    g = torch.Generator().manual_seed(1000 + seed)
    B, V, _ = verts.shape
    pick = torch.randint(V, (B, N), generator=g)
    p = torch.gather(verts, 1, pick.unsqueeze(-1).expand(-1, -1, 3))
    uv = p[..., :2] / p[..., 2:3] + torch.randn(B, N, 2, generator=g) * spread
    d = torch.cat([uv, torch.ones(B, N, 1)], dim=-1)
    mask = torch.rand(B, N, generator=g) > 0.1
    return d, mask


def _tables(mano, B: int):
    faces = mano.f
    return faces, vertex_face_table(faces, int(mano.v_template.shape[0]))


def _kw(B: int):
    return dict(band=torch.full((B,), BAND / F_PX), front_tol=torch.full((B,), FRONT / F_PX), z_tol=ZTOL, k=16)


def _same(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], keys=None):
    for k in (keys or a.keys()):
        assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype, k
        assert torch.equal(a[k], b[k]), k


# --------------------------------------------------------------------------- (1) hit_first == pre-change
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hit_first_is_the_pre_change_function_bitwise(mano, seed):
    verts = _hand_verts(mano, seed)
    d, mask = _rays(verts, seed)
    faces, vf = _tables(mano, 2)
    ref = _assoc_ref(d, mask, verts, faces, vf, **_kw(2))
    assert 0.2 < ref["hit"].float().mean() < 0.95 and ref["valid"].any()          # a non-trivial scene
    assert set(ref) == {"vid", "A", "q", "z", "valid", "hit"}
    new = associate_rays(d, mask, verts, faces, vf, **_kw(2))
    assert set(new) == set(ref)                                                    # no new key in the default
    _same(new, ref)
    _same(associate_rays(d, mask, verts, faces, vf, **_kw(2), mode="hit_first"), ref)
    # contour-only arguments are inert in hit_first
    _same(associate_rays(d, mask, verts, faces, vf, **_kw(2), mode="hit_first",
                         normals=vertex_normals(verts, faces), gap=0.5), ref)
    # a small chunk (several chunks per call) changes nothing either
    _same(associate_rays(d, mask, verts, faces, vf, **_kw(2), chunk=97), ref)


def test_unknown_mode_and_missing_normals_raise(mano):
    verts = _hand_verts(mano, 0)
    d, mask = _rays(verts, 0, N=8)
    faces, vf = _tables(mano, 2)
    with pytest.raises(ValueError):
        associate_rays(d, mask, verts, faces, vf, **_kw(2), mode="nearest")
    with pytest.raises(ValueError):
        associate_rays(d, mask, verts, faces, vf, **_kw(2), mode="contour")       # no normals


def test_vertex_normals_are_unit_outward_and_area_weighted(mano):
    verts = _hand_verts(mano, 3)
    faces = mano.f
    n = vertex_normals(verts, faces)
    assert n.shape == verts.shape and torch.allclose(n.norm(dim=-1), torch.ones(2, 778), atol=1e-5)
    # outward: the vertex nearest the camera faces it
    near = verts[..., 2].argmin(-1)
    for b in range(2):
        d = verts[b, near[b]] / verts[b, near[b], 2]
        assert float((n[b, near[b]] * d).sum()) < -0.5
    # area weighting on a tiny fan: one big and one small triangle sharing vertex 0
    v = torch.tensor([[[0., 0., 1.], [2., 0., 1.], [0., 2., 1.], [-0.1, 0., 1.], [0., -0.1, 1.]]])
    f = torch.tensor([[0, 1, 2], [0, 4, 3]])                  # (+z, area 2) and (-z, area 0.005)
    nz = vertex_normals(v, f)[0, 0, 2]
    assert float(nz) > 0.99
    # a vertex used by no face has a zero normal, not NaN
    v6 = torch.cat([v, torch.tensor([[[5., 5., 5.]]])], dim=1)
    n6 = vertex_normals(v6, f)
    assert torch.isfinite(n6).all() and float(n6[0, 5].abs().sum()) == 0.0


# --------------------------------------------------------------------------- (2) contour: mass, reduction
def _mass(out):
    q = out["q"]
    return (q.unsqueeze(-1) * out["A"]).sum(-1) + (1.0 - q)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_contour_conserves_unit_mass_and_reduces_to_hit_first(mano, seed):
    verts = _hand_verts(mano, seed, flex=1.0)
    d, mask = _rays(verts, seed, N=500)
    faces, vf = _tables(mano, 2)
    nrm = vertex_normals(verts, faces)
    hf = associate_rays(d, mask, verts, faces, vf, **_kw(2))
    ct = associate_rays(d, mask, verts, faces, vf, **_kw(2), mode="contour", normals=nrm, gap=0.01)
    assert ct["vid"].shape == (2, 500, 2 * R_MAX) and ct["A"].shape == ct["z"].shape == ct["ctr"].shape
    assert ct["q"].shape == ct["valid"].shape == ct["hit"].shape == (2, 500)
    assert ct["ctr"].dtype == torch.bool
    live = mask
    for out in (hf, ct):                                                           # sum w + r = 1 per live event
        assert float((_mass(out) - 1.0).abs()[live].max()) <= 1e-6
    # dead slots stay empty
    assert not ct["A"][~mask].any() and not ct["ctr"][~mask].any() and not ct["valid"][~mask].any()
    trig = ct["ctr"].any(-1)
    assert int(trig.sum()) > 0, "the scene never triggers the two-hypothesis branch; the test is vacuous"
    # a triggered event: hit, q = 1, r = 0, both hypotheses carry mass, ctr only on records 4..7
    assert ct["hit"][trig].all() and ct["valid"][trig].all() and (ct["q"][trig] == 1.0).all()
    assert not ct["ctr"][..., :R_MAX].any()
    fg = (ct["A"] * ct["ctr"]).sum(-1)[trig]
    assert (fg > 0).all() and (fg < 1).all() and (ct["A"][..., :R_MAX].sum(-1)[trig] > 0).all()
    zc = ct["z"][trig]
    # the trigger is zfront <= t_hit - gap; z_near is the kernel mean over the layer (<= zfront + z_tol)
    assert (zc[:, R_MAX:].amax(-1) <= zc[:, :R_MAX].amin(-1) - 0.01 + ZTOL + 1e-6).all()
    assert (zc[:, R_MAX:].amax(-1) < zc[:, :R_MAX].amin(-1)).all()                         # fg strictly in front
    # not triggered: the hit_first record set, records 4..7 weigh nothing, z equal over the record axis
    nt = ~trig & mask
    assert not ct["A"][..., R_MAX:][nt].any()
    for k in ("valid", "hit"):
        assert torch.equal(ct[k][nt], hf[k][nt])
    hit_nt = nt & hf["hit"]
    assert (hit_nt & ct["hit"]).sum() == hit_nt.sum()
    for k, sl in (("vid", slice(0, R_MAX)), ("A", slice(0, R_MAX))):
        assert torch.equal(ct[k][..., sl][hit_nt], hf[k][hit_nt])                      # hit events: as before
    assert torch.equal(ct["q"][hit_nt], hf["q"][hit_nt])
    assert torch.equal(ct["z"][..., 0][hit_nt], hf["z"][hit_nt])


@pytest.mark.parametrize("seed", [0, 1])
def test_contour_without_trigger_or_culling_equals_hit_first(mano, seed):
    verts = _hand_verts(mano, seed, flex=1.0)
    d, mask = _rays(verts, seed, N=500)
    faces, vf = _tables(mano, 2)
    hf = associate_rays(d, mask, verts, faces, vf, **_kw(2))
    # gap = 1e6 never triggers; zero normals (n . d = 0) cull nothing: contour == hit_first, every event
    ct = associate_rays(d, mask, verts, faces, vf, **_kw(2), mode="contour",
                        normals=torch.zeros_like(verts), gap=1e6)
    assert not ct["ctr"].any() and not ct["A"][..., R_MAX:].any()
    for k in ("vid", "A"):
        assert torch.equal(ct[k][..., :R_MAX], hf[k])
    for k in ("q", "valid", "hit"):
        assert torch.equal(ct[k], hf[k])
    assert torch.equal(ct["z"][..., 0], hf["z"]) and torch.equal(ct["z"][..., R_MAX - 1], hf["z"])
    # with the true normals the hit events still equal hit_first (culling touches the near layer only)
    ct2 = associate_rays(d, mask, verts, faces, vf, **_kw(2), mode="contour",
                         normals=vertex_normals(verts, faces), gap=1e6)
    h = hf["hit"]
    assert torch.equal(ct2["hit"], h) and torch.equal(ct2["A"][..., :R_MAX][h], hf["A"][h])
    assert torch.equal(ct2["vid"][..., :R_MAX][h], hf["vid"][h]) and not ct2["ctr"].any()
    # culling only ever removes near candidates: events that hit_first sent to `none` stay `none`
    assert not (ct2["valid"] & ~hf["valid"]).any()


def test_contour_gap_is_a_threshold(mano):
    verts = _hand_verts(mano, 0, flex=1.0)
    d, mask = _rays(verts, 0, N=500)
    faces, vf = _tables(mano, 2)
    nrm = vertex_normals(verts, faces)
    n_by_gap = [int(associate_rays(d, mask, verts, faces, vf, **_kw(2), mode="contour", normals=nrm,
                                   gap=g)["ctr"].any(-1).sum()) for g in (0.0, 0.01, 0.03, 0.1)]
    assert n_by_gap == sorted(n_by_gap, reverse=True) and n_by_gap[0] > n_by_gap[-1]


# --------------------------------------------------------------------------- (3) the minimal scene
def _scene(fg_faces_camera: bool = True, ray_dx: float = 0.0, ray_dy_px: float = -1.0):
    """Background triangle at z = 0.6 (large, covers the view) and a small foreground triangle at z = 0.4
    (half-size 0.01 m, ~5 px at f = 200). The ray goes `ray_dy_px` px below the foreground's lower edge."""
    f_px = 200.0
    bg = [[-0.5, -0.5, 0.6], [0.0, 0.5, 0.6], [0.5, -0.5, 0.6]]                   # faces the camera (-z)
    fg = [[-0.01, -0.01, 0.4], [0.0, 0.01, 0.4], [0.01, -0.01, 0.4]]
    if not fg_faces_camera:
        fg = fg[::-1]
    verts = torch.tensor([bg + fg], dtype=torch.float32)                          # ids 0..2 background, 3..5 foreground
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    edge_y = -0.01 / 0.4                                                           # lower edge in ray units
    ray = torch.tensor([[[ray_dx, edge_y + ray_dy_px / f_px, 1.0]]])
    kw = dict(band=torch.tensor([BAND / f_px]), front_tol=torch.tensor([FRONT / f_px]), z_tol=ZTOL, k=16)
    return verts, faces, vertex_face_table(faces, 6), ray, torch.ones(1, 1, dtype=torch.bool), kw


def test_minimal_occluded_silhouette_scene():
    verts, faces, vf, ray, mask, kw = _scene()
    n = vertex_normals(verts, faces)
    assert (n[0, :, 2] < 0).all()                                                  # both triangles face the camera
    # the ray misses the foreground triangle
    t_fg, _, _ = ray_triangle(ray.view(1, 3), verts[0, 3], verts[0, 4], verts[0, 5])
    assert not torch.isfinite(t_fg).any()
    hf = associate_rays(ray, mask, verts, faces, vf, **kw)
    assert bool(hf["hit"][0, 0]) and float(hf["q"][0, 0]) == 1.0
    assert set(hf["vid"][0, 0, :3].tolist()) == {0, 1, 2} and float(hf["z"][0, 0]) == pytest.approx(0.6, abs=1e-5)
    ct = associate_rays(ray, mask, verts, faces, vf, **kw, mode="contour", normals=n, gap=0.01)
    assert bool(ct["hit"][0, 0]) and bool(ct["valid"][0, 0]) and float(ct["q"][0, 0]) == 1.0
    assert float(_mass(ct)[0, 0]) == pytest.approx(1.0, abs=1e-6)                  # r = 0, sum A = 1
    ctr = ct["ctr"][0, 0]
    assert ctr[R_MAX:].any() and not ctr[:R_MAX].any()
    fg_mass = float((ct["A"][0, 0] * ctr).sum())
    bg_mass = float((ct["A"][0, 0] * ~ctr).sum())
    assert 0.0 < fg_mass < 1.0 and bg_mass == pytest.approx(1.0 - fg_mass, abs=1e-6)
    # alpha = q_near: the kernel of the nearest foreground vertex (5.1 px from the ray, sigma = band / 2 = 8 px)
    assert fg_mass == pytest.approx(0.816, abs=0.01)
    assert set(ct["vid"][0, 0][ctr & (ct["A"][0, 0] > 0)].tolist()) == {3, 4, 5}   # the foreground layer: all three in band
    assert set(ct["vid"][0, 0, :3].tolist()) == {0, 1, 2}
    assert float(ct["z"][0, 0, 0]) == pytest.approx(0.6, abs=1e-5)                 # hit records: t_hit
    assert float(ct["z"][0, 0, R_MAX]) == pytest.approx(0.4, abs=1e-3)             # foreground records: z_near (closest-point depth)
    # the hit_first answer is the background alone
    assert float((hf["A"][0, 0] * (hf["vid"][0, 0] >= 3)).sum()) == 0.0


def test_scene_controls():
    # back-facing foreground: culled from the near layer, no occlusion, contour == hit_first for this event
    verts, faces, vf, ray, mask, kw = _scene(fg_faces_camera=False)
    n = vertex_normals(verts, faces)
    assert (n[0, 3:, 2] > 0).all()
    hf = associate_rays(ray, mask, verts, faces, vf, **kw)
    ct = associate_rays(ray, mask, verts, faces, vf, **kw, mode="contour", normals=n, gap=0.01)
    assert not ct["ctr"].any() and torch.equal(ct["A"][..., :R_MAX], hf["A"]) and torch.equal(ct["vid"][..., :R_MAX], hf["vid"])
    # a ray inside the foreground triangle hits it: nothing in front of the hit, no second hypothesis
    verts, faces, vf, ray, mask, kw = _scene(ray_dy_px=+1.0)
    hf = associate_rays(ray, mask, verts, faces, vf, **kw)
    assert set(hf["vid"][0, 0, :3].tolist()) == {3, 4, 5}
    ct = associate_rays(ray, mask, verts, faces, vf, **kw, mode="contour", normals=vertex_normals(verts, faces), gap=0.01)
    assert not ct["ctr"].any() and torch.equal(ct["A"][..., :R_MAX], hf["A"])
    # the same ray 1 px outside with a gap larger than the depth separation (0.2 m): no trigger
    verts, faces, vf, ray, mask, kw = _scene()
    ct = associate_rays(ray, mask, verts, faces, vf, **kw, mode="contour", normals=vertex_normals(verts, faces), gap=0.25)
    assert not ct["ctr"].any()
    # no hit and nothing in band: none, the whole event goes to the residual slot
    verts, faces, vf, ray, mask, kw = _scene()
    ray2 = ray.clone()
    ray2[..., 0] = 3.0                                                             # far off to the side: nothing in band
    ct = associate_rays(ray2, mask, verts, faces, vf, **kw, mode="contour", normals=vertex_normals(verts, faces), gap=0.01)
    assert not ct["valid"].any() and float(_mass(ct)[0, 0]) == pytest.approx(1.0)


# --------------------------------------------------------------------------- (4) the network
def _net(mano, **kw):
    torch.manual_seed(0)
    return XYZMeshNet(mano.weights, mano.f, **kw)


def _randomise_readout(net):
    """fc2 is zero-initialised (a fresh net outputs 0 whatever it reads); give it weights so every module counts."""
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        net.fc2.weight.copy_(torch.randn(net.fc2.weight.shape, generator=g) * 0.05)
        net.fc2.bias.copy_(torch.randn(net.fc2.bias.shape, generator=g) * 0.01)
    return net


def _n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_parameter_counts(mano):
    base = _net(mano)
    assert _n_params(base) == N_PARAMS_DEFAULT                  # == the pre-change network; contour=False adds nothing
    assert base.rel[0].weight.shape == (64, REL_IN) and not base.contour
    ctr = _net(mano, contour=True)
    assert _n_params(ctr) == N_PARAMS_DEFAULT + 64              # one more relation input: 64 weights, no bias change
    assert ctr.rel[0].weight.shape == (64, REL_IN + 1)
    # same parameter names / shapes except rel.0.weight
    sb, sc = base.state_dict(), ctr.state_dict()
    assert sb.keys() == sc.keys()
    assert [k for k in sb if sb[k].shape != sc[k].shape] == ["rel.0.weight"]
    # the default draw is untouched by the new argument: identical initial weights under the same seed
    again = _net(mano, contour=False)
    assert all(torch.equal(sb[k], again.state_dict()[k]) for k in sb)


def _net_inputs(mano, seed=0, N=300):
    verts = _hand_verts(mano, seed, flex=1.0)
    d, mask = _rays(verts, seed, N=N)
    faces, vf = _tables(mano, 2)
    g = torch.Generator().manual_seed(7)
    tok = torch.randn(2, N, 7, generator=g) * mask.unsqueeze(-1)
    prev48 = torch.randn(2, 48, generator=g) * 0.1
    return verts, d, mask, faces, vf, tok, prev48


def test_contour_net_forward_backward(mano):
    verts, d, mask, faces, vf, tok, prev48 = _net_inputs(mano)
    ct = associate_rays(d, mask, verts, faces, vf, **_kw(2), mode="contour",
                        normals=vertex_normals(verts, faces), gap=0.01)
    assert ct["ctr"].any()
    net = _randomise_readout(_net(mano, contour=True)).train()
    delta, stats, aux = net(tok, d, mask, verts, ct, prev48)
    assert delta.shape == (2, 51) and torch.isfinite(delta).all()
    assert aux["slots"].shape == (2, 17, 2 * 128 + 2) and aux["w"].shape == ct["A"].shape == (2, 300, 2 * R_MAX)
    assert set(stats) == {"xyz_valid", "xyz_hit", "xyz_resid_mass", "xyz_verts_direct", "xyz_verts_prop", "xyz_ctr_mass"}
    live = float(mask.sum())
    assert float(stats["xyz_ctr_mass"]) == pytest.approx(float((aux["w"] * ct["ctr"]).sum()) / live, rel=1e-5)
    assert 0.0 < float(stats["xyz_ctr_mass"]) < 1.0
    # total mass: matched + residual == the live events
    assert float(aux["w"].sum() + aux["r"].sum()) == pytest.approx(live, rel=1e-5)
    delta.square().sum().backward()
    g = net.rel[0].weight.grad
    assert g.shape == (64, REL_IN + 1) and torch.isfinite(g).all() and float(g[:, -1].abs().sum()) > 0   # the ctr input learns
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    # the flag matters: flipping it off changes the output
    ct2 = dict(ct, ctr=torch.zeros_like(ct["ctr"]))
    delta2, _, _ = net(tok, d, mask, verts, ct2, prev48)
    assert not torch.equal(delta, delta2)
    # a contour net refuses hit_first records instead of guessing the flag
    hf = associate_rays(d, mask, verts, faces, vf, **_kw(2))
    with pytest.raises(ValueError):
        net(tok, d, mask, verts, hf, prev48)


def test_default_net_stats_and_shapes_unchanged(mano):
    verts, d, mask, faces, vf, tok, prev48 = _net_inputs(mano)
    hf = associate_rays(d, mask, verts, faces, vf, **_kw(2))
    net = _randomise_readout(_net(mano)).eval()
    delta, stats, aux = net(tok, d, mask, verts, hf, prev48)
    assert delta.shape == (2, 51)
    assert set(stats) == {"xyz_valid", "xyz_hit", "xyz_resid_mass", "xyz_verts_direct", "xyz_verts_prop"}
    assert set(aux) == {"slots", "O", "S", "w", "r"} and aux["w"].shape == (2, 300, R_MAX)


# --------------------------------------------------------------------------- (5) verts_geo
def test_verts_geo_none_is_the_default_path_bitwise(mano):
    verts, d, mask, faces, vf, tok, prev48 = _net_inputs(mano)
    hf = associate_rays(d, mask, verts, faces, vf, **_kw(2))
    net = _randomise_readout(_net(mano)).eval()
    a = net(tok, d, mask, verts, hf, prev48)
    b = net(tok, d, mask, verts, hf, prev48, verts_geo=None)
    assert torch.equal(a[0], b[0]) and all(torch.equal(a[1][k], b[1][k]) for k in a[1])
    assert all(torch.equal(a[2][k], b[2][k]) for k in a[2])


def test_verts_geo_same_mesh_reproduces_the_default_to_rounding(mano):
    """`verts_geo = verts` recomputes z_prior from the vertices: sum_r A_r zj_r equals the Moeller-Trumbore hit depth
    mathematically (the hit point is the barycentric combination), so the two paths agree to float rounding.
    (Bitwise equality is not available: the recomputation is different arithmetic by construction.)"""
    verts, d, mask, faces, vf, tok, prev48 = _net_inputs(mano)
    hf = associate_rays(d, mask, verts, faces, vf, **_kw(2))
    net = _randomise_readout(_net(mano)).eval()
    base = net(tok, d, mask, verts, hf, prev48)
    geo = net(tok, d, mask, verts, hf, prev48, verts_geo=verts.clone())
    assert torch.allclose(base[0], geo[0], atol=2e-4, rtol=1e-3)
    assert torch.allclose(base[2]["slots"], geo[2]["slots"], atol=2e-3, rtol=1e-3)
    assert all(torch.equal(base[1][k], geo[1][k]) for k in ("xyz_valid", "xyz_hit", "xyz_resid_mass", "xyz_verts_direct"))


def test_verts_geo_geometry_comes_from_verts_geo_alone(mano):
    """Assignment from mesh T, geometry from mesh P: dXYZ / z_prior / edges follow the formulas on P and the `verts`
    argument no longer matters."""
    vT = _hand_verts(mano, 0, flex=0.9)
    vP = vT + torch.randn(2, 1, 3) * 0.004 + torch.randn(2, 778, 3) * 0.002          # prev: shifted + jittered copy
    d, mask = _rays(vT, 0, N=200)
    faces, vf = _tables(mano, 2)
    assoc = associate_rays(d, mask, vT, faces, vf, **_kw(2))                         # the oracle's assignment
    g = torch.Generator().manual_seed(7)
    tok = torch.randn(2, 200, 7, generator=g) * mask.unsqueeze(-1)
    prev48 = torch.randn(2, 48, generator=g) * 0.1
    net = _randomise_readout(_net(mano)).eval()
    captured = {}

    def _grab(mod, inp):                                    # first call = the matched records; returns None (keeps the input)
        captured.setdefault("feat", inp[0].detach().clone())

    net.rel.register_forward_pre_hook(_grab)
    out_geo = net(tok, d, mask, vP, assoc, prev48, verts_geo=vP)
    feat = captured.pop("feat")
    junk = _hand_verts(mano, 5)
    out_junk = net(tok, d, mask, junk, assoc, prev48, verts_geo=vP)                  # `verts` is ignored
    assert torch.equal(out_geo[0], out_junk[0]) and torch.equal(out_geo[2]["slots"], out_junk[2]["slots"])
    out_plain = net(tok, d, mask, vP, assoc, prev48)                                 # no verts_geo: T's z_prior
    assert not torch.allclose(out_geo[0], out_plain[0], atol=1e-6)
    # the formulas, in float64
    w = assoc["q"].unsqueeze(-1) * assoc["A"] * mask.unsqueeze(-1)
    rb, rn, rr = (w > 0).nonzero(as_tuple=True)
    vid, wr = assoc["vid"][rb, rn, rr], w[rb, rn, rr].double()
    V = vP.double()[rb, vid]
    dd = d.double()[rb, rn]
    zj = (V * dd).sum(-1) / (dd * dd).sum(-1)
    key = rb * 200 + rn
    zsum = torch.zeros(2 * 200, dtype=torch.float64).index_add(0, key, wr * zj)
    wsum = torch.zeros(2 * 200, dtype=torch.float64).index_add(0, key, wr)
    zp = (zsum / wsum)[key]
    dxyz = (zp.unsqueeze(-1) * dd - V) / L_REL
    assert feat.shape == (rb.numel(), REL_IN)
    assert torch.allclose(feat[:, 7:10].double(), dxyz, atol=2e-3, rtol=1e-4)
    assert torch.allclose(feat[:, 10].double(), (zp - 0.45) / 0.1, atol=1e-4)
    # the oracle's own z (T's hit depth) differs from the recomputed z_prior on P
    assert float((assoc["z"][rb, rn].double() - zp).abs().max()) > 1e-3


# --------------------------------------------------------------------------- (6) model keys
def _cfg(**model_over):
    from config import load_config
    cfg = load_config(str(ROOT / "configs" / "xa" / "xa_c1_s3407.yaml"))
    cfg["MANO"]["NPZ"] = NPZ
    cfg["MODEL"].update(model_over)
    return cfg


def _model(**over) -> MNISTModel:
    torch.manual_seed(0)
    m = MNISTModel(cfg=_cfg(**over))
    _randomise_readout(m.xyz)
    return m.float().eval()


def _batch(B=2, n=600, seed=0, target=None, dev_prev=None) -> EventPacketBatch:
    g = torch.Generator().manual_seed(seed)
    ev = torch.zeros(B * n, 5)
    for b in range(B):
        ev[b * n:(b + 1) * n, 0] = b
    ev[:, 1] = (122 + 22 * torch.randn(B * n, generator=g)).clamp(0, 239).round()
    ev[:, 2] = (91 + 22 * torch.randn(B * n, generator=g)).clamp(0, 179).round()
    ev[:, 3] = torch.rand(B * n, generator=g) * 0.05
    ev[:, 4] = (torch.rand(B * n, generator=g) > 0.5).float()
    prev = torch.zeros(B, 51)
    prev[:, 2] = 0.45
    prev[:, 3:6] = torch.randn(B, 3, generator=g) * 0.3
    prev[:, 6:] = torch.randn(B, 45, generator=g) * 0.4
    K = torch.tensor([[603.4507, 0, 325.09183], [0, 602.95654, 242.09796], [0, 0, 1.0]]).view(1, 3, 3).expand(B, -1, -1)
    return EventPacketBatch(
        events=ev, ptr=torch.arange(B + 1) * n, sequence_id=torch.zeros(B, dtype=torch.long),
        t_start_us=torch.zeros(B, dtype=torch.long), t_end_us=torch.full((B,), 50000),
        delta_t_s=torch.full((B,), 0.05), is_sequence_start=torch.zeros(B, dtype=torch.bool),
        is_sequence_end=torch.zeros(B, dtype=torch.bool),
        target=torch.zeros(B, 51) if target is None else target,
        prev_state=prev if dev_prev is None else dev_prev, betas=torch.zeros(B, 10), camera_K=K.clone(), lnes=None)


@pytest.fixture(scope="module")
def default_model():
    return _model()


def test_model_defaults_are_the_pre_change_model(default_model):
    m = default_model
    assert (m.xyz_assoc_mode, m.xyz_assoc_prev, m.xyz_contour_gap) == ("hit_first", "self", 0.01)
    assert m.xyz_target_override is None and not m.xyz.contour
    assert _n_params(m) == N_PARAMS_DEFAULT
    m.forward_packet(_batch())
    assert set(m.route_stats) == {"xyz_valid", "xyz_hit", "xyz_resid_mass", "xyz_verts_direct", "xyz_verts_prop"}


@pytest.mark.parametrize("over", [{"XYZ_ASSOC_MODE": "nearest"}, {"XYZ_ASSOC_MODE": "Contour "},
                                  {"XYZ_ASSOC_PREV": "gt"}, {"XYZ_ASSOC_PREV": ""},
                                  {"XYZ_CONTOUR_GAP": -0.01}, {"XYZ_CONTOUR_GAP": float("nan")},
                                  {"XYZ_CONTOUR_GAP": float("inf")}])
def test_model_rejects_illegal_values(over):
    with pytest.raises(ValueError):
        MNISTModel(cfg=_cfg(**over))


def test_model_rejects_the_keys_on_other_encoders():
    with pytest.raises(ValueError, match="belong to ENCODER=xyz_mesh"):
        MNISTModel(cfg=_cfg(ENCODER="event_gnn", XYZ_ASSOC_PREV="target"))


def test_contour_model(default_model):
    m = _model(XYZ_ASSOC_MODE="contour", XYZ_CONTOUR_GAP=0.01)
    assert m.xyz.contour and _n_params(m) == N_PARAMS_DEFAULT + 64
    batch = _batch(B=2, n=1500, seed=3)
    with torch.no_grad():
        out = m.forward_packet(batch)
    assert out.shape == (2, 51) and torch.isfinite(out).all()
    assert "xyz_ctr_mass" in m.route_stats and 0.0 <= float(m.route_stats["xyz_ctr_mass"]) < 1.0
    assert float(m.route_stats["xyz_resid_mass"]) >= 0.0
    # training step shape: loss finite, gradients flow into the ctr input
    m.train()
    pred = m.forward_packet(_batch(B=2, n=1500, seed=3, target=_batch(B=2, seed=9).prev_state))
    loss, _ = m._compute_loss(pred, _batch(B=2, seed=9).prev_state, torch.zeros(2, 10))
    loss.backward()
    assert torch.isfinite(loss) and float(m.xyz.rel[0].weight.grad[:, -1].abs().sum()) > 0
    # the default model is unaffected by the existence of the option
    with torch.no_grad():
        default_model.forward_packet(batch)
    assert "xyz_ctr_mass" not in default_model.route_stats


def test_oracle_model_needs_a_real_target():
    m = _model(XYZ_ASSOC_PREV="target")
    assert m.xyz_assoc_prev == "target" and _n_params(m) == N_PARAMS_DEFAULT
    batch = _batch()                                           # an evaluation packet: target is all zeros
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="ground truth"):
            m.forward_packet(batch)
        m.xyz_target_override = torch.zeros(2, 51)
        with pytest.raises(RuntimeError):
            m.forward_packet(batch)
        m.xyz_target_override = None
        batch2 = _batch(target=batch.prev_state.clone())       # a training batch carries batch.target
        out = m.forward_packet(batch2)
        assert torch.isfinite(out).all()


def test_oracle_with_target_equal_prev_is_the_self_model(default_model):
    """If the window-end ground truth *is* prev, the oracle's assignment is the model's own: same hits, output equal
    to float rounding (the oracle recomputes z_prior on the vertices instead of using the hit depth)."""
    orc = _model(XYZ_ASSOC_PREV="target").eval()
    batch = _batch(B=1, n=1200, seed=5)
    with torch.no_grad():
        ref = default_model.forward_packet(batch)
        ref_stats = {k: float(v) for k, v in default_model.route_stats.items()}
        orc.xyz_target_override = batch.prev_state.clone()
        got = orc.forward_packet(batch)
        stats = {k: float(v) for k, v in orc.route_stats.items()}
    assert stats["xyz_hit"] == ref_stats["xyz_hit"] and stats["xyz_valid"] == ref_stats["xyz_valid"]
    assert stats["xyz_resid_mass"] == pytest.approx(ref_stats["xyz_resid_mass"], abs=1e-6)
    assert torch.allclose(got, ref, atol=3e-4, rtol=1e-3)
    # a different ground truth changes the assignment and the output
    with torch.no_grad():
        shifted = batch.prev_state.clone()
        shifted[:, 0:2] += 0.02
        orc.xyz_target_override = shifted
        orc.xyz_keep_aux = True
        out2 = orc.forward_packet(batch)
        stats2 = {k: float(v) for k, v in orc.route_stats.items()}
    assert not torch.allclose(out2, ref, atol=1e-4) and stats2["xyz_hit"] != ref_stats["xyz_hit"]
    assert "verts_assoc" in orc.xyz_last and not torch.equal(orc.xyz_last["verts_assoc"], orc.xyz_last["verts"])
    # the override beats batch.target
    with torch.no_grad():
        orc.xyz_target_override = batch.prev_state.clone()
        got2 = orc.forward_packet(_batch(B=1, n=1200, seed=5, target=shifted))
    assert torch.equal(got2, got)


def test_oracle_train_step_is_finite():
    m = _model(XYZ_ASSOC_PREV="target").train()
    tgt = _batch(B=2, seed=11).prev_state
    pred = m.forward_packet(_batch(B=2, n=1500, seed=4, target=tgt))
    loss, _ = m._compute_loss(pred, tgt, torch.zeros(2, 10))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.xyz.parameters())


# --------------------------------------------------------------------------- track_sequence hands over pos51[end]
class _Stub:
    """Just enough of a model for `track_sequence`: records the oracle hand-over, returns prev."""
    encoder_name = "stub"

    def __init__(self, oracle: bool):
        self.seen = []
        if oracle:
            self.xyz_assoc_prev = "target"
            self.xyz_target_override = None

    def set_hand_context(self, betas, camera_K):
        pass

    def forward_packet(self, batch):
        if hasattr(self, "xyz_target_override"):
            assert self.xyz_target_override is not None
            self.seen.append(self.xyz_target_override.clone())
        return batch.prev_state


@pytest.mark.skipif(not (ROOT / "data" / "hand_data51" / "splits_semkine.json").exists(), reason="needs the data root")
def test_track_sequence_hands_the_window_end_ground_truth_to_the_oracle(mano, monkeypatch):
    from config import load_config
    from semkine import eval_track as ET
    from semkine.dataset import sequences_for_split
    monkeypatch.chdir(ROOT)
    cfg = load_config(str(ROOT / "configs" / "xa" / "xa_orc_s3407.yaml"))
    root = Path(cfg["DATA"]["ROOT"])
    seqs = dict(sequences_for_split(root, "val_core", None))
    real_load = ET.load_sequence

    def short(root_, d, seq):                                   # two runs are enough
        events, offsets, aux, pos51 = real_load(root_, d, seq)
        aux = dict(aux)
        aux["valid_runs_ms"] = np.asarray(aux["valid_runs_ms"]).reshape(-1, 2)[:2]
        return events, offsets, aux, pos51

    monkeypatch.setattr(ET, "load_sequence", short)
    _, _, aux, pos51 = short(root, seqs["zgz_local"], "zgz_local")
    ends = np.concatenate([np.arange(a + 49, b, 50) for a, b in aux["valid_runs_ms"]])
    dev = torch.device("cpu")
    orc = _Stub(True)
    ET.track_sequence(orc, mano, cfg, root, seqs["zgz_local"], "zgz_local", 50, dev, np.random.default_rng(0), 1.0)
    assert len(orc.seen) == len(ends) > 50
    got = torch.cat(orc.seen).numpy()
    assert got.shape == (len(ends), 51) and np.array_equal(got, pos51[ends].astype(got.dtype))
    assert orc.xyz_target_override is None                       # reset after the sequence
    plain = _Stub(False)
    ET.track_sequence(plain, mano, cfg, root, seqs["zgz_local"], "zgz_local", 50, dev, np.random.default_rng(0), 1.0)
    assert not hasattr(plain, "xyz_target_override") and not hasattr(plain, "xyz_assoc_prev")


# --------------------------------------------------------------------------- tools
def _tool(name):
    spec = importlib.util.spec_from_file_location(f"xa_tool_{name}", ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_run(root: Path, name: str, steps=(500, 1000, 6000)) -> Path:
    run = root / name
    run.mkdir(parents=True)
    for n in steps:
        (run / f"{name}-step={n}.ckpt").touch()
    (run / "last.ckpt").touch()
    return run


def test_make_s36_row_step_resolution(tmp_path, capsys):
    mod = _tool("make_s36_row")
    run = _fake_run(tmp_path, "xa_c1_s3407")
    assert mod.resolve_step_ckpt(run, "last") == (6000, run / "xa_c1_s3407-step=6000.ckpt")
    assert mod.resolve_step_ckpt(run, 1000) == (1000, run / "xa_c1_s3407-step=1000.ckpt")
    with pytest.raises(FileNotFoundError):
        mod.resolve_step_ckpt(run, 2000)
    (run / "epoch=0-step=6000.ckpt").touch()                     # a second 6000 checkpoint is ambiguous
    with pytest.raises(RuntimeError):
        mod.resolve_step_ckpt(run, "last")
    short = _fake_run(tmp_path, "xa_c1_s3408", steps=(500, 2500, 3000))        # no step=6000: the largest step, loudly
    assert mod.resolve_step_ckpt(short, "last")[0] == 3000 and "WARNING" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError):
        mod.resolve_step_ckpt(tmp_path, "last")
    # argument parsing and the row file name
    import argparse
    assert [mod.step_policy_arg(x) for x in ("selection", "LAST", "3000")] == ["selection", "last", 3000]
    for bad in ("abc", "0", "-5", "1.5"):
        with pytest.raises(argparse.ArgumentTypeError):
            mod.step_policy_arg(bad)
    assert mod.row_tag("val_core", None, "selection") == ""                  # the default file name is unchanged
    assert mod.row_tag("val_core", None, "last") == "_last" and mod.row_tag("val_core", None, 3000) == "_step3000"
    assert mod.row_tag("test", "zgz", "selection") == "_test_zgz" and mod.row_tag("test", "zgz", "last") == "_test_zgz_last"


def test_make_s36_row_eval_seed_wiring(tmp_path, monkeypatch):
    """`eval_seed(step_policy='last')` on a fake run directory (empty checkpoint files, no selection JSON): the
    step=6000 checkpoint is the one loaded, no selection file and no drift assertion are involved; the default
    policy still insists on the selection file."""
    mod = _tool("make_s36_row")
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(mod, "REPO", tmp_path)
    run = _fake_run(tmp_path / "outputs" / "semkine", "fake_s3407")
    (run / "training_metadata.json").write_text(json.dumps({"config_path": str(ROOT / "configs/xa/xa_c1_s3407.yaml")}))
    loaded = []

    class _M:
        def to(self, *_):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(mod.MNISTModel, "load_from_checkpoint", classmethod(lambda cls, p, **kw: loaded.append(p) or _M()))
    monkeypatch.setattr(mod, "sequences_for_split", lambda *a, **k: [("zgz_local", "val"), ("zgz_global", "val")])
    fake = lambda m, mano, cfg, root, d, s, *a, **k: {"seq": s, "n_frames": 10, **{x: 7.0 for x in mod.METRICS}}   # noqa: E731
    monkeypatch.setattr(mod.ET, "track_sequence", fake)
    agg, *_ = mod.eval_seed("fake", 3407, torch.device("cpu"), step_policy="last")
    assert agg["step"] == 6000 and agg["ckpt"] == str(run / "fake_s3407-step=6000.ckpt")
    assert loaded == [agg["ckpt"]] and agg["overall"]["mpjpe_ra_mm"] == 7.0
    agg, *_ = mod.eval_seed("fake", 3407, torch.device("cpu"), step_policy=1000)
    assert agg["step"] == 1000
    with pytest.raises(AssertionError, match="no selection file"):
        mod.eval_seed("fake", 3407, torch.device("cpu"))


def test_report_table_tag(tmp_path, monkeypatch, capsys):
    mod = _tool("report_table")
    monkeypatch.setattr(mod, "OUT", tmp_path)

    def row_json(x):
        blk = {k: {"mpjpe_ra_mm": x, "mpvpe_ra_mm": x} for k in ("overall", "local", "global")}
        return {"two_seed_mean": blk, "per_seed": {"3407": blk, "3408": blk}, "latency_ms_scaled_full1p75": 1.0,
                "macs_forward_packet": 1e9, "params_total": 1e6}

    (tmp_path / "a_main_row.json").write_text(json.dumps(row_json(11.11)))
    (tmp_path / "a_main_row_last.json").write_text(json.dumps(row_json(22.22)))
    assert "11.11" in mod.row("a") and "22.22" in mod.row("a", tag="_last") and "11.11" not in mod.row("a", tag="_last")
    for argv in (["a"], ["--tag", "last", "a"], ["--tag", "_last", "a"]):
        monkeypatch.setattr(sys, "argv", ["report_table.py", *argv])
        mod.main()
        text = capsys.readouterr().out
        assert ("11.11" in text) == (argv == ["a"]) and ("22.22" in text) == (argv != ["a"])


# --------------------------------------------------------------------------- the real configs on real data (CPU)
DATA_OK = (ROOT / "data" / "hand_data51" / "splits_semkine.json").exists()


def _arm_cfg(arm: str):
    from config import load_config
    cfg = load_config(str(ROOT / "configs" / "xa" / f"xa_{arm}_s3407.yaml"))
    cfg["DATA"]["ROOT"] = str(ROOT / cfg["DATA"]["ROOT"])
    cfg["MANO"]["NPZ"] = NPZ
    return cfg


@pytest.fixture(scope="module")
def real_train_batch():
    from semkine.dataset import build_dataset
    from semkine.events import collate_packets
    cfg = _arm_cfg("c1")
    ds = build_dataset(cfg, "train", np.load(NPZ)["hands_components"].astype(np.float32), train=True)
    idx = torch.randint(len(ds), (6,), generator=torch.Generator().manual_seed(0)).tolist()
    return collate_packets([ds[i] for i in idx])


@pytest.mark.skipif(not DATA_OK, reason="needs the data root")
@pytest.mark.parametrize("arm", ["orc", "ctr"])
@pytest.mark.parametrize("amp", [False, True], ids=["fp32", "bf16"])
def test_arm_config_training_step_on_a_real_batch(arm, amp, real_train_batch):
    """`training_step` (forward + loss) and backward of the xa_orc / xa_ctr configs on real training samples."""
    torch.manual_seed(0)
    m = MNISTModel(cfg=_arm_cfg(arm))
    _randomise_readout(m.xyz)
    m.train()
    assert (m.xyz_assoc_prev, m.xyz_assoc_mode) == (("target", "hit_first") if arm == "orc" else ("self", "contour"))
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        loss = m.training_step(real_train_batch, 0)              # orc: the assignment mesh is FK(batch.target)
    loss.backward()
    assert torch.isfinite(loss)
    norms = {n: float(p.grad.float().norm()) for n, p in m.named_parameters()}
    assert all(np.isfinite(v) for v in norms.values()) and sum(norms.values()) > 0
    for prefix in ("xyz.rel.", "xyz.vin.", "xyz.edge.", "xyz.layers.", "xyz.slot_proj.", "xyz.fc1.", "xyz.fc2."):
        assert any(v > 0 for n, v in norms.items() if n.startswith(prefix)), prefix
    st = {k: float(v) for k, v in m.route_stats.items()}
    assert all(np.isfinite(v) for v in st.values()) and 0.0 < st["xyz_valid"] <= 1.0
    if arm == "ctr":
        assert 0.0 < st["xyz_ctr_mass"] < 1.0 and float(m.xyz.rel[0].weight.grad[:, -1].abs().sum()) > 0
    else:
        assert "xyz_ctr_mass" not in st


@pytest.mark.skipif(not DATA_OK, reason="needs the data root")
def test_oracle_forward_packet_on_a_real_eval_packet():
    from semkine import eval_track as ET
    from semkine.dataset import sequences_for_split
    cfg = _arm_cfg("orc")
    root = Path(cfg["DATA"]["ROOT"])
    d = dict(sequences_for_split(root, "val_core", None))["zgz_local"]
    events, offsets, aux, pos51 = ET.load_sequence(root, d, "zgz_local")
    end = 1399                                                    # a busy 50 ms step of the first run
    ev5 = ET._window_events(events, offsets, None, end, 50)
    pkt = ET.make_eval_packet(ev5, torch.from_numpy(pos51[end - 49].copy()).view(1, -1),
                              torch.tensor(aux["betas"], dtype=torch.float32).view(1, -1),
                              torch.tensor(aux["camera_K"], dtype=torch.float32).view(1, 3, 3), 50, torch.device("cpu"))
    assert not pkt.target.any()                                   # evaluation packets carry no target
    torch.manual_seed(0)
    orc = MNISTModel(cfg=cfg)
    _randomise_readout(orc.xyz)
    orc.eval()
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="ground truth"):
            orc.forward_packet(pkt)
        orc.xyz_target_override = torch.from_numpy(pos51[end].copy()).view(1, -1)
        out = orc.forward_packet(pkt)
        st = {k: float(v) for k, v in orc.route_stats.items()}
    assert out.shape == (1, 51) and torch.isfinite(out).all() and 0.0 < st["xyz_hit"] <= st["xyz_valid"] <= 1.0
    own = MNISTModel(cfg=_arm_cfg("c1"))
    _randomise_readout(own.xyz)
    own.eval()
    with torch.no_grad():
        ref = own.forward_packet(pkt)
    assert not torch.allclose(out, ref, atol=1e-5)               # the oracle assignment changes the update
