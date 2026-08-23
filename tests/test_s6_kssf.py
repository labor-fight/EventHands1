#!/usr/bin/env python3
r"""S6 gates for the Kinematic Skinning Semantic Field.

Two kinds of check live here, and the distinction matters for what a PASS means.

**Geometric gates** ask whether the field is what it claims to be, and they are checked against
an independent reference rather than against themselves. The rasteriser is compared to a
brute-force implementation that loops over every pixel and tests every face; the contour distance
is compared to an exhaustive point-to-segment minimum; the normal is compared to a central
difference of the distance field. A rasteriser that agrees with brute force is correct, full stop,
which is a much stronger statement than "the barycentric coordinates sum to one".

**Mechanism gates** ask whether the field carries the information the method needs. The one that
matters for S9 is localisation: if the field's skinning weights are to tell an estimator which
joints a given event constrains, then perturbing joint `k` must move the pixels whose weights name
`k` and leave the others alone. That is checked directly, per joint, for all fifteen.

`z_near` rejection is exercised on purpose: about 2% of frames in some sequences carry a
degenerate label (the fitted hand ends up straddling the image plane, minimum vertex depth around
-5 cm). The field must stay finite and well-formed on those, and must drop nothing on the frames
where the label is sane.
"""
from __future__ import annotations

import glob
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
from semkine import kssf as KS                                      # noqa: E402
from semkine.dataset import _read_meta51                            # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H, W, SCALE = 180, 240, 0.375


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def mano():
    return ManoLayer(REPO / "assets/mano_right.npz", add_mean=False).to(DEV).eval()


@pytest.fixture(scope="module")
def field(mano):
    return KS.KSSF(mano, H, W, SCALE).to(DEV)


@pytest.fixture(scope="module")
def poses(mano):
    """A batch of real poses whose labels are sane, plus the camera intrinsics they go with.

    Frames are drawn from several sequences so the batch spans hand scales and orientations,
    and frames with a degenerate label are filtered out here so the geometric gates measure the
    rasteriser rather than the annotation.
    """
    metas = sorted(glob.glob(str(REPO / "data/hand_data51/*/*.meta")))
    assert metas, "hand_data51 not found"
    P, Bt, Km = [], [], []
    for m in metas[::11][:6]:
        base = m[:-5]
        p = _read_meta51(base + ".meta")
        aux = np.load(base + "_aux.npz", allow_pickle=True)
        b = torch.tensor(aux["betas"], dtype=torch.float32, device=DEV).view(1, -1)
        k = torch.tensor(aux["camera_K"], dtype=torch.float32, device=DEV).view(1, 3, 3)
        idx = np.linspace(0, len(p) - 1, 24).astype(int)
        pp = torch.from_numpy(p[idx]).to(DEV)
        with torch.no_grad():
            d = decode_to_mano_inputs(pp, "mano_full_axis_angle",
                                      mano.hands_components, mano.hands_mean)
            v, _ = mano(b.expand(len(pp), -1), d["global_orient"],
                        d["local_full_aa"], d["transl"])
        keep = v[..., 2].amin(-1) > 0.10
        pp = pp[keep]
        P.append(pp)
        Bt.append(b.expand(len(pp), -1))
        Km.append(k.expand(len(pp), -1, -1))
    return torch.cat(P), torch.cat(Bt), torch.cat(Km)


@pytest.fixture(scope="module")
def verts(mano, poses):
    p, b, _ = poses
    with torch.no_grad():
        d = decode_to_mano_inputs(p, "mano_full_axis_angle",
                                  mano.hands_components, mano.hands_mean)
        v, _ = mano(b, d["global_orient"], d["local_full_aa"], d["transl"])
    return v


@pytest.fixture(scope="module")
def fields(field, verts, poses):
    _, _, k = poses
    with torch.no_grad():
        return field.rasterize(verts, k)


# ----------------------------------------------------------------------------- reference
def brute_force_raster(field, verts, camera_K, bi: int):
    """Reference rasteriser: every pixel against every face, nearest front surface wins.

    Deliberately written in the least clever way available -- no bounding boxes, no scatter, no
    tie-break encoding -- so that agreement with it is evidence about the fast path rather than
    a shared bug.
    """
    uv = field.project(verts, camera_K)[bi]                      # (V, 2)
    z = verts[bi, :, 2]
    f = field.faces
    tri = uv[f]                                                  # (F, 3, 2)
    triz = z[f]
    ok = (triz > field.z_near).all(-1)

    gy, gx = torch.meshgrid(torch.arange(H, device=uv.device, dtype=uv.dtype),
                            torch.arange(W, device=uv.device, dtype=uv.dtype), indexing="ij")
    p = torch.stack([gx, gy], -1).reshape(-1, 1, 2)              # (HW, 1, 2)
    a, b, c = tri[None, :, 0], tri[None, :, 1], tri[None, :, 2]

    def edge(u, v, q):
        return ((v[..., 0] - u[..., 0]) * (q[..., 1] - u[..., 1])
                - (v[..., 1] - u[..., 1]) * (q[..., 0] - u[..., 0]))

    w0, w1, w2 = edge(b, c, p), edge(c, a, p), edge(a, b, p)
    area = w0 + w1 + w2
    s = torch.where(area >= 0, 1.0, -1.0)
    inside = (w0 * s >= 0) & (w1 * s >= 0) & (w2 * s >= 0) & (area.abs() > 1e-12) & ok[None]
    bary = torch.stack([w0, w1, w2], -1) / torch.where(area.abs() > 1e-12, area,
                                                       torch.ones_like(area))[..., None]
    inv_z = (bary / triz[None].clamp_min(1e-6)).sum(-1)
    zz = torch.where(inside & (inv_z > 0), 1.0 / inv_z.clamp_min(1e-12),
                     torch.full_like(inv_z, 1e6))
    depth, which = zz.min(dim=1)
    fid = torch.where(depth < 1e5, which, torch.full_like(which, -1))
    return fid.view(H, W), depth.view(H, W)


# ----------------------------------------------------------------------------- geometry
def test_rasteriser_agrees_with_brute_force(field, verts, poses, fields):
    """The hard gate on the fast path: identical coverage, and identical depth where covered."""
    _, _, k = poses
    n_pix, n_bad_cov, n_bad_depth = 0, 0, 0
    for bi in range(min(4, verts.shape[0])):
        ref_f, ref_d = brute_force_raster(field, verts, k, bi)
        got_f = fields.face_id[bi]
        got_d = fields.depth[bi]
        cov_ref, cov_got = ref_f >= 0, got_f >= 0
        n_pix += cov_ref.numel()
        n_bad_cov += int((cov_ref != cov_got).sum())
        both = cov_ref & cov_got
        # Face ids may differ where two faces are exactly coincident in depth; the depth is the
        # quantity that must match, and it is compared at the scale of the field (metres).
        n_bad_depth += int(((ref_d[both] - got_d[both]).abs() > 1e-5).sum())
    assert n_bad_cov / n_pix < 1e-4, f"coverage disagrees on {n_bad_cov}/{n_pix} pixels"
    assert n_bad_depth / n_pix < 1e-4, f"depth disagrees on {n_bad_depth}/{n_pix} pixels"


def test_no_faces_dropped_on_sane_labels(field, verts, poses):
    """Nothing may be silently discarded when the label is well behaved."""
    _, _, k = poses
    with torch.no_grad():
        field.rasterize(verts, k)
    assert field.n_faces_behind == 0
    assert field.n_faces_oversized == 0, (
        f"{field.n_faces_oversized} faces exceeded the {KS.MAX_FACE_BBOX}px candidate window; "
        f"largest span was {field.max_face_span}px")
    assert field.max_face_span <= KS.MAX_FACE_BBOX


def test_barycentric_and_skinning_are_valid_distributions(fields):
    rep = KS.geometry_report(fields)
    assert rep["bary_sum_max_dev"] < 1e-5
    assert rep["bary_min"] >= -1e-6
    # The plan's gate is that the top-k skinning weights carry essentially all the mass, so that
    # the truncation is a compression and not a loss. MANO rows really are this sparse.
    assert rep["lbs_sum_min"] > 0.98, rep["lbs_sum_min"]
    assert rep["lbs_sum_max"] < 1.0 + 1e-5


def test_background_carries_no_signal(fields):
    """Every field that is only defined on the hand must be exactly zero off it."""
    rep = KS.geometry_report(fields)
    assert rep["bg_semantic_max"] == 0.0
    assert rep["bg_lbs_max"] == 0.0
    assert rep["bg_invdepth_max"] == 0.0
    assert rep["foreground_frac"] < 0.5, "silhouette implausibly large"
    assert rep["foreground_frac"] > 0.002, "silhouette implausibly small"


def test_splat_silhouette_is_contained_in_the_raster(field, verts, poses, fields):
    """The existing point splat must be a subset of the raster, up to one pixel of rounding.

    Stated this way rather than as an IoU on purpose: 778 splatted vertices cannot cover a
    surface spanning thousands of pixels, so the splat is a sparse sample of the same silhouette
    and an IoU against it would fail for a *correct* rasteriser. What is falsifiable is
    containment -- a splatted vertex landing off the raster means the raster has a hole.
    """
    _, _, k = poses
    B = verts.shape[0]
    uv = field.project(verts, k)
    ui, vi = uv[..., 0].round().long(), uv[..., 1].round().long()
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (verts[..., 2] > field.z_near)
    sil = torch.zeros(B, H, W, device=DEV)
    bidx = torch.arange(B, device=DEV)[:, None].expand_as(ui)
    lin = bidx * H * W + vi.clamp(0, H - 1) * W + ui.clamp(0, W - 1)
    sil.view(-1)[lin[ok]] = 1.0
    rep = KS.geometry_report(fields, splat_sil=sil)
    assert rep["splat_outside_raster_frac"] < 0.01, rep["splat_outside_raster_frac"]
    # And the raster must be substantially denser, which is the point of replacing the splat.
    assert rep["raster_over_splat_area"] > 1.5, rep["raster_over_splat_area"]


# ----------------------------------------------------------------------------- contour / SDF
def test_sdf_sign_matches_the_silhouette(fields):
    rep = KS.geometry_report(fields)
    assert rep["sdf_inside_max"] <= 0.0
    assert rep["sdf_outside_min"] >= 0.0
    assert rep["normal_unit_max_dev"] < 1e-5
    assert 0.02 < rep["band_frac"] < 0.9
    # Only the measure-zero set of pixels lying exactly on the contour may lack a normal.
    assert rep["zero_normal_in_band_frac"] < 1e-3, rep["zero_normal_in_band_frac"]


def test_sdf_equals_exhaustive_point_to_segment_distance(field, verts, poses, fields):
    """The distance must be the exact segment distance, not a pixel-grid approximation."""
    _, _, k = poses
    uv = field.project(verts, k)
    on = field.occluding_contour(uv, verts)
    worst = 0.0
    for bi in range(min(3, verts.shape[0])):
        sel = on[bi]
        q0, q1 = uv[bi][field.edge_v[sel, 0]], uv[bi][field.edge_v[sel, 1]]
        band = fields.sdf[bi].abs() < KS.KSSF.SDF_BAND_PX * 1.5
        ys, xs = torch.nonzero(band, as_tuple=True)
        pick = torch.randperm(len(ys), device=DEV)[:400]
        g = torch.stack([xs[pick], ys[pick]], -1).to(uv.dtype)
        d = q1 - q0
        t = (((g[:, None] - q0[None]) * d[None]).sum(-1)
             / (d * d).sum(-1).clamp_min(1e-12)[None]).clamp(0, 1)
        ref = (g[:, None] - (q0[None] + t[..., None] * d[None])).norm(dim=-1).min(dim=1).values
        got = fields.sdf[bi][ys[pick], xs[pick]].abs()
        worst = max(worst, float((ref - got).abs().max()))
    assert worst < 1e-3, f"sdf deviates from exhaustive segment distance by {worst}px"


def test_normal_is_the_gradient_of_the_distance_field(field, verts, poses, fields):
    """Central differences of the sdf must reproduce `sdf_normal`.

    This is the property S8 depends on: the event residual projects onto `grad SDF`, so a normal
    that merely points "roughly outward" would bias every measurement. Pixels adjacent to the
    silhouette are excluded because the sign flip there makes the difference quotient meaningless
    at one-pixel spacing.
    """
    bad, tot = 0, 0
    for bi in range(min(3, verts.shape[0])):
        s = fields.sdf[bi]
        interior = (s.abs() > 1.5) & (s.abs() < KS.KSSF.SDF_BAND_PX - 1.5)
        gx = (s[1:-1, 2:] - s[1:-1, :-2]) * 0.5
        gy = (s[2:, 1:-1] - s[:-2, 1:-1]) * 0.5
        g = torch.stack([gx, gy], -1)
        m = interior[1:-1, 1:-1] & (g.norm(dim=-1) > 0.5)
        ghat = g[m] / g[m].norm(dim=-1, keepdim=True)
        nhat = fields.sdf_normal[bi][1:-1, 1:-1][m]
        cos = (ghat * nhat).sum(-1)
        tot += int(m.sum())
        bad += int((cos < 0.9).sum())
    # A minority of pixels sit on a medial ridge where the closest contour segment changes between
    # neighbours; there the finite difference is not the gradient of anything smooth.
    assert tot > 500, "not enough interior band pixels to test"
    assert bad / tot < 0.05, f"normal disagrees with grad(sdf) on {bad}/{tot} pixels"


def test_contour_is_a_small_closed_subset_of_edges(field, verts, poses):
    _, _, k = poses
    uv = field.project(verts, k)
    on = field.occluding_contour(uv, verts)
    frac = on.float().mean(-1)
    # A silhouette is a curve on a surface: it must be a small fraction of all edges, and it must
    # not be empty for any pose.
    assert float(frac.min()) > 0.01, float(frac.min())
    assert float(frac.max()) < 0.35, float(frac.max())


# ----------------------------------------------------------------------------- query
def test_query_gathers_exactly_what_indexing_would(fields):
    B = fields.face_id.shape[0]
    g = torch.Generator(device="cpu").manual_seed(0)
    n = 4096
    bi = torch.randint(0, B, (n,), generator=g).to(DEV)
    xs = torch.randint(0, W, (n,), generator=g).to(DEV)
    ys = torch.randint(0, H, (n,), generator=g).to(DEV)
    q = fields.query(torch.stack([xs, ys], -1).float(), bi)
    assert torch.equal(q["face_id"], fields.face_id[bi, ys, xs])
    assert torch.equal(q["bary"], fields.bary[bi, ys, xs])
    assert torch.equal(q["sdf"], fields.sdf[bi, ys, xs])
    assert torch.equal(q["lbs_indices"], fields.lbs_indices[bi, ys, xs])
    assert torch.equal(q["sdf_normal"], fields.sdf_normal[bi, ys, xs])


def test_query_clamps_out_of_range_coordinates(fields):
    xy = torch.tensor([[-5.0, -5.0], [1e4, 1e4]], device=DEV)
    q = fields.query(xy, torch.zeros(2, dtype=torch.long, device=DEV))
    assert torch.equal(q["face_id"][0], fields.face_id[0, 0, 0])
    assert torch.equal(q["face_id"][1], fields.face_id[0, H - 1, W - 1])


def test_surface_point_reproduces_the_interpolated_vertex(field, verts, fields):
    """`surface_point` must land on the mesh: reprojecting it must return the pixel it came from."""
    _, _, k = None, None, None
    for bi in range(min(2, verts.shape[0])):
        ys, xs = torch.nonzero(fields.face_id[bi] >= 0, as_tuple=True)
        pick = torch.randperm(len(ys), device=DEV)[:512]
        fid = fields.face_id[bi][ys[pick], xs[pick]]
        bary = fields.bary[bi][ys[pick], xs[pick]]
        X = KS.surface_point(fields, field.mano, verts[bi], fid, bary)
        # The depth of that point must equal the depth buffer at the same pixel.
        d = fields.depth[bi][ys[pick], xs[pick]]
        assert float((X[:, 2] - d).abs().max()) < 1e-4


# ----------------------------------------------------------------------------- mechanism
def test_semantic_code_separates_fingers_and_is_continuous(field):
    """The per-vertex code must actually distinguish the parts it claims to.

    Both halves matter. If the finger components did not separate, the code would carry no part
    identity and S9's routing would have nothing to condition on. And the code must be a blend of
    per-joint labels rather than an argmax part id, because an argmax injects a jump into a field
    that feeds a Jacobian.

    The second half is stated as a Lipschitz bound in the skinning weights, not as plain spatial
    continuity. Where the index and middle fingers touch, the weights themselves are nearly
    one-hot on different joints, so the code *must* jump there -- that discontinuity belongs to
    the hand, not to the encoding. What the encoding owes us is that it never adds a jump of its
    own: `code = W L` gives `||dcode|| <= ||dW||_1 max_j ||L_j||` exactly, and that inequality is
    checked on every edge of the mesh.
    """
    sem = field.vert_semantic
    assert sem.shape[1] == KS.SEMANTIC_DIM
    w = field.mano.weights
    part = w.argmax(-1)
    ang = sem[:, 1:3]
    # Vertices dominated by different fingers must be separated on the circle; the five fingers
    # are 72 degrees apart, so their mean directions must be well apart.
    means = []
    for f in range(5):
        m = (part >= 1 + 3 * f) & (part <= 3 + 3 * f)
        if int(m.sum()) > 10:
            means.append(ang[m].mean(0))
    assert len(means) == 5
    M = torch.stack(means)
    M = M / M.norm(dim=-1, keepdim=True)
    cos = M @ M.T
    off = cos[~torch.eye(5, dtype=torch.bool, device=cos.device)]
    assert float(off.max()) < 0.95, "two fingers share a direction in the semantic code"

    # The angular and chain-depth components must be exactly linear in the skinning weights.
    import math
    fo = torch.zeros(16, dtype=torch.long, device=w.device)
    do = torch.zeros(16, dtype=w.dtype, device=w.device)
    for fi in range(5):
        for dd in range(3):
            fo[1 + 3 * fi + dd] = fi
            do[1 + 3 * fi + dd] = dd / 2.0
    a = fo.to(w.dtype) * (2.0 * math.pi / 5.0)
    L = torch.stack([torch.cos(a), torch.sin(a), do], -1)             # (16, 3)
    assert torch.allclose(sem[:, 1:4], w @ L, atol=1e-6), "code is not a linear blend of W"

    # Lipschitz bound across every mesh edge, with the constant derived from the label tables.
    J = field.mano.J_regressor @ field.mano.v_template
    c_ang = float(L.norm(dim=-1).max())
    radial = (w @ J - J[0:1]).norm(dim=-1)
    c_rad = float((J - J[0:1]).norm(dim=-1).max()
                  / (radial.max() - radial.min()).clamp_min(1e-8))
    C = math.sqrt(c_rad ** 2 + c_ang ** 2)
    e = field.edge_v
    dw = (w[e[:, 0]] - w[e[:, 1]]).abs().sum(-1)
    jump = (sem[e[:, 0]] - sem[e[:, 1]]).norm(dim=-1)
    slack = jump - C * dw
    assert float(slack.max()) < 1e-5, (
        f"code jumps by {float(jump.max()):.3f} where the weights only move "
        f"{float(dw[slack.argmax()]):.3f}; the encoding is adding a discontinuity")


def test_lbs_topk_captures_the_mass(field):
    w = field.mano.weights
    top = torch.topk(w, KS.LBS_TOPK, dim=-1).values.sum(-1)
    assert float(top.min()) > 0.98, float(top.min())


@pytest.mark.parametrize("joint", list(range(1, 16)))
def test_joint_perturbation_moves_the_pixels_that_name_that_joint(field, mano, poses, joint):
    r"""The S6 mechanism gate, per joint.

    Rotate joint `k` by a small angle and ask where the field changed. If the skinning weights in
    the field are to tell an estimator which joints an event constrains, the change must be
    concentrated on pixels whose top-k weights include `k` or one of its descendants -- the set
    LBS itself says that joint moves. A field that changed everywhere, or in the wrong place,
    would make S9's routing meaningless no matter how good the router is.

    "Descendants" is the right set rather than `k` alone: rotating an MCP carries the whole finger
    with it, so the distal vertices move even though their weight on `k` is zero.
    """
    p, b, k_mat = poses
    n = min(6, p.shape[0])
    p, b, k_mat = p[:n], b[:n], k_mat[:n]

    # Descendants of `joint` in MANO's chain: joints 1..15 are five chains of three.
    f_idx, d_idx = (joint - 1) // 3, (joint - 1) % 3
    family = [1 + 3 * f_idx + d for d in range(d_idx, 3)]

    def render(pp):
        with torch.no_grad():
            d = decode_to_mano_inputs(pp, "mano_full_axis_angle",
                                      mano.hands_components, mano.hands_mean)
            v, _ = mano(b, d["global_orient"], d["local_full_aa"], d["transl"])
            return field.rasterize(v, k_mat)

    f0 = render(p)
    p1 = p.clone()
    # 51D layout is [t3 | R3 | residual45]; joint j occupies residual slots 3(j-1)..3(j-1)+2.
    p1[:, 6 + 3 * (joint - 1) + 1] += 0.25
    f1 = render(p1)

    moved = (f0.inv_depth - f1.inv_depth).abs() + (f0.visibility - f1.visibility).abs()
    thresh = 0.02
    changed = moved > thresh
    assert int(changed.sum()) > 20, f"joint {joint}: perturbation produced no visible change"

    fam = torch.tensor(family, device=DEV)
    names = (f0.lbs_indices.unsqueeze(-1) == fam).any(-1).any(-1) & (f0.visibility > 0)
    # Where did the change land? Everything is measured on the pre-perturbation field, since that
    # is the one an estimator would route with.
    inside_fg = changed & (f0.visibility > 0)
    hit = int((inside_fg & names).sum())
    tot = int(inside_fg.sum())
    assert tot > 0
    precision = hit / tot
    assert precision > 0.5, (
        f"joint {joint}: only {precision:.2%} of the changed foreground pixels are ones whose "
        f"skinning weights name joints {family}")


def test_rasterisation_is_deterministic(field, verts, poses):
    _, _, k = poses
    with torch.no_grad():
        a = field.rasterize(verts, k)
        b = field.rasterize(verts, k)
    assert torch.equal(a.face_id, b.face_id)
    assert torch.equal(a.bary, b.bary)
    assert torch.equal(a.sdf, b.sdf)
    assert torch.equal(a.lbs_weights, b.lbs_weights)


def test_degenerate_labels_stay_finite(field, mano):
    """A frame whose label puts the hand behind the image plane must not produce NaNs.

    These exist in the data (about 2% of frames in a few sequences), so the field has to survive
    them rather than assume they were filtered upstream.
    """
    metas = sorted(glob.glob(str(REPO / "data/hand_data51/*/*.meta")))
    found = None
    for m in metas:
        base = m[:-5]
        p = _read_meta51(base + ".meta")
        aux = np.load(base + "_aux.npz", allow_pickle=True)
        b = torch.tensor(aux["betas"], dtype=torch.float32, device=DEV).view(1, -1)
        kk = torch.tensor(aux["camera_K"], dtype=torch.float32, device=DEV).view(1, 3, 3)
        idx = np.linspace(0, len(p) - 1, 200).astype(int)
        pp = torch.from_numpy(p[idx]).to(DEV)
        with torch.no_grad():
            d = decode_to_mano_inputs(pp, "mano_full_axis_angle",
                                      mano.hands_components, mano.hands_mean)
            v, _ = mano(b.expand(len(pp), -1), d["global_orient"],
                        d["local_full_aa"], d["transl"])
        bad = v[..., 2].amin(-1) < field.z_near
        if bool(bad.any()):
            found = (v[bad][:4], kk.expand(int(bad.sum()), -1, -1)[:4])
            break
    if found is None:
        pytest.skip("no degenerate label found in the scan")
    v, kk = found
    with torch.no_grad():
        f = field.rasterize(v, kk)
    for name in ("bary", "depth", "inv_depth", "sdf", "sdf_normal", "semantic", "lbs_weights"):
        t = getattr(f, name)
        assert torch.isfinite(t).all(), f"{name} not finite on a degenerate label"
    assert field.n_faces_behind > 0, "the degenerate frame should have tripped the z_near guard"
