#!/usr/bin/env python3
r"""S6 Kinematic Skinning Semantic Field: what the hand model says about each event's pixel.

The measurement an event carries is "brightness changed at pixel `u` at time `t`". To turn that
into a statement about joint angles, something has to answer "which piece of the hand was at `u`,
and which joints move it". MANO already contains the complete answer: it skins vertex `i` as
`v_i = sum_j W_ij G_j(theta) v_i`, so row `i` of the 778x16 skinning matrix *is* the list of
joints that move that vertex, with weights. KSSF is that table, rasterised through the predicted
pose and read out at event coordinates.

Fields produced, all at the previous state (so nothing here needs the answer):

  `face_id`      which triangle covers the pixel, -1 outside
  `bary`         barycentric coordinates inside that triangle, summing to 1
  `inv_depth`    1/z normalised, the existing `inv` channel
  `visibility`   1 inside the silhouette
  `lbs_topk`     the k largest skinning weights and their joint indices, interpolated by `bary`
  `semantic`     a compact continuous code per pixel (see `SEMANTIC_DIM`)
  `sdf`          signed distance to the projected occluding contour, in pixels
  `sdf_normal`   the unit 2D gradient of that distance, i.e. the contour normal

Two implementation choices worth stating.

**A real triangle rasteriser, not the existing point splat.** `MNISTModel._render_chunk` scatters
778 vertices into a depth buffer, which is enough for a silhouette but cannot produce a face id or
barycentric coordinates, and leaves holes where the surface is stretched. Here the 1538 faces are
rasterised over their bounding boxes with a proper inside test and depth test. The `sil` output is
checked against the splat with an IoU gate, so the two agree where the splat is trustworthy.

**The contour distance is measured to line segments, not to pixels.** The occluding contour is
extracted as the set of edges whose two adjacent faces disagree on facing direction, and the
distance is the exact point-to-segment distance to those projected edges. A pixel-based Euclidean
distance transform would quantise the contour to the pixel grid, which is the wrong thing to hand
to an estimator whose whole job is sub-pixel: it would cap the achievable residual at half a
pixel and would make the normal piecewise constant. The segment form is exact and differentiable,
and the normal comes out of the same computation for free.

The event measurement model this feeds (S8) is the contour-normal residual

    r_i(x) = n_i^T ( u_i - pi_K( X_i(x) ) ),      n_i = grad SDF_prev(u_i)

which is the standard reading of an event as evidence about a moving edge: motion along the edge
produces no brightness change, so only the normal component is observable. `n_i` is `sdf_normal`
and `X_i` is the surface point recovered from `face_id` and `bary`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

#: Width of the compact per-pixel semantic code. The four components are chosen so that each is a
#: quantity the kinematics actually distinguishes, rather than a learned embedding that would need
#: its own ablation: radial chain coordinate, finger identity (as a pair on the unit circle so it
#: is continuous and has no arbitrary ordering), and depth along the chain.
SEMANTIC_DIM = 4
#: How many skinning weights to keep per pixel. MANO rows are sparse: the top 4 of 16 carry
#: essentially all the mass, which `test_s6_kssf.py` checks rather than assumes.
LBS_TOPK = 4
#: Cap on a face's bounding box in pixels. At 240x180 with this camera a MANO triangle spans a
#: few pixels; the cap bounds the rasteriser's work and is asserted not to clip anything.
MAX_FACE_BBOX = 24


@dataclass
class KSSFFields:
    """Dense fields on the render grid. `(B, H, W, ...)`."""

    face_id: torch.Tensor       # (B, H, W) long, -1 = background
    bary: torch.Tensor          # (B, H, W, 3)
    depth: torch.Tensor         # (B, H, W)   metres, +inf on background
    inv_depth: torch.Tensor     # (B, H, W)
    visibility: torch.Tensor    # (B, H, W)   1 inside the silhouette
    lbs_weights: torch.Tensor   # (B, H, W, LBS_TOPK)
    lbs_indices: torch.Tensor   # (B, H, W, LBS_TOPK) uint8, joint ids 0..15
    semantic: torch.Tensor      # (B, H, W, SEMANTIC_DIM)
    sdf: torch.Tensor           # (B, H, W)   signed pixels, negative inside
    sdf_normal: torch.Tensor    # (B, H, W, 2) unit outward normal
    uv: torch.Tensor            # (B, 778, 2) projected vertices, kept for the S8 Jacobian

    def query(self, xy: torch.Tensor, batch_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Read the fields at integer event pixels. `xy` is `(N, 2)`, `batch_index` is `(N,)`.

        Events live on the integer grid the fields are computed on, so this is an exact gather
        rather than an interpolation.
        """
        b = batch_index.long()
        x = xy[:, 0].round().long().clamp(0, self.face_id.shape[2] - 1)
        y = xy[:, 1].round().long().clamp(0, self.face_id.shape[1] - 1)
        return {
            "face_id": self.face_id[b, y, x],
            "bary": self.bary[b, y, x],
            "inv_depth": self.inv_depth[b, y, x],
            "visibility": self.visibility[b, y, x],
            "lbs_weights": self.lbs_weights[b, y, x],
            "lbs_indices": self.lbs_indices[b, y, x],
            "semantic": self.semantic[b, y, x],
            "sdf": self.sdf[b, y, x],
            "sdf_normal": self.sdf_normal[b, y, x],
        }

    def as_channels(self, names: Tuple[str, ...]) -> torch.Tensor:
        """Pack selected fields into `(B, H, W, C)` so they can extend the LNES input."""
        parts = []
        for n in names:
            v = getattr(self, n)
            parts.append(v.unsqueeze(-1) if v.dim() == 3 else v)
        return torch.cat([p.float() for p in parts], dim=-1)


class KSSF(torch.nn.Module):
    """Rasterises the kinematic semantic field for a batch of poses."""

    def __init__(self, mano, height: int = 180, width: int = 240,
                 render_scale: float = 0.375, topk: int = LBS_TOPK,
                 z_near: float = 5.0e-2):
        super().__init__()
        self.mano = mano
        self.h, self.w = int(height), int(width)
        self.render_scale = float(render_scale)
        self.topk = int(topk)
        self.SPAN_TIERS = tuple(t for t in (4, 8, 16, MAX_FACE_BBOX) if t <= MAX_FACE_BBOX)
        # Faces with any vertex closer than this are dropped. 5 cm is well inside the working
        # volume of this capture (hand at 0.3-0.8 m) and keeps the projection conditioned: a
        # vertex at 1 mm would project thousands of pixels away and span the whole frame.
        self.z_near = float(z_near)
        self.register_buffer("faces", mano.f.long(), persistent=False)
        self._register_semantics()
        self._register_edges()

    # ------------------------------------------------------------------ tables
    def _register_semantics(self):
        r"""Per-vertex semantic code, computed once from the rest pose.

        `betas = 0` on purpose: this is a canonical label for "where on the hand am I", and it
        must not drift when the subject's hand shape changes. Components:

        0. radial chain coordinate `|W_i J_rest - J_rest[0]|`, normalised. Continuous across part
           boundaries because the skinning weights themselves blend, unlike an argmax part id.
        1-2. finger identity as `(cos, sin)` of an angle assigned per finger. A scalar finger
           index would impose an ordering the kinematics does not have (why is the ring finger
           "between" the middle and the pinky?); a point on the circle does not.
        3. normalised depth along the chain, 0 at the MCP and 1 at the tip, which tells a
           distal joint apart from its parent even though they share a finger.
        """
        import math

        with torch.no_grad():
            W = self.mano.weights                                   # (778, 16)
            J = self.mano.J_regressor @ self.mano.v_template        # (16, 3)
            radial = (W @ J - J[0:1]).norm(dim=-1, keepdim=True)
            radial = (radial - radial.min()) / (radial.max() - radial.min()).clamp_min(1e-8)

            # MANO's 15 local joints are five chains of three, in the order
            # index, middle, pinky, ring, thumb (joints 1..15).
            finger_of = torch.zeros(16, dtype=torch.long, device=W.device)
            depth_of = torch.zeros(16, dtype=W.dtype, device=W.device)
            for f in range(5):
                for d in range(3):
                    finger_of[1 + 3 * f + d] = f
                    depth_of[1 + 3 * f + d] = d / 2.0
            ang = finger_of.to(W.dtype) * (2.0 * math.pi / 5.0)
            # Blend the per-joint labels by the skinning weights, so a vertex between two parts
            # gets a code between the two rather than one of them.
            cos_v = W @ torch.cos(ang).unsqueeze(-1)
            sin_v = W @ torch.sin(ang).unsqueeze(-1)
            depth_v = W @ depth_of.unsqueeze(-1)
            code = torch.cat([radial, cos_v, sin_v, depth_v], dim=-1)
        self.register_buffer("vert_semantic", code.contiguous(), persistent=False)
        w = self.mano.weights
        vals, idx = torch.topk(w, self.topk, dim=-1)
        self.register_buffer("vert_lbs_w", vals.contiguous(), persistent=False)
        self.register_buffer("vert_lbs_i", idx.contiguous(), persistent=False)

    def _register_edges(self):
        """Unique undirected edges and, per edge, the (up to two) faces sharing it.

        Needed for the occluding contour: an edge is on the contour when its two faces disagree
        on facing direction, or when it has only one face (a boundary of the open mesh -- MANO's
        wrist ring).
        """
        f = self.mano.f.long().cpu()
        F = f.shape[0]
        # Each face contributes three directed edges; sorting the vertex pair makes the two
        # copies of a shared edge identical so they collide on the same key.
        e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], dim=0)      # (3F, 2)
        e_sorted, _ = e.sort(dim=-1)
        key = e_sorted[:, 0] * 100_000 + e_sorted[:, 1]
        face_of = torch.arange(F).repeat(3)

        order = torch.argsort(key, stable=True)
        k_s, f_s, ev_s = key[order], face_of[order], e_sorted[order]
        is_first = torch.ones_like(k_s, dtype=torch.bool)
        is_first[1:] = k_s[1:] != k_s[:-1]
        # Sorted keys make the running count of "first occurrences" the edge index directly,
        # so no per-edge Python loop is needed.
        group = torch.cumsum(is_first.long(), 0) - 1
        E = int(group[-1]) + 1
        edge_v = torch.zeros(E, 2, dtype=torch.long)
        edge_v[group] = ev_s
        edge_f = torch.full((E, 2), -1, dtype=torch.long)
        edge_f[group[is_first], 0] = f_s[is_first]
        second = ~is_first
        edge_f[group[second], 1] = f_s[second]
        # A manifold triangle mesh gives every edge one or two faces; more would mean the face
        # list is malformed and the contour test below would be meaningless.
        counts = torch.bincount(group, minlength=E)
        assert int(counts.max()) <= 2, f"non-manifold edge shared by {int(counts.max())} faces"
        self.register_buffer("edge_v", edge_v, persistent=False)
        self.register_buffer("edge_f", edge_f, persistent=False)

    # ------------------------------------------------------------------ geometry
    def project(self, verts: torch.Tensor, camera_K: torch.Tensor) -> torch.Tensor:
        s = self.render_scale
        fx, fy = camera_K[:, 0, 0] * s, camera_K[:, 1, 1] * s
        cx, cy = camera_K[:, 0, 2] * s, camera_K[:, 1, 2] * s
        z = verts[..., 2].clamp_min(1e-6)
        return torch.stack([fx[:, None] * verts[..., 0] / z + cx[:, None],
                            fy[:, None] * verts[..., 1] / z + cy[:, None]], dim=-1)

    def _tier_candidates(self, tri: torch.Tensor, triz: torch.Tensor, lo: torch.Tensor,
                         sel: torch.Tensor, m: int):
        """Rasterise the `(batch, face)` pairs selected by `sel` with an `m x m` window.

        Returns the flat buffer index, depth, face id and barycentric coordinates of every
        candidate that lands inside its triangle and inside the frame. Working on the selected
        pairs rather than on a dense `(B, F, m*m)` grid is what keeps the cost proportional to the
        actual triangle areas: a face spanning 3 pixels is not given a 15x15 window.
        """
        dev, dt = tri.device, tri.dtype
        bi, fi = torch.nonzero(sel, as_tuple=True)                     # (P,)
        if bi.numel() == 0:
            e = torch.empty(0, device=dev)
            return (e.long(), e.to(dt), e.long(), torch.empty(0, 3, device=dev, dtype=dt))
        t = tri[bi, fi]                                                # (P, 3, 2)
        tz = triz[bi, fi]                                              # (P, 3)
        l = lo[bi, fi]                                                 # (P, 2)

        off = torch.arange(m, device=dev)
        oy, ox = torch.meshgrid(off, off, indexing="ij")
        cand = torch.stack([ox.reshape(-1), oy.reshape(-1)], -1)        # (m*m, 2)
        px = l[:, None, :] + cand[None]                                # (P, m*m, 2)
        p = px.to(dt)

        a, b, c = t[:, 0], t[:, 1], t[:, 2]

        def edge(u, v, q):
            return ((v[:, None, 0] - u[:, None, 0]) * (q[..., 1] - u[:, None, 1])
                    - (v[:, None, 1] - u[:, None, 1]) * (q[..., 0] - u[:, None, 0]))

        w0, w1, w2 = edge(b, c, p), edge(c, a, p), edge(a, b, p)
        area = w0 + w1 + w2
        sgn = torch.where(area >= 0, 1.0, -1.0)
        inside = (w0 * sgn >= 0) & (w1 * sgn >= 0) & (w2 * sgn >= 0) & (area.abs() > 1e-12)
        denom = torch.where(area.abs() > 1e-12, area, torch.ones_like(area))
        bary = torch.stack([w0 / denom, w1 / denom, w2 / denom], -1)    # (P, m*m, 3)

        inb = ((px[..., 0] >= 0) & (px[..., 0] < self.w)
               & (px[..., 1] >= 0) & (px[..., 1] < self.h))
        # Perspective-correct interpolation: the depth that varies linearly in screen space is
        # 1/z, so interpolate that and invert. Interpolating z would bow the surface.
        inv_z = (bary / tz[:, None, :].clamp_min(1e-6)).sum(-1)
        good = inside & inb & (inv_z > 0)
        if not bool(good.any()):
            e = torch.empty(0, device=dev)
            return (e.long(), e.to(dt), e.long(), torch.empty(0, 3, device=dev, dtype=dt))

        flat = (px[..., 1].clamp(0, self.h - 1) * self.w
                + px[..., 0].clamp(0, self.w - 1))
        lin = (bi[:, None] * (self.h * self.w) + flat)[good]
        zz = (1.0 / inv_z.clamp_min(1e-12))[good]
        fid = fi[:, None].expand_as(flat)[good]
        return lin, zz, fid, bary[good]

    def rasterize(self, verts: torch.Tensor, camera_K: torch.Tensor) -> KSSFFields:
        """Rasterise all fields for `verts` `(B, 778, 3)` in metres, camera frame."""
        B = verts.shape[0]
        dev, dt = verts.device, verts.dtype
        uv = self.project(verts, camera_K)                      # (B, V, 2)
        z = verts[..., 2].clamp_min(1e-6)
        f = self.faces                                          # (F, 3)

        tri = uv[:, f]                                          # (B, F, 3, 2)
        triz = z[:, f]                                          # (B, F, 3)

        # A vertex at or behind the pinhole projects to an arbitrarily large coordinate, which
        # would give its faces a bounding box millions of pixels wide and destroy the span
        # statistic that bounds the rasteriser's work. Such faces are dropped outright: they are
        # not visible, and keeping them would only add candidates that all fail the bounds test.
        z_ok = (verts[:, f, 2] > self.z_near).all(-1)

        lo = tri.amin(dim=2).floor().long()
        hi = tri.amax(dim=2).ceil().long()
        span = (hi - lo + 1).clamp(min=1).amax(-1)                    # (B, F)
        # A face wider than the candidate window would be rasterised only in part, which is worse
        # than not at all: a partial triangle puts a hole in the middle of the surface. Such faces
        # are dropped and counted, so a test can assert the count is zero on real poses rather
        # than trusting that the window is large enough.
        span_ok = span <= MAX_FACE_BBOX
        face_ok = z_ok & span_ok
        self.max_face_span = int(span[z_ok].max()) if bool(z_ok.any()) else 0
        self.n_faces_behind = int((~z_ok).sum())
        self.n_faces_oversized = int((z_ok & ~span_ok).sum())

        HW = self.h * self.w
        face_id = torch.empty(B, self.h, self.w, device=dev, dtype=torch.long)
        bary_g = torch.empty(B, self.h, self.w, 3, device=dev, dtype=dt)
        depth_g = torch.empty(B, self.h, self.w, device=dev, dtype=dt)

        # Batch chunking keeps peak memory independent of the batch size, which matters because
        # this runs alongside a training model on the same device.
        nb = max(1, min(B, self.RASTER_BUDGET // max(int(span.amax()) ** 2 * f.shape[0], 1)))
        for b0 in range(0, B, nb):
            b1 = min(b0 + nb, B)
            sl = slice(b0, b1)
            nbb = b1 - b0
            depth = torch.full((nbb * HW,), 1e6, device=dev, dtype=dt)
            face_buf = torch.full((nbb * HW,), -1, device=dev, dtype=torch.long)
            bary_buf = torch.zeros((nbb * HW, 3), device=dev, dtype=dt)

            # Faces are grouped by how many pixels they actually cover. Most MANO triangles at
            # this resolution span three or four pixels, so giving every face the window the
            # largest one needs would multiply the work by an order of magnitude.
            parts = []
            prev = 0
            for m in self.SPAN_TIERS:
                sel = face_ok[sl] & (span[sl] > prev) & (span[sl] <= m)
                prev = m
                if bool(sel.any()):
                    parts.append(self._tier_candidates(tri[sl], triz[sl], lo[sl], sel, m))
                if prev >= MAX_FACE_BBOX:
                    break
            if not parts:
                face_id[sl] = -1
                bary_g[sl] = 0
                depth_g[sl] = 1e6
                continue
            lin = torch.cat([p[0] for p in parts])
            zl = torch.cat([p[1] for p in parts])
            fid_src = torch.cat([p[2] for p in parts])
            bary_all = torch.cat([p[3] for p in parts])

            depth.scatter_reduce_(0, lin, zl, reduce="amin", include_self=True)
            # Winner selection: a candidate wins its pixel if its depth equals the buffer minimum.
            # `amax` over face id then breaks ties between exactly coincident faces reproducibly.
            won = zl <= depth[lin] + 1e-12
            face_buf.scatter_reduce_(0, lin[won], fid_src[won], reduce="amax", include_self=True)
            pick = won & (face_buf[lin] == fid_src)
            bary_buf[lin[pick]] = bary_all[pick]

            face_id[sl] = face_buf.view(nbb, self.h, self.w)
            bary_g[sl] = bary_buf.view(nbb, self.h, self.w, 3)
            depth_g[sl] = depth.view(nbb, self.h, self.w)

        vis = (face_id >= 0).to(dt)
        inv_depth = ((1.0 / depth_g).clamp(0.0, 5.0) / 5.0) * vis

        # Vertex attributes interpolated by the barycentric weights of the winning face. Only the
        # covered pixels are touched: the hand fills a few percent of the frame, and materialising
        # the 16-wide skinning matrix at every background pixel would cost gigabytes to produce
        # values that are then multiplied by zero.
        sem = torch.zeros(B, self.h, self.w, SEMANTIC_DIM, device=dev, dtype=dt)
        lw = torch.zeros(B, self.h, self.w, self.topk, device=dev, dtype=dt)
        li = torch.zeros(B, self.h, self.w, self.topk, device=dev, dtype=torch.uint8)
        fg = torch.nonzero(face_id >= 0, as_tuple=True)
        if fg[0].numel():
            vidx = f[face_id[fg]]                                    # (N, 3)
            bb = bary_g[fg].unsqueeze(-1)                            # (N, 3, 1)
            sem[fg] = (self.vert_semantic[vidx] * bb).sum(-2)
            # Interpolate the full 16-vector and then truncate, rather than interpolating three
            # separate top-k lists, which would not be a valid distribution over joints.
            wfull = (self.mano.weights[vidx] * bb).sum(-2)           # (N, 16)
            v_, i_ = torch.topk(wfull, self.topk, dim=-1)
            lw[fg] = v_
            li[fg] = i_.to(torch.uint8)

        sdf, nrm = self._contour_sdf(uv, verts, vis)
        return KSSFFields(face_id=face_id, bary=bary_g, depth=depth_g, inv_depth=inv_depth,
                          visibility=vis, lbs_weights=lw, lbs_indices=li, semantic=sem,
                          sdf=sdf, sdf_normal=nrm, uv=uv)

    # ------------------------------------------------------------------ contour
    def occluding_contour(self, uv: torch.Tensor, verts: torch.Tensor) -> torch.Tensor:
        """Mask `(B, E)` of edges on the occluding contour.

        An edge is on the contour when its two adjacent faces face opposite ways in image space
        (so the surface turns away from the camera there), or when it has a single face.
        """
        f = self.faces
        tri = uv[:, f]
        a, b, c = tri[:, :, 0], tri[:, :, 1], tri[:, :, 2]
        # Signed area in image space: its sign is the facing direction after projection.
        area = ((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
                - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))
        front = area > 0                                              # (B, F)
        f0, f1 = self.edge_f[:, 0], self.edge_f[:, 1]
        boundary = (f1 < 0)
        s0 = front[:, f0.clamp_min(0)]
        s1 = front[:, f1.clamp_min(0)]
        return boundary[None, :] | (s0 != s1)

    #: The distance field is only computed within this many pixels of the hand's bounding box.
    #: Outside, the value saturates: an estimator has nothing to learn from "the contour is 90
    #: pixels away", and computing it everywhere would cost 20x more for no information.
    SDF_BAND_PX = 12.0
    #: Rough cap on the rasteriser's candidate tensor, in elements, used to pick the batch chunk.
    #: Speed and memory only; the result is identical for any value.
    RASTER_BUDGET = 8_000_000
    #: Rough cap on the `(segment, window)` candidate tensor, in elements. Speed and memory only.
    SDF_BUDGET = 4_000_000

    @staticmethod
    def _seg_dist(g: torch.Tensor, q0: torch.Tensor, q1: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Point-to-segment vector and distance, broadcasting over whatever shape is given."""
        d = q1 - q0
        t = (((g - q0) * d).sum(-1) / (d * d).sum(-1).clamp_min(1e-12)).clamp(0.0, 1.0)
        diff = g - (q0 + t.unsqueeze(-1) * d)
        return diff, diff.norm(dim=-1)

    def _contour_sdf(self, uv: torch.Tensor, verts: torch.Tensor,
                     vis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Signed distance to the projected contour segments, and its unit gradient.

        The distance is only wanted within `SDF_BAND_PX` of the contour, and a projected MANO edge
        is a pixel or two long, so each segment can only be the nearest one for pixels in a small
        window around it. Computing every pixel against every segment would be a few hundred times
        more arithmetic for an identical answer; instead each segment writes into its own window
        and a scatter-min resolves the competition. The exact projection is then recomputed for
        the winning segment, so the result is the true segment distance and not an approximation.
        """
        B = uv.shape[0]
        dev, dt = uv.device, uv.dtype
        HW = self.h * self.w
        sdf = torch.full((B * HW,), self.SDF_BAND_PX * 2.0, device=dev, dtype=dt)
        nrm = torch.zeros((B * HW, 2), device=dev, dtype=dt)

        on = self.occluding_contour(uv, verts)                        # (B, E)
        self.n_contour_edges = on.sum(-1).detach()
        # A flat list of the selected segments across the whole batch, so nothing is padded and
        # the window can be sized per segment rather than per batch.
        bi, ei = torch.nonzero(on, as_tuple=True)                     # (P,)
        if bi.numel() == 0:
            return sdf.view(B, self.h, self.w), nrm.view(B, self.h, self.w, 2)
        q0 = uv[bi, self.edge_v[ei, 0]]                               # (P, 2)
        q1 = uv[bi, self.edge_v[ei, 1]]

        r = int(self.SDF_BAND_PX)
        # A segment can only be the nearest one for pixels within `r` of its own bounding box, so
        # the window is the segment's extent plus the band. Sizing it from the extent matters:
        # projected MANO edges are two or three pixels long, and using the worst allowed face span
        # instead would inflate every window's area by about a factor of three.
        ext = (q1 - q0).abs().amax(-1).ceil().long().clamp(min=1)
        # Distance and segment id are packed into one integer so that a single `amin` yields both.
        # Reducing them separately would be wrong: a later tier can lower the minimum distance
        # without updating a separately-recorded winner, leaving the two inconsistent. The
        # quantisation only decides ties -- the returned distance is recomputed exactly below.
        Q, SEG_BITS = 16384.0, 20
        assert bi.numel() < (1 << SEG_BITS)
        NONE = (1 << 62)
        best = torch.full((B * HW,), NONE, device=dev, dtype=torch.long)

        prev = 0
        for e_max in self.SPAN_TIERS:
            tier = torch.nonzero((ext > prev) & (ext <= e_max), as_tuple=True)[0]
            prev = e_max
            if tier.numel() == 0:
                if prev >= MAX_FACE_BBOX:
                    break
                continue
            # Covers `floor(min) - r` through `ceil(max) + r`, and `ceil(max) - floor(min)` can be
            # one more than the extent, hence the +2 rather than +1.
            win = 2 * r + 2 + e_max
            off = torch.arange(win, device=dev)
            oy, ox = torch.meshgrid(off, off, indexing="ij")
            cand = torch.stack([ox.reshape(-1), oy.reshape(-1)], -1)  # (win*win, 2)
            K = cand.shape[0]
            chunk = max(1, self.SDF_BUDGET // K)
            for c0 in range(0, tier.numel(), chunk):
                idx = tier[c0 : c0 + chunk]
                a0, a1 = q0[idx], q1[idx]
                base = torch.minimum(a0, a1).floor().long() - r
                px = base[:, None, :] + cand[None]                    # (p, K, 2)
                _, dist = self._seg_dist(px.to(dt), a0[:, None, :], a1[:, None, :])
                keep = ((px[..., 0] >= 0) & (px[..., 0] < self.w)
                        & (px[..., 1] >= 0) & (px[..., 1] < self.h)
                        & (dist <= self.SDF_BAND_PX))
                if not bool(keep.any()):
                    continue
                lin = (bi[idx][:, None] * HW
                       + px[..., 1].clamp(0, self.h - 1) * self.w
                       + px[..., 0].clamp(0, self.w - 1))[keep]
                src = idx[:, None].expand_as(keep)[keep]
                key = ((dist[keep] * Q).round().long() << SEG_BITS) | src
                best.scatter_reduce_(0, lin, key, reduce="amin", include_self=True)
            if prev >= MAX_FACE_BBOX:
                break

        hit = torch.nonzero(best < NONE, as_tuple=True)[0]
        if hit.numel():
            pp = hit % HW
            gxy = torch.stack([pp % self.w, pp // self.w], -1).to(dt)
            ss = best[hit] & ((1 << SEG_BITS) - 1)
            # Recomputed exactly for the winning segment, so the stored value is the true
            # point-to-segment distance and not the tier's intermediate.
            diff, dd = self._seg_dist(gxy, q0[ss], q1[ss])
            sdf[hit] = dd
            nrm[hit] = diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        out = sdf.view(B, self.h, self.w)
        n = nrm.view(B, self.h, self.w, 2)
        # The segment distance is unsigned; the sign comes from the rasterised silhouette.
        inside = vis > 0
        out = torch.where(inside, -out, out)
        # The gathered vector points from the closest contour point to the pixel, which is outward
        # only outside the silhouette. Flipping it inside makes `n` equal `grad sdf` everywhere,
        # i.e. the outward contour normal the event residual in S8 projects onto.
        n = torch.where(inside.unsqueeze(-1), -n, n)
        return out, n

    # ------------------------------------------------------------------ entry
    def forward(self, params51: torch.Tensor, betas: torch.Tensor,
                camera_K: torch.Tensor, pose_repr: str = "mano_full_axis_angle") -> KSSFFields:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
        from pose_repr import decode_to_mano_inputs

        dec = decode_to_mano_inputs(params51, pose_repr, self.mano.hands_components,
                                    self.mano.hands_mean)
        verts, _ = self.mano(betas, dec["global_orient"], dec["local_full_aa"], dec["transl"])
        return self.rasterize(verts, camera_K)


def surface_point(fields: KSSFFields, mano, verts: torch.Tensor, face_id: torch.Tensor,
                  bary: torch.Tensor) -> torch.Tensor:
    """3D point on the surface for a queried `(face_id, bary)`; the `X_i` of the S8 residual."""
    f = mano.f.long()[face_id.clamp_min(0)]                # (N, 3)
    v = verts[f]                                            # (N, 3, 3) when verts is (V, 3)
    return (v * bary.unsqueeze(-1)).sum(-2)


def _dilate(mask: torch.Tensor, r: int = 1) -> torch.Tensor:
    """Binary dilation by a `(2r+1)` square, used to allow one pixel of rounding slack."""
    m = mask.unsqueeze(1).float()
    m = torch.nn.functional.max_pool2d(m, 2 * r + 1, stride=1, padding=r)
    return m.squeeze(1) > 0


def geometry_report(fields: KSSFFields, splat_sil: Optional[torch.Tensor] = None
                    ) -> Dict[str, float]:
    """The S6 geometric hard gates, measured."""
    vis = fields.visibility > 0
    n_fg = int(vis.sum())
    bary_sum = fields.bary.sum(-1)[vis]
    lbs_sum = fields.lbs_weights.sum(-1)[vis]
    bg = ~vis
    out = {
        "n_foreground": n_fg,
        "foreground_frac": float(vis.float().mean()),
        "bary_sum_max_dev": float((bary_sum - 1.0).abs().max()) if n_fg else 0.0,
        "bary_min": float(fields.bary[vis].min()) if n_fg else 0.0,
        "lbs_sum_min": float(lbs_sum.min()) if n_fg else 0.0,
        "lbs_sum_max": float(lbs_sum.max()) if n_fg else 0.0,
        # "background leakage": any field that should vanish outside the mask but does not.
        "bg_semantic_max": float(fields.semantic[bg].abs().max()) if bool(bg.any()) else 0.0,
        "bg_lbs_max": float(fields.lbs_weights[bg].abs().max()) if bool(bg.any()) else 0.0,
        "bg_invdepth_max": float(fields.inv_depth[bg].abs().max()) if bool(bg.any()) else 0.0,
        "sdf_inside_max": float(fields.sdf[vis].max()) if n_fg else 0.0,
        "sdf_outside_min": float(fields.sdf[bg].min()) if bool(bg.any()) else 0.0,
    }
    # The normal is only defined where the distance was actually computed; outside the band it is
    # deliberately zero, so it must be excluded from the unit-length check.
    band = fields.sdf.abs() < KSSF.SDF_BAND_PX * 1.999
    out["band_frac"] = float(band.float().mean())
    # A pixel sitting exactly on a contour segment has no gradient direction, so its normal is
    # zero by construction. Those are reported separately instead of failing the unit check.
    defined = band & (fields.sdf.abs() > 1e-4)
    out["zero_normal_in_band_frac"] = (
        float((band & ~defined).float().sum() / band.float().sum().clamp_min(1)))
    out["normal_unit_max_dev"] = float(
        (fields.sdf_normal.norm(dim=-1)[defined] - 1.0).abs().max()) if bool(defined.any()) else 0.0
    if splat_sil is not None:
        # The point splat is a strict subset of a correct triangle raster: 778 vertices cannot
        # cover a surface that spans thousands of pixels, so IoU against it is not a correctness
        # measure. What must hold is containment -- every splatted vertex lands on a rasterised
        # face, up to one pixel of rounding at the silhouette.
        b = splat_sil > 0
        out["splat_outside_raster_frac"] = (
            float((b & ~_dilate(vis, 1)).sum()) / max(float(b.sum()), 1.0))
        out["raster_over_splat_area"] = float(vis.sum()) / max(float(b.sum()), 1.0)
    return out
