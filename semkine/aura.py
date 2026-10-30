r"""AURA association variants of the S37-XYZ C1 module (docs/EXPERIMENT_RECORDS.md#aura-prereg-20261024).

The C1 association (`semkine.xyz_mesh.associate_rays`, hit_first) casts each event ray against the previous-state MANO
mesh: the nearest positive hit on the faces around the 16 angularly nearest projected vertices (barycentric weights,
q = 1), else a "near" record set inside a 16 px image band (the front depth layer, <= 4 vertices, Gaussian angular
weights, q < 1), else nothing (the residual slot). Its "aura" is that image band. This module re-implements the rule
(`associate("c1", ...)` equals the library bit for bit) and builds single-factor variants of it:

* aura extent / units / construction:
    surf      surface hits only (no band)
    b8 / b32  band 8 / 32 px (sigma = band / 2), candidates unchanged
    mM / m10 / m20   metric band: candidate j is in the band when its perpendicular distance to the ray is <= r
              (`R_MATCH` for mM, 1 / 2 cm), Gaussian sigma = r / 2 on the same distance; reference, same set and
              front layer stay angular as in C1
    aura10 / aura20  an Aura-Mesh-like metric shell: hits as C1; otherwise the candidate face with the smallest
              ray-to-triangle distance delta (attained on an edge), recorded when delta <= r on that face's vertices
              with the foot point's barycentrics, q = 1 - (delta / r)^2, z = depth of the closest ray point
* 3D logic: p2d / p2dvis (the 4 nearest projected [z-buffer visible] vertices inside the 16 px band, no ray / surface
  test, hit = 0), p2dh (p2dvis with C1's hit bit put back), vis (C1 with its near candidates restricted to z-buffer
  visible vertices: the occlusion test of the original design contract)
* binding nulls: `shuffle_assoc` (a deterministic derangement among associated events, global or within one hard LBS
  part; the support multiset and per-vertex masses are unchanged), `none_assoc` (every event to the residual)
* probe helpers: occupancy masses, joint nodes and joint binning (the coarse 2D baseline), the shell-node graph
* D: `apply_aura` (the association hook of `MODEL.XYZ_AURA`) and `AuraXYZMeshNet` (`MODEL.XYZ_AURA: shell`: the
  near records live on 778 shell nodes V + 1 cm n, linked to their skin vertex)

Everything here is opt-in; production imports this module only when MODEL.XYZ_AURA is set.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from .xyz_mesh import (ASSOC_CHUNK, L_EDGE, L_REL, R_MAX, Z_REF, Z_SCALE, XYZMeshNet, lbs_slot_pool, ray_triangle,
                       vertex_normals)

#: C1's association constants (model/model.py defaults of the C1 recipe)
BAND_PX, FRONT_PX, Z_TOL, K_CAND = 16.0, 3.0, 0.01, 16
#: mM: 16 px at the fit-set median anchor vertex depth over the fit-set median focal (frozen before any reading;
#: tools/aura/constants.py, docs/EXPERIMENT_RECORDS.md#aura-prereg-20261024)
R_MATCH = 0.03662  # m: median anchor vertex depth 0.5177 m (fit subjects, every 50th endpoint, GT anchors) * 16 px / median f 226.20 px
RULES = ("c1", "surf", "b8", "b32", "mM", "m10", "m20", "aura10", "aura20", "p2d", "p2dvis", "p2dh", "vis")
METRIC_R = {"m10": 0.01, "m20": 0.02, "aura10": 0.01, "aura20": 0.02}
BAND_OF = {"b8": 8.0, "b32": 32.0}
#: z-buffer visibility (a batched copy of tools/s44/evflow.py::visible: 3 x 3 vertex splats, 1 cm tolerance)
IMG_H, IMG_W, VIS_TOL = 180, 240, 0.010
P2D_M = 4
SHELL_R = 0.01
#: MODEL.XYZ_AURA values; `shell` is structural (AuraXYZMeshNet), the others replace the association
AURA_MODES = ("legacy", "none", "shuffle", "surf", "b8", "b32", "mmatch", "m10", "aura10", "p2dh", "vis", "shell")
D_RULE = {"surf": "surf", "b8": "b8", "b32": "b32", "mmatch": "mM", "m10": "m10", "aura10": "aura10",
          "p2dh": "p2dh", "vis": "vis"}


def _empty(B: int, N: int, dev) -> Dict[str, torch.Tensor]:
    return {"vid": torch.zeros(B, N, R_MAX, dtype=torch.long, device=dev),
            "A": torch.zeros(B, N, R_MAX, dtype=torch.float32, device=dev),
            "q": torch.zeros(B, N, dtype=torch.float32, device=dev),
            "z": torch.zeros(B, N, dtype=torch.float32, device=dev),
            "valid": torch.zeros(B, N, dtype=torch.bool, device=dev),
            "hit": torch.zeros(B, N, dtype=torch.bool, device=dev)}


def metric_radius(rule: str) -> float:
    if rule == "mM":
        if R_MATCH is None:
            raise RuntimeError("R_MATCH (the mM radius) is not frozen yet")
        return float(R_MATCH)
    return METRIC_R[rule]


# --------------------------------------------------------------------------------------------------- visibility
@torch.no_grad()
def project(verts: torch.Tensor, intr: Sequence[torch.Tensor]) -> torch.Tensor:
    """(B, V, 2) pixel coordinates of camera-frame vertices; `intr` = (fx, fy, cx, cy), each (B,)."""
    fx, fy, cx, cy = intr
    z = verts[..., 2].clamp(min=1e-6)
    return torch.stack([fx[:, None] * verts[..., 0] / z + cx[:, None], fy[:, None] * verts[..., 1] / z + cy[:, None]],
                       -1)


@torch.no_grad()
def visible(verts: torch.Tensor, intr: Sequence[torch.Tensor]) -> torch.Tensor:
    """(B, V) bool: `tools/s44/evflow.py::visible` per sample (vertex depth within 1 cm of the 3 x 3 vertex-splat
    z-buffer at its rounded pixel, inside the 240 x 180 image, z > 0)."""
    uv = project(verts.float(), intr)
    z = verts[..., 2].float()
    B, V = z.shape
    dev = z.device
    ui, vi = uv[..., 0].round().long(), uv[..., 1].round().long()
    base = torch.arange(B, device=dev).view(B, 1) * (IMG_H * IMG_W)
    zb = torch.full((B * IMG_H * IMG_W,), 1e9, dtype=z.dtype, device=dev)
    for dyy in (-1, 0, 1):
        for dxx in (-1, 0, 1):
            uu, vv = ui + dxx, vi + dyy
            ok = (uu >= 0) & (uu < IMG_W) & (vv >= 0) & (vv < IMG_H) & (z > 0)
            zb.scatter_reduce_(0, (base + vv * IMG_W + uu)[ok], z[ok], reduce="amin", include_self=True)
    inb = (ui >= 0) & (ui < IMG_W) & (vi >= 0) & (vi < IMG_H) & (z > 0)
    zz = zb[base + vi.clamp(0, IMG_H - 1) * IMG_W + ui.clamp(0, IMG_W - 1)]
    return inb & (z <= zz + VIS_TOL)


# --------------------------------------------------------------------------------------------------- association
@torch.no_grad()
def _segment_ray(dd: torch.Tensor, Pa: torch.Tensor, Pb: torch.Tensor):
    """Closest points between the line t * dd (dd (m, 3), d_z = 1) and the segments Pa + s (Pb - Pa), s in [0, 1]
    (Pa, Pb (m, C, 3, 3): three edges of C faces). Returns distance, s, t, all (m, C, 3)."""
    u = dd.view(-1, 1, 1, 3)
    e = Pb - Pa
    a = (u * u).sum(-1)
    b = (u * e).sum(-1)
    c = (e * e).sum(-1)
    p = (u * Pa).sum(-1)
    g = (e * Pa).sum(-1)
    den = a * c - b * b
    s = torch.where(den > 1e-12 * a * c.clamp_min(1e-30), (b * p - a * g) / den.clamp_min(1e-30), torch.zeros_like(b))
    s = s.clamp(0.0, 1.0)
    t = (b * s + p) / a
    gap = t.unsqueeze(-1) * u - (Pa + s.unsqueeze(-1) * e)
    return gap.norm(dim=-1), s, t


@torch.no_grad()
def _core(d, mask, verts, faces, vert_faces, band, front_tol, z_tol, k=K_CAND, chunk=ASSOC_CHUNK, near="layer",
          metric_r=None, cand_vis=None, shell_r=None):
    """`associate_rays(mode="hit_first")` with three opt-in hooks; with near="layer" and no hook it executes the same
    operations in the same order as the library (bitwise test in tests/test_aura.py).

    near: "layer" (C1's front-layer near records), "none" (surface hits only), "shell" (metric shell of radius
    `shell_r` metres, see the module docstring). metric_r (B,) metres: the band test and kernel on the perpendicular
    ray distance. cand_vis (B, V) bool: near candidates restricted to visible vertices (as contour's back-face filter)."""
    B, N, _ = d.shape
    V = verts.shape[1]
    dev = d.device
    out = _empty(B, N, dev)
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
        near_k = d2.topk(kk, dim=-1, largest=False)
        nid = near_k.indices                                                        # (m, kk)
        nang = near_k.values.clamp_min(0.0).sqrt()                                   # ascending
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
        if near == "shell":
            # ---- metric shell: the candidate face nearest to the ray (distance attained on an edge)
            Pa, Pb = P, P[:, :, [1, 2, 0]]
            dist, s, te = _segment_ray(dd, Pa, Pb)                                   # (m, C, 3)
            dist = dist.masked_fill((cf < 0).unsqueeze(-1) | (te <= 1e-4), float("inf"))
            de, ei = dist.min(-1)                                                    # per face: best edge
            se = s.gather(-1, ei.unsqueeze(-1)).squeeze(-1)
            tf = te.gather(-1, ei.unsqueeze(-1)).squeeze(-1)
            delta, fsel2 = de.min(-1)                                                # per event: best face
            e_sel = ei.gather(1, fsel2[:, None]).squeeze(1)
            s_sel = se.gather(1, fsel2[:, None]).squeeze(1)
            t_sel = tf.gather(1, fsel2[:, None]).squeeze(1)
            f2 = cfs.gather(1, fsel2[:, None]).squeeze(1)
            r_ = shell_r[b].float() if torch.is_tensor(shell_r) else torch.full_like(delta, float(shell_r))
            is_near = (~is_hit) & torch.isfinite(delta) & (delta <= r_)
            wv = torch.zeros(m, 3, device=dev, dtype=torch.float32)
            ar = torch.arange(m, device=dev)
            wv[ar, e_sel] = 1.0 - s_sel
            wv[ar, (e_sel + 1) % 3] = s_sel
            A_near = torch.cat([wv, torch.zeros_like(wv[:, :1])], 1)
            fv2 = faces[f2]
            vid_near = torch.cat([fv2, fv2[:, :1]], dim=1)
            q_near = (1.0 - (delta / r_).square()).clamp(min=0.0)
            z_near = t_sel
        else:
            # ---- near: the front depth layer among the in-band angular neighbours
            bnd = band[b].float().unsqueeze(1)
            ftl = front_tol[b].float().unsqueeze(1)
            Vn = Vf[b.view(m, 1), nid]                                               # (m, kk, 3)
            zj = (Vn * dd.unsqueeze(1)).sum(-1) / (dd * dd).sum(-1, keepdim=True)    # closest-point depth
            if cand_vis is not None:
                okc = cand_vis[b.view(m, 1), nid]
                nang_c = nang.masked_fill(~okc, float("inf"))
                ref = nang_c.min(-1, keepdim=True).values
            else:
                nang_c = nang
                ref = nang[:, :1]
            if metric_r is None:
                inband = (nang_c <= bnd) & (zj > 1e-4)
            else:
                pj = (Vn - zj.unsqueeze(-1) * dd.unsqueeze(1)).norm(dim=-1)          # perpendicular distance (m)
                rr = metric_r[b].float().unsqueeze(1)
                inband = (pj <= rr) & (zj > 1e-4) & torch.isfinite(nang_c)
            same = inband & (nang_c <= ref + ftl)
            zfront = torch.where(same, zj, torch.full_like(zj, float("inf"))).min(-1).values
            layer = inband & (zj <= zfront.unsqueeze(1) + float(z_tol))
            rank = torch.where(layer, col.unsqueeze(0), col.unsqueeze(0) + kk)
            cols = rank.argsort(dim=-1)[:, :R_MAX]                                   # first R_MAX in layer
            kept = layer.gather(1, cols)
            if metric_r is None:
                ang = nang.gather(1, cols)
                sig = bnd / 2.0
            else:
                ang = pj.gather(1, cols)
                sig = rr / 2.0
            kw = torch.exp(-0.5 * (ang / sig).square()) * kept
            is_near = (~is_hit) & kept.any(-1)
            if near == "none":
                is_near = torch.zeros_like(is_near)
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


@torch.no_grad()
def _p2d(d, mask, verts, band, vis=None, m_near=P2D_M, chunk=ASSOC_CHUNK):
    """2D proximity: the `m_near` nearest projected vertices (optionally only visible ones) inside `band` (B,) ray
    units, Gaussian weights (sigma = band / 2), q = kernel of the nearest, z = weighted closest-point depth, hit = 0."""
    B, N, _ = d.shape
    V = verts.shape[1]
    dev = d.device
    out = _empty(B, N, dev)
    flat = mask.reshape(-1).nonzero().squeeze(1)
    if flat.numel() == 0:
        return out
    Vf = verts.float()
    zv = Vf[..., 2]
    bad = zv <= 1e-6
    if vis is not None:
        bad = bad | ~vis
    vu = (Vf[..., 0] / zv.clamp_min(1e-6)).masked_fill(bad, 1e6)
    vv = (Vf[..., 1] / zv.clamp_min(1e-6)).masked_fill(bad, 1e6)
    kk = max(1, min(int(m_near), V))
    parts = {key: [] for key in out}
    for i0 in range(0, int(flat.numel()), int(chunk)):
        rows = flat[i0:i0 + chunk]
        b = rows // N
        dd = d.reshape(-1, 3)[rows].float()
        m = rows.numel()
        d2 = (vu[b] - dd[:, 0:1]).square() + (vv[b] - dd[:, 1:2]).square()
        near_k = d2.topk(kk, dim=-1, largest=False)
        nid = near_k.indices
        nang = near_k.values.clamp_min(0.0).sqrt()
        bnd = band[b].float().unsqueeze(1)
        Vn = Vf[b.view(m, 1), nid]
        zj = (Vn * dd.unsqueeze(1)).sum(-1) / (dd * dd).sum(-1, keepdim=True)
        inb = (nang <= bnd) & (zj > 1e-4)
        sig = bnd / 2.0
        kw = torch.exp(-0.5 * (nang / sig).square()) * inb
        A = kw / kw.sum(-1, keepdim=True).clamp_min(1e-12)
        ok = inb.any(-1)
        q = torch.exp(-0.5 * (nang[:, 0] / sig[:, 0]).square()) * inb[:, 0]
        q = torch.where(ok & ~inb[:, 0], (kw.max(-1).values), q)                    # nearest out of band, others in
        z = (A * zj).sum(-1)
        if kk < R_MAX:
            nid = torch.nn.functional.pad(nid, (0, R_MAX - kk))
            A = torch.nn.functional.pad(A, (0, R_MAX - kk))
        parts["vid"].append(nid[:, :R_MAX] * ok[:, None])
        parts["A"].append(A[:, :R_MAX] * ok[:, None])
        parts["q"].append(q * ok)
        parts["z"].append(z * ok)
        parts["valid"].append(ok)
        parts["hit"].append(torch.zeros_like(ok))
    for key, dst in out.items():
        dst.view(-1, *dst.shape[2:])[flat] = torch.cat(parts[key]).to(dst.dtype)
    return out


@torch.no_grad()
def associate(rule: str, d, mask, verts, faces, vert_faces, intr, *, c1=None, chunk=ASSOC_CHUNK,
              vis: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """One association rule (`RULES`) on the anchor mesh. d (B, N, 3) rays with d_z = 1; mask (B, N); verts (B, V, 3)
    metres, camera frame; faces (F, 3); vert_faces (V, M); intr = (fx, fy, cx, cy) render-frame intrinsics, (B,) each.
    `c1`: the C1 association of the same inputs (p2dh reads its hit bit; computed when absent). `vis` (B, V): the
    z-buffer visibility (computed when absent and needed). Returns the `associate_rays` dict (R = 4 records)."""
    if rule not in RULES:
        raise ValueError(f"unknown association rule {rule!r}; one of {RULES}")
    fx, fy = intr[0].float(), intr[1].float()
    f_px = (fx * fy).sqrt()
    band = BAND_PX / f_px
    front = FRONT_PX / f_px
    B = d.shape[0]
    if rule in ("p2dvis", "p2dh", "vis") and vis is None:
        vis = visible(verts, intr)
    if rule == "c1":
        return _core(d, mask, verts, faces, vert_faces, band, front, Z_TOL, chunk=chunk)
    if rule == "surf":
        return _core(d, mask, verts, faces, vert_faces, band, front, Z_TOL, chunk=chunk, near="none")
    if rule in BAND_OF:
        return _core(d, mask, verts, faces, vert_faces, BAND_OF[rule] / f_px, front, Z_TOL, chunk=chunk)
    if rule in ("mM", "m10", "m20"):
        r = torch.full((B,), metric_radius(rule), device=d.device)
        return _core(d, mask, verts, faces, vert_faces, band, front, Z_TOL, chunk=chunk, metric_r=r)
    if rule in ("aura10", "aura20"):
        return _core(d, mask, verts, faces, vert_faces, band, front, Z_TOL, chunk=chunk, near="shell",
                     shell_r=metric_radius(rule))
    if rule == "vis":
        return _core(d, mask, verts, faces, vert_faces, band, front, Z_TOL, chunk=chunk, cand_vis=vis)
    if rule == "p2d":
        return _p2d(d, mask, verts, band, chunk=chunk)
    out = _p2d(d, mask, verts, band, vis=vis, chunk=chunk)                          # p2dvis / p2dh
    if rule == "p2dh":
        if c1 is None:
            c1 = _core(d, mask, verts, faces, vert_faces, band, front, Z_TOL, chunk=chunk)
        out["hit"] = c1["hit"] & out["valid"]
    return out


def none_assoc(assoc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Every event to the residual slot (q = 0, valid = hit = 0)."""
    return {k: torch.zeros_like(v) for k, v in assoc.items()}


# --------------------------------------------------------------------------------------------------- binding nulls
def _hash31(x: torch.Tensor) -> torch.Tensor:
    """A deterministic 31-bit integer mix (int64 in, int64 in [0, 2^31) out; no random state)."""
    x = x & 0x7FFFFFFF
    x = x ^ (x >> 16)
    x = (x * 0x45D9F3B) & 0x7FFFFFFF
    x = x ^ (x >> 16)
    x = (x * 0x45D9F3B) & 0x7FFFFFFF
    return x ^ (x >> 16)


def packet_seeds(sequence_id: torch.Tensor, t_end_us: torch.Tensor) -> torch.Tensor:
    """(B,) int64 seeds of the D shuffle from the packet identity (no random state)."""
    return _hash31(sequence_id.long() * 1000003 + torch.div(t_end_us.long(), 1000, rounding_mode="floor"))


@torch.no_grad()
def shuffle_assoc(assoc, d, mask, verts, seeds, part_of_vertex: Optional[torch.Tensor] = None):
    """Deterministic derangement of the association among the associated events of each packet (mask & valid & q > 0):
    events are ordered by a hash of (packet seed, event index) within their group (the packet, or the packet and the
    hard LBS part of the event's main vertex when `part_of_vertex` (V,) is given) and every event takes the record set
    (vid, A, q, hit) of the next one in that order (cyclically); z is recomputed as the A-weighted closest-point depth
    of the event's own ray at the borrowed vertices. The multiset of (vid, q A) records per packet, hence the support
    and the per-vertex masses, is unchanged. Groups of one keep their own records. Returns (assoc, info) with the
    partner index (B, N) (-1 where not eligible)."""
    B, N = mask.shape
    dev = mask.device
    elig = mask & assoc["valid"] & (assoc["q"] > 0)
    idx = torch.arange(N, device=dev).view(1, N).expand(B, N)
    h = _hash31(seeds.long().view(B, 1) * 131071 + idx)                              # (B, N) in [0, 2^31)
    if part_of_vertex is not None:
        w = assoc["q"].unsqueeze(-1) * assoc["A"]
        main = assoc["vid"].gather(2, w.argmax(-1, keepdim=True)).squeeze(-1)        # (B, N)
        grp = part_of_vertex.long()[main]
    else:
        grp = torch.zeros(B, N, dtype=torch.long, device=dev)
    bb = torch.arange(B, device=dev).view(B, 1).expand(B, N)
    big = torch.tensor(1 << 31, device=dev, dtype=torch.long)
    gkey = torch.where(elig, bb * 64 + grp, torch.full_like(grp, -1))                # group id, -1 = not eligible
    sort_key = torch.where(elig, (bb * 64 + grp) * big + h, torch.full_like(h, (B * 64 + 1)) * big + bb * N + idx)
    order = torch.argsort(sort_key.reshape(-1), stable=True)
    g_sorted = gkey.reshape(-1)[order]
    n_el = int(elig.sum())
    partner_flat = torch.full((B * N,), -1, dtype=torch.long, device=dev)
    if n_el:
        gs = g_sorted[:n_el]                                                         # eligible come first
        pos = torch.arange(n_el, device=dev)
        new = torch.ones(n_el, dtype=torch.bool, device=dev)
        new[1:] = gs[1:] != gs[:-1]
        start = torch.cummax(torch.where(new, pos, torch.zeros_like(pos)), 0).values
        last = torch.ones(n_el, dtype=torch.bool, device=dev)
        last[:-1] = new[1:]
        end = torch.flip(torch.cummin(torch.flip(torch.where(last, pos, torch.full_like(pos, n_el)), [0]), 0).values,
                         [0])
        n = end - start + 1
        nxt = start + torch.remainder(pos - start + 1, n)
        src = order[:n_el]
        partner_flat[src] = torch.remainder(order[nxt], N)                          # partners share the packet
    partner = partner_flat.view(B, N)
    take = partner >= 0
    pidx = partner.clamp_min(0)
    out = {}
    for key in ("vid", "A"):
        g = assoc[key].gather(1, pidx.unsqueeze(-1).expand(-1, -1, assoc[key].shape[-1]))
        out[key] = torch.where(take.unsqueeze(-1), g, assoc[key])
    for key in ("q", "valid", "hit"):
        g = assoc[key].gather(1, pidx)
        out[key] = torch.where(take, g, assoc[key])
    Vr = torch.gather(verts.float().unsqueeze(1).expand(-1, N, -1, -1), 2,
                      out["vid"].unsqueeze(-1).expand(-1, -1, -1, 3))                # (B, N, R, 3)
    dd = d.float()
    zj = (Vr * dd.unsqueeze(2)).sum(-1) / (dd * dd).sum(-1, keepdim=True)
    An = out["A"] / out["A"].sum(-1, keepdim=True).clamp_min(1e-12)
    z_new = (An * zj).sum(-1)
    out["z"] = torch.where(take, z_new, assoc["z"])
    return out, {"partner": partner, "eligible": elig}


@torch.no_grad()
def shuffle_locality(assoc, info, d, intr, part_of_vertex, finger_of_part) -> Dict[str, torch.Tensor]:
    """Per eligible event: image distance (px) between the event and its partner, and whether the two main vertices
    share the hard part / the finger group."""
    fx, fy, cx, cy = intr
    partner, elig = info["partner"], info["eligible"]
    B, N = partner.shape
    px = torch.stack([d[..., 0] * fx[:, None] + cx[:, None], d[..., 1] * fy[:, None] + cy[:, None]], -1)
    pp = px.gather(1, partner.clamp_min(0).unsqueeze(-1).expand(-1, -1, 2))
    dist = (px - pp).norm(dim=-1)
    w = assoc["q"].unsqueeze(-1) * assoc["A"]
    main = assoc["vid"].gather(2, w.argmax(-1, keepdim=True)).squeeze(-1)
    pmain = main.gather(1, partner.clamp_min(0))
    part, ppart = part_of_vertex[main], part_of_vertex[pmain]
    return {"dist_px": dist[elig], "same_part": (part == ppart)[elig],
            "same_finger": (finger_of_part[part] == finger_of_part[ppart])[elig],
            "self": (partner == torch.arange(N, device=d.device).view(1, N))[elig]}


# --------------------------------------------------------------------------------------------------- probe helpers
@torch.no_grad()
def occupancy(assoc, mask, V: int):
    """(B, V) masses of the surface (hit) and near records: sum of q A per vertex."""
    B, N, R = assoc["vid"].shape
    w = assoc["q"].unsqueeze(-1) * assoc["A"] * mask.float().unsqueeze(-1)
    hit = assoc["hit"].unsqueeze(-1).expand(-1, -1, R)
    key = (torch.arange(B, device=w.device).view(B, 1, 1) * V + assoc["vid"]).reshape(-1)
    wf = w.reshape(-1).float()
    hf = hit.reshape(-1)
    m_hit = torch.zeros(B * V, device=w.device).index_add(0, key, wf * hf.float())
    m_near = torch.zeros(B * V, device=w.device).index_add(0, key, wf * (~hf).float())
    return m_hit.view(B, V), m_near.view(B, V)


def joint_weights(lbs_w: torch.Tensor) -> torch.Tensor:
    """(16, V) LBS weights normalised per joint (the skinning-weighted surface centroid of each joint)."""
    w = lbs_w.float().t()
    return w / w.sum(1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def joint_assoc(d, mask, joints) -> Dict[str, torch.Tensor]:
    """The coarse 2D baseline: every live event to its nearest projected joint node (no band), q = 1, z = the
    closest-point depth of the ray at that joint. vid indexes the 16 joint nodes (record 0; records 1-3 weigh 0)."""
    B, N, _ = d.shape
    J = joints.float()
    zj = J[..., 2].clamp_min(1e-6)
    ju, jv = J[..., 0] / zj, J[..., 1] / zj
    d2 = (ju.unsqueeze(1) - d[..., 0:1]).square() + (jv.unsqueeze(1) - d[..., 1:2]).square()   # (B, N, 16)
    j = d2.argmin(-1)
    Jsel = J.gather(1, j.unsqueeze(-1).expand(-1, -1, 3))
    dd = d.float()
    z = (Jsel * dd).sum(-1) / (dd * dd).sum(-1)
    out = _empty(B, N, d.device)
    live = mask & (z > 1e-4)
    out["vid"][..., 0] = j * live
    out["A"][..., 0] = live.float()
    out["q"] = live.float()
    out["z"] = z * live
    out["valid"] = live
    return out


def shell_tables(ring_idx: torch.Tensor, ring_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Neighbour tables of the 2V-node skin + shell graph: skin j -> [ring(j), V + j]; shell V + j -> [V + ring(j), j]."""
    V, K = ring_idx.shape
    ar = torch.arange(V, device=ring_idx.device).unsqueeze(1)
    skin = torch.cat([ring_idx, ar + V], 1)
    shell = torch.cat([ring_idx + V, ar], 1)
    m = torch.cat([ring_mask, torch.ones(V, 1, device=ring_mask.device, dtype=ring_mask.dtype)], 1)
    return torch.cat([skin, shell], 0), torch.cat([m, m], 0)


@torch.no_grad()
def shell_positions(verts: torch.Tensor, faces: torch.Tensor, r: float = SHELL_R) -> torch.Tensor:
    """(B, 2V, 3): the skin vertices, then the shell nodes V + r n (outward area-weighted vertex normals)."""
    n = vertex_normals(verts.float(), faces)
    return torch.cat([verts.float(), verts.float() + r * n], 1)


@torch.no_grad()
def node_edge_geometry(pos: torch.Tensor, idx: torch.Tensor, emask: torch.Tensor) -> torch.Tensor:
    """`XYZMeshNet.edge_geometry` on an arbitrary node table: (B, V', K', 4) = [(P_l - P_j), |P_l - P_j|] / L_EDGE."""
    B, Vn, _ = pos.shape
    K = idx.shape[1]
    dv = pos[:, idx.reshape(-1)].view(B, Vn, K, 3) - pos.unsqueeze(2)
    g = torch.cat([dv, dv.norm(dim=-1, keepdim=True)], dim=-1) / L_EDGE
    return g * emask.view(1, Vn, K, 1)


# --------------------------------------------------------------------------------------------------- D: MODEL.XYZ_AURA
@torch.no_grad()
def apply_aura(mode: str, assoc, d, mask, verts, faces, vert_faces, intr, batch, lbs_hard, k: int = K_CAND):
    """The association hook of `MODEL.XYZ_AURA` (model/model.py `_xyz_mesh_forward`, right after `associate_rays`):
    returns the association the C1 network reads. `shell` and `legacy` leave it unchanged (shell is structural)."""
    if mode in ("legacy", "shell"):
        return assoc
    if mode == "none":
        return none_assoc(assoc)
    if mode == "shuffle":
        seeds = packet_seeds(batch.sequence_id.to(d.device), batch.t_end_us.to(d.device))
        return shuffle_assoc(assoc, d, mask, verts, seeds)[0]
    if mode not in D_RULE:
        raise ValueError(f"MODEL.XYZ_AURA must be one of {AURA_MODES}, got {mode!r}")
    if int(k) != K_CAND:
        raise ValueError("MODEL.XYZ_AURA variants are defined on C1's 16 candidates")
    return associate(D_RULE[mode], d, mask, verts, faces, vert_faces, intr, c1=assoc)


class AuraXYZMeshNet(XYZMeshNet):
    """`MODEL.XYZ_AURA: shell`: C1 with literal shell nodes. The association is C1's; surface (hit) records stay on the
    778 skin vertices and near records go to 778 shell nodes at V + 1 cm n (their dXYZ is relative to the shell node);
    the MeshGNN runs on the 2V-node graph (skin 1-ring, shell 1-ring, skin-shell edge), shell nodes use their skin
    vertex's LBS weights. No new parameters: every initial value and the data stream equal C1's."""

    def __init__(self, lbs_weights: torch.Tensor, faces: torch.Tensor, *args, aura: str = "shell", **kwargs):
        super().__init__(lbs_weights, faces, *args, **kwargs)
        if aura != "shell":
            raise ValueError(f"AuraXYZMeshNet builds the structural variant `shell` only, got {aura!r}")
        if (self.contour or not self.geom or self.residual_mode != "legacy" or self.xag_gnn != "legacy"
                or self.sign):
            raise ValueError("XYZ_AURA is defined on the C1 recipe only (hit_first, geometry, legacy residual / GNN)")
        self.aura = aura
        idx2, m2 = shell_tables(self.ring_idx, self.ring_mask)
        self.register_buffer("shell_idx", idx2, persistent=False)
        self.register_buffer("shell_mask", m2, persistent=False)
        self.register_buffer("lbs_w2", torch.cat([self.lbs_w, self.lbs_w], 0), persistent=False)
        self.register_buffer("lbs_hard2", torch.cat([self.lbs_hard, self.lbs_hard], 0), persistent=False)
        self.register_buffer("shell_faces", faces.detach().long().clone(), persistent=False)

    def forward(self, tok, d, mask, verts, assoc, prev48, ablate: bool = False, verts_geo=None,
                capture: bool = False, h3_transform=None, verts_gt=None):
        if verts_geo is not None or verts_gt is not None or h3_transform is not None:
            raise ValueError("XYZ_AURA shell does not take the oracle / probe inputs")
        faces = self.shell_faces
        B, N, _ = tok.shape
        V = verts.shape[1]
        V2 = 2 * V
        C = self.hidden
        dev = tok.device
        wdt = self.rel[0].weight.dtype
        acc = torch.float32 if self.training else torch.float64
        mask_f = mask.to(torch.float32)
        w = assoc["q"].unsqueeze(-1) * assoc["A"] * mask_f.unsqueeze(-1)                # (B, N, R)
        r = (1.0 - assoc["q"]) * mask_f
        if ablate:
            w = torch.zeros_like(w)
            r = torch.zeros_like(r)
        pos = shell_positions(verts, faces)                                             # (B, 2V, 3)
        rb, rn, rr = (w > 0).nonzero(as_tuple=True)
        vrec = assoc["vid"][rb, rn, rr]
        hrec = assoc["hit"][rb, rn]
        node = vrec + V * (~hrec).long()
        wrec = w[rb, rn, rr]
        drec = d[rb, rn].float()
        zrec = assoc["z"][rb, rn]
        X = zrec.unsqueeze(-1) * drec
        dxyz = (X - pos[rb, node]) / L_REL
        zn = ((zrec - Z_REF) / Z_SCALE).unsqueeze(-1)
        feat = torch.cat([tok[rb, rn].float(), dxyz, zn, torch.ones_like(zn), hrec.float().unsqueeze(-1)], dim=-1)
        phi = self.rel(feat.to(wdt))
        key = rb * V2 + node
        mass = torch.zeros(B * V2, device=dev, dtype=acc).index_add(0, key, wrec.to(acc))
        num = torch.zeros(B * V2, C, device=dev, dtype=acc).index_add(0, key, wrec.to(acc).unsqueeze(-1) * phi.to(acc))
        has = mass > 0
        mean = (num / mass.clamp_min(1e-12).unsqueeze(-1)).to(torch.float32) * has.unsqueeze(-1)
        peak = torch.full((B * V2, C), -1e4, device=dev, dtype=torch.float32).scatter_reduce(
            0, key.unsqueeze(-1).expand(-1, C), phi.float(), reduce="amax")
        peak = peak * has.unsqueeze(-1)
        x = torch.cat([mean, peak, torch.log1p(mass.to(torch.float32)).unsqueeze(-1)], -1)
        O = has.view(B, V2).to(torch.float32)
        H = self.vin(x.view(B, V2, 2 * C + 1).to(wdt))
        keep_capture = capture or self.xag_capture_enabled
        H0 = H if keep_capture else None
        g = node_edge_geometry(pos, self.shell_idx, self.shell_mask)
        gemb = torch.relu(self.edge(g.to(wdt)))
        S = O
        for layer in self.layers:
            H, S = layer(H, S, gemb, self.shell_idx, self.shell_mask)
        H3 = H if keep_capture else None
        e = lbs_slot_pool(H, S, O, self.lbs_w2, self.lbs_hard2)                          # (B, 16, 2C+2)
        eb, en = (r > 0).nonzero(as_tuple=True)
        tr = tok[eb, en].float()
        feat_r = torch.cat([tr, torch.zeros(tr.shape[0], self.rel_in - tr.shape[1], device=dev)], dim=-1)
        phi_r = self.rel(feat_r.to(wdt))
        rw = r[eb, en]
        rs = torch.zeros(B, device=dev, dtype=acc).index_add(0, eb, rw.to(acc))
        rnum = torch.zeros(B, C, device=dev, dtype=acc).index_add(0, eb, rw.to(acc).unsqueeze(-1) * phi_r.to(acc))
        rhas = (rs > 0).unsqueeze(-1)
        rmean = (rnum / rs.clamp_min(1e-12).unsqueeze(-1)).to(torch.float32) * rhas
        rpeak = torch.full((B, C), -1e4, device=dev, dtype=torch.float32).scatter_reduce(
            0, eb.unsqueeze(-1).expand(-1, C), phi_r.float(), reduce="amax") * rhas
        live = mask_f.sum(1).clamp_min(1.0)
        bslot = torch.cat([rmean, rpeak, (rs.to(torch.float32) / live).unsqueeze(-1), rhas.to(torch.float32)], dim=-1)
        slots = torch.cat([e.to(torch.float32), bslot.unsqueeze(1)], dim=1)
        zs = torch.relu(self.slot_proj(slots.reshape(B, -1, 1).to(wdt))).view(B, -1)
        h = torch.relu(self.fc1(torch.cat([zs, prev48.to(zs.dtype)], dim=-1)))
        delta = self.fc2(h)
        n_live = mask_f.sum()
        stats = {"xyz_valid": assoc["valid"].float().sum() / n_live.clamp_min(1.0),
                 "xyz_hit": assoc["hit"].float().sum() / n_live.clamp_min(1.0),
                 "xyz_resid_mass": r.sum() / n_live.clamp_min(1.0),
                 "xyz_verts_direct": O.sum(1).mean(), "xyz_verts_prop": S.sum(1).mean()}
        aux = {"slots": slots, "O": O, "S": S, "w": w, "r": r}
        if keep_capture:
            self.xag_last_capture = {k_: v_.detach().clone() for k_, v_ in {
                "H0": H0, "H3": H3, "O": O, "S": S, "slots": slots, "zs": zs, "bslot": bslot, "prev48": prev48}.items()}
            aux.update({k_: v_ for k_, v_ in self.xag_last_capture.items() if k_ not in aux})
        return delta, stats, aux


__all__ = ["AURA_MODES", "AuraXYZMeshNet", "BAND_PX", "D_RULE", "FRONT_PX", "K_CAND", "METRIC_R", "R_MATCH", "RULES",
           "SHELL_R", "Z_TOL", "apply_aura", "associate", "joint_assoc", "joint_weights", "node_edge_geometry",
           "none_assoc", "occupancy", "packet_seeds", "project", "shell_positions", "shell_tables", "shuffle_assoc",
           "shuffle_locality", "visible"]
