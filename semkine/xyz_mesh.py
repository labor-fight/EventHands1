#!/usr/bin/env python3
r"""S37-XYZ (2026-09-29, the user's drawing): the previous state's full MANO mesh, in camera XYZ, is
the carrier; events meet it along their camera rays; three learned modules read it.

    prev 51D --MANO FK--> V (778 x 3, camera frame, metres), fixed faces
    events (u, v, t, p) + K --> rays d = K^-1 [u, v, 1] (d_z = 1, so a ray parameter is a depth)
    ray x mesh association, no parameters:
        surface  the nearest positive ray/triangle hit; barycentric weights over the hit face, q = 1
        near     no hit, but vertices within the band (angle): the front depth layer among them, at
                 most four, angular-kernel weights, q = kernel of the nearest one (< 1)
        none     valid = 0, q = 0: the whole event goes to the residual slot
    (1) shared relation encoder phi(token7, dXYZ, z_prior, valid, hit) on every (event, vertex)
        record *before* pooling; residual records use the same phi with valid = 0, zero geometry
    vertex aggregation: mean / max / mass at every vertex; unobserved vertices stay in the graph
    (2) one MeshGNN on the fixed face 1-ring, messages gated by the 3D edge vectors, direct (O) and
        propagated (S) support kept apart; all-zero features stay zero
    LBS pooling into 16 ordered joint slots; the residual summary b is slot 17
    (3) one common readout: 17 slots + prev[3:51] -> one 51D delta; x_k = prev + delta

Every event's unit mass is split between vertices (w_ij = q_i A_ij) and the residual slot
(r_i = 1 - q_i), so nothing an event says is dropped by a finite association band.

`geom=False` is the matched control (C0): the same association, records, weights, widths and
parameter count, with the explicit 3D channels (dXYZ, z_prior, edge vectors) zero at the inputs of
the learned modules.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional, Tuple

import torch
from torch import nn

#: relation-encoder input: token7 + dXYZ (3) + z_prior (1) + valid (1) + hit (1)
REL_IN = 7 + 3 + 1 + 1 + 1
#: vertices per event record set (3 for a surface hit, <= 4 for a near-silhouette event)
R_MAX = 4
#: metres per unit of the event-to-vertex offset; surface offsets are ~2-5 mm, near ones ~1-3 cm
L_REL = 0.01
#: z_prior enters as (z - Z_REF) / Z_SCALE
Z_REF, Z_SCALE = 0.45, 0.1
#: metres per unit of a mesh edge vector (MANO edges are ~3-8 mm)
L_EDGE = 0.005
#: rows of the (events x 778) angular block and of the candidate-face block per chunk
ASSOC_CHUNK = 32768
#: barycentric slack of the ray/triangle test (float32; covers the shared-edge crack)
BARY_EPS = 1e-5
N_JOINTS = 16


# region agent log
_AGENT_LOG = "/data1/lyq/code/mesh/EventHands1/.cursor/debug-466f12.log"


def agent_log(hyp: str, loc: str, msg: str, data: dict, run_id: Optional[str] = None) -> None:
    try:
        with open(_AGENT_LOG, "a") as f:
            f.write(json.dumps({"sessionId": "466f12", "runId": run_id or os.environ.get("AGENT_RUN_ID", "adhoc"),
                                "hypothesisId": hyp, "location": loc, "message": msg, "data": data,
                                "timestamp": int(time.time() * 1000)}, default=float) + "\n")
    except Exception:
        pass
# endregion


# --------------------------------------------------------------------------- fixed tables
def stride_sample(ptr: torch.Tensor, max_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """`EventGNN._sample`'s uniform-stride subsample (<= `max_nodes` per packet), with the padded
    width trimmed to the largest packet: a packet's positions depend only on its own count, so the
    trim changes nothing but the cost."""
    dev = ptr.device
    counts = (ptr[1:] - ptr[:-1]).clamp(min=0)
    n = counts.clamp(max=int(max_nodes))
    N = max(int(n.max()) if n.numel() else 0, 1)
    pos = torch.arange(N, device=dev)
    mask = pos.unsqueeze(0) < n.unsqueeze(1)
    step = counts.to(torch.float32) / n.clamp(min=1).to(torch.float32)
    src = ptr[:-1].unsqueeze(1) + (pos.unsqueeze(0).to(torch.float32) * step.unsqueeze(1)).long()
    src = torch.minimum(src, (ptr[1:] - 1).clamp(min=0).unsqueeze(1))
    return src.clamp(min=0), mask


def vertex_face_table(faces: torch.Tensor, n_verts: int) -> torch.Tensor:
    """`(V, M)` long: the faces incident to every vertex, -1 padded."""
    f = faces.long().reshape(-1).cpu()
    fid = torch.arange(faces.shape[0]).repeat_interleave(3)
    order = torch.argsort(f, stable=True)
    fs, fids = f[order], fid[order]
    deg = torch.bincount(f, minlength=n_verts)
    first = torch.cat([torch.zeros(1, dtype=torch.long), deg.cumsum(0)[:-1]])
    slot = torch.arange(f.numel()) - first[fs]
    table = torch.full((n_verts, int(deg.max())), -1, dtype=torch.long)
    table[fs, slot] = fids
    return table


def mesh_ring(faces: torch.Tensor, n_verts: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Face 1-ring of every vertex: `(idx (V, K) long, self where padded; emask (V, K) float)`."""
    f = faces.long().cpu()
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = torch.unique(torch.cat([e, e.flip(1)]), dim=0)
    e = e[e[:, 0] != e[:, 1]]
    deg = torch.bincount(e[:, 0], minlength=n_verts)
    K = int(deg.max())
    idx = torch.arange(n_verts).unsqueeze(1).repeat(1, K)
    emask = torch.zeros(n_verts, K)
    first = torch.cat([torch.zeros(1, dtype=torch.long), deg.cumsum(0)[:-1]])
    slot = torch.arange(e.shape[0]) - first[e[:, 0]]
    idx[e[:, 0], slot] = e[:, 1]
    emask[e[:, 0], slot] = 1.0
    return idx, emask


# --------------------------------------------------------------------------- association
@torch.no_grad()
def ray_triangle(d: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor,
                 eps: float = BARY_EPS) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Moeller-Trumbore for rays from the camera origin. `d (..., 3)`, `p* (..., 3)` broadcast.
    Returns `(t, u, v)`, `t = inf` where the ray misses; with `d_z = 1`, `t` is the hit depth."""
    e1, e2 = p1 - p0, p2 - p0
    shape = torch.broadcast_shapes(d.shape, e1.shape)
    d, e1, e2, p0 = d.expand(shape), e1.expand(shape), e2.expand(shape), p0.expand(shape)
    pvec = torch.cross(d, e2, dim=-1)
    det = (e1 * pvec).sum(-1)
    ok = det.abs() > 1e-14
    inv = 1.0 / torch.where(ok, det, torch.ones_like(det))
    tvec = -p0
    u = (tvec * pvec).sum(-1) * inv
    qvec = torch.cross(tvec, e1, dim=-1)
    v = (d * qvec).sum(-1) * inv
    t = (e2 * qvec).sum(-1) * inv
    good = ok & (u >= -eps) & (v >= -eps) & (u + v <= 1.0 + eps) & (t > 1e-4)
    return torch.where(good, t, torch.full_like(t, float("inf"))), u, v


@torch.no_grad()
def associate_rays(d: torch.Tensor, mask: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor,
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


# --------------------------------------------------------------------------- learned modules
class XYZMeshLayer(nn.Module):
    r"""One message round on the fixed 1-ring, gated by the 3D edge:

        m_{l->j} = S_l sigmoid(W_g gemb_jl + c) * W_m H_l
        S'_j     = S_j OR any(S_l : l in ring(j))
        H'_j     = H_j + S'_j relu(W_s H_j + mean_{supported l} m_{l->j})

    `W_m`, `W_s` have no bias, so an all-zero feature field stays zero: the geometry modulates what
    the events put on the mesh and cannot light up an unobserved region on its own."""

    def __init__(self, hidden: int, edge_dim: int):
        super().__init__()
        self.msg = nn.Linear(hidden, hidden, bias=False)
        self.slf = nn.Linear(hidden, hidden, bias=False)
        self.gate = nn.Linear(edge_dim, hidden)

    def forward(self, H, S, gemb, idx, emask):
        B, V, C = H.shape
        K = idx.shape[1]
        flat = idx.reshape(-1)
        Mn = self.msg(H)[:, flat].view(B, V, K, C)
        Sn = S[:, flat].view(B, V, K) * emask.unsqueeze(0).to(S.dtype)
        gate = torch.sigmoid(self.gate(gemb))
        w = Sn.unsqueeze(-1).to(Mn.dtype)
        agg = (gate * Mn * w).sum(2) / Sn.sum(2, keepdim=True).clamp_min(1.0).to(Mn.dtype)
        S_next = torch.maximum(S, Sn.amax(2))
        H = H + S_next.unsqueeze(-1).to(H.dtype) * torch.relu(self.slf(H) + agg)
        return H, S_next


def lbs_slot_pool(H, S, O, W, hard):
    """16 ordered joint slots `(B, 16, 2C+2)` = [LBS-weighted mean over propagated support,
    hard-part max over propagated support, direct-support share, propagated-support share]."""
    B, V, C = H.shape
    J = W.shape[1]
    Wd = W.to(H.dtype)
    Sd = S.to(H.dtype)
    a = Wd.unsqueeze(0) * Sd.unsqueeze(-1)                                              # (B, V, J)
    wsum = a.sum(1)
    mean = torch.einsum("bvj,bvc->bjc", a, H) / wsum.clamp_min(1e-6).unsqueeze(-1)
    mean = mean * (wsum > 0).unsqueeze(-1).to(H.dtype)
    sup = S > 0
    src = H.masked_fill(~sup.unsqueeze(-1), -1e4)
    peak = torch.full((B, J, C), -1e4, device=H.device, dtype=H.dtype)
    peak = peak.scatter_reduce(1, hard.view(1, V, 1).expand(B, V, C), src, reduce="amax")
    count = torch.zeros(B, J, device=H.device, dtype=torch.long)
    count.scatter_add_(1, hard.view(1, V).expand(B, V), sup.long())
    peak = peak * (count > 0).unsqueeze(-1).to(H.dtype)
    wtot = W.float().sum(0).clamp_min(1e-6)
    cov = (W.float().unsqueeze(0) * O.float().unsqueeze(-1)).sum(1) / wtot
    spt = (W.float().unsqueeze(0) * S.float().unsqueeze(-1)).sum(1) / wtot
    return torch.cat([mean, peak, cov.unsqueeze(-1).to(H.dtype), spt.unsqueeze(-1).to(H.dtype)], -1)


class XYZMeshNet(nn.Module):
    """The three learned modules of the drawing and the deterministic pooling between them."""

    N_SLOTS = N_JOINTS + 1

    def __init__(self, lbs_weights: torch.Tensor, faces: torch.Tensor, hidden: int = 128,
                 n_layers: int = 3, rel_hidden: int = 64, slot_dim: int = 32,
                 readout_hidden: int = 256, edge_dim: int = 32, geom: bool = True):
        super().__init__()
        V = int(lbs_weights.shape[0])
        self.hidden = int(hidden)
        self.geom = bool(geom)
        self.slot_in = 2 * self.hidden + 2
        # (1) shared relation encoder
        self.rel = nn.Sequential(nn.Linear(REL_IN, rel_hidden), nn.ReLU(inplace=True),
                                 nn.Linear(rel_hidden, self.hidden))
        # (2) one MeshGNN: vertex input projection (no bias), shared edge embedding, L rounds
        self.vin = nn.Linear(2 * self.hidden + 1, self.hidden, bias=False)
        self.edge = nn.Linear(4, edge_dim)
        self.layers = nn.ModuleList([XYZMeshLayer(self.hidden, edge_dim) for _ in range(int(n_layers))])
        # (3) one common readout: slot-specific compression (grouped 1x1 conv), then a joint MLP
        self.slot_proj = nn.Conv1d(self.N_SLOTS * self.slot_in, self.N_SLOTS * slot_dim, 1,
                                   groups=self.N_SLOTS)
        self.fc1 = nn.Linear(self.N_SLOTS * slot_dim + 48, readout_hidden)
        self.fc2 = nn.Linear(readout_hidden, 51)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        idx, emask = mesh_ring(faces, V)
        self.register_buffer("ring_idx", idx, persistent=False)
        self.register_buffer("ring_mask", emask, persistent=False)
        self.register_buffer("vert_faces", vertex_face_table(faces, V), persistent=False)
        self.register_buffer("lbs_w", lbs_weights.detach().float().clone(), persistent=False)
        self.register_buffer("lbs_hard", lbs_weights.detach().argmax(1), persistent=False)

    @torch.no_grad()
    def edge_geometry(self, verts: torch.Tensor) -> torch.Tensor:
        """`(B, V, K, 4)` = [(V_l - V_j) / L_EDGE, |V_l - V_j| / L_EDGE] on the 1-ring, camera frame."""
        B, V, _ = verts.shape
        K = self.ring_idx.shape[1]
        vf = verts.float()
        dv = vf[:, self.ring_idx.reshape(-1)].view(B, V, K, 3) - vf.unsqueeze(2)
        g = torch.cat([dv, dv.norm(dim=-1, keepdim=True)], dim=-1) / L_EDGE
        return g * self.ring_mask.view(1, V, K, 1)

    def forward(self, tok, d, mask, verts, assoc, prev48, ablate: bool = False):
        B, N, _ = tok.shape
        V = verts.shape[1]
        C = self.hidden
        dev = tok.device
        wdt = self.rel[0].weight.dtype
        acc = torch.float32 if self.training else torch.float64
        mask_f = mask.to(torch.float32)
        w = assoc["q"].unsqueeze(-1) * assoc["A"] * mask_f.unsqueeze(-1)                # (B, N, R)
        r = (1.0 - assoc["q"]) * mask_f                                                 # (B, N)
        if ablate:
            w = torch.zeros_like(w)
            r = torch.zeros_like(r)
        # ---- matched records through phi, then onto the vertices
        rb, rn, rr = (w > 0).nonzero(as_tuple=True)
        vrec = assoc["vid"][rb, rn, rr]
        wrec = w[rb, rn, rr]
        zrec = assoc["z"][rb, rn]
        X = zrec.unsqueeze(-1) * d[rb, rn].float()
        dxyz = (X - verts[rb, vrec].float()) / L_REL
        zn = ((zrec - Z_REF) / Z_SCALE).unsqueeze(-1)
        if not self.geom:
            dxyz = torch.zeros_like(dxyz)
            zn = torch.zeros_like(zn)
        feat = torch.cat([tok[rb, rn].float(), dxyz, zn, torch.ones_like(zn),
                          assoc["hit"][rb, rn].float().unsqueeze(-1)], dim=-1)
        phi = self.rel(feat.to(wdt))                                                    # (R, C)
        key = rb * V + vrec
        mass = torch.zeros(B * V, device=dev, dtype=acc).index_add(0, key, wrec.to(acc))
        num = torch.zeros(B * V, C, device=dev, dtype=acc).index_add(
            0, key, wrec.to(acc).unsqueeze(-1) * phi.to(acc))
        has = mass > 0
        mean = (num / mass.clamp_min(1e-12).unsqueeze(-1)).to(torch.float32) * has.unsqueeze(-1)
        peak = torch.full((B * V, C), -1e4, device=dev, dtype=torch.float32).scatter_reduce(
            0, key.unsqueeze(-1).expand(-1, C), phi.float(), reduce="amax")
        peak = peak * has.unsqueeze(-1)
        x = torch.cat([mean, peak, torch.log1p(mass.to(torch.float32)).unsqueeze(-1)], -1)
        O = has.view(B, V).to(torch.float32)
        H = self.vin(x.view(B, V, 2 * C + 1).to(wdt))
        # ---- MeshGNN on the fixed 1-ring
        g = self.edge_geometry(verts)
        if not self.geom:
            g = torch.zeros_like(g)
        gemb = torch.relu(self.edge(g.to(wdt)))
        S = O
        for layer in self.layers:
            H, S = layer(H, S, gemb, self.ring_idx, self.ring_mask)
        e = lbs_slot_pool(H, S, O, self.lbs_w, self.lbs_hard)                            # (B, 16, 2C+2)
        # ---- residual (unmatched / weakly matched) summary with the same phi
        eb, en = (r > 0).nonzero(as_tuple=True)
        tr = tok[eb, en].float()
        feat_r = torch.cat([tr, torch.zeros(tr.shape[0], REL_IN - tr.shape[1], device=dev)], dim=-1)
        phi_r = self.rel(feat_r.to(wdt))
        rw = r[eb, en]
        rs = torch.zeros(B, device=dev, dtype=acc).index_add(0, eb, rw.to(acc))
        rnum = torch.zeros(B, C, device=dev, dtype=acc).index_add(
            0, eb, rw.to(acc).unsqueeze(-1) * phi_r.to(acc))
        rhas = (rs > 0).unsqueeze(-1)
        rmean = (rnum / rs.clamp_min(1e-12).unsqueeze(-1)).to(torch.float32) * rhas
        rpeak = torch.full((B, C), -1e4, device=dev, dtype=torch.float32).scatter_reduce(
            0, eb.unsqueeze(-1).expand(-1, C), phi_r.float(), reduce="amax") * rhas
        live = mask_f.sum(1).clamp_min(1.0)
        bslot = torch.cat([rmean, rpeak, (rs.to(torch.float32) / live).unsqueeze(-1),
                           rhas.to(torch.float32)], dim=-1)
        slots = torch.cat([e.to(torch.float32), bslot.unsqueeze(1)], dim=1)             # (B, 17, 2C+2)
        # ---- common readout
        zs = torch.relu(self.slot_proj(slots.reshape(B, -1, 1).to(wdt))).view(B, -1)
        h = torch.relu(self.fc1(torch.cat([zs, prev48.to(zs.dtype)], dim=-1)))
        delta = self.fc2(h)
        n_live = mask_f.sum()
        stats = {
            "xyz_valid": assoc["valid"].float().sum() / n_live.clamp_min(1.0),
            "xyz_hit": assoc["hit"].float().sum() / n_live.clamp_min(1.0),
            "xyz_resid_mass": r.sum() / n_live.clamp_min(1.0),
            "xyz_verts_direct": O.sum(1).mean(),
            "xyz_verts_prop": S.sum(1).mean(),
        }
        return delta, stats, {"slots": slots, "O": O, "S": S, "w": w, "r": r}


__all__ = ["ASSOC_CHUNK", "L_EDGE", "L_REL", "REL_IN", "R_MAX", "XYZMeshLayer", "XYZMeshNet",
           "agent_log", "associate_rays", "lbs_slot_pool", "mesh_ring", "ray_triangle",
           "stride_sample", "vertex_face_table"]
