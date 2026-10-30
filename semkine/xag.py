"""Opt-in XAG routing and XCL oracle controls.

The source labels are geometric proxies.  ``part_route`` searches the original
top-k candidate set, within the original hit/near branch. ``face_route`` replaces
only existing positive-mass routes. Neither operation recovers unmatched events.
XCL ``full_oracle`` additionally replaces depth/quality/support and recalls known
sources; it is a non-deployable ceiling, separate from the historical routes.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

ASSOC_MODES = ("base", "part_route", "face_route", "full_oracle")
ORACLE_GNN_MODES = ("oracle_dense", "oracle_observed")
GNN_MODES = ("legacy", "global_residual") + ORACLE_GNN_MODES
SOURCE_FIELDS = ("xag_source_index", "xag_source_ms", "xag_part", "xag_face", "xag_bary", "xag_valid")


def validate_config(model_cfg: Mapping, *, xyz_mesh: bool) -> tuple[str, str]:
    """Fail closed on combinations outside the C1 XAG comparison."""
    present = any(k in model_cfg for k in ("XAG_ASSOC", "XAG_GNN"))
    if present and not xyz_mesh:
        raise ValueError("MODEL.XAG_ASSOC / XAG_GNN belong to ENCODER=xyz_mesh")
    assoc = model_cfg.get("XAG_ASSOC", "base")
    gnn = model_cfg.get("XAG_GNN", "legacy")
    if assoc not in ASSOC_MODES:
        raise ValueError(f"MODEL.XAG_ASSOC must be one of {ASSOC_MODES}, got {assoc!r}")
    if gnn not in GNN_MODES:
        raise ValueError(f"MODEL.XAG_GNN must be one of {GNN_MODES}, got {gnn!r}")
    if gnn in ORACLE_GNN_MODES and assoc != "base":
        raise ValueError("XCL oracle GNN controls require XAG_ASSOC=base; oracle crossings are not registered")
    if assoc != "base" or gnn != "legacy":
        expected = {"XYZ_ASSOC_MODE": "hit_first", "XYZ_ASSOC_PREV": "self",
                    "XYZ_RESIDUAL_MODE": "legacy", "XYZ_GEOM": True,
                    "ENCODER_HIDDEN": 128, "ENCODER_LAYERS": 3,
                    "XYZ_SLOT_DIM": 32, "XYZ_READOUT_HIDDEN": 256}
        bad = [k for k, v in expected.items() if model_cfg.get(k, v) != v]
        if bad:
            raise ValueError("XAG requires the C1 hit_first/self, geometry, legacy residual and fixed "
                             "128x3/32/256 interfaces; incompatible: " + ", ".join(bad))
    return assoc, gnn


@torch.no_grad()
def sample_sources(batch, src: torch.Tensor, mask: torch.Tensor, *,
                   require_depth: bool = False) -> dict[str, torch.Tensor]:
    """Gather the exact production stride indices, including augmentation provenance."""
    n = int(batch.events.shape[0])
    out = {}
    for name in SOURCE_FIELDS + (("xag_depth",) if require_depth else ()):
        value = getattr(batch, name, None)
        if value is None:
            raise RuntimeError(f"XAG oracle routing requires batch.{name}; never infer sources from target")
        shape = (n, 3) if name == "xag_bary" else (n,)
        dtype = torch.float32 if name in ("xag_bary", "xag_depth") else torch.bool if name == "xag_valid" else torch.int64
        if value.shape != shape or value.dtype != dtype or value.device != src.device:
            raise ValueError(f"batch.{name} must be {shape} {dtype} on {src.device}")
        out[name] = value[src]
    # Padded copies and synthetic/hot events are never oracle eligible.
    out["xag_valid"] = out["xag_valid"] & mask & (out["xag_source_index"] >= 0) & (out["xag_source_ms"] >= 0)
    return out


@torch.no_grad()
def face_parts(lbs_weights: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Exact XA pseudo-part convention: hard-majority, tie by summed LBS weight.

    Float64 sums and ascending joint order reproduce step1_common.face_part_table.
    """
    hard = lbs_weights.argmax(-1)
    hj = hard[faces.long()]
    a, b, c = hj.unbind(-1)
    majority = torch.where(a == b, a, torch.where(a == c, a, torch.where(b == c, b, -1)))
    scores = lbs_weights.double()[faces.long()].sum(1)
    allowed = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, hj, True)
    tie = scores.masked_fill(~allowed, -torch.inf).argmax(-1)
    return torch.where(majority >= 0, majority, tie)


@torch.no_grad()
def route_association(mode, assoc, d, mask, verts, faces, vert_faces, lbs_weights,
                      band, front_tol, z_tol, *, source_part, source_face,
                      source_bary, source_valid, source_depth=None, k=16, chunk=4096):
    """Return ``(association, audit)``; historical routes change only vid/A.

    Audit masks have the original padded event shape. Unknowns, source failures,
    and absent in-branch candidates fall back byte-for-byte. A repaired route
    retains the baseline event's sum(A), so q*A mass and residual mass are fixed.
    """
    if mode not in ASSOC_MODES:
        raise ValueError(f"unknown XAG association {mode!r}")
    if mode == "full_oracle":
        if assoc["vid"].shape != (*mask.shape, 4) or assoc["z"].shape != mask.shape:
            raise ValueError("full_oracle requires the four-record hit_first association")
        if source_depth is None:
            raise RuntimeError("full_oracle requires source_depth from batch.xag_depth; never infer depth from prev")
        if source_depth.shape != mask.shape or source_depth.device != mask.device or source_depth.dtype != torch.float32:
            raise ValueError("full_oracle source_depth must be float32 with the event-mask shape/device")
        known = source_valid & (source_face >= 0) & (source_face < len(faces))
        known = known & torch.isfinite(source_bary).all(-1) & (source_bary >= 0).all(-1) & (source_bary.sum(-1) > 0)
        if bool((known & mask & (~torch.isfinite(source_depth) | (source_depth <= 0))).any()):
            raise ValueError("full_oracle known sources require finite positive source_depth in metres")
        eligible = known & mask
        out = dict(assoc)
        for name in ("vid", "A", "q", "z", "valid", "hit"):
            out[name] = torch.zeros_like(assoc[name])
        b, n = eligible.nonzero(as_tuple=True)
        fv = faces.long()[source_face[b, n]]
        bary = source_bary[b, n].to(out["A"].dtype)
        bary = bary / bary.sum(-1, keepdim=True)
        out["vid"][b, n] = torch.cat([fv, fv[:, :1]], -1)
        out["A"][b, n] = torch.cat([bary, torch.zeros_like(bary[:, :1])], -1)
        out["q"][b, n] = 1
        out["z"][b, n] = source_depth[b, n].to(out["z"].dtype)
        out["valid"][b, n] = True
        out["hit"][b, n] = True
        return out, {"xag_known": eligible, "xag_eligible": eligible,
                     "xag_replaced": eligible, "xag_fallback": mask & ~eligible,
                     "xag_recalled": eligible & ~assoc["valid"], "xag_none": mask & ~eligible}
    eligible = mask & (assoc["q"] > 0) & assoc["valid"] & source_valid
    known = source_valid & (source_part >= 0) & (source_part < lbs_weights.shape[1])
    if mode == "face_route":
        known = source_valid & (source_face >= 0) & (source_face < len(faces))
        known = known & torch.isfinite(source_bary).all(-1) & (source_bary >= 0).all(-1) & (source_bary.sum(-1) > 0)
    eligible = eligible & known
    replaced = torch.zeros_like(mask, dtype=torch.bool)
    if mode == "base":
        return assoc, {"xag_known": known & mask, "xag_eligible": eligible, "xag_replaced": replaced}
    if assoc["vid"].shape[-1] != 4 or assoc["z"].ndim != 2:
        raise ValueError("XAG routes require the four-record hit_first association")
    out = dict(assoc)
    out["vid"] = assoc["vid"].clone()
    out["A"] = assoc["A"].clone()
    B, N = mask.shape
    rows = eligible.reshape(-1).nonzero().flatten()
    if mode == "face_route":
        b, n = rows // N, rows % N
        fv = faces.long()[source_face[b, n]]
        weights = source_bary[b, n].to(assoc["A"].dtype)
        weights = weights / weights.sum(-1, keepdim=True)
        weights = torch.cat([weights, torch.zeros_like(weights[:, :1])], -1)
        weights = weights * assoc["A"][b, n].sum(-1, keepdim=True)
        out["vid"][b, n] = torch.cat([fv, fv[:, :1]], -1)
        out["A"][b, n] = weights
        replaced[b, n] = True
    elif rows.numel():
        from .xyz_mesh import ray_triangle
        vf = verts.float()
        z = vf[..., 2]
        uv = (vf[..., :2] / z.clamp_min(1e-6).unsqueeze(-1)).masked_fill((z <= 1e-6).unsqueeze(-1), 1e6)
        fp = face_parts(lbs_weights, faces)
        vp = lbs_weights.argmax(-1)
        kk = max(1, min(int(k), vf.shape[1]))
        rank0 = torch.arange(kk, device=d.device)
        for sub in rows.split(int(chunk)):
            b, n = sub // N, sub % N
            dd = d[b, n].float()
            wanted = source_part[b, n]
            distance = (uv[b] - dd[:, None, :2]).square().sum(-1)
            near = distance.topk(kk, largest=False, dim=-1)
            nid, nang = near.indices, near.values.clamp_min(0).sqrt()
            cf = vert_faces[nid].reshape(len(sub), -1)
            safe = cf.clamp_min(0)
            tri = faces.long()[safe]
            points = vf[b[:, None, None], tri]
            t, u, v = ray_triangle(dd[:, None], points[:, :, 0], points[:, :, 1], points[:, :, 2])
            t = t.masked_fill((cf < 0) | (fp[safe] != wanted[:, None]), torch.inf)
            depth, sel = t.min(-1)
            hit_ok = torch.isfinite(depth) & assoc["hit"][b, n]
            chosen = safe.gather(1, sel[:, None]).squeeze(1)
            uu, vv = u.gather(1, sel[:, None]).squeeze(1), v.gather(1, sel[:, None]).squeeze(1)
            wh = torch.stack([(1 - uu - vv).clamp_min(0), uu.clamp_min(0), vv.clamp_min(0), torch.zeros_like(uu)], -1)
            wh = wh / wh.sum(-1, keepdim=True).clamp_min(1e-12)
            vh = faces.long()[chosen]
            vh = torch.cat([vh, vh[:, :1]], -1)
            # Preserve the original near rule, but choose its reference/front layer
            # among matching-part candidates inside the *original* top-k set.
            vn = vf[b[:, None], nid]
            zj = (vn * dd[:, None]).sum(-1) / dd.square().sum(-1, keepdim=True)
            part_ok = vp[nid] == wanted[:, None]
            nc = nang.masked_fill(~part_ok, torch.inf)
            ref = nc.min(-1, keepdim=True).values
            inband = part_ok & (nc <= band[b, None]) & (zj > 1e-4)
            same = inband & (nc <= ref + front_tol[b, None])
            front = zj.masked_fill(~same, torch.inf).min(-1, keepdim=True).values
            layer = inband & (zj <= front + float(z_tol))
            rank = torch.where(layer, rank0[None], rank0[None] + kk)
            cols = rank.argsort(-1)[:, :min(4, kk)]
            keep = layer.gather(1, cols)
            wn = torch.exp(-0.5 * (nang.gather(1, cols) / (band[b, None] / 2)).square()) * keep
            wn = wn / wn.sum(-1, keepdim=True).clamp_min(1e-12)
            vnid = nid.gather(1, cols)
            if kk < 4:
                wn = torch.nn.functional.pad(wn, (0, 4 - kk))
                vnid = torch.nn.functional.pad(vnid, (0, 4 - kk))
            near_ok = keep.any(-1) & ~assoc["hit"][b, n] & (wn.sum(-1) > 0)
            ok = hit_ok | near_ok
            new_vid = torch.where(hit_ok[:, None], vh, vnid)
            new_a = torch.where(hit_ok[:, None], wh, wn) * assoc["A"][b, n].sum(-1, keepdim=True)
            out["vid"][b[ok], n[ok]] = new_vid[ok]
            out["A"][b[ok], n[ok]] = new_a[ok]
            replaced[b[ok], n[ok]] = True
    return out, {"xag_known": known & mask, "xag_eligible": eligible,
                 "xag_replaced": replaced, "xag_fallback": mask & ~replaced}


class GlobalResidual(nn.Module):
    """Support-preserving global messages after one unchanged local mesh round."""

    def __init__(self, hidden=128, heads=4, ff_hidden=256, n_parts=16):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.identity = nn.Linear(n_parts, hidden, bias=False)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, ff_hidden), nn.GELU(), nn.Linear(ff_hidden, hidden))

    def forward(self, H, S, lbs_weights):
        support = S > 0
        # MultiheadAttention would return NaN with all keys masked. The dummy key
        # is used only for an empty row; outputs are then masked to exact zero.
        key_mask = ~support
        key_mask = key_mask.clone()
        key_mask[:, 0] = key_mask[:, 0] & support.any(-1)
        value = self.norm1(H)
        position = self.identity(lbs_weights.to(H.dtype)).unsqueeze(0)
        query = value + position  # LBS identity never enters the value stream.
        update, _ = self.attention(query, query, value, key_padding_mask=key_mask, need_weights=False)
        live = support.unsqueeze(-1).to(H.dtype)
        H = (H + update * live) * live
        H = (H + self.ff(self.norm2(H)) * live) * live
        return H

