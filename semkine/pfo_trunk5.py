r"""PFO per-vertex implementation of the MeshGNN control "self" (docs/EXPERIMENT_RECORDS.md#pfo-p3c-prereg-20261103).

With every 1-ring index replaced by the vertex itself (semkine/pfo_trunk.py / pfo_trunk4.py "self"), the edge geometry
is zero, so the gate of a round is the per-channel constant sigmoid(W_g relu(b_e) + c); the mean over the ring is the
vertex's own message, and support never spreads (S = O). A round is then a per-vertex residual layer

    H_j <- H_j + O_j relu(W_s H_j + gate * W_m H_j)

and unobserved vertices keep their features. `PFOMeshNet5` computes exactly that, on the observed vertices only, with the
parameters (names, shapes, initial values) of the pfo_trunk4 "self" variant, so a checkpoint trained with pfo_trunk4
"self" (+ flags) loads unchanged. Variant strings are pfo_trunk4's with mp "vtx" for "self" (e.g. "vtx", "vtx+pos");
tests/test_pfo_trunk5.py checks equality with pfo_trunk4.
"""
from __future__ import annotations

import torch

from . import pfo_trunk2 as T2
from . import pfo_trunk4 as T4
from . import xyz_mesh as XM

_NATIVE = {"cls": XM.XYZMeshNet}
_STATE = {"variant": None}


def to_trunk4(variant):
    """The pfo_trunk4 variant string of a trunk5 one ("vtx+pos" -> "self+pos")."""
    parts = [p for p in variant.split("+") if p]
    if "vtx" not in parts:
        raise ValueError(f"trunk5 variants use mp 'vtx': {variant!r}")
    out = "+".join("self" if p == "vtx" else p for p in parts)
    mp, _ = T4.parse(out)
    if mp != "self":
        raise ValueError(f"bad variant {variant!r}")
    return out


def from_trunk4(variant):
    """The trunk5 string of a pfo_trunk4 "self" variant, None for any other mp."""
    mp, _ = T4.parse(variant)
    return None if mp != "self" else "+".join("vtx" if p == "self" else p for p in variant.split("+") if p)


class VertexLayer(XM.XYZMeshLayer):
    """An `XYZMeshLayer` evaluated on self edges: the per-vertex form of its forward (same parameters)."""

    def forward(self, h, gemb0):
        gate = torch.sigmoid(self.gate(gemb0))
        return h + torch.relu(self.slf(h) + gate * self.msg(h))


class PFOMeshNet5(T4.PFOMeshNet4):
    variant = "self"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for layer in self.layers:
            layer.__class__ = VertexLayer

    def forward(self, tok, d, mask, verts, assoc, prev48, ablate: bool = False, verts_geo=None, capture: bool = False,
                h3_transform=None):
        if verts_geo is not None or self.sign or h3_transform is not None or self.xag_h3_transform is not None:
            raise ValueError("PFO trunk5 covers the S45 / S45-SM forward only")
        fl = self.flags
        B, N, _ = tok.shape
        V = verts.shape[1]
        C = self.hidden
        dev = tok.device
        wdt = self.rel[0].weight.dtype
        acc = torch.float32 if self.training else torch.float64
        mask_f = mask.to(torch.float32)
        w = assoc["q"].unsqueeze(-1) * assoc["A"] * mask_f.unsqueeze(-1)
        r = (1.0 - assoc["q"]) * mask_f
        if ablate:
            w = torch.zeros_like(w)
            r = torch.zeros_like(r)
        rb, rn, rr = (w > 0).nonzero(as_tuple=True)
        vrec = assoc["vid"][rb, rn, rr]
        wrec = w[rb, rn, rr]
        drec = d[rb, rn].float()
        zrec = assoc["z"][rb, rn]
        vxyz = verts[rb, vrec].float()
        X = zrec.unsqueeze(-1) * drec
        dxyz = (X - vxyz) / XM.L_REL
        R = T2.rodrigues(prev48[:, 0:3].float()) if fl & {"pos", "hframe"} else None
        if "hframe" in fl:
            dxyz = torch.einsum("rji,rj->ri", R[rb], dxyz)
        zn = ((zrec - XM.Z_REF) / XM.Z_SCALE).unsqueeze(-1)
        feat = torch.cat([tok[rb, rn].float(), dxyz, zn, torch.ones_like(zn),
                          assoc["hit"][rb, rn].float().unsqueeze(-1)], dim=-1)
        if "pos" in fl:
            j0 = torch.einsum("v,bvk->bk", self.j0_reg, verts.float())
            pos = torch.einsum("rji,rj->ri", R[rb], vxyz - j0[rb]) / T2.POS_SCALE
            phi = self.rel[2](self.rel[1](self.rel[0](feat.to(wdt)) + self.pos_in(pos.to(wdt))))
        else:
            phi = self.rel(feat.to(wdt))
        key = rb * V + vrec
        mass = torch.zeros(B * V, device=dev, dtype=acc).index_add(0, key, wrec.to(acc))
        num = torch.zeros(B * V, C, device=dev, dtype=acc).index_add(0, key, wrec.to(acc).unsqueeze(-1) * phi.to(acc))
        has = mass > 0
        mean = (num / mass.clamp_min(1e-12).unsqueeze(-1)).to(torch.float32) * has.unsqueeze(-1)
        peak = torch.full((B * V, C), -1e4, device=dev, dtype=torch.float32).scatter_reduce(
            0, key.unsqueeze(-1).expand(-1, C), phi.float(), reduce="amax")
        peak = peak * has.unsqueeze(-1)
        x = torch.cat([mean, peak, torch.log1p(mass.to(torch.float32)).unsqueeze(-1)], -1)
        O = has.view(B, V).to(torch.float32)
        H = self.vin(x.view(B, V, 2 * C + 1).to(wdt))
        # the "self" rounds on the observed vertices only (edge geometry of a self edge is zero)
        gemb0 = torch.relu(self.edge(torch.zeros(1, 4, device=dev, dtype=wdt)))
        Hf = H.reshape(B * V, C)
        obs = has.nonzero(as_tuple=True)[0]
        h = Hf[obs]
        for layer in self.layers:
            h = layer(h, gemb0)
        H = Hf.index_copy(0, obs, h).view(B, V, C)
        S = O
        e = XM.lbs_slot_pool(H, S, O, self.lbs_w, self.lbs_hard)
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
        p48 = torch.zeros_like(prev48) if "noprev" in fl else prev48
        h = torch.relu(self.fc1(torch.cat([zs, p48.to(zs.dtype)], dim=-1)))
        delta = self.fc2(h)
        n_live = mask_f.sum()
        stats = {"xyz_valid": assoc["valid"].float().sum() / n_live.clamp_min(1.0),
                 "xyz_hit": assoc["hit"].float().sum() / n_live.clamp_min(1.0),
                 "xyz_resid_mass": r.sum() / n_live.clamp_min(1.0), "xyz_verts_direct": O.sum(1).mean(),
                 "xyz_verts_prop": S.sum(1).mean()}
        return delta, stats, {"slots": slots, "O": O, "S": S, "w": w, "r": r}


def install(variant):
    v4 = to_trunk4(variant)
    if XM.XYZMeshNet is not _NATIVE["cls"]:
        raise RuntimeError("XYZMeshNet already patched")
    XM.XYZMeshNet = type(f"PFOMeshNet5_{variant.replace('+', '_')}", (PFOMeshNet5,), {"variant": v4})
    _STATE["variant"] = variant
    return variant


def uninstall():
    XM.XYZMeshNet = _NATIVE["cls"]
    _STATE["variant"] = None


def installed():
    return _STATE["variant"]


__all__ = ["PFOMeshNet5", "VertexLayer", "from_trunk4", "install", "installed", "to_trunk4", "uninstall"]
