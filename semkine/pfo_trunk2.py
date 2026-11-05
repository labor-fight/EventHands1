r"""PFO trunk variants, second set (docs/EXPERIMENT_RECORDS.md#pfo-p2b-prereg-20261103): opt-in subclasses of S45's
`XYZMeshNet` with a re-written forward that builds the relation records like the native one and changes one thing:

  base2   the native computation (checks the re-written forward; equal to `XYZMeshNet.forward`, tests/test_pfo_trunk2.py)
  pos     each record also carries its vertex's position in the previous-state hand frame, R_prev^T (V_j - J0_prev) / 5 cm
          (J0 = the MANO wrist from the joint regressor, R_prev = the previous root rotation), through a zero-initialised,
          bias-free 3 -> rel_hidden map added to phi's first layer (built in an isolated RNG scope): the initial function
          is the base's. Motivation: an LBS slot averages per-vertex features that carry no vertex position, so a rotation
          (displacement linear in position) largely cancels in the slot mean.
  hframe  the record offset dXYZ is expressed in the previous-state hand frame (R_prev^T dXYZ) instead of the camera frame
  noprev  the common readout sees zeros instead of the previous pose prev[3:51] (fc1 keeps its shape)
`install(variant)` replaces `semkine.xyz_mesh.XYZMeshNet` before the model is built; `uninstall()` restores it.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import xyz_mesh as XM

VARIANTS = ("base2", "pos", "hframe", "noprev")
POS_SCALE = 0.05
_NATIVE = {"cls": XM.XYZMeshNet}
_STATE = {"variant": None}


def rodrigues(aa):
    """(B, 3) axis-angle -> (B, 3, 3)."""
    th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = aa / th
    K = torch.zeros(aa.shape[0], 3, 3, dtype=aa.dtype, device=aa.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    th = th.unsqueeze(-1)
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand_as(K)
    return eye + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


class PFOMeshNet2(XM.XYZMeshNet):
    variant = "base2"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.variant == "pos":
            rel_hidden = self.rel[0].out_features
            with torch.random.fork_rng(devices=[]):
                self.pos_in = nn.Linear(3, rel_hidden, bias=False)
            nn.init.zeros_(self.pos_in.weight)
            with np.load(str(_mano_npz()), allow_pickle=False) as z:
                self.register_buffer("j0_reg", torch.tensor(np.asarray(z["J_regressor"][0], np.float32)),
                                     persistent=False)
        elif self.variant not in VARIANTS:
            raise ValueError(self.variant)

    def forward(self, tok, d, mask, verts, assoc, prev48, ablate: bool = False, verts_geo=None, capture: bool = False,
                h3_transform=None):
        if verts_geo is not None or self.sign or h3_transform is not None or self.xag_h3_transform is not None:
            raise ValueError("PFO trunk2 covers the S45 / S45-SM forward only")
        v = self.variant
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
        R = None
        if v in ("pos", "hframe"):
            R = rodrigues(prev48[:, 0:3].float())                                      # (B, 3, 3) prev root
        if v == "hframe":
            dxyz = torch.einsum("rji,rj->ri", R[rb], dxyz)                              # R^T dXYZ
        zn = ((zrec - XM.Z_REF) / XM.Z_SCALE).unsqueeze(-1)
        cols = [tok[rb, rn].float(), dxyz, zn, torch.ones_like(zn), assoc["hit"][rb, rn].float().unsqueeze(-1)]
        feat = torch.cat(cols, dim=-1)
        if v == "pos":
            j0 = torch.einsum("v,bvk->bk", self.j0_reg, verts.float())                  # (B, 3) prev wrist
            pos = torch.einsum("rji,rj->ri", R[rb], vxyz - j0[rb]) / POS_SCALE
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
        g = self.edge_geometry(verts)
        gemb = torch.relu(self.edge(g.to(wdt)))
        S = O
        for layer in self.layers:
            H, S = layer(H, S, gemb, self.ring_idx, self.ring_mask)
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
        p48 = torch.zeros_like(prev48) if v == "noprev" else prev48
        h = torch.relu(self.fc1(torch.cat([zs, p48.to(zs.dtype)], dim=-1)))
        delta = self.fc2(h)
        n_live = mask_f.sum()
        stats = {"xyz_valid": assoc["valid"].float().sum() / n_live.clamp_min(1.0),
                 "xyz_hit": assoc["hit"].float().sum() / n_live.clamp_min(1.0),
                 "xyz_resid_mass": r.sum() / n_live.clamp_min(1.0), "xyz_verts_direct": O.sum(1).mean(),
                 "xyz_verts_prop": S.sum(1).mean()}
        return delta, stats, {"slots": slots, "O": O, "S": S, "w": w, "r": r}


def _mano_npz():
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "assets/mano_right.npz"


def install(variant):
    if variant not in VARIANTS:
        raise ValueError(f"variant in {VARIANTS}")
    if XM.XYZMeshNet is not _NATIVE["cls"]:
        raise RuntimeError("XYZMeshNet already patched")
    XM.XYZMeshNet = type(f"PFOMeshNet2_{variant}", (PFOMeshNet2,), {"variant": variant})
    _STATE["variant"] = variant
    return variant


def uninstall():
    XM.XYZMeshNet = _NATIVE["cls"]
    _STATE["variant"] = None


def installed():
    return _STATE["variant"]


__all__ = ["PFOMeshNet2", "VARIANTS", "install", "installed", "rodrigues", "uninstall"]
