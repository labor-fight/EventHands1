r"""PFO composable trunk (for the final candidate and its leave-one-out ablations): one opt-in subclass of S45's
`XYZMeshNet` whose variant string combines the single changes of semkine/pfo_trunk.py, pfo_trunk2.py and pfo_trunk3.py:

  mp:   full (native) | nomp | self | rewire | l1 | gshare      (exactly one; default full)
  pos | hframe | noprev                                        (any subset)
  e.g. "nomp+pos", "full+noprev", "gshare". Construction runs the native constructor first (the same random draws),
then applies the changes in the order of the single-change modules (MeshGNN, then the isolated-RNG `pos_in`), so
parameter names and initial values equal the single-change variants'. The forward is pfo_trunk2's re-written forward
with the MeshGNN options; tests/test_pfo_trunk4.py checks equality with every single-change variant.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import pfo_trunk as T1
from . import pfo_trunk2 as T2
from . import pfo_trunk3 as T3
from . import xyz_mesh as XM

MP = ("full", "nomp", "self", "rewire", "l1", "gshare")
FLAGS = ("pos", "hframe", "noprev")
_NATIVE = {"cls": XM.XYZMeshNet}
_STATE = {"variant": None}


def parse(variant):
    parts = [p for p in variant.split("+") if p]
    mp = [p for p in parts if p in MP]
    flags = {p for p in parts if p in FLAGS}
    bad = [p for p in parts if p not in MP and p not in FLAGS]
    if bad or len(mp) > 1:
        raise ValueError(f"bad variant {variant!r}")
    return (mp[0] if mp else "full"), flags


class PFOMeshNet4(XM.XYZMeshNet):
    variant = "full"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mp, self.flags = parse(self.variant)
        if self.mp == "nomp":
            del self.layers
            del self.edge
            self.layers = nn.ModuleList()
            self.edge = nn.Identity()
        elif self.mp == "self":
            V, K = self.ring_idx.shape
            self.ring_idx = torch.arange(V).unsqueeze(1).repeat(1, K)
        elif self.mp == "rewire":
            idx, mask, info = T1.rewired_ring(self.ring_idx.cpu(), self.ring_mask.cpu())
            self.ring_idx, self.ring_mask = idx, mask
            self.rewire_info = info
        elif self.mp == "l1":
            self.layers = nn.ModuleList(list(self.layers)[:1])
        elif self.mp == "gshare":
            first = self.layers[0]
            shared = []
            for layer in self.layers:
                s = T3.SharedGateLayer.__new__(T3.SharedGateLayer)
                nn.Module.__init__(s)
                s.msg, s.slf = layer.msg, layer.slf
                shared.append(s)
            self.gate_shared = first.gate
            self.layers = nn.ModuleList(shared)
        if "pos" in self.flags:
            rel_hidden = self.rel[0].out_features
            with torch.random.fork_rng(devices=[]):
                self.pos_in = nn.Linear(3, rel_hidden, bias=False)
            nn.init.zeros_(self.pos_in.weight)
            with np.load(str(T2._mano_npz()), allow_pickle=False) as z:
                self.register_buffer("j0_reg", torch.tensor(np.asarray(z["J_regressor"][0], np.float32)),
                                     persistent=False)

    def forward(self, tok, d, mask, verts, assoc, prev48, ablate: bool = False, verts_geo=None, capture: bool = False,
                h3_transform=None):
        if verts_geo is not None or self.sign or h3_transform is not None or self.xag_h3_transform is not None:
            raise ValueError("PFO trunk4 covers the S45 / S45-SM forward only")
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
        S = O
        if self.mp != "nomp":
            g = _NATIVE["cls"].edge_geometry(self, verts)
            gemb = torch.relu(self.edge(g.to(wdt)))
            if self.mp == "gshare":
                gate = torch.sigmoid(self.gate_shared(gemb))
                for layer in self.layers:
                    H, S = layer(H, S, gate, self.ring_idx, self.ring_mask)
            else:
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
    parse(variant)
    if XM.XYZMeshNet is not _NATIVE["cls"]:
        raise RuntimeError("XYZMeshNet already patched")
    XM.XYZMeshNet = type(f"PFOMeshNet4_{variant.replace('+', '_')}", (PFOMeshNet4,), {"variant": variant})
    _STATE["variant"] = variant
    return variant


def uninstall():
    XM.XYZMeshNet = _NATIVE["cls"]
    _STATE["variant"] = None


def installed():
    return _STATE["variant"]


__all__ = ["FLAGS", "MP", "PFOMeshNet4", "install", "installed", "parse", "uninstall"]
