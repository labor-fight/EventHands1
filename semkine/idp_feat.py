r"""IDP record feature (docs/EXPERIMENT_RECORDS.md#idp-prereg-20261105): B2's trunk (semkine/pfo_trunk4.py, variant
"gshare") with two per-record scalars computed from the prior mesh alone, entering the relation encoder's first layer
through a zero-initialised, bias-free map (the XVM `sign_in` / PFO `pos_in` pattern, built inside an isolated RNG scope):
with the map at zero the forward equals PFOMeshNet4("gshare") exactly, and the constructor's random draws are B2's.

For the finger (or the palm) of the record's vertex (`lbs_hard`), at the prior pose, camera frame:
  fore  projected MCP -> tip length divided by the length an extended, fronto-parallel chain would have at the MCP's
        depth (normalised image coordinates, so no intrinsics enter): 1 = extended and facing the camera, smaller when
        the finger is flexed or points at the camera; the palm uses wrist -> middle MCP
  flex  the sum of the two inter-phalanx angles, degrees; 0 for the palm
Joints are regressed from the prior vertices (MANO J_regressor + the fingertip vertices, OpenPose order). The two
features enter as [(fore - 0.8) / 0.2, flex / 90]. The residual slot (off-hand events) is unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import pfo_trunk2 as T2
from . import pfo_trunk4 as T4
from . import xyz_mesh as XM

FEAT_IN = 2
FORE_CENTER, FORE_SCALE, FLEX_SCALE = 0.8, 0.2, 90.0
MCP_OF_FINGER = (5, 9, 17, 13, 1)            # OpenPose MCP of MANO's finger order: index, middle, pinky, ring, thumb
PALM = 5
_NATIVE = {"cls": XM.XYZMeshNet}
_STATE = {"on": False}


class IDPMeshNet(T4.PFOMeshNet4):
    variant = "gshare"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.flags or self.mp != "gshare":
            raise ValueError("IDP builds on the gshare trunk only")
        rel_hidden = self.rel[0].out_features
        with torch.random.fork_rng(devices=[]):
            self.fore_in = nn.Linear(FEAT_IN, rel_hidden, bias=False)
        nn.init.zeros_(self.fore_in.weight)
        with np.load(str(T2._mano_npz()), allow_pickle=False) as z:
            self.register_buffer("j_reg", torch.tensor(np.asarray(z["J_regressor"], np.float32)), persistent=False)
            self.register_buffer("tip_ids", torch.tensor(np.asarray(z["fingertip_vertex_ids"], np.int64)),
                                 persistent=False)
            self.register_buffer("op21", torch.tensor(np.asarray(z["mano16_tips_to_openpose21"], np.int64)),
                                 persistent=False)
        part_finger = torch.full((16,), PALM, dtype=torch.long)
        for p in range(1, 16):
            part_finger[p] = (p - 1) // 3
        self.register_buffer("part_finger", part_finger, persistent=False)
        self.register_buffer("mcp_of_finger", torch.tensor(MCP_OF_FINGER, dtype=torch.long), persistent=False)

    @torch.no_grad()
    def finger_geometry(self, verts):
        """`(B, 6, 2)`: [fore, flex_deg] per finger (MANO order index, middle, pinky, ring, thumb) and the palm."""
        with torch.autocast(device_type=verts.device.type, enabled=False):
            V = verts.float()
            B = V.shape[0]
            J = torch.cat([torch.einsum("ji,bik->bjk", self.j_reg, V), V[:, self.tip_ids]], 1)[:, self.op21]
            n = J[..., :2] / J[..., 2:3].clamp_min(1e-6)
            out = V.new_zeros(B, 6, 2)
            for f in range(5):
                b = int(self.mcp_of_finger[f])
                seg = [J[:, b + k + 1] - J[:, b + k] for k in range(3)]
                chain = seg[0].norm(dim=-1) + seg[1].norm(dim=-1) + seg[2].norm(dim=-1)
                px = (n[:, b + 3] - n[:, b]).norm(dim=-1)
                out[:, f, 0] = (px * J[:, b, 2].clamp_min(1e-6) / chain.clamp_min(1e-6)).clamp(0.0, 1.5)
                ang = torch.zeros(B, device=V.device)
                for k in range(2):
                    c = (seg[k] * seg[k + 1]).sum(-1) / (seg[k].norm(dim=-1) * seg[k + 1].norm(dim=-1)).clamp_min(1e-9)
                    ang = ang + torch.acos(c.clamp(-1.0, 1.0))
                out[:, f, 1] = ang * (180.0 / np.pi)
            seg = J[:, 9] - J[:, 0]
            out[:, PALM, 0] = ((n[:, 9] - n[:, 0]).norm(dim=-1) * J[:, 0, 2].clamp_min(1e-6)
                               / seg.norm(dim=-1).clamp_min(1e-6)).clamp(0.0, 1.5)
            return out

    def record_features(self, geo, rb, vrec):
        """`(R, 2)` scaled [fore, flex] of the finger of every record's vertex."""
        g = geo[rb, self.part_finger[self.lbs_hard[vrec]]]
        return torch.stack([(g[:, 0] - FORE_CENTER) / FORE_SCALE, g[:, 1] / FLEX_SCALE], -1)

    def forward(self, tok, d, mask, verts, assoc, prev48, ablate: bool = False, verts_geo=None, capture: bool = False,
                h3_transform=None):
        if verts_geo is not None or self.sign or h3_transform is not None or self.xag_h3_transform is not None:
            raise ValueError("IDP trunk covers the S45 / S45-SM forward only")
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
        zn = ((zrec - XM.Z_REF) / XM.Z_SCALE).unsqueeze(-1)
        feat = torch.cat([tok[rb, rn].float(), dxyz, zn, torch.ones_like(zn),
                          assoc["hit"][rb, rn].float().unsqueeze(-1)], dim=-1)
        extra = self.record_features(self.finger_geometry(verts), rb, vrec)                  # IDP: the only change
        phi = self.rel[2](self.rel[1](self.rel[0](feat.to(wdt)) + self.fore_in(extra.to(wdt))))
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
        g = _NATIVE["cls"].edge_geometry(self, verts)
        gemb = torch.relu(self.edge(g.to(wdt)))
        gate = torch.sigmoid(self.gate_shared(gemb))
        for layer in self.layers:
            H, S = layer(H, S, gate, self.ring_idx, self.ring_mask)
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
        h = torch.relu(self.fc1(torch.cat([zs, prev48.to(zs.dtype)], dim=-1)))
        delta = self.fc2(h)
        n_live = mask_f.sum()
        stats = {"xyz_valid": assoc["valid"].float().sum() / n_live.clamp_min(1.0),
                 "xyz_hit": assoc["hit"].float().sum() / n_live.clamp_min(1.0),
                 "xyz_resid_mass": r.sum() / n_live.clamp_min(1.0), "xyz_verts_direct": O.sum(1).mean(),
                 "xyz_verts_prop": S.sum(1).mean()}
        return delta, stats, {"slots": slots, "O": O, "S": S, "w": w, "r": r}


def install():
    if XM.XYZMeshNet is not _NATIVE["cls"]:
        raise RuntimeError("XYZMeshNet already patched")
    XM.XYZMeshNet = IDPMeshNet
    _STATE["on"] = True
    return "gshare+fore"


def uninstall():
    XM.XYZMeshNet = _NATIVE["cls"]
    _STATE["on"] = False


def installed():
    return "gshare+fore" if _STATE["on"] else None


__all__ = ["FEAT_IN", "IDPMeshNet", "MCP_OF_FINGER", "PALM", "install", "installed", "uninstall"]
