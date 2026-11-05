r"""PFO2 arm (b), first-moment ("torque") slot pooling (docs/EXPERIMENT_RECORDS.md#pfo2-s1b-prereg-20261103; registered
only if P0-D passes): semkine/pfo2_trunk.py's trunk plus, per joint j, the first moment of H3 about the joint in the
previous hand frame (P0-D's feature, tools/pfo2/p0d.py `moments`)

    first_j = sum_v a_vj H_v (x) R^T (v - J_j) / 1 cm / sum_v a_vj,   a_vj = W_vj S_v

entering the readout through a zero-initialised grouped projection (C x 3 -> slot_dim per joint, no bias) added to the
joint slot's compressed features before their ReLU (the residual slot gets nothing), so the initial function equals the
base's. Variant strings: semkine/pfo2_trunk.py's with "+mom" (e.g. "gshare+mom", "gshare+self+mom").
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import pfo2_trunk as P2
from . import pfo_trunk2 as T2
from . import xyz_mesh as XM

L_M = 0.01
_NATIVE = {"cls": XM.XYZMeshNet}
_STATE = {"variant": None}


def parse(variant):
    parts = [p for p in variant.split("+") if p]
    mom = "mom" in parts
    rest = "+".join(p for p in parts if p != "mom")
    form, mods = P2.parse(rest)
    return form, mods, mom


def first_moments(H, S, W, verts, joints, R):
    """(B, J, C, 3): sum_v a_vj H_v (x) R^T (v - J_j) / L_M / sum_v a_vj with a = W S, by batched products."""
    B, V, C = H.shape
    J = W.shape[1]
    a = W.unsqueeze(0).to(H.dtype) * S.unsqueeze(-1).to(H.dtype)                          # (B, V, J)
    den = a.sum(1).clamp_min(1e-6)                                                         # (B, J)
    M0 = torch.bmm(a.transpose(1, 2), H)                                                   # (B, J, C)
    Av = (a.unsqueeze(-1) * verts.to(H.dtype).unsqueeze(2)).reshape(B, V, J * 3)
    M1 = torch.bmm(Av.transpose(1, 2), H).view(B, J, 3, C)                                  # sum a v_k H_c
    first = M1 - M0.unsqueeze(2) * joints.to(H.dtype).unsqueeze(-1)                         # (B, J, 3, C)
    first = torch.einsum("bkl,bjkc->bjcl", R.to(H.dtype), first)                            # R^T, -> (B, J, C, 3)
    return first / L_M / den.view(B, J, 1, 1)


class PFO2MeshNetB(P2.PFO2MeshNet):
    mom = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mom:
            J = self.lbs_w.shape[1]
            slot_dim = self.slot_proj.out_channels // self.N_SLOTS
            with torch.random.fork_rng(devices=[]):
                self.mom_proj = nn.Conv1d(J * self.hidden * 3, J * slot_dim, 1, groups=J, bias=False)
            nn.init.zeros_(self.mom_proj.weight)
            with np.load(str(T2._mano_npz()), allow_pickle=False) as z:
                self.register_buffer("j_reg", torch.tensor(np.asarray(z["J_regressor"], np.float32)), persistent=False)

    def moment_term(self, H, S, verts, prev48):
        B = H.shape[0]
        joints = torch.einsum("jv,bvk->bjk", self.j_reg, verts.float())
        R = T2.rodrigues(prev48[:, 0:3].float())
        first = first_moments(H.float(), S, self.lbs_w, verts.float(), joints, R)
        m = self.mom_proj(first.reshape(B, -1, 1).to(self.mom_proj.weight.dtype))          # (B, J * slot_dim, 1)
        slot_dim = self.slot_proj.out_channels // self.N_SLOTS
        return F.pad(m, (0, 0, 0, slot_dim))                                                # residual slot: 0

    def forward(self, tok, d, mask, verts, assoc, prev48, ablate: bool = False, verts_geo=None, capture: bool = False,
                h3_transform=None):
        if not self.mom:
            return super().forward(tok, d, mask, verts, assoc, prev48, ablate, verts_geo, capture, h3_transform)
        held = {}
        native_pool = XM.lbs_slot_pool

        def pool(H, S, O, W, hard):
            held["H"], held["S"] = H, S
            return native_pool(H, S, O, W, hard)
        proj = self.slot_proj

        def proj_forward(x):
            return nn.Conv1d.forward(proj, x) + self.moment_term(held["H"], held["S"], verts, prev48).to(x.dtype)
        XM.lbs_slot_pool = pool
        proj.forward = proj_forward
        try:
            return super().forward(tok, d, mask, verts, assoc, prev48, ablate, verts_geo, capture, h3_transform)
        finally:
            XM.lbs_slot_pool = native_pool
            del proj.forward


def install(variant):
    form, mods, mom = parse(variant)
    if XM.XYZMeshNet is not _NATIVE["cls"]:
        raise RuntimeError("XYZMeshNet already patched")
    XM.XYZMeshNet = type(f"PFO2MeshNetB_{variant.replace('+', '_')}", (PFO2MeshNetB,),
                         {"variant": form, "mods": frozenset(mods), "mom": mom})
    _STATE["variant"] = variant
    return variant


def uninstall():
    XM.XYZMeshNet = _NATIVE["cls"]
    _STATE["variant"] = None


def installed():
    return _STATE["variant"]


__all__ = ["PFO2MeshNetB", "first_moments", "install", "installed", "parse", "uninstall"]
