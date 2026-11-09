r"""PMR stage S3b ladder variants of D1's R (docs/EXPERIMENT_RECORDS.md#pmr-s3b-prereg-20261109): opt-in subclasses of
semkine/pmr_trunk.py's `PMRNet` (unchanged), each one change against R (variant "r"), installed in place of
`semkine.xyz_mesh.XYZMeshNet` (`install(variant)`):

  rconst   gate ablation: the edge gates do not read the prior geometry; each gate is the sigmoid of a learned vector per
           edge type (zero-initialised, so 0.5 at the start). The geometry embedding is built (R's draws) and deleted.
  rnohub   hand-node ablation: messages only on the kinematic-tree edges; the hand node keeps its own input
           (whole-hand statistics, residual slot, root slice) and the root / translation readout, and no part reads it.
           The hub message maps are built (R's draws) and deleted.
  rnomesh  no event-mesh association: the node statistics do not use the router's point on the prior mesh. An on-hand
           event keeps its part weights (the skinning weights of its routed face, a template quantity) but is placed on
           its camera ray at the prior wrist joint's depth, X_i = z(J_0) d_i; no residual to the prior surface is
           formed (res = 0), so the moments carry only the event cloud about each prior joint (mean lever, lever x
           lever), and phi reads (X_i - J_0) / L_RES and the wrist depth in place of the residual and the routed depth.
Rounds 1 / 2: R with MODEL.ENCODER_LAYERS 1 / 2 (no code change; registered in tools/pmr/arms2.py).
Every added parameter is built inside an isolated RNG scope, so at one seed the shared parameters start equal to R's.
"""
from __future__ import annotations

import torch
from torch import nn

from . import pmr_trunk as PM
from . import xyz_mesh as XM

LADDER = ("rconst", "rnohub", "rnomesh")
_STATE = {"variant": None}


class PMRLadderNet(PM.PMRNet):
    variant = "r"
    ladder = "rconst"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.ladder not in LADDER:
            raise ValueError(f"PMR ladder variant in {LADDER}")
        if self.ladder == "rconst":
            del self.edge_emb, self.gate
            with torch.random.fork_rng(devices=[]):
                self.gate_const = nn.Parameter(torch.zeros(len(PM.EDGE_TYPES), self.hidden))
        elif self.ladder == "rnohub":
            for rd in self.rounds:
                rd.msg = nn.ModuleList(list(rd.msg)[:2])                   # parent->child, child->parent

    # ---------------------------------------------------------------- rconst
    def gates(self, J):
        if self.ladder != "rconst":
            return super().gates(J)
        g = torch.sigmoid(self.gate_const)[self.e_type]                     # (E, C)
        return g.unsqueeze(0).expand(J.shape[0], -1, -1).to(self.gate_const.dtype)

    # ---------------------------------------------------------------- rnohub
    def propagate(self, h, gate=None, messages=True):
        if self.ladder != "rnohub" or not messages:
            return super().propagate(h, gate, messages)
        for rd in self.rounds:
            upd = rd.slf(h)
            for k, lin in enumerate(rd.msg):                               # k in (0, 1): the tree edges only
                sel = getattr(self, f"e_sel{k}")
                msg = gate[:, sel] * lin(h)[:, self.e_src[sel]]
                upd = upd + torch.einsum("ev,bec->bvc", getattr(self, f"e_scatter{k}").to(msg.dtype), msg)
            h = h + torch.relu(upd)
        return h

    # ---------------------------------------------------------------- rnomesh
    def geometry(self, mask, d, verts, assoc, prev48, ablate, acc):
        g = super().geometry(mask, d, verts, assoc, prev48, ablate, acc)
        if self.ladder != "rnomesh":
            return g
        on = (g["m"] > 0).unsqueeze(-1)
        zj = g["J"][:, :1, 2:3]                                             # (B, 1, 1) prior wrist depth
        X = zj * d.to(acc)                                                  # (B, N, 3) ray point at that depth
        g["P"] = torch.where(on, X, torch.zeros_like(X))
        g["res"] = torch.zeros_like(g["res"])
        g["X_rel"] = torch.where(on, X - g["J"][:, :1], torch.zeros_like(X))
        g["z_wrist"] = zj
        return g

    def node_inputs(self, tok, d, mask, verts, assoc, prev48, ablate=False):
        if self.ladder != "rnomesh":
            return super().node_inputs(tok, d, mask, verts, assoc, prev48, ablate)
        B, N, _ = tok.shape
        dev = tok.device
        acc = torch.float32 if self.training else torch.float64
        with torch.autocast(device_type=dev.type, enabled=False):
            g = self.geometry(mask, d, verts, assoc, prev48, ablate, acc)
            Mp, sp, Mh, sh = self.statistics(g)
            on = (g["m"] > 0).unsqueeze(-1).float()
            zn = ((g["z_wrist"].float() - XM.Z_REF) / XM.Z_SCALE).expand(B, N, 1)
            f_on = torch.cat([tok.float(), g["X_rel"].float() / PM.L_RES, zn * on, on], -1)
            f_off = torch.cat([tok.float(), torch.zeros(B, N, 5, device=dev)], -1)
        wdt = self.phi[0].weight.dtype
        phi_on = self.phi(f_on.to(wdt))
        phi_off = self.phi(f_off.to(wdt))
        with torch.autocast(device_type=dev.type, enabled=False):
            phi_on, phi_off = phi_on.to(acc), phi_off.to(acc)
            Phi_p = torch.einsum("bnk,bnc->bkc", g["omega"], phi_on) / Mp.clamp_min(1e-12).unsqueeze(-1)
            Phi_h = torch.einsum("bn,bnc->bc", g["m"], phi_on) / Mh.clamp_min(1e-12).unsqueeze(-1)
            rs = g["r"].sum(1)
            Phi_r = torch.einsum("bn,bnc->bc", g["r"], phi_off) / rs.clamp_min(1e-12).unsqueeze(-1)
            live = g["mask_f"].sum(1).clamp_min(1.0)
            p3 = prev48.to(acc).view(B, PM.N_PARTS, 3)
            x_part = torch.cat([Phi_p, sp, p3], -1)
            x_hub = torch.cat([Phi_h, sh, Phi_r, (rs / live).unsqueeze(-1), (rs > 0).to(acc).unsqueeze(-1), p3[:, 0]], -1)
        return {"x_part": x_part, "x_hub": x_hub, "Mp": Mp, "Mh": Mh, "sp": sp, "sh": sh, **g}


def install(variant):
    if variant not in LADDER:
        raise ValueError(f"PMR ladder variant in {LADDER}")
    if XM.XYZMeshNet is not PM._NATIVE["cls"]:
        raise RuntimeError("XYZMeshNet already patched")
    XM.XYZMeshNet = type(f"PMRLadderNet_{variant}", (PMRLadderNet,), {"variant": "r", "ladder": variant})
    _STATE["variant"] = variant
    return variant


def uninstall():
    XM.XYZMeshNet = PM._NATIVE["cls"]
    _STATE["variant"] = None


def installed():
    return _STATE["variant"]


__all__ = ["LADDER", "PMRLadderNet", "install", "installed", "uninstall"]
