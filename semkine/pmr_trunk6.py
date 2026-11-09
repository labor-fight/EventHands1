r"""PMR stage S4c refinements of D1's gates (docs/EXPERIMENT_RECORDS.md#pmr-s4c-prereg-20261109): opt-in subclasses of
semkine/pmr_trunk.py's `PMRNet` (unchanged), installed in place of `semkine.xyz_mesh.XYZMeshNet` (`install(variant)`).
The gates stay a function of the prior FK joint geometry, computed once per forward and shared by the rounds; only the
geometric features they read change (D1: the edge's joint vector in camera axes / 2 cm, its length, its image-plane share
and the destination depth, plus the edge type).

  ghand   articulation geometry in the hand frame: the joint vector rotated into the prior root frame, R_0^T v / 2 cm,
          and its length (+ type); invariant to the hand's global pose (the ladder's geometry-free gate, d1g0, traded
          local for global accuracy; the camera-axis features change with the global pose)
  gview   observability only: length, image-plane share, destination depth (+ type); no direction
Each builds a new geometry embedding (its input width differs) in an isolated RNG scope after D1's construction and
deletes D1's; the shared parameters start equal to D1's at one seed. The node inputs are D1's, so D1's R-noprop /
R-mlp are the controls.
"""
from __future__ import annotations

import torch
from torch import nn

from . import pmr_trunk as PM
from . import xyz_mesh as XM

VARIANTS = ("ghand", "gview")
GEO_IN = {"ghand": 3 + 1 + 4, "gview": 1 + 1 + 1 + 4}
_STATE = {"variant": None}


def gate_features(kind, J, F, src, dst, etype):
    """(B, E, GEO_IN[kind]) gate inputs from the prior joints J (B, 16, 3) and node frames F (B, 16, 3, 3)."""
    anchor = torch.cat([J, J[:, :1]], 1)
    v = anchor[:, dst] - anchor[:, src]
    ln = v.norm(dim=-1, keepdim=True)
    oh = torch.nn.functional.one_hot(etype, len(PM.EDGE_TYPES)).to(J.dtype).unsqueeze(0).expand(J.shape[0], -1, -1)
    if kind == "ghand":
        R0 = F[:, 1]                                                       # part 1's frame = the root's rotation
        vh = torch.einsum("bji,bej->bei", R0, v)
        return torch.cat([vh / PM.L_EDGE, ln / PM.L_EDGE, oh], -1)
    fs = v[..., :2].norm(dim=-1, keepdim=True) / ln.clamp_min(1e-9)
    zd = (anchor[:, dst, 2:3] - XM.Z_REF) / XM.Z_SCALE
    return torch.cat([ln / PM.L_EDGE, fs, zd, oh], -1)


class PMRGateNet(PM.PMRNet):
    variant = "r"
    gate_kind = "ghand"

    def __init__(self, *args, edge_dim=32, **kwargs):
        super().__init__(*args, edge_dim=edge_dim, **kwargs)
        del self.edge_emb
        with torch.random.fork_rng(devices=[]):
            self.edge_emb = nn.Linear(GEO_IN[self.gate_kind], edge_dim)

    def gates_from(self, J, F):
        dt = self.edge_emb.weight.dtype
        g = gate_features(self.gate_kind, J.to(dt), F.to(dt), self.e_src, self.e_dst, self.e_type)
        return torch.sigmoid(self.gate(torch.relu(self.edge_emb(g))))

    def forward(self, tok, d, mask, verts, assoc, prev48, ablate: bool = False, verts_geo=None, capture: bool = False,
                h3_transform=None, messages: bool = True):
        if verts_geo is not None or h3_transform is not None or self.xag_h3_transform is not None:
            raise ValueError("PMR covers the S45 / S45-SM forward only")
        n = self.node_inputs(tok, d, mask, verts, assoc, prev48, ablate)
        h0 = self.encode(n["x_part"], n["x_hub"])
        hL = self.propagate(h0, self.gates_from(n["J"], n["F"]) if messages else None, messages=messages)
        delta = self.readout(hL)
        n_live = mask.float().sum()
        stats = {"xyz_valid": assoc["valid"].float().sum() / n_live.clamp_min(1.0),
                 "xyz_hit": assoc["hit"].float().sum() / n_live.clamp_min(1.0),
                 "xyz_resid_mass": n["r"].float().sum() / n_live.clamp_min(1.0),
                 "pmr_parts_live": (n["Mp"] > 0).float().sum(1).mean()}
        aux = {"w": n["w"], "r": n["r"]}
        if capture or self.xag_capture_enabled:
            aux.update({k: n[k].detach() for k in ("x_part", "x_hub", "Mp", "Mh", "J", "F")})
            aux.update(h0=h0.detach(), hL=hL.detach())
        return delta, stats, aux


def install(variant):
    if variant not in VARIANTS:
        raise ValueError(f"PMR S4c gate variant in {VARIANTS}")
    if XM.XYZMeshNet is not PM._NATIVE["cls"]:
        raise RuntimeError("XYZMeshNet already patched")
    XM.XYZMeshNet = type(f"PMRGateNet_{variant}", (PMRGateNet,), {"variant": "r", "gate_kind": variant})
    _STATE["variant"] = variant
    return variant


def uninstall():
    XM.XYZMeshNet = PM._NATIVE["cls"]
    _STATE["variant"] = None


def installed():
    return _STATE["variant"]


__all__ = ["GEO_IN", "PMRGateNet", "VARIANTS", "gate_features", "install", "installed", "uninstall"]
