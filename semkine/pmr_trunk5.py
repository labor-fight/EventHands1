r"""PMR stage S4b refinement of D1 (docs/EXPERIMENT_RECORDS.md#pmr-s4b-prereg-20261109): a position-aware event encoder,
opt-in subclass of semkine/pmr_trunk.py's `PMRNet` (unchanged), installed in place of `semkine.xyz_mesh.XYZMeshNet`.

  pos / posn / posm   phi also reads where the event lies on its part: the lever of its routed point about the prior
                      joint of its dominant part p* = argmax_p omega_ip, in that part's prior frame, / L_LEV (zero for an
                      unmatched event). The part channels stay sums over events (event-additive); now they can carry the
                      residual's spatial structure on the part nonlinearly (B2's vertex records see the offset to their
                      vertex; D1's learned channels saw only the residual, the position entered through the fixed
                      moments). The lever enters phi's first layer through a zero-initialised, bias-free map built in an
                      isolated RNG scope (initial function = D1's; shared parameters start equal at one seed).
                      pos / posn / posm = R / R-noprop / R-mlp with these node inputs.
"""
from __future__ import annotations

import torch
from torch import nn

from . import pmr_trunk as PM
from . import xyz_mesh as XM

VARIANTS = ("pos", "posn", "posm")
BASE = {"pos": "r", "posn": "noprop", "posm": "mlp"}
_STATE = {"variant": None}


class PMRPosNet(PM.PMRNet):
    variant = "r"
    refine = "pos"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rel_hidden = self.phi[0].out_features
        with torch.random.fork_rng(devices=[]):
            self.lev_in = nn.Linear(3, rel_hidden, bias=False)
        nn.init.zeros_(self.lev_in.weight)

    def _phi(self, f, lev):
        wdt = self.phi[0].weight.dtype
        return self.phi[2](self.phi[1](self.phi[0](f.to(wdt)) + self.lev_in(lev.to(wdt))))

    def node_inputs(self, tok, d, mask, verts, assoc, prev48, ablate=False):
        B, N, _ = tok.shape
        dev = tok.device
        acc = torch.float32 if self.training else torch.float64
        with torch.autocast(device_type=dev.type, enabled=False):
            g = self.geometry(mask, d, verts, assoc, prev48, ablate, acc)
            Mp, sp, Mh, sh = self.statistics(g)
            on = (g["m"] > 0).unsqueeze(-1)
            pstar = g["omega"].argmax(-1)                                              # (B, N) dominant part
            bidx = torch.arange(B, device=dev).view(B, 1)
            Jp = g["J"][bidx, pstar]                                                   # (B, N, 3)
            Fp = g["F"][bidx, pstar]                                                   # (B, N, 3, 3)
            lev = torch.einsum("bnji,bnj->bni", Fp, g["P"] - Jp) / PM.L_LEV
            lev = torch.where(on, lev, torch.zeros_like(lev)).float()
            onf = on.float()
            zn = ((assoc["z"].float() - XM.Z_REF) / XM.Z_SCALE).unsqueeze(-1)
            f_on = torch.cat([tok.float(), g["res"].float() / PM.L_RES, zn * onf, onf], -1)
            f_off = torch.cat([tok.float(), torch.zeros(B, N, 5, device=dev)], -1)
        phi_on = self._phi(f_on, lev)                                                  # ambient autocast
        phi_off = self._phi(f_off, torch.zeros_like(lev))
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
        return {"x_part": x_part, "x_hub": x_hub, "Mp": Mp, "Mh": Mh, "sp": sp, "sh": sh, "lev": lev, **g}


def install(variant):
    if variant not in VARIANTS:
        raise ValueError(f"PMR S4b refinement in {VARIANTS}")
    if XM.XYZMeshNet is not PM._NATIVE["cls"]:
        raise RuntimeError("XYZMeshNet already patched")
    XM.XYZMeshNet = type(f"PMRPosNet_{variant}", (PMRPosNet,), {"variant": BASE[variant], "refine": variant})
    _STATE["variant"] = variant
    return variant


def uninstall():
    XM.XYZMeshNet = PM._NATIVE["cls"]
    _STATE["variant"] = None


def installed():
    return _STATE["variant"]


__all__ = ["BASE", "PMRPosNet", "VARIANTS", "install", "installed", "uninstall"]
