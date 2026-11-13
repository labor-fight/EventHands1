r"""ABL trunk variant (docs/EXPERIMENT_RECORDS.md#abl-prereg-20261111): D1 R without the event-prior geometric residual, an
opt-in subclass of semkine/pmr_trunk.py's `PMRNet` (unchanged), installed in place of `semkine.xyz_mesh.XYZMeshNet`.

  rnores   every event keeps its correspondence (part weights and the routed point P on the prior mesh) but its residual
           res = z d - P to the prior surface is set to zero: the node moments lose mean res and lever x res, and phi reads
           zeros in place of res / L_RES. The routed depth, the tokens, the gates and the readout are D1 R's.
No parameter is added or removed, so at one seed the network starts equal to D1 R.
"""
from __future__ import annotations

import torch

from . import pmr_trunk as PM
from . import xyz_mesh as XM

VARIANTS = ("rnores",)
_STATE = {"variant": None}


class PMRNoResNet(PM.PMRNet):
    variant = "r"

    def geometry(self, mask, d, verts, assoc, prev48, ablate, acc):
        g = super().geometry(mask, d, verts, assoc, prev48, ablate, acc)
        g["res"] = torch.zeros_like(g["res"])
        return g


def install(variant):
    if variant not in VARIANTS:
        raise ValueError(f"ABL trunk variant in {VARIANTS}")
    if XM.XYZMeshNet is not PM._NATIVE["cls"]:
        raise RuntimeError("XYZMeshNet already patched")
    XM.XYZMeshNet = type(f"PMRNoResNet_{variant}", (PMRNoResNet,), {"variant": "r"})
    _STATE["variant"] = variant
    return variant


def uninstall():
    XM.XYZMeshNet = PM._NATIVE["cls"]
    _STATE["variant"] = None


__all__ = ["PMRNoResNet", "VARIANTS", "install", "uninstall"]
