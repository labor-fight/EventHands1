r"""PMR stage S7 combinations of ghand with the hand-node refinements (docs/EXPERIMENT_RECORDS.md#pmr-s7-prereg-20261109):
classes composed from the frozen refinement subclasses, installed in place of `semkine.xyz_mesh.XYZMeshNet`
(`install(variant)`).

  ghandhp   ghand's hand-frame gates (semkine/pmr_trunk6.py) + hp's typed palm <-> hand edges (semkine/pmr_trunk4.py)
  ghandhub  ghand's hand-frame gates + hub's per-part low-rank hand-node adapters (semkine/pmr_trunk7.py)
Python's method resolution composes them: ghand overrides forward (gates from the hand-frame geometry) and replaces the
edge embedding; hp overrides the edge types and propagate; hub overrides propagate and adds its adapters. Each adds its
modules in its own isolated RNG scope, so the shared parameters start equal to D1's at one seed and the initial function is
D1's. The one join: hp retypes the palm <-> hand edges (types 4 / 5), and ghand's gate one-hot reads D1's four types, so the
gates read hp's own gate types (hub->palm as hub->part, palm->hub as part->hub), exactly as hp's own gates do.
Both keep D1's node inputs and readout, so D1's R-noprop / R-mlp are the controls.
"""
from __future__ import annotations

import torch

from . import pmr_trunk as PM
from . import pmr_trunk4 as P4
from . import pmr_trunk6 as P6
from . import pmr_trunk7 as P7
from . import xyz_mesh as XM

VARIANTS = ("ghandhp", "ghandhub")
_STATE = {"variant": None}


class PMRGateHPNet(P6.PMRGateNet, P4.PMRRefineNet):
    variant = "r"
    gate_kind = "ghand"
    refine = "hp"

    def gates_from(self, J, F):
        dt = self.edge_emb.weight.dtype
        g = P6.gate_features(self.gate_kind, J.to(dt), F.to(dt), self.e_src, self.e_dst, self.e_type_gate)
        return torch.sigmoid(self.gate(torch.relu(self.edge_emb(g))))


class PMRGateHubNet(P6.PMRGateNet, P7.PMRHubNet):
    variant = "r"
    gate_kind = "ghand"


CLASSES = {"ghandhp": PMRGateHPNet, "ghandhub": PMRGateHubNet}


def install(variant):
    if variant not in VARIANTS:
        raise ValueError(f"PMR S7 combination in {VARIANTS}")
    if XM.XYZMeshNet is not PM._NATIVE["cls"]:
        raise RuntimeError("XYZMeshNet already patched")
    XM.XYZMeshNet = type(f"PMRCombo_{variant}", (CLASSES[variant],), {})
    _STATE["variant"] = variant
    return variant


def uninstall():
    XM.XYZMeshNet = PM._NATIVE["cls"]
    _STATE["variant"] = None


def installed():
    return _STATE["variant"]


__all__ = ["CLASSES", "PMRGateHPNet", "PMRGateHubNet", "VARIANTS", "install", "installed", "uninstall"]
