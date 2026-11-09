r"""PMR design D2 (docs/EXPERIMENT_RECORDS.md#pmr-s9-prereg-20261109): D1 with full-rank typed part -> hand-node edges, an
opt-in subclass of semkine/pmr_trunk.py's `PMRNet` (unchanged), installed in place of `semkine.xyz_mesh.XYZMeshNet`
(`install(variant)`).

  hubfull   in every round part j's message to the hand node adds W_j h_j (W_j: C -> C per part, zero-initialised), on top
            of D1's shared map and gate. The hand node then receives every part's state through its own map, the same
            information R-mlp's root / translation readout gets from the concatenated part states, while the joints keep
            D1's kinematic-tree propagation and per-joint local readout. The rank-8 version (S6 / S8, d1hub) changed
            nothing over three seeds; this is the full-capacity test of the same hand-node hypothesis.
The maps are built in an isolated RNG scope after D1's construction, so the shared parameters start equal to D1's at one
seed and the initial function is D1's. Gates (prior FK joint geometry, shared by the rounds), node inputs and readout are
D1's, so D1's R-noprop / R-mlp are the controls; with the messages zeroed the maps carry nothing.
"""
from __future__ import annotations

import torch
from torch import nn

from . import pmr_trunk as PM
from . import xyz_mesh as XM

VARIANTS = ("hubfull",)
_STATE = {"variant": None}


class PMRHubFullNet(PM.PMRNet):
    variant = "r"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        C = self.hidden
        with torch.random.fork_rng(devices=[]):
            self.p2h_w = nn.ModuleList([PM.NodeLinear(PM.N_PARTS, C, C, zero=True) for _ in self.rounds])

    def propagate(self, h, gate=None, messages=True):
        if not messages:
            return super().propagate(h, gate, messages)
        for r, rd in enumerate(self.rounds):
            upd = rd.slf(h)
            for k, lin in enumerate(rd.msg):
                sel = getattr(self, f"e_sel{k}")
                m = lin(h)[:, self.e_src[sel]]
                if k == 3:                                                   # part -> hand: part j's own full map
                    m = m + self.p2h_w[r](h[:, :PM.N_PARTS])[:, self.e_src[sel]]
                msg = gate[:, sel] * m
                upd = upd + torch.einsum("ev,bec->bvc", getattr(self, f"e_scatter{k}").to(msg.dtype), msg)
            h = h + torch.relu(upd)
        return h


def install(variant):
    if variant not in VARIANTS:
        raise ValueError(f"PMR D2 variant in {VARIANTS}")
    if XM.XYZMeshNet is not PM._NATIVE["cls"]:
        raise RuntimeError("XYZMeshNet already patched")
    XM.XYZMeshNet = type(f"PMRHubFullNet_{variant}", (PMRHubFullNet,), {"variant": "r"})
    _STATE["variant"] = variant
    return variant


def uninstall():
    XM.XYZMeshNet = PM._NATIVE["cls"]
    _STATE["variant"] = None


def installed():
    return _STATE["variant"]


__all__ = ["PMRHubFullNet", "VARIANTS", "install", "installed", "uninstall"]
