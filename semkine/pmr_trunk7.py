r"""PMR stage S6 refinement of D1's hand-node edges (docs/EXPERIMENT_RECORDS.md#pmr-s6-prereg-20261109): an opt-in
subclass of semkine/pmr_trunk.py's `PMRNet` (unchanged), installed in place of `semkine.xyz_mesh.XYZMeshNet`
(`install(variant)`).

  hub   per-part low-rank adapters on the hand-node edges. D1 sends every part -> hand message through one shared map and
        averages the 16 (each part counts 1 / 16), and the hand node answers every part with one message. Here, in every
        round, part j's message to the hand node adds B_j A h_j and the hand node's message to part j adds B'_j A' h_hub
        (A, A': C -> 8 shared; B_j, B'_j: 8 -> C per part, zero-initialised). The hand node can then weigh each part's
        evidence on its own (palm and finger evidence of translation and root rotation) and send each part its own share
        of the common motion. The S4 refinement hp typed the palm <-> hand edges and corrected translation better
        (zgz_global wrist 47 -> 26 mm); this extends it to every part at a low rank.
The adapters are built in an isolated RNG scope after D1's construction, so the shared parameters start equal to D1's at
one seed and the initial function is D1's. The gates (prior FK joint geometry, shared by the rounds) and the node inputs
are D1's, so D1's R-noprop / R-mlp are the controls; with the messages zeroed the adapters carry nothing.
"""
from __future__ import annotations

import torch
from torch import nn

from . import pmr_trunk as PM
from . import xyz_mesh as XM

VARIANTS = ("hub",)
RANK = 8
_STATE = {"variant": None}


class PMRHubNet(PM.PMRNet):
    variant = "r"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        C = self.hidden
        with torch.random.fork_rng(devices=[]):
            self.p2h_a = nn.ModuleList([nn.Linear(C, RANK, bias=False) for _ in self.rounds])
            self.p2h_b = nn.ModuleList([PM.NodeLinear(PM.N_PARTS, RANK, C, zero=True) for _ in self.rounds])
            self.h2p_a = nn.ModuleList([nn.Linear(C, RANK, bias=False) for _ in self.rounds])
            self.h2p_b = nn.ModuleList([PM.NodeLinear(PM.N_PARTS, RANK, C, zero=True) for _ in self.rounds])

    def propagate(self, h, gate=None, messages=True):
        if not messages:
            return super().propagate(h, gate, messages)
        P = PM.N_PARTS
        for r, rd in enumerate(self.rounds):
            upd = rd.slf(h)
            for k, lin in enumerate(rd.msg):
                sel = getattr(self, f"e_sel{k}")
                m = lin(h)[:, self.e_src[sel]]
                if k == 3:                                                   # part -> hand: part j's own adapter
                    m = m + self.p2h_b[r](self.p2h_a[r](h[:, :P]))[:, self.e_src[sel]]
                elif k == 2:                                                 # hand -> part: one adapter per target
                    a = self.h2p_a[r](h[:, PM.HUB]).unsqueeze(1).expand(-1, P, -1)
                    m = m + self.h2p_b[r](a)[:, self.e_dst[sel]]
                msg = gate[:, sel] * m
                upd = upd + torch.einsum("ev,bec->bvc", getattr(self, f"e_scatter{k}").to(msg.dtype), msg)
            h = h + torch.relu(upd)
        return h


def install(variant):
    if variant not in VARIANTS:
        raise ValueError(f"PMR S6 variant in {VARIANTS}")
    if XM.XYZMeshNet is not PM._NATIVE["cls"]:
        raise RuntimeError("XYZMeshNet already patched")
    XM.XYZMeshNet = type(f"PMRHubNet_{variant}", (PMRHubNet,), {"variant": "r"})
    _STATE["variant"] = variant
    return variant


def uninstall():
    XM.XYZMeshNet = PM._NATIVE["cls"]
    _STATE["variant"] = None


def installed():
    return _STATE["variant"]


__all__ = ["PMRHubNet", "RANK", "VARIANTS", "install", "installed", "uninstall"]
