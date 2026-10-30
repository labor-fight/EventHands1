"""MRF probe downstream (docs/EXPERIMENT_RECORDS.md#mrf-prereg-20261024 Stage B).

`MRFTail` is the AGG778 R2 probe tail (`tools/agg778/heads.Tail`: input projection without bias, module G, the LBS
slots, the 17-group `slot_proj`, `fc1(+prev48)`, zero-initialised `fc2`) with module G replaced by one variant of
`semkine.mrf.run_g`. The common parameters are built first in Tail's order, so under one seed every variant starts from
the same values for them (`legacy` is `Tail` exactly); the extra blocks of `global` / `global_loc` are built last in a
CPU fork, so these two start from the same values as each other. Probe names map to variants:

    N0d self3   L3 legacy   L3np noprop   K12np ring4np   K12 ring4   Gg global   Gl global_loc
    N0 none     Dfix diffuse   Lg0 nogeo
"""
from __future__ import annotations

import torch
from torch import nn

from semkine.mrf import BallIndex, LocalResidual, local_keys, run_g
from semkine.xag import GlobalResidual
from semkine.xyz_mesh import lbs_slot_pool
from tools.agg778.heads import Tail

G_OF = {"N0d": "self3", "L3": "legacy", "L3np": "noprop", "K12np": "ring4np", "K12": "ring4", "Gg": "global",
        "Gl": "global_loc", "N0": "none", "Dfix": "diffuse", "Lg0": "nogeo"}


class MRFTail(Tail):
    def __init__(self, f_in: int, xyz, variant: str = "legacy", faces=None, **kw):
        super().__init__(f_in, xyz, **kw)
        if variant not in G_OF.values():
            raise ValueError(f"unknown receptive-field variant {variant!r}")
        self.variant = variant
        with torch.random.fork_rng(devices=[]):
            if variant == "global":
                self.blocks = nn.ModuleList([GlobalResidual() for _ in self.layers])
            elif variant == "global_loc":
                self.blocks = nn.ModuleList([LocalResidual() for _ in self.layers])
        if variant == "global_loc":
            keys, kmask = local_keys(self.ring_idx, self.ring_mask)
            self.register_buffer("keys", keys, persistent=False)
            self.register_buffer("kmask", kmask, persistent=False)
        if variant in ("ring4", "ring4np"):
            if faces is None:
                raise ValueError("the ball variants need the mesh faces")
            self.ball = BallIndex(torch.as_tensor(faces).long().cpu(), int(self.lbs_w.shape[0]))

    def forward(self, feat, O, edge_g, bslot, prev48, verts=None):
        B = feat.shape[0]
        H = self.inp(feat) * O.unsqueeze(-1)
        H, S = run_g(self.variant, H, O, layers=self.layers, edge=self.edge, edge_g=edge_g, verts=verts,
                     ring_idx=self.ring_idx, ring_mask=self.ring_mask, lbs_w=self.lbs_w,
                     ball=getattr(self, "ball", None), blocks=getattr(self, "blocks", None),
                     keys=getattr(self, "keys", None), kmask=getattr(self, "kmask", None))
        e = lbs_slot_pool(H, S, O, self.lbs_w, self.lbs_hard)
        slots = torch.cat([e.float(), bslot.unsqueeze(1).float()], 1)
        zs = torch.relu(self.slot_proj(slots.reshape(B, -1, 1))).view(B, -1)
        h = torch.relu(self.fc1(torch.cat([zs, prev48.float()], -1)))
        return self.fc2(h)
