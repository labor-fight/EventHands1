r"""PMR stage S4 refinements of D1 (docs/EXPERIMENT_RECORDS.md#pmr-s4-prereg-20261109): opt-in subclasses of
semkine/pmr_trunk.py's `PMRNet` (unchanged), each one change against D1, installed in place of
`semkine.xyz_mesh.XYZMeshNet` (`install(variant)`).

  dual       every part node also receives its event moments in the camera frame (the 22 statistics of
  dualn      `moments` about the part's prior joint, camera axes), beside D1's moments in its parent's prior frame:
  dualm      the hand node's common-motion estimate (camera frame) and a part's own evidence then share a frame, so the
             common motion can be removed linearly, and the part's parent-frame moments carry its local increment.
             The extra statistics enter through a zero-initialised, bias-free map added to the part encoder (initial
             function = D1's), built in an isolated RNG scope (shared parameters start equal to D1's at one seed).
             dual / dualn / dualm = R / R-noprop / R-mlp with these node inputs (the screen's three arms).
  hp         typed hand-node edges: the palm <-> hand edges get their own message maps (types hub->palm, palm->hub), so
             the palm's rigid evidence reaches the hand node with weight 1 instead of 1/16 of the part->hub mean; the
             finger <-> hand edges keep D1's maps. The node inputs are D1's, so D1's R-noprop / R-mlp are its controls.
Width and rounds stay D1's (semkine/xag.py pins the configuration interfaces).
"""
from __future__ import annotations

import torch
from torch import nn

from . import pmr_trunk as PM
from . import xyz_mesh as XM

VARIANTS = ("dual", "dualn", "dualm", "hp")
BASE = {"dual": "r", "dualn": "noprop", "dualm": "mlp", "hp": "r"}
EXTRA_STATS = PM.HUB_STATS                    # 22: the camera-frame moments of a part about its joint
#: hp edge types: D1's four, then hub->palm and palm->hub
HP_TYPES = PM.EDGE_TYPES + ("hub2palm", "palm2hub")
GATE_TYPE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 2, 5: 3}
_STATE = {"variant": None}


def hp_edges():
    """D1's 62 edges with the two palm <-> hand edges retyped (4 = hub->palm, 5 = palm->hub)."""
    out = []
    for s, t, k in PM.edges():
        if k == 2 and t == 0:
            k = 4
        elif k == 3 and s == 0:
            k = 5
        out.append((s, t, k))
    return out


class PMRRefineNet(PM.PMRNet):
    variant = "r"
    refine = "dual"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        C = self.hidden
        if self.refine in ("dual", "dualn", "dualm"):
            with torch.random.fork_rng(devices=[]):
                self.extra_in = PM.NodeLinear(PM.N_PARTS, EXTRA_STATS, C)
            nn.init.zeros_(self.extra_in.weight)
            nn.init.zeros_(self.extra_in.bias)
        elif self.refine == "hp":
            E = hp_edges()
            self.register_buffer("e_type", torch.tensor([e[2] for e in E]), persistent=False)
            # the gate's edge-type input stays D1's four types (hub->palm reads as hub->part, palm->hub as part->hub)
            self.register_buffer("e_type_gate", torch.tensor([GATE_TYPE[e[2]] for e in E]), persistent=False)
            indeg = torch.zeros(len(HP_TYPES), PM.N_NODES)
            for _, t, k in E:
                indeg[k, t] += 1
            for k in range(len(HP_TYPES)):
                sel = torch.tensor([i for i, e in enumerate(E) if e[2] == k])
                S = torch.zeros(len(sel), PM.N_NODES)
                S[torch.arange(len(sel)), self.e_dst[sel]] = 1.0 / indeg[k, self.e_dst[sel]]
                self.register_buffer(f"e_sel{k}", sel, persistent=False)
                self.register_buffer(f"e_scatter{k}", S, persistent=False)
            with torch.random.fork_rng(devices=[]):
                for rd in self.rounds:
                    for _ in range(2):
                        rd.msg.append(nn.Linear(C, C, bias=False))
            with torch.no_grad():
                for rd in self.rounds:                                       # start the palm maps as D1's hub maps
                    rd.msg[4].weight.copy_(rd.msg[2].weight)
                    rd.msg[5].weight.copy_(rd.msg[3].weight)

    def node_inputs(self, tok, d, mask, verts, assoc, prev48, ablate=False):
        n = super().node_inputs(tok, d, mask, verts, assoc, prev48, ablate)
        if self.refine in ("dual", "dualn", "dualm"):
            B = n["J"].shape[0]
            with torch.autocast(device_type=tok.device.type, enabled=False):
                eye = torch.eye(3, dtype=n["J"].dtype, device=n["J"].device).expand(B, PM.N_PARTS, 3, 3)
                _, cam = PM.moments(n["omega"], n["P"], n["res"], n["J"], eye)
            n["x_extra"] = cam
        return n

    def encode(self, x_part, x_hub, x_extra=None):
        h = super().encode(x_part, x_hub)
        if x_extra is None:
            return h
        add = self.extra_in(x_extra.to(self.enc_hub.weight.dtype))
        return torch.cat([h[:, :PM.N_PARTS] + add, h[:, PM.N_PARTS:]], 1)

    def gates(self, J):
        if self.refine != "hp":
            return super().gates(J)
        g = PM.edge_features(J.to(self.edge_emb.weight.dtype), self.e_src, self.e_dst, self.e_type_gate)
        return torch.sigmoid(self.gate(torch.relu(self.edge_emb(g))))

    def propagate(self, h, gate=None, messages=True):
        if self.refine != "hp" or not messages:
            return super().propagate(h, gate, messages)
        for rd in self.rounds:
            upd = rd.slf(h)
            for k, lin in enumerate(rd.msg):
                sel = getattr(self, f"e_sel{k}")
                msg = gate[:, sel] * lin(h)[:, self.e_src[sel]]
                upd = upd + torch.einsum("ev,bec->bvc", getattr(self, f"e_scatter{k}").to(msg.dtype), msg)
            h = h + torch.relu(upd)
        return h

    def forward(self, tok, d, mask, verts, assoc, prev48, ablate: bool = False, verts_geo=None, capture: bool = False,
                h3_transform=None, messages: bool = True):
        if verts_geo is not None or h3_transform is not None or self.xag_h3_transform is not None:
            raise ValueError("PMR covers the S45 / S45-SM forward only")
        n = self.node_inputs(tok, d, mask, verts, assoc, prev48, ablate)
        h0 = self.encode(n["x_part"], n["x_hub"], n.get("x_extra"))
        if self.variant == "mlp":
            hL = h0
            delta = self.fc2(torch.relu(self.fc1(h0.reshape(h0.shape[0], -1))))
        else:
            use = self.variant == "r" and messages
            hL = self.propagate(h0, self.gates(n["J"]) if use else None, messages=use)
            delta = self.readout(hL)
        n_live = mask.float().sum()
        stats = {"xyz_valid": assoc["valid"].float().sum() / n_live.clamp_min(1.0),
                 "xyz_hit": assoc["hit"].float().sum() / n_live.clamp_min(1.0),
                 "xyz_resid_mass": n["r"].float().sum() / n_live.clamp_min(1.0),
                 "pmr_parts_live": (n["Mp"] > 0).float().sum(1).mean()}
        aux = {"w": n["w"], "r": n["r"]}
        if capture or self.xag_capture_enabled:
            aux.update({k: n[k].detach() for k in ("x_part", "x_hub", "Mp", "Mh", "J", "F")})
            if "x_extra" in n:
                aux["x_extra"] = n["x_extra"].detach()
            aux.update(h0=h0.detach(), hL=hL.detach())
        return delta, stats, aux


def install(variant):
    if variant not in VARIANTS:
        raise ValueError(f"PMR refinement in {VARIANTS}")
    if XM.XYZMeshNet is not PM._NATIVE["cls"]:
        raise RuntimeError("XYZMeshNet already patched")
    XM.XYZMeshNet = type(f"PMRRefineNet_{variant}", (PMRRefineNet,), {"variant": BASE[variant], "refine": variant})
    _STATE["variant"] = variant
    return variant


def uninstall():
    XM.XYZMeshNet = PM._NATIVE["cls"]
    _STATE["variant"] = None


def installed():
    return _STATE["variant"]


__all__ = ["BASE", "HP_TYPES", "PMRRefineNet", "VARIANTS", "hp_edges", "install", "installed", "uninstall"]
