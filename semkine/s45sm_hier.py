r"""S45-SM part-first decoding of the surface-patch routers (T0P and its slim versions srp / dw).

A patch router scores 279 surface patches that never cross the 16 MANO parts and routes a node to the most probable
patch. Here the node's part is decided first, by the part with the largest summed patch probability, and the patch
is then the most probable patch inside that part; the face is looked up in that patch as before. Implemented as a
wrapper of the patch head (`PartFirst`: logits outside the chosen part become -inf), so every path that calls
`net.patch` (the router's own forward and route, semkine/s45sm_fast.py, semkine/s45sm_ctx.py) decodes part-first.
No parameter changes; the weights are the base router's.

Opt-in. `register()` teaches the S45-SM hooks the router kinds in KINDS (built and loaded as their base kind, routed
exactly as "t0p", with the part-first head), wrapping whatever is installed then, so it composes with
semkine/s45sm_srp.py and semkine/s45sm_dw.py registered first; `unregister()` restores what it wrapped.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from . import s45sm as SM
from . import s45sm_fast as FAST

KINDS = {"srph": "srp", "dwh": "dw", "t0ph": "t0p"}
_STATE = {"registered": False, "prev": None}


def patch_part(net):
    """The part of each patch (all faces of a patch share one part)."""
    part = net.face_part[net.patch_faces]                                       # (P, M)
    first = part[:, 0]
    if not bool(((part == first.unsqueeze(1)) | ~net.patch_faces_valid).all()):
        raise ValueError("a patch crosses parts")
    return first


class PartFirst(nn.Module):
    """The patch head with logits outside the most probable part (summed patch probability) set to -inf."""

    def __init__(self, linear, part_of_patch, n_parts):
        super().__init__()
        self.linear = linear
        self.register_buffer("part_of_patch", part_of_patch.long().clone(), persistent=False)
        self.register_buffer("onehot", F.one_hot(part_of_patch.long(), n_parts).float(), persistent=False)

    def forward(self, z):
        logits = self.linear(z)
        prob = torch.softmax(logits.float(), -1)
        part = (prob @ self.onehot).argmax(-1, keepdim=True)                    # (..., 1)
        return logits.masked_fill(self.part_of_patch != part, float("-inf"))


def part_first(net):
    """`net` with its patch head decoded part-first (in place; the parameters are unchanged)."""
    if isinstance(net.patch, PartFirst):
        return net
    net.patch = PartFirst(net.patch, patch_part(net), int(net.face_part.max()) + 1).to(net.patch.weight.device)
    return net


def register():
    if _STATE["registered"]:
        return
    prev = {"routers": SM.ROUTERS, "load": SM._load_router, "route_nodes": SM.route_nodes, "fast_route": FAST.route}

    def load_router(router, checkpoint):
        if router not in KINDS:
            return prev["load"](router, checkpoint)
        return part_first(prev["load"](KINDS[router], checkpoint))

    def route_nodes(net, router, *args, **kwargs):
        return prev["route_nodes"](net, "t0p" if router in KINDS else router, *args, **kwargs)

    def fast_route(net, router, out, select):
        return prev["fast_route"](net, "t0p" if router in KINDS else router, out, select)
    SM.ROUTERS = prev["routers"] + tuple(KINDS)
    SM._load_router, SM.route_nodes, FAST.route = load_router, route_nodes, fast_route
    _STATE.update(registered=True, prev=prev)


def unregister():
    prev = _STATE["prev"]
    if prev is not None:
        SM.ROUTERS, SM._load_router, SM.route_nodes, FAST.route = (prev["routers"], prev["load"], prev["route_nodes"],
                                                                   prev["fast_route"])
    _STATE.update(registered=False, prev=None)


__all__ = ["KINDS", "PartFirst", "part_first", "patch_part", "register", "unregister"]
