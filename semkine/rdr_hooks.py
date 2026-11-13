"""RDR routers behind the PFO router kind "pfr" (docs/EXPERIMENT_RECORDS.md#rdr-prereg-20261110): after `enable()`, a
checkpoint written by semkine/rdr_router.py loads wherever a "pfr" checkpoint is expected (semkine/pfo_router.py `load`
dispatches on the checkpoint's own kind), and the fast evaluation path (semkine/s45sm_fast.py) calls the RDR network's
own forward / decoding. The training-time path (semkine/s45sm.py `route_nodes`) already calls `net(...)` and
`net.route(...)`, which the RDR network implements. Opt-in; a "pfr" checkpoint behaves exactly as before.
"""
from __future__ import annotations

import torch

from . import pfo_router as PR
from . import rdr_router as RD
from . import s45sm_fast as FAST

_STATE = {"on": False}


def enable():
    if _STATE["on"]:
        return
    native_load, native_forward, native_route = PR.load, FAST.router_forward, FAST.route

    def load(path, map_location="cpu"):
        obj = torch.load(str(path), map_location="cpu")
        if obj.get("kind") == RD.KIND:
            return RD.load(path, map_location=map_location)
        return native_load(path, map_location=map_location)

    def router_forward(net, events, ptr, delta_t_s, src, mask):
        if isinstance(net, RD.RDRNet):
            return net.forward_at(events, ptr, delta_t_s, src, mask)
        return native_forward(net, events, ptr, delta_t_s, src, mask)

    def route(net, router, out, select):
        if isinstance(net, RD.RDRNet):
            face, bary = net.route(out, select)
            return torch.where(select, face, torch.zeros_like(face)), bary
        return native_route(net, router, out, select)

    PR.load, FAST.router_forward, FAST.route = load, router_forward, route
    _STATE["on"] = True


__all__ = ["enable"]
