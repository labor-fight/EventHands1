r"""S45-SM slim patch router `srp`: S45-RT's T0P surface-patch router (semkine/s45rt_router.py T0PNet, unchanged code:
event image, residual trunk + FPN, per-node MLP, 279 patch logits + rest-template point + on-hand logit, routing to the
predicted patch's face nearest to the predicted point) at widths that fit the S45-SM parameter budget:

  trunk 16 / 24 / 48 / 56, FPN 32, head 56  ->  137,787 parameters (T0P 32 / 64 / 128 / 256, 128, 256: 1,606,683)

so that S45 (459,059) + srp stays within 600,000. The SR router (semkine/s45sm.py) has the same trunk family but
predicts 16 parts and finds the face from its regressed point inside the whole part; srp keeps T0P's patch head.

Opt-in. `register()` teaches the frozen S45-SM hooks the router kind "srp" (routed exactly as "t0p", built at these
widths); entries that run srp arms call it once before installing a hook (tools/s45sm/srp_run.py), `unregister()`
restores the frozen functions. `build` / `save_checkpoint` / `load` are T0P's under the srp widths.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

from . import s45rt_router as RT
from . import s45sm as SM
from . import s45sm_fast as FAST

KIND = "srp"
TRUNK = (16, 24, 48, 56)
FPN = 32
HEAD = 56
_NATIVE = {"routers": SM.ROUTERS, "load": SM._load_router, "route_nodes": SM.route_nodes, "fast_route": FAST.route}
_STATE = {"registered": False}


@contextmanager
def widths():
    """T0P's module constants at the srp widths while a net is built (T0PNet reads them in __init__)."""
    old = (RT.TRUNK, RT.FPN, RT.HEAD)
    RT.TRUNK, RT.FPN, RT.HEAD = TRUNK, FPN, HEAD
    try:
        yield
    finally:
        RT.TRUNK, RT.FPN, RT.HEAD = old


def build(npz):
    with widths():
        return RT.build(npz)


def save_checkpoint(net, path, **meta):
    RT.save_checkpoint(net, path, kind=KIND, trunk=list(TRUNK), fpn=FPN, head=HEAD, **meta)


def load(path, map_location="cpu"):
    obj = torch.load(str(path), map_location=map_location)
    if obj.get("kind") != KIND or (tuple(obj["trunk"]), obj["fpn"], obj["head"]) != (TRUNK, FPN, HEAD):
        raise ValueError("not an srp checkpoint at the registered widths: " + str(path))
    with widths():
        net, obj = RT.load(path, map_location=map_location)
    return net, obj


def _load_router(router, checkpoint):
    if router != KIND:
        return _NATIVE["load"](router, checkpoint)
    net, _ = load(checkpoint)
    for p in net.parameters():
        p.requires_grad_(False)
    return net.eval()


def _as_t0p(router):
    return "t0p" if router == KIND else router


def _route_nodes(net, router, *args, **kwargs):
    return _NATIVE["route_nodes"](net, _as_t0p(router), *args, **kwargs)


def _fast_route(net, router, out, select):
    return _NATIVE["fast_route"](net, _as_t0p(router), out, select)


def register():
    """Idempotent: the kind "srp" in the frozen S45-SM hooks (install, router loading, both routing paths)."""
    if _STATE["registered"]:
        return
    if SM.ROUTERS is not _NATIVE["routers"] or SM._load_router is not _NATIVE["load"] \
            or SM.route_nodes is not _NATIVE["route_nodes"] or FAST.route is not _NATIVE["fast_route"]:
        raise RuntimeError("S45-SM router functions already patched")
    SM.ROUTERS = _NATIVE["routers"] + (KIND,)
    SM._load_router = _load_router
    SM.route_nodes = _route_nodes
    FAST.route = _fast_route
    _STATE["registered"] = True


def unregister():
    SM.ROUTERS = _NATIVE["routers"]
    SM._load_router = _NATIVE["load"]
    SM.route_nodes = _NATIVE["route_nodes"]
    FAST.route = _NATIVE["fast_route"]
    _STATE["registered"] = False


__all__ = ["FPN", "HEAD", "KIND", "TRUNK", "build", "load", "register", "save_checkpoint", "unregister", "widths"]
