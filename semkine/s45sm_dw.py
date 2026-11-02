r"""S45-SM depthwise patch router `dw` (docs/EXPERIMENT_RECORDS.md#s45sm-r8-prereg-20261031): S45-RT's T0P surface-patch
router (semkine/s45rt_router.py T0PNet code: event image, residual trunk + FPN, per-node MLP, 279 patch logits +
rest-template point + on-hand logit, routing to the predicted patch's face nearest to the predicted point) with every
3 x 3 convolution of the four trunk stages factorized into a depthwise 3 x 3 and a pointwise 1 x 1 (`DWBlock`), so the
trunk keeps wide channels inside the S45-SM parameter budget:

  dw   trunk 32 / 64 / 96 / 128 (depthwise-separable), FPN 40, head 72  ->  137,291 parameters
  srp  trunk 16 / 24 / 48 / 56 (full 3 x 3), FPN 32, head 56            ->  137,787 parameters (semkine/s45sm_srp.py)
  T0P  trunk 32 / 64 / 128 / 256 (full 3 x 3), FPN 128, head 256        ->  1,606,683 parameters

S45 (459,059) + dw stays within 600,000. The stem, the FPN laterals / smoothing convolution and the head are T0P's.

Opt-in. `register()` teaches the S45-SM hooks the router kind "dw" (routed exactly as "t0p", built at these widths),
wrapping whatever is installed then, so it composes with semkine/s45sm_srp.py when that is registered first;
`unregister()` restores what it wrapped. `build` / `save_checkpoint` / `load` are T0P's under the dw blocks and widths.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F

from . import s45rt_router as RT
from . import s45sm as SM
from . import s45sm_fast as FAST

KIND = "dw"
TRUNK = (32, 64, 96, 128)
FPN = 40
HEAD = 72
_STATE = {"registered": False, "prev": None}


class DWBlock(nn.Module):
    """T0P's residual block with each 3 x 3 convolution as depthwise 3 x 3 + pointwise 1 x 1 (skip unchanged)."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.dw1 = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.pw1 = nn.Conv2d(cin, cout, 1, bias=False)
        self.gn1 = nn.GroupNorm(8, cout)
        self.dw2 = nn.Conv2d(cout, cout, 3, 1, 1, groups=cout, bias=False)
        self.pw2 = nn.Conv2d(cout, cout, 1, bias=False)
        self.gn2 = nn.GroupNorm(8, cout)
        self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.GroupNorm(8, cout))

    def forward(self, x):
        h = F.relu(self.gn1(self.pw1(self.dw1(x))))
        return F.relu(self.gn2(self.pw2(self.dw2(h))) + self.skip(x))


@contextmanager
def widths():
    """T0P's module constants and block at the dw design while a net is built (T0PNet reads them in __init__)."""
    old = (RT.TRUNK, RT.FPN, RT.HEAD, RT.Block)
    RT.TRUNK, RT.FPN, RT.HEAD, RT.Block = TRUNK, FPN, HEAD, DWBlock
    try:
        yield
    finally:
        RT.TRUNK, RT.FPN, RT.HEAD, RT.Block = old


def build(npz):
    with widths():
        return RT.build(npz)


def save_checkpoint(net, path, **meta):
    RT.save_checkpoint(net, path, kind=KIND, block="dw", trunk=list(TRUNK), fpn=FPN, head=HEAD, **meta)


def load(path, map_location="cpu"):
    obj = torch.load(str(path), map_location=map_location)
    if obj.get("kind") != KIND or obj.get("block") != "dw" or \
            (tuple(obj["trunk"]), obj["fpn"], obj["head"]) != (TRUNK, FPN, HEAD):
        raise ValueError("not a dw checkpoint at the registered design: " + str(path))
    with widths():
        net, obj = RT.load(path, map_location=map_location)
    return net, obj


def register():
    if _STATE["registered"]:
        return
    prev = {"routers": SM.ROUTERS, "load": SM._load_router, "route_nodes": SM.route_nodes, "fast_route": FAST.route}

    def load_router(router, checkpoint):
        if router != KIND:
            return prev["load"](router, checkpoint)
        net, _ = load(checkpoint)
        for p in net.parameters():
            p.requires_grad_(False)
        return net.eval()

    def route_nodes(net, router, *args, **kwargs):
        return prev["route_nodes"](net, "t0p" if router == KIND else router, *args, **kwargs)

    def fast_route(net, router, out, select):
        return prev["fast_route"](net, "t0p" if router == KIND else router, out, select)
    SM.ROUTERS = prev["routers"] + (KIND,)
    SM._load_router, SM.route_nodes, FAST.route = load_router, route_nodes, fast_route
    _STATE.update(registered=True, prev=prev)


def unregister():
    prev = _STATE["prev"]
    if prev is not None:
        SM.ROUTERS, SM._load_router, SM.route_nodes, FAST.route = (prev["routers"], prev["load"], prev["route_nodes"],
                                                                   prev["fast_route"])
    _STATE.update(registered=False, prev=None)


__all__ = ["DWBlock", "FPN", "HEAD", "KIND", "TRUNK", "build", "load", "register", "save_checkpoint", "unregister",
           "widths"]
