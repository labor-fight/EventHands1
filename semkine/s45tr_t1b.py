r"""S45-TR arm T1b: correspondence routing by the frozen T0 predictor (docs/EXPERIMENT_RECORDS.md#s45tr-prereg-20261030).

With `MODEL.XAG_ASSOC: face_route`, S45's `_xyz_mesh_forward` asks `semkine.xag.sample_sources` for each sampled
event's source face / barycentrics and `semkine.xag.route_association` swaps them into the events the ray association
already matched (q > 0, valid), keeping each event's association mass, q and z. T1d answers with the XAG ground truth.
T1b answers with the frozen T0 network instead: face = T0's argmax face, bary = T0's template point projected onto
that face, and an event is eligible only when T0's on-hand probability is at least tau (0.5); every other event keeps
S45's association. No ground-truth source field is read; T0's weights are not model parameters (not in the state
dict, no gradient), so the pose network trains with S45's loss alone.

T0 runs under the caller's autocast, on the batch's device, from the batch's events only. Its node positions are
checked against the `src` / `mask` S45 passes in, so the routes are attached to the right events.

Opt-in only: `install(checkpoint, tau)` replaces `semkine.xag.sample_sources` for the process.
"""
from __future__ import annotations

import torch

from . import xag as XAG

_ORIGINAL = XAG.sample_sources
_STATE = {"net": None, "tau": None, "checkpoint": None, "on": {}}


@torch.no_grad()
def predicted_sources(batch, src, mask, *, require_depth=False):
    if require_depth:
        raise RuntimeError("T1b routes faces only; full_oracle depth is not defined for it")
    dev = src.device
    net = _STATE["on"].get(dev)
    if net is None:
        net = _STATE["net"].to(dev)
        _STATE["on"][dev] = net
    out = net(batch.events, batch.ptr, batch.delta_t_s)
    if out["src"].shape != src.shape or not torch.equal(out["src"][mask], src[mask]) or not torch.equal(out["mask"], mask):
        raise RuntimeError("T0 nodes are not S45's sampled events")
    prob = torch.sigmoid(out["onhand"].float())
    valid = mask & (prob >= _STATE["tau"])
    face, bary, _ = net.correspondence(out, valid)
    zero = torch.zeros_like(face)
    return {"xag_source_index": zero, "xag_source_ms": zero, "xag_part": out["part"].argmax(-1),
            "xag_face": face, "xag_bary": bary.to(torch.float32), "xag_valid": valid}


def install(checkpoint, tau: float = 0.5) -> None:
    from .s45tr_t0 import load
    net, _ = load(checkpoint)
    for p in net.parameters():
        p.requires_grad_(False)
    _STATE.update(net=net, tau=float(tau), checkpoint=str(checkpoint), on={})
    XAG.sample_sources = predicted_sources


def uninstall() -> None:
    XAG.sample_sources = _ORIGINAL
    _STATE.update(net=None, tau=None, checkpoint=None, on={})


def installed():
    if XAG.sample_sources is not predicted_sources:
        return None
    return {"checkpoint": _STATE["checkpoint"], "tau": _STATE["tau"]}


__all__ = ["install", "installed", "predicted_sources", "uninstall"]
