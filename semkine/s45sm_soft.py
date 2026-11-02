r"""S45-SM soft prior-free association `spfa` (docs/EXPERIMENT_RECORDS.md#s45sm-r3-prereg-20261031). Opt-in.

PFA (semkine/s45sm.py) puts every node the router calls on-hand on its routed face with q = 1. `spfa` is the same
association with one change: q = the router's probability of its own route, p(on-hand) * p(top part) (top patch for
T0P), so an uncertain route moves the rest of the event's mass, 1 - q, to S45's residual slot (`semkine/xyz_mesh.py`:
records weigh q * A, the residual weighs 1 - q, exactly as S45's own near-surface records with q < 1). The faces,
barycentrics, depths and the eligible nodes are PFA's; no parameter, no new record column.

The router's probabilities reach the association through a one-entry stash: `predicted_sources` stores them with the
face tensor it returns, and `spfa_association` accepts them only for that same tensor object (model.py passes the
source dict's tensors straight through), then clears the stash. Built on the latency path (semkine/s45sm_fast.py).
"""
from __future__ import annotations

import torch

from . import s45sm as SM
from . import s45sm_fast as FAST
from . import xag as XAG

_STASH = {"face": None, "conf": None}
_STATE = {"on": False}


def route_confidence(router, out):
    """p(on-hand) * p(top part) (SR) or p(top patch) (T0P) per node; the route's own label is the argmax."""
    logits = out["patch"] if router == "t0p" else out["part"]
    return torch.sigmoid(out["onhand"].float()) * torch.softmax(logits.float(), -1).max(-1).values


@torch.no_grad()
def predicted_sources(batch, src, mask, *, require_depth=False):
    if require_depth:
        raise RuntimeError("S45-SM routes faces; full_oracle depth is not defined for it")
    st = SM._STATE
    net = SM._net(src.device)
    out = FAST.router_forward(net, batch.events, batch.ptr, batch.delta_t_s, src, mask)
    routed = mask & (torch.sigmoid(out["onhand"].float()) >= st["onhand_tau"])
    face, bary = FAST.route(net, st["router"], out, routed)
    conf = route_confidence(st["router"], out)
    _STASH.update(face=face, conf=torch.where(routed, conf, torch.zeros_like(conf)))
    zero = torch.zeros_like(face)
    return {"xag_source_index": zero, "xag_source_ms": zero, "xag_part": net.face_part[face], "xag_face": face,
            "xag_bary": bary.to(torch.float32), "xag_valid": routed}


@torch.no_grad()
def spfa_association(mode, assoc, d, mask, verts, faces, vert_faces, lbs_weights, band, front_tol, z_tol, *,
                     source_part, source_face, source_bary, source_valid, source_depth=None, k=16, chunk=4096):
    """PFA's records with q = the stashed route probability on the eligible nodes."""
    if _STASH["face"] is None or source_face is not _STASH["face"]:
        raise RuntimeError("spfa: the route probabilities do not belong to these routes")
    conf = _STASH["conf"]
    _STASH.update(face=None, conf=None)
    out, audit = FAST.pfa_association(mode, assoc, d, mask, verts, faces, vert_faces, lbs_weights, band, front_tol,
                                      z_tol, source_part=source_part, source_face=source_face,
                                      source_bary=source_bary, source_valid=source_valid, source_depth=source_depth,
                                      k=k, chunk=chunk)
    out["q"] = torch.where(audit["xag_eligible"], conf.to(out["q"].dtype), torch.zeros_like(out["q"]))
    return out, audit


def install(router, checkpoint, assoc, onhand_tau=0.5):
    if assoc != "spfa":
        raise ValueError("semkine.s45sm_soft installs assoc 'spfa' only")
    state = FAST.install(router, checkpoint, "pfa", onhand_tau)
    XAG.sample_sources = predicted_sources
    XAG.route_association = spfa_association
    _STASH.update(face=None, conf=None)
    _STATE["on"] = True
    return {**state, "assoc": "spfa"}


def uninstall():
    _STASH.update(face=None, conf=None)
    _STATE["on"] = False
    FAST.uninstall()


def installed():
    state = FAST.installed()
    return None if state is None or not _STATE["on"] else {**state, "assoc": "spfa"}


__all__ = ["install", "installed", "predicted_sources", "route_confidence", "spfa_association", "uninstall"]
