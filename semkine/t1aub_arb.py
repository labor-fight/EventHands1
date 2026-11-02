r"""T1A-UB round 2 (docs/EXPERIMENT_RECORDS.md#t1aub-r2-prereg-20261102): state-arbitrated part decoding of a patch router
under PFA. Opt-in; `install(sigma_px)` wraps whatever routes are installed (semkine/s45sm.py or s45sm_fast.py with a
patch router of kind t0p / dw / srp and PFA); `uninstall()` restores them.

The router alone proposes which parts an event may come from; the previous state only arbitrates between the router's two
most probable parts. For every routed node:

    p(c)     = the router's part probability (its patch softmax summed over the part's patches)
    c1, c2   = the two most probable parts; each candidate's patch = its most probable patch inside the part, its face =
               the patch's face nearest to the predicted rest-template point, its barycentrics = that point projected
               onto the face (the router's own rule, restricted to the candidate's part)
    d_c      = image distance between the event's ray and the candidate surface point placed on the previous-state mesh
               (in render-frame pixels: ray units x the render focal length)
    score_c  = log p(c) - d_c^2 / (2 sigma^2),  the routed candidate = argmax score

sigma = inf (`install(None)`) is part-first decoding (semkine/s45sm_hier.py): the router alone. `install(mode="ray")`
arbitrates with the state's 3D ray association instead: S45's ray / mesh association (semkine.xyz_mesh.associate_rays,
native, on the previous-state mesh) names a part per event (the part of its strongest associated vertex); when that part is
one of the router's two candidates it is chosen, otherwise the router's first part. In both modes the chosen part's surface
point is the router's (its face / barycentrics for that part), never the ray hit. Arbitration can only move
an event between two parts the router supports; it never places evidence where the router puts none, and the residual
geometry (dXYZ = event ray - the chosen point on the previous mesh) is PFA's. `last()` returns the fraction of routed
nodes whose part the state changed in the last forward (descriptive).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import s45sm as SM
from . import s45sm_fast as FAST
from . import t1aub as T1AUB
from . import xag as XAG
from .s45tr_t0 import project_bary

N_PARTS = 16
MODES = ("point", "ray")
_STATE = {"on": False, "mode": "point", "sigma_px": None, "band_px": 16.0, "prev": None, "stash": None, "tables": {},
          "last": None}


def _tables(net, device):
    key = (id(net), str(device))
    t = _STATE["tables"].get(key)
    if t is None:
        part = net.face_part[net.patch_faces]                                         # (P, M)
        first = part[:, 0]
        if not bool(((part == first.unsqueeze(1)) | ~net.patch_faces_valid).all()):
            raise ValueError("a patch crosses parts")
        t = {"patch_part": first.to(device), "onehot": F.one_hot(first, N_PARTS).float().to(device),
             "cent": net.tmpl[net.faces].mean(1).to(device)}
        _STATE["tables"][key] = t
    return t


@torch.no_grad()
def candidates(net, out):
    """Top-2 parts per node: (log p (B, N, 2), parts (B, N, 2), [face_1, face_2], [bary_1, bary_2])."""
    t = _tables(net, out["patch"].device)
    prob = torch.softmax(out["patch"].float(), -1)                                    # (B, N, P)
    top = (prob @ t["onehot"]).topk(2, dim=-1)
    T1AUB._add_macs(prob.numel() * N_PARTS)                                          # the part-probability product
    point = out["coord"].float() * 0.01
    faces, barys = [], []
    for j in range(2):
        inside = t["patch_part"].view(1, 1, -1) == top.indices[..., j:j + 1]
        patch = prob.masked_fill(~inside, -1.0).argmax(-1)                            # most probable patch in the part
        cand, ok = net.patch_faces[patch], net.patch_faces_valid[patch]               # (B, N, M)
        dist = (t["cent"][cand] - point.unsqueeze(-2)).norm(dim=-1).masked_fill(~ok, float("inf"))
        face = cand.gather(-1, dist.argmin(-1, keepdim=True)).squeeze(-1)
        faces.append(face)
        barys.append(project_bary(point, net.tmpl[net.faces[face]]))
    return top.values.clamp_min(1e-12).log(), top.indices, faces, barys


@torch.no_grad()
def arb_sources(batch, src, mask, *, require_depth=False):
    if require_depth:
        raise RuntimeError("arbitrated routes are face routes")
    net = SM._net(src.device)
    st = SM._STATE
    if FAST._STATE["on"]:
        out = FAST.router_forward(net, batch.events, batch.ptr, batch.delta_t_s, src, mask)
    else:
        out = net(batch.events, batch.ptr, batch.delta_t_s)
        if out["src"].shape != src.shape or not torch.equal(out["src"][mask], src[mask]) or not torch.equal(out["mask"], mask):
            raise RuntimeError("router nodes are not S45's sampled events")
    routed = out["mask"] & (torch.sigmoid(out["onhand"].float()) >= st["onhand_tau"]) & mask
    lp, parts, faces, barys = candidates(net, out)
    _STATE["stash"] = {"lp": lp, "face2": faces[1], "bary2": barys[1], "part": parts}
    zero = torch.zeros_like(faces[0])
    return {"xag_source_index": zero, "xag_source_ms": zero, "xag_part": parts[..., 0], "xag_face": faces[0],
            "xag_bary": barys[0].to(torch.float32), "xag_valid": routed}


def _reprojection(verts, faces, face, bary):
    B = verts.shape[0]
    tri = verts.float()[torch.arange(B, device=verts.device).view(B, 1, 1), faces.long()[face]]   # (B, N, 3, 3)
    X = (bary.float().unsqueeze(-1) * tri).sum(-2)
    return X[..., :2] / X[..., 2:3].clamp_min(SM.Z_MIN)


@torch.no_grad()
def arb_association(mode, assoc, d, mask, verts, faces, vert_faces, lbs_weights, band, front_tol, z_tol, *,
                    source_part, source_face, source_bary, source_valid, source_depth=None, k=16, chunk=4096):
    st = _STATE["stash"]
    _STATE["stash"] = None
    if st is None or st["face2"].shape != source_face.shape:
        raise RuntimeError("arbitration needs this forward's router candidates")
    face, bary = source_face, source_bary
    sigma = _STATE["sigma_px"]
    live = source_valid & mask
    if _STATE["mode"] == "ray":
        asc = SM._NATIVE["associate_rays"](d, live, verts, faces, vert_faces, band, front_tol, z_tol, k=k)
        best = asc["vid"].gather(-1, asc["A"].argmax(-1, keepdim=True)).squeeze(-1)
        ray_part = lbs_weights.argmax(-1)[best]
        pick = live & asc["valid"] & (ray_part == st["part"][..., 1])
        face = torch.where(pick, st["face2"], source_face)
        bary = torch.where(pick.unsqueeze(-1), st["bary2"].to(source_bary.dtype), source_bary)
        n = live.sum().clamp_min(1)
        _STATE["last"] = {"switched": float((pick & live).sum() / n), "routed": int(live.sum()),
                          "ray_in_top2": float((live & asc["valid"] & ((ray_part == st["part"][..., 0])
                                                                      | (ray_part == st["part"][..., 1]))).sum() / n)}
    elif sigma is not None:
        e1 = (_reprojection(verts, faces, source_face, source_bary) - d[..., :2].float()).norm(dim=-1)
        e2 = (_reprojection(verts, faces, st["face2"], st["bary2"]) - d[..., :2].float()).norm(dim=-1)
        s = (float(sigma) * band.float() / _STATE["band_px"]).view(-1, 1)            # pixels -> ray units, per packet
        pick = (st["lp"][..., 1] - 0.5 * (e2 / s) ** 2) > (st["lp"][..., 0] - 0.5 * (e1 / s) ** 2)
        face = torch.where(pick, st["face2"], source_face)
        bary = torch.where(pick.unsqueeze(-1), st["bary2"].to(source_bary.dtype), source_bary)
        n = live.sum().clamp_min(1)
        _STATE["last"] = {"switched": float((pick & live).sum() / n), "routed": int(live.sum())}
    else:
        _STATE["last"] = {"switched": 0.0, "routed": int(live.sum())}
    return _STATE["prev"]["route_association"](mode, assoc, d, mask, verts, faces, vert_faces, lbs_weights, band,
                                               front_tol, z_tol, source_part=source_part, source_face=face,
                                               source_bary=bary, source_valid=source_valid, k=k, chunk=chunk)


def install(sigma_px, band_px=16.0, mode="point"):
    """Wrap the installed router routes (a patch router under PFA) with state arbitration; mode "point" with sigma_px
    None = part-first decoding; mode "ray" = the ray-association part gated by the router's two candidates."""
    if mode not in MODES or (mode == "ray" and sigma_px is not None):
        raise ValueError("mode point (sigma_px or None) or ray (no sigma)")
    if _STATE["on"]:
        raise RuntimeError("arbitration already installed")
    st = SM.installed()
    if st is None or st["assoc"] != "pfa":
        raise RuntimeError("arbitration wraps an installed router under PFA")
    _STATE.update(on=True, mode=mode, sigma_px=None if sigma_px is None else float(sigma_px), band_px=float(band_px),
                  prev={"sample_sources": XAG.sample_sources, "route_association": XAG.route_association},
                  stash=None, last=None)
    XAG.sample_sources = arb_sources
    XAG.route_association = arb_association
    return installed()


def uninstall():
    if _STATE["on"]:
        XAG.sample_sources = _STATE["prev"]["sample_sources"]
        XAG.route_association = _STATE["prev"]["route_association"]
    _STATE.update(on=False, mode="point", sigma_px=None, prev=None, stash=None, tables={}, last=None)


def installed():
    return {"mode": _STATE["mode"], "sigma_px": _STATE["sigma_px"], "band_px": _STATE["band_px"]} if _STATE["on"] else None


def last():
    return _STATE["last"]


__all__ = ["MODES", "arb_association", "arb_sources", "candidates", "install", "installed", "last", "uninstall"]
