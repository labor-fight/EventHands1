r"""S45-SM: state-free surface routing for S45 (docs/EXPERIMENT_RECORDS.md#s45sm-prereg-20261031).

S45's ray association is computed against the previous-state mesh, so when the state is wrong the evidence is assigned
to the wrong surface and the loop has no restoring force (17 networks in Results.md: every one that kept it fails to
contract from the mean hand; every one with a state-independent surface assignment contracts). S45-CT showed that what
matters is the packet's occupancy of the true surface, not per-event pairing. This module supplies that assignment
from the events alone, in two opt-in pieces (`install` patches module attributes, `uninstall` restores them):

Routers (see only the current packet's events: no previous state, no K, no betas)
  t0p  S45-RT's surface-patch router (semkine/s45rt_router.py, frozen checkpoint), used without its
       patch-confidence filter: every node with on-hand probability >= tau is routed.
  sr   the lean surface router SRNet below: T0P's event image and trunk shape, shrunk, with a 16-way part
       head instead of 279 patches. Face = the face of the predicted part whose rest-template centroid is
       nearest the predicted rest-template point; barycentrics = that point projected onto the face.

How the routes enter S45 (config `MODEL.XAG_ASSOC: face_route` either way, so no ground-truth source is read)
  face_route  S45-TR T1d's mechanism unchanged (semkine/xag.py route_association): only events S45's ray
              association matched (q > 0) take the routed face; everything else stays S45's.
  pfa         prior-free association: S45's ray casting is skipped (associate_rays sees an empty mask and
              returns its all-zero association) and every live node the router calls on-hand is assigned to its
              routed face (barycentric weights, q = 1, valid = hit = 1) at the camera depth of the routed point on
              the previous-state mesh, so dXYZ is the offset between the event's ray and where the state puts
              that surface point; the other nodes go to S45's residual slot. No parameter changes; the previous
              state enters only through the residual geometry, never through the assignment.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import xag as XAG
from . import xyz_mesh as XM
from .encoder import EV_X, EV_Y, event_tokens
from .s45rt_router import Block, event_image
from .s45tr_t0 import project_bary
from .xyz_mesh import stride_sample

MAX_NODES = 2048
HEIGHT, WIDTH = 180, 240
N_PARTS = 16
#: SR widths (fixed in the pre-registration): stem/stages, FPN, per-node head
TRUNK = (16, 24, 48, 64)
FPN = 32
HEAD = 64
#: SR loss weights: part cross-entropy, rest-template point smooth-L1 (cm), on-hand BCE (T0P's, part for patch)
WEIGHTS = {"part": 1.0, "coord": 1.0, "onhand": 0.5}
ROUTERS = ("t0p", "sr")
ASSOCS = ("face_route", "pfa")
#: PFA depth floor (m): a diverged state can put a routed point behind the camera; the ray parameter stays positive
Z_MIN = 0.05
CHUNK = 65536


def face_part_table(lbs_weights: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Part of every face = the majority skinning part of its three vertices (ties: lowest part), as T0P's tables."""
    vpart = np.asarray(lbs_weights).argmax(1)
    return np.array([np.bincount(vpart[t], minlength=N_PARTS).argmax() for t in np.asarray(faces, np.int64)], np.int64)


class SRNet(nn.Module):
    """Lean surface router: event image -> small residual CNN + FPN (1/4 resolution) -> per-node MLP ->
    part logits (16), rest-template point (cm), on-hand logit."""

    def __init__(self, faces: torch.Tensor, verts: torch.Tensor, face_part: torch.Tensor):
        super().__init__()
        c = TRUNK
        self.stem = nn.Sequential(nn.Conv2d(4, c[0], 3, 1, 1, bias=False), nn.GroupNorm(8, c[0]), nn.ReLU())
        chans = (c[0],) + c
        self.stages = nn.ModuleList(Block(chans[i], chans[i + 1], 2) for i in range(4))     # 1/2 .. 1/16
        self.lateral = nn.ModuleList(nn.Conv2d(ch, FPN, 1) for ch in c[1:])                  # 1/4, 1/8, 1/16
        self.smooth = nn.Sequential(nn.Conv2d(FPN, FPN, 3, 1, 1), nn.GroupNorm(8, FPN), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(FPN + 7, HEAD), nn.ReLU(inplace=True), nn.Linear(HEAD, HEAD),
                                  nn.ReLU(inplace=True))
        self.part = nn.Linear(HEAD, N_PARTS)
        self.coord = nn.Linear(HEAD, 3)
        self.onhand = nn.Linear(HEAD, 1)
        self.register_buffer("faces", faces.long().clone(), persistent=False)
        self.register_buffer("tmpl", verts.float().clone(), persistent=False)
        self.register_buffer("face_part", face_part.long().clone(), persistent=False)
        members = [torch.nonzero(face_part == k).flatten() for k in range(N_PARTS)]
        width = max(len(m) for m in members)
        table = torch.zeros(N_PARTS, width, dtype=torch.long)
        valid = torch.zeros(N_PARTS, width, dtype=torch.bool)
        for k, m in enumerate(members):
            table[k, :len(m)], valid[k, :len(m)] = m, True
        self.register_buffer("part_faces", table, persistent=False)
        self.register_buffer("part_faces_valid", valid, persistent=False)
        self.register_buffer("centroid", verts.float()[faces.long()].mean(1), persistent=False)

    def trunk(self, image):
        x = self.stem(image)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        top = self.lateral[2](feats[3])                                                    # 1/16
        for level in (1, 0):                                                               # 1/8, then 1/4
            lat = self.lateral[level](feats[level + 1])
            top = F.interpolate(top, size=lat.shape[-2:], mode="nearest") + lat
        return self.smooth(top)                                                            # (B, FPN, H/4, W/4)

    def forward(self, events, ptr, delta_t_s):
        B = int(ptr.numel() - 1)
        src, mask = stride_sample(ptr, MAX_NODES)
        tok = event_tokens(events, ptr, delta_t_s, HEIGHT, WIDTH)
        fmap = self.trunk(event_image(events, ptr, delta_t_s).to(self.stem[0].weight.dtype))
        if tok.shape[0] == 0:
            tok = events.new_zeros((1, 7))
            ev = events.new_zeros((1, events.shape[1]))
            src = torch.zeros_like(src)
        else:
            ev = events
        flat = src.reshape(-1)
        node_ev = ev[flat]
        h4, w4 = fmap.shape[-2:]
        yy = (node_ev[:, EV_Y].long() * h4 // HEIGHT).clamp(0, h4 - 1)
        xx = (node_ev[:, EV_X].long() * w4 // WIDTH).clamp(0, w4 - 1)
        bb = torch.arange(B, device=events.device).repeat_interleave(src.shape[1])
        feat = fmap[bb, :, yy, xx]                                                         # (B * N, FPN)
        z = self.head(torch.cat([feat.to(tok.dtype), tok[flat]], -1)).view(B, src.shape[1], -1)
        z = z * mask.unsqueeze(-1).to(z.dtype)
        return {"z": z, "part": self.part(z), "coord": self.coord(z), "onhand": self.onhand(z).squeeze(-1),
                "src": src, "mask": mask}

    def route(self, out, select):
        """`(face, bary)` for every node; faces are looked up for the selected nodes only (others: face 0)."""
        part = out["part"].float().argmax(-1)
        point = out["coord"].float() * 0.01                                                # metres
        face = torch.zeros(select.shape, dtype=torch.long, device=select.device)
        rows = select.reshape(-1).nonzero().flatten()
        fp, pt = part.reshape(-1), point.reshape(-1, 3)
        out_face = face.view(-1)
        for chunk in rows.split(CHUNK):
            cand = self.part_faces[fp[chunk]]                                              # (n, M)
            ok = self.part_faces_valid[fp[chunk]]
            d = (self.centroid[cand] - pt[chunk].unsqueeze(1)).norm(dim=-1).masked_fill(~ok, float("inf"))
            out_face[chunk] = cand.gather(1, d.argmin(1, keepdim=True)).squeeze(1)
        bary = project_bary(point, self.tmpl[self.faces[face]])
        return face, bary


def build(npz):
    with np.load(str(npz), allow_pickle=False) as z:
        verts = torch.tensor(np.asarray(z["v_template"], np.float32))
        faces = torch.tensor(np.asarray(z["f"], np.int64))
        part = torch.from_numpy(face_part_table(np.asarray(z["weights"]), np.asarray(z["f"], np.int64)))
    return SRNet(faces, verts, part)


def n_params(net: nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def targets(net: SRNet, batch, src, mask):
    """Per-node labels at the sampled positions (XAG pseudo-sources, as T0 / T0P): valid, face, part, point (m)."""
    s = XAG.sample_sources(batch, src, mask)
    F_ = net.faces.shape[0]
    valid = s["xag_valid"] & (s["xag_face"] >= 0) & (s["xag_face"] < F_)
    valid = valid & torch.isfinite(s["xag_bary"]).all(-1) & (s["xag_bary"] >= 0).all(-1) & (s["xag_bary"].sum(-1) > 0)
    face = s["xag_face"].clamp(0, F_ - 1)
    w = s["xag_bary"].clamp_min(0.0)
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
    point = (w.unsqueeze(-1) * net.tmpl[net.faces[face]]).sum(-2)
    return {"valid": valid, "face": face, "part": net.face_part[face], "point": point}


def loss(net: SRNet, out, tgt):
    mask, valid = out["mask"], tgt["valid"]
    parts = {"onhand": F.binary_cross_entropy_with_logits(out["onhand"][mask].float(), valid[mask].float())}
    if bool(valid.any()):
        parts["part"] = F.cross_entropy(out["part"][valid].float(), tgt["part"][valid])
        parts["coord"] = F.smooth_l1_loss(out["coord"][valid].float(), tgt["point"][valid] / 0.01)
    else:
        parts.update(part=parts["onhand"] * 0.0, coord=parts["onhand"] * 0.0)
    return sum(WEIGHTS[k] * v for k, v in parts.items()), parts


def save_checkpoint(net: SRNet, path, **meta):
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "faces": net.faces.cpu(),
                "tmpl": net.tmpl.cpu(), "face_part": net.face_part.cpu(), "weights": WEIGHTS, "trunk": TRUNK,
                "fpn": FPN, "head": HEAD, **meta}, str(path))


def load(path, map_location="cpu"):
    obj = torch.load(str(path), map_location=map_location)
    if (tuple(obj["trunk"]), obj["fpn"], obj["head"]) != (TRUNK, FPN, HEAD):
        raise ValueError("SR checkpoint widths differ from this module")
    net = SRNet(obj["faces"], obj["tmpl"], obj["face_part"])
    net.load_state_dict(obj["state_dict"], strict=True)
    return net.eval(), obj


# --------------------------------------------------------------------------- hooks
_NATIVE = {"sample_sources": XAG.sample_sources, "route_association": XAG.route_association,
           "associate_rays": XM.associate_rays}
if any(f.__module__ != m for f, m in ((_NATIVE["sample_sources"], "semkine.xag"),
                                      (_NATIVE["route_association"], "semkine.xag"),
                                      (_NATIVE["associate_rays"], "semkine.xyz_mesh"))):
    raise ImportError("semkine.s45sm must be imported before any other XAG / association hook is installed")
_STATE = {"router": None, "net": None, "checkpoint": None, "assoc": None, "onhand_tau": None, "on": {}}


def _load_router(router, checkpoint):
    if router == "t0p":
        from . import s45rt_router as RT
        net, _ = RT.load(checkpoint)
    else:
        net, _ = load(checkpoint)
    for p in net.parameters():
        p.requires_grad_(False)
    return net.eval()


def _net(device):
    net = _STATE["on"].get(device)
    if net is None:
        net = _STATE["net"].to(device)
        _STATE["on"][device] = net
    return net


@torch.no_grad()
def route_nodes(net, router, events, ptr, delta_t_s, onhand_tau):
    """Router forward on a packet batch: `(out, routed (B, N) bool, face (B, N), bary (B, N, 3), onhand prob)`."""
    out = net(events, ptr, delta_t_s)
    onhand = torch.sigmoid(out["onhand"].float())
    routed = out["mask"] & (onhand >= onhand_tau)
    if router == "t0p":
        face, bary, _ = net.route(out, routed)
    else:
        face, bary = net.route(out, routed)
    return out, routed, face, bary, onhand


@torch.no_grad()
def predicted_sources(batch, src, mask, *, require_depth=False):
    if require_depth:
        raise RuntimeError("S45-SM routes faces; full_oracle depth is not defined for it")
    net = _net(src.device)
    out, routed, face, bary, _ = route_nodes(net, _STATE["router"], batch.events, batch.ptr, batch.delta_t_s,
                                             _STATE["onhand_tau"])
    if out["src"].shape != src.shape or not torch.equal(out["src"][mask], src[mask]) or not torch.equal(out["mask"], mask):
        raise RuntimeError("router nodes are not S45's sampled events")
    zero = torch.zeros_like(face)
    return {"xag_source_index": zero, "xag_source_ms": zero, "xag_part": net.face_part[face], "xag_face": face,
            "xag_bary": bary.to(torch.float32), "xag_valid": routed & mask}


def empty_association(d, mask, verts, faces, vert_faces, band, front_tol, z_tol, k=16, chunk=XM.ASSOC_CHUNK,
                      mode="hit_first", normals=None, gap=0.01):
    """PFA: S45's own association on an empty mask = its all-zero record set, same shapes and dtypes, no ray casting."""
    if mode != "hit_first":
        raise ValueError("PFA replaces the hit_first association only")
    return _NATIVE["associate_rays"](d, torch.zeros_like(mask), verts, faces, vert_faces, band, front_tol, z_tol, k=k,
                                     chunk=chunk)


@torch.no_grad()
def pfa_association(mode, assoc, d, mask, verts, faces, vert_faces, lbs_weights, band, front_tol, z_tol, *,
                    source_part, source_face, source_bary, source_valid, source_depth=None, k=16, chunk=4096):
    """Every live routed node -> its routed face at the depth of the routed point on the previous-state mesh."""
    if mode != "face_route":
        raise ValueError("PFA is installed under MODEL.XAG_ASSOC: face_route")
    if assoc["vid"].shape != (*mask.shape, XM.R_MAX) or assoc["z"].shape != mask.shape:
        raise ValueError("PFA expects the four-record hit_first association shapes")
    known = source_valid & (source_face >= 0) & (source_face < len(faces))
    known = known & torch.isfinite(source_bary).all(-1) & (source_bary >= 0).all(-1) & (source_bary.sum(-1) > 0)
    eligible = known & mask
    out = {name: torch.zeros_like(assoc[name]) for name in ("vid", "A", "q", "z", "valid", "hit")}
    b, n = eligible.nonzero(as_tuple=True)
    fv = faces.long()[source_face[b, n]]                                                   # (m, 3)
    bary = source_bary[b, n].float()
    bary = bary / bary.sum(-1, keepdim=True)
    point = (bary.unsqueeze(-1) * verts.float()[b.unsqueeze(1), fv]).sum(1)                # routed point on prev mesh
    out["vid"][b, n] = torch.cat([fv, fv[:, :1]], -1)
    out["A"][b, n] = torch.cat([bary, torch.zeros_like(bary[:, :1])], -1).to(out["A"].dtype)
    out["q"][b, n] = 1
    out["z"][b, n] = point[:, 2].clamp_min(Z_MIN).to(out["z"].dtype)
    out["valid"][b, n] = True
    out["hit"][b, n] = True
    return out, {"xag_known": known & mask, "xag_eligible": eligible, "xag_replaced": eligible,
                 "xag_fallback": mask & ~eligible, "xag_recalled": eligible, "xag_none": mask & ~eligible}


def install(router, checkpoint, assoc, onhand_tau=0.5):
    if router not in ROUTERS or assoc not in ASSOCS:
        raise ValueError(f"router in {ROUTERS}, assoc in {ASSOCS}")
    if _STATE["net"] is not None:
        raise RuntimeError("S45-SM routing already installed")
    if XAG.sample_sources is not _NATIVE["sample_sources"] or XM.associate_rays is not _NATIVE["associate_rays"] \
            or XAG.route_association is not _NATIVE["route_association"]:
        raise RuntimeError("another hook already patched XAG / associate_rays")
    _STATE.update(router=router, net=_load_router(router, checkpoint), checkpoint=str(checkpoint), assoc=assoc,
                  onhand_tau=float(onhand_tau), on={})
    XAG.sample_sources = predicted_sources
    if assoc == "pfa":
        XAG.route_association = pfa_association
        XM.associate_rays = empty_association
    return installed()


def uninstall():
    XAG.sample_sources = _NATIVE["sample_sources"]
    XAG.route_association = _NATIVE["route_association"]
    XM.associate_rays = _NATIVE["associate_rays"]
    _STATE.update(router=None, net=None, checkpoint=None, assoc=None, onhand_tau=None, on={})


def installed():
    if _STATE["net"] is None:
        return None
    return {"router": _STATE["router"], "checkpoint": _STATE["checkpoint"], "assoc": _STATE["assoc"],
            "onhand_tau": _STATE["onhand_tau"], "router_params": n_params(_STATE["net"])}


__all__ = ["ASSOCS", "FPN", "HEAD", "N_PARTS", "ROUTERS", "SRNet", "TRUNK", "WEIGHTS", "build", "empty_association",
           "face_part_table", "install", "installed", "load", "loss", "n_params", "pfa_association",
           "predicted_sources", "route_nodes", "save_checkpoint", "targets", "uninstall"]
