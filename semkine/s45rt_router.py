r"""S45-RT T0P: the surface-patch router (docs/EXPERIMENT_RECORDS.md#s45rt-prereg-20261031), and arm T1p's hook.

Input: a packet batch's events only, as S45-TR T0 (no prior, no K, no betas). Unlike T0's local kNN event graph, T0P
sees the whole packet as an image, so the evidence for "which part of the hand is this" can come from the hand's
overall event layout:

  image    4 x 180 x 240 per packet: per polarity the last event time / packet duration, and log1p(event count)
           (built with a stable sort and unique-index writes, deterministic under torch's deterministic mode)
  trunk    stem + four stride-2 residual stages (32 / 64 / 128 / 256 channels, GroupNorm 8), FPN-style nearest
           upsampling back to 1/4 resolution (45 x 60), 128 channels
  head     per S45 stride-sampled node (<= 2048, the events S45's association sees): the trunk feature at the node's
           pixel (1/4 resolution, integer lookup) + token7 -> MLP 135 -> 256 -> 256 -> patch logits, rest-template
           point (cm), on-hand logit

Patches: each MANO face gets the part of the majority of its vertices' skinning argmax; inside each part, farthest
point sampling over template face-centroid geodesic distances (semkine/s45ct_route.face_geodesic) until every face is
within PATCH_RADIUS_M of a centre; each face joins its geodesically nearest centre. Patches never cross parts.

Routing (T1p): an event is routed only when on-hand probability >= ONHAND_TAU and the top patch probability >=
PATCH_TAU; its face is the face of the predicted patch nearest (Euclidean, rest template) to the predicted template
point, and its barycentrics that point projected onto the face (S45-TR T0's `project_bary`). As in S45-TR T1b, the
hook answers `semkine.xag.sample_sources` and `route_association` swaps the routes only into events S45's ray
association already matched (q > 0, valid). The router's weights are not model parameters (not in the state dict,
no gradient). Opt-in only: `install(checkpoint, onhand_tau, patch_tau)`.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import xag as XAG
from .encoder import EV_BATCH, EV_P, EV_T, EV_X, EV_Y, _packet_dt, event_tokens
from .s45tr_t0 import project_bary
from .xyz_mesh import stride_sample

MAX_NODES = 2048
HEIGHT, WIDTH = 180, 240
PATCH_RADIUS_M = 0.012
N_PARTS = 16
TRUNK = (32, 64, 128, 256)
FPN = 128
HEAD = 256
#: loss weights (fixed in docs/EXPERIMENT_RECORDS.md#s45rt-prereg-20261031)
WEIGHTS = {"patch": 1.0, "coord": 1.0, "onhand": 0.5}
#: face lookups at inference run in chunks of this many nodes
CHUNK = 65536


# --------------------------------------------------------------------------- patches
def patch_table(npz, radius_m: float = PATCH_RADIUS_M):
    """`(patch_of_face (F,), face_part (F,), centres (P,))`, all int64; deterministic."""
    from .s45ct_route import face_geodesic
    with np.load(str(npz), allow_pickle=False) as z:
        tmpl = np.asarray(z["v_template"], np.float64)
        faces = np.asarray(z["f"], np.int64)
        vpart = np.asarray(z["weights"]).argmax(1)
    face_part = np.array([np.bincount(vpart[t], minlength=N_PARTS).argmax() for t in faces], np.int64)
    dist = face_geodesic(tmpl, faces).astype(np.float64)
    cent = tmpl[faces].mean(1)
    patch = np.full(len(faces), -1, np.int64)
    centres = []
    for part in range(N_PARTS):
        idx = np.flatnonzero(face_part == part)
        if not len(idx):
            continue
        start = int(idx[np.argmin(np.linalg.norm(cent[idx] - cent[idx].mean(0), axis=1))])
        chosen = [start]
        d = dist[start, idx].copy()
        while d.max() > radius_m:
            nxt = int(idx[int(np.argmax(d))])
            chosen.append(nxt)
            d = np.minimum(d, dist[nxt, idx])
        patch[idx] = len(centres) + np.argmin(dist[np.ix_(chosen, idx)], axis=0)
        centres.extend(chosen)
    assert (patch >= 0).all()
    return patch, face_part, np.asarray(centres, np.int64)


def table_sha256(npz) -> str:
    import hashlib
    h = hashlib.sha256()
    for a in patch_table(npz):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- image
def event_image(events: torch.Tensor, ptr: torch.Tensor, delta_t_s: torch.Tensor,
                height: int = HEIGHT, width: int = WIDTH) -> torch.Tensor:
    """`(B, 4, H, W)`: per polarity the last event time / packet duration, then per polarity log1p(count)."""
    B = int(ptr.numel() - 1)
    img = events.new_zeros((B * 4 * height * width,), dtype=torch.float32)
    n = events.shape[0]
    if n == 0:
        return img.view(B, 4, height, width)
    dt = _packet_dt(delta_t_s, events).float()
    b = events[:, EV_BATCH].long()
    x = events[:, EV_X].long().clamp(0, width - 1)
    y = events[:, EV_Y].long().clamp(0, height - 1)
    p = (events[:, EV_P] > 0.5).long()
    t = (events[:, EV_T].float() / dt).clamp(0.0, 1.0)
    pix = (b * 4 + p) * (height * width) + y * width + x            # time channel of (packet, polarity, pixel)
    idx = torch.arange(n, device=events.device)
    order = torch.argsort(pix * (n + 1) + idx)                      # unique keys: deterministic order
    key, ts = pix[order], t[order]
    last = torch.ones_like(key, dtype=torch.bool)
    last[:-1] = key[1:] != key[:-1]
    first = torch.ones_like(key, dtype=torch.bool)
    first[1:] = key[1:] != key[:-1]
    pos = torch.arange(n, device=events.device)
    count = (pos[last] - pos[first] + 1).float()
    ukey = key[last]
    img[ukey] = ts[last]                                            # time-sorted packets: the last event wins
    img[ukey + 2 * height * width] = torch.log1p(count)
    return img.view(B, 4, height, width)


# --------------------------------------------------------------------------- network
class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.gn1 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.gn2 = nn.GroupNorm(8, cout)
        self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.GroupNorm(8, cout))

    def forward(self, x):
        h = F.relu(self.gn1(self.conv1(x)))
        return F.relu(self.gn2(self.conv2(h)) + self.skip(x))


class T0PNet(nn.Module):
    def __init__(self, faces: torch.Tensor, verts: torch.Tensor, patch_of_face: torch.Tensor, face_part: torch.Tensor):
        super().__init__()
        n_patches = int(patch_of_face.max()) + 1
        self.stem = nn.Sequential(nn.Conv2d(4, TRUNK[0], 3, 1, 1, bias=False), nn.GroupNorm(8, TRUNK[0]), nn.ReLU())
        chans = (TRUNK[0],) + TRUNK
        self.stages = nn.ModuleList(Block(chans[i], chans[i + 1], 2) for i in range(4))    # 1/2 .. 1/16
        self.lateral = nn.ModuleList(nn.Conv2d(c, FPN, 1) for c in TRUNK[1:])                # 1/4, 1/8, 1/16
        self.smooth = nn.Sequential(nn.Conv2d(FPN, FPN, 3, 1, 1), nn.GroupNorm(8, FPN), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(FPN + 7, HEAD), nn.ReLU(inplace=True), nn.Linear(HEAD, HEAD),
                                  nn.ReLU(inplace=True))
        self.patch = nn.Linear(HEAD, n_patches)
        self.coord = nn.Linear(HEAD, 3)
        self.onhand = nn.Linear(HEAD, 1)
        self.register_buffer("faces", faces.long().clone(), persistent=False)
        self.register_buffer("tmpl", verts.float().clone(), persistent=False)
        self.register_buffer("patch_of_face", patch_of_face.long().clone(), persistent=False)
        self.register_buffer("face_part", face_part.long().clone(), persistent=False)
        members = [torch.nonzero(patch_of_face == k).flatten() for k in range(n_patches)]
        width = max(len(m) for m in members)
        table = torch.zeros(n_patches, width, dtype=torch.long)
        valid = torch.zeros(n_patches, width, dtype=torch.bool)
        for k, m in enumerate(members):
            table[k, :len(m)], valid[k, :len(m)] = m, True
        self.register_buffer("patch_faces", table, persistent=False)
        self.register_buffer("patch_faces_valid", valid, persistent=False)
        self.n_patches = n_patches

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
        return {"z": z, "patch": self.patch(z), "coord": self.coord(z), "onhand": self.onhand(z).squeeze(-1),
                "src": src, "mask": mask}

    def route(self, out, select):
        """`(face, bary, patch probability)` for every node; faces are looked up for the selected nodes only."""
        prob = torch.softmax(out["patch"].float(), -1)
        pprob, patch = prob.max(-1)
        point = out["coord"].float() * 0.01                                                # metres
        face = torch.zeros(select.shape, dtype=torch.long, device=select.device)
        rows = select.reshape(-1).nonzero().flatten()
        fp, pt = patch.reshape(-1), point.reshape(-1, 3)
        cent = self.tmpl[self.faces].mean(1)                                               # (F, 3)
        out_face = face.view(-1)
        for chunk in rows.split(CHUNK):
            cand = self.patch_faces[fp[chunk]]                                             # (n, M)
            ok = self.patch_faces_valid[fp[chunk]]
            d = (cent[cand] - pt[chunk].unsqueeze(1)).norm(dim=-1).masked_fill(~ok, float("inf"))
            out_face[chunk] = cand.gather(1, d.argmin(1, keepdim=True)).squeeze(1)
        tri = self.tmpl[self.faces[face]]
        bary = project_bary(point, tri)
        return face, bary, pprob


def template(npz):
    with np.load(str(npz), allow_pickle=False) as z:
        return torch.tensor(np.asarray(z["v_template"], np.float32)), torch.tensor(np.asarray(z["f"], np.int64))


def build(npz):
    verts, faces = template(npz)
    patch, part, _ = patch_table(npz)
    return T0PNet(faces, verts, torch.from_numpy(patch), torch.from_numpy(part))


# --------------------------------------------------------------------------- training
def targets(net: T0PNet, batch, src, mask):
    """Per-node labels at the sampled positions (XAG pseudo-sources, as S45-TR T0): valid, face, patch, point (m)."""
    s = XAG.sample_sources(batch, src, mask)
    F_ = net.faces.shape[0]
    valid = s["xag_valid"] & (s["xag_face"] >= 0) & (s["xag_face"] < F_)
    valid = valid & torch.isfinite(s["xag_bary"]).all(-1) & (s["xag_bary"] >= 0).all(-1) & (s["xag_bary"].sum(-1) > 0)
    face = s["xag_face"].clamp(0, F_ - 1)
    w = s["xag_bary"].clamp_min(0.0)
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
    point = (w.unsqueeze(-1) * net.tmpl[net.faces[face]]).sum(-2)
    return {"valid": valid, "face": face, "patch": net.patch_of_face[face], "point": point}


def loss(net: T0PNet, out, tgt):
    mask, valid = out["mask"], tgt["valid"]
    parts = {"onhand": F.binary_cross_entropy_with_logits(out["onhand"][mask].float(), valid[mask].float())}
    if bool(valid.any()):
        parts["patch"] = F.cross_entropy(out["patch"][valid].float(), tgt["patch"][valid])
        parts["coord"] = F.smooth_l1_loss(out["coord"][valid].float(), tgt["point"][valid] / 0.01)
    else:
        parts.update(patch=parts["onhand"] * 0.0, coord=parts["onhand"] * 0.0)
    return sum(WEIGHTS[k] * v for k, v in parts.items()), parts


def save_checkpoint(net: T0PNet, path, **meta):
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "faces": net.faces.cpu(), "tmpl": net.tmpl.cpu(), "patch_of_face": net.patch_of_face.cpu(),
                "face_part": net.face_part.cpu(), "weights": WEIGHTS, "radius_m": PATCH_RADIUS_M, **meta}, str(path))


def load(path, map_location="cpu"):
    obj = torch.load(str(path), map_location=map_location)
    net = T0PNet(obj["faces"], obj["tmpl"], obj["patch_of_face"], obj["face_part"])
    net.load_state_dict(obj["state_dict"], strict=True)
    return net.eval(), obj


# --------------------------------------------------------------------------- T1p hook
_ORIGINAL = XAG.sample_sources
_STATE = {"net": None, "onhand_tau": None, "patch_tau": None, "checkpoint": None, "on": {}}


@torch.no_grad()
def predicted_sources(batch, src, mask, *, require_depth=False):
    if require_depth:
        raise RuntimeError("T1p routes faces only; full_oracle depth is not defined for it")
    dev = src.device
    net = _STATE["on"].get(dev)
    if net is None:
        net = _STATE["net"].to(dev)
        _STATE["on"][dev] = net
    out = net(batch.events, batch.ptr, batch.delta_t_s)
    if out["src"].shape != src.shape or not torch.equal(out["src"][mask], src[mask]) or not torch.equal(out["mask"], mask):
        raise RuntimeError("T0P nodes are not S45's sampled events")
    onhand = torch.sigmoid(out["onhand"].float())
    pprob = torch.softmax(out["patch"].float(), -1).max(-1).values
    valid = mask & (onhand >= _STATE["onhand_tau"]) & (pprob >= _STATE["patch_tau"])
    face, bary, _ = net.route(out, valid)
    zero = torch.zeros_like(face)
    return {"xag_source_index": zero, "xag_source_ms": zero, "xag_part": net.face_part[face],
            "xag_face": face, "xag_bary": bary.to(torch.float32), "xag_valid": valid}


def install(checkpoint, onhand_tau: float = 0.5, patch_tau: float = 0.5) -> None:
    net, _ = load(checkpoint)
    for p in net.parameters():
        p.requires_grad_(False)
    _STATE.update(net=net, onhand_tau=float(onhand_tau), patch_tau=float(patch_tau), checkpoint=str(checkpoint), on={})
    XAG.sample_sources = predicted_sources


def uninstall() -> None:
    XAG.sample_sources = _ORIGINAL
    _STATE.update(net=None, onhand_tau=None, patch_tau=None, checkpoint=None, on={})


def installed():
    if XAG.sample_sources is not predicted_sources:
        return None
    return {"checkpoint": _STATE["checkpoint"], "onhand_tau": _STATE["onhand_tau"], "patch_tau": _STATE["patch_tau"]}


__all__ = ["T0PNet", "WEIGHTS", "build", "event_image", "install", "installed", "load", "loss", "patch_table",
           "predicted_sources", "save_checkpoint", "table_sha256", "targets", "uninstall"]
