r"""S45-TR T0: event -> MANO correspondence predictor (docs/EXPERIMENT_RECORDS.md#s45tr-prereg-20261030).

Input: a packet batch's events, stride-sampled exactly as S45 samples them (`stride_sample(ptr, 2048)`), so node `i`
of a packet is the event S45's association sees at position `i`. No state enters (no prior, no K, no betas).

Network: the S36 EventGNN operator on the sampled tokens (`semkine.s45tr_t1a.node_features`: causal kNN over the
`window` preceding nodes, k 8, three rounds, hidden 96), then a per-node head that also reads the packet context
(mean and max of the node features), `MLP([h_i, mean, max]) -> 256 -> 256`, and four outputs:

  face    1538 logits, the MANO face the event's source pixel lies on (evaluated only where needed: a seeded subset of
          at most FACE_NODES labelled nodes per batch in training, the routed nodes at inference)
  part    16 logits, the face's skinning part (XA pseudo-part)
  coord   3 values, the source point on the MANO rest template (v_template), in centimetres
  onhand  1 logit, whether the event has a hand source (XAG valid)

The correspondence a downstream router uses is `(face, bary)`: the argmax face, and the barycentric coordinates of the
predicted template point projected onto that face's plane, clamped to the triangle (`project_bary`).

Training labels are the XAG per-event pseudo-sources (`outputs/xag/sources`), defined before augmentation and carried
through it; hot pixels and unknown events have no source (onhand target 0, no face / part / coord term).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .encoder import event_tokens
from .event_gnn import EventGNN
from .s45tr_t1a import node_features
from .xyz_mesh import stride_sample

MAX_NODES = 2048
HEAD = 256
#: loss weights (fixed in the T0 manifest before any T0 reading)
WEIGHTS = {"face": 1.0, "part": 0.5, "coord": 1.0, "onhand": 0.5}
#: the face cross-entropy reads at most this many labelled nodes per batch (a seeded random subset); the face logits
#: of every node would be ~1e9 values per 512-packet batch. Part, coord and on-hand terms read every node.
FACE_NODES = 131072
#: face logits are evaluated in chunks of this many nodes at inference
CHUNK = 131072


def template(npz):
    with np.load(str(npz), allow_pickle=False) as z:
        return (torch.tensor(np.asarray(z["v_template"], np.float32)), torch.tensor(np.asarray(z["f"], np.int64)))


class T0Net(nn.Module):
    def __init__(self, faces: torch.Tensor, verts: torch.Tensor, gnn: dict, n_parts: int = 16):
        super().__init__()
        self.gnn_cfg = dict(gnn)
        H = int(gnn["HIDDEN"])
        self.gnn = EventGNN(height=180, width=240, hidden=H, feat_dim=H, k=int(gnn["K"]), n_layers=int(gnn["LAYERS"]),
                            max_nodes=MAX_NODES, window=int(gnn["WINDOW"]), t_scale=float(gnn["T_SCALE"]),
                            node_attrs="token7", readout=False)
        self.head = nn.Sequential(nn.Linear(3 * H, HEAD), nn.ReLU(inplace=True), nn.Linear(HEAD, HEAD),
                                  nn.ReLU(inplace=True))
        self.face = nn.Linear(HEAD, int(faces.shape[0]))
        self.part = nn.Linear(HEAD, n_parts)
        self.coord = nn.Linear(HEAD, 3)
        self.onhand = nn.Linear(HEAD, 1)
        self.register_buffer("faces", faces.long().clone(), persistent=False)
        self.register_buffer("tmpl", verts.float().clone(), persistent=False)

    def forward(self, events, ptr, delta_t_s):
        B = int(ptr.numel() - 1)
        src, mask = stride_sample(ptr, MAX_NODES)
        tok = event_tokens(events, ptr, delta_t_s, 180, 240)
        if tok.shape[0] == 0:
            tok = events.new_zeros((1, 7))
            src = torch.zeros_like(src)
        tok = tok[src.reshape(-1)].view(B, -1, tok.shape[-1]) * mask.unsqueeze(-1)
        h = node_features(self.gnn, tok, mask)
        m = mask.unsqueeze(-1).to(h.dtype)
        mean = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        peak = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values * mask.any(1, keepdim=True).to(h.dtype)
        ctx = torch.cat([h, mean.unsqueeze(1).expand_as(h), peak.unsqueeze(1).expand_as(h)], -1)
        z = self.head(ctx)
        return {"z": z, "part": self.part(z), "coord": self.coord(z),
                "onhand": self.onhand(z).squeeze(-1), "src": src, "mask": mask}

    def face_argmax(self, z, select):
        """`(B, N)` argmax face of the selected nodes (0 elsewhere), in chunks."""
        face = torch.zeros(select.shape, dtype=torch.long, device=z.device)
        rows = select.reshape(-1).nonzero().flatten()
        flat = z.reshape(-1, z.shape[-1])
        out = face.view(-1)
        for chunk in rows.split(CHUNK):
            out[chunk] = self.face(flat[chunk].to(self.face.weight.dtype)).argmax(-1)
        return face

    def template_point(self, face, bary):
        """`(..., 3)` metres: the template point of `(face, bary)`."""
        tri = self.tmpl[self.faces[face]]                                    # (..., 3, 3)
        w = bary / bary.sum(-1, keepdim=True).clamp_min(1e-12)
        return (w.unsqueeze(-1) * tri).sum(-2)

    def correspondence(self, out, select=None):
        """`(face, bary, template point)` of a forward output, for the selected nodes (all live nodes by default)."""
        face = self.face_argmax(out["z"], out["mask"] if select is None else select)
        tri = self.tmpl[self.faces[face]]
        bary = project_bary(out["coord"].float() * 0.01, tri)
        return face, bary, (bary.unsqueeze(-1) * tri).sum(-2)


def project_bary(p: torch.Tensor, tri: torch.Tensor) -> torch.Tensor:
    """Barycentric coordinates of `p (..., 3)` projected onto the plane of `tri (..., 3, 3)`, clamped to the triangle."""
    a, b, c = tri.unbind(-2)
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = (v0 * v0).sum(-1), (v0 * v1).sum(-1), (v1 * v1).sum(-1)
    d20, d21 = (v2 * v0).sum(-1), (v2 * v1).sum(-1)
    den = (d00 * d11 - d01 * d01).clamp_min(1e-18)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    bary = torch.stack([1.0 - v - w, v, w], -1).clamp_min(0.0)
    return bary / bary.sum(-1, keepdim=True).clamp_min(1e-12)


def targets(net: T0Net, batch, src, mask):
    """Per-node labels gathered at the sampled positions: valid, face, part, template point (metres)."""
    from .xag import sample_sources
    s = sample_sources(batch, src, mask)
    valid = s["xag_valid"] & (s["xag_face"] >= 0) & (s["xag_face"] < net.faces.shape[0])
    valid = valid & torch.isfinite(s["xag_bary"]).all(-1) & (s["xag_bary"] >= 0).all(-1) & (s["xag_bary"].sum(-1) > 0)
    face = s["xag_face"].clamp(0, net.faces.shape[0] - 1)
    point = net.template_point(face, s["xag_bary"].clamp_min(0.0))
    return {"valid": valid, "face": face, "part": s["xag_part"], "point": point}


def loss(net: T0Net, out, tgt, generator=None):
    mask, valid = out["mask"], tgt["valid"]
    on = F.binary_cross_entropy_with_logits(out["onhand"][mask].float(), valid[mask].float())
    parts = {"onhand": on}
    if bool(valid.any()):
        rows = valid.reshape(-1).nonzero().flatten()
        if rows.numel() > FACE_NODES:
            keep = torch.randperm(rows.numel(), generator=generator, device="cpu")[:FACE_NODES].to(rows.device)
            rows = rows[keep]
        z = out["z"].reshape(-1, out["z"].shape[-1])[rows]
        parts["face"] = F.cross_entropy(net.face(z).float(), tgt["face"].reshape(-1)[rows])
        pv = valid & (tgt["part"] >= 0)
        parts["part"] = F.cross_entropy(out["part"][pv].float(), tgt["part"][pv]) if bool(pv.any()) else on * 0.0
        parts["coord"] = F.smooth_l1_loss(out["coord"][valid].float(), tgt["point"][valid] / 0.01)
    else:
        parts.update(face=on * 0.0, part=on * 0.0, coord=on * 0.0)
    total = sum(WEIGHTS[k] * v for k, v in parts.items())
    return total, parts


def load(path, map_location="cpu"):
    obj = torch.load(str(path), map_location=map_location)
    net = T0Net(obj["faces"], obj["tmpl"], obj["gnn"])
    net.load_state_dict(obj["state_dict"], strict=True)
    return net.eval(), obj


__all__ = ["T0Net", "WEIGHTS", "load", "loss", "project_bary", "targets", "template"]
