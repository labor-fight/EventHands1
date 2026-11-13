r"""RDR routers (docs/EXPERIMENT_RECORDS.md#rdr-prereg-20261110): r_s2d's network (semkine/pfo_router.py, stem "s2d")
redesigned inside its own parameter and MAC budget (137,252 parameters, 58.3 M dense + 12.5 M per-node MACs).

  roi   the dense trunk runs only on an event-centred CROP x CROP window (centre = midpoint of the 5 % / 95 % quantiles of
        the packet's event coordinates; events only, no prior state). The saved pixels buy resolution: a full-resolution
        3x3 stem and a feature map at 1/2 of the crop (56 x 56 cells on the hand instead of about 15-20 cells across it at
        1/4 of the whole image). The 3x3 smoothing convolution becomes depthwise + pointwise to pay for the larger window.
        Nodes outside the window get zero image features (their token is kept).
  set   (training only) a per-(packet, label part) mean term: smooth-L1 between the mean predicted and the mean label
        template point of every part with >= 3 labelled nodes. The trunk aggregates per part, so this is the error it sees.
  hier  T0P's output form at this size: classification over the 279 rest-template patches (none crosses a part) through a
        width-16 bottleneck, the point regressed as before, face = the face of the predicted patch nearest to the point.
        The part is the predicted patch's part.
Opt-in: nothing on an existing path imports this module.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from . import s45rt_router as RT
from . import s45sm as SM
from .encoder import EV_BATCH, EV_X, EV_Y, event_tokens
from .s45rt_router import Block, event_image
from .xyz_mesh import stride_sample

KIND = "rdr"
CROP = 112
Q_LO, Q_HI = 0.05, 0.95
TRUNK = (16, 24, 48, 64)
FPN = 32
HEAD = 64
BOTTLENECK = 16
SET_MIN_NODES = 3
BUDGET_PARAMS = 137_252


class RDRNet(nn.Module):
    def __init__(self, faces, verts, face_part, patch_of_face=None, *, hier=False, crop=CROP):
        super().__init__()
        c = TRUNK
        self.design = {"roi": True, "hier": bool(hier), "crop": int(crop), "trunk": list(c), "fpn": FPN, "head": HEAD,
                       "bottleneck": BOTTLENECK if hier else None, "centre": [Q_LO, Q_HI]}
        self.crop = int(crop)
        self.stem = nn.Sequential(nn.Conv2d(4, c[0], 3, 1, 1, bias=False), nn.GroupNorm(8, c[0]), nn.ReLU())
        chans = (c[0],) + c
        self.stages = nn.ModuleList(Block(chans[i], chans[i + 1], 2) for i in range(4))      # 1/2 .. 1/16 of the crop
        self.lateral = nn.ModuleList(nn.Conv2d(ch, FPN, 1) for ch in c)
        self.smooth = nn.Sequential(nn.Conv2d(FPN, FPN, 3, 1, 1, groups=FPN, bias=False), nn.Conv2d(FPN, FPN, 1),
                                    nn.GroupNorm(8, FPN), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(FPN + 7, HEAD), nn.ReLU(inplace=True), nn.Linear(HEAD, HEAD),
                                  nn.ReLU(inplace=True))
        self.coord = nn.Linear(HEAD, 3)
        self.onhand = nn.Linear(HEAD, 1)
        self.register_buffer("faces", faces.long().clone(), persistent=False)
        self.register_buffer("tmpl", verts.float().clone(), persistent=False)
        self.register_buffer("face_part", face_part.long().clone(), persistent=False)
        self.register_buffer("centroid", verts.float()[faces.long()].mean(1), persistent=False)
        if hier:
            pof = patch_of_face.long().clone()
            self.n_patches = int(pof.max()) + 1
            self.patch = nn.Sequential(nn.Linear(HEAD, BOTTLENECK, bias=False), nn.Linear(BOTTLENECK, self.n_patches))
            self.register_buffer("patch_of_face", pof, persistent=False)
            members = [torch.nonzero(pof == k).flatten() for k in range(self.n_patches)]
            pp = torch.stack([face_part.long()[m].mode().values for m in members])
            self.register_buffer("patch_part", pp, persistent=False)
            name = "patch_faces"
        else:
            self.part = nn.Linear(HEAD, SM.N_PARTS)
            members = [torch.nonzero(face_part == k).flatten() for k in range(SM.N_PARTS)]
            name = "part_faces"
        width = max(len(m) for m in members)
        table = torch.zeros(len(members), width, dtype=torch.long)
        valid = torch.zeros(len(members), width, dtype=torch.bool)
        for k, m in enumerate(members):
            table[k, :len(m)], valid[k, :len(m)] = m, True
        self.register_buffer(name, table, persistent=False)
        self.register_buffer(name + "_valid", valid, persistent=False)

    # ------------------------------------------------------------------ crop
    def origin(self, events, ptr):
        """(B,) x0, y0 of the window: centred on the midpoint of the 5 % / 95 % quantiles of each packet's events."""
        B = int(ptr.numel() - 1)
        dev = events.device
        counts = (ptr[1:] - ptr[:-1]).long()
        n = events.shape[0]
        out = []
        for col, size in ((EV_X, SM.WIDTH), (EV_Y, SM.HEIGHT)):
            if n == 0:
                mid = torch.full((B,), size // 2, dtype=torch.long, device=dev)
            else:
                key = events[:, EV_BATCH].long() * 1024 + events[:, col].long().clamp(0, size - 1)
                s = torch.sort(key).values
                span = (counts - 1).clamp_min(0).to(torch.float64)
                lo = (ptr[:-1].long() + (span * Q_LO).long()).clamp(0, n - 1)
                hi = (ptr[:-1].long() + (span * Q_HI).long()).clamp(0, n - 1)
                mid = ((s[lo] % 1024) + (s[hi] % 1024)) // 2
                mid = torch.where(counts > 0, mid, torch.full_like(mid, size // 2))
            out.append((mid - self.crop // 2).clamp(0, size - self.crop))
        return out[0], out[1]

    def trunk(self, image):
        x = self.stem(image)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        top = self.lateral[3](feats[3])
        for j in (2, 1, 0):
            lat = self.lateral[j](feats[j])
            top = F.interpolate(top, size=lat.shape[-2:], mode="nearest") + lat
        return self.smooth(top)                                                            # (B, FPN, crop/2, crop/2)

    # ------------------------------------------------------------------ forward
    def forward_at(self, events, ptr, delta_t_s, src, mask):
        B = int(ptr.numel() - 1)
        tok = event_tokens(events, ptr, delta_t_s, SM.HEIGHT, SM.WIDTH)
        x0, y0 = self.origin(events, ptr)
        image = event_image(events, ptr, delta_t_s)
        ar = torch.arange(self.crop, device=image.device)
        bi = torch.arange(B, device=image.device)[:, None, None]
        image = image[bi, :, (y0[:, None] + ar)[:, :, None], (x0[:, None] + ar)[:, None, :]].permute(0, 3, 1, 2)
        fmap = self.trunk(image.contiguous().to(self.stem[0].weight.dtype))
        if tok.shape[0] == 0:
            tok = events.new_zeros((1, 7))
            ev = events.new_zeros((1, events.shape[1]))
            src = torch.zeros_like(src)
        else:
            ev = events
        flat = src.reshape(-1)
        node_ev = ev[flat]
        bb = torch.arange(B, device=events.device).repeat_interleave(src.shape[1])
        xr = node_ev[:, EV_X].long() - x0[bb]
        yr = node_ev[:, EV_Y].long() - y0[bb]
        inside = (xr >= 0) & (xr < self.crop) & (yr >= 0) & (yr < self.crop)
        hf, wf = fmap.shape[-2:]
        yy = (yr * hf // self.crop).clamp(0, hf - 1)
        xx = (xr * wf // self.crop).clamp(0, wf - 1)
        feat = fmap[bb, :, yy, xx] * inside.unsqueeze(-1).to(fmap.dtype)
        z = self.head(torch.cat([feat.to(tok.dtype), tok[flat]], -1)).view(B, src.shape[1], -1)
        z = z * mask.unsqueeze(-1).to(z.dtype)
        out = {"z": z, "coord": self.coord(z), "onhand": self.onhand(z).squeeze(-1), "src": src, "mask": mask,
               "inside": inside.view(B, -1)}
        if self.design["hier"]:
            out["patch"] = self.patch(z)
        else:
            out["part"] = self.part(z)
        return out

    def forward(self, events, ptr, delta_t_s):
        src, mask = stride_sample(ptr, SM.MAX_NODES)
        return self.forward_at(events, ptr, delta_t_s, src, mask)

    def route(self, out, select):
        """`(face, bary)` for every node (faces looked up for the selected nodes): T0P's decoding with patches, sr1's with parts."""
        if self.design["hier"]:
            return RT.T0PNet.route(self, out, select)[:2]
        return SM.SRNet.route(self, out, select)

    def predicted_part(self, out):
        if self.design["hier"]:
            return self.patch_part[out["patch"].float().argmax(-1)]
        return out["part"].float().argmax(-1)


# ---------------------------------------------------------------------- training
def targets(net, batch, src, mask):
    if getattr(net, "design", {}).get("hier"):
        t = RT.targets(net, batch, src, mask)
        t["part"] = net.face_part[t["face"]]
        return t
    return SM.targets(net, batch, src, mask)


def set_term(out, tgt):
    """Smooth-L1 between the mean predicted and the mean label point (cm) of every (packet, label part) group."""
    valid = tgt["valid"]
    if not bool(valid.any()):
        return out["coord"].float().sum() * 0.0
    B = valid.shape[0]
    group = (torch.arange(B, device=valid.device)[:, None] * SM.N_PARTS + tgt["part"])[valid]
    n = B * SM.N_PARTS
    pred = out["coord"][valid].float()
    label = (tgt["point"][valid] / 0.01).float()
    sp = torch.zeros(n, 3, device=pred.device, dtype=pred.dtype).index_add_(0, group, pred)
    sl = torch.zeros(n, 3, device=pred.device, dtype=pred.dtype).index_add_(0, group, label)
    cnt = torch.bincount(group, minlength=n)
    keep = cnt >= SET_MIN_NODES
    if not bool(keep.any()):
        return pred.sum() * 0.0
    diff = (sp - sl)[keep] / cnt[keep].unsqueeze(-1).to(pred.dtype)
    return F.smooth_l1_loss(diff, torch.zeros_like(diff))


def loss(net, out, tgt, set_weight=0.0):
    total, parts = (RT.loss if getattr(net, "design", {}).get("hier") else SM.loss)(net, out, tgt)
    if set_weight > 0:
        parts["set"] = set_term(out, tgt)
        total = total + float(set_weight) * parts["set"]
    return total, parts


def build(npz, hier=False, crop=CROP):
    import numpy as np
    with np.load(str(npz), allow_pickle=False) as z:
        verts = torch.tensor(np.asarray(z["v_template"], np.float32))
        faces = torch.tensor(np.asarray(z["f"], np.int64))
        part = torch.from_numpy(SM.face_part_table(np.asarray(z["weights"]), np.asarray(z["f"], np.int64)))
    patch = None
    if hier:
        p, fp, _ = RT.patch_table(npz)
        patch, part = torch.from_numpy(p), torch.from_numpy(fp)
    return RDRNet(faces, verts, part, patch, hier=hier, crop=crop)


def save_checkpoint(net, path, **meta):
    obj = {"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "faces": net.faces.cpu(),
           "tmpl": net.tmpl.cpu(), "face_part": net.face_part.cpu(), "kind": KIND, "design": net.design, **meta}
    if net.design["hier"]:
        obj["patch_of_face"] = net.patch_of_face.cpu()
    torch.save(obj, str(path))


def load(path, map_location="cpu"):
    obj = torch.load(str(path), map_location=map_location)
    if obj.get("kind") != KIND:
        raise ValueError("not an RDR router checkpoint: " + str(path))
    d = obj["design"]
    net = RDRNet(obj["faces"], obj["tmpl"], obj["face_part"], obj.get("patch_of_face"), hier=d["hier"], crop=d["crop"])
    net.load_state_dict(obj["state_dict"], strict=True)
    return net.eval(), obj


__all__ = ["BUDGET_PARAMS", "CROP", "KIND", "RDRNet", "build", "load", "loss", "save_checkpoint", "set_term", "targets"]
