r"""PFO slim SRNet routers (docs/EXPERIMENT_RECORDS.md#pfo-prereg-20261103 and the router sections): sr1's chain
"4-channel event image -> CNN / FPN at 1/4 resolution -> per-event integer query + token7 -> part (16) / rest-template
point / on-hand heads -> part-then-nearest-centroid face + clipped barycentrics", with configurable cost:

  stem    "s1"  sr1's 3x3 stride-1 convolution at full resolution (4 -> c0)
          "s2d" lossless 2x2 pixel unshuffle (4 -> 16 channels at 1/2) + 1x1 convolution (16 -> c0)
  stages  four residual blocks (semkine/s45rt_router.py `Block`), stride 2 each; with "s2d" the stage outputs are at
          1/4, 1/8, 1/16, 1/32 and the FPN takes the finest three plus the top (still a 1/4 output map)
  fpn     lateral 1x1 to `fpn` channels, nearest top-down sum; `smooth` "conv3" (sr1) | "none"
  head    per-node MLP (fpn + 7 -> head -> head), part / point / on-hand linear heads (sr1's)
Every variant routes exactly like sr1 (same tables, same decoding); `register()` teaches the S45-SM hooks the router kind
"pfr" (loaded from a PFO checkpoint, routed as "sr").
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from . import s45sm as SM
from . import s45sm_fast as FAST
from .encoder import EV_X, EV_Y, event_tokens
from .s45rt_router import Block, event_image
from .xyz_mesh import stride_sample

KIND = "pfr"
_STATE = {"registered": False, "prev": None}


class SlimSR(SM.SRNet):
    """SRNet with a configurable stem / widths / smoothing / head (see the module docstring)."""

    def __init__(self, faces, verts, face_part, stem="s1", trunk=(16, 24, 48, 64), fpn=32, smooth="conv3", head=64):
        nn.Module.__init__(self)
        self.design = {"stem": stem, "trunk": list(trunk), "fpn": int(fpn), "smooth": smooth, "head": int(head)}
        c = tuple(int(x) for x in trunk)
        if stem == "s1":
            self.stem = nn.Sequential(nn.Conv2d(4, c[0], 3, 1, 1, bias=False), nn.GroupNorm(8, c[0]), nn.ReLU())
        elif stem == "s2d":
            self.stem = nn.Sequential(nn.Conv2d(16, c[0], 1, bias=False), nn.GroupNorm(8, c[0]), nn.ReLU())
        else:
            raise ValueError(stem)
        chans = (c[0],) + c
        self.stages = nn.ModuleList(Block(chans[i], chans[i + 1], 2) for i in range(4))
        # FPN inputs: the stage outputs at 1/4, 1/8, 1/16 (s1: stages 2-4; s2d: stages 1-3, plus stage 4 as the top)
        self.levels = (1, 2, 3) if stem == "s1" else (0, 1, 2, 3)
        self.lateral = nn.ModuleList(nn.Conv2d(c[i], fpn, 1) for i in self.levels)
        if smooth == "conv3":
            self.smooth = nn.Sequential(nn.Conv2d(fpn, fpn, 3, 1, 1), nn.GroupNorm(8, fpn), nn.ReLU())
        elif smooth == "none":
            self.smooth = nn.ReLU()
        else:
            raise ValueError(smooth)
        self.head = nn.Sequential(nn.Linear(fpn + 7, head), nn.ReLU(inplace=True), nn.Linear(head, head),
                                  nn.ReLU(inplace=True))
        self.part = nn.Linear(head, SM.N_PARTS)
        self.coord = nn.Linear(head, 3)
        self.onhand = nn.Linear(head, 1)
        self.register_buffer("faces", faces.long().clone(), persistent=False)
        self.register_buffer("tmpl", verts.float().clone(), persistent=False)
        self.register_buffer("face_part", face_part.long().clone(), persistent=False)
        members = [torch.nonzero(face_part == k).flatten() for k in range(SM.N_PARTS)]
        width = max(len(m) for m in members)
        table = torch.zeros(SM.N_PARTS, width, dtype=torch.long)
        valid = torch.zeros(SM.N_PARTS, width, dtype=torch.bool)
        for k, m in enumerate(members):
            table[k, :len(m)], valid[k, :len(m)] = m, True
        self.register_buffer("part_faces", table, persistent=False)
        self.register_buffer("part_faces_valid", valid, persistent=False)
        self.register_buffer("centroid", verts.float()[faces.long()].mean(1), persistent=False)

    def trunk(self, image):
        if self.design["stem"] == "s2d":
            image = F.pixel_unshuffle(image, 2)
        x = self.stem(image)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        lv = self.levels
        top = self.lateral[-1](feats[lv[-1]])
        for j in range(len(lv) - 2, -1, -1):
            lat = self.lateral[j](feats[lv[j]])
            top = F.interpolate(top, size=lat.shape[-2:], mode="nearest") + lat
        return self.smooth(top)


def build(npz, **design):
    import numpy as np
    with np.load(str(npz), allow_pickle=False) as z:
        verts = torch.tensor(np.asarray(z["v_template"], np.float32))
        faces = torch.tensor(np.asarray(z["f"], np.int64))
        part = torch.from_numpy(SM.face_part_table(np.asarray(z["weights"]), np.asarray(z["f"], np.int64)))
    return SlimSR(faces, verts, part, **design)


def save_checkpoint(net, path, **meta):
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "faces": net.faces.cpu(),
                "tmpl": net.tmpl.cpu(), "face_part": net.face_part.cpu(), "kind": KIND, "design": net.design, **meta},
               str(path))


def load(path, map_location="cpu"):
    obj = torch.load(str(path), map_location=map_location)
    if obj.get("kind") != KIND:
        raise ValueError("not a PFO slim router checkpoint: " + str(path))
    net = SlimSR(obj["faces"], obj["tmpl"], obj["face_part"], **obj["design"])
    net.load_state_dict(obj["state_dict"], strict=True)
    return net.eval(), obj


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
        return prev["route_nodes"](net, "sr" if router == KIND else router, *args, **kwargs)

    def fast_route(net, router, out, select):
        return prev["fast_route"](net, "sr" if router == KIND else router, out, select)
    SM.ROUTERS = prev["routers"] + (KIND,)
    SM._load_router, SM.route_nodes, FAST.route = load_router, route_nodes, fast_route
    _STATE.update(registered=True, prev=prev)


def unregister():
    prev = _STATE["prev"]
    if prev is not None:
        SM.ROUTERS, SM._load_router, SM.route_nodes, FAST.route = (prev["routers"], prev["load"], prev["route_nodes"],
                                                                   prev["fast_route"])
    _STATE.update(registered=False, prev=None)


__all__ = ["KIND", "SlimSR", "build", "load", "register", "save_checkpoint", "unregister"]
