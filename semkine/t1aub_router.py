r"""T1A-UB router variants: S45-SM's depthwise patch router `dw` (semkine/s45sm_dw.py: T0P's design, depthwise-separable
stages 32 / 64 / 96 / 128, FPN 40, head 72, 137,291 parameters) with exactly one change each:

  dwt   time-binned input: the event image gets four more channels, per polarity log1p(count) in the first and in the
        second half of the packet, so the trunk sees where the edges were early and late (motion direction)
  dwr   a finer feature level: the lateral of the first stage (1/2 resolution) is merged into the FPN and the per-node
        feature is sampled at 1/2 resolution (the middle-phalanx confusions are narrower than a 1/4-resolution cell)
  dweg  T1a's event graph inside the router: the S36 causal kNN EventGNN operator (k 8, window 32, 3 rounds, as T1a's, at
        hidden 16) on the node tokens; its node feature joins the per-node head's input next to the FPN feature and token7

Routing is T0P's rule (most probable patch; its face nearest the predicted rest-template point). Opt-in: `register()`
teaches the S45-SM hooks the kinds in KINDS (built / loaded here, routed as "t0p") and a fast-path forward for them;
`unregister()` restores what it wrapped. Composes with semkine/s45sm_dw.py / s45sm_srp.py / s45sm_hier.py registered
before or after.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from . import s45rt_router as RT
from . import s45sm as SM
from . import s45sm_dw as DW
from . import s45sm_fast as FAST
from .encoder import EV_BATCH, EV_P, EV_T, EV_X, EV_Y, _packet_dt, event_tokens
from .event_gnn import EventGNN
from .s45tr_t1a import node_features
from .xyz_mesh import stride_sample

KINDS = ("dwt", "dwr", "dweg")
EG = {"HIDDEN": 16, "K": 8, "WINDOW": 32, "LAYERS": 3, "T_SCALE": 1.0}
_STATE = {"registered": False, "prev": None}


def event_image8(events, ptr, delta_t_s, height=RT.HEIGHT, width=RT.WIDTH):
    """`(B, 8, H, W)`: RT.event_image's four channels, then per polarity log1p(count) in the packet's first / second half."""
    base = FAST.event_image(events, ptr, delta_t_s, height, width)
    B = int(ptr.numel() - 1)
    n = events.shape[0]
    halves = events.new_zeros((B * 4 * height * width,), dtype=torch.float32)
    if n:
        dt = _packet_dt(delta_t_s, events).float()
        b = events[:, EV_BATCH].long()
        x = events[:, EV_X].long().clamp(0, width - 1)
        y = events[:, EV_Y].long().clamp(0, height - 1)
        p = (events[:, EV_P] > 0.5).long()
        late = ((events[:, EV_T].float() / dt) >= 0.5).long()
        halves.index_add_(0, ((b * 4 + p * 2 + late) * height + y) * width + x, torch.ones(n, device=events.device))
    return torch.cat([base, torch.log1p(halves).view(B, 4, height, width)], 1)


class VariantNet(RT.T0PNet):
    """`dw` with the one change of its kind (module docstring)."""

    def __init__(self, faces, verts, patch_of_face, face_part, kind):
        if kind not in KINDS:
            raise ValueError("router variant kind in " + str(KINDS))
        with DW.widths():
            super().__init__(faces, verts, patch_of_face, face_part)
        self.kind = kind
        if kind == "dwt":
            self.stem[0] = nn.Conv2d(8, DW.TRUNK[0], 3, 1, 1, bias=False)
        elif kind == "dwr":
            self.lateral_hi = nn.Conv2d(DW.TRUNK[0], DW.FPN, 1)
        else:
            self.evgnn = EventGNN(height=RT.HEIGHT, width=RT.WIDTH, hidden=EG["HIDDEN"], feat_dim=EG["HIDDEN"], k=EG["K"],
                                  n_layers=EG["LAYERS"], max_nodes=RT.MAX_NODES, window=EG["WINDOW"],
                                  t_scale=EG["T_SCALE"], node_attrs="token7", readout=False)
            self.head[0] = nn.Linear(DW.FPN + 7 + EG["HIDDEN"], DW.HEAD)

    def image(self, events, ptr, delta_t_s):
        if self.kind == "dwt":
            return event_image8(events, ptr, delta_t_s)
        return FAST.event_image(events, ptr, delta_t_s)

    def trunk(self, image):
        x = self.stem(image)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        top = self.lateral[2](feats[3])
        for level in (1, 0):
            lat = self.lateral[level](feats[level + 1])
            top = F.interpolate(top, size=lat.shape[-2:], mode="nearest") + lat
        if self.kind == "dwr":
            lat = self.lateral_hi(feats[0])
            top = F.interpolate(top, size=lat.shape[-2:], mode="nearest") + lat
        return self.smooth(top)

    def head_out(self, fmap, events, ptr, delta_t_s, tok, src, mask):
        B = int(ptr.numel() - 1)
        if tok.shape[0] == 0:
            tok = events.new_zeros((1, 7))
            ev = events.new_zeros((1, events.shape[1]))
            src = torch.zeros_like(src)
        else:
            ev = events
        flat = src.reshape(-1)
        node_ev = ev[flat]
        h4, w4 = fmap.shape[-2:]
        yy = (node_ev[:, EV_Y].long() * h4 // RT.HEIGHT).clamp(0, h4 - 1)
        xx = (node_ev[:, EV_X].long() * w4 // RT.WIDTH).clamp(0, w4 - 1)
        bb = torch.arange(B, device=events.device).repeat_interleave(src.shape[1])
        feat = fmap[bb, :, yy, xx]
        parts = [feat.to(tok.dtype), tok[flat]]
        if self.kind == "dweg":
            ntok = tok[flat].view(B, src.shape[1], -1) * mask.unsqueeze(-1).to(tok.dtype)
            g = node_features(self.evgnn, ntok, mask)
            parts.append(g.reshape(B * src.shape[1], -1).to(tok.dtype))
        z = self.head(torch.cat(parts, -1)).view(B, src.shape[1], -1)
        z = z * mask.unsqueeze(-1).to(z.dtype)
        return {"z": z, "patch": self.patch(z), "coord": self.coord(z), "onhand": self.onhand(z).squeeze(-1),
                "src": src, "mask": mask}

    def forward(self, events, ptr, delta_t_s):
        src, mask = stride_sample(ptr, RT.MAX_NODES)
        tok = event_tokens(events, ptr, delta_t_s, RT.HEIGHT, RT.WIDTH)
        fmap = self.trunk(self.image(events, ptr, delta_t_s).to(self.stem[0].weight.dtype))
        return self.head_out(fmap, events, ptr, delta_t_s, tok, src, mask)


def fast_forward(net, events, ptr, delta_t_s, src, mask):
    """FAST.router_forward for a variant: S45's nodes, the memoized tokens, the trunk replayed from a CUDA graph."""
    tok = FAST.event_tokens(events, ptr, delta_t_s, SM.HEIGHT, SM.WIDTH)
    fmap = FAST.trunk(net, net.image(events, ptr, delta_t_s).to(net.stem[0].weight.dtype))
    return net.head_out(fmap, events, ptr, delta_t_s, tok, src, mask)


def build(npz, kind):
    verts, faces = RT.template(npz)
    patch, part, _ = RT.patch_table(npz)
    return VariantNet(faces, verts, torch.from_numpy(patch), torch.from_numpy(part), kind)


def save_checkpoint(net, path, **meta):
    RT.save_checkpoint(net, path, kind=net.kind, block="dw", trunk=list(DW.TRUNK), fpn=DW.FPN, head=DW.HEAD,
                       variant=net.kind, **meta)


def load(path, map_location="cpu"):
    obj = torch.load(str(path), map_location=map_location)
    if obj.get("variant") not in KINDS or (tuple(obj["trunk"]), obj["fpn"], obj["head"]) != (DW.TRUNK, DW.FPN, DW.HEAD):
        raise ValueError("not a T1A-UB router variant checkpoint: " + str(path))
    net = VariantNet(obj["faces"], obj["tmpl"], obj["patch_of_face"], obj["face_part"], obj["variant"])
    net.load_state_dict(obj["state_dict"], strict=True)
    return net.eval(), obj


def register():
    if _STATE["registered"]:
        return
    prev = {"routers": SM.ROUTERS, "load": SM._load_router, "route_nodes": SM.route_nodes, "fast_route": FAST.route,
            "fast_forward": FAST.router_forward}

    def load_router(router, checkpoint):
        if router not in KINDS:
            return prev["load"](router, checkpoint)
        net, obj = load(checkpoint)
        if obj["variant"] != router:
            raise ValueError("router kind differs from the checkpoint's variant")
        for p in net.parameters():
            p.requires_grad_(False)
        return net.eval()

    def route_nodes(net, router, *args, **kwargs):
        return prev["route_nodes"](net, "t0p" if router in KINDS else router, *args, **kwargs)

    def fast_route(net, router, out, select):
        return prev["fast_route"](net, "t0p" if router in KINDS else router, out, select)

    def router_forward(net, *args, **kwargs):
        if isinstance(net, VariantNet):
            return fast_forward(net, *args, **kwargs)
        return prev["fast_forward"](net, *args, **kwargs)
    SM.ROUTERS = prev["routers"] + KINDS
    SM._load_router, SM.route_nodes, FAST.route, FAST.router_forward = load_router, route_nodes, fast_route, router_forward
    _STATE.update(registered=True, prev=prev)


def unregister():
    prev = _STATE["prev"]
    if prev is not None:
        SM.ROUTERS, SM._load_router, SM.route_nodes = prev["routers"], prev["load"], prev["route_nodes"]
        FAST.route, FAST.router_forward = prev["fast_route"], prev["fast_forward"]
    _STATE.update(registered=False, prev=None)


__all__ = ["EG", "KINDS", "VariantNet", "build", "event_image8", "fast_forward", "load", "register", "save_checkpoint",
           "unregister"]
