r"""PCR prior-conditioned router (docs/EXPERIMENT_RECORDS.md#pcr-prereg-20261105): PFO's r_s2d (semkine/pfo_router.py
SlimSR, stem "s2d") with one more input, the previous state's part map, so the router can use the hand's 3-D layout to
decide which finger / phalanx an event belongs to where the event image alone is ambiguous (foreshortened fingers).

Prior map: MANO FK of the previous state (the model's decoding, ManoLayer add_mean=False + decode_to_mano_inputs
"mano_full_axis_angle"), the 778 vertices projected with camera_K x 0.375 (the model's render scale), bilinearly splatted
into 16 part channels (each vertex's dominant LBS part) at half resolution (90 x 120, the s2d stem's grid: half-res cell
(i, j) covers full-res pixels 2j, 2j+1), blurred with a 3 x 3 Gaussian (sigma 1) and mapped to 1 - exp(-density).
Trunk: x = stem_conv(pixel_unshuffle(event image)) + prior_in(prior map), then r_s2d's GroupNorm / ReLU / stages / FPN /
head unchanged. `prior_in` (16 -> c0, 1 x 1, no bias) is built inside an isolated RNG scope and zero-initialised, so the
constructor draws r_s2d's initial weights and, while `prior_in` is zero, the outputs equal SlimSR's exactly.

Routing / decoding are r_s2d's (part argmax, nearest-centroid face in the part, clipped barycentrics). `register()` teaches
the S45-SM hooks the kind "pcr"; `install(checkpoint, tau)` installs PFA with this router and the prior-aware source
sampler `pcr_sources` (the sampler reads batch.prev_state / betas / camera_K, which every packet carries).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import pfo_router as PR
from . import s45sm as SM
from . import s45sm_fast as FAST
from . import xag as XAG
from .encoder import EV_X, EV_Y, event_tokens
from .s45rt_router import event_image
from .xyz_mesh import stride_sample

KIND = "pcr"
N_PRIOR = 16
RENDER_SCALE = 0.375
H2, W2 = SM.HEIGHT // 2, SM.WIDTH // 2
_STATE = {"registered": False, "prev": None, "mano": {}, "installed": False}


class PCRNet(PR.SlimSR):
    def __init__(self, faces, verts, face_part, vert_part, **design):
        super().__init__(faces, verts, face_part, **design)
        if self.design["stem"] != "s2d":
            raise ValueError("PCR builds on the s2d stem")
        c0 = self.stem[0].out_channels
        with torch.random.fork_rng(devices=[]):
            self.prior_in = nn.Conv2d(N_PRIOR, c0, 1, bias=False)
        nn.init.zeros_(self.prior_in.weight)
        self.register_buffer("vert_part", vert_part.long().clone(), persistent=False)

    def trunk_prior(self, image, prior):
        x = F.pixel_unshuffle(image, 2)
        x = self.stem[0](x) + self.prior_in(prior.to(x.dtype))
        x = self.stem[2](self.stem[1](x))
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

    def head_out(self, fmap, events, ptr, tok, src, mask):
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
        yy = (node_ev[:, EV_Y].long() * h4 // SM.HEIGHT).clamp(0, h4 - 1)
        xx = (node_ev[:, EV_X].long() * w4 // SM.WIDTH).clamp(0, w4 - 1)
        bb = torch.arange(B, device=events.device).repeat_interleave(src.shape[1])
        feat = fmap[bb, :, yy, xx]
        z = self.head(torch.cat([feat.to(tok.dtype), tok[flat]], -1)).view(B, src.shape[1], -1)
        z = z * mask.unsqueeze(-1).to(z.dtype)
        return {"z": z, "part": self.part(z), "coord": self.coord(z), "onhand": self.onhand(z).squeeze(-1),
                "src": src, "mask": mask}

    def forward(self, events, ptr, delta_t_s, prior):
        """Training forward (S45's nodes; native tokens / event image)."""
        src, mask = stride_sample(ptr, SM.MAX_NODES)
        tok = event_tokens(events, ptr, delta_t_s, SM.HEIGHT, SM.WIDTH)
        fmap = self.trunk_prior(event_image(events, ptr, delta_t_s).to(self.stem[0].weight.dtype), prior)
        return self.head_out(fmap, events, ptr, tok, src, mask)


# --------------------------------------------------------------------------- prior map
def _mano(device):
    key = str(device)
    if key not in _STATE["mano"]:
        from pathlib import Path
        from mano_layer import ManoLayer
        npz = Path(__file__).resolve().parents[1] / "assets/mano_right.npz"
        _STATE["mano"][key] = ManoLayer(str(npz), add_mean=False).to(device).eval()
    return _STATE["mano"][key]


_BLUR = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0


@torch.no_grad()
def project_prior(prev, betas, camera_K):
    """`(u, v, z)` full-resolution pixel coordinates of the prior's 778 vertices, `(B, 778)` each."""
    from pose_repr import decode_to_mano_inputs
    dev = prev.device
    with torch.autocast(device_type=dev.type, enabled=False):
        m = _mano(dev)
        B = prev.shape[0]
        dec = decode_to_mano_inputs(prev.float(), "mano_full_axis_angle", m.hands_components, m.hands_mean)
        V, _ = m(betas.float().expand(B, -1), dec["global_orient"], dec["local_full_aa"], dec["transl"])
        K = camera_K.float().expand(B, -1, -1) * RENDER_SCALE
        z = V[..., 2]
        zs = z.clamp_min(1e-3)
        u = K[:, 0, 0, None] * V[..., 0] / zs + K[:, 0, 2, None]
        v = K[:, 1, 1, None] * V[..., 1] / zs + K[:, 1, 2, None]
        return u, v, z


@torch.no_grad()
def prior_maps(prev, betas, camera_K, vert_part):
    """`(B, 16, 90, 120)` part map of the prior (see the module docstring), float32."""
    dev = prev.device
    with torch.autocast(device_type=dev.type, enabled=False):
        u, v, z = project_prior(prev, betas, camera_K)
        B, Vn = u.shape
        u2, v2 = (u - 0.5) / 2.0, (v - 0.5) / 2.0
        x0, y0 = torch.floor(u2), torch.floor(v2)
        fx, fy = u2 - x0, v2 - y0
        ch = vert_part.view(1, Vn).expand(B, Vn)
        bidx = torch.arange(B, device=dev).view(B, 1).expand(B, Vn)
        dens = torch.zeros(B * N_PRIOR * H2 * W2, device=dev)
        ok_z = z > 1e-3
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = (x0 + dx).long(), (y0 + dy).long()
            ok = ok_z & (xi >= 0) & (xi < W2) & (yi >= 0) & (yi < H2)
            idx = ((bidx * N_PRIOR + ch) * H2 + yi.clamp(0, H2 - 1)) * W2 + xi.clamp(0, W2 - 1)
            dens.index_add_(0, idx[ok], w[ok])
        dens = dens.view(B * N_PRIOR, 1, H2, W2)
        dens = F.conv2d(F.pad(dens, (1, 1, 1, 1)), _BLUR.to(dev).view(1, 1, 3, 3)).view(B, N_PRIOR, H2, W2)
        return 1.0 - torch.exp(-dens)


# --------------------------------------------------------------------------- build / save / load
def build(npz, **design):
    with np.load(str(npz), allow_pickle=False) as z:
        verts = torch.tensor(np.asarray(z["v_template"], np.float32))
        faces = torch.tensor(np.asarray(z["f"], np.int64))
        w = np.asarray(z["weights"])
        part = torch.from_numpy(SM.face_part_table(w, np.asarray(z["f"], np.int64)))
        vpart = torch.from_numpy(w.argmax(1).astype(np.int64))
    return PCRNet(faces, verts, part, vpart, **design)


def save_checkpoint(net, path, **meta):
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "faces": net.faces.cpu(),
                "tmpl": net.tmpl.cpu(), "face_part": net.face_part.cpu(), "vert_part": net.vert_part.cpu(), "kind": KIND,
                "design": net.design, "prior": {"channels": N_PRIOR, "res": [H2, W2], "render_scale": RENDER_SCALE},
                **meta}, str(path))


def load(path, map_location="cpu"):
    obj = torch.load(str(path), map_location=map_location)
    if obj.get("kind") != KIND:
        raise ValueError("not a PCR checkpoint: " + str(path))
    net = PCRNet(obj["faces"], obj["tmpl"], obj["face_part"], obj["vert_part"], **obj["design"])
    net.load_state_dict(obj["state_dict"], strict=True)
    return net.eval(), obj


# --------------------------------------------------------------------------- inference hooks
def pcr_forward(net, batch, src, mask):
    """`FAST.router_forward` with the prior: the memoized tokens / event image, S45's nodes, no CUDA graph."""
    prior = prior_maps(batch.prev_state, batch.betas, batch.camera_K, net.vert_part)
    tok = FAST.event_tokens(batch.events, batch.ptr, batch.delta_t_s, SM.HEIGHT, SM.WIDTH)
    image = FAST.event_image(batch.events, batch.ptr, batch.delta_t_s).to(net.stem[0].weight.dtype)
    fmap = net.trunk_prior(image, prior)
    return net.head_out(fmap, batch.events, batch.ptr, tok, src, mask)


@torch.no_grad()
def pcr_sources(batch, src, mask, *, require_depth=False):
    if require_depth:
        raise RuntimeError("PCR routes faces only")
    st = SM._STATE
    net = SM._net(src.device)
    out = pcr_forward(net, batch, src, mask)
    routed = mask & (torch.sigmoid(out["onhand"].float()) >= st["onhand_tau"])
    face, bary = FAST.route(net, "sr", out, routed)
    zero = torch.zeros_like(face)
    return {"xag_source_index": zero, "xag_source_ms": zero, "xag_part": net.face_part[face], "xag_face": face,
            "xag_bary": bary.to(torch.float32), "xag_valid": routed}


def register():
    if _STATE["registered"]:
        return
    prev = {"routers": SM.ROUTERS, "load": SM._load_router, "route_nodes": SM.route_nodes, "fast_route": FAST.route,
            "fast_install": FAST.install}

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

    def fast_install(router, checkpoint, assoc, onhand_tau=0.5):
        state = prev["fast_install"](router, checkpoint, assoc, onhand_tau)
        if router == KIND:
            XAG.sample_sources = pcr_sources
            state = {**state, "sample_sources": "semkine.pcr_router.pcr_sources"}
        return state
    SM.ROUTERS = prev["routers"] + (KIND,)
    SM._load_router, SM.route_nodes, FAST.route, FAST.install = load_router, route_nodes, fast_route, fast_install
    _STATE.update(registered=True, prev=prev)


def unregister():
    prev = _STATE["prev"]
    if prev is not None:
        SM.ROUTERS, SM._load_router, SM.route_nodes = prev["routers"], prev["load"], prev["route_nodes"]
        FAST.route, FAST.install = prev["fast_route"], prev["fast_install"]
    _STATE.update(registered=False, prev=None)


__all__ = ["H2", "KIND", "N_PRIOR", "PCRNet", "W2", "build", "load", "pcr_forward", "pcr_sources", "prior_maps",
           "project_prior", "register", "save_checkpoint", "unregister"]
