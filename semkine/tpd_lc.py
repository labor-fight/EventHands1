r"""TPD problem 1, long-context router LC (docs/EXPERIMENT_RECORDS.md#tpd-lc-prereg-20261105).

r_s2d (semkine/pfo_router.py SlimSR, stem "s2d") classifies the events of one 50 ms packet from an image of that packet
alone. LC adds the 200 ms of events before the packet as two more input channels and is fine-tuned from r_s2d in the
training form (the router learns to use the history; inference-only context was tried before, S45-SM R6C):

    image = [event_image(current packet) (4 ch, r_s2d's), history_image(previous 200 ms) (2 ch)]
    history_image = polarity-merged: time of the most recent history event at the pixel / 200 ms (1 = just before the
                    packet, 0 where none), log1p(count)
The stem's 1x1 convolution grows from 16 to 24 input channels (pixel unshuffle by 2); the 8 new columns start at zero,
so before fine-tuning LC is r_s2d exactly, whatever the history. Nodes, tokens, heads and decoding are r_s2d's: only the
current packet's events are classified. The history is causal (past events only); the router stage is exempt from the
asynchronous requirement (#tpd-prereg-20261105).

Data (training): B2's / r_s2d's router dataset instance with the current window fixed to PACKET_MS and, per sample, the
history events of [start - HIST_MS, start - 1] (clipped at the recording start) under the same geometric augmentation
realisation, their own keep draw and hot pixels at the same rate (a separate per-sample stream), no polarity augmentation
(the history channels merge polarities). `lc0` is the control: the same network and fine-tune with the history input
always empty.

Deployment: `install(checkpoint, tau, history)` puts LC behind the S45-SM routing hooks (PFA); the packet's history is
attached by `attach_history` from the sequence the engine is reading (`engine_hooks`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

from . import domrand as DR
from . import pfo_router as PR
from . import s45sm as SM
from . import s45sm_fast as FAST
from . import xag as XAG
from .dataset import SemKineDataset
from .encoder import EV_BATCH, EV_T, EV_X, EV_Y, event_tokens
from .events import EV_COLS, EventPacketBatch, collate_packets
from .s45rt_router import event_image
from .xyz_mesh import stride_sample

KIND = "tpd_lc"
HIST_MS = 200
HIST_CH = 2
PACKET_MS = 50
HIST_SALT = 51105


# --------------------------------------------------------------------------- network
def history_image(events, ptr, span_s, height=SM.HEIGHT, width=SM.WIDTH):
    """`(B, 2, H, W)`: the time of the most recent history event at the pixel / span (events in time order), log1p count."""
    B = int(ptr.numel() - 1)
    plane = height * width
    n = events.shape[0]
    if n == 0:
        return events.new_zeros((B, HIST_CH, height, width), dtype=torch.float32)
    b = events[:, EV_BATCH].long()
    x = events[:, EV_X].long().clamp(0, width - 1)
    y = events[:, EV_Y].long().clamp(0, height - 1)
    t = (events[:, EV_T].float() / float(span_s)).clamp(0.0, 1.0)
    pix = b * plane + y * width + x
    idx = torch.arange(n, device=events.device)
    last = torch.full((B * plane,), -1, dtype=torch.long, device=events.device)
    last.scatter_reduce_(0, pix, idx, reduce="amax", include_self=True)
    time = torch.where(last >= 0, t[last.clamp_min(0)], torch.zeros((), device=events.device))
    count = torch.zeros(B * plane, dtype=torch.float32, device=events.device).index_add_(
        0, pix, torch.ones(n, dtype=torch.float32, device=events.device))
    return torch.stack([time.view(B, height, width), torch.log1p(count).view(B, height, width)], 1)


class LCRouter(PR.SlimSR):
    """r_s2d with the history channels (see the module docstring)."""

    def __init__(self, faces, verts, face_part, hist_ms=HIST_MS, **design):
        super().__init__(faces, verts, face_part, **design)
        if self.design["stem"] != "s2d":
            raise ValueError("LC extends the s2d stem")
        c0 = self.design["trunk"][0]
        self.stem[0] = nn.Conv2d(4 * (4 + HIST_CH), c0, 1, bias=False)
        self.hist_ms = int(hist_ms)

    def forward(self, events, ptr, delta_t_s, hist_events, hist_ptr):
        B = int(ptr.numel() - 1)
        src, mask = stride_sample(ptr, SM.MAX_NODES)
        tok = event_tokens(events, ptr, delta_t_s, SM.HEIGHT, SM.WIDTH)
        image = torch.cat([event_image(events, ptr, delta_t_s), history_image(hist_events, hist_ptr, self.hist_ms * 1e-3)], 1)
        fmap = self.trunk(image.to(self.stem[0].weight.dtype))
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


def from_r_s2d(path, hist_ms=HIST_MS):
    """LC with r_s2d's weights; the stem's history columns are zero (LC == r_s2d before fine-tuning)."""
    base, _ = PR.load(path)
    net = LCRouter(base.faces, base.tmpl, base.face_part, hist_ms=hist_ms, **base.design)
    sd = base.state_dict()
    own = net.state_dict()
    for k, v in sd.items():
        if k == "stem.0.weight":
            w = torch.zeros_like(own[k])
            w[:, : v.shape[1]] = v
            own[k] = w
        else:
            own[k] = v.clone()
    net.load_state_dict(own, strict=True)
    return net


def save_checkpoint(net, path, **meta):
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "faces": net.faces.cpu(),
                "tmpl": net.tmpl.cpu(), "face_part": net.face_part.cpu(), "kind": KIND, "design": net.design,
                "hist_ms": net.hist_ms, **meta}, str(path))


def load(path, map_location="cpu"):
    obj = torch.load(str(path), map_location=map_location)
    if obj.get("kind") != KIND:
        raise ValueError("not a TPD LC router checkpoint: " + str(path))
    net = LCRouter(obj["faces"], obj["tmpl"], obj["face_part"], hist_ms=obj["hist_ms"], **obj["design"])
    net.load_state_dict(obj["state_dict"], strict=True)
    return net.eval(), obj


# --------------------------------------------------------------------------- data
@dataclass
class HistBatch:
    base: EventPacketBatch
    hist_events: torch.Tensor          # (N_h, 5) [packet, x, y, t from the history start (s), polarity]
    hist_ptr: torch.Tensor             # (B + 1,)

    @property
    def batch_size(self):
        return self.base.batch_size

    def to(self, device):
        return HistBatch(self.base.to(device), self.hist_events.to(device, non_blocking=True),
                         self.hist_ptr.to(device, non_blocking=True))


def collate_hist(items):
    base = collate_packets([it[0] for it in items])
    hist = [it[1] for it in items]
    counts = np.array([len(h) for h in hist], np.int64)
    ptr = np.zeros(len(hist) + 1, np.int64)
    np.cumsum(counts, out=ptr[1:])
    ev = np.zeros((int(ptr[-1]), EV_COLS), np.float32)
    if ptr[-1]:
        ev[:, EV_BATCH] = np.repeat(np.arange(len(hist), dtype=np.float32), counts)
        ev[:, 1:] = np.concatenate([h for h in hist if len(h)], 0)
    return HistBatch(base, torch.from_numpy(ev), torch.from_numpy(ptr))


class HistDataset(SemKineDataset):
    """The router dataset instance with a fixed current window and the history events (built by `upgrade`)."""

    def _setup_hist(self, hist_ms, packet_ms, empty):
        self.hist_ms, self.fixed_window, self.hist_empty = int(hist_ms), int(packet_ms), bool(empty)
        return self

    def history(self, idx, packet):
        if self.hist_empty:
            return np.zeros((0, 4), np.float32)
        si, end = int(self.index[idx][0]), int(packet.meta["end_idx"])
        h = self._handle(si)
        start = end - int(packet.meta["window_ms"]) + 1
        h0 = start - self.hist_ms                                   # time origin of the history channels
        a, b = max(h0, 0), start - 1
        if b < a:
            return np.zeros((0, 4), np.float32)
        rng = np.random.default_rng([self.seed, int(idx), HIST_SALT])
        xs, ys, ps, us = self._raw_events(h, b, b - a + 1)
        us = us + (a - h0) * 1000
        if self.domrand.enabled:
            dr = packet.meta["domrand"]
            if len(xs):
                keep = DR.transform_events(xs, ys, dr, h.camera_K, self.render_scale, self.width, self.height)
                m = DR.keep_mask(len(xs), dr.keep, rng)
                if m is not None:
                    keep = keep & m
                xs, ys, ps, us = xs[keep], ys[keep], ps[keep], us[keep]
            n_hot = int(rng.poisson(self.domrand.hot_pixel_rate * self.height * self.width * 2 * self.hist_ms)) \
                if self.domrand.hot_pixel_rate > 0 else 0
            hot = DR.sample_hot_pixels(n_hot, self.hist_ms, self.width, self.height, rng)
            if hot is not None:
                xs = np.concatenate([xs, hot[:, 1]])
                ys = np.concatenate([ys, hot[:, 2]])
                ps = np.concatenate([ps, hot[:, 3]])
                us = np.concatenate([us, hot[:, 0] * 1000 + rng.integers(0, 1000, len(hot))])
                order = np.argsort(us, kind="stable")
                xs, ys, ps, us = xs[order], ys[order], ps[order], us[order]
        out = np.empty((len(xs), 4), np.float32)
        if len(xs):
            out[:, 0], out[:, 1], out[:, 2], out[:, 3] = xs, ys, us.astype(np.float32) * 1e-6, ps
        return out

    def __getitem__(self, idx):
        packet = SemKineDataset.__getitem__(self, idx)
        return packet, self.history(idx, packet)


def upgrade(ds, hist_ms=HIST_MS, packet_ms=PACKET_MS, empty=False):
    if type(ds) is not SemKineDataset:
        raise TypeError("upgrade expects semkine.dataset.SemKineDataset")
    if ds.input_mode != "raw_packed" or not ds.train:
        raise ValueError("LC trains on the raw-packet training dataset")
    ds.__class__ = HistDataset
    return ds._setup_hist(hist_ms, packet_ms, empty)


# --------------------------------------------------------------------------- deployment hooks
_DEPLOY = {"net": None, "on": {}, "tau": None, "ctx": None, "empty": False}


def _net(device):
    net = _DEPLOY["on"].get(device)
    if net is None:
        net = _DEPLOY["net"].to(device)
        _DEPLOY["on"][device] = net
    return net


def history_events(ctx, end, packet_ms, hist_ms, device):
    """`(hist_events (N, 5), hist_ptr (2,))` of [end - packet_ms - hist_ms + 1, end - packet_ms] for one packet, times from
    the history origin, in the eval packet layout (no augmentation)."""
    from .eval_track import _window_events
    start = int(end) - packet_ms + 1
    h0 = start - hist_ms
    a, b = max(h0, 0), start - 1
    if b < a or _DEPLOY["empty"]:
        ev = np.zeros((0, EV_COLS), np.float32)
    else:
        ev = np.array(_window_events(ctx.events, ctx.offsets, ctx.tsub, b, b - a + 1), np.float32, copy=True)
        ev[:, EV_T] += (a - h0) * 1e-3
    return torch.from_numpy(ev).to(device), torch.tensor([0, len(ev)], dtype=torch.int64, device=device)


@torch.no_grad()
def lc_sources(batch, src, mask, *, require_depth=False):
    if require_depth:
        raise RuntimeError("LC routes faces only")
    net = _net(src.device)
    hist, hptr = getattr(batch, "hist_events", None), getattr(batch, "hist_ptr", None)
    if hist is None:
        raise RuntimeError("LC needs the packet's history (attach_history)")
    out = net(batch.events, batch.ptr, batch.delta_t_s, hist, hptr)
    if not torch.equal(out["src"], src) or not torch.equal(out["mask"], mask):
        raise RuntimeError("router nodes are not S45's sampled events")
    routed = mask & (torch.sigmoid(out["onhand"].float()) >= _DEPLOY["tau"])
    face, bary = FAST.route(net, "sr", out, routed)
    zero = torch.zeros_like(face)
    return {"xag_source_index": zero, "xag_source_ms": zero, "xag_part": net.face_part[face], "xag_face": face,
            "xag_bary": bary.to(torch.float32), "xag_valid": routed & mask}


def install(checkpoint, tau=0.5, empty=False):
    """Load LC for deployment. The source sampler is switched to `lc_sources` after the S45-SM / PFA hooks are in place
    (`engine_hooks` does it after the engine's `nets.load`; `use()` for code that installed the hooks itself)."""
    net, obj = load(checkpoint)
    for p in net.parameters():
        p.requires_grad_(False)
    _DEPLOY.update(net=net, on={}, tau=float(tau), empty=bool(empty))
    return {"router": KIND, "checkpoint": str(checkpoint), "hist_ms": int(obj["hist_ms"]), "tau": float(tau),
            "empty_history": bool(empty)}


def use():
    if _DEPLOY["net"] is None or SM.installed() is None:
        raise RuntimeError("install LC and the S45-SM / PFA hooks first")
    XAG.sample_sources = lc_sources


def engine_hooks():
    """Make PFO's frozen engine hand LC the packet's history: remember the sequence context it opens
    (`tools.pfo.nets.sequence`) and attach the history to every packet it builds (`attach_xag_sources`, called by
    `Net.step` after `nets.load` set it to the identity for deployed routes)."""
    from tools.pfo import nets as N
    from . import eval_track as ET
    native_seq, native_load = N.sequence, N.load

    def sequence(*args, **kwargs):
        ctx = native_seq(*args, **kwargs)
        _DEPLOY["ctx"] = ctx
        return ctx

    def attach(batch, cfg, seq, offsets, end, window, device):
        ctx = _DEPLOY["ctx"]
        if ctx is None or ctx.seq != seq or ctx.offsets is not offsets:
            raise RuntimeError("LC history: the engine's sequence context is not the packet's")
        batch.hist_events, batch.hist_ptr = history_events(ctx, end, int(window), _DEPLOY["net"].hist_ms, device)
        return batch

    def load(*args, **kwargs):
        net = native_load(*args, **kwargs)
        ET.attach_xag_sources = attach
        XAG.sample_sources = lc_sources
        return net
    N.sequence, N.load = sequence, load


__all__ = ["HIST_CH", "HIST_MS", "HistBatch", "HistDataset", "KIND", "LCRouter", "PACKET_MS", "collate_hist",
           "engine_hooks", "from_r_s2d", "history_events", "history_image", "install", "lc_sources", "load",
           "save_checkpoint", "upgrade", "use"]
