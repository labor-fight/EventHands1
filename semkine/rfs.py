r"""Route-first sampling (RFS) for EvHandTrack (opt-in; docs/EXPERIMENT_RECORDS.md#real-rfs-20261112).

EvHandTrack samples its nodes (stride rule, budget 2048 per packet; the streaming rule at readings) from every event of
the packet and routes them afterwards; the nodes the router calls off-hand go to S45's residual slot. On the real DAVIS
data most events are off-hand (desk scene, forearm, noise), so most of the budget lands on them. RFS routes first:

  1. the router (r_s2d, prior-free, the deployed one) evaluates every event of the packet: on-hand = p >= tau;
  2. eligible = on-hand events, plus off-hand events within NEAR px (square window, canvas pixels) of an on-hand event
     of the same packet (the hand's vicinity: they keep feeding the residual slot as before);
  3. nodes = the same sampling rule run over the eligible events only.

Everything else -- event tokens, the router's routes at the nodes, PFA, the residual slot, the trunk -- is unchanged, and
no previous state, label or K enters the selection. `install(tau, near_px)` patches semkine.xyz_mesh.stride_sample
(every model of this family looks it up at call time) and registers every EventPacketBatch at construction so the
sampler can find the batch of a given `ptr`; `install_stream(...)` patches semkine.tpd_str.prepare_packet (streaming
readings). `uninstall()` restores both.
"""
from __future__ import annotations

import weakref

import torch
import torch.nn.functional as F

from . import events as EVM
from . import xyz_mesh as XM

_STATE = {"on": False, "tau": None, "near": None, "native_stride": None, "native_init": None, "native_prepare": None,
          "chunk": 64, "max_rows": None, "stats": {"packets": 0, "events": 0, "eligible": 0, "onhand": 0}}
_REG: dict = {}


def _register_init(native):
    def __init__(self, *args, **kwargs):
        native(self, *args, **kwargs)
        ptr = getattr(self, "ptr", None)
        if torch.is_tensor(ptr):
            _REG[id(ptr)] = weakref.ref(self)
            if len(_REG) > 4096:                                   # drop dead entries now and then
                for k in [k for k, r in _REG.items() if r() is None]:
                    _REG.pop(k, None)
    return __init__


def _batch_of(ptr):
    r = _REG.get(id(ptr))
    b = r() if r is not None else None
    if b is None or b.ptr is not ptr:
        raise RuntimeError("RFS: no EventPacketBatch registered for this ptr (batch built without __init__?)")
    return b


@torch.no_grad()
def eligible(events, ptr, delta_t_s, tau=None, near=None):
    """`(eligible (N,) bool, onhand (N,) bool)` for every event of a packet batch (router on all events)."""
    from . import s45sm as SM
    from . import s45sm_fast as FAST
    tau = _STATE["tau"] if tau is None else tau
    near = _STATE["near"] if near is None else near
    dev = events.device
    n_all = int(events.shape[0])
    on = torch.zeros(n_all, dtype=torch.bool, device=dev)
    if n_all == 0:
        return on, on
    net = SM._net(dev)
    B = int(ptr.numel() - 1)
    counts = (ptr[1:] - ptr[:-1]).clamp(min=0)
    for b0, b1 in _spans(counts, B):
        a0, a1 = int(ptr[b0]), int(ptr[b1])
        if a1 <= a0:
            continue
        sub_ptr = ptr[b0:b1 + 1] - ptr[b0]
        sub_ev = events[a0:a1].clone()
        sub_ev[:, EVM.EV_BATCH] = sub_ev[:, EVM.EV_BATCH] - b0
        c = counts[b0:b1]
        nmax = max(int(c.max()), 1)
        pos = torch.arange(nmax, device=dev)
        mask = pos.unsqueeze(0) < c.unsqueeze(1)
        src = (sub_ptr[:-1].unsqueeze(1) + pos.unsqueeze(0)).clamp(max=max(a1 - a0 - 1, 0))
        out = FAST.router_forward(net, sub_ev, sub_ptr, delta_t_s[b0:b1], src, mask)
        p = torch.sigmoid(out["onhand"].float())
        on[a0 + src[mask]] = p[mask] >= float(tau)
    FAST._MEMO.update(key=None, value=None)                          # our sub-batch tensors die here: no stale key
    if not near or near <= 0:
        return on, on
    from . import s45sm as SM2
    Hh, Ww = int(SM2.HEIGHT), int(SM2.WIDTH)
    bidx = events[:, EVM.EV_BATCH].long()
    x = events[:, EVM.EV_X].long().clamp(0, Ww - 1)
    y = events[:, EVM.EV_Y].long().clamp(0, Hh - 1)
    k = 2 * int(near) + 1
    elig = on.clone()
    for b0 in range(0, B, _STATE["chunk"]):
        b1 = min(B, b0 + _STATE["chunk"])
        a0, a1 = int(ptr[b0]), int(ptr[b1])
        if a1 <= a0:
            continue
        img = torch.zeros(b1 - b0, 1, Hh, Ww, device=dev)
        sel = on[a0:a1]
        img[bidx[a0:a1][sel] - b0, 0, y[a0:a1][sel], x[a0:a1][sel]] = 1.0
        img = F.max_pool2d(img, (1, k), stride=1, padding=(0, int(near)))
        img = F.max_pool2d(img, (k, 1), stride=1, padding=(int(near), 0))
        elig[a0:a1] |= img[bidx[a0:a1] - b0, 0, y[a0:a1], x[a0:a1]] > 0
    return elig, on


def _spans(counts, B):
    """Packet chunks for the router pass: `chunk` packets each; with `max_rows` set (a memory setting), a chunk whose
    padded size (packets x its largest packet's events) exceeds it is halved until it fits or holds one packet. The
    router's per-event output does not depend on the chunking beyond floating-point rounding (eval-mode BatchNorm,
    one image per packet, a per-event head)."""
    out, b0, cap = [], 0, _STATE.get("max_rows")
    while b0 < B:
        b1 = min(B, b0 + _STATE["chunk"])
        if cap:
            while b1 - b0 > 1 and (b1 - b0) * int(counts[b0:b1].max()) > int(cap):
                b1 = b0 + (b1 - b0) // 2
        out.append((b0, b1))
        b0 = b1
    return out


def stride_over(flags, ptr, max_nodes):
    """`stride_sample`'s rule run over the flagged events of each packet: `(src (B, N), mask (B, N))`."""
    dev = ptr.device
    B = int(ptr.numel() - 1)
    idx = torch.nonzero(flags).squeeze(1)                              # grouped by packet, arrival order
    bel = torch.bucketize(idx, ptr[1:], right=True) if idx.numel() else idx
    c = torch.bincount(bel, minlength=B)[:B] if idx.numel() else torch.zeros(B, dtype=torch.long, device=dev)
    first = torch.zeros(B, dtype=torch.long, device=dev)
    first[1:] = torch.cumsum(c, 0)[:-1]
    n = c.clamp(max=int(max_nodes))
    N = max(int(n.max()) if n.numel() else 0, 1)
    pos = torch.arange(N, device=dev)
    mask = pos.unsqueeze(0) < n.unsqueeze(1)
    step = c.to(torch.float32) / n.clamp(min=1).to(torch.float32)
    rank = (pos.unsqueeze(0).to(torch.float32) * step.unsqueeze(1)).long()
    rank = torch.minimum(rank, (c - 1).clamp(min=0).unsqueeze(1))
    gi = (first.unsqueeze(1) + rank).clamp(max=max(idx.numel() - 1, 0))
    src = idx[gi] if idx.numel() else torch.zeros(B, N, dtype=torch.long, device=dev)
    src = torch.where(mask, src, torch.zeros_like(src))
    return src, mask


def stride_sample_rfs(ptr, max_nodes):
    b = _batch_of(ptr)
    m = _STATE.get("memo")
    if m is not None and m[0]() is ptr and m[1]() is b.events and m[2] == int(max_nodes) and m[3] == b.events._version:
        return m[4]
    elig, on = eligible(b.events, b.ptr, b.delta_t_s)
    import os
    if os.environ.get("RFS_DEBUG") and _STATE["stats"]["packets"] < 3 * int(ptr.numel() - 1):
        ev = b.events
        print("RFS_DEBUG batch:", {"B": int(ptr.numel() - 1), "events": int(ev.shape[0]),
              "x": [float(ev[:, EVM.EV_X].min()), float(ev[:, EVM.EV_X].max())],
              "y": [float(ev[:, EVM.EV_Y].min()), float(ev[:, EVM.EV_Y].max())],
              "t": [float(ev[:, EVM.EV_T].min()), float(ev[:, EVM.EV_T].max())],
              "p": [float(ev[:, EVM.EV_P].min()), float(ev[:, EVM.EV_P].max())],
              "dt": [float(b.delta_t_s.min()), float(b.delta_t_s.max())],
              "onhand": float(on.float().mean()), "eligible": float(elig.float().mean())}, flush=True)
        from . import s45sm as SMd
        net = SMd._net(ev.device)
        print("RFS_DEBUG router:", {"kind": SMd._STATE.get("router"), "ckpt": SMd._STATE.get("checkpoint"),
              "training": bool(net.training), "bn_training": sorted({m.training for m in net.modules()}),
              "dtype": str(next(net.parameters()).dtype), "tau": SMd._STATE.get("onhand_tau"),
              "autocast": torch.is_autocast_enabled()}, flush=True)
    st = _STATE["stats"]
    st["packets"] += int(ptr.numel() - 1)
    st["events"] += int(b.events.shape[0])
    st["eligible"] += int(elig.sum())
    st["onhand"] += int(on.sum())
    res = stride_over(elig, ptr, max_nodes)
    _STATE["memo"] = (weakref.ref(ptr), weakref.ref(b.events), int(max_nodes), b.events._version, res)
    return res


def install(tau=0.5, near_px=16, packet=True, max_rows=None):
    """packet=True (training, packet-form readings): patch stride_sample. packet=False (streaming readings): set the rule
    only; install_stream() then patches tpd_str.prepare_packet (tpd_str.install owns stride_sample there).
    max_rows: memory cap of the router pass's padded chunk (see `_spans`; None = 64-packet chunks as before)."""
    if _STATE["on"]:
        raise RuntimeError("RFS already installed")
    _STATE.update(on=True, tau=float(tau), near=int(near_px), max_rows=int(max_rows) if max_rows else None)
    if packet:
        _STATE.update(native_stride=XM.stride_sample, native_init=EVM.EventPacketBatch.__init__)
        EVM.EventPacketBatch.__init__ = _register_init(_STATE["native_init"])
        XM.stride_sample = stride_sample_rfs
        # the routers sample their own nodes through names bound at import (S45-SM's routing checks that they equal
        # the model's): rebind those too, so model and router draw the same (memoized) route-first nodes
        _STATE["bound"] = []
        import importlib
        for name in ("semkine.s45sm", "semkine.pfo_router"):
            mod = importlib.import_module(name)
            if getattr(mod, "stride_sample", None) is _STATE["native_stride"]:
                _STATE["bound"].append(mod)
                mod.stride_sample = stride_sample_rfs
    rec = {"rfs": True, "tau": float(tau), "near_px": int(near_px), "packet": bool(packet)}
    if max_rows:
        rec["max_rows"] = int(max_rows)
    return rec


def install_stream():
    """Streaming readings (semkine/tpd_str.py): the streaming rule over the eligible events of the packet and of its
    history (the history window is classified as its own packet, i.e. as the previous packet was)."""
    from . import tpd_str as TF
    if not _STATE["on"]:
        raise RuntimeError("install() first")
    _STATE["native_prepare"] = TF.prepare_packet

    def prepare_packet(batch, events, offsets, tsub, end, window=TF.WINDOW_MS, budget=TF.BUDGET):
        start = int(end) - int(window) + 1
        dev = batch.events.device
        hist = torch.from_numpy(TF.history_events(events, offsets, tsub, start)).to(dev)
        hp = torch.tensor([0, hist.shape[0]], dtype=torch.long, device=dev)
        c0 = torch.tensor([int(offsets[start])], dtype=torch.long, device=dev)
        fp, _ = eligible(batch.events, batch.ptr, batch.delta_t_s)
        if hist.shape[0]:
            hev = hist.clone()
            hev[:, EVM.EV_T] = hev[:, EVM.EV_T] + float(window) * 1e-3      # the window's own time origin
            fh, _ = eligible(hev, hp, batch.delta_t_s[:1])
        else:
            fh = torch.zeros(0, dtype=torch.bool, device=dev)
        idx = torch.nonzero(fp).squeeze(1)
        ev_on = batch.events[idx]
        ptr_on = torch.tensor([0, int(idx.numel())], dtype=torch.long, device=dev)
        hist_on = hist[fh]
        hp_on = torch.tensor([0, int(hist_on.shape[0])], dtype=torch.long, device=dev)
        # the streaming rule over the eligible sub-stream; the counter phase stays the stream's own (offsets[start])
        s_on, mask, info = TF.stream_nodes(ev_on, ptr_on, hist_on, hp_on, c0, budget, TF.WINDOW_MS, None)
        src = idx[s_on] if idx.numel() else torch.zeros_like(s_on)
        src = torch.where(mask, src, torch.zeros_like(src))
        TF._STATE["pending"] = {"ptr": batch.ptr, "events": batch.events, "hist": hist, "hist_ptr": hp, "c0": c0,
                                "src": src, "mask": mask, "nodes": int(info["nodes_max"])}
        return TF._STATE["pending"]
    TF.prepare_packet = prepare_packet


def uninstall():
    from . import tpd_str as TF
    if _STATE["native_prepare"] is not None:
        TF.prepare_packet = _STATE["native_prepare"]
    if _STATE["native_stride"] is not None:
        XM.stride_sample = _STATE["native_stride"]
        EVM.EventPacketBatch.__init__ = _STATE["native_init"]
        for mod in _STATE.get("bound") or []:
            mod.stride_sample = _STATE["native_stride"]
    _STATE.update(on=False, tau=None, near=None, native_stride=None, native_init=None, native_prepare=None, memo=None,
                  max_rows=None)
    _REG.clear()


def stats():
    return dict(_STATE["stats"])
