r"""S45-SM router context (docs/EXPERIMENT_RECORDS.md#s45sm-r6c-prereg-20261031). Opt-in; nothing changes unless installed.

The events-only router may look at the events of the last `ctx_ms` before the packet end (the packet plus up to
ctx_ms - 50 ms before it, clipped at the start of the valid run, as the evaluator clips its windows) instead of the
packet alone. S45 still sees the packet only, the routed nodes stay S45's stride-sampled nodes of the packet, and the
router's weights do not change (every router was trained on 30-300 ms packets, AUG.SPEED_AUG). No previous state, K
or betas.

  forward     `router_forward_ctx(net, ptr, ctx, src, mask)`: the event image and the 7-d tokens of the context
              packet. Node i of packet b is the packet's event src[b, i]; the packet's events are the context's last
              events (checked when the context is attached), so it is the context's event
              src[b, i] + ctx_ptr[b + 1] - ptr[b + 1]. The head reads the context feature map at that event's pixel
              and its context token. A context equal to the packet gives `semkine/s45sm_fast.router_forward`'s output
              bit for bit (tests/test_s45sm_ctx.py).
  evaluation  `install_eval(ctx_ms)` wraps semkine.eval_track's `load_sequence` (records the sequence's valid runs),
              `_window_events` (for a 50 ms window it also cuts the context window min(ctx_ms, end - run start + 1))
              and `make_eval_packet` (attaches that context to the packet built from those very events, as an
              attribute of the packet's event tensor; a batch rebuilt from other tensors carries none).
  routing     `install_routing()` replaces `semkine.s45sm.predicted_sources` and `semkine.s45sm_fast.predicted_sources`
              (and the installed `semkine.xag.sample_sources`, if it is one of them) by wrappers that route with the
              context whenever the batch's events carry one and fall through to the original otherwise, so training
              batches and packets without context route exactly as before.
"""
from __future__ import annotations

import numpy as np
import torch

from . import s45sm as SM
from . import s45sm_fast as FAST
from . import xag as XAG
from .encoder import EV_X, EV_Y

ATTR = "s45sm_ctx"
PACKET_MS = 50
CONTEXTS = (100, 150, 200, 300)
T_TOL_S = 2e-6
_NATIVE = {"sm": SM.predicted_sources, "fast": FAST.predicted_sources}
_STATE = {"ms": None, "runs": None, "pending": None, "et": None}


def context_of(batch):
    return getattr(batch.events, ATTR, None)


def context_nodes(ptr, cptr, src, mask):
    """Row of every live node's event in the context batch (the packet's events are the context's last events)."""
    B = int(ptr.numel() - 1)
    return torch.where(mask, src + (cptr[1:] - ptr[1:]).view(B, 1), torch.zeros_like(src))


def router_forward_ctx(net, ptr, ctx, src, mask):
    """`FAST.router_forward` with the context's event image and tokens; the nodes are the packet's (module docstring)."""
    ev, cptr, dt = ctx["events"], ctx["ptr"], ctx["delta_t_s"]
    B = int(ptr.numel() - 1)
    tok = FAST._NATIVE_TOKENS(ev, cptr, dt, SM.HEIGHT, SM.WIDTH)
    fmap = FAST.trunk(net, FAST.event_image(ev, cptr, dt).to(net.stem[0].weight.dtype))
    flat = context_nodes(ptr, cptr, src, mask).reshape(-1)
    node_ev = ev[flat]
    h4, w4 = fmap.shape[-2:]
    yy = (node_ev[:, EV_Y].long() * h4 // SM.HEIGHT).clamp(0, h4 - 1)
    xx = (node_ev[:, EV_X].long() * w4 // SM.WIDTH).clamp(0, w4 - 1)
    bb = torch.arange(B, device=ev.device).repeat_interleave(src.shape[1])
    feat = fmap[bb, :, yy, xx]
    z = net.head(torch.cat([feat.to(tok.dtype), tok[flat]], -1)).view(B, src.shape[1], -1)
    z = z * mask.unsqueeze(-1).to(z.dtype)
    out = {"z": z, "coord": net.coord(z), "onhand": net.onhand(z).squeeze(-1), "src": src, "mask": mask}
    out["patch" if hasattr(net, "patch") else "part"] = (net.patch if hasattr(net, "patch") else net.part)(z)
    return out


def _wrap(native):
    @torch.no_grad()
    def predicted_sources(batch, src, mask, *, require_depth=False):
        ctx = context_of(batch)
        if ctx is None or require_depth or batch.events.shape[0] == 0:
            return native(batch, src, mask, require_depth=require_depth)
        st = SM._STATE
        net = SM._net(src.device)
        out = router_forward_ctx(net, batch.ptr, ctx, src, mask)
        routed = mask & (torch.sigmoid(out["onhand"].float()) >= st["onhand_tau"])
        face, bary = FAST.route(net, st["router"], out, routed)
        zero = torch.zeros_like(face)
        return {"xag_source_index": zero, "xag_source_ms": zero, "xag_part": net.face_part[face], "xag_face": face,
                "xag_bary": bary.to(torch.float32), "xag_valid": routed}
    predicted_sources._s45sm_ctx = native
    return predicted_sources


def install_routing():
    if getattr(SM.predicted_sources, "_s45sm_ctx", None) is not None:
        return
    if SM.predicted_sources is not _NATIVE["sm"] or FAST.predicted_sources is not _NATIVE["fast"]:
        raise RuntimeError("another hook already replaced predicted_sources")
    SM.predicted_sources, FAST.predicted_sources = _wrap(_NATIVE["sm"]), _wrap(_NATIVE["fast"])
    if XAG.sample_sources is _NATIVE["sm"]:
        XAG.sample_sources = SM.predicted_sources
    elif XAG.sample_sources is _NATIVE["fast"]:
        XAG.sample_sources = FAST.predicted_sources


def uninstall_routing():
    if XAG.sample_sources is SM.predicted_sources:
        XAG.sample_sources = _NATIVE["sm"]
    elif XAG.sample_sources is FAST.predicted_sources:
        XAG.sample_sources = _NATIVE["fast"]
    SM.predicted_sources, FAST.predicted_sources = _NATIVE["sm"], _NATIVE["fast"]


def check_tail(ev5: np.ndarray, ctx5: np.ndarray, ctx_ms: int):
    """The packet's events are the context's last events: pixel and polarity equal, time shifted by ctx_ms - 50 ms."""
    n, m = len(ev5), len(ctx5)
    if m < n:
        raise RuntimeError("the context has fewer events than the packet")
    tail = ctx5[m - n:]
    if not np.array_equal(tail[:, [0, 1, 2, 4]], ev5[:, [0, 1, 2, 4]]):
        raise RuntimeError("the packet's events are not the context's last events")
    shift = (ctx_ms - PACKET_MS) * 1e-3
    if n and float(np.abs(tail[:, 3].astype(np.float64) - ev5[:, 3].astype(np.float64) - shift).max()) > T_TOL_S:
        raise RuntimeError("the packet's event times are not the context's shifted by ctx_ms - 50 ms")


def attach(batch, ev5: np.ndarray, ctx5: np.ndarray, ctx_ms: int):
    check_tail(ev5, ctx5, ctx_ms)
    dev = batch.events.device
    setattr(batch.events, ATTR, {
        "events": torch.from_numpy(np.ascontiguousarray(ctx5)).to(dev),
        "ptr": torch.tensor([0, len(ctx5)], dtype=torch.int64, device=dev),
        "delta_t_s": torch.tensor([ctx_ms * 1e-3], dtype=torch.float32, device=dev), "ms": int(ctx_ms)})
    return batch


def context_window(end: int, ctx_ms: int):
    """The context window of a 50 ms evaluation window ending at `end`: min(ctx_ms, end - run start + 1)."""
    starts = [a for a, b in _STATE["runs"] or () if a <= int(end) < b]
    if len(starts) != 1:
        raise RuntimeError("endpoint outside every valid run (or no sequence loaded)")
    return min(int(ctx_ms), int(end) - starts[0] + 1)


def install_eval(ctx_ms: int):
    from . import eval_track as ET
    ctx_ms = int(ctx_ms)
    if ctx_ms not in CONTEXTS:
        raise ValueError(f"ctx_ms in {CONTEXTS}")
    if _STATE["ms"] is not None:
        raise RuntimeError("router context already installed")
    load, window_of, packet_of = ET.load_sequence, ET._window_events, ET.make_eval_packet

    def load_sequence(root, legacy_dir, seq):
        result = load(root, legacy_dir, seq)
        _STATE["runs"] = [tuple(map(int, r)) for r in np.asarray(result[2]["valid_runs_ms"]).reshape(-1, 2)]
        _STATE["pending"] = None
        return result

    def window_events(events, offsets, tsub, end, window):
        ev5 = window_of(events, offsets, tsub, end, window)
        _STATE["pending"] = None
        if int(window) == PACKET_MS:
            w = context_window(int(end), ctx_ms)
            if w > PACKET_MS:
                _STATE["pending"] = (ev5, window_of(events, offsets, tsub, end, w), w)
        return ev5

    def make_eval_packet(ev5, prev, betas, camera_K, window_ms, device):
        batch = packet_of(ev5, prev, betas, camera_K, window_ms, device)
        pending, _STATE["pending"] = _STATE["pending"], None
        if pending is not None and pending[0] is ev5 and int(window_ms) == PACKET_MS and len(ev5):
            attach(batch, ev5, pending[1], pending[2])
        return batch

    ET.load_sequence, ET._window_events, ET.make_eval_packet = load_sequence, window_events, make_eval_packet
    _STATE.update(ms=ctx_ms, et=(load, window_of, packet_of), runs=None, pending=None)
    return ctx_ms


def uninstall_eval():
    from . import eval_track as ET
    if _STATE["et"] is not None:
        ET.load_sequence, ET._window_events, ET.make_eval_packet = _STATE["et"]
    _STATE.update(ms=None, runs=None, pending=None, et=None)


def installed():
    return None if _STATE["ms"] is None else {"ctx_ms": _STATE["ms"],
                                               "routing": getattr(SM.predicted_sources, "_s45sm_ctx", None) is not None}


__all__ = ["ATTR", "CONTEXTS", "attach", "check_tail", "context_nodes", "context_of", "context_window", "install_eval",
           "install_routing", "installed", "router_forward_ctx", "uninstall_eval", "uninstall_routing"]
