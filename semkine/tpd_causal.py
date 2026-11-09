r"""TPD causal node selection for the closed-loop network (docs/EXPERIMENT_RECORDS.md#tpd-causal-prereg-20261105; goal
constraint 1: the deployed closed-loop input must be asynchronous -- prefix-stable and partition-invariant).

B2's trunk takes at most CAP = 2048 events of a packet as nodes by a uniform stride over the packet (`semkine.xyz_mesh
.stride_sample`): an event's selection depends on how many events arrive after it (91 % of zgz_global's 50 ms packets and
27 % of zgz_local's exceed the cap). Here the stride comes from the past:

    nodes = the packet's events j (arrival order) with j % k == 0 and j // k < CAP,  k = max(1, ceil(n_prev / CAP))
    n_prev = the number of events in the PACKET_MS before the packet (the previous packet of the evaluation grid)

(semkine/acu_pr.py `causal_stride`, ACU's causal stride with the previous interval's rate). The tokens: B2's seven
scalars are prefix-stable on the fixed 50 ms grid except the same / opposite-polarity pixel times of a pixel that has not
fired yet in the packet, which `semkine.encoder.sae_times` fills with the packet's actual span (last - first event time,
a future quantity); `event_tokens_causal` fills them with the packet duration Δt instead (known in advance; the token
value of a "never fired" pixel becomes exactly 1). The rest is unchanged (position, polarity, t / Δt, inter-event time;
PFA geometry given the node's route). The router is packet-based and exempt (#tpd-prereg-20261105); it keeps its own
tokens and is evaluated at the trunk's nodes (semkine/s45sm_fast.py path).

Opt-in: `install()` replaces `semkine.xyz_mesh.stride_sample` and `semkine.encoder.event_tokens` (model/model.py imports
both at call time; the routers bound their own names at import) by the causal versions, the stride reading the batch's k
from `set_k`; `engine_hooks()` makes PFO's frozen engine compute k for every packet it builds from the recording it is
reading.
"""
from __future__ import annotations

import math

import torch

from . import encoder as ENC
from . import xyz_mesh as XM

CAP = 2048
PACKET_MS = 50
_STATE = {"native": XM.stride_sample, "tokens": ENC.event_tokens, "k": None, "on": False, "ctx": None, "tokens_prev": None}
if _STATE["native"].__module__ != "semkine.xyz_mesh":
    raise ImportError("semkine.tpd_causal must be imported before stride_sample is patched")


def event_tokens_causal(events, ptr, delta_t_s, height, width, eps=ENC.EPS_DT):
    """`semkine.encoder.event_tokens` with the "never fired" pixel time = the packet duration (not its actual span)."""
    n = events.shape[0]
    if n == 0:
        return events.new_zeros((0, ENC.TOKEN_DIM))
    dt = ENC._packet_dt(delta_t_s, events)
    dt_ie = ENC.inter_event_dt(events, ptr)
    sae_s, sae_o = ENC.sae_times(events, ptr, height, width, fallback=0.0)
    sae_s = torch.where(sae_s > 0, sae_s, dt)
    sae_o = torch.where(sae_o > 0, sae_o, dt)
    return torch.stack([
        events[:, ENC.EV_X] / float(width),
        events[:, ENC.EV_Y] / float(height),
        2.0 * events[:, ENC.EV_P] - 1.0,
        events[:, ENC.EV_T] / dt,
        torch.log1p(dt_ie / eps),
        (sae_s / dt).clamp(0.0, 1.0),
        (sae_o / dt).clamp(0.0, 1.0),
    ], dim=-1)


def k_from_prev(n_prev, cap=CAP):
    """k = max(1, ceil(n_prev / cap)) per packet (same interval length as the packet)."""
    n = torch.as_tensor(n_prev, dtype=torch.int64)
    return torch.clamp((n + cap - 1) // cap, min=1)


def causal_stride(ptr, k, cap=CAP):
    """`(src (B, N) long, mask (B, N) bool)` like `stride_sample`: per packet the events j with j % k == 0, j // k < cap."""
    from .acu_pr import causal_stride as acu
    src, mask = acu(ptr, k, cap=int(cap))
    return src, mask


def set_k(k):
    _STATE["k"] = None if k is None else torch.as_tensor(k, dtype=torch.int64)


def stride_sample_causal(ptr, max_nodes):
    k = _STATE["k"]
    if k is None or k.numel() != int(ptr.numel() - 1):
        raise RuntimeError("causal stride: no k for this batch (set_k)")
    return causal_stride(ptr, k.to(ptr.device), cap=int(max_nodes))


def install():
    if _STATE["on"]:
        raise RuntimeError("causal stride already installed")
    if XM.stride_sample is not _STATE["native"]:
        raise RuntimeError("stride_sample is patched by someone else")
    XM.stride_sample = stride_sample_causal
    _STATE["tokens_prev"] = ENC.event_tokens
    ENC.event_tokens = event_tokens_causal
    _STATE["on"] = True


def retoken():
    """Re-assert the causal tokens after a later hook replaced `semkine.encoder.event_tokens` (semkine/s45sm_fast.py
    `install` puts its memo there; the router keeps its own bound names either way)."""
    if _STATE["on"]:
        ENC.event_tokens = event_tokens_causal


def uninstall():
    XM.stride_sample = _STATE["native"]
    if _STATE["tokens_prev"] is not None:
        ENC.event_tokens = _STATE["tokens_prev"]
    _STATE.update(on=False, k=None, ctx=None, tokens_prev=None)


def n_prev_events(offsets, end, packet_ms=PACKET_MS):
    """Events in the packet_ms before the packet ending at `end` (clipped at the recording start)."""
    start = int(end) - packet_ms + 1
    a = max(start - packet_ms, 0)
    return int(offsets[start]) - int(offsets[a])


def engine_hooks():
    """PFO's frozen engine: remember the sequence it opens and set k for every packet it builds (chained after whatever
    `attach_xag_sources` the engine's `nets.load` installed)."""
    from tools.pfo import nets as N
    from . import eval_track as ET
    native_seq, native_load = N.sequence, N.load

    def sequence(*args, **kwargs):
        ctx = native_seq(*args, **kwargs)
        _STATE["ctx"] = ctx
        return ctx

    def load(*args, **kwargs):
        net = native_load(*args, **kwargs)
        retoken()
        inner = ET.attach_xag_sources

        def attach(batch, cfg, seq, offsets, end, window, device):
            ctx = _STATE["ctx"]
            if ctx is None or ctx.seq != seq or ctx.offsets is not offsets or int(window) != PACKET_MS:
                raise RuntimeError("causal stride: the engine's sequence context is not the packet's")
            set_k([k_from_prev(n_prev_events(offsets, end))])
            return inner(batch, cfg, seq, offsets, end, window, device)
        ET.attach_xag_sources = attach
        return net
    N.sequence, N.load = sequence, load


__all__ = ["CAP", "PACKET_MS", "causal_stride", "engine_hooks", "event_tokens_causal", "install", "k_from_prev",
           "n_prev_events", "retoken", "set_k", "stride_sample_causal", "uninstall"]
