r"""TPD stage STR (docs/EXPERIMENT_RECORDS.md#tpd-str-prereg-20261106): the streaming input form of the closed-loop trunk.

The causal form (semkine/tpd_causal.py) fixes a packet's stride from the previous packet's count and cuts the packet after
2048 nodes, so a packet whose rate rose loses its newest events (zgz_global: 36.5 % of the 50 ms packets, 6.9 % of the
events), and a pixel that has not fired in the packet takes the packet duration. Here every event's inputs are fixed at
its arrival from itself and the stream before it; the packet (the update interval) enters only through t_q:

  rate    n_i = the events of the stream that arrived before i with time >= t_i - W                 (W = 50 ms)
  stride  k_i = max(1, ceil(n_i / BUDGET))                                                          (BUDGET = 2048)
  keep    c_i mod k_i == 0, c_i = the stream's running event counter                                 (no cap)
  tokens  x / W_img, y / H_img, polarity, (t_i - t_q) / Δ, log1p(gap to the stream's previous event / eps),
          same / opposite-polarity pixel age min(t_i - T_last, Δ) / Δ over the stream (not reset at the updates)
with t_q the last state update and Δ the update interval (50 ms on the evaluation grid). Times are integer microseconds;
the rate uses the running maximum of the stream's times, so it stays prefix-stable even if a timestamp stepped back (the
recordings' never do). Every age is clamped at Δ = W, so a packet needs only the W before it ("history"). The router
(r_s2d) stays per update interval (exempt, #tpd-prereg-20261105) and is evaluated at the kept events, so an event's PFA
record is fixed at the update, not at its arrival.

Pieces: `stream_nodes` / `stream_tokens` (batched, ragged history); the datasets (rollout packet 0 carries the W before it
under the sample's geometric and polarity maps with its own keep draw; plain packets, evaluation-style, the raw W before
them); `collate_str`; `str_model_class` (FinalModel with these inputs; rollout packet k >= 1 takes packet k-1 as its
history); `install` / `engine_hooks` for PFO's frozen engine; `Stream` (event-by-event ingestion: the prefix / partition
checks and the deployment description).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import domrand as DR
from . import encoder as ENC
from . import tpd_final as TF
from . import tpd_rol as R
from . import xyz_mesh as XM
from .events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y

BUDGET = 2048
WINDOW_MS = 50
STR_SALT = 61106
#: the training-only guard on the kept events of one packet (the 9 training subjects' 50 ms packets reach 5,385 at most
#: without augmentation); hits are logged, evaluation has no cap
TRAIN_CAP = 6144
_OFF = 1 << 21                      # us added before the segmented keys (times >= -W, < 2 s)
_SEG = 1 << 23                      # us per sample segment of the keys
_STATE = {"on": False, "native": XM.stride_sample, "tokens_prev": None, "ctx": None, "pending": None}
if _STATE["native"].__module__ != "semkine.xyz_mesh":
    raise ImportError("semkine.tpd_str must be imported before stride_sample is patched")
#: tpd_final's dataset upgrades as imported (tools/tpd/ns_str.py then points tpd_final's names at this module's)
_FINAL = {"upgrade_rollout": TF.upgrade_rollout, "upgrade_plain": TF.upgrade_plain}


# --------------------------------------------------------------------------- the stream of one batch
def _us(t):
    return torch.round(t.double() * 1e6).long()


def _combined(events, ptr, hist, hist_ptr):
    """The stream of every sample, history then packet: (events (N_h + N, 5), ptr (B + 1), rows of the packet events).
    Gathers only (no index assignment), so the deterministic flags accept it on CUDA."""
    B = int(ptr.numel() - 1)
    dev = events.device
    if hist is None:
        hist, hist_ptr = events.new_zeros((0, events.shape[1])), torch.zeros(B + 1, dtype=torch.long, device=dev)
    hist_ptr = hist_ptr.to(dev)
    nh, n = hist_ptr[1:] - hist_ptr[:-1], ptr[1:] - ptr[:-1]
    cptr = torch.zeros(B + 1, dtype=torch.long, device=dev)
    cptr[1:] = torch.cumsum(nh + n, 0)
    bc = torch.repeat_interleave(torch.arange(B, device=dev), nh + n)
    o = torch.arange(int(cptr[-1]), device=dev) - cptr[:-1][bc]
    from_hist = o < nh[bc]
    row = torch.where(from_hist, hist_ptr[:-1][bc] + o, hist.shape[0] + ptr[:-1][bc] + o - nh[bc])
    out = torch.cat([hist.to(events.dtype), events], 0)[row]
    out[:, EV_BATCH] = bc.to(events.dtype)
    bp = events[:, EV_BATCH].long()
    rows_p = cptr[:-1][bp] + nh[bp] + torch.arange(events.shape[0], device=dev) - ptr[:-1][bp]
    return out, cptr, rows_p


def stream_nodes(events, ptr, hist, hist_ptr, c0, budget=BUDGET, window_ms=WINDOW_MS, cap=None):
    """`(src (B, N) long, mask (B, N) bool, info)` like `stride_sample`: the packet events kept by the streaming rule,
    in arrival order; `c0` (B,) is the stream counter of each packet's first event. `cap` (training only) keeps the first
    `cap` kept events of a packet; info counts the packets it cut."""
    B = int(ptr.numel() - 1)
    dev = events.device
    if events.shape[0] == 0:
        return (torch.zeros(B, 1, dtype=torch.long, device=dev), torch.zeros(B, 1, dtype=torch.bool, device=dev),
                {"cap_hits": 0, "nodes_max": 0})
    comb, cptr, rows_p = _combined(events, ptr, hist, hist_ptr)
    b = comb[:, EV_BATCH].long()
    key = b * _SEG + _us(comb[:, EV_T]) + _OFF
    if bool(((key - b * _SEG) < 0).any() or ((key - b * _SEG) >= _SEG).any()):
        raise ValueError("stream times outside the key range")
    run = torch.cummax(key, 0).values                                   # segmented: every segment starts above the last
    lo = torch.searchsorted(run, run - int(window_ms) * 1000, right=False)
    n_before = torch.arange(run.numel(), device=dev) - lo               # arrived before i, time >= t_i - W
    n_i = n_before[rows_p]
    k = torch.clamp((n_i + budget - 1) // budget, min=1)
    bp = events[:, EV_BATCH].long()
    c = c0.to(dev).long()[bp] + torch.arange(events.shape[0], device=dev) - ptr[:-1][bp]
    keep = (c % k) == 0
    kc = torch.zeros(events.shape[0] + 1, dtype=torch.long, device=dev)
    kc[1:] = torch.cumsum(keep.long(), 0)
    first = kc[ptr[:-1]]
    counts = kc[ptr[1:]] - first
    hits = 0
    if cap is not None:
        hits = int((counts > int(cap)).sum())
        counts = counts.clamp(max=int(cap))
    N = max(int(counts.max()), 1)
    kept = torch.nonzero(keep).squeeze(1)                               # grouped by packet, arrival order
    pos = torch.arange(N, device=dev)
    mask = pos.unsqueeze(0) < counts.unsqueeze(1)
    flat = (first.unsqueeze(1) + pos.unsqueeze(0)).clamp(max=max(kept.numel() - 1, 0))
    src = kept[flat] if kept.numel() else torch.zeros(B, N, dtype=torch.long, device=dev)
    src = torch.where(mask, src, torch.zeros_like(src))
    return src, mask, {"cap_hits": hits, "nodes_max": int(counts.max())}


def stream_tokens(events, ptr, delta_t_s, height, width, hist, hist_ptr, eps=ENC.EPS_DT):
    """`(N, 7)` tokens of the packet events, B2's seven scalars computed over the stream (history then packet)."""
    n = events.shape[0]
    if n == 0:
        return events.new_zeros((0, ENC.TOKEN_DIM))
    comb, cptr, rows_p = _combined(events, ptr, hist, hist_ptr)
    comb[:, EV_T] = comb[:, EV_T] + WINDOW_MS * 1e-3                   # history times >= 0 for the pixel-time keys
    dt = ENC._packet_dt(delta_t_s, events)
    dt_ie = ENC.inter_event_dt(comb, cptr)[rows_p]
    sae_s, sae_o = ENC.sae_times(comb, cptr, height, width, fallback=-1.0)
    sae_s, sae_o = sae_s[rows_p], sae_o[rows_p]
    sae_s = torch.where(sae_s > 0, sae_s, dt)
    sae_o = torch.where(sae_o > 0, sae_o, dt)
    return torch.stack([
        events[:, EV_X] / float(width),
        events[:, EV_Y] / float(height),
        2.0 * events[:, EV_P] - 1.0,
        events[:, EV_T] / dt,
        torch.log1p(dt_ie / eps),
        (sae_s / dt).clamp(0.0, 1.0),
        (sae_o / dt).clamp(0.0, 1.0),
    ], dim=-1)


# --------------------------------------------------------------------------- batches
@dataclass
class StrPacketBatch(TF.CausalPacketBatch):
    hist_events: Optional[torch.Tensor] = None     # (N_h, 5): the W before each packet, times relative to its start
    hist_ptr: Optional[torch.Tensor] = None        # (B + 1,)
    c0: Optional[torch.Tensor] = None              # (B,) stream counter of the packet's first event

    def to(self, device) -> "StrPacketBatch":
        def mv(t):
            return t.to(device, non_blocking=True) if torch.is_tensor(t) else t
        return StrPacketBatch(**{k: mv(v) for k, v in self.__dict__.items()})


def str_batch(items):
    b = TF.causal_batch(items)
    c0 = torch.tensor([int(it.meta["str_c0"]) for it in items], dtype=torch.int64)
    hist = [it.meta.get("str_hist") for it in items]
    if all(h is None for h in hist):
        return StrPacketBatch(**b.__dict__, c0=c0)
    if any(h is None for h in hist):
        raise ValueError("history on some packets of a batch only")
    counts = np.array([len(h) for h in hist], np.int64)
    hp = np.zeros(len(items) + 1, np.int64)
    np.cumsum(counts, out=hp[1:])
    ev = np.zeros((int(hp[-1]), 5), np.float32)
    if hp[-1]:
        ev[:, EV_BATCH] = np.repeat(np.arange(len(items), dtype=np.float32), counts)
        ev[:, 1:] = np.concatenate([h for h in hist if len(h)], 0)
    return StrPacketBatch(**b.__dict__, hist_events=torch.from_numpy(ev), hist_ptr=torch.from_numpy(hp), c0=c0)


def collate_str(items):
    if isinstance(items[0], R.RolloutItem):
        k = len(items[0].packets)
        if any(len(it.packets) != k for it in items):
            raise ValueError("rollout items of different lengths in one batch")
        return R.RolloutBatch([str_batch([it.packets[i] for it in items]) for i in range(k)])
    return str_batch(items)


def _hist_array(xs, ys, ps, us, span_ms):
    """(N, 4) [x, y, t, p] with t = (us - span) * 1e-6 (relative to the packet start, < 0)."""
    out = np.empty((len(xs), 4), np.float32)
    if len(xs):
        out[:, 0], out[:, 1] = xs, ys
        out[:, 2] = (np.asarray(us, np.int64) - int(span_ms) * 1000).astype(np.float32) * 1e-6
        out[:, 3] = ps
    return out


# --------------------------------------------------------------------------- datasets
class StrRolloutDataset(TF.FinalRolloutDataset):
    """FinalRolloutDataset's samples, draw for draw, with packet 0's history and every packet's stream counter. The
    history is the W before packet 0 (clipped at the recording start) under the sample's geometric map and polarity
    maps, with its own keep draw (stream STR_SALT); the counter starts at the history's first event."""

    def __getitem__(self, idx: int):
        rng = np.random.default_rng([self.seed, int(idx), self.salt])
        si, ends, run_a, run_b = self.ends(idx)
        P = self.packet_ms
        h = self._handle(si)
        starts = [e - P + 1 for e in ends]
        camera_K = h.camera_K.copy()
        states = np.stack([self._to_target(h.pos51[starts[0]])] + [self._to_target(h.pos51[e]) for e in ends])
        dr = DR.sample_params(self.domrand, rng, self.height * self.width * 2 * P)
        raw = []
        for s, e in zip(starts, ends):
            xs, ys, ps, us = self._raw_events(h, e, P)
            rows = np.arange(int(h.offsets[s]), int(h.offsets[e + 1]), dtype=np.int64) if self.xag_source_root else None
            if self.domrand.enabled:
                if rows is None:
                    xs, ys, ps, us = self._domrand_events(xs, ys, ps, us, P, dr, camera_K, rng)
                else:
                    xs, ys, ps, us, rows = self._domrand_events(xs, ys, ps, us, P, dr, camera_K, rng, source_rows=rows)
            raw.append((xs, ys, ps, us, rows))
        if self.domrand.enabled:
            states, camera_K_new = DR.transform_labels(states, dr, camera_K, h.j0, self.render_scale)
        else:
            camera_K_new = camera_K
        prev0 = states[0].copy()
        noise = self._sample_prev_noise(rng)
        if noise is not None:
            prev0 = prev0 + noise
        flip = self.polarity_flip and int(rng.integers(0, 2)) == 1
        swap_mask = None
        if self.pixel_polarity_swap and self.train:
            swap_mask = rng.integers(0, 2, size=(self.height, self.width)).astype(bool)
        crng = np.random.default_rng([self.seed, int(idx), TF.FINAL_SALT])
        n0 = TF.augmented_count(self, h, starts[0] - P, starts[0] - 1, dr, camera_K, crng)
        hrng = np.random.default_rng([self.seed, int(idx), STR_SALT])
        span = min(P, int(starts[0]))
        if span > 0:
            hx, hy, hp, hu = self._raw_events(h, starts[0] - 1, span)
            if self.domrand.enabled:
                hx, hy, hp, hu = self._domrand_events(hx, hy, hp, hu, span, dr, camera_K, hrng)
            hist = _hist_array(hx, hy, self._polarity(hx, hy, hp, flip, swap_mask), hu, span)
        else:
            hist = np.zeros((0, 4), np.float32)
        c0 = len(hist)
        packets = []
        for i, (xs, ys, ps, us, rows) in enumerate(raw):
            p_eff = self._polarity(xs, ys, ps, flip, swap_mask)
            meta = {"window_ms": P, "end_idx": int(ends[i]), "seq": h.seq, "category": h.category, "domrand": dr,
                    "rollout_index": int(idx), "rollout_step": i, "n_prev": n0 if i == 0 else len(raw[i - 1][0]),
                    "str_c0": c0}
            if i == 0:
                meta["str_hist"] = hist
            packets.append(self._pack(
                si, h, xs, ys, p_eff, us, P, t_start_us=int(starts[i]) * 1000, t_end_us=(int(ends[i]) + 1) * 1000,
                target=states[i + 1], prev=prev0 if i == 0 else states[i], camera_K=camera_K_new,
                is_start=bool(starts[i] == run_a), is_end=bool(ends[i] == run_b - 1), meta=meta, source_rows=rows))
            c0 += len(xs)
        return R.RolloutItem(packets)


class StrPlainDataset(TF.FinalPlainDataset):
    """Evaluation-style packets (no augmentation) with the raw W before them and the recording's event counter."""

    def __getitem__(self, idx: int):
        if self.domrand.enabled or self.polarity_flip or (self.pixel_polarity_swap and self.train):
            raise RuntimeError("STR plain packets are evaluation-style (no augmentation)")
        packet = TF.FinalPlainDataset.__getitem__(self, idx)
        h = self._handle(int(self.index[idx][0]))
        end, window = int(packet.meta["end_idx"]), int(packet.meta["window_ms"])
        start = end - window + 1
        span = min(WINDOW_MS, start)
        if span > 0:
            hx, hy, hp, hu = self._raw_events(h, start - 1, span)
            packet.meta["str_hist"] = _hist_array(hx, hy, hp, hu, span)
        else:
            packet.meta["str_hist"] = np.zeros((0, 4), np.float32)
        packet.meta["str_c0"] = int(h.offsets[start])
        return packet


def upgrade_rollout(ds, k, packet_ms=R.PACKET_MS):
    _FINAL["upgrade_rollout"](ds, k, packet_ms)
    ds.__class__ = StrRolloutDataset
    return ds


def upgrade_plain(ds):
    _FINAL["upgrade_plain"](ds)
    ds.__class__ = StrPlainDataset
    return ds


def link_history(packets):
    """Rollout packet k >= 1: its history is packet k-1 (times shifted by that packet's duration); checks the counters."""
    for k in range(1, len(packets)):
        prev, pk = packets[k - 1], packets[k]
        ev = prev.events.clone()
        if ev.shape[0]:
            ev[:, EV_T] = ev[:, EV_T] - prev.delta_t_s[ev[:, EV_BATCH].long()]
        pk.hist_events, pk.hist_ptr = ev, prev.ptr
        if not torch.equal(pk.c0, prev.c0 + (prev.ptr[1:] - prev.ptr[:-1])):
            raise RuntimeError("stream counters of consecutive rollout packets disagree")


# --------------------------------------------------------------------------- the model
_CLASS = {}


def str_model_class(base=None):
    """`base` (default: semkine/tpd_final.py's FinalModel) with the streaming nodes and tokens; built once."""
    if "cls" in _CLASS:
        return _CLASS["cls"]
    base = base if base is not None else TF.final_model_class()

    class STRModel(base):
        def __init__(self, cfg=None, num_classes=None):
            super().__init__(cfg, num_classes)
            s = self.cfg["STR"]
            self.str_budget, self.str_window = int(s["BUDGET"]), int(s["WINDOW_MS"])
            self.str_cap = s.get("TRAIN_CAP")
            self._str_info = []

        def _nodes(self, batch):
            src, mask, info = stream_nodes(batch.events, batch.ptr, batch.hist_events, batch.hist_ptr, batch.c0,
                                           self.str_budget, self.str_window, self.str_cap if self.training else None)
            if self.training:
                self._str_info.append(info)
            return src, mask

        def _tokens(self, batch):
            return stream_tokens(batch.events, batch.ptr, batch.delta_t_s, self.render_h, self.render_w,
                                 batch.hist_events, batch.hist_ptr)

        def training_step(self, batch, batch_nb):
            if isinstance(batch, R.RolloutBatch):
                link_history(batch.packets)
            self._str_info = []
            loss = super().training_step(batch, batch_nb)
            if self._str_info:
                B = batch.batch_size
                self.log("train_str_cap_hits", float(sum(i["cap_hits"] for i in self._str_info)), sync_dist=True,
                         batch_size=B, reduce_fx="sum")
                self.log("train_str_nodes_max", float(max(i["nodes_max"] for i in self._str_info)), sync_dist=True,
                         batch_size=B, reduce_fx="max")
            return loss

        def _start_check(self, pk):
            """First rollout step: the streaming inputs of packet 0 and the INIT_FROM weights."""
            self._rol_checked = True
            src, mask, info = stream_nodes(pk.events, pk.ptr, pk.hist_events, pk.hist_ptr, pk.c0, self.str_budget,
                                           self.str_window, None)
            counts = mask.sum(1).float()
            rec = {"global_step": int(self.global_step), "rank": int(self.global_rank), "causal": True, "stream": True,
                   "autocast": bool(torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()),
                   "nodes_mean": float(counts.mean()), "nodes_max": int(info["nodes_max"]),
                   "hist_mean": float((pk.hist_ptr[1:] - pk.hist_ptr[:-1]).float().mean()),
                   "events_mean": float((pk.ptr[1:] - pk.ptr[:-1]).float().mean())}
            init = self.cfg["MODEL"].get("INIT_FROM")
            if init:
                sd = torch.load(init, map_location="cpu")["state_dict"]
                own = self.state_dict()
                rec["init_equal"] = bool(set(sd) == set(own) and all(torch.equal(own[k].detach().cpu(), sd[k]) for k in sd))
            out = Path(self.cfg["TRAIN"]["OUTPUT_DIR"]) / f"final_start_check_rank{self.global_rank}.json"
            out.write_text(json.dumps(rec, indent=2) + "\n")
            if not rec.get("init_equal", True):
                raise RuntimeError("STR start check failed: " + json.dumps(rec))

    _CLASS["cls"] = STRModel
    return STRModel


# --------------------------------------------------------------------------- PFO's engine (one packet per batch)
def stride_sample_str(ptr, max_nodes):
    p = _STATE["pending"]
    if p is None or p["ptr"].shape != ptr.shape or not torch.equal(p["ptr"], ptr.to(p["ptr"].device)):
        raise RuntimeError("STR nodes: no streaming state for this batch (engine_hooks)")
    return p["src"], p["mask"]


def event_tokens_str(events, ptr, delta_t_s, height, width, eps=ENC.EPS_DT):
    p = _STATE["pending"]
    if p is None or p["events"] is not events:
        raise RuntimeError("STR tokens: no streaming state for this batch (engine_hooks)")
    return stream_tokens(events, ptr, delta_t_s, height, width, p["hist"], p["hist_ptr"], eps)


def install():
    if _STATE["on"]:
        raise RuntimeError("STR already installed")
    if XM.stride_sample is not _STATE["native"]:
        raise RuntimeError("stride_sample is patched by someone else")
    XM.stride_sample = stride_sample_str
    _STATE["tokens_prev"] = ENC.event_tokens
    ENC.event_tokens = event_tokens_str
    _STATE["on"] = True


def retoken():
    if _STATE["on"]:
        ENC.event_tokens = event_tokens_str


def uninstall():
    XM.stride_sample = _STATE["native"]
    if _STATE["tokens_prev"] is not None:
        ENC.event_tokens = _STATE["tokens_prev"]
    _STATE.update(on=False, ctx=None, pending=None, tokens_prev=None)


def history_events(events, offsets, tsub, start, span_ms=WINDOW_MS):
    """(N, 5) [batch 0, x, y, t, p] of the span_ms before ms `start` (clipped at the recording start), t < 0 relative
    to `start`; the integer-microsecond arithmetic of the training history."""
    span = min(int(span_ms), int(start))
    if span <= 0:
        return np.zeros((0, 5), np.float32)
    a, a0, a1 = start - span, int(offsets[start - span]), int(offsets[start])
    ev = np.asarray(events[a0:a1])
    us = np.repeat(np.arange(span, dtype=np.int64), np.diff(offsets[a:start + 1]).astype(np.int64)) * 1000
    if tsub is not None:
        us = us + np.asarray(tsub[a0:a1]).astype(np.int64)
    out = np.zeros((len(ev), 5), np.float32)
    out[:, 1:] = _hist_array(ev[:, 0], ev[:, 1], np.clip(ev[:, 2], 0, 1), us, span)
    return out


def prepare_packet(batch, events, offsets, tsub, end, window=WINDOW_MS, budget=BUDGET):
    """Streaming state of an engine packet ending at ms `end`: history, counter, the kept events."""
    start = int(end) - int(window) + 1
    dev = batch.events.device
    hist = torch.from_numpy(history_events(events, offsets, tsub, start)).to(dev)
    hp = torch.tensor([0, hist.shape[0]], dtype=torch.long, device=dev)
    c0 = torch.tensor([int(offsets[start])], dtype=torch.long, device=dev)
    src, mask, info = stream_nodes(batch.events, batch.ptr, hist, hp, c0, budget, WINDOW_MS, None)
    _STATE["pending"] = {"ptr": batch.ptr, "events": batch.events, "hist": hist, "hist_ptr": hp, "c0": c0,
                         "src": src, "mask": mask, "nodes": int(info["nodes_max"])}
    return _STATE["pending"]


def engine_hooks():
    """PFO's frozen engine: remember the sequence it opens; for every packet it builds, the streaming state (chained
    after whatever `attach_xag_sources` the engine's `nets.load` installed)."""
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
            if ctx is None or ctx.seq != seq or ctx.offsets is not offsets or int(window) != WINDOW_MS:
                raise RuntimeError("STR: the engine's sequence context is not the packet's")
            if int(batch.ptr.numel()) != 2:
                raise RuntimeError("STR: the engine builds one packet per batch")
            prepare_packet(batch, ctx.events, ctx.offsets, ctx.tsub, end)
            return inner(batch, cfg, seq, offsets, end, window, device)
        ET.attach_xag_sources = attach
        return net
    N.sequence, N.load = sequence, load


# --------------------------------------------------------------------------- event-by-event ingestion
class Stream:
    """One sensor stream ingested in arbitrary chunks. Each arriving event's keep decision and token are computed from
    the stream before it (the last W of history kept) and the last update time; `update()` returns the update interval's
    kept events and their tokens and starts the next interval. The batched functions are prefix-stable, so the result
    does not depend on the chunking: `tests/test_tpd_str.py` and `tools/tpd/asynccheck_str.py` check it against
    `stream_nodes` / `stream_tokens` on the whole packet."""

    def __init__(self, height, width, budget=BUDGET, window_ms=WINDOW_MS, counter=0, history=None):
        self.h, self.w, self.budget, self.window = int(height), int(width), int(budget), int(window_ms)
        self.counter = int(counter)
        self.past = history if history is not None else torch.zeros((0, 5))      # times relative to the interval start
        self.cur = torch.zeros((0, 5))
        self.kept, self.tokens = [], []

    def ingest(self, chunk):
        """chunk (M, 5) [*, x, y, t, p] with t relative to the interval start, in arrival order."""
        if chunk.shape[0] == 0:
            return
        chunk = chunk.clone()
        chunk[:, EV_BATCH] = 0
        upto = torch.cat([self.cur, chunk], 0)
        ptr = torch.tensor([0, upto.shape[0]])
        hp = torch.tensor([0, self.past.shape[0]])
        c0 = torch.tensor([self.counter])
        src, mask, _ = stream_nodes(upto, ptr, self.past, hp, c0, self.budget, self.window, None)
        sel = src[0][mask[0]]
        sel = sel[sel >= self.cur.shape[0]]
        tok = stream_tokens(upto, ptr, torch.tensor([self.window * 1e-3]), self.h, self.w, self.past, hp)
        self.kept.append(sel)
        self.tokens.append(tok[sel])
        self.cur = upto

    def update(self):
        """(kept indices into the interval's events, (n, 7) tokens); the interval's events become the history."""
        kept = torch.cat(self.kept) if self.kept else torch.zeros(0, dtype=torch.long)
        tok = torch.cat(self.tokens, 0) if self.tokens else torch.zeros((0, ENC.TOKEN_DIM))
        span = self.window * 1e-3
        past = torch.cat([self.past, self.cur], 0)
        past[:, EV_T] = past[:, EV_T] - span
        self.past = past[past[:, EV_T] >= -span]
        self.counter += self.cur.shape[0]
        self.cur, self.kept, self.tokens = torch.zeros((0, 5)), [], []
        return kept, tok


__all__ = ["BUDGET", "STR_SALT", "Stream", "StrPacketBatch", "StrPlainDataset", "StrRolloutDataset", "TRAIN_CAP",
           "WINDOW_MS", "collate_str", "engine_hooks", "event_tokens_str", "history_events", "install", "link_history",
           "prepare_packet", "retoken", "str_batch", "str_model_class", "stream_nodes", "stream_tokens",
           "stride_sample_str", "uninstall", "upgrade_plain", "upgrade_rollout"]
