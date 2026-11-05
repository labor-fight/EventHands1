r"""ACU stage 2: the causal per-event pipeline shared by every 2x2 cell (docs/EXPERIMENT_RECORDS.md#acu-s2-prereg-20261104).

Every quantity an event contributes is computed when it arrives, from itself, earlier events and the state of the last
readout; nothing reads a later event of the window:

  nodes    the trunk's nodes are PR's causal-stride window nodes (`semkine.xyz_mesh.stride_sample` -> semkine/acu_pr.py
           `causal_stride` with the batch's k from the history's event rate); replaces the count-dependent stride
  tokens   the trunk's seven-scalar tokens (`semkine.encoder.event_tokens`, read by model.py at forward time) over the
           history + window events of each sample: the inter-event time and the same / opposite polarity pixel times run
           across the readout boundary, "never fired" is H (a constant, not the packet's span); t_norm = (t - t_p) / dt
           with t_p the prior's time inside the window (0 except for B1 training samples), dt the window length; window
           rows are returned
  routes   `semkine.xag.sample_sources` -> PR's routes at the trunk's nodes (part, rest-template point, on-hand >= tau ->
           face, barycentrics), computed with the batch's history; the association is PFA (point-to-point) or CA
  B1       optional, as semkine/acu_b1.py: the prior time inside the window (training) and the readout's age term
Data: training packets carry the history and the prior's time (`install_dataset(b1=...)`, `collate_causal`); evaluation
packets get the history from the recording (`eval_batch`). Opt-in; `uninstall()` restores everything.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from . import acu_b1 as B1
from . import acu_hist as AH
from . import acu_pr as PR
from . import dataset as D
from . import domrand as DR
from . import encoder as ENC
from . import xag as XAG
from . import xyz_mesh as XM
from .events import EV_BATCH, EV_COLS, EV_P, EV_T, EV_X, EV_Y, collate_packets

_ORIGINAL_GETITEM = D.SemKineDataset.__getitem__
_STATE = {"b1": False, "batch": None, "k": None, "net": None, "ckpt": None, "tau": 0.5, "native": {}, "on": False}


# --------------------------------------------------------------------------- data
def _getitem(self, idx: int):
    """semkine/acu_hist.py's history packet, with B1's prior time when installed with b1=True."""
    if not self.train:
        return _ORIGINAL_GETITEM(self, idx)
    if (self.input_mode != "raw_packed" or self._fast_on or self.prev_lag_ms is not None or self.unroll_pair
            or self.clip_k or self.triplet or self.fixed_window is not None):
        raise RuntimeError("the causal pipeline is defined on the raw-packet training path only")
    rng = np.random.default_rng((self.seed * 1_000_003 + idx) & 0x7FFFFFFF)
    si, end, window, run_a = self._window(idx, rng)
    h = self._handle(si)
    start = end - window + 1
    h0 = max(int(run_a), start - AH.H_MS)
    ext = end - h0 + 1
    xs, ys, ps, us = self._raw_events(h, end, ext)
    source_rows = np.arange(int(h.offsets[h0]), int(h.offsets[end + 1]), dtype=np.int64) if self.xag_source_root else None
    target = self._to_target(h.pos51[end])
    prior_ms = B1.draw_prior_ms(self.seed, idx, int(start), int(end))[0] if _STATE["b1"] else int(start)
    prev = self._to_target(h.pos51[prior_ms])
    camera_K = h.camera_K.copy()
    dr = DR.sample_params(self.domrand, rng, self.height * self.width * 2 * ext)
    if self.domrand.enabled:
        if source_rows is None:
            xs, ys, ps, us = self._domrand_events(xs, ys, ps, us, ext, dr, camera_K, rng)
        else:
            xs, ys, ps, us, source_rows = self._domrand_events(xs, ys, ps, us, ext, dr, camera_K, rng,
                                                               source_rows=source_rows)
        stacked, camera_K = DR.transform_labels(np.stack([prev, target]), dr, camera_K, h.j0, self.render_scale)
        prev, target = stacked[0], stacked[1]
    noise = self._sample_prev_noise(rng)
    if noise is not None:
        prev = prev + noise
    flip = self.polarity_flip and int(rng.integers(0, 2)) == 1
    swap_mask = rng.integers(0, 2, size=(self.height, self.width)).astype(bool) if self.pixel_polarity_swap else None
    p_eff = self._polarity(xs, ys, ps, flip, swap_mask)
    shift = (start - h0) * 1000
    win = us >= shift
    packet = self._pack(
        si, h, xs[win], ys[win], p_eff[win], us[win] - shift, window,
        t_start_us=int(start) * 1000, t_end_us=(int(end) + 1) * 1000, target=target, prev=prev, camera_K=camera_K,
        is_start=bool(start == int(self.index[idx][2])), is_end=bool(end == int(self.index[idx][3]) - 1),
        meta={"window_ms": window, "end_idx": end, "seq": h.seq, "category": h.category, "domrand": dr},
        source_rows=None if source_rows is None else source_rows[win])
    hist = np.empty((int((~win).sum()), 4), np.float32)
    if len(hist):
        hist[:, 0], hist[:, 1] = xs[~win], ys[~win]
        hist[:, 2] = (us[~win].astype(np.float64) - shift) * 1e-6
        hist[:, 3] = p_eff[~win]
    pp0 = max(int(run_a), h0 - AH.H_MS)
    n_pp = int(h.offsets[h0] - h.offsets[pp0]) * float(dr.keep)
    packet.meta.update(acu_hist=hist, acu_n_hist=int(len(hist)), acu_hist_len_s=(start - h0) * 1e-3,
                       acu_k_hist=int(PR.stride_k(n_pp, (h0 - pp0) * 1e-3, (start - h0) * 1e-3)),
                       acu_prior_offset_s=float(prior_ms - start) * 1e-3)
    return packet


@dataclass
class CausalBatch(AH.HistBatch):
    prior_offset_s: Optional[torch.Tensor] = None

    def to(self, device) -> "CausalBatch":
        def mv(t):
            return t.to(device, non_blocking=True) if torch.is_tensor(t) else t
        return CausalBatch(**{k: mv(v) for k, v in self.__dict__.items()})


def collate_causal(items):
    h = AH.collate_hist(items)
    off = torch.tensor([float(it.meta.get("acu_prior_offset_s", 0.0)) for it in items], dtype=torch.float32)
    return CausalBatch(**h.__dict__, prior_offset_s=off)


def install_dataset(b1: bool):
    _STATE["b1"] = bool(b1)
    D.SemKineDataset.__getitem__ = _getitem


def install_collate(train_module):
    _STATE["native"]["collate"] = (train_module, train_module.collate_packets)
    train_module.collate_packets = collate_causal


# --------------------------------------------------------------------------- evaluation packets
def eval_batch(ctx, end, run_a, prev_t, device, window=50):
    """The evaluation packet of [end - window + 1, end] with the stream's history (prior at the window start)."""
    from . import eval_track as ET
    ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, int(end), window)
    b = ET.make_eval_packet(ev5, prev_t, ctx.betas, ctx.K, window, device)
    hist, n_hist, hist_len_s, k_hist = AH.eval_history(ctx.events, ctx.offsets, ctx.tsub, int(end), int(run_a), window)
    hev = torch.zeros(len(hist), EV_COLS, device=device)
    if len(hist):
        hev[:, 1:] = torch.from_numpy(hist).to(device)
    return CausalBatch(**b.__dict__, hist_events=hev, hist_ptr=torch.tensor([0, len(hist)], device=device),
                       n_hist=torch.tensor([n_hist], device=device), hist_len_s=torch.tensor([hist_len_s], device=device,
                                                                                            dtype=torch.float64),
                       k_hist=torch.tensor([k_hist], device=device),
                       prior_offset_s=torch.zeros(1, device=device))


def run_start(aux, end):
    runs = np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2)
    hit = np.nonzero((runs[:, 0] <= int(end)) & (int(end) < runs[:, 1]))[0]
    if not len(hit):
        raise ValueError(f"endpoint {end} outside every valid run")
    return int(runs[hit[0], 0])


# --------------------------------------------------------------------------- model hooks
def _causal_nodes(ptr, max_nodes):
    k = _STATE["k"]
    if k is None:
        raise RuntimeError("causal nodes outside forward_packet")
    if int(max_nodes) != PR.N_CAP:
        raise ValueError("the causal stride caps at PR.N_CAP")
    return PR.causal_stride(ptr, k)


def _all_events(batch):
    """History + window events of every sample in time order and the window rows' positions in that list."""
    ev, hev = batch.events, batch.hist_events
    B = batch.batch_size
    dev = ev.device
    nb_h = batch.hist_ptr[1:] - batch.hist_ptr[:-1]
    bid = torch.cat([hev[:, EV_BATCH], ev[:, EV_BATCH]]).long()
    rank = torch.cat([torch.arange(hev.shape[0], device=dev) - batch.hist_ptr[:-1][hev[:, EV_BATCH].long()],
                      torch.arange(ev.shape[0], device=dev) - batch.ptr[:-1][ev[:, EV_BATCH].long()]
                      + nb_h[ev[:, EV_BATCH].long()]])
    order = torch.argsort(bid * (int(rank.max().item()) + 1 if rank.numel() else 1) + rank)
    allev = torch.cat([hev, ev], 0)[order]
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.shape[0], device=dev)
    cnt = nb_h + (batch.ptr[1:] - batch.ptr[:-1])
    aptr = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), cnt.cumsum(0)])
    return allev, aptr, inv[hev.shape[0]:], B


def causal_tokens(events, ptr, delta_t_s, height, width, eps=ENC.EPS_DT):
    """The trunk's tokens of the window events of `_STATE['batch']` (see the module docstring)."""
    batch = _STATE["batch"]
    if batch is None or getattr(batch, "hist_events", None) is None:
        raise RuntimeError("causal tokens need the batch history")
    if events.data_ptr() != batch.events.data_ptr() or events.shape != batch.events.shape:
        raise RuntimeError("causal tokens are defined for the forward's own window events")
    n = events.shape[0]
    if n == 0:
        return events.new_zeros((0, ENC.TOKEN_DIM))
    allev, aptr, rows, B = _all_events(batch)
    dt = delta_t_s.clamp_min(eps)[events[:, EV_BATCH].long()]
    dt_ie_all = ENC.inter_event_dt(allev, aptr)
    same_all, opp_all = ENC.sae_times(allev, aptr, height, width, fallback=PR.H_S)
    same, opp = same_all[rows].clamp(max=PR.H_S), opp_all[rows].clamp(max=PR.H_S)
    dt_ie = dt_ie_all[rows]
    off = batch.prior_offset_s.to(events.device).float()[events[:, EV_BATCH].long()] if \
        getattr(batch, "prior_offset_s", None) is not None else torch.zeros_like(dt)
    return torch.stack([events[:, EV_X] / float(width), events[:, EV_Y] / float(height), 2.0 * events[:, EV_P] - 1.0,
                        (events[:, EV_T] - off) / dt, torch.log1p(dt_ie / eps), (same / dt).clamp(0.0, 1.0),
                        (opp / dt).clamp(0.0, 1.0)], dim=-1)


@torch.no_grad()
def pr_sources(batch, src, mask, *, require_depth=False):
    """PR's routes at the trunk's nodes (S45-SM's source layout)."""
    if require_depth:
        raise RuntimeError("PR routes faces only")
    net = _net(src.device)
    out = net(batch.events, batch.ptr, batch.delta_t_s, batch.hist_events, batch.hist_ptr, _STATE["k"], batch.k_hist)
    if out["src"].shape != src.shape or not torch.equal(out["mask"], mask) or not torch.equal(out["src"][mask], src[mask]):
        raise RuntimeError("PR nodes are not the trunk's nodes")
    routed = mask & (torch.sigmoid(out["onhand"].float()) >= _STATE["tau"])
    face, bary = net.route({"part": out["part"].float(), "coord": out["coord"].float()}, routed)
    face = torch.where(routed, face, torch.zeros_like(face))
    zero = torch.zeros_like(face)
    return {"xag_source_index": zero, "xag_source_ms": zero, "xag_part": net.face_part[face], "xag_face": face,
            "xag_bary": bary.to(torch.float32), "xag_valid": routed}


def _net(device):
    net = _STATE["net"].get(str(device)) if isinstance(_STATE["net"], dict) else None
    if net is None:
        base, _ = PR.load(_STATE["ckpt"], map_location="cpu")
        for p in base.parameters():
            p.requires_grad_(False)
        net = base.to(device).eval()
        _STATE["net"] = {**(_STATE["net"] or {}), str(device): net}
    return net


def install_model(model_cls, pr_checkpoint, assoc, onhand_tau=0.5, b1=False):
    """forward_packet records the batch (history, prior offset, causal stride) for the hooks; nodes, tokens and routes as
    the module docstring; assoc "pfa" (semkine/s45sm.py PFA) or "ca" (semkine/acu_ca.py)."""
    from . import s45sm as SM
    native_fp = model_cls.forward_packet

    def forward_packet(self, batch):
        if getattr(batch, "hist_events", None) is None:
            raise RuntimeError("the causal pipeline needs packets with history")
        _STATE["batch"] = batch
        _STATE["k"] = batch.k_win()
        if b1:
            off = batch.prior_offset_s.to(batch.delta_t_s.device).float()
            B1._STATE["age"] = ((batch.delta_t_s.float() - off) / B1.AGE_UNIT_S).clamp_min(0.0)
        try:
            return native_fp(self, batch)
        finally:
            _STATE["batch"], _STATE["k"] = None, None
            if b1:
                B1._STATE["age"] = None
    _STATE["native"].update(forward_packet=(model_cls, native_fp), stride=XM.stride_sample, tokens=ENC.event_tokens,
                            sources=XAG.sample_sources, route=XAG.route_association, rays=XM.associate_rays)
    model_cls.forward_packet = forward_packet
    XM.stride_sample = _causal_nodes
    ENC.event_tokens = causal_tokens
    _STATE.update(ckpt=str(pr_checkpoint), tau=float(onhand_tau), net=None, on=True)
    XAG.sample_sources = pr_sources
    XM.associate_rays = SM.empty_association
    if assoc == "pfa":
        XAG.route_association = SM.pfa_association
    elif assoc == "ca":
        from . import acu_ca as CA
        XAG.route_association = CA.ca_association
        CA.install_intrinsics(model_cls)
    else:
        raise ValueError("assoc pfa | ca")
    return {"nodes": "semkine.acu_pr.causal_stride", "tokens": "semkine.acu_causal.causal_tokens",
            "sources": "semkine.acu_causal.pr_sources", "assoc": assoc, "pr_checkpoint": str(pr_checkpoint),
            "onhand_tau": float(onhand_tau), "b1": bool(b1)}


def uninstall():
    D.SemKineDataset.__getitem__ = _ORIGINAL_GETITEM
    nat = _STATE["native"]
    if "collate" in nat:
        mod, fn = nat["collate"]
        mod.collate_packets = fn
    if "forward_packet" in nat:
        cls, fn = nat["forward_packet"]
        cls.forward_packet = fn
        XM.stride_sample, ENC.event_tokens = nat["stride"], nat["tokens"]
        XAG.sample_sources, XAG.route_association, XM.associate_rays = nat["sources"], nat["route"], nat["rays"]
    _STATE.update(b1=False, batch=None, k=None, net=None, ckpt=None, native={}, on=False)


__all__ = ["CausalBatch", "causal_tokens", "collate_causal", "eval_batch", "install_collate", "install_dataset",
           "install_model", "pr_sources", "run_start", "uninstall"]
