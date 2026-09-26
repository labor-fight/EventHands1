"""Exact causal-prefix execution of EventGNN's existing trainable operators.

This runner owns no parameters and is deliberately not an nn.Module. Its caller
owns the encoder and explicit stream state. Arrival-final tokens differ from S37's
packet tokens: old S37 checkpoints are initializations, not equivalent predictors.

Each append computes new nodes once per layer, with the last ``window`` features
at every layer as predecessors. Their older influences remain encoded when a
node leaves that working cache. Readout uses the last ``max_nodes`` events within
the fixed horizon. Recomputing only this readout window is NOT the reference;
``full_reference`` independently evaluates the complete causal prefix.

Graph arithmetic is O(M*W + M*L*K*C*C) for M new events, feature caches are
O(L*W*C + N*C). Sparse SAE uses O(P+M) storage and sorting/searching of visited
pixel/polarity keys, with P <= 2*image_pixels. Contract checks synchronize on GPU;
neither their cost nor the sparse-map sort is claimed to meet a latency target.
No gradient is detached: use torch.no_grad/inference_mode for deployment and
explicitly reset after an optimizer update. Training graphs can retain history
beyond the bounded tensor caches; truncated backpropagation is not implicit.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Optional, Tuple, Union

import torch

from .event_gnn import EventGNN


StreamID = Union[int, str]
Readout = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class CausalEventState:
    """Caller-owned state; tensors must not be mutated by the caller.

    ``sae_keys`` and ``sae_timestamps`` are a sorted sparse dictionary, with key
    ``((y * width + x) * 2 + polarity)``. Equal timestamps retain arrival order.
    ``recent_layers[l]`` stores h^l; layer 0 is the embedded token.
    """

    recent_xyp: torch.Tensor
    recent_timestamps: torch.Tensor
    recent_layers: Tuple[torch.Tensor, ...]
    readout_xyp: torch.Tensor
    readout_timestamps: torch.Tensor
    readout_h: torch.Tensor
    sae_keys: torch.Tensor
    sae_timestamps: torch.Tensor
    last_timestamp: Optional[int]
    n_events: int
    stream_id: StreamID
    parameter_version: tuple
    runner_signature: tuple
    device: torch.device
    dtype: torch.dtype
    compute_dtype: torch.dtype


class CausalEventEncoder:
    """Reuse an EventGNN with explicit ``append -> readout`` stream execution.

    Input xyp is (M,3), integral sensor pixels and polarity 0/1 stored in the
    encoder parameter dtype. Times are (M,) int64 microseconds on the same device.
    Query support is [now-horizon, now); querying before an accepted event is an
    error. Starting a different stream or changing weights requires state=None.
    """

    def __init__(self, encoder: EventGNN, horizon_us: int = 50000,
                 max_events_per_append: int = 8192, linear_backend: str = "torch"):
        if not isinstance(encoder, EventGNN):
            raise TypeError("encoder must be an EventGNN")
        if encoder.in_dim != 7 or encoder.node_attrs != "token7" or not encoder.readout:
            raise ValueError("streaming requires token7, no extra channels, and a readout")
        if min(encoder.window, encoder.k, encoder.max_nodes, encoder.width, encoder.height) < 1:
            raise ValueError("graph, readout and image dimensions must be positive")
        if float(encoder.t_scale) != 1.0:
            raise ValueError("the frozen streaming distance uses t_scale=1")
        for name, value in (("horizon_us", horizon_us),
                            ("max_events_per_append", max_events_per_append)):
            if (isinstance(value, bool) or not isinstance(value, Integral)
                    or not 0 < value <= torch.iinfo(torch.int64).max):
                raise ValueError(f"{name} must be a positive integer")
        self.encoder = encoder
        self.horizon_us = int(horizon_us)
        self.max_events_per_append = int(max_events_per_append)
        if linear_backend not in ("torch", "fixed_fp32"):
            raise ValueError("linear_backend must be torch or fixed_fp32")
        self.linear_backend = linear_backend

    def _linear(self, layer, x):
        # CPU keeps nn.Linear; CUDA prefix execution can select a fixed arithmetic
        # schedule. This changes neither parameters nor the legacy model modules.
        if self.linear_backend == "fixed_fp32" and x.is_cuda:
            from .streaming_linear import fixed_linear
            return fixed_linear(x, layer)
        return layer(x)

    def _version(self) -> tuple:
        return tuple((name, id(p), p._version, p.device, p.dtype)
                     for name, p in self.encoder.named_parameters())

    def _signature(self) -> tuple:
        e = self.encoder
        return (id(self), self.horizon_us, e.window, e.k, e.max_nodes, e.width,
                e.height, e.hidden, len(e.layers), e.in_dim, e.t_scale,
                e.node_attrs, e.readout, e.feat_dim, self.linear_backend)

    def _validate_state(self, state: CausalEventState, stream_id: StreamID) -> None:
        if not isinstance(state, CausalEventState):
            raise TypeError("state must be CausalEventState or None")
        if state.stream_id != stream_id:
            raise ValueError("stream identity changed; reset with state=None")
        if state.parameter_version != self._version():
            raise ValueError("encoder parameters changed; reset after a weight/device/dtype update")
        if state.runner_signature != self._signature():
            raise ValueError("runner or graph configuration changed; reset required")

    def _validate_input(self, xyp: torch.Tensor, timestamps: torch.Tensor,
                        state: Optional[CausalEventState], stream_id: StreamID,
                        enforce_budget: bool) -> None:
        if not isinstance(xyp, torch.Tensor) or not isinstance(timestamps, torch.Tensor):
            raise TypeError("xyp and timestamps must be tensors")
        if xyp.ndim != 2 or xyp.shape[1] != 3:
            raise ValueError("xyp must have shape (events, 3)")
        if timestamps.ndim != 1 or timestamps.shape[0] != xyp.shape[0]:
            raise ValueError("timestamps must have one entry per event")
        weight = self.encoder.embed.weight
        if not xyp.is_floating_point() or xyp.dtype != weight.dtype:
            raise TypeError("xyp dtype must match encoder parameters")
        if timestamps.dtype != torch.int64:
            raise TypeError("timestamps must be int64 microseconds")
        if xyp.device != weight.device or timestamps.device != xyp.device:
            raise ValueError("input and encoder devices must match")
        if not isinstance(stream_id, (int, str)) or isinstance(stream_id, bool):
            raise TypeError("stream_id must be an integer or string")
        if state is not None:
            self._validate_state(state, stream_id)
            if state.device != xyp.device or state.dtype != xyp.dtype:
                raise ValueError("input device/dtype changed within a stream")
        n = len(xyp)
        if enforce_budget and n > self.max_events_per_append:
            raise ValueError("append exceeds max_events_per_append; no events were accepted")
        if not n:
            return
        e = self.encoder
        if not bool(torch.isfinite(xyp).all()):
            raise ValueError("event coordinates/polarities must be finite")
        xy, p = xyp[:, :2], xyp[:, 2]
        if not bool(((xy == xy.floor()).all() & (xy >= 0).all()
                     & (xyp[:, 0] < e.width).all() & (xyp[:, 1] < e.height).all())):
            raise ValueError("event pixels must be integral and within the calibrated image")
        if not bool(((p == 0) | (p == 1)).all()):
            raise ValueError("polarity must be 0 or 1")
        if not bool((timestamps[1:] >= timestamps[:-1]).all()):
            raise ValueError("event timestamps must be nondecreasing; sorting is not implicit")
        first, last = int(timestamps[0]), int(timestamps[-1])
        if state is not None and state.last_timestamp is not None:
            if first < state.last_timestamp:
                raise ValueError("new event precedes an already accepted event")
            oldest = int(state.sae_timestamps.min()) if state.sae_timestamps.numel() else first
            if state.recent_timestamps.numel():
                oldest = min(oldest, int(state.recent_timestamps[0]))
        else:
            oldest = first
        if last - min(first, oldest) > torch.iinfo(torch.int64).max:
            raise ValueError("timestamp differences exceed int64 range")

    def _empty(self, xyp: torch.Tensor, timestamps: torch.Tensor,
               stream_id: StreamID) -> CausalEventState:
        xyp, timestamps = xyp.clone(), timestamps.clone()
        h = xyp.new_empty((0, self.encoder.hidden))
        return CausalEventState(
            xyp, timestamps, tuple(h for _ in range(len(self.encoder.layers) + 1)),
            xyp, timestamps, h, timestamps, timestamps, None, 0, stream_id,
            self._version(), self._signature(), xyp.device, xyp.dtype, h.dtype)

    def _tokens(self, xyp: torch.Tensor, ts: torch.Tensor,
                old_keys: torch.Tensor, old_times: torch.Tensor,
                last_timestamp: Optional[int]):
        """Vectorized sparse SAE with ordinal causality, including zero-age hits."""
        n = len(ts)
        keys = ((xyp[:, 1].long() * self.encoder.width + xyp[:, 0].long()) * 2
                + xyp[:, 2].long())
        ordinal = torch.arange(1, n + 1, device=ts.device, dtype=torch.int64)
        all_keys = torch.cat((old_keys, keys))
        all_times = torch.cat((old_times, ts))
        packed = torch.cat((old_keys * (n + 1), keys * (n + 1) + ordinal))
        order = packed.argsort()
        sorted_packed = packed[order]
        sorted_keys, sorted_times = all_keys[order], all_times[order]

        def age(query_keys):
            pos = torch.searchsorted(sorted_packed, query_keys * (n + 1) + ordinal) - 1
            safe = pos.clamp_min(0)
            hit = (pos >= 0) & (sorted_keys[safe] == query_keys)
            # For a missing key choose ts itself before subtraction, avoiding an
            # irrelevant overflow against an unrelated timestamp.
            prev = torch.where(hit, sorted_times[safe], ts)
            delta = (ts - prev).clamp_max(self.horizon_us)
            return torch.where(hit, delta, torch.full_like(delta, self.horizon_us))

        same, opposite = age(keys), age(keys ^ 1)
        predecessor = ts[:1] if last_timestamp is None else ts.new_tensor([last_timestamp])
        gap = ts - torch.cat((predecessor, ts[:-1]))
        time_dtype = torch.float64 if xyp.dtype == torch.float64 else torch.float32
        gap_f = gap.to(time_dtype)
        token = torch.stack((xyp[:, 0] / self.encoder.width,
                             xyp[:, 1] / self.encoder.height,
                             2 * xyp[:, 2] - 1,
                             (gap_f / self.horizon_us).clamp_max(1),
                             torch.log1p(gap_f),
                             same.to(time_dtype) / self.horizon_us,
                             opposite.to(time_dtype) / self.horizon_us), dim=-1).to(xyp.dtype)
        last = torch.ones(len(sorted_keys), dtype=torch.bool, device=ts.device)
        last[:-1] = sorted_keys[:-1] != sorted_keys[1:]
        return token, sorted_keys[last], sorted_times[last]

    @staticmethod
    def _cat(old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        return torch.cat((old, new), dim=0) if len(old) else new

    def _append_edges(self, old_xy: torch.Tensor, old_t: torch.Tensor,
                      xyp: torch.Tensor, ts: torch.Tensor):
        e = self.encoder
        xy, times = self._cat(old_xy, xyp), self._cat(old_t, ts)
        dst = torch.arange(len(ts), device=ts.device) + len(old_t)
        candidate = dst[:, None] - torch.arange(1, e.window + 1, device=ts.device)
        safe = candidate.clamp_min(0)
        age = ts[:, None] - times[safe]
        time_dtype = torch.float64 if xyp.dtype == torch.float64 else torch.float32
        dp = torch.stack(((xy[safe, 0] - xyp[:, None, 0]) / e.width,
                          (xy[safe, 1] - xyp[:, None, 1]) / e.height,
                          -age.to(time_dtype) / self.horizon_us), dim=-1)
        valid = (candidate >= 0) & (age <= self.horizon_us)
        distances = dp.square().sum(-1).masked_fill(~valid, float("inf"))
        near = distances.topk(min(e.k, e.window), dim=-1, largest=False)
        indices = safe.gather(1, near.indices)
        mask = torch.isfinite(near.values)
        delta = dp.gather(1, near.indices[..., None].expand(-1, -1, 3))
        delta = torch.where(mask[..., None], delta, torch.zeros_like(delta))
        return indices, delta, mask

    def append(self, xyp: torch.Tensor, timestamps: torch.Tensor,
               state: Optional[CausalEventState] = None, stream_id: StreamID = 0
               ) -> CausalEventState:
        self._validate_input(xyp, timestamps, state, stream_id, enforce_budget=True)
        state = self._empty(xyp[:0], timestamps[:0], stream_id) if state is None else state
        if not len(timestamps):
            return state
        token, sae_keys, sae_times = self._tokens(
            xyp, timestamps, state.sae_keys, state.sae_timestamps, state.last_timestamp)
        h = torch.relu(self._linear(self.encoder.embed, token))
        if state.n_events and h.dtype != state.compute_dtype:
            raise ValueError("autocast/compute dtype changed; reset required")
        idx, dp, mask = self._append_edges(state.recent_xyp, state.recent_timestamps,
                                           xyp, timestamps)
        new_layers = [h]
        for layer_index, layer in enumerate(self.encoder.layers):
            source = self._cat(state.recent_layers[layer_index], h)
            message = torch.relu(self._linear(layer.lin, torch.cat(
                (source[idx] - h[:, None], dp.to(h.dtype)), dim=-1)))
            message = torch.where(mask[..., None], message, torch.zeros_like(message))
            h = h + message.sum(1) / mask.sum(1, keepdim=True).clamp_min(1).to(h.dtype)
            new_layers.append(h)
        w, n = self.encoder.window, self.encoder.max_nodes
        return CausalEventState(
            self._cat(state.recent_xyp, xyp)[-w:].clone(),
            self._cat(state.recent_timestamps, timestamps)[-w:].clone(),
            tuple(self._cat(old, new)[-w:].clone() for old, new in zip(state.recent_layers, new_layers)),
            self._cat(state.readout_xyp, xyp)[-n:].clone(),
            self._cat(state.readout_timestamps, timestamps)[-n:].clone(),
            self._cat(state.readout_h, h)[-n:].clone(), sae_keys, sae_times,
            int(timestamps[-1]), state.n_events + len(timestamps), stream_id,
            self._version(), self._signature(), xyp.device, xyp.dtype, h.dtype)

    @staticmethod
    def _query_time(now_us, last_timestamp: Optional[int]) -> int:
        if isinstance(now_us, bool) or not isinstance(now_us, Integral):
            raise TypeError("now_us must be an integer microsecond timestamp")
        now = int(now_us)
        if not torch.iinfo(torch.int64).min <= now <= torch.iinfo(torch.int64).max:
            raise ValueError("query timestamp is outside int64 range")
        if last_timestamp is not None and now < last_timestamp:
            raise ValueError("query precedes an already accepted event")
        return now

    def _read(self, xyp: torch.Tensor, timestamps: torch.Tensor, h: torch.Tensor,
              now: int) -> Readout:
        # Compare endpoints directly, avoiding overflow of now - timestamp.
        lower = max(now - self.horizon_us, torch.iinfo(torch.int64).min)
        live = (timestamps >= lower) & (timestamps < now)
        xyp, h = xyp[live], h[live]
        mask = torch.ones((1, len(h)), dtype=torch.bool, device=h.device)
        if len(h):
            feat = self.encoder.proj(torch.cat((h.mean(0), h.max(0).values))[None])
        else:
            feat = h.new_zeros((1, self.encoder.feat_dim))
        return feat, h[None], xyp[None, :, 0], xyp[None, :, 1], mask

    def readout(self, state: CausalEventState, now_us: int) -> Readout:
        self._validate_state(state, state.stream_id)
        now = self._query_time(now_us, state.last_timestamp)
        return self._read(state.readout_xyp, state.readout_timestamps, state.readout_h, now)

    def full_reference(self, xyp: torch.Tensor, timestamps: torch.Tensor,
                       now_us: int, stream_id: StreamID = 0, *, return_layers=False):
        """Independent full-prefix graph construction and synchronous layer calls.

        No call to append or _append_edges. All prefix nodes participate in the
        graph; only the final readout is restricted to the last max_nodes events.
        """
        self._validate_input(xyp, timestamps, None, stream_id, enforce_budget=False)
        now = self._query_time(now_us, int(timestamps[-1]) if len(timestamps) else None)
        e, count = self.encoder, len(timestamps)
        if not count:
            empty = xyp.new_empty((0, e.hidden))
            result = self._read(xyp, timestamps, empty, now)
            return (result, tuple(empty for _ in range(len(e.layers) + 1))) if return_layers else result
        token, _, _ = self._tokens(xyp, timestamps, timestamps[:0], timestamps[:0], None)
        nodes = torch.arange(count, device=xyp.device)
        candidates = nodes[:, None] - torch.arange(1, e.window + 1, device=xyp.device)
        bounded = candidates.clamp_min(0)
        dt = timestamps[:, None] - timestamps[bounded]
        time_dtype = torch.float64 if xyp.dtype == torch.float64 else torch.float32
        offsets = torch.stack(((xyp[bounded, 0] - xyp[:, None, 0]) / e.width,
                               (xyp[bounded, 1] - xyp[:, None, 1]) / e.height,
                               -dt.to(time_dtype) / self.horizon_us), dim=-1)
        allowed = (candidates >= 0) & (dt <= self.horizon_us)
        squared = offsets.square().sum(-1).masked_fill(~allowed, float("inf"))
        chosen = squared.topk(min(e.k, e.window), largest=False, dim=-1)
        idx = bounded.gather(1, chosen.indices)[None]
        mask = torch.isfinite(chosen.values)
        dp = offsets.gather(1, chosen.indices[..., None].expand(-1, -1, 3))
        dp = torch.where(mask[..., None], dp, torch.zeros_like(dp))[None]
        h = torch.relu(self._linear(e.embed, token))[None]
        layers = [h[0]] if return_layers else None
        for layer in e.layers:
            if self.linear_backend == "fixed_fp32" and h.is_cuda:
                # Independently materialize the full-prefix gather, as EdgeConv
                # does. Only the separately verified affine primitive is shared
                # with append; there are no caches or append-edge calls here.
                k = idx.shape[-1]
                neighbors = h.gather(1, idx.reshape(1, count * k, 1).expand(
                    1, count * k, e.hidden)).reshape(1, count, k, e.hidden)
                messages = torch.relu(self._linear(layer.lin, torch.cat(
                    (neighbors - h.unsqueeze(2), dp.to(h.dtype)), dim=-1)))
                messages = messages * mask[None, ..., None].to(h.dtype)
                h = h + messages.sum(2) / mask[None].sum(2, keepdim=True).clamp_min(1).to(h.dtype)
            else:
                h = h + layer(h, idx, dp.to(h.dtype), mask[None].to(h.dtype))
            if return_layers:
                layers.append(h[0])
        n = e.max_nodes
        result = self._read(xyp[-n:], timestamps[-n:], h[0, -n:], now)
        return (result, tuple(layers)) if return_layers else result
