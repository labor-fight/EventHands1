"""Explicit-state, event-only S37 integration for the causal-cache research arm.

This is a new input/runtime contract, not a reproduction of S37 checkpoints.
Existing weights can initialize it; short-interval tracking and cold start still
require training and independent validation. No annotation enters this API.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any

import torch

from .routed_readout import pool_joint_evidence
from .streaming import CausalEventEncoder


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be an integer, not a rounded time')
    result = int(value)
    if not torch.iinfo(torch.int64).min <= result <= torch.iinfo(torch.int64).max:
        raise ValueError(f'{name} is outside int64 range')
    return result


@dataclass(frozen=True)
class StreamTrackingState:
    pose: torch.Tensor
    graph: Any
    stream_id: int
    last_query_us: int
    pending_events: int
    last_event_us: int | None
    camera_K: torch.Tensor
    betas: torch.Tensor
    model_version: tuple


@dataclass(frozen=True)
class MeshOutput:
    pose: torch.Tensor
    vertices: torch.Tensor
    joints: torch.Tensor
    query_us: int
    state: StreamTrackingState


class SparseS37Tracker:
    """Reuse S37's learned operators; retain all streaming state in the caller.

    ``frame_period_us`` is explicit because a checkpoint trained at 50 ms is not
    validated at a shorter interval. Delta is never silently time-scaled. The
    first implementation uses a fixed neutral pose at 0.45 m and zero MANO beta
    (the MANO model's shape origin), not sequence GT or an external initializer.
    ``camera_K`` is calibration at the legacy full sensor resolution; the S37
    projection applies its existing render scale. Event pixels remain 240x180.

    A pose/shape/K update recomputes all readout routing. The only persistent
    feature cache is state-independent. Use torch.no_grad() for deployment;
    absence of it deliberately preserves gradients for bounded unroll Debug.
    """

    def __init__(self, model, *, frame_period_us: int, horizon_us: int = 50_000,
                 max_events_per_append: int = 8192):
        if not getattr(model, 'routed', False) or model.encoder_name != 'event_gnn':
            raise ValueError('SparseS37Tracker requires the S37 routed EventGNN')
        forbidden = ('prev_render', 'fk_graph', 'mesh_graph', 'mesh_query',
                     'event_guided_mesh', 'root_covmap')
        if any(bool(getattr(model, name, False)) for name in forbidden):
            raise ValueError('unsupported optional model path for streaming S37')
        if getattr(model, 'route_prev_override', None) is not None:
            raise ValueError('oracle routing is forbidden in streaming inference')
        if not model.predict_delta or not model.active_head:
            raise ValueError('S37 delta/active heads are required')
        frame_period_us = _integer(frame_period_us, 'frame_period_us')
        horizon_us = _integer(horizon_us, 'horizon_us')
        if not 0 < frame_period_us <= horizon_us:
            raise ValueError('require 0 < frame_period_us <= horizon_us')
        self.model = model
        self.frame_period_us = int(frame_period_us)
        self.encoder = CausalEventEncoder(model.event_encoder, horizon_us=horizon_us,
                                         max_events_per_append=max_events_per_append,
                                         linear_backend="fixed_fp32")

    def _model_version(self):
        return tuple((name, id(p), p._version, p.device, p.dtype)
                     for name, p in self.model.named_parameters())

    def _validate_state(self, state):
        if not isinstance(state, StreamTrackingState):
            raise TypeError('state must be StreamTrackingState')
        if state.model_version != self._model_version():
            raise ValueError('model parameters changed; initialize a new tracking state')

    def initialize(self, camera_K: torch.Tensor, *, stream_id: int,
                   start_us: int) -> StreamTrackingState:
        stream_id = _integer(stream_id, 'stream_id')
        start_us = _integer(start_us, 'start_us')
        dev = next(self.model.parameters()).device
        K = torch.as_tensor(camera_K, device=dev, dtype=torch.float32).reshape(1, 3, 3)
        if not bool(torch.isfinite(K).all()) or not bool((K[:, (0, 1), (0, 1)] > 0).all()):
            raise ValueError('finite positive focal lengths required')
        if not torch.equal(K[:, 2], K.new_tensor([[0., 0., 1.]])):
            raise ValueError('camera_K must use the calibrated pinhole convention')
        pose = torch.zeros(1, 51, device=dev, dtype=torch.float32)
        pose[:, 2] = 0.45
        return StreamTrackingState(pose, None, int(stream_id), int(start_us), 0,
                                   None, K.clone(), torch.zeros(1, 10, device=dev),
                                   self._model_version())

    def append(self, xyp: torch.Tensor, timestamps: torch.Tensor,
               state: StreamTrackingState) -> StreamTrackingState:
        self._validate_state(state)
        if timestamps.numel() and int(timestamps[0]) < state.last_query_us:
            raise ValueError('an event cannot arrive behind the last query watermark')
        graph = self.encoder.append(xyp, timestamps, state=state.graph,
                                    stream_id=state.stream_id)
        if not timestamps.numel():
            return state if state.graph is not None else replace(state, graph=graph)
        return replace(state, graph=graph,
                       pending_events=state.pending_events + int(timestamps.numel()),
                       last_event_us=int(timestamps[-1]))

    def query(self, now_us: int, state: StreamTrackingState) -> MeshOutput:
        self._validate_state(state)
        now_us = _integer(now_us, 'now_us')
        if now_us != state.last_query_us + self.frame_period_us:
            raise ValueError('query cadence changed; reset or use the declared schedule')
        if state.last_event_us is not None and state.last_event_us >= now_us:
            raise ValueError('query consumes only events with timestamp < query time')
        if self.model.route_prev_override is not None:
            raise ValueError('oracle routing is forbidden in streaming inference')
        if not torch.equal(state.betas, torch.zeros_like(state.betas)):
            raise ValueError('this prototype uses a fixed mean shape, not sequence shape')
        prev = state.pose
        if prev.dtype != torch.float32 or not bool(torch.isfinite(prev).all()):
            raise ValueError('recurrent pose must be finite float32')
        pose = prev
        if state.graph is not None:
            if not state.pending_events:
                # Static output still validates state, but does not recompute
                # evidence that will not be integrated. Both controls share it.
                self.encoder._validate_state(state.graph, state.stream_id)
            else:
                feat, h, px, py, mask = self.encoder.readout(state.graph, now_us)
                if not bool(mask.any()):
                    raise RuntimeError('unserved events expired before their mesh query')
                a, _, _ = self.model._route_nodes(px, py, mask, prev,
                                                 state.betas, state.camera_K)
                evidence, _ = pool_joint_evidence(h, a, mask)
                if self.model.ablate_evidence:
                    evidence = torch.zeros_like(evidence)
                delta = self.model._decode_active(feat, prev, evidence)
                if self.model.prevpos_embed:
                    delta = delta + self.model.prev_mlp(prev.to(delta.dtype))
                # Match the intended S37 addition without rounding the FP32 prior.
                pose = prev + delta.to(torch.float32)
        if not bool(torch.isfinite(pose).all()):
            raise FloatingPointError('non-finite predicted pose')
        # Final decoding is explicit; no latency measurement may stop at pose51.
        with torch.autocast(device_type=pose.device.type, enabled=False):
            vertices, joints = self.model._fk(pose, state.betas)
        if not bool(torch.isfinite(vertices).all() and torch.isfinite(joints).all()):
            raise FloatingPointError('non-finite decoded mesh')
        new_state = replace(state, pose=pose, last_query_us=now_us, pending_events=0)
        return MeshOutput(pose, vertices, joints, now_us, new_state)
