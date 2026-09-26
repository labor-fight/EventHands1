"""Event-guided graphs on the previous state's complete MANO vertex set.

The vertex IDs retain their MANO order. Events rank geometrically nearby vertices;
they do not create nodes or discard unobserved vertices. Geometry participates in
messages only when a sender carries current or retained event evidence. The final
readout is a fixed, column-normalised LBS mean. Optional explicit local memory
updates only observed vertices and reads their new states for the pose delta.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Optional, Tuple

import torch
from torch import nn

from .event_gnn import EdgeConv
from .mesh_memory import EventDrivenVertexMemory


OBS_DIM = 6


@dataclass(frozen=True)
class MeshVertexState:
    """Caller-owned local memory in permanent MANO vertex order.

    ``hidden`` is floating point ``(B,V,C)`` and must contain finite values;
    ``seen`` is boolean ``(B,V)``. The caller resets this state at sequence
    boundaries, and may detach it at a truncated-backpropagation boundary.
    Neither tensor describes a frozen vertex position in world coordinates.
    """

    hidden: torch.Tensor
    seen: torch.Tensor

    def detach(self) -> "MeshVertexState":
        return MeshVertexState(self.hidden.detach(), self.seen.detach())


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _geometry_neighbors(value: int, k: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= k:
        raise ValueError("geometry_neighbors must be an integer between zero and k")
    return int(value)


def _graph_options(k: int, candidates: int, event_weight: float,
                   geometry_scale: float, chunk_size: int) -> Tuple[int, int, float, float, int]:
    k = _positive_int("k", k)
    candidates = _positive_int("candidates", candidates)
    chunk_size = _positive_int("chunk_size", chunk_size)
    if candidates < k:
        raise ValueError("candidates must be at least k")
    event_weight = float(event_weight)
    geometry_scale = float(geometry_scale)
    if not math.isfinite(event_weight) or event_weight < 0:
        raise ValueError("event_weight must be finite and non-negative")
    if not math.isfinite(geometry_scale) or geometry_scale <= 0:
        raise ValueError("geometry_scale must be finite and positive")
    return k, candidates, event_weight, geometry_scale, chunk_size


def _check_inputs(vertices: torch.Tensor, observations: torch.Tensor) -> None:
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape (B, V, 3)")
    if observations.ndim != 3 or observations.shape[-1] != OBS_DIM:
        raise ValueError(f"observations must have shape (B, V, {OBS_DIM})")
    if vertices.shape[:2] != observations.shape[:2]:
        raise ValueError("vertices and observations must have the same batch and vertex counts")
    if vertices.shape[1] < 1:
        raise ValueError("the graph must contain at least one vertex")
    if vertices.device != observations.device:
        raise ValueError("vertices and observations must be on the same device")
    if not vertices.is_floating_point() or not observations.is_floating_point():
        raise ValueError("vertices and observations must be floating-point tensors")


@torch.no_grad()
def build_event_guided_graph(vertices: torch.Tensor, observations: torch.Tensor,
                             k: int = 8, candidates: int = 32,
                             event_weight: float = 1.0, geometry_scale: float = 0.02,
                             chunk_size: int = 64, geometry_neighbors: int = 0
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return original vertex IDs, relative 3D offsets and valid-edge masks.

    First take the nearest ``candidates`` vertices in 3D, excluding self. Rank
    those candidates by ``distance_squared / geometry_scale**2`` plus
    ``event_weight * mean((obs_i - obs_j)**2)`` and retain ``k`` neighbours.
    Stable sorting resolves geometric ties by vertex ID and score ties by the
    preceding geometric order. Query chunks bound pairwise working storage to
    ``B * chunk_size * V`` rather than ``B * V * V``.

    With ``geometry_neighbors > 0``, reserve that many nearest geometric
    neighbours. Fill remaining slots with currently observed candidates ranked
    by the score, then with remaining geometric neighbours if needed. For a
    query lacking observations, score observed senders by geometry alone: its
    missing measurement must not masquerade as a zero-valued measurement.
    ``event_weight=0`` always returns the nearest ``k`` geometric neighbours.
    The default zero reservation retains the original graph implementation.

    Shapes are ``idx: (B,V,K)``, ``dp: (B,V,K,3)`` and
    ``edge_mask: (B,V,K)``. ``K = min(k, V-1)`` except that a one-vertex graph
    has one masked self-padding entry. ``dp`` is divided by ``geometry_scale``;
    it and the mask are float32, while indices are int64. Graph construction is
    detached from autograd; event observations still enter the learned encoder.
    """
    k, candidates, event_weight, geometry_scale, chunk_size = _graph_options(
        k, candidates, event_weight, geometry_scale, chunk_size)
    geometry_neighbors = _geometry_neighbors(geometry_neighbors, k)
    _check_inputs(vertices, observations)
    B, V, _ = vertices.shape
    K = max(1, min(k, V - 1))
    dev = vertices.device
    idx = torch.zeros((B, V, K), dtype=torch.long, device=dev)
    dp = torch.zeros((B, V, K, 3), dtype=torch.float32, device=dev)
    edge_mask = torch.zeros((B, V, K), dtype=torch.float32, device=dev)
    if V == 1 or B == 0:
        return idx, dp, edge_mask

    points = vertices.float()
    obs = observations.float()
    n_candidates = min(candidates, V - 1)
    scale_squared = geometry_scale * geometry_scale
    for start in range(0, V, chunk_size):
        stop = min(start + chunk_size, V)
        Q = stop - start
        query = points[:, start:stop]
        # Avoid the norm-expansion cdist path: close points on a translated hand
        # otherwise suffer cancellation before neighbours are ranked.
        distance_squared = torch.cdist(
            query, points, p=2, compute_mode="donot_use_mm_for_euclid_dist").square_()
        rows = torch.arange(Q, device=dev)
        distance_squared[:, rows, rows + start] = float("inf")
        candidate_idx = torch.argsort(distance_squared, dim=-1, stable=True)[..., :n_candidates]
        candidate_obs = obs.gather(
            1, candidate_idx.reshape(B, -1, 1).expand(-1, -1, OBS_DIM)
        ).reshape(B, Q, n_candidates, OBS_DIM)
        event_distance = (obs[:, start:stop, None] - candidate_obs).square().mean(-1)
        score = distance_squared.gather(-1, candidate_idx) / scale_squared
        if geometry_neighbors == 0:
            score = score + event_weight * event_distance
            selected = torch.argsort(score, dim=-1, stable=True)[..., :K]
            neighbour_idx = candidate_idx.gather(-1, selected)
        elif event_weight == 0.0:
            neighbour_idx = candidate_idx[..., :K]
        else:
            n_geometric = min(geometry_neighbors, K)
            query_observed = obs[:, start:stop, 0] > 0
            score = score + event_weight * torch.where(
                query_observed.unsqueeze(-1), event_distance, torch.zeros_like(event_distance))
            # Remove reserved entries before ranking, so neither the observed
            # selection nor the fallback can duplicate a geometric edge.
            remaining_idx = candidate_idx[..., n_geometric:]
            remaining_score = score[..., n_geometric:].masked_fill(
                candidate_obs[..., n_geometric:, 0] <= 0, float("inf"))
            # Finite scores (observed sources) precede the inf-scored missing
            # sources. Stable ties retain the original nearest-neighbour order.
            selected = torch.argsort(remaining_score, dim=-1, stable=True)[..., :K - n_geometric]
            neighbour_idx = torch.cat((candidate_idx[..., :n_geometric],
                                       remaining_idx.gather(-1, selected)), dim=-1)
        neighbour_points = points.gather(
            1, neighbour_idx.reshape(B, -1, 1).expand(-1, -1, 3)
        ).reshape(B, Q, K, 3)
        idx[:, start:stop] = neighbour_idx
        dp[:, start:stop] = (neighbour_points - query.unsqueeze(2)) / geometry_scale
        edge_mask[:, start:stop] = 1.0
    return idx, dp, edge_mask


class EventGuidedMeshEncoder(nn.Module):
    """Observe all vertices, pass supported messages, then pool with fixed LBS.

    ``forward(observations, vertices)`` returns joint evidence ``(B,J,C)`` and
    a boolean support map ``(B,J)``; MANO uses ``V=778`` and ``J=16``. Every
    layer expands support by one graph hop. Unobserved vertices remain graph
    nodes and may receive evidence, but an unsupported sender emits no message.

    With ``memory=True``, ``forward_step`` reads and returns explicit vertex
    memory; only current observations write it. Graph support can propagate
    temporarily, while current observation support alone opens joint heads.
    Empty packets traverse every trainable layer, preserving DDP connections,
    but produce exactly zero pooled evidence even with nonzero retained memory.
    Diagnostic tensors are detached and replaced on every forward call.
    """

    def __init__(self, lbs_weights: torch.Tensor, hidden: int = 128, n_layers: int = 3,
                 k: int = 8, candidates: int = 32, event_weight: float = 1.0,
                 geometry_scale: float = 0.02, memory: bool = False,
                 geometry_neighbors: int = 0):
        super().__init__()
        self.hidden = _positive_int("hidden", hidden)
        if isinstance(n_layers, bool) or not isinstance(n_layers, Integral) or n_layers < 0:
            raise ValueError("n_layers must be a non-negative integer")
        self.k, self.candidates, self.event_weight, self.geometry_scale, _ = _graph_options(
            k, candidates, event_weight, geometry_scale, 64)
        self.geometry_neighbors = _geometry_neighbors(geometry_neighbors, self.k)
        if not isinstance(memory, bool):
            raise ValueError("memory must be boolean")
        self.memory_enabled = memory
        if lbs_weights.ndim != 2 or min(lbs_weights.shape) < 1:
            raise ValueError("lbs_weights must have shape (V, J), with V and J positive")
        weights = lbs_weights.detach().float().clone()
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("lbs_weights must be finite and non-negative")
        self.n_nodes, self.n_joints = weights.shape
        sums = weights.sum(0, keepdim=True)
        # A zero-weight column describes an unsupported joint and pools to zero.
        normalised = weights / torch.where(sums > 0, sums, torch.ones_like(sums))
        self.register_buffer("lbs_weights", weights)
        self.register_buffer("lbs_pool_weights", normalised.transpose(0, 1).contiguous())
        self.obs_embed = nn.Linear(OBS_DIM, self.hidden)
        self.layers = nn.ModuleList([EdgeConv(self.hidden) for _ in range(int(n_layers))])
        self.memory = EventDrivenVertexMemory(self.hidden, self.n_nodes) if memory else None
        self.last_graph = None
        self.last_node_features = None
        self.last_support = None
        self.last_observed = None
        self.last_innovation = None
        self.last_readout_features = None

    def forward(self, observations: torch.Tensor, vertices: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.memory_enabled:
            evidence, joint_support, _ = self.forward_step(observations, vertices)
            return evidence, joint_support
        h, support, _ = self._convolve(observations, vertices)
        self.last_innovation = None
        self.last_readout_features = h.detach()
        return self._pool(h, support)

    def _convolve(self, observations: torch.Tensor, vertices: torch.Tensor,
                  state: Optional[MeshVertexState] = None
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _check_inputs(vertices, observations)
        if observations.shape[1] != self.n_nodes:
            raise ValueError("observation vertex count must match lbs_weights")
        idx, dp, graph_mask = build_event_guided_graph(
            vertices, observations, k=self.k, candidates=self.candidates,
            event_weight=self.event_weight, geometry_scale=self.geometry_scale,
            geometry_neighbors=self.geometry_neighbors)
        B, V, K = idx.shape
        observed = observations[..., 0] > 0
        support = observed
        h = torch.relu(self.obs_embed(observations.to(self.obs_embed.weight.dtype)))
        h = h * support.unsqueeze(-1).to(h.dtype)
        if state is not None:
            # Context can spread through the graph without writing into the
            # persistent memory of an unobserved receiver.
            h = h + torch.where(state.seen.unsqueeze(-1), state.hidden,
                                torch.zeros_like(state.hidden)).to(h.dtype)
            support = support | state.seen
        edge_features = dp.to(h.dtype)
        valid_edges = graph_mask.bool()
        for layer in self.layers:
            sender_support = support.gather(1, idx.reshape(B, V * K)).reshape(B, V, K)
            supported_edges = valid_edges & sender_support
            message = layer(h, idx, edge_features, supported_edges.to(h.dtype))
            support = support | supported_edges.any(-1)
            h = (h + message) * support.unsqueeze(-1).to(h.dtype)

        self.last_graph = (idx.detach(), dp.detach(), graph_mask.detach())
        self.last_node_features = h.detach()
        self.last_support = support.detach()
        self.last_observed = observed.detach()
        return h, support, observed

    def _pool(self, features: torch.Tensor, support: torch.Tensor
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        evidence = torch.einsum("jv,bvc->bjc", self.lbs_pool_weights.to(features.dtype), features)
        # This boolean readout depends only on fixed positive skinning weights,
        # avoiding underflow of very small weights when using mixed precision.
        joint_support = (support.to(torch.float32) @ (self.lbs_weights > 0).to(torch.float32)) > 0
        return evidence, joint_support

    def forward_step(self, observations: torch.Tensor, vertices: torch.Tensor,
                     state: Optional[MeshVertexState] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor, MeshVertexState]:
        """Read graph context; commit memory only at currently observed vertices.

        Pool ``observed_current * H_next`` with fixed LBS weights. Reading the
        updated state preserves sustained event information when the GRU reaches
        a fixed point; its innovation remains available as a diagnostic. Current
        observation support alone opens the decoder gates, so retained history
        cannot cause another motion update during an empty packet. All layers
        are evaluated even for empty packets, preserving DDP connections.
        """
        if self.memory is None:
            raise RuntimeError("forward_step requires memory=True")
        _check_inputs(vertices, observations)
        if state is not None:
            if not isinstance(state, MeshVertexState):
                raise ValueError("state must be a MeshVertexState or None")
            expected_shape = (*observations.shape[:2], self.hidden)
            if (state.hidden.shape != expected_shape or not state.hidden.is_floating_point()
                    or state.hidden.device != observations.device):
                raise ValueError("state.hidden must be floating point (B,V,C) on the input device")
            if (state.seen.shape != observations.shape[:2] or state.seen.dtype != torch.bool
                    or state.seen.device != observations.device):
                raise ValueError("state.seen must be boolean (B,V) on the input device")
        h, _, observed = self._convolve(observations, vertices, state)
        next_hidden, innovation = self.memory(h, observed, None if state is None else state.hidden)
        next_seen = observed if state is None else state.seen | observed
        next_state = MeshVertexState(next_hidden, next_seen)
        self.last_innovation = innovation.detach()
        readout_features = torch.where(observed.unsqueeze(-1), next_hidden,
                                       torch.zeros_like(next_hidden))
        self.last_readout_features = readout_features.detach()
        evidence, joint_support = self._pool(readout_features, observed)
        return evidence, joint_support, next_state


__all__ = ["OBS_DIM", "MeshVertexState", "build_event_guided_graph", "EventGuidedMeshEncoder"]
