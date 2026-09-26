"""Explicit event-driven vertex memory for mesh tracking.

The caller owns the state and passes it between consecutive packets. Missing
observations leave that vertex's memory unchanged. A separate innovation tensor
contains only the current packet's memory update, so a delta decoder need not
interpret retained history as a fresh motion observation.

This module stores latent features, not vertex coordinates or MANO parameters.
Holding its state does not constrain the final mesh's physical motion.
"""
from __future__ import annotations

from numbers import Integral
from typing import Optional, Tuple

import torch
from torch import nn


class EventDrivenVertexMemory(nn.Module):
    """Update observed vertices and return ``(next_state, innovation)``.

    ``features`` and ``previous`` have shape ``(B,V,H)``; ``observed`` has shape
    ``(B,V)`` and must mark actual current-packet observation availability, not
    whether an embedding is nonzero or the vertex was observed in the past.
    With ``previous=None`` a new independent sequence starts from zero memory.

    ``innovation = next_state - previous`` only at observed vertices; it is
    exactly zero elsewhere. Consumers must also gate biased delta heads using
    current observations to preserve the empty-packet state identity.

    No history is cached on the module. Sequence boundaries, batch identities,
    and truncated-gradient boundaries are controlled explicitly by the caller.
    """

    def __init__(self, hidden: int, n_vertices: int = 778):
        super().__init__()
        for name, value in (("hidden", hidden), ("n_vertices", n_vertices)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.hidden = int(hidden)
        self.n_vertices = int(n_vertices)
        self.cell = nn.GRUCell(self.hidden, self.hidden)

    def forward(self, features: torch.Tensor, observed: torch.Tensor,
                previous: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[1:] != (self.n_vertices, self.hidden):
            raise ValueError("features must have shape (B, n_vertices, hidden)")
        if not features.is_floating_point():
            raise ValueError("features must be floating-point")
        if observed.shape != features.shape[:2] or observed.dtype != torch.bool:
            raise ValueError("observed must be a boolean tensor with shape (B, n_vertices)")
        if observed.device != features.device:
            raise ValueError("observed and features must be on the same device")
        if features.device != self.cell.weight_ih.device:
            raise ValueError("features and the memory module must be on the same device")
        if previous is None:
            # Keep persistent memory at parameter precision, even when the graph
            # emits bf16 features under mixed precision.
            previous = features.new_zeros(features.shape, dtype=self.cell.weight_ih.dtype)
        elif (previous.shape != features.shape or previous.device != features.device
              or not previous.is_floating_point()):
            raise ValueError("previous must be floating-point with the same shape/device as features")

        # Torch 2.1 CUDA's fused GRUCell has no bf16 implementation. Casting the
        # inputs alone is insufficient because autocast would cast its matmuls
        # again. Keep the recurrent cell at parameter precision (fp32 in mixed
        # training), while the surrounding graph/heads still use autocast.
        with torch.autocast(device_type=features.device.type, enabled=False):
            proposal = self.cell(
                features.to(self.cell.weight_ih.dtype).reshape(-1, self.hidden),
                previous.to(self.cell.weight_hh.dtype).reshape(-1, self.hidden),
            ).reshape_as(previous).to(previous.dtype)
        mask = observed.unsqueeze(-1)
        # where retains the original values exactly; multiplicative blending
        # or casting the entire state to proposal precision could round history.
        next_state = torch.where(mask, proposal, previous)
        innovation = torch.where(mask, next_state - previous, torch.zeros_like(previous))
        return next_state, innovation


__all__ = ["EventDrivenVertexMemory"]
