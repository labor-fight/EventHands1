"""Equal-capacity residual evidence controls for the S37 XYZ experiments.

The existing residual slot remains in ``XYZMeshNet``.  This optional branch
adds 64 readout inputs in all four arms:

* A copies a packet-wide content summary and packet-wide count/quality metadata
  to 16 slots, with fixed centers serving only as distinct slot identities;
* B pools content in a fixed 4 x 4 sensor grid;
* C uses B's grid and metadata but exchanges content between residual events;
* D uses B's metadata with the content summary identically zero.

Content uses only token columns 2:7.  Raw xy, source cell IDs, and the exchange
map never enter its encoder.  C retains each destination's residual weight,
so in weak-quality strata it breaks both content-position and content-quality
binding; the exact-r=1 stratum preserves quality.  Diagnostics
measure the exchange instead of assuming that it changes the representation.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

RESIDUAL_MODES = ("legacy", "A", "B", "C", "D")


def residual_permutation(live: torch.Tensor, seed: int = 19073,
                         groups: torch.Tensor | None = None) -> torch.Tensor:
    """Deterministic packet-local content exchange, independent of global RNG.

    Hash the ordinal of each live residual event, then rotate each stratum's
    hash ordering by half its length.  With two or more live events in a
    stratum this is a derangement. ``groups`` must be integer IDs in [0, 19].
    Padding width, unrelated batch members and train/eval mode cannot change
    a packet's permutation.  Non-residual entries retain their own index.
    """
    B, N = live.shape
    index = torch.arange(N, device=live.device).expand(B, -1)
    if N == 0:
        return index
    if groups is None:
        groups = torch.zeros_like(index)
    count = torch.zeros(B, 20, device=live.device, dtype=torch.long)
    count.scatter_add_(1, groups, live.long())
    start = torch.cat([torch.zeros_like(count[:, :1]), count.cumsum(1)[:, :-1]], 1)
    ordinal = (live.long().cumsum(1) - 1).clamp_min(0)
    h = (ordinal + int(seed)) & 0xFFFFFFFF
    h = ((h ^ (h >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    h = ((h ^ (h >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    h = (h + groups * (1 << 32)).masked_fill(~live, torch.iinfo(torch.int64).max)
    order = h.argsort(dim=1, stable=True)
    inverse = torch.empty_like(order).scatter_(1, order, index)
    own_count, own_start = count.gather(1, groups), start.gather(1, groups)
    source_rank = own_start + (inverse - own_start + (own_count + 1) // 2) % own_count.clamp_min(1)
    source_rank = source_rank.clamp(max=N - 1)
    source = order.gather(1, source_rank)
    return torch.where(live, source, index)


class XYZResidualBranch(nn.Module):
    """A-D residual representation with identical modules and parameter count.

    ``forward(tok, r, mask)`` returns ``(flat[B, 64], aux)``.  The fixed
    centers are the only location inputs to the shared slot MLP.  Empty slots
    are gated to exact zero after the MLP, including its biases.
    """

    def __init__(self, mode: str, grid: int = 4, content_dim: int = 16,
                 slot_dim: int = 4, content_hidden: int = 32,
                 slot_hidden: int = 32, permutation_seed: int = 19073):
        super().__init__()
        if mode not in RESIDUAL_MODES[1:]:
            raise ValueError(f"residual branch mode must be A/B/C/D, got {mode!r}")
        if int(grid) <= 0:
            raise ValueError("grid must be positive")
        self.mode = mode
        self.grid = int(grid)
        self.n_slots = self.grid ** 2
        self.content_dim = int(content_dim)
        self.slot_dim = int(slot_dim)
        self.out_dim = self.n_slots * self.slot_dim
        self.permutation_seed = int(permutation_seed)
        self.content = nn.Sequential(nn.Linear(5, content_hidden), nn.ReLU(),
                                     nn.Linear(content_hidden, self.content_dim))
        self.slot = nn.Sequential(nn.Linear(self.content_dim + 5, slot_hidden), nn.ReLU(),
                                  nn.Linear(slot_hidden, self.slot_dim))
        center = (torch.arange(self.grid, dtype=torch.float32) + 0.5) / self.grid
        yy, xx = torch.meshgrid(center, center, indexing="ij")
        self.register_buffer("centers", torch.stack([xx, yy], -1).reshape(self.n_slots, 2),
                             persistent=False)

    def _mean(self, features: torch.Tensor, weight: torch.Tensor,
              cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, C = features.shape
        count = torch.zeros(B, self.n_slots, device=features.device, dtype=torch.float32)
        mass = torch.zeros_like(count)
        count.scatter_add_(1, cell, (weight > 0).float())
        mass.scatter_add_(1, cell, weight.float())
        numerator = torch.zeros(B, self.n_slots, C, device=features.device, dtype=torch.float32)
        numerator.scatter_add_(1, cell.unsqueeze(-1).expand(-1, -1, C),
                               features.float() * weight.unsqueeze(-1).float())
        mean = numerator / mass.clamp_min(1e-12).unsqueeze(-1)
        return mean, mass, count

    def forward(self, tok: torch.Tensor, r: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if tok.ndim != 3 or tok.shape[-1] != 7 or tok.shape[:2] != r.shape or r.shape != mask.shape:
            raise ValueError("expected tok[B,N,7], r[B,N], mask[B,N]")
        B, N, _ = tok.shape
        dtype = self.content[0].weight.dtype
        weight = r.float() * mask.float()
        live = weight > 0
        # xy only determines the destination grid.  It is never encoded as content.
        xy = tok[..., :2].float()
        ij = (xy * self.grid).floor().long().clamp(0, self.grid - 1)
        cell = ij[..., 1] * self.grid + ij[..., 0]
        content = self.content(tok[..., 2:].to(dtype))
        permutation = torch.arange(N, device=tok.device).expand(B, -1)
        exchanged = content
        # Four weak-quality strata plus an exact-r=1 stratum preserve full
        # residual quality exactly. Weak strata have width .25. Coarse time
        # and polarity are preserved; fine values can still differ.
        quality_bin = (weight * 4).floor().long().clamp(0, 4)
        time_bin = (tok[..., 3].float() * 2).floor().long().clamp(0, 1)
        polarity_bin = (tok[..., 2] > 0).long()
        group = quality_bin * 4 + time_bin * 2 + polarity_bin
        group_count = torch.zeros(B, 20, device=tok.device, dtype=torch.long)
        group_count.scatter_add_(1, group, live.long())
        eligible = live & (group_count.gather(1, group) > 1)
        if self.mode == "C":
            permutation = residual_permutation(live, self.permutation_seed, group)
            exchanged = content.gather(1, permutation.unsqueeze(-1).expand(-1, -1, self.content_dim))
        original_mean, mass, count = self._mean(content, weight, cell)
        local_mean, _, _ = self._mean(exchanged, weight, cell)
        packet_live = mask.float().sum(1, keepdim=True).clamp_min(1.0)
        if self.mode == "A":
            total_mass = weight.sum(1, keepdim=True)
            total_count = live.float().sum(1, keepdim=True)
            packet_mean = (content.float() * weight.unsqueeze(-1)).sum(1) / total_mass.clamp_min(1e-12)
            means = packet_mean.unsqueeze(1).expand(-1, self.n_slots, -1)
            mass = total_mass.expand(-1, self.n_slots)
            count = total_count.expand(-1, self.n_slots)
            centers = self.centers.unsqueeze(0).expand(B, -1, -1)
        else:
            means = local_mean
            if self.mode == "D":
                # Keep all content parameters and initialization; gradients are zero.
                means = means * 0.0
            centers = self.centers.unsqueeze(0).expand(B, -1, -1)
        metadata = torch.cat([
            (torch.log1p(count) / torch.log1p(packet_live)).unsqueeze(-1),
            (mass / packet_live).unsqueeze(-1),
            (mass / count.clamp_min(1.0)).unsqueeze(-1), centers,
        ], -1)
        has = mass > 0
        slots = self.slot(torch.cat([means, metadata], -1).to(dtype)) * has.unsqueeze(-1).to(dtype)
        source_cell = cell.gather(1, permutation)
        source_xy = xy.gather(1, permutation.unsqueeze(-1).expand(-1, -1, 2))
        denominator = live.sum().clamp_min(1).float()
        swap_fraction = ((permutation != torch.arange(N, device=tok.device)) & live).sum().float() / denominator
        crosscell_fraction = ((source_cell != cell) & live).sum().float() / denominator
        swap_distance = ((source_xy - xy).norm(dim=-1) * live).sum() / denominator
        changed = (permutation != torch.arange(N, device=tok.device)) & live
        crosscell = (source_cell != cell) & live
        mass_denominator = weight.sum().clamp_min(1e-12)
        quality_diff = (weight.gather(1, permutation) - weight).abs()
        full_residual = live & (weight == 1.0)
        full_count = full_residual.sum().clamp_min(1).float()
        occupied = (mass > 0).float()
        feature_diff = ((local_mean - original_mean).abs().mean(-1) * occupied).sum() / occupied.sum().clamp_min(1)
        aux = {
            "residual_extra": slots.reshape(B, self.out_dim),
            "residual_slot_features": slots,
            "residual_content": means,
            "residual_metadata": metadata,
            "residual_mass": mass,
            "residual_count": count,
            "residual_cell": cell,
            "residual_permutation": permutation,
            "residual_group": group,
            "residual_eligible": eligible,
            "residual_eligible_fraction": eligible.sum().float() / denominator,
            "residual_swap_fraction": swap_fraction,
            "residual_eligible_swap_fraction": (changed & eligible).sum().float() / eligible.sum().clamp_min(1),
            "residual_weighted_swap_fraction": (changed * weight).sum() / mass_denominator,
            "residual_crosscell_fraction": crosscell_fraction,
            "residual_weighted_crosscell_fraction": (crosscell * weight).sum() / mass_denominator,
            "residual_swap_distance": swap_distance,
            "residual_quality_diff": (quality_diff * live).sum() / denominator,
            "residual_r1_crosscell_fraction": (crosscell & full_residual).sum().float() / full_count,
            "residual_r1_quality_diff": (quality_diff * full_residual).sum() / full_count,
            "residual_feature_diff": feature_diff,
        }
        return aux["residual_extra"], aux


__all__ = ["RESIDUAL_MODES", "XYZResidualBranch", "residual_permutation"]
