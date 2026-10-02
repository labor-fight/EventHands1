#!/usr/bin/env python3
"""U1a: matched 17-slot measurement heads over the unchanged S38 evidence.

Slots are fixed: 0 translation delta, 1 root reference-rotation residual, and
2..16 the fifteen MANO residual-pose joints, each producing three numbers.
The caller owns root SO(3) conversion, finger residual/mean conventions,
state feedback, filtering and the empty-packet hold.

The packed observation support is identical in shared and untied modes:
  slot 0: projected feature + all ordered routed evidence + the full prev51;
  slots 1..16: the encoder's unprojected state-free pool only.
Type/joint embeddings identify coordinates; they do not select another MLP
in shared mode. LayerNorm normalizes only the last (feature) dimension of
one slot. There is no state-to-state message, local gate or cross-slot norm.

Translation predicts the complete delta target previously represented by
S38's routed translation head PLUS prev_mlp[:3]. The caller must not add an
extra dedicated prev MLP, otherwise the prior is counted twice.
"""
from __future__ import annotations

import torch
from torch import nn


class U1aReadout(nn.Module):
    """One shared or seventeen untied, identically structured 3D node heads.

    ``forward(pooled, feat, evidence, prev)`` returns
    ``(t_delta (B,3), root_raw (B,3), finger_raw (B,45))``.
    The 45 finger outputs remain residual45 coordinates, not full axis-angle
    with MANO hands_mean already added.
    """

    N_JOINTS = 16
    N_SLOTS = 17
    PREV_DIM = 51

    def __init__(self, *, pooled_dim: int, feat_dim: int, evidence_dim: int,
                 hidden: int = 64, type_dim: int = 8, joint_dim: int = 16,
                 mode: str = "shared"):
        super().__init__()
        dims = {"pooled_dim": pooled_dim, "feat_dim": feat_dim,
                "evidence_dim": evidence_dim, "hidden": hidden,
                "type_dim": type_dim, "joint_dim": joint_dim}
        for name, value in dims.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            setattr(self, name, int(value))
        if mode not in ("shared", "untied"):
            raise ValueError(f"mode must be shared or untied, got {mode!r}")
        self.mode = mode

        # Fixed field widths and masks, independently of head-sharing mode.
        widths = (("pool", self.pooled_dim), ("feat", self.feat_dim),
                  ("evidence", self.N_JOINTS * self.evidence_dim),
                  ("prev", self.PREV_DIM))
        self.field_slices = {}
        offset = 0
        for name, width in widths:
            self.field_slices[name] = slice(offset, offset + width)
            offset += width
        self.field_dim = offset
        self.input_dim = self.field_dim + self.type_dim + self.joint_dim
        field_mask = torch.zeros(self.N_SLOTS, self.field_dim, dtype=torch.bool)
        for name in ("feat", "evidence", "prev"):
            field_mask[0, self.field_slices[name]] = True
        field_mask[1:, self.field_slices["pool"]] = True
        self.register_buffer("field_mask", field_mask, persistent=False)
        # Type 0 translation, 1 root rotation, 2 finger rotation. Both root
        # coordinates use joint id 0; finger slots use MANO joint ids 1..15.
        self.register_buffer("slot_type", torch.tensor([0, 1] + [2] * 15),
                             persistent=False)
        self.register_buffer("slot_joint", torch.tensor([0, 0] + list(range(1, 16))),
                             persistent=False)
        self.type_embed = nn.Embedding(3, self.type_dim)
        self.joint_embed = nn.Embedding(self.N_JOINTS, self.joint_dim)
        if self.mode == "shared":
            self.shared_head = self._make_head()
        else:
            self.node_heads = nn.ModuleList([self._make_head() for _ in range(self.N_SLOTS)])

    def _make_head(self) -> nn.Sequential:
        head = nn.Sequential(nn.LayerNorm(self.input_dim),
                             nn.Linear(self.input_dim, self.hidden), nn.GELU(),
                             nn.Linear(self.hidden, 3))
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    def _check_inputs(self, pooled, feat, evidence, prev) -> None:
        if pooled.ndim != 2 or pooled.shape[1] != self.pooled_dim:
            raise ValueError(f"pooled must have shape (B, {self.pooled_dim}), "
                             f"got {tuple(pooled.shape)}")
        B = pooled.shape[0]
        expected = (("feat", feat, (B, self.feat_dim)),
                    ("evidence", evidence, (B, self.N_JOINTS, self.evidence_dim)),
                    ("prev", prev, (B, self.PREV_DIM)))
        for name, value, shape in expected:
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
            if value.device != pooled.device:
                raise ValueError(f"{name} and pooled must be on the same device")
        if pooled.device != self.type_embed.weight.device:
            raise ValueError("inputs and U1aReadout must be on the same device")

    def pack_fields(self, pooled, feat, evidence, prev) -> torch.Tensor:
        """Fixed masked fields, shape (B,17,field_dim); no identities yet.

        torch.where gives masked fields literal zeros, even when excluded
        values are non-finite; multiplying by zero would not do that.
        """
        self._check_inputs(pooled, feat, evidence, prev)
        dtype = self.type_embed.weight.dtype
        fields = torch.cat([pooled.to(dtype=dtype), feat.to(dtype=dtype),
                            evidence.flatten(1).to(dtype=dtype), prev.to(dtype=dtype)], -1)
        return torch.where(self.field_mask.unsqueeze(0), fields.unsqueeze(1),
                           fields.new_zeros(()))

    def node_inputs(self, pooled, feat, evidence, prev) -> torch.Tensor:
        """The exact input shared by both modes, shape (B,17,input_dim)."""
        fields = self.pack_fields(pooled, feat, evidence, prev)
        identity = torch.cat([self.type_embed(self.slot_type),
                              self.joint_embed(self.slot_joint)], -1)
        return torch.cat([fields, identity.unsqueeze(0).expand(fields.shape[0], -1, -1)], -1)

    def forward(self, pooled, feat, evidence, prev):
        x = self.node_inputs(pooled, feat, evidence, prev)
        if self.mode == "shared":
            out = self.shared_head(x)
        else:
            out = torch.stack([head(x[:, j]) for j, head in enumerate(self.node_heads)], dim=1)
        return out[:, 0], out[:, 1], out[:, 2:].reshape(x.shape[0], 45)


@torch.no_grad()
def copy_shared_to_untied(shared: U1aReadout, untied: U1aReadout) -> None:
    """Copy identities and the same head to all slots for an equality control.

    This is an initialization/test helper, not a constraint during training.
    Shared and untied modes otherwise retain their own parameter topology.
    """
    if shared.mode != "shared" or untied.mode != "untied":
        raise ValueError("copy_shared_to_untied expects shared then untied modules")
    for name in ("pooled_dim", "feat_dim", "evidence_dim", "hidden", "type_dim", "joint_dim"):
        if getattr(shared, name) != getattr(untied, name):
            raise ValueError(f"shared and untied must have equal {name}")
    untied.type_embed.load_state_dict(shared.type_embed.state_dict())
    untied.joint_embed.load_state_dict(shared.joint_embed.state_dict())
    head_state = shared.shared_head.state_dict()
    for head in untied.node_heads:
        head.load_state_dict(head_state)


__all__ = ["U1aReadout", "copy_shared_to_untied"]
