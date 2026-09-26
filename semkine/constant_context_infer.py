"""Opt-in, inference-only execution of the already fitted CC0 constant bias.

``enable_constant_context_inference(encoder)`` updates an existing S37
EventGNN in place. Its original embed/layers/proj and parameter names remain;
``context.bias`` is preserved if present, or registered as zeros so the full
CC0 state_dict can subsequently be loaded strictly. No checkpoint is read by
this module. The caller owns model/config identity and whole-model eval mode.

The original EventGNN forward supplies the unchanged three-layer nodes and g
with its internal readout disabled. One final readout follows the constant
residual. Only the residual mask is needed: no additional context idx/dp are
built. The original W32 graph remains. Eight-slot broadcast/mask/sum/div and
autocast precision are deliberately retained for finite-precision parity.

This module has no scratch/training imports and is not a training replacement.
Installation freezes the encoder; forward runs under no_grad; train(True) is
rejected. Nothing in this implementation establishes a latency bound.
"""
from __future__ import annotations

import torch
from torch import nn

from .event_gnn import EventGNN


def constant_context_mask(mask: torch.Tensor) -> torch.Tensor:
    """Exactly the original SC0 L valid-slot mask, including sparse padding."""
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("mask must be bool with shape (batch,n)")
    previous = mask.long().cumsum(-1) - mask.long()
    degree = previous.clamp_max(8) * mask.long()
    slot = torch.arange(8, device=mask.device).view(1, 1, 8)
    return (mask.unsqueeze(-1) & (slot < degree.unsqueeze(-1))).to(torch.float32)


class ConstantContextBias(nn.Module):
    """CC0 bias and its original eight-slot arithmetic; no source features."""
    def __init__(self, bias: nn.Parameter | None = None, *, device=None, dtype=None):
        super().__init__()
        if bias is not None and (not isinstance(bias, nn.Parameter) or bias.shape != (128,)):
            raise ValueError("existing bias must be one nn.Parameter of shape (128,)")
        self.bias = bias if bias is not None else nn.Parameter(torch.zeros(128, device=device, dtype=dtype))

    def forward(self, h: torch.Tensor, emask: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3 or h.shape[-1] != 128:
            raise ValueError("h must have shape (batch,n,128)")
        if emask.ndim != 3 or emask.shape[:2] != h.shape[:2] or emask.shape[-1] != 8:
            raise ValueError("emask must have shape (batch,n,8)")
        if self.bias.device != h.device or emask.device != h.device:
            raise ValueError("bias, h and emask must share a device")
        dtype = self.bias.dtype
        if h.device.type == "cuda" and torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
        elif h.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            dtype = torch.get_autocast_cpu_dtype()
        message = self.bias.to(dtype=dtype).view(1, 1, 1, 128).expand(*emask.shape, 128)
        live = emask.to(message.dtype)
        message = message * live.unsqueeze(-1)
        return message.sum(2) / live.sum(2, keepdim=True).clamp_min(1.0)


class ConstantContextInference(EventGNN):
    """Installed EventGNN subclass; original graph and layers, one final pool."""
    def train(self, mode: bool = True):
        if mode:
            raise RuntimeError("constant-context optimized encoder is inference-only")
        return super().train(False)

    @torch.no_grad()
    def forward(self, events, ptr, delta_t_s, extra=None, return_nodes=False):
        if self.training or self.readout:
            raise RuntimeError("inference installation requires eval mode and base readout=False")
        _, h, g, px, py, mask = EventGNN.forward(
            self, events, ptr, delta_t_s, extra, return_nodes=True)
        if h.shape[1] == 0:
            # The original all-empty fast path uses the embed weight dtype,
            # even inside BF16 autocast. It never calls proj/max/context.
            out = torch.zeros(int(ptr.numel() - 1), self.feat_dim,
                              device=events.device, dtype=self.embed.weight.dtype)
        else:
            emask = constant_context_mask(mask)
            h = (h + self.context(h, emask)) * mask.unsqueeze(-1)
            # Preserve the original expression order, including mixed-empty
            # rows and signed zeros. No special per-row empty shortcut.
            any_node = mask.any(1, keepdim=True).to(h.dtype)
            live = mask.sum(1, keepdim=True).clamp_min(1.0).to(h.dtype)
            mean = h.sum(1) / live
            peak = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
            out = self.proj(torch.cat([mean, peak * any_node], dim=-1))
            out = out * any_node
        result = out, h, g, px, py, mask
        return result if return_nodes else out


def enable_constant_context_inference(encoder: EventGNN) -> ConstantContextInference:
    """Install once, in place; preserve all existing parameter objects/keys.

    Accept either a freshly constructed source EventGNN (register zero bias,
    then strictly load the complete CC0 state_dict) or a loaded CC0 encoder
    whose only context state is bias[128]. No original module is duplicated.
    This is an explicit inference boundary, not reversible training setup.
    """
    if not isinstance(encoder, EventGNN) or isinstance(encoder, ConstantContextInference):
        raise TypeError("requires an EventGNN without the inference installation")
    if encoder.hidden != 128 or len(encoder.layers) != 3 or encoder.k != 8 or encoder.window != 32:
        raise ValueError("requires S37 hidden128/three-layers/k8/W32")
    if not encoder.readout or not isinstance(getattr(encoder, "proj", None), nn.Module):
        raise ValueError("requires the existing global proj and original readout=True")
    if getattr(encoder, "context_mode", "L") != "L":
        raise ValueError("only the registered CC0 L-mask constant context is supported")
    if "_edges" in encoder.__dict__ or "forward" in encoder.__dict__:
        raise ValueError("remove instance graph/forward overrides before inference installation")
    old_context = getattr(encoder, "context", None)
    if old_context is None:
        context = ConstantContextBias(device=encoder.embed.weight.device, dtype=encoder.embed.weight.dtype)
    else:
        if not isinstance(old_context, nn.Module) or set(old_context.state_dict()) != {"bias"}:
            raise ValueError("existing context must contain only the CC0 bias")
        context = ConstantContextBias(old_context.bias)
    encoder.context = context
    encoder.__class__ = ConstantContextInference
    encoder.readout = False  # Set once; forward never mutates this flag.
    for name in ("_context_p", "last_edges", "last_support"):
        if name in encoder.__dict__:
            delattr(encoder, name)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder
