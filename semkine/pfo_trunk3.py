r"""PFO cheaper message passing (registered only if the phase-2a MeshGNN controls show that propagation is load-bearing):
opt-in subclasses of S45's `XYZMeshNet` that keep the native forward and reduce the MeshGNN's cost.

  l1      one message round instead of three (the second and third `XYZMeshLayer` are removed after the native
          constructor, so every remaining parameter starts as in the base)
  gshare  the three rounds share one edge gate: the gate MLP of round 1 is evaluated once per forward and reused by
          rounds 2 and 3 (their own gate layers are removed); the per-edge gate is the dominant MeshGNN cost
`install(variant)` replaces `semkine.xyz_mesh.XYZMeshNet` before the model is built; `uninstall()` restores it.
"""
from __future__ import annotations

import torch

from . import xyz_mesh as XM

VARIANTS = ("l1", "gshare")
_NATIVE = {"cls": XM.XYZMeshNet}
_STATE = {"variant": None}


class SharedGateLayer(XM.XYZMeshLayer):
    """`XYZMeshLayer` whose gate is computed by the first round and passed in (`gate` precomputed)."""

    def forward(self, H, S, gate, idx, emask):
        B, V, C = H.shape
        K = idx.shape[1]
        flat = idx.reshape(-1)
        Mn = self.msg(H)[:, flat].view(B, V, K, C)
        Sn = S[:, flat].view(B, V, K) * emask.unsqueeze(0).to(S.dtype)
        w = Sn.unsqueeze(-1).to(Mn.dtype)
        agg = (gate * Mn * w).sum(2) / Sn.sum(2, keepdim=True).clamp_min(1.0).to(Mn.dtype)
        S_next = torch.maximum(S, Sn.amax(2))
        H = H + S_next.unsqueeze(-1).to(H.dtype) * torch.relu(self.slf(H) + agg)
        return H, S_next


class PFOMeshNet3(XM.XYZMeshNet):
    variant = "l1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.variant == "l1":
            self.layers = torch.nn.ModuleList(list(self.layers)[:1])
        elif self.variant == "gshare":
            first = self.layers[0]
            shared = []
            for layer in self.layers:
                s = SharedGateLayer.__new__(SharedGateLayer)
                torch.nn.Module.__init__(s)
                s.msg, s.slf = layer.msg, layer.slf
                shared.append(s)
            self.gate_shared = first.gate
            self.layers = torch.nn.ModuleList(shared)
            self._gemb_to_gate = True
        else:
            raise ValueError(self.variant)

    def forward(self, *args, **kwargs):
        if self.variant != "gshare":
            return super().forward(*args, **kwargs)
        # the native forward computes gemb once and hands it to every layer: wrap the layers so the first call turns
        # gemb into the shared gate and every layer receives that gate
        cache = {}
        native = [layer.forward for layer in self.layers]

        def make(fwd):
            def f(H, S, gemb, idx, emask):
                if "gate" not in cache:
                    cache["gate"] = torch.sigmoid(self.gate_shared(gemb))
                return fwd(H, S, cache["gate"], idx, emask)
            return f
        try:
            for layer, fwd in zip(self.layers, native):
                layer.forward = make(fwd)
            return super().forward(*args, **kwargs)
        finally:
            for layer in self.layers:
                del layer.forward


def install(variant):
    if variant not in VARIANTS:
        raise ValueError(f"variant in {VARIANTS}")
    if XM.XYZMeshNet is not _NATIVE["cls"]:
        raise RuntimeError("XYZMeshNet already patched")
    XM.XYZMeshNet = type(f"PFOMeshNet3_{variant}", (PFOMeshNet3,), {"variant": variant})
    _STATE["variant"] = variant
    return variant


def uninstall():
    XM.XYZMeshNet = _NATIVE["cls"]
    _STATE["variant"] = None


def installed():
    return _STATE["variant"]


__all__ = ["PFOMeshNet3", "SharedGateLayer", "VARIANTS", "install", "installed", "uninstall"]
