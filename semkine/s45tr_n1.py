r"""S45-TR arm N1: the readout sees no previous-pose vector (docs/EXPERIMENT_RECORDS.md#s45tr-prereg-20261030).

S45 (`semkine.xyz_mesh.XYZMeshNet`) reads out the 51D update from `fc1([zs, prev48])`, where `prev48 = prev[3:51]`
is the previous root rotation and finger pose. S45-DEEP measured that the update shrinks the fingers toward the
training mean pose by 10-17 % per step; a readout can only do that if it knows the current pose, which `prev48`
hands it directly. N1 feeds that vector as zeros, in training and at evaluation alike; nothing else changes.

`prev48` has no other use inside `XYZMeshNet.forward` (the association, the record geometry and the mesh edges all
come from `verts`, the posed previous mesh, which still carries the previous shape). The 48 input columns of `fc1`
stay in the state dict with their initial values: their input is exactly zero, so their gradient is exactly zero and
Adam never moves them. Parameter count, parameter order and initial values are S45's.

Opt-in only: `install()` replaces `semkine.xyz_mesh.XYZMeshNet` for the process; the model constructor imports the
class at build time, so it must run before the model is built or loaded.
"""
from __future__ import annotations

import torch

from . import xyz_mesh as X

_ORIGINAL = X.XYZMeshNet


class N1XYZMeshNet(_ORIGINAL):
    """S45 with the readout's previous-pose input held at zero."""

    def forward(self, tok, d, mask, verts, assoc, prev48, *args, **kwargs):
        return super().forward(tok, d, mask, verts, assoc, torch.zeros_like(prev48), *args, **kwargs)


def install() -> None:
    """Route the model constructor to `N1XYZMeshNet` for this process (idempotent)."""
    X.XYZMeshNet = N1XYZMeshNet


def installed() -> bool:
    return X.XYZMeshNet is N1XYZMeshNet


__all__ = ["N1XYZMeshNet", "install", "installed"]
