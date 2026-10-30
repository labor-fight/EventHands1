r"""C1SF arm ZR: the relation encoder's depth channel relative to the previous state's root depth
(docs/EXPERIMENT_RECORDS.md#c1sf-prereg-20261029).

C1 (`semkine.xyz_mesh.XYZMeshNet`) feeds every matched record `zn = (z - Z_REF) / Z_SCALE`, an absolute
camera depth. This arm changes only the reference:

    zn = (z - z_root) / Z_SCALE,    z_root = depth of J_regressor[0] . V_prev   (per sample)

`J_regressor[0]` is the MANO wrist row of the joint regressor, applied to the posed previous mesh that the
association already uses; it approximates the forward-kinematics wrist (the residual is the pose-blendshape and
skinning offset of the wrist vertices, a few millimetres at most, see tests/test_c1sf.py). `X = z d`, `dXYZ`,
the edge geometry, the association and every parameter are C1's. The module file `semkine/xyz_mesh.py` stays
untouched: the subclass sets the module constant `Z_REF` to the per-record reference for the duration of
`XYZMeshNet.forward`, whose only use of `Z_REF` is that one line. With a constant reference of 0.45 m the
output is C1's bit for bit.

Opt-in only: `install(npz)` replaces `semkine.xyz_mesh.XYZMeshNet` for the process; the model constructor
imports it at build time, so it must run before the model is built or loaded. Buffers are non-persistent,
so the state dict, the parameter count and the initial values are C1's.
"""
from __future__ import annotations

import numpy as np
import torch

from . import xyz_mesh as X

_ORIGINAL = X.XYZMeshNet


def wrist_weights(npz_path) -> torch.Tensor:
    """MANO joint regressor row 0 (the wrist), `(778,)` float32, sums to one."""
    with np.load(str(npz_path), allow_pickle=False) as z:
        w = np.asarray(z["J_regressor"], np.float64)[0]
    if w.shape != (778,) or not np.isfinite(w).all() or abs(w.sum() - 1.0) > 1e-6:
        raise ValueError("unexpected MANO J_regressor wrist row")
    return torch.from_numpy(w.astype(np.float32))


def record_rows(assoc, mask) -> torch.Tensor:
    """The batch index of every matched record, in `XYZMeshNet.forward`'s order (same expression)."""
    mask_f = mask.to(torch.float32)
    w = assoc["q"].unsqueeze(-1) * assoc["A"] * mask_f.unsqueeze(-1)
    rb, _, _ = (w > 0).nonzero(as_tuple=True)
    return rb


class ZRelXYZMeshNet(_ORIGINAL):
    """C1 with `zn` relative to the previous root depth; nothing else differs."""

    WRIST = None  # set by install()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.WRIST is None:
            raise RuntimeError("c1sf_zrel.install() must run before the model is built")
        self.register_buffer("c1sf_wrist", self.WRIST.clone(), persistent=False)

    def root_depth(self, verts: torch.Tensor) -> torch.Tensor:
        """`(B,)` metres: wrist-row regression of the posed previous mesh, camera z."""
        return torch.einsum("v,bv->b", self.c1sf_wrist.to(torch.float32), verts[..., 2].to(torch.float32))

    def forward(self, tok, d, mask, verts, assoc, prev48, ablate: bool = False, verts_geo=None, **kwargs):
        if verts_geo is not None:
            raise ValueError("C1SF ZR is defined on the C1 recipe (no oracle geometry)")
        reference = X.Z_REF
        if not ablate:
            reference = self.root_depth(verts)[record_rows(assoc, mask)]
        saved = X.Z_REF
        X.Z_REF = reference
        try:
            return super().forward(tok, d, mask, verts, assoc, prev48, ablate=ablate, verts_geo=verts_geo, **kwargs)
        finally:
            X.Z_REF = saved


def install(npz_path) -> None:
    """Route the model constructor to `ZRelXYZMeshNet` for this process (idempotent)."""
    ZRelXYZMeshNet.WRIST = wrist_weights(npz_path)
    X.XYZMeshNet = ZRelXYZMeshNet


def installed() -> bool:
    return X.XYZMeshNet is ZRelXYZMeshNet


__all__ = ["ZRelXYZMeshNet", "install", "installed", "record_rows", "wrist_weights"]
