r"""S45-PX: the previous state's MANO vertex position as three more relation-encoder inputs
(docs/EXPERIMENT_RECORDS.md#s45px-prereg-20261030).

S45 (`semkine.xyz_mesh.XYZMeshNet` on the C1 recipe) feeds every matched (event, vertex) record 13 numbers:
token7, dXYZ = (z d - V_j) / L_REL, zn = (z - Z_REF) / Z_SCALE, valid, hit. Because X = z d is fixed by the event
pixel and z, the vertex's camera position V_j = z d - L_REL dXYZ is already a (nonlinear) function of the record;
where the vertex sits on the hand is not. The two arms add the associated vertex V_j of the previous mesh, three
numbers per record (13 -> 16):

    camera   p = (V_j - (0, 0, Z_REF)) / Z_SCALE         absolute camera XYZ (the literal request)
    mano     p = (V_j - W) / Z_SCALE,  W = J_regressor[0] . V_prev
             the vertex relative to the previous wrist in camera axes, i.e. MANO's own frame without its
             translation; W is the wrist row of the joint regressor on the posed previous mesh, as C1SF ZR

They enter the relation encoder's first layer (before its ReLU) through a separate zero-initialised, bias-free
3 -> rel_hidden map built last in an isolated RNG scope: the XVM C1+sign slot of `XYZMeshNet.forward`
(`sign_in`, `sign_features`), reused unchanged. Every S45 tensor, its initial value and the random stream after
construction are S45's, and the initial function is S45's exactly, so training step 1 reproduces S45's step 1
(prediction, objective, every S45 gradient); from step 2 on only the learned map differs.

Opt-in only: `install(frame, npz)` replaces `semkine.xyz_mesh.XYZMeshNet` for the process (the model constructor
imports it at build time, so it must run before the model is built or loaded); `semkine/xyz_mesh.py` is unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import xyz_mesh as X

_ORIGINAL = X.XYZMeshNet
FRAMES = ("camera", "mano")
PX_IN = 3


def wrist_weights(npz_path) -> torch.Tensor:
    """MANO joint regressor row 0 (the wrist), `(778,)` float32, sums to one (as semkine/c1sf_zrel.py)."""
    with np.load(str(npz_path), allow_pickle=False) as z:
        w = np.asarray(z["J_regressor"], np.float64)[0]
    if w.shape != (778,) or not np.isfinite(w).all() or abs(w.sum() - 1.0) > 1e-6:
        raise ValueError("unexpected MANO J_regressor wrist row")
    return torch.from_numpy(w.astype(np.float32))


class PXXYZMeshNet(_ORIGINAL):
    """S45 plus the associated previous-mesh vertex position in every matched record; nothing else differs."""

    FRAME = None  # set by install()
    WRIST = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.FRAME not in FRAMES or self.WRIST is None:
            raise RuntimeError("s45px.install() must run before the model is built")
        if self.sign or self.contour or self.xag_gnn != "legacy" or self.residual_mode != "legacy" or not self.geom:
            raise ValueError("S45-PX is defined on the S45 (C1) recipe only")
        with torch.random.fork_rng(devices=[]):
            self.sign_in = nn.Linear(PX_IN, self.rel[0].out_features, bias=False)
        nn.init.zeros_(self.sign_in.weight)
        self.sign = True
        self.px_frame = self.FRAME
        self.register_buffer("px_wrist", self.WRIST.clone(), persistent=False)

    def sign_features(self, verts, rb, vrec, drec, pol):
        """`(R, 3)` per matched record: the associated vertex of the previous mesh in the arm's frame."""
        v = verts.float()
        p = v[rb, vrec]
        if self.px_frame == "mano":
            wrist = torch.einsum("v,bvc->bc", self.px_wrist.to(torch.float32), v)
            return (p - wrist[rb]) / X.Z_SCALE
        return (p - p.new_tensor((0.0, 0.0, X.Z_REF))) / X.Z_SCALE

    def forward(self, tok, d, mask, verts, assoc, prev48, ablate: bool = False, verts_geo=None, **kwargs):
        if verts_geo is not None:
            raise ValueError("S45-PX is defined on the S45 recipe (no oracle geometry)")
        return super().forward(tok, d, mask, verts, assoc, prev48, ablate=ablate, **kwargs)


def install(frame, npz_path) -> None:
    """Route the model constructor to `PXXYZMeshNet` in `frame` for this process (idempotent)."""
    if frame not in FRAMES:
        raise ValueError(f"S45-PX frame must be one of {FRAMES}, got {frame!r}")
    PXXYZMeshNet.FRAME = frame
    PXXYZMeshNet.WRIST = wrist_weights(npz_path)
    X.XYZMeshNet = PXXYZMeshNet


def installed():
    """The installed frame, or None."""
    return PXXYZMeshNet.FRAME if X.XYZMeshNet is PXXYZMeshNet else None
