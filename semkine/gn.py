#!/usr/bin/env python3
r"""S14 active-set Gauss-Newton / Levenberg-Marquardt refinement.

The network's 51D output is a proposal. This module treats the S8 residual

    r_i = n_i^T (u_i - π(X_i(x)))

as an actual least-squares problem and takes a few damped Newton steps on the joints the
router selected. S8 found that `Lambda` is genuinely singular -- hidden joints contribute no
rows -- so a Cholesky solve on the full 51D system would fail for the right reason. The solve
is therefore:

* restricted to the active set (root always, plus the routed fingers)
* ridge-stabilised (`λ I`), which is LM
* allowed to fall back to a least-squares solve when the active block is still rank-deficient

The plan's original "Cholesky-only" instruction is amended for the reason recorded in
`EXPERIMENT_LOG.md` S8. Refinement is optional, off by default, and must not be able to move
an inactive coordinate: that would undo S9.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from . import jacobian as JA
from . import lie
from . import oracle as OR
from .kssf import KSSF

DOF = JA.DOF


def active_tangent_mask(active_groups: np.ndarray) -> torch.Tensor:
    """Boolean `(51,)` mask of tangent coordinates belonging to active groups."""
    keep = torch.zeros(DOF, dtype=torch.bool)
    for gi, (_n, sl) in enumerate(OR.GROUPS):
        if bool(active_groups[gi]):
            keep[sl] = True
    return keep


def lm_solve(J: torch.Tensor, r: torch.Tensor, damp: float,
             keep: torch.Tensor) -> torch.Tensor:
    """Solve `(J_k^T J_k + λ I) dx_k = -J_k^T r` on the kept coordinates, zeros elsewhere.

    `J` is `(N, 51)`, `r` is `(N,)`. Uses a least-squares solve rather than Cholesky so a
    rank-deficient active block -- a hidden finger, a single event -- is a small step, not a
    crash.
    """
    dx = J.new_zeros(DOF)
    idx = torch.where(keep)[0]
    if idx.numel() == 0 or J.shape[0] == 0:
        return dx
    Jk = J[:, idx].double()
    H = Jk.T @ Jk
    H = H + float(damp) * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
    rhs = -(Jk.T @ r.double())
    step = torch.linalg.lstsq(H, rhs.unsqueeze(-1)).solution.squeeze(-1)
    dx[idx] = step.to(J.dtype)
    return dx


@dataclass
class GNLMRefiner:
    """A few LM iterations at the previous state, using KSSF residuals."""

    mano: object
    field: Optional[KSSF] = None
    n_iters: int = 3
    damp: float = 1e-2
    step_scale: float = 1.0
    render_scale: float = 0.375

    def __post_init__(self):
        if self.field is None:
            self.field = KSSF(self.mano, 180, 240, self.render_scale)

    @torch.no_grad()
    def refine(self, x51: torch.Tensor, events: np.ndarray, betas: torch.Tensor,
               camera_K: torch.Tensor, active_groups: Optional[np.ndarray] = None
               ) -> Tuple[torch.Tensor, dict]:
        """Return a refined 51D state and a small diagnostic dict."""
        if active_groups is None:
            active_groups = np.ones(OR.N_GROUPS, dtype=bool)
        keep = active_tangent_mask(active_groups).to(x51.device)
        if events is None or len(events) == 0:
            return x51, {"n_iters": 0, "n_residuals": 0, "step_norm": 0.0}

        hm = self.mano.hands_mean.to(x51.dtype)
        j0 = (self.mano.J_regressor @ (
            self.mano.v_template + torch.einsum("bl,mkl->bmk", betas, self.mano.shapedirs)
        ))[:, 0]
        x = x51.reshape(1, DOF).to(x51.dtype)
        xy = torch.as_tensor(np.ascontiguousarray(events[:, :2]),
                             device=x.device, dtype=x.dtype)
        b = torch.zeros(xy.shape[0], dtype=torch.long, device=x.device)
        last_norm = 0.0
        used = 0
        for _ in range(self.n_iters):
            state = lie.state_from_51d(x, hm, j0)
            fk = JA.forward_kinematics(self.mano, state, betas)
            vi = torch.arange(778, device=x.device)
            bi = torch.zeros(778, dtype=torch.long, device=x.device)
            verts = JA.skin(fk, self.mano.weights[vi], bi, vi).reshape(1, 778, 3)
            fields = self.field.rasterize(verts, camera_K)
            q = fields.query(xy, b)
            vis = q["visibility"] > 0.5
            nrm = q["sdf_normal"].norm(dim=-1) > 0.5
            m = vis & nrm
            if int(m.sum()) < 8:
                break
            pix = JA.query_jacobian(self.mano, fk, camera_K, self.render_scale,
                                    b[m], q["face_id"][m], q["bary"][m],
                                    normal=q["sdf_normal"][m], pixel=xy[m])
            if pix.J_r is None:
                break
            dx = lm_solve(pix.J_r, pix.r, self.damp, keep)
            last_norm = float(dx.norm())
            x = lie.retract_51d(x, self.step_scale * dx.view(1, DOF), hm, j0)
            used += 1
        return x, {"n_iters": used, "n_residuals": int(len(events)), "step_norm": last_norm}
