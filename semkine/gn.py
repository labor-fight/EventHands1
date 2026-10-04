#!/usr/bin/env python3
"""S14 active-set Levenberg-Marquardt solve with inactive coordinates held fixed."""
from __future__ import annotations


import numpy as np
import torch

from . import jacobian as JA
from . import oracle as OR

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
