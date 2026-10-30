"""R1 event-weighted support in the fixed MANO joint-coordinate basis.

The single summed Gram C and its per-slot diagonal are empirical support
features.  They are neither precision nor a calibrated posterior covariance.
No unweighted contour-point geometry token enters this readout.
"""
from __future__ import annotations

import torch
from .model import TypedSupportAdapter


def joint_gram_features(a, lam):
    """Return C[B,51,51] and diagG[B,51,H], without H dense Grams.

Cost is O(B*N*(51^2+51*H)); the retained outputs use 51^2+51*H
numbers per batch row, in addition to the input factors.
    """
    if a.ndim != 3 or a.shape[-1] != 51 or lam.ndim != 3 or lam.shape[:2] != a.shape[:2]:
        raise ValueError("expected a[B,N,51] and lambda[B,N,H]")
    if not bool(torch.isfinite(a).all() and torch.isfinite(lam).all()) or bool(torch.any(lam < 0)):
        raise ValueError("finite a and nonnegative finite lambda required")
    # BF16 accumulation would unnecessarily perturb the support basis.  Keep
    # the statistic in float32, or float64 for explicit numerical audits.
    dtype = torch.float64 if a.dtype == torch.float64 or lam.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=a.device.type, enabled=False):
        aa, ll = a.to(dtype), lam.to(dtype)
        diagG = torch.bmm(aa.square().transpose(1, 2), ll)
        C = torch.bmm(aa.transpose(1, 2), aa * ll.sum(-1, keepdim=True))
    if not bool(torch.isfinite(C).all() and torch.isfinite(diagG).all()):
        raise ArithmeticError("nonfinite joint support Gram")
    return C, diagG


class JointGramSupportAdapter(TypedSupportAdapter):
    """Same parameterization and count as the R0 point-token adapter.

Only the support token's meaning changes: [signed-log C_jk, log diagG_jh]
in fixed joint coordinates replaces [unweighted a_lj, log lambda_lh].
    """

    reader = "joint_gram"

    def forward(self, f, a, lam, conditions):
        if f.ndim != 3 or f.shape[1:] != (51, self.h):
            raise ValueError(f"expected f[B,51,{self.h}]")
        if a.ndim != 3 or a.shape[0] != len(f) or a.shape[2] != 51:
            raise ValueError("expected a[B,N,51]")
        if lam.shape != (len(f), a.shape[1], self.h):
            raise ValueError("expected lambda[B,N,H]")
        if conditions.shape != (len(f), 3) or not bool(torch.isfinite(f).all() and torch.isfinite(conditions).all()):
            raise ValueError("finite direction/condition inputs with shape[B,3] required")
        C, diagG = joint_gram_features(a, lam)
        features = torch.cat((torch.sign(C) * torch.log1p(C.abs()), torch.log1p(diagG)), -1)
        rows = (diagG.sum(-1) > 0).to(features.dtype)
        tokens = self.support(features.to(f.dtype))
        pooled = (tokens * rows[..., None]).sum(1) / rows.sum(1, keepdim=True).clamp_min(1)
        ff = torch.sign(f) * torch.log1p(f.abs())
        features = torch.cat((self.direction(ff.flatten(1)), pooled, conditions), -1)
        live = (rows.sum(1, keepdim=True) > 0).to(features.dtype)
        return self.output(features) * live
