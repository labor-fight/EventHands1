#!/usr/bin/env python3
r"""S13 triggered absolute anchoring.

A recursive tracker has no fixed point of reference. Every step is a correction to the previous
estimate, so whatever error survives a step is carried into the next one, and over tens of seconds
the state walks away from the truth in a direction nothing in the loop can see -- the residual keeps
looking small because it is measured against the drifted state. The repository's own history shows
this directly: a zero-training constant blend of the recursive tracker with a frozen absolute
predictor moved 19.26 mm to 19.32 mm, close to neutral overall, which for a blend applied
indiscriminately at every step is the expected outcome. Applied indiscriminately, an absolute
predictor costs as much on the frames where the tracker is right as it gains where it has drifted.

So the question is not whether to fuse but *when*, and with what weight.

### When: three signals, all internal

The trigger never looks at ground truth. It fires on

1. **Normalised innovation squared.** The filter's own consistency statistic. Sustained NIS above
   the chi-square band means the measurements keep disagreeing with the prediction by more than the
   claimed uncertainty allows, which is what drift looks like from inside the filter.
2. **Posterior log-determinant.** The filter's admitted uncertainty. After a stretch of few events
   the covariance has grown and the state is cheap to correct; this is the "free" trigger.
3. **Events outside the silhouette.** A geometric check the filter cannot fake: if the state were
   right, edges would appear at the rendered contour. A large fraction of events landing far
   outside the projected hand means the hypothesis is in the wrong place, and unlike the first two
   this signal does not depend on the filter's own noise model being correct.

Any one crossing its threshold arms the anchor; hysteresis and a cooldown keep it from firing every
step, which would reduce it to the constant blend that history already found neutral.

### With what weight: tangent-space Gaussian fusion

Given the tracker's `(x_trk, P)` and the anchor's `(x_abs, Sigma)`, the fused estimate is the
maximum-likelihood point of the two Gaussians written on the manifold:

    x^+ = argmin_x  ||Log(x_trk^-1 x)||^2_{P^-1} + ||Log(x_abs^-1 x)||^2_{Sigma^-1} .

Linearising at `x_trk`, with `d = Log(x_trk^-1 x_abs)` and the right Jacobian `J_r` accounting for
the second residual being measured at a different point,

    delta = (P^-1 + J^-T Sigma^-1 J^-1)^-1 J^-T Sigma^-1 d,     x^+ = x_trk Exp(delta),

iterated to convergence, and the fused covariance is the inverse of the summed information. The
Euclidean shortcut -- averaging the two 51-dimensional vectors -- is wrong for the same reason it is
always wrong on rotations, and here it is wrong in the worst place: the anchor fires precisely when
the two estimates are far apart, which is where the difference between the manifold mean and the
coordinate mean is largest.

### What the anchor is

Per the plan's correction 7, the first version reuses the frozen absolute LNES predictor. It is
dense and expensive, but it is event-only, requires no training, already exists, and runs at a low
duty cycle by construction because it only runs when triggered. A long-context raw-event anchor is
an upgrade, not a prerequisite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from . import jacobian as JA
from . import lie

DOF = JA.DOF


# --------------------------------------------------------------------------- fusion
#: Metric on the root for fusion. `"decoupled"` treats the wrist as `R^3 x SO(3)`, `"se3"` uses the
#: full group. See :func:`fuse` for why the default is the former.
ROOT_METRIC = "decoupled"


def _block_Jr_inv(d: torch.Tensor, root_metric: str = ROOT_METRIC) -> torch.Tensor:
    """Block-diagonal `J_r(d)^-1` on the 51-dimensional tangent space."""
    d = d.reshape(-1)
    out = torch.eye(DOF, device=d.device, dtype=d.dtype)
    if root_metric == "se3":
        out[0:6, 0:6] = torch.linalg.inv(lie.se3_Jr(d[0:6].view(1, 6))[0])
    else:
        # Translation is Euclidean, so its block is the identity; only the wrist rotation curves.
        out[3:6, 3:6] = lie.so3_Jr_inv(d[3:6].view(1, 3))[0]
    for k in range(15):
        s = 6 + 3 * k
        out[s:s + 3, s:s + 3] = lie.so3_Jr_inv(d[s:s + 3].view(1, 3))[0]
    return out


def local_metric(a: torch.Tensor, b: torch.Tensor, hands_mean, j0,
                 root_metric: str = ROOT_METRIC) -> torch.Tensor:
    """Tangent vector from `a` to `b` under the chosen root metric."""
    if root_metric == "se3":
        return lie.local_coordinates_51d(a, b, hands_mean, j0).reshape(DOF)
    d = lie.local_coordinates_51d(a, b, hands_mean, j0).reshape(DOF).clone()
    d[0:3] = (b.reshape(DOF)[0:3] - a.reshape(DOF)[0:3]).to(d.dtype)
    d[3:6] = lie.so3_log(lie.so3_exp(a.reshape(1, DOF)[:, 3:6]).transpose(-1, -2)
                         @ lie.so3_exp(b.reshape(1, DOF)[:, 3:6]))[0].to(d.dtype)
    return d


def retract_metric(x: torch.Tensor, d: torch.Tensor, hands_mean, j0,
                   root_metric: str = ROOT_METRIC) -> torch.Tensor:
    """Move `x` by tangent vector `d` under the chosen root metric; inverse of `local_metric`."""
    if root_metric == "se3":
        return lie.retract_51d(x, d.view(1, DOF), hands_mean, j0)
    out = lie.retract_51d(x, torch.cat([torch.zeros_like(d[0:6]), d[6:]]).view(1, DOF),
                          hands_mean, j0)
    out = out.clone()
    out[:, 0:3] = x.reshape(1, DOF)[:, 0:3] + d[0:3]
    R = lie.so3_exp(x.reshape(1, DOF)[:, 3:6]) @ lie.so3_exp(d[3:6].view(1, 3))
    out[:, 3:6] = lie.so3_log(R)
    return out


def fuse(x_trk: torch.Tensor, P: torch.Tensor, x_abs: torch.Tensor, Sigma: torch.Tensor,
         hands_mean: torch.Tensor, j0: torch.Tensor, iters: int = 3,
         root_metric: str = ROOT_METRIC) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Tangent-space Gaussian fusion of two pose estimates. Returns `(x_fused, P_fused)`.

    Iterated because the linearisation point moves: one step is a good approximation when the two
    estimates are close, and the anchor fires precisely when they are not.

    `root_metric` is a decision worth stating rather than defaulting into. The state lives on
    `SE(3) x SO(3)^15` and S3 fixed that convention, but the *metric* used to average two states is
    a separate choice, and on `SE(3)` the geodesic mean couples rotation into translation through
    the `Log` map: fusing two estimates that straddle the truth by 0.6 rad returns a wrist 5.9 mm
    off the true midpoint, entirely in translation, with the joints exact. That is not an error --
    it is the correct mean under that metric -- but it is the wrong mean for this problem, whose
    error measure is millimetres of vertex position and is linear in world translation. So the
    default treats the wrist as `R^3 x SO(3)`: translation averaged where it is measured, rotation
    on its manifold. `"se3"` is kept so the bias can be reproduced rather than asserted.
    """
    dt = P.dtype
    xt = x_trk.reshape(1, DOF).to(dt)
    xa = x_abs.reshape(1, DOF).to(dt)
    x = xt.clone()
    Pi = torch.linalg.inv(P)
    Si = torch.linalg.inv(Sigma)

    def normal_equations(x):
        r"""`(A, b)` of the Gauss-Newton step at `x`.

        The residual of endpoint `i` is `r_i = Log(x_i^-1 x)`, and stepping `x <- x Exp(delta)`
        gives `Log(x_i^-1 x Exp(delta)) = Log(Exp(r_i) Exp(delta)) ~ r_i + J_r(r_i)^-1 delta`.
        So the Jacobian is evaluated at `r_i`, which is the *negative* of the more convenient
        `d_i = Log(x^-1 x_i)`. Using `d_i` instead is a sign error that survives every symmetric
        test case -- the two terms cancel there either way -- and only shows up as a bias when the
        two covariances differ, which is the case the anchor exists for.
        """
        A = torch.zeros(DOF, DOF, device=x.device, dtype=dt)
        b = torch.zeros(DOF, device=x.device, dtype=dt)
        for xi, Wi in ((xt, Pi), (xa, Si)):
            d = local_metric(x, xi, hands_mean, j0, root_metric)
            J = _block_Jr_inv(-d, root_metric)
            JW = J.T @ Wi
            A = A + JW @ J
            b = b + JW @ d
        return 0.5 * (A + A.T), b

    for _ in range(iters):
        A, b = normal_equations(x)
        step = torch.linalg.solve(A, b)
        x = retract_metric(x, step, hands_mean, j0, root_metric)
        if float(step.norm()) < 1e-14:
            break
    A, _ = normal_equations(x)
    Pf = torch.linalg.inv(A)
    return x, 0.5 * (Pf + Pf.T)


def outside_silhouette_fraction(field, events_xy: torch.Tensor, batch_index: torch.Tensor,
                                margin_px: float = 4.0) -> float:
    r"""Fraction of events lying more than `margin_px` outside the projected hand.

    The one drift signal that does not go through the filter's noise model. Events are produced by
    moving edges, and under the current hypothesis those edges are at the rendered contour, so a
    large fraction landing far outside means the hypothesis is misplaced -- regardless of what the
    covariance claims. The margin absorbs the contour's own uncertainty and the sensor's spatial
    jitter; without it, every silhouette boundary event would count as an outlier.
    """
    q = field.query(events_xy, batch_index)
    return float((q["sdf"] > margin_px).to(torch.float64).mean())


# --------------------------------------------------------------------------- trigger
@dataclass
class AnchorTrigger:
    r"""Decides when to spend an absolute prediction. Hysteresis plus cooldown, no ground truth."""

    #: NIS above `nis_dim * nis_ratio_on` arms the trigger; below `..._off` disarms it
    nis_ratio_on: float = 2.0
    nis_ratio_off: float = 1.2
    nis_dim: int = DOF
    #: trace of the pose covariance above which the state is considered cheap to correct
    trace_on: float = 0.05
    #: Fraction of events outside the silhouette. Calibrated on a real frame at a 4 px margin:
    #: the correct state gives 0.00, 1 cm of translational drift still gives 0.00, 2 cm gives 0.23,
    #: 4 cm gives 0.36 and 8 cm gives 0.65. So this signal detects gross displacement, not
    #: millimetre drift, and the threshold is set where it starts to mean something rather than at
    #: a round number that would never fire.
    outside_on: float = 0.20
    outside_off: float = 0.10
    #: minimum steps between two firings, and consecutive armed steps before the first
    cooldown: int = 20
    persistence: int = 2

    _cool: int = 0
    _armed: int = 0
    n_steps: int = 0
    n_fired: int = 0
    reasons: List[str] = field(default_factory=list)

    def reset(self) -> None:
        self._cool = 0
        self._armed = 0

    def rate(self) -> float:
        return self.n_fired / max(self.n_steps, 1)

    def __call__(self, nis: Optional[float] = None, trace: Optional[float] = None,
                 outside: Optional[float] = None) -> bool:
        self.n_steps += 1
        if self._cool > 0:
            self._cool -= 1
            self._armed = 0
            return False
        why = []
        if nis is not None and nis > self.nis_ratio_on * self.nis_dim:
            why.append("nis")
        if trace is not None and trace > self.trace_on:
            why.append("trace")
        if outside is not None and outside > self.outside_on:
            why.append("outside")
        # Persistence before firing: a single bad packet is noise, and the anchor is expensive
        # enough that spending it on noise is the failure mode that made the constant blend neutral.
        if why:
            self._armed += 1
        else:
            below = ((nis is None or nis < self.nis_ratio_off * self.nis_dim)
                     and (outside is None or outside < self.outside_off))
            self._armed = 0 if below else self._armed
        if self._armed >= self.persistence:
            self._armed = 0
            self._cool = self.cooldown
            self.n_fired += 1
            self.reasons.append("+".join(why))
            return True
        return False

    def stats(self) -> Dict[str, float]:
        from collections import Counter
        return {"n_steps": self.n_steps, "n_fired": self.n_fired, "fire_rate": self.rate(),
                "reasons": dict(Counter(self.reasons))}
