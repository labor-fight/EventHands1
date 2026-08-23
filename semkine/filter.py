#!/usr/bin/env python3
r"""S12 continuous-time filtering of the hand state on its own manifold.

The tracker as it stands is a memoryless map: it sees the previous pose and one packet of events and
emits a new pose. Nothing in it represents how certain that pose is, so nothing can trade the
measurement off against the prior -- and with events, the strength of the measurement varies by
orders of magnitude from packet to packet. A filter is the principled place to put that trade-off,
and on a manifold it has to be built rather than assumed: averaging two 51-dimensional axis-angle
vectors is not averaging two rotations.

### State and process model

The state is the S3 `LieState` -- root in `SE(3)`, fifteen joints in `SO(3)` -- augmented with a
body-frame velocity `v` in the 51-dimensional tangent space. Between measurements,

    X^-_k = X^+_{k-1} ⊕ (Δt · v^+_{k-1}),      v^-_k = e^{-γΔt} v^+_{k-1},

where `⊕` is the right retraction of S3 (`R Exp(δφ)`), so the velocity lives in the body frame and
the model is coordinate-free. Damping `γ` encodes that a hand at rest stays at rest: without it the
prediction extrapolates the last motion forever, and a hand that stops moving stops producing
events, so there is nothing to correct the extrapolation with.

The error-state covariance propagates with

    Φ = [[ J_r(Δt v)^-1 · Ad,  Δt · J ],  [ 0,  e^{-γΔt} I ]],     P^- = Φ P^+ Φ^T + Q(Δt),

and `Q` from a white-noise-acceleration model integrated over the interval,

    Q(Δt) = σ_a² [[ Δt³/3 · I,  Δt²/2 · I ], [ Δt²/2 · I,  Δt · I ]].

The cross term is not decoration: with a diagonal `Q` the filter would believe position and
velocity errors are independent, which they are not when the position error is produced by
integrating the velocity error, and the result is an overconfident prediction.

### Measurement

The network's output is treated as a noisy measurement of the pose, expressed in the tangent space
at the predicted state:

    z_k = Log( (X^-_k)^{-1} ⊗ X^{net}_k ),    H = [I  0],    ν = z_k,

which is the error-state formulation (Barrau & Bonnabel, IEEE TAC 2017): the residual is a tangent
vector, the update is a retraction, and nothing is ever added in coordinates.

Its covariance is where the event structure enters. Rather than a learned uncertainty head -- which
would need retraining and, per the repository's own S9-era finding, tends to go lazy next to a
direct regression path -- `Σ` is built from the Fisher information of the packet that produced the
measurement (S8):

    Σ = ( Λ_k / s + λ I )^{-1},

so a packet that says little about a joint yields a large variance on that joint and the filter
leans on its prior. This closes the loop the plan asks for: the same information matrix that routes
the update in S9 sets the weight the update gets in S12, instead of the two being independent
heuristics that happen to be applied in sequence.

With no events there is no measurement: the state propagates and `P` grows. That is the principled
form of the existing `ZERO_EVENT_GATE`, which returns the previous pose bitwise and leaves the
tracker's confidence unchanged -- fine as a special case, wrong as a general rule, because a hand
that has been unobserved for a second is not as well localised as one observed a millisecond ago.

The update is applied in Joseph form,

    P^+ = (I - KH) P^- (I - KH)^T + K Σ K^T,

which stays symmetric positive semidefinite under floating-point arithmetic where the algebraically
equivalent `(I - KH)P^-` does not. Over a thousand-step sequence the short form loses symmetry and
then positive definiteness, and a filter with an indefinite covariance produces gains with the wrong
sign rather than an error.

### What a PASS requires

Calibration, not accuracy: the covariance has to mean something. The pre-registered gates are a
rank correlation of at least 0.30 between the predicted standard deviation and the realised error,
normalised innovation squared coverage inside 90-98%, and a reduction in long-horizon drift.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from . import jacobian as JA
from . import lie

DOF = JA.DOF                # 51: 6 root + 45 joints
NSTATE = 2 * DOF            # pose error and velocity


def error_transport(delta: torch.Tensor) -> torch.Tensor:
    r"""How a pose error at the old state appears at the new one: `Ad(Exp(-u))`, block-diagonal.

    Derivation, in the right-multiplicative convention S3 fixes. An error `e` means the true state
    is `X Exp(e)`. Both true and nominal states are propagated by the same body-frame increment
    `u = dt * v`, so the propagated true state is `X Exp(e) Exp(u)` and the propagated nominal is
    `X Exp(u)`. The new error `e'` is defined by `X Exp(u) Exp(e') = X Exp(e) Exp(u)`, hence

        Exp(e') = Exp(-u) Exp(e) Exp(u)   =>   e' = Ad(Exp(-u)) e .

    This is the adjoint, *not* the inverse right Jacobian. The two are easy to confuse -- both
    appear in error-state derivations, both reduce to the identity for small `u` -- and the
    difference is first order in `u`, so a filter using the wrong one is wrong at exactly the scale
    where the correction matters. `SO(3)`'s adjoint is the rotation matrix itself; the root's is the
    `SE(3)` adjoint, which is where the lever arm between wrist rotation and translation enters.
    """
    d = delta.reshape(-1)
    dev, dt = d.device, d.dtype
    out = torch.zeros(DOF, DOF, device=dev, dtype=dt)
    R_r, p_r = lie.se3_exp(-d[0:6].view(1, 6))
    out[0:6, 0:6] = lie.se3_adjoint(R_r, p_r)[0]
    for k in range(15):
        s = 6 + 3 * k
        out[s:s + 3, s:s + 3] = lie.so3_exp(-d[s:s + 3].view(1, 3))[0]
    return out


def velocity_coupling(delta: torch.Tensor, dt: float) -> torch.Tensor:
    r"""How a velocity error enters the pose error: `dt * J_r(u)`, block-diagonal.

    A perturbed velocity `v + dv` retracts by `u + dt*dv`, and
    `Exp(u + dt*dv) = Exp(u) Exp(J_r(u) dt dv)` to first order, so the pose error picks up
    `dt * J_r(u)`. The right Jacobian rather than the identity: at a 50 ms step and a fast finger
    the increment is a fifth of a radian, where `J_r` differs from `I` by percent-level terms that
    the covariance would otherwise attribute to the wrong direction.
    """
    d = delta.reshape(-1)
    dev, dtp = d.device, d.dtype
    out = torch.zeros(DOF, DOF, device=dev, dtype=dtp)
    out[0:6, 0:6] = lie.se3_Jr(d[0:6].view(1, 6))[0]
    for k in range(15):
        s = 6 + 3 * k
        out[s:s + 3, s:s + 3] = lie.so3_Jr(d[s:s + 3].view(1, 3))[0]
    return dt * out


def process_matrices(dt: float, gamma: float, sigma_a: float, dof: int = DOF,
                     device=None, dtype=torch.float64,
                     delta: Optional[torch.Tensor] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""`(Phi, Q)` for one interval of length `dt`, given the increment `delta = dt * v` taken.

    With `delta = None` both manifold corrections collapse to the identity, which is the
    small-angle approximation: adequate at 1 ms, wrong at 50.
    """
    I = torch.eye(dof, device=device, dtype=dtype)
    A = I if delta is None else error_transport(delta)
    B = dt * I if delta is None else velocity_coupling(delta, dt)
    Phi = torch.zeros(NSTATE, NSTATE, device=device, dtype=dtype)
    Phi[:dof, :dof] = A
    Phi[:dof, dof:] = B
    Phi[dof:, dof:] = float(np.exp(-gamma * dt)) * I
    s2 = sigma_a ** 2
    Q = torch.zeros(NSTATE, NSTATE, device=device, dtype=dtype)
    Q[:dof, :dof] = s2 * dt ** 3 / 3.0 * I
    Q[:dof, dof:] = s2 * dt ** 2 / 2.0 * I
    Q[dof:, :dof] = s2 * dt ** 2 / 2.0 * I
    Q[dof:, dof:] = s2 * dt * I
    return Phi, Q


def sigma_from_information(Lam: Optional[torch.Tensor], divisor: float, floor: float,
                           ceil: float, device=None, dtype=torch.float64) -> torch.Tensor:
    r"""Measurement covariance from the packet's Fisher information, `(Lam/scale + lam I)^-1`.

    The floor is not a numerical guard, it is a statement: even a packet dense with events does not
    determine the pose better than the network that reads it, so the measurement variance cannot go
    below the regressor's own error. The ceiling caps how uninformative a near-empty packet is
    allowed to look, which keeps the gain from underflowing to zero and freezing the state.
    """
    I = torch.eye(DOF, device=device, dtype=dtype)
    if Lam is None:
        return ceil * I
    M = Lam.to(dtype).to(device) / max(divisor, 1e-12) + (1.0 / ceil) * I
    S = torch.linalg.inv(0.5 * (M + M.T))
    S = 0.5 * (S + S.T)
    # Clamp the spectrum rather than the entries: clamping entries of a covariance can leave it
    # indefinite, and an indefinite `Sigma` gives a Kalman gain with no meaning.
    w, V = torch.linalg.eigh(S)
    return V @ torch.diag(w.clamp(floor, ceil)) @ V.T


@dataclass
class LieFilter:
    r"""Error-state filter on `SE(3) x SO(3)^15` with a velocity augmentation."""

    #: velocity damping, s^-1. A hand's motion decorrelates in well under a second.
    gamma: float = 8.0
    #: process acceleration noise, rad/s^2 (and m/s^2 on the root translation block)
    sigma_a: float = 60.0
    #: initial pose and velocity standard deviations
    p0_pose: float = 0.05
    p0_vel: float = 1.0
    #: Converts the packet's Fisher information, whose units are inverse squared pixels times the
    #: squared projection Jacobian, into a pose measurement precision. A *divisor*: raising it makes
    #: every measurement looser and the filter more reliant on its prior.
    info_divisor: float = 2.0e4
    #: spectral floor and ceiling of the measurement covariance, rad^2
    sigma_floor: float = 1.0e-4
    sigma_ceil: float = 1.0e0
    exact_retraction_jacobian: bool = True
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float64

    x: Optional[torch.Tensor] = field(default=None, repr=False)     # (51,) pose, 51D convention
    v: Optional[torch.Tensor] = field(default=None, repr=False)     # (51,) tangent velocity
    P: Optional[torch.Tensor] = field(default=None, repr=False)     # (102, 102)
    hands_mean: Optional[torch.Tensor] = field(default=None, repr=False)
    j0: Optional[torch.Tensor] = field(default=None, repr=False)
    nis: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    n_measured: int = 0
    n_propagated: int = 0

    # ------------------------------------------------------------------ lifecycle
    def start(self, x0: torch.Tensor, hands_mean: torch.Tensor, j0: torch.Tensor) -> None:
        self.device = x0.device
        self.hands_mean, self.j0 = hands_mean, j0
        self.x = x0.reshape(1, DOF).to(self.dtype)
        self.v = torch.zeros(DOF, device=self.device, dtype=self.dtype)
        P = torch.zeros(NSTATE, NSTATE, device=self.device, dtype=self.dtype)
        P[:DOF, :DOF] = self.p0_pose ** 2 * torch.eye(DOF, device=self.device, dtype=self.dtype)
        P[DOF:, DOF:] = self.p0_vel ** 2 * torch.eye(DOF, device=self.device, dtype=self.dtype)
        self.P = P
        self.nis, self.trace = [], []
        self.n_measured = self.n_propagated = 0

    # ------------------------------------------------------------------ prediction
    def predict(self, dt: float) -> torch.Tensor:
        delta = dt * self.v
        Phi, Q = process_matrices(dt, self.gamma, self.sigma_a, DOF, self.device, self.dtype,
                                  delta if self.exact_retraction_jacobian else None)
        self.x = lie.retract_51d(self.x, delta.view(1, DOF), self.hands_mean, self.j0)
        self.v = float(np.exp(-self.gamma * dt)) * self.v
        self.P = Phi @ self.P @ Phi.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        self.n_propagated += 1
        return self.x

    # ------------------------------------------------------------------ correction
    def update(self, z51: torch.Tensor, Lam: Optional[torch.Tensor] = None,
               dt: float = 1.0) -> torch.Tensor:
        """Fuse a network prediction `z51` measured with information `Lam`."""
        if z51 is None:
            return self.x
        Sigma = sigma_from_information(Lam, self.info_divisor, self.sigma_floor,
                                       self.sigma_ceil, self.device, self.dtype)
        # Innovation in the tangent space at the predicted state: `Log(x^-1 z)` under the same
        # right-multiplicative convention S3 fixes, so it composes with the retraction below.
        nu = lie.local_coordinates_51d(self.x, z51.reshape(1, DOF).to(self.dtype),
                                       self.hands_mean, self.j0).reshape(DOF)
        Ppp = self.P[:DOF, :DOF]
        S = Ppp + Sigma
        S = 0.5 * (S + S.T)
        L = torch.linalg.cholesky(S)
        # `H = [I 0]`, so `P H^T` is the first block column; written out rather than materialising
        # `H`, which would be a 51x102 matrix of ones and zeros.
        PHt = self.P[:, :DOF]
        K = torch.cholesky_solve(PHt.T, L).T                        # (102, 51)
        d = K @ nu
        self.x = lie.retract_51d(self.x, d[:DOF].view(1, DOF), self.hands_mean, self.j0)
        self.v = self.v + d[DOF:]

        IKH = torch.eye(NSTATE, device=self.device, dtype=self.dtype)
        IKH[:, :DOF] -= K
        self.P = IKH @ self.P @ IKH.T + K @ Sigma @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # Normalised innovation squared, the standard consistency statistic: its expectation is the
        # innovation dimension, so a filter claiming more precision than it has shows up here long
        # before it shows up in the error.
        y = torch.cholesky_solve(nu.view(-1, 1), L)
        self.nis.append(float((nu.view(1, -1) @ y)))
        self.trace.append(float(torch.diagonal(self.P[:DOF, :DOF]).sum()))
        self.n_measured += 1
        return self.x

    # ------------------------------------------------------------------ reporting
    def pose_sigma(self) -> torch.Tensor:
        """Per-coordinate predicted standard deviation, for the calibration gate."""
        return torch.diagonal(self.P[:DOF, :DOF]).clamp_min(0).sqrt()

    def stats(self) -> Dict[str, float]:
        n = np.array(self.nis, dtype=np.float64)
        out = {"n_measured": self.n_measured, "n_propagated": self.n_propagated,
               "trace_mean": float(np.mean(self.trace)) if self.trace else 0.0}
        if n.size:
            # Coverage against the chi-square quantiles at the innovation dimension: the fraction
            # of steps whose NIS falls inside the 95% band, which is the quantity the gate is on.
            from scipy.stats import chi2
            lo, hi = chi2.ppf(0.025, DOF), chi2.ppf(0.975, DOF)
            out.update(nis_mean=float(n.mean()), nis_dim=DOF,
                       nis_coverage=float(((n >= lo) & (n <= hi)).mean()))
        return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, computed here so the module does not depend on scipy for the main gate."""
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")
