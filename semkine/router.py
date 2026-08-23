#!/usr/bin/env python3
r"""S9 RDOR: root-disentangled observability routing.

The question a packet of events poses is not only "what is the new pose" but "which degrees of
freedom does this packet actually say anything about". RDOR answers it from the measurement geometry
rather than from a learned score, in three steps.

**1. Build the information.** Each event at pixel `u` is a statement about a moving edge, so it
constrains the state only along the contour normal (S6, S8):

    r_i(x) = n_i^T ( u_i - pi_K( X_i(x) ) ),   J_i = -n_i^T (d pi/d X) (d X_i/d x),
    Lambda = sum_i J_i^T R_i^-1 J_i .

Events that fall on no rasterised face carry no such statement and are dropped; under this
measurement model they are not weak evidence, they are no evidence.

**2. Eliminate the root as a nuisance.** During tracking the wrist pose is not known, and a rigid
wrist motion sweeps edges across the fingers too, so a finger's raw information block credits it
with evidence that the root could equally explain. The Schur complement

    Lambda_{a|r} = Lambda_aa - Lambda_ar ( Lambda_rr + P_r^-1 )^-1 Lambda_ra

removes exactly that part, and `P_r` lets the filter's current root covariance enter -- a
well-determined wrist subtracts less, which closes the loop between S12 and the router instead of
leaving the two independent.

**3. Select under a budget, greedily, by conditional log-determinant gain.**

    IG(S) = log det( I + Lambda_{S|r} ),    gain(g | S) = IG(S + g) - IG(S)

`IG` is monotone submodular in `S`, so greedy selection is within `1 - 1/e` of the optimum
(Shamaiah et al., SPL 2010; Krause & Guestrin). The root is always selected: it is the one block
that events constrain unconditionally, and eliminating it as a nuisance is a statement about how to
*credit* the fingers, not a decision to leave the wrist un-updated.

Two pieces of structure on top of the plain greedy step:

*Kinematic closure.* Selecting a distal joint also selects its ancestors within the same finger. A
fingertip's apparent motion is produced jointly by the chain that carries it, and updating the tip
while pinning its parent puts the error into the one joint that happens to be selected.

*Hysteresis and minimum dwell.* A group must exceed a gain threshold to switch on and fall below a
lower one to switch off, and stays on for at least `min_dwell` steps. Without it the mask chatters
at the threshold from step to step, which is visible as jitter and is the standard remedy in
switching estimators.

### What this is measured against

S7 established the reference points on the frozen checkpoint: a ground-truth oracle at matched
update rate beats a random policy by 1.38 mm, so routing carries real signal, but constant δ-trust
reaches within 0.2 mm of the oracle at a 50 ms step. RDOR is therefore not evaluated against the
unmodified baseline -- that bar is too low to mean anything -- but against δ-trust and against the
oracle, with the oracle as the ceiling it is trying to approach without looking at the answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from . import jacobian as JA
from . import lie
from . import oracle as OR
from .kssf import KSSF

N_GROUPS = OR.N_GROUPS
ROOT = slice(0, 6)


def greedy_logdet_select(Lam_cond: torch.Tensor, budget: int,
                         gain_floor: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    r"""Greedy maximisation of `log det(I + Lambda_SS)` over joint groups.

    `Lam_cond` is the `(45, 45)` joint block after root elimination. Returns the boolean selection
    over the 15 joint groups and the marginal gain each selected group contributed.

    The gain is evaluated on the *joint* block of the currently selected set, not on each group's
    own diagonal block, which is what makes it conditional: two joints that explain the same edge
    motion cannot both be credited for it.
    """
    n = 15
    sel = np.zeros(n, dtype=bool)
    gains = np.zeros(n, dtype=np.float64)
    idx: List[int] = []
    cur = 0.0
    for _ in range(min(budget, n)):
        best_g, best_gain = -1, gain_floor
        for g in range(n):
            if sel[g]:
                continue
            cand = idx + [g]
            cols = np.concatenate([np.arange(3 * c, 3 * c + 3) for c in sorted(cand)])
            t = torch.as_tensor(cols, device=Lam_cond.device)
            sub = Lam_cond[t][:, t]
            ig = float(JA.information_gain(sub))
            if not np.isfinite(ig):
                continue
            if ig - cur > best_gain:
                best_g, best_gain = g, ig - cur
        if best_g < 0:
            break
        sel[best_g] = True
        gains[best_g] = best_gain
        idx.append(best_g)
        cur += best_gain
    return sel, gains


def close_kinematic(sel: np.ndarray) -> np.ndarray:
    """Selecting a joint also selects its ancestors within the same finger.

    MANO's 15 local joints are five chains of three in order, so a joint's ancestors are the
    earlier entries of its own triple.
    """
    out = sel.copy()
    for f in range(5):
        base = 3 * f
        for d in (2, 1):
            if out[base + d]:
                out[base + d - 1] = True
    return out


@dataclass
class RDORPolicy:
    """Inference-time router with the same interface as the S7 oracle, but blind to ground truth."""

    mano: object
    camera_K: Optional[torch.Tensor] = None
    betas: Optional[torch.Tensor] = None
    height: int = 180
    width: int = 240
    render_scale: float = 0.375
    #: number of joint groups allowed on per step; `None` uses `gain_on` alone
    budget: Optional[int] = 6
    #: switch-on and switch-off thresholds on the marginal log-det gain, in nats
    gain_on: float = 1.0
    gain_off: float = 0.25
    min_dwell: int = 2
    #: measurement standard deviation in pixels, the `R_i` of the Fisher sum
    sigma_px: float = 1.0
    #: cap on events used per step; the information sum saturates long before the packet does
    max_events: int = 20000
    #: ablation switch. `False` scores each joint on its own information block, crediting it with
    #: whatever the unresolved wrist could equally explain -- the arm S9 must beat.
    eliminate_root: bool = True
    device: Optional[torch.device] = None

    _field: Optional[KSSF] = field(default=None, repr=False)
    _dwell: np.ndarray = field(default=None, repr=False)
    _prev_active: np.ndarray = field(default=None, repr=False)
    _j0: Optional[torch.Tensor] = field(default=None, repr=False)
    n_steps: int = 0
    n_updated: int = 0
    n_possible: int = 0
    n_no_evidence: int = 0
    gain_log: List[float] = field(default_factory=list)

    def __post_init__(self):
        self._dwell = np.zeros(N_GROUPS, dtype=np.int64)
        self._prev_active = np.ones(N_GROUPS, dtype=bool)

    def begin_sequence(self, betas: torch.Tensor, camera_K: torch.Tensor) -> None:
        self.betas = betas
        self.camera_K = camera_K
        self.device = betas.device
        if self._field is None:
            self._field = KSSF(self.mano, self.height, self.width,
                               self.render_scale).to(self.device)
        v = self.mano.v_template + torch.einsum("bl,mkl->bmk", betas, self.mano.shapedirs)
        self._j0 = (self.mano.J_regressor @ v)[:, 0]

    def reset(self) -> None:
        self._dwell[:] = 0
        self._prev_active[:] = True

    def rate(self) -> float:
        return self.n_updated / max(self.n_possible, 1)

    @torch.no_grad()
    def information(self, prev51: torch.Tensor, events: np.ndarray) -> Optional[torch.Tensor]:
        """`Lambda` for one packet, evaluated entirely at the previous state."""
        state = lie.state_from_51d(prev51, self.mano.hands_mean, self._j0)
        fk = JA.forward_kinematics(self.mano, state, self.betas)
        vi = torch.arange(778, device=self.device)
        bi0 = torch.zeros(778, dtype=torch.long, device=self.device)
        verts = JA.skin(fk, self.mano.weights[vi], bi0, vi).reshape(1, 778, 3)
        f = self._field.rasterize(verts, self.camera_K)

        xy = torch.as_tensor(np.ascontiguousarray(events[:, :2]),
                             device=self.device).long()
        if xy.shape[0] > self.max_events:
            step = xy.shape[0] // self.max_events + 1
            xy = xy[::step]
        b = torch.zeros(xy.shape[0], dtype=torch.long, device=self.device)
        q = f.query(xy.to(prev51.dtype), b)
        # An event on no face is not weak evidence about the pose, it is none: the measurement
        # model has nothing to say about a pixel with no surface behind it.
        ok = (q["face_id"] >= 0) & (q["sdf_normal"].norm(dim=-1) > 0.5)
        if int(ok.sum()) < 50:
            return None
        out = JA.query_jacobian(self.mano, fk, self.camera_K, self.render_scale,
                                b[ok], q["face_id"][ok], q["bary"][ok],
                                normal=q["sdf_normal"][ok],
                                pixel=xy[ok].to(prev51.dtype))
        sigma = torch.full((int(ok.sum()),), self.sigma_px,
                           device=self.device, dtype=out.J_r.dtype)
        return JA.fisher(out.J_r, sigma=sigma)

    def __call__(self, prev: torch.Tensor, pred: torch.Tensor, gt_prev: np.ndarray,
                 gt_cur: np.ndarray, n_events: int = 0,
                 events: Optional[np.ndarray] = None, **kw) -> torch.Tensor:
        self.n_steps += 1
        self.n_possible += N_GROUPS
        Lam = None if events is None or len(events) == 0 else self.information(prev, events)
        if Lam is None:
            # No usable measurement. The principled response is to update nothing and let the
            # prior stand, which is the same rule `ZERO_EVENT_GATE` applies globally.
            self.n_no_evidence += 1
            self._dwell = np.maximum(self._dwell - 1, 0)
            active = self._prev_active & (self._dwell > 0)
            self.n_updated += int(active.sum())
            return OR.apply_active(prev, pred, active)

        cond = (JA.schur_eliminate(Lam[None], slice(6, 51), ROOT)[0] if self.eliminate_root
                else Lam[6:, 6:])
        budget = self.budget if self.budget is not None else 15
        sel, gains = greedy_logdet_select(cond, budget, gain_floor=self.gain_off)
        sel = close_kinematic(sel)
        self.gain_log.extend(float(x) for x in gains[gains > 0])

        active = np.zeros(N_GROUPS, dtype=bool)
        active[0] = True                     # the root is always updated
        for g in range(15):
            on = self._prev_active[1 + g]
            gain = gains[g]
            # Asymmetric thresholds plus a dwell floor: switching on demands more evidence than
            # staying on, so the mask does not chatter at the boundary from one packet to the next.
            if sel[g] and (gain >= self.gain_on or (on and gain >= self.gain_off)):
                active[1 + g] = True
                self._dwell[1 + g] = self.min_dwell
            elif self._dwell[1 + g] > 0:
                active[1 + g] = True
                self._dwell[1 + g] -= 1
        self._prev_active = active.copy()
        self.n_updated += int(active.sum())
        return OR.apply_active(prev, pred, active)

    def stats(self) -> Dict[str, float]:
        return {
            "update_rate": self.rate(),
            "n_steps": self.n_steps,
            "no_evidence_frac": self.n_no_evidence / max(self.n_steps, 1),
            "gain_median": float(np.median(self.gain_log)) if self.gain_log else 0.0,
        }
