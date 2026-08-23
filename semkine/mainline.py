#!/usr/bin/env python3
r"""The SemKine mainline: routing, filtering and anchoring as one estimator.

The three pieces built in S9, S12 and S13 are not three post-processing steps applied in sequence.
They are one recursive estimator, and this module is where that claim becomes code.

The joint that makes it one thing is the packet's Fisher information `Lambda` (S8), computed at the
previous state through the kinematic semantic field (S6). It is used three times, and this is the
only place it is computed:

* **Routing** (S9). Which degrees of freedom this packet is allowed to move, by greedy conditional
  log-determinant gain after eliminating the root as a nuisance.
* **Weighting** (S12). How far to move them, through `Sigma = (Lambda/s + lambda I)^-1` as the
  measurement covariance of the network's prediction.
* **Triggering** (S13). When the tracker has drifted far enough to spend an absolute prediction,
  read off the filter's own posterior and innovation statistics.

Routing is implemented as *zero measurement information* on the unrouted coordinates rather than as
a hard overwrite of them. The two are the same decision expressed differently, but the filter form
is the more defensible one and is strictly more informative: a coordinate the packet says nothing
about keeps its prior mean and its prior covariance *grows* by the process noise, which is the
correct statement of "I did not observe this". A hard skip -- the S7 oracle's mechanism, which was
right for an upper-bound experiment -- freezes the value and leaves the confidence untouched, which
is the same error the `ZERO_EVENT_GATE` makes globally.

The network is used only as a measurement. It is not retrained, wrapped or distilled here: every
arm in `tools/run_semkine.py` shares one checkpoint and one event stream, so every comparison is
paired at inference time and the 0.4 mm replicate floor does not apply to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from . import anchor as AN
from . import filter as FT
from . import jacobian as JA
from . import lie
from . import oracle as OR
from . import router as RD

DOF = JA.DOF


@dataclass
class SemKinePolicy:
    """Drop-in `active_policy` for `semkine.eval_track.track_sequence`.

    Returns the state to feed back, which is the filter's posterior rather than the network's raw
    output. With `use_router`, `use_filter` and `anchor_model` all off it is the identity, and that
    parity is the first gate any run of this must pass.
    """

    mano: object
    step_ms: int = 50
    use_filter: bool = True
    use_router: bool = True
    #: a frozen absolute predictor `f(events_window) -> (1, 51)`; `None` disables anchoring
    anchor_model: Optional[object] = None
    router_kwargs: dict = field(default_factory=dict)
    filter_kwargs: dict = field(default_factory=dict)
    trigger_kwargs: dict = field(default_factory=dict)
    #: covariance attributed to the anchor's own prediction, rad^2 per coordinate
    anchor_sigma: float = 2.5e-3

    _router: Optional[RD.RDORPolicy] = field(default=None, repr=False)
    _filter: Optional[FT.LieFilter] = field(default=None, repr=False)
    _trigger: Optional[AN.AnchorTrigger] = field(default=None, repr=False)
    _started: bool = False
    _j0: Optional[torch.Tensor] = field(default=None, repr=False)
    _j0_64: Optional[torch.Tensor] = field(default=None, repr=False)
    _betas: Optional[torch.Tensor] = field(default=None, repr=False)
    _K: Optional[torch.Tensor] = field(default=None, repr=False)
    n_steps: int = 0
    n_updated: int = 0
    n_possible: int = 0
    n_anchored: int = 0
    outside_log: List[float] = field(default_factory=list)

    # ------------------------------------------------------------------ lifecycle
    def begin_sequence(self, betas: torch.Tensor, camera_K: torch.Tensor) -> None:
        self._betas, self._K = betas, camera_K
        v = self.mano.v_template + torch.einsum("bl,mkl->bmk", betas, self.mano.shapedirs)
        # Two copies on purpose. The MANO layer and the checkpoint run in float32, so anything
        # touching them has to match; the filter and the fusion run in float64, because a 102x102
        # covariance propagated over thousands of steps loses its small eigenvalues in float32 and
        # those are exactly the directions the router and the trigger read.
        self._j0 = (self.mano.J_regressor @ v)[:, 0]
        self._j0_64 = self._j0.double()
        self._router = RD.RDORPolicy(mano=self.mano, **self.router_kwargs)
        self._router.begin_sequence(betas, camera_K)
        self._filter = FT.LieFilter(**self.filter_kwargs)
        self._trigger = AN.AnchorTrigger(**self.trigger_kwargs)
        self._started = False

    def reset(self) -> None:
        """Called at the start of each valid run: the state is re-initialised, the counters are not.

        Deliberate. A filter carried across a run boundary would be conditioning on a different
        segment of the recording, which the protocol forbids -- each valid run gets exactly one
        ground-truth initialisation and nothing else.
        """
        self._started = False
        if self._trigger is not None:
            self._trigger.reset()
        if self._router is not None:
            self._router.reset()

    def rate(self) -> float:
        return self.n_updated / max(self.n_possible, 1)

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def __call__(self, prev: torch.Tensor, pred: torch.Tensor, gt_prev, gt_cur,
                 n_events: int = 0, events: Optional[np.ndarray] = None,
                 lnes: Optional[torch.Tensor] = None, **kw) -> torch.Tensor:
        self.n_steps += 1
        self.n_possible += OR.N_GROUPS
        hm = self.mano.hands_mean
        dt = self.step_ms / 1000.0

        if not self.use_filter and not self.use_router:
            self.n_updated += OR.N_GROUPS
            return pred

        if not self._started:
            # The run's single ground-truth initialisation, handed to the filter as its prior mean.
            self._filter.start(prev.to(torch.float64), hm.to(torch.float64), self._j0_64)
            self._started = True

        Lam = None
        if events is not None and len(events):
            Lam = self._router.information(prev, events)

        active = np.ones(OR.N_GROUPS, dtype=bool)
        if self.use_router and Lam is not None:
            active = self._route(Lam)
        self.n_updated += int(active.sum())

        if not self.use_filter:
            return OR.apply_active(prev, pred, active)

        self._filter.predict(dt)
        Lam_eff = self._mask_information(Lam, active)
        self._filter.update(pred.to(torch.float64), Lam_eff, dt)

        if self.anchor_model is not None:
            self._maybe_anchor(events, lnes)
        return self._filter.x.to(pred.dtype)

    # ------------------------------------------------------------------ pieces
    def _route(self, Lam: torch.Tensor) -> np.ndarray:
        r = self._router
        cond = (JA.schur_eliminate(Lam[None], slice(6, 51), RD.ROOT)[0] if r.eliminate_root
                else Lam[6:, 6:])
        sel, gains = RD.greedy_logdet_select(cond, r.budget if r.budget is not None else 15,
                                             gain_floor=r.gain_off)
        sel = RD.close_kinematic(sel)
        active = np.zeros(OR.N_GROUPS, dtype=bool)
        active[0] = True
        for g in range(15):
            on = r._prev_active[1 + g]
            if sel[g] and (gains[g] >= r.gain_on or (on and gains[g] >= r.gain_off)):
                active[1 + g] = True
                r._dwell[1 + g] = r.min_dwell
            elif r._dwell[1 + g] > 0:
                active[1 + g] = True
                r._dwell[1 + g] -= 1
        r._prev_active = active.copy()
        return active

    def _mask_information(self, Lam: Optional[torch.Tensor],
                          active: np.ndarray) -> Optional[torch.Tensor]:
        r"""Zero the information of the unrouted coordinates.

        Rows *and* columns, so the result stays symmetric positive semidefinite and the coordinate
        is genuinely uninformed rather than merely uncorrected: leaving the cross terms would let a
        routed coordinate's update leak into a skipped one through the gain.
        """
        if Lam is None or bool(active.all()):
            return Lam
        keep = torch.ones(DOF, dtype=torch.bool, device=Lam.device)
        for gi, (_n, sl) in enumerate(OR.GROUPS):
            if not active[gi]:
                keep[sl] = False
        M = Lam.clone()
        M[~keep, :] = 0.0
        M[:, ~keep] = 0.0
        return M

    def _maybe_anchor(self, events, lnes=None) -> None:
        f = self._filter
        nis = f.nis[-1] if f.nis else None
        trace = f.trace[-1] if f.trace else None
        outside = None
        if events is not None and len(events) and self._router._field is not None:
            outside = self._outside_fraction(events)
            self.outside_log.append(outside)
        if not self._trigger(nis=nis, trace=trace, outside=outside):
            return
        # The anchor is handed the same window the tracker saw, in the tracker's own input format,
        # so its prediction is a genuinely independent read of identical evidence rather than a
        # differently preprocessed one.
        z = self.anchor_model(lnes if lnes is not None else events)
        if z is None:
            return
        S = torch.eye(DOF, device=f.x.device, dtype=torch.float64) * self.anchor_sigma
        x, P = AN.fuse(f.x, f.P[:DOF, :DOF], z.to(torch.float64), S,
                       self.mano.hands_mean.to(torch.float64), self._j0_64)
        f.x = x
        f.P[:DOF, :DOF] = P
        self.n_anchored += 1

    def _outside_fraction(self, events) -> float:
        r = self._router
        x32 = self._filter.x.to(self._j0.dtype)
        state = lie.state_from_51d(x32, self.mano.hands_mean, self._j0)
        fk = JA.forward_kinematics(self.mano, state, self._betas)
        vi = torch.arange(778, device=x32.device)
        bi = torch.zeros(778, dtype=torch.long, device=x32.device)
        verts = JA.skin(fk, self.mano.weights[vi], bi, vi).reshape(1, 778, 3)
        fields = r._field.rasterize(verts, self._K)
        xy = torch.as_tensor(np.ascontiguousarray(events[:, :2]),
                             device=x32.device).to(x32.dtype)
        b = torch.zeros(xy.shape[0], dtype=torch.long, device=xy.device)
        return AN.outside_silhouette_fraction(fields, xy, b)

    # ------------------------------------------------------------------ reporting
    def stats(self) -> Dict[str, float]:
        out = {"update_rate": self.rate(), "n_steps": self.n_steps,
               "n_anchored": self.n_anchored}
        if self._filter is not None:
            out.update({f"filter_{k}": v for k, v in self._filter.stats().items()})
        if self._trigger is not None:
            out.update({f"trigger_{k}": v for k, v in self._trigger.stats().items()
                        if not isinstance(v, dict)})
        if self.outside_log:
            out["outside_median"] = float(np.median(self.outside_log))
        return out
