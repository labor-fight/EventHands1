#!/usr/bin/env python3
r"""S7 oracle upper bound on selective updating.

The whole active-estimation branch of the plan rests on one empirical claim: that it helps to
update only the degrees of freedom a packet of events actually constrains, and to leave the rest
alone. S9's router, S10's per-joint head and S12's selective filtering all inherit their
justification from it. So before building any of them, the claim is tested with an oracle that is
allowed to cheat: it reads the ground truth to decide *which* coordinates to update, but never
what to update them to. If knowing the answer to the routing question does not improve tracking,
no learned router can, and the plan's own instruction is to stop the active mainline there.

Two things this is careful about.

**The skip is real.** A retained coordinate keeps its previous value and that value is what gets
fed back on the next step. The tempting alternative -- multiplying the predicted update by a mask
-- does not test selective updating, because with `PREDICT_DELTA` the network's output already is
`prev + delta`, so masking the delta and masking the output are different operations and only one
of them is a skip.

**The controls matter more than the oracle.** An oracle that skips 70% of coordinates will look
good for a reason that has nothing to do with routing: any shrinkage toward the previous state
reduces jitter, which is why delta-trust alone was worth half a millimetre for free. So each oracle
is paired with a random policy skipping at the same rate and with the constant delta-trust arm. The
oracle only counts if it beats those.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

#: Coordinate groups of the 51D vector: the root as one 6-dimensional block, then a block per
#: MANO joint. Matches `jacobian.GROUPS`, so a routing decision means the same thing in both.
GROUPS: Tuple[Tuple[str, slice], ...] = (("root", slice(0, 6)),) + tuple(
    (f"j{k}", slice(6 + 3 * k, 9 + 3 * k)) for k in range(15))
N_GROUPS = len(GROUPS)


def _geodesic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation angle between two axis-angle vectors, `(..., 3)` -> `(...)`.

    Via the rotation matrices rather than `|a - b|`: axis-angle differences are not angles, and
    near a wrap the two disagree by a factor of several.
    """
    def R(v):
        th = np.linalg.norm(v, axis=-1, keepdims=True)
        k = v / np.clip(th, 1e-12, None)
        K = np.zeros(v.shape[:-1] + (3, 3), dtype=np.float64)
        K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
        K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
        K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
        I = np.eye(3)
        s, c = np.sin(th)[..., None], np.cos(th)[..., None]
        return I + s * K + (1 - c) * (K @ K)
    M = np.swapaxes(R(a.astype(np.float64)), -1, -2) @ R(b.astype(np.float64))
    tr = np.clip((M[..., 0, 0] + M[..., 1, 1] + M[..., 2, 2] - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(tr)


def group_motion(gt_prev: np.ndarray, gt_cur: np.ndarray, hands_mean: np.ndarray
                 ) -> np.ndarray:
    """Ground-truth motion per group, `(16,)`, in radians (root translation in metres-as-radians).

    The root's two halves are combined by taking the larger of the rotation angle and the
    translation in metres scaled to a comparable size; the exact scaling only shifts where the
    threshold sweep lands, and the sweep is reported in full rather than at one operating point.
    """
    out = np.zeros(N_GROUPS, dtype=np.float64)
    dt = float(np.linalg.norm(gt_cur[0:3] - gt_prev[0:3]))
    dr = float(_geodesic(gt_prev[3:6], gt_cur[3:6]))
    out[0] = max(dr, dt / 0.05)          # 5 cm of translation counts like 1 rad of rotation
    a = gt_prev[6:51] + hands_mean
    b = gt_cur[6:51] + hands_mean
    out[1:] = _geodesic(a.reshape(15, 3), b.reshape(15, 3))
    return out


@dataclass
class ActivePolicy:
    """Decides, per step, which coordinate groups to update.

    `mode`:
      `oracle`  update a group only when the ground truth says it moved by at least `thresh`
      `random`  update a random subset of the same expected size, as the shrinkage control
      `events`  update the groups... all of them, gated on the packet's event count, which is the
                cheap heuristic S9 must beat
      `all`     identity, for a parity check against the untouched evaluator
    """

    mode: str = "all"
    thresh: float = 0.02
    hands_mean: Optional[np.ndarray] = None
    seed: int = 0
    event_thresh: int = 0
    _rng: np.random.Generator = field(default=None, repr=False)
    n_steps: int = 0
    n_updated: int = 0
    n_possible: int = 0
    active_hist: List[int] = field(default_factory=list)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        assert self.mode in ("all", "oracle", "random", "events"), self.mode
        if self.mode == "oracle":
            assert self.hands_mean is not None

    def reset(self) -> None:
        """Called at the start of each valid run; the counters are cumulative on purpose."""
        return None

    def rate(self) -> float:
        return self.n_updated / max(self.n_possible, 1)

    def __call__(self, prev: torch.Tensor, pred: torch.Tensor, gt_prev: np.ndarray,
                 gt_cur: np.ndarray, n_events: int = 0,
                 events: Optional[np.ndarray] = None, **kw) -> torch.Tensor:
        self.n_steps += 1
        self.n_possible += N_GROUPS
        if self.mode == "all":
            self.n_updated += N_GROUPS
            return pred

        if self.mode == "oracle":
            active = group_motion(gt_prev, gt_cur, self.hands_mean) >= self.thresh
        elif self.mode == "events":
            active = np.full(N_GROUPS, n_events >= self.event_thresh)
        else:
            # Same expected number of active groups as the oracle it controls for, but chosen
            # without looking at the motion.
            active = self._rng.random(N_GROUPS) < self.thresh

        self.n_updated += int(active.sum())
        self.active_hist.append(int(active.sum()))
        return apply_active(prev, pred, active)


def apply_active(prev: torch.Tensor, pred: torch.Tensor, active) -> torch.Tensor:
    """Take `pred` on the active groups and keep `prev` on the rest. The real skip."""
    out = prev.clone()
    for gi, (_name, sl) in enumerate(GROUPS):
        if bool(active[gi]):
            out[:, sl] = pred[:, sl]
    return out
