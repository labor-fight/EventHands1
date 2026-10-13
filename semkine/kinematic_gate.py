"""S43 kinematic-subtree evidence gate (docs/EXPERIMENT_RECORDS.md#s43-prereg, §1 item 4): local finger updates follow
local event evidence along the kinematic chain.

`KinematicGateFilter` is `semkine.anchored.AdaptiveFilter` (event-count gains per block, the S39 stack) with one more
factor on the finger rows: finger joint j's update `out_j - prev_j` is multiplied by `g_j = min(1, n_j / n0)`, where

    n_j = number of RECENT event pixels (latest event within the last `hop_ms` of the evidence window, both polarities
          pooled) whose nearest projected vertex of the PREVIOUS mesh (within `radius_px`) belongs to j's kinematic subtree.

A vertex belongs to the joint that dominates its skinning weights; the subtree of j is j and its descendants along the
finger (MANO parents). So a fingertip event supports the distal, middle and proximal joints of its finger (any of them can
have moved the tip), an event on the proximal phalanx supports only the proximal joint, and palm events support no finger
joint. Without local evidence a joint keeps its local angle; it still moves rigidly with its parents and the root (FK), so
the mesh is never torn. Root rotation and translation keep the block gains of the S39 stack. In a sequence where the whole
hand moves every finger has evidence and every g_j = 1: one mechanism for local and global, no label switch.

Inference only, batch rows independent. `window_ms` (evalx passes it for `takes_window` models) is the evidence window of
the step; recent = LNES value >= (w - hop) / w, computed in float32 exactly as the LNES itself is, so the test is exact.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from semkine.anchored import AdaptiveFilter

__all__ = ["KinematicGateFilter", "deadband_state", "gate_state", "subtree_matrix", "support_matrix"]


def support_matrix(mano) -> torch.Tensor:
    """(778, 16) float: column 0 = the vertex belongs to the palm (dominant skinning joint 0, moved by the root alone),
    column j = 1..15 = the vertex lies on finger joint j's kinematic subtree (`subtree_matrix` of its dominant joint)."""
    dom = mano.weights.argmax(dim=1)
    sub = subtree_matrix([int(p) for p in mano.parents.tolist()]).to(mano.weights.device)
    return torch.cat([(dom == 0).float()[:, None], sub[dom]], dim=1)


def subtree_matrix(parents) -> torch.Tensor:
    """(16, 15) float: [d, j - 1] = 1 iff joint d is in the subtree of finger joint j (j = 1..15)."""
    n = len(parents)
    m = torch.zeros(n, n - 1)
    for d in range(n):
        a = d
        while a > 0:
            m[d, a - 1] = 1.0
            a = int(parents[a])
    return m


class KinematicGateFilter(AdaptiveFilter):
    takes_window = True

    def __init__(self, trk_model, spec: dict, n0: float, radius_px: float = 8.0, hop_ms: int = 50,
                 chunk: int = 4096, n0_root: float = float("inf"), eps_f: float = 0.0, eps_r: float = 0.0):
        super().__init__(trk_model, **AdaptiveFilter.canonical_spec(spec))
        self.eps_f, self.eps_r = float(eps_f), float(eps_r)                  # update dead bands (rad), deviation 7
        if not (n0 > 0 and n0_root > 0):
            raise ValueError(f"n0 / n0_root must be positive (inf = gate off), got {n0} / {n0_root}")
        self.n0, self.radius_px, self.hop_ms, self.chunk = float(n0), float(radius_px), int(hop_ms), int(chunk)
        self.n0_root = float(n0_root)
        self.register_buffer("vert_support", support_matrix(trk_model.mano), persistent=False)   # (778, 16)
        self.last_counts = None                                             # (B, 16) of the latest call, for reports

    def spec(self) -> dict:
        s = super().spec()
        s.update(kin_n0=self.n0, kin_n0_root=self.n0_root, kin_radius_px=self.radius_px, kin_hop_ms=self.hop_ms,
                 deadband_finger_rad=self.eps_f, deadband_root_rad=self.eps_r)
        return s

    @torch.no_grad()
    def joint_counts(self, x, prevpos, window_ms: Optional[int], betas=None, camera_K=None) -> torch.Tensor:
        """(B, 16) float: recent event pixels on the palm (column 0) and supporting each finger joint (1..15) under
        the previous mesh."""
        trk = self.trk_model
        b = x.shape[0]
        betas_f, k_f = trk._resolve_betas_K(prevpos, betas, camera_K)
        verts, _ = trk._fk(prevpos.float(), betas_f)
        fx, fy, cx, cy = trk._intrinsics(k_f)
        vx, vy, vz = verts.float().unbind(-1)
        z = vz.clamp(min=1e-6)
        uv = torch.stack([fx[:, None] * vx / z + cx[:, None], fy[:, None] * vy / z + cy[:, None]], dim=-1)  # (B, V, 2)
        ev = x[..., :2].float().amax(dim=-1)                                # (B, H, W) latest normalised time, any polarity
        if window_ms is None or int(window_ms) <= self.hop_ms:
            recent = ev > 0
        else:
            w = int(window_ms)
            thr = torch.tensor(float(w - self.hop_ms), dtype=torch.float32) / float(w)
            recent = ev >= thr.to(ev.device)
        out = torch.zeros(b, self.vert_support.shape[1], device=x.device)
        r2 = self.radius_px ** 2
        for i in range(b):
            pix = recent[i].nonzero()                                       # (P, 2) row, col
            if not len(pix):
                continue
            p = torch.stack([pix[:, 1].float(), pix[:, 0].float()], dim=-1)  # pixel centres in render coordinates
            for s in range(0, len(p), self.chunk):
                d2 = torch.cdist(p[s:s + self.chunk], uv[i]).square()        # (P, V)
                best, idx = d2.min(dim=1)
                ok = best <= r2
                if ok.any():
                    out[i] += self.vert_support[idx[ok]].sum(dim=0)
        return out

    def forward(self, x, prevpos, betas=None, camera_K=None, n_events=None, window_ms=None):
        out = super().forward(x, prevpos, betas=betas, camera_K=camera_K, n_events=n_events)
        prev = prevpos.to(out.dtype)
        if not (math.isinf(self.n0) and math.isinf(self.n0_root)):
            cnt = self.joint_counts(x, prevpos, window_ms, betas, camera_K)
            self.last_counts = cnt
            out = gate_state(prev, out, cnt, self.n0, self.n0_root)
        return deadband_state(prev, out, self.eps_f, self.eps_r)


def gate_state(prev: torch.Tensor, out: torch.Tensor, cnt: torch.Tensor, n0: float, n0_root: float) -> torch.Tensor:
    """Apply the evidence gates to a filtered update `prev -> out` (B, 51): finger joint j's rows move by
    min(1, cnt[:, j] / n0) of their update; root rotation (geodesic) and translation by min(1, cnt[:, 0] / n0_root)."""
    res = out
    if not math.isinf(n0):
        g = (cnt[:, 1:] / n0).clamp(max=1.0).to(out.dtype)                  # (B, 15)
        f_prev, f_out = prev[:, 6:].reshape(-1, 15, 3), out[:, 6:].reshape(-1, 15, 3)
        res = torch.cat([res[:, :6], (f_prev + g[..., None] * (f_out - f_prev)).reshape(-1, 45)], dim=1)
    if not math.isinf(n0_root):
        from semkine.anchored import _t_conj, _t_log, _t_qmul, _t_quat
        g = (cnt[:, :1] / n0_root).clamp(max=1.0).to(torch.float64)          # (B, 1)
        p, o = prev.to(torch.float64), res.to(torch.float64)
        t_new = p[:, :3] + g * (o[:, :3] - p[:, :3])
        q_prev = _t_quat(p[:, 3:6])
        rel = _t_log(_t_qmul(_t_conj(q_prev), _t_quat(o[:, 3:6])))
        r_new = _t_log(_t_qmul(q_prev, _t_quat(g * rel)))
        res = torch.cat([t_new.to(out.dtype), r_new.to(out.dtype), res[:, 6:]], dim=1)
    return res


def deadband_state(prev: torch.Tensor, out: torch.Tensor, eps_f: float, eps_r: float) -> torch.Tensor:
    """Deviation 7: write a finger joint only if its axis-angle change |out_j - prev_j| reaches `eps_f` (rad), the root
    rotation only if its geodesic change reaches `eps_r` (rad); held rows are copied from `prev` exactly (so the mesh
    cache does not touch their subtrees). Translation is never held. eps = 0 returns `out` unchanged."""
    if eps_f <= 0.0 and eps_r <= 0.0:
        return out
    res = out
    if eps_f > 0.0:
        f_prev, f_out = prev[:, 6:].reshape(-1, 15, 3), out[:, 6:].reshape(-1, 15, 3)
        keep = (f_out - f_prev).norm(dim=-1, keepdim=True) < eps_f                # (B, 15, 1)
        res = torch.cat([res[:, :6], torch.where(keep, f_prev, f_out).reshape(-1, 45)], dim=1)
    if eps_r > 0.0:
        from semkine.anchored import _t_conj, _t_log, _t_qmul, _t_quat
        p, o = prev.to(torch.float64), res.to(torch.float64)
        ang = _t_log(_t_qmul(_t_conj(_t_quat(p[:, 3:6])), _t_quat(o[:, 3:6]))).norm(dim=-1, keepdim=True)
        hold = ang < eps_r                                                          # (B, 1)
        res = torch.cat([res[:, :3], torch.where(hold, prev[:, 3:6], res[:, 3:6]), res[:, 6:]], dim=1)
    return res
