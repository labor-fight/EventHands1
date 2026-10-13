"""S43 event-driven sparse asynchronous hand tracker (docs/EXPERIMENT_RECORDS.md#s43-prereg, §1 item 7).

The runtime form of an S43 checkpoint (a `MNISTModel` with a `sparse_*` trunk) plus its deployed filter:

    ingest(xs, ys, ps, t_ms)   each event writes one cell of the per-pixel, per-polarity surface of active events (SAE:
                               the latest millisecond seen there), O(1) per event; nothing else runs
    query(t_ms)                at ANY time: LNES(t) from the SAE (latest event of the window [t - w + 1, t], normalised
                               exactly as `semkine.events.splat_event_image`), splat of the cached mesh, the trunk on the
                               active sites only (`sparse_trunk.SparseExecutor`), prev_mlp, zero-event hold, the S39-stack
                               count-gain filter and the kinematic-subtree gate (`semkine.kinematic_gate`); the new state
                               then updates the mesh cache locally (`IncrementalMano`)

The mesh is cached in the root frame, verts = R_root Q + J_0 + transl (MANO factorises so: Q does not depend on the root
rotation or the translation). A change of root rotation / translation is a rigid transform of Q (O(V)); a change of finger
joints C re-skins only the vertices with skinning weight on the subtrees of C and adds the posedirs rank update
(sum over the changed joints of P_j vec(R_j' - R_j)) through the unchanged blend transform to every other vertex. A joint
the gate holds is not in C: no work for it. `IncrementalMano.check()` compares the cache with the full MANO forward.

At a 50 ms query cadence on the protocol's steps the runtime computes the function `tools/s43/gate_eval.py` evaluates
(LNES from the SAE is the evaluator's LNES bit for bit; the trunk differs only by summation order, <= 1e-4).
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from semkine.anchored import AdaptiveFilter
from semkine.kinematic_gate import deadband_state, gate_state, subtree_matrix, support_matrix

__all__ = ["IncrementalMano", "EventDrivenTracker"]

_NEVER = -(2 ** 40)


def _rodrigues(aa: torch.Tensor) -> torch.Tensor:
    """(N, 3) -> (N, 3, 3): the MANO layer's own `batch_rodrigues`."""
    import mano_layer
    return mano_layer.batch_rodrigues(aa)


class IncrementalMano:
    """Root-frame MANO mesh cache for one hand (fixed betas) with local updates. All float32 on `device`."""

    def __init__(self, mano, betas: torch.Tensor, device, refresh_every: int = 100):
        self.mano, self.device = mano, device
        self.refresh_every = int(refresh_every)          # full recomputation every N updates bounds float drift
        b = betas.reshape(1, -1).to(device=device, dtype=torch.float32)
        self.v_shaped = (mano.v_template + torch.einsum("bl,mkl->bmk", b, mano.shapedirs))[0]      # (V, 3)
        self.J = torch.einsum("ik,ji->jk", self.v_shaped, mano.J_regressor)                        # (16, 3)
        self.parents = [int(p) for p in mano.parents.tolist()]
        self.W = mano.weights                                                                       # (V, 16)
        self.P = mano.posedirs                                                                      # (V, 3, 135)
        self.hands_mean = mano.hands_mean.reshape(15, 3)
        sub = subtree_matrix(self.parents).to(device)                                               # (16, 15)
        # (15, V) bool: vertex has weight on a joint of finger joint j's subtree
        self.touch = ((self.W > 0).float() @ sub > 0).t().contiguous()
        self.children = [[c for c in range(16) if self.parents[c] == j] for j in range(16)]
        self.stats = {"updates": 0, "joints_changed": 0, "verts_reskinned": 0, "rigid_only": 0}

    def _local_rot(self, params: torch.Tensor) -> torch.Tensor:
        """(16, 3, 3) joint rotations of a 51-d state (root + mean-added finger axis-angles, as `decode_to_mano_inputs`)."""
        aa = torch.cat([params[3:6].reshape(1, 3), params[6:51].reshape(15, 3) + self.hands_mean], dim=0)
        return _rodrigues(aa)

    def _chain(self, R: torch.Tensor):
        """K_j (16, 3, 4): the blend transforms in the root frame (the MANO chain without the root's [R_0 | J_0])."""
        H = [torch.eye(4, device=self.device)]
        for i in range(1, 16):
            T = torch.eye(4, device=self.device)
            T[:3, :3] = R[i]
            T[:3, 3] = self.J[i] - self.J[self.parents[i]]
            H.append(H[self.parents[i]] @ T)
        H = torch.stack(H)
        K = H[:, :3, :].clone()
        K[:, :, 3] = H[:, :3, 3] - torch.einsum("jab,jb->ja", H[:, :3, :3], self.J)
        return K

    def reset(self, params: torch.Tensor):
        p = params.reshape(-1).float().to(self.device)
        self.params = p.clone()
        self.R = self._local_rot(p)
        feat = (self.R[1:] - torch.eye(3, device=self.device)).reshape(-1)
        self.v_posed = self.v_shaped + torch.einsum("l,mkl->mk", feat, self.P)
        self.K = self._chain(self.R)
        self.A = torch.einsum("vj,jab->vab", self.W, self.K)                         # (V, 3, 4) per-vertex blend
        self.Q = torch.einsum("vab,vb->va", self.A[:, :, :3], self.v_posed) + self.A[:, :, 3]
        self._place()

    def _place(self):
        R0 = self.R[0]
        self.verts = self.Q @ R0.t() + self.J[0] + self.params[0:3]

    def update(self, params: torch.Tensor) -> Dict[str, int]:
        """Move the cache to `params`; returns what had to be recomputed."""
        p = params.reshape(-1).float().to(self.device)
        if self.refresh_every and (self.stats["updates"] + 1) % self.refresh_every == 0:
            self.stats["updates"] += 1
            self.stats["refresh"] = self.stats.get("refresh", 0) + 1
            self.reset(p)
            return {"joints_changed": -1, "verts_reskinned": int(self.Q.shape[0])}
        old = self.params
        fing = (p[6:51] != old[6:51]).reshape(15, 3).any(dim=1)
        changed = fing.nonzero().flatten() + 1                                       # MANO joint ids 1..15
        info = {"joints_changed": int(len(changed)), "verts_reskinned": 0}
        self.params = p.clone()
        if len(changed):
            Rn = self._local_rot(p)
            dfeat = torch.zeros(15, 3, 3, device=self.device)
            dfeat[changed - 1] = Rn[changed] - self.R[changed]
            dv = torch.einsum("l,mkl->mk", dfeat.reshape(-1), self.P)                # rank update of the posed rest mesh
            self.R = torch.cat([Rn[0:1], self.R[1:]])                                 # root handled by _place
            self.R[changed] = Rn[changed]
            self.v_posed = self.v_posed + dv
            touch = self.touch[changed - 1].any(dim=0)                                # vertices re-skinned
            self.K = self._chain(self.R)
            idx = touch.nonzero().flatten()
            A_new = torch.einsum("vj,jab->vab", self.W[idx], self.K)
            keep = (~touch).nonzero().flatten()
            self.Q[keep] = self.Q[keep] + torch.einsum("vab,vb->va", self.A[keep, :, :3], dv[keep])
            self.A[idx] = A_new
            self.Q[idx] = torch.einsum("vab,vb->va", A_new[:, :, :3], self.v_posed[idx]) + A_new[:, :, 3]
            info["verts_reskinned"] = int(len(idx))
        else:
            self.R = torch.cat([self._local_rot(p)[0:1], self.R[1:]])
        self._place()
        self.stats["updates"] += 1
        self.stats["joints_changed"] += info["joints_changed"]
        self.stats["verts_reskinned"] += info["verts_reskinned"]
        self.stats["rigid_only"] += int(info["joints_changed"] == 0)
        return info

    @torch.no_grad()
    def check(self, betas: torch.Tensor) -> float:
        """max |cached verts - full MANO forward verts| (m)."""
        from pose_repr import decode_to_mano_inputs
        dec = decode_to_mano_inputs(self.params[None], "mano_full_axis_angle", self.mano.hands_components,
                                    self.mano.hands_mean)
        v, _ = self.mano(betas.reshape(1, -1).float().to(self.device), dec["global_orient"], dec["local_full_aa"],
                         dec["transl"])
        return float((v[0] - self.verts).abs().max())


class _Fixed(nn.Module):
    """A tracker stand-in that returns a precomputed output: lets the runtime reuse AdaptiveFilter's exact arithmetic."""

    def __init__(self):
        super().__init__()
        self.z = None

    def forward(self, x, prevpos, betas=None, camera_K=None):
        return self.z


class EventDrivenTracker:
    """See the module docstring. `model`: eval-mode MNISTModel with a sparse trunk; `spec`: the AdaptiveFilter spec of
    the deployed stack (None = raw tracker); `n0`: the kinematic gate threshold (inf = off); `window_ms`: the evidence
    window W; `hop_ms`: the recent-evidence span of the count gains and of the gate (50 = the protocol's packet)."""

    def __init__(self, model, spec: Optional[dict], n0: float = float("inf"), radius_px: float = 8.0,
                 window_ms: int = 200, hop_ms: int = 50, executor: str = "sparse", device=None,
                 height: int = 180, width: int = 240, n0_root: float = float("inf"), host_state: bool = False,
                 eps_f: float = 0.0, eps_r: float = 0.0):
        import copy
        from sparse_trunk import SparseExecutor
        if not getattr(model, "sparse_trunk", False):
            raise ValueError("EventDrivenTracker runs an S43 checkpoint (MODEL.CNN_BACKBONE sparse_*)")
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.exec = SparseExecutor(model.conv1, model.rn) if executor == "sparse" else None
        # host_state: the 51-d state, prev_mlp, the filter / gate arithmetic and the mesh cache live on the host (CPU);
        # the device keeps the SAE, the splat and the trunk. Same arithmetic, other rounding of the tiny 51-d / 4x4 ops.
        self.host = torch.device("cpu") if host_state else self.device
        self.prev_mlp = copy.deepcopy(model.prev_mlp).to(self.host).eval()
        self.transl_head = None if getattr(model, "transl_head", None) is None else \
            copy.deepcopy(model.transl_head).to(self.host).eval()
        self.mano_h = copy.deepcopy(model.mano).to(self.host).eval() if host_state else model.mano
        self.filter = None if spec is None else AdaptiveFilter.from_spec(_Fixed(), spec).to(self.host).eval()
        self.n0, self.n0_root, self.r2 = float(n0), float(n0_root), float(radius_px) ** 2
        self.eps_f, self.eps_r = float(eps_f), float(eps_r)                          # update dead bands (rad)
        self.window_ms, self.hop_ms = int(window_ms), int(hop_ms)
        self.h, self.w = height, width
        self.vert_support = support_matrix(model.mano).to(self.device)              # (778, 16): palm, finger subtrees
        self.sae = torch.full((height * width * 2,), _NEVER, dtype=torch.long, device=self.device)
        self.ev_n = {}                                     # events per millisecond (host), for the count gain
        self.timing = {"ingest": 0.0, "query": 0.0, "n_query": 0, "n_events": 0}
        self.macs = []

    # ---- context and reset
    def set_hand(self, betas: torch.Tensor, camera_K: torch.Tensor):
        """A new event stream (sequence): its hand and camera; the SAE and the event-count history start empty (a stream
        has its own clock, so another stream's timestamps must not survive into it)."""
        self.sae.fill_(_NEVER)
        self.ev_n = {}
        self.betas = betas.reshape(1, -1).float().to(self.device)
        self.K = camera_K.reshape(1, 3, 3).float().to(self.device)
        self.mesh = IncrementalMano(self.mano_h, self.betas.to(self.host), self.host)

    def reset(self, state51: torch.Tensor, floor_ms: int):
        """New segment: the state, the earliest millisecond the window may reach (training's clip-run-start rule)."""
        self.state = state51.reshape(1, 51).float().to(self.host)
        self.floor = int(floor_ms)
        self.mesh.reset(self.state[0])
        if self.filter is not None:
            self.filter.reset_state(self.state)

    # ---- events
    @torch.no_grad()
    def ingest(self, xs, ys, ps, t_ms):
        """Events (arrays, time-ordered) with integer millisecond stamps: SAE[pixel, polarity] = max(SAE, t)."""
        t0 = time.perf_counter()
        xs = torch.as_tensor(xs, dtype=torch.long, device=self.device)
        ys = torch.as_tensor(ys, dtype=torch.long, device=self.device)
        ps = torch.as_tensor(ps, dtype=torch.long, device=self.device).clamp(0, 1)
        t = torch.full_like(xs, int(t_ms))
        self.sae.scatter_reduce_(0, (ys * self.w + xs) * 2 + ps, t, reduce="amax")
        self.ev_n[int(t_ms)] = self.ev_n.get(int(t_ms), 0) + int(xs.numel())
        self.timing["ingest"] += time.perf_counter() - t0
        self.timing["n_events"] += int(xs.numel())

    def _recent_count(self, t_ms: int) -> int:
        lo = t_ms - self.hop_ms + 1
        for k in [k for k in self.ev_n if k < lo]:
            del self.ev_n[k]
        return int(sum(n for k, n in self.ev_n.items() if k <= t_ms))

    # ---- one query
    def lnes(self, t_ms: int):
        w = min(self.window_ms, t_ms - self.floor + 1)
        start = t_ms - w + 1
        s = self.sae.view(self.h, self.w, 2)
        # a 0-dim device tensor, not a Python float: CUDA divides by a host scalar as a multiplication by its reciprocal,
        # which rounds differently from numpy's float32 division in `splat_event_image`
        val = (s - start).to(torch.float32) / torch.tensor(float(w), dtype=torch.float32, device=s.device)
        return torch.where(s >= start, val, torch.zeros_like(val)), w

    def render(self):
        m = self.model
        if getattr(self, "render_full_fk", False):          # diagnostic: the model's own FK render of the state
            return m._render_prev(self.state.float().to(self.device), self.betas, self.K)[0]
        fx, fy, cx, cy = m._intrinsics(self.K)
        v = self.mesh.verts.to(self.device, non_blocking=True)
        return m._splat(v[None], fx, fy, cx, cy, m.render_h, m.render_w)[0]          # (H, W, C)

    @torch.no_grad()
    def query(self, t_ms: int) -> Dict:
        t0 = time.perf_counter()
        m = self.model
        lnes, w = self.lnes(t_ms)
        n_ev = self._recent_count(t_ms)
        prev = self.state
        empty = not bool((lnes != 0).any())
        if empty:
            z = prev.clone()
            macs = 0
        else:
            img = torch.cat([lnes, self.render().to(lnes.dtype)], dim=-1)
            if self.exec is not None:
                logits, macs, _ = self.exec.run(img)
                logits, feats = logits[None], self.exec.last_feats[None]
            else:
                x = img.permute(2, 0, 1)[None]
                from sparse_trunk import input_mask
                m0 = input_mask(x)
                feats, _ = m.rn.pooled(m.conv1(x) * m0, m0)
                logits, macs = m.rn.fc(feats), 0
            logits, feats = logits.to(self.host), feats.to(self.host)
            pm = self.prev_mlp(prev)
            if not m.prev_mlp_transl:
                t = m.slices.transl
                pm = torch.cat([pm[:, : t.start], torch.zeros_like(pm[:, t]), pm[:, t.stop:]], dim=-1)
            z = prev + (logits + pm)
            if self.transl_head is not None:
                tz = prev[:, :3] + self.transl_head(torch.cat([feats.to(prev.dtype), prev[:, :3]], dim=1))
                z = torch.cat([tz, z[:, 3:]], dim=1)
        out = z
        if self.filter is not None:
            self.filter.trk_model.z = z
            # the filter reads its input only for the empty-packet test: hand it a host scalar of the right shape
            out = self.filter(torch.full((1, 1), 0.0 if empty else 1.0, device=self.host), prev, n_events=n_ev)
        gains = None
        if not empty and not (np.isinf(self.n0) and np.isinf(self.n0_root)):
            cnt = self._joint_counts(lnes, w).to(self.host)
            out = gate_state(prev.to(out.dtype), out, cnt[None], self.n0, self.n0_root)
            gains = cnt
        if not empty:
            out = deadband_state(prev.to(out.dtype), out, self.eps_f, self.eps_r)
        info = self.mesh.update(out[0])
        self.state = out
        self.macs.append(macs)
        self.timing["query"] += time.perf_counter() - t0
        self.timing["n_query"] += 1
        return {"state": out, "macs": macs, "n_events": n_ev, "window": w, "mesh": info,
                "gains": None if gains is None else gains.cpu().numpy()}

    def _joint_counts(self, lnes, w):
        ev = lnes.amax(dim=-1)
        if w <= self.hop_ms:
            recent = ev > 0
        else:
            thr = torch.tensor(float(w - self.hop_ms), dtype=torch.float32) / float(w)
            recent = ev >= thr.to(ev.device)
        pix = recent.nonzero()
        cnt = torch.zeros(self.vert_support.shape[1], device=self.device)
        if not len(pix):
            return cnt
        m = self.model
        fx, fy, cx, cy = m._intrinsics(self.K)
        v = self.mesh.verts.to(self.device)
        zz = v[:, 2].clamp(min=1e-6)
        uv = torch.stack([fx[0] * v[:, 0] / zz + cx[0], fy[0] * v[:, 1] / zz + cy[0]], dim=-1)
        p = torch.stack([pix[:, 1].float(), pix[:, 0].float()], dim=-1)
        for s in range(0, len(p), 4096):
            d2 = torch.cdist(p[s:s + 4096], uv).square()
            best, idx = d2.min(dim=1)
            ok = best <= self.r2
            if ok.any():
                cnt += self.vert_support[idx[ok]].sum(dim=0)
        return cnt
