#!/usr/bin/env python3
r"""XKF (docs/EXPERIMENT_RECORDS.md#xkf-20261014): the S37 XYZ path as a motion model plus a prev-free event
measurement, fused with gains taken from residual statistics.

    x_I = XYZ(events_W50, prev)                 unchanged association + MANO graph + bank readout (state96, no t),
                                                trained as a motion model (small prev noise)
    z_D = Direct(LNES_long, camera)             never reads prev (semkine/xkf_direct.py)
    x   = fuse(x_I, z_D; K)                     per block: t (per axis), root (SO(3), isotropic), 15 finger joints

Gains (no backprop through the fusion; E-3DPSM keeps Q, R constant and reports input-dependent ones worse):
running second moments of the residuals e_I = x_I - y, e_D = z_D - y, e_0 = prev - y on self rows (all rows before
the self curriculum starts), synchronised across ranks, and the minimum-variance combination of two correlated
estimates  K = (V_I - C) / (V_I + V_D - 2C)  clipped to [0, 1].  Both inputs are detached before fusing, so the direct
branch can never learn to cancel the incremental branch's bias: it stays a measurement.

`XKF_KMODE` (evaluation ablations of one checkpoint): fused | i (K = 0) | d (K = 1) | d_zero (prev fused with z_D,
the "measurement plus zero-motion smoother" control). An empty W50 packet holds prev exactly in every mode.
Loss: log10(L_I) + log10(L_D); the two branches share no parameter, so L_I has exactly the XDL form.
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from semkine.lie import so3_exp, so3_log
from semkine.xkf_direct import DirectBranch, dgp_translation, matrix_to_rot6d, rot6d_to_matrix
from semkine.xyz_recovery_study import RecoveryXYZModel

#: gain groups: t_x, t_y, t_z, root, 15 finger joints
N_GROUPS = 19
KMODES = ('fused', 'i', 'd', 'd_zero')
#: innovation clip of the root fusion (rad)
ROOT_CLIP = math.radians(60.0)


def _root_residual(aa, aa_ref):
    """Tangent residual Log(R_ref^T R) `(B, 3)`."""
    return so3_log(so3_exp(aa_ref.float()).transpose(-1, -2) @ so3_exp(aa.float()))


def group_residuals(x, y):
    """`(B, 19)` squared-norm-ready residual vectors packed as a list of per-group tensors: t axes (B,1) x3,
    root (B,3), finger joints (B,3) x15."""
    e_t = (x[:, :3] - y[:, :3]).float()
    e_r = _root_residual(x[:, 3:6], y[:, 3:6])
    e_f = (x[:, 6:51] - y[:, 6:51]).float().view(-1, 15, 3)
    return e_t, e_r, e_f


def group_moments(ea, eb):
    """Per-group mean inner product `(19,)` of two residual triples."""
    ta, ra, fa = ea
    tb, rb, fb = eb
    return torch.cat([(ta * tb).mean(0), (ra * rb).sum(-1).mean(0, keepdim=True), (fa * fb).sum(-1).mean(0)])


def nearest_aa(aa, ref):
    """Axis-angle branch of the same rotation closest to `ref` (avoids a 2*pi jump for additive successors)."""
    th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    alt = aa * (1.0 - 2.0 * math.pi / th)
    use_alt = (alt - ref).norm(dim=-1, keepdim=True) < (aa - ref).norm(dim=-1, keepdim=True)
    return torch.where(use_alt & (th > 1e-6), alt, aa)


def fuse_states(a, b, k):
    """a, b `(B, 51)`, k `(19,)` or `(B, 19)` gains in [0, 1]: a + K (b - a) per group; root on SO(3)."""
    k = k.expand(a.shape[0], N_GROUPS) if k.dim() == 1 else k
    out = a.float().clone()
    out[:, :3] = a[:, :3].float() + k[:, :3] * (b[:, :3].float() - a[:, :3].float())
    Ra = so3_exp(a[:, 3:6].float())
    innov = so3_log(Ra.transpose(-1, -2) @ so3_exp(b[:, 3:6].float()))
    n = innov.norm(dim=-1, keepdim=True)
    innov = innov * torch.clamp(ROOT_CLIP / n.clamp_min(1e-9), max=1.0)
    root = nearest_aa(so3_log(Ra @ so3_exp(k[:, 3:4] * innov)), a[:, 3:6].float())
    # a zero gain keeps the incoming root exactly (no exp/log round trip)
    out[:, 3:6] = torch.where(k[:, 3:4] > 0, root, a[:, 3:6].float())
    kf = k[:, 4:19].repeat_interleave(3, dim=1)
    out[:, 6:51] = a[:, 6:51].float() + kf * (b[:, 6:51].float() - a[:, 6:51].float())
    return out


class XKFModel(RecoveryXYZModel):
    MODEL_KEYS = RecoveryXYZModel.MODEL_KEYS | frozenset({'XKF', 'XKF_KMODE', 'XKF_EMA', 'XKF_KP_WEIGHT',
                                                        'XKF_HM_WEIGHT', 'XKF_FK_WEIGHT'})

    def __init__(self, cfg=None, num_classes=None):
        mc = (cfg or {}).get('MODEL', {})
        if mc.get('XKF') is not True:
            raise ValueError('XKFModel requires MODEL.XKF=true')
        if mc.get('XDL_STATE_T', True) is not False:
            raise ValueError('XKF requires MODEL.XDL_STATE_T=false (no absolute translation in the motion branch)')
        super().__init__(cfg=cfg, num_classes=num_classes)
        self.kmode = str(mc.get('XKF_KMODE', 'fused'))
        if self.kmode not in KMODES:
            raise ValueError(f'XKF_KMODE must be one of {KMODES}')
        self.xkf_ema = float(mc.get('XKF_EMA', 0.99))
        self.kp_weight = float(mc.get('XKF_KP_WEIGHT', 300.0))
        self.hm_weight = float(mc.get('XKF_HM_WEIGHT', 30.0))
        self.direct_fk_weight = float(mc.get('XKF_FK_WEIGHT', 30000.0))
        self.direct = DirectBranch(self.render_h, self.render_w)
        # second moments: V_I, V_D, V_0, C_ID, C_0D per gain group (start: trust the motion branch)
        self.register_buffer('xkf_moments', torch.tensor([[1e-4] * N_GROUPS, [1.0] * N_GROUPS, [1e-4] * N_GROUPS,
                                                          [0.0] * N_GROUPS, [0.0] * N_GROUPS]))
        self.register_buffer('xkf_updates', torch.zeros((), dtype=torch.long))

    # ------------------------------------------------------------------ gains
    def gains(self, mode=None):
        mode = mode or self.kmode
        V_I, V_D, V_0, C_ID, C_0D = self.xkf_moments
        if mode == 'i':
            return torch.zeros(N_GROUPS, device=V_I.device)
        if mode == 'd':
            return torch.ones(N_GROUPS, device=V_I.device)
        if mode == 'd_zero':
            return ((V_0 - C_0D) / (V_0 + V_D - 2 * C_0D).clamp_min(1e-12)).clamp(0, 1)
        return ((V_I - C_ID) / (V_I + V_D - 2 * C_ID).clamp_min(1e-12)).clamp(0, 1)

    # ------------------------------------------------------------------ direct measurement
    def _lnes_float(self, batch):
        if batch.lnes is None or batch.lnes_window_ms is None:
            raise RuntimeError('XKF needs a long-window LNES (batch.lnes and batch.lnes_window_ms)')
        lnes = batch.lnes
        W = batch.lnes_window_ms.to(lnes.device).float().view(-1, 1, 1, 1)
        if lnes.dtype in (torch.int16, torch.int32, torch.int64):
            code = lnes.float()
            return torch.where(code > 0, (code - 1.0) / W, torch.zeros_like(code)), W.view(-1)
        return lnes.float(), W.view(-1)

    def direct_measurement(self, batch):
        """z_D `(B, 51)` and auxiliaries; reads events, camera and betas only (never batch.prev_state)."""
        lnes, W = self._lnes_float(batch)
        B = lnes.shape[0]
        betas, camera = self._resolve_betas_K(lnes, batch.betas, batch.camera_K)
        cam = self._cam_planes(camera)
        out = self.direct(lnes, cam, W)
        R = rot6d_to_matrix(out['root6d'].float())
        with torch.autocast(device_type=lnes.device.type, enabled=False):
            root = so3_log(R)
            pose0 = torch.cat([torch.zeros(B, 3, device=lnes.device), root, out['fingers'].float()], -1)
            _, joints0 = self._fk(pose0, betas.float())
            fx, fy, cx, cy = self._intrinsics(camera.float())
            kp = out['kp_px'].float()
            uv = torch.stack([(kp[..., 0] - cx[:, None]) / fx[:, None], (kp[..., 1] - cy[:, None]) / fy[:, None]], -1)
            w = F.softplus(out['kp_logit'].float()) + 1e-3
            t, covariance = dgp_translation(joints0.float(), uv, w, return_covariance=True)
        z = torch.cat([t, root, out['fingers'].float()], -1)
        return z, {'R': R, 'kp_px': kp, 'heat': out['heat'], 'joints0': joints0, 'w': w,
                   'dgp_covariance': covariance}

    # ------------------------------------------------------------------ forward
    def branches(self, batch):
        x_i = RecoveryXYZModel.forward_packet(self, batch)
        z_d, aux = self.direct_measurement(batch)
        return x_i, z_d, aux

    def forward_packet(self, batch):
        x_i, z_d, _ = self.branches(batch)
        prev = batch.prev_state.float()
        if self.kmode == 'd':
            # Pure measurement must remain independent of prev even when the root innovation exceeds 60 degrees.
            # The innovation clip belongs to fusion, and would otherwise contaminate the G1 / G5 D-only control.
            fused = z_d.detach()
        elif self.kmode == 'd_zero':
            fused = fuse_states(prev, z_d.detach(), self.gains('d_zero'))
        else:
            fused = fuse_states(x_i.detach().float(), z_d.detach(), self.gains())
        return torch.where((batch.counts <= 0)[:, None], prev, fused)

    # ------------------------------------------------------------------ training
    def _direct_loss(self, z, aux, batch):
        y = batch.target.float()
        betas, camera = self._resolve_betas_K(y, batch.betas, batch.camera_K)
        with torch.autocast(device_type=y.device.type, enabled=False):
            Rg = so3_exp(y[:, 3:6])
            l_root = (aux['R'].float() - Rg).square().sum(dim=(-2, -1)).mean() / 6.0
            l_f = (z[:, 6:51] - y[:, 6:51]).square().mean()
            l_t = (z[:, :3] - y[:, :3]).square().mean()
            _, jg = self._fk(y, betas.float())
            fx, fy, cx, cy = self._intrinsics(camera.float())
            zc = jg[..., 2].clamp_min(1e-3)
            gx = fx[:, None] * jg[..., 0] / zc + cx[:, None]
            gy = fy[:, None] * jg[..., 1] / zc + cy[:, None]
            inside = (gx >= 0) & (gx < self.render_w) & (gy >= 0) & (gy < self.render_h) & (jg[..., 2] > 1e-3)
            target_px = torch.stack([gx, gy], -1)
            err = (aux['kp_px'] - target_px).abs().sum(-1) / float(self.render_w)
            l_kp = (err * inside).sum() / inside.sum().clamp_min(1.0)
            l_hm = self.direct.heatmap_loss(aux['heat'], target_px, inside)
            root = self.FK_ROOT_JOINT
            pred_rel = aux['joints0'] - aux['joints0'][:, root:root + 1]
            gt_rel = jg - jg[:, root:root + 1]
            l_fk = (pred_rel - gt_rel).square().mean()
        loss = (self.lambda_pose * l_f + self.lambda_r * l_root + self.lambda_t * l_t + self.kp_weight * l_kp
                + self.hm_weight * l_hm + self.direct_fk_weight * l_fk) / self.normalizer
        return loss, {'d_root': l_root, 'd_fingers': l_f, 'd_t': l_t, 'd_kp': l_kp,
                      'd_hm': l_hm, 'd_fk': l_fk}

    def _moment_rows(self, main, take):
        live = set((main.counts > 0).nonzero().squeeze(1).tolist())
        # Bootstrap on noisy-GT rows only before self burn starts. Once the curriculum is live, zero self rows
        # means zero local weight; every rank still enters _update_moments and its collective reductions.
        if self.xdl_self and self.self_probability(int(self.global_step)) > 0:
            return [r for r in take if r in live]
        return sorted(live)

    @torch.no_grad()
    def _update_moments(self, x_i, z_d, prev, y, rows):
        """EMA of the per-group residual moments over `rows`, weighted by row counts across ranks (a rank with no
        usable row contributes zero weight but still joins the all-reduce)."""
        idx = torch.as_tensor(list(rows), device=y.device, dtype=torch.long)
        n = torch.tensor(float(idx.numel()), device=y.device)
        if idx.numel():
            x_i, z_d, prev, y = x_i[idx], z_d[idx], prev[idx], y[idx]
            e_i, e_d, e_0 = group_residuals(x_i, y), group_residuals(z_d, y), group_residuals(prev, y)
            batch = torch.stack([group_moments(e_i, e_i), group_moments(e_d, e_d), group_moments(e_0, e_0),
                                 group_moments(e_i, e_d), group_moments(e_0, e_d)]) * n
        else:
            batch = torch.zeros_like(self.xkf_moments, device=y.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch)
            dist.all_reduce(n)
        if float(n) <= 0:
            return
        batch = batch / n
        m = self.xkf_ema if int(self.xkf_updates) > 0 else 0.0
        self.xkf_moments.mul_(m).add_((1.0 - m) * batch.to(self.xkf_moments))
        self.xkf_updates += 1

    def training_step(self, batch, batch_nb):
        if not isinstance(batch, dict) or 'main' not in batch:
            raise ValueError('XKF trains on the XDL clip batches (main + prefix)')
        main, take = self._burn_main(batch)
        x_i, z_d, aux = self.branches(main)
        y = main.target.float()
        l_i, parts = self._compute_loss(x_i, main.target, main.betas)
        l_d, dparts = self._direct_loss(z_d, aux, main)
        rows = self._moment_rows(main, take)
        self._update_moments(x_i.detach().float(), z_d.detach().float(), main.prev_state.float(), y, rows)
        loss = l_i.float().log10() + l_d.float().log10()
        k = self.gains()
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_loss_i', l_i.float().log10(), sync_dist=True)
        self.log('train_loss_d', l_d.float().log10(), sync_dist=True)
        for name, v in dparts.items():
            self.log(f'train_{name}', v, sync_dist=True)
        self.log('train_k_t', k[:3].mean(), sync_dist=True)
        self.log('train_k_root', k[3], sync_dist=True)
        self.log('train_k_finger', k[4:].mean(), sync_dist=True)
        return loss


__all__ = ['KMODES', 'N_GROUPS', 'XKFModel', 'fuse_states', 'group_moments', 'group_residuals', 'nearest_aa']
