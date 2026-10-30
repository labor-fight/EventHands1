#!/usr/bin/env python3
r"""XRF (docs/EXPERIMENT_RECORDS.md#xrf-20261015): the S37 XYZ path as a refiner of a prev-free event measurement,
followed by a constant-gain filter against the previous state.

    z_D = Direct(LNES_long, camera)                   never reads prev (semkine/xkf_direct.py: HandDGP translation)
    z_R = XYZ(events_W50, hypothesis = z_D)           unchanged stride_sample / associate_rays / XYZMeshNet / bank
                                                      readout (state96); the hypothesis takes the place of prev
    x   = prev (+) K * (z_R (-) prev)                 per group (t axes, root on SO(3), 15 finger joints)

The measurement chain z_D -> z_R never sees prev, so a wrong prev cannot bias it; the only use of prev is the
filter, whose gains are constants calibrated after training on held-out training-identity blocks (never zgz):
steady-state Kalman gains of a random-walk state, K = P/(P+R) with P = (Q + sqrt(Q^2 + 4QR))/2, Q the per-step
GT motion variance and R the measurement residual variance (tools/xrf/calibrate.py). XYZ refinement around a
hypothesis follows mesh-aligned refinement (PyMAF, WiLoR CVPR 2025): association is reliable when the hypothesis is
close (XA: part agreement 80.6% from GT vs 41.9% from closed-loop prev).

Training: per packet, no burn. The refiner's hypothesis is GT + noise with probability p(step) (1 -> XRF_HYP_GT_END
over XRF_HYP_RAMP steps, so the refiner learns before the direct branch is useful) and otherwise the detached
direct estimate + small noise. Loss log10(L_R) + log10(L_D).

`XRF_KMODE` (evaluation of one checkpoint): fused | r (K = 1 on z_R: no filter) | d (K = 1 on z_D: no refiner) |
d_fused (filter on z_D). An empty W50 packet holds prev exactly.
"""
from __future__ import annotations

import dataclasses

import torch

from semkine.lie import so3_exp, so3_log
from semkine.xkf import N_GROUPS, XKFModel, fuse_states, group_moments, group_residuals
from semkine.xyz_recovery_study import RecoveryXYZModel

KMODES = ('fused', 'r', 'd', 'd_fused')


class XRFModel(XKFModel):
    MODEL_KEYS = XKFModel.MODEL_KEYS | frozenset({'XRF', 'XRF_KMODE', 'XRF_REFINE_ITERS', 'XRF_HYP_NOISE',
                                                   'XRF_HYP_GT_END', 'XRF_HYP_RAMP', 'XRF_D_INIT'})

    def __init__(self, cfg=None, num_classes=None):
        mc = (cfg or {}).get('MODEL', {})
        if mc.get('XRF') is not True:
            raise ValueError('XRFModel requires MODEL.XRF=true')
        if mc.get('XDL_SELF', False):
            raise ValueError('XRF trains per packet: MODEL.XDL_SELF must be false')
        cfg = dict(cfg)
        cfg['MODEL'] = dict(mc, XKF=True, XKF_KMODE='fused')
        super().__init__(cfg=cfg, num_classes=num_classes)
        self.xrf_kmode = str(mc.get('XRF_KMODE', 'fused'))
        if self.xrf_kmode not in KMODES:
            raise ValueError(f'XRF_KMODE must be one of {KMODES}')
        self.refine_iters = int(mc.get('XRF_REFINE_ITERS', 1))
        self.hyp_noise = tuple(float(v) for v in mc.get('XRF_HYP_NOISE', (0.01, 0.05, 0.05)))
        self.hyp_gt_end = float(mc.get('XRF_HYP_GT_END', 0.25))
        self.hyp_ramp = int(mc.get('XRF_HYP_RAMP', 2000))
        # calibrated filter gains (tools/xrf/calibrate.py writes them); 1 = no filter until calibrated
        self.register_buffer('xrf_gains', torch.ones(N_GROUPS))
        self.register_buffer('xrf_gains_d', torch.ones(N_GROUPS))
        init = mc.get('XRF_D_INIT')
        if init:
            self._init_direct_from_dense(init)

    # ------------------------------------------------------------------ warm start of the direct trunk
    def _init_direct_from_dense(self, path):
        """Trunk from a dense LNES tracker checkpoint (S39 `dt_dz_l3`): `rn.*` -> `direct.trunk.*` except fc; the
        adapter keeps the two LNES input channels, the render channels are dropped and the new planes start at zero."""
        sd = torch.load(path, map_location='cpu')['state_dict']
        own = self.direct.state_dict()
        with torch.no_grad():
            for k, v in sd.items():
                if k.startswith('rn.') and not k.startswith('rn.fc.'):
                    tk = 'trunk.' + k[3:]
                    if tk in own and own[tk].shape == v.shape:
                        own[tk].copy_(v)
            w = torch.zeros_like(own['adapter.weight'])
            w[:, :2] = sd['conv1.weight'][:, :2]
            own['adapter.weight'].copy_(w)
            own['adapter.bias'].copy_(sd['conv1.bias'])
        self.direct.load_state_dict(own)

    # ------------------------------------------------------------------ measurement chain
    def refine(self, batch, hyp):
        z = hyp
        for _ in range(self.refine_iters):
            z = RecoveryXYZModel.forward_packet(self, dataclasses.replace(batch, prev_state=z))
        return z

    def forward_packet(self, batch):
        prev = batch.prev_state.float()
        z_d, aux = self.direct_measurement(batch)
        if self.xrf_kmode == 'd':
            out = z_d
        elif self.xrf_kmode == 'd_fused':
            out = fuse_states(prev, z_d.detach(), self.xrf_gains_d)
        else:
            z_r = self.refine(batch, z_d.detach())
            out = z_r if self.xrf_kmode == 'r' else fuse_states(prev, z_r.detach().float(), self.xrf_gains)
        return torch.where((batch.counts <= 0)[:, None], prev, out.float())

    # ------------------------------------------------------------------ training
    def hyp_gt_probability(self, step):
        frac = min(1.0, max(0.0, float(step) / max(1, self.hyp_ramp)))
        return 1.0 + frac * (self.hyp_gt_end - 1.0)

    def _hypothesis(self, y, z_d, step):
        B = y.shape[0]
        g = torch.Generator(device='cpu').manual_seed(int(self.cfg.get('SEED', 3407)) * 100003 + int(step))
        use_gt = (torch.rand(B, generator=g) < self.hyp_gt_probability(step)).to(y.device)
        sig = torch.tensor([self.hyp_noise[0]] * 3 + [self.hyp_noise[1]] * 3 + [self.hyp_noise[2]] * 45,
                           device=y.device, dtype=torch.float32)
        noise = torch.randn(B, 51, generator=g).to(y.device) * sig
        base = torch.where(use_gt[:, None], y.float(), z_d.detach().float())
        return base + noise, use_gt

    def training_step(self, batch, batch_nb):
        main = batch['main'] if isinstance(batch, dict) else batch
        step = int(self.global_step)
        z_d, aux = self.direct_measurement(main)
        y = main.target.float()
        hyp, use_gt = self._hypothesis(y, z_d, step)
        z_r = self.refine(main, hyp)
        l_r, _ = self._compute_loss(z_r, main.target, main.betas)
        l_d, dparts = self._direct_loss(z_d, aux, main)
        loss = l_r.float().log10() + l_d.float().log10()
        with torch.no_grad():
            live = main.counts > 0
            er = group_residuals(z_r[live].float(), y[live]) if bool(live.any()) else None
            ed = group_residuals(z_d[live].float(), y[live]) if bool(live.any()) else None
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_loss_r', l_r.float().log10(), sync_dist=True)
        self.log('train_loss_d', l_d.float().log10(), sync_dist=True)
        self.log('train_hyp_gt_p', self.hyp_gt_probability(step), sync_dist=True)
        for name, v in dparts.items():
            self.log(f'train_{name}', v, sync_dist=True)
        if er is not None:
            mr, md = group_moments(er, er), group_moments(ed, ed)
            self.log('train_r_t_rms_mm', mr[:3].sum().sqrt() * 1000, sync_dist=True)
            self.log('train_d_t_rms_mm', md[:3].sum().sqrt() * 1000, sync_dist=True)
            self.log('train_r_root_rms_deg', mr[3].sqrt() * 57.2958, sync_dist=True)
            self.log('train_d_root_rms_deg', md[3].sqrt() * 57.2958, sync_dist=True)
        return loss


def steady_state_gain(Q, R):
    """Random-walk steady-state Kalman gain per group: P = (Q + sqrt(Q^2 + 4 Q R)) / 2, K = P / (P + R)."""
    Q, R = torch.as_tensor(Q, dtype=torch.float64), torch.as_tensor(R, dtype=torch.float64)
    P = (Q + torch.sqrt(Q * Q + 4 * Q * R)) / 2
    return (P / (P + R).clamp_min(1e-30)).clamp(0, 1).float()


__all__ = ['KMODES', 'XRFModel', 'steady_state_gain']
