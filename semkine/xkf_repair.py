"""Opt-in XKF repair without changing legacy source or checkpoint structure.

The XYZ association, observation bank and MANO graph remain the original ones.
Only the residual addition can be kept in FP32, and only the I training target
can become clean GT motion instead of correction toward the current GT state.
Measurement/fusion moments always score absolute states against current GT.
D auxiliary coefficients are relative to LOSS.LAMBDA_T: 0.5 adds half the
original translation-loss weight to that auxiliary's B-by-3 translation MSE.
"""
from __future__ import annotations

import math

import torch

from semkine.xkf import XKFModel


class XKFRepairModel(XKFModel):
    MODEL_KEYS = XKFModel.MODEL_KEYS | frozenset({
        'XKF_REPAIR_FP32_ADD', 'XKF_REPAIR_MOTION_TARGET',
        'XKF_REPAIR_D_AUX_2D_WEIGHT', 'XKF_REPAIR_D_AUX_3D_WEIGHT',
    })

    def __init__(self, cfg=None, num_classes=None):
        mc = (cfg or {}).get('MODEL', {})
        for key in ('XKF_REPAIR_FP32_ADD', 'XKF_REPAIR_MOTION_TARGET'):
            if not isinstance(mc.get(key, False), bool):
                raise ValueError(f'MODEL.{key} must be bool')
        for key in ('XKF_REPAIR_D_AUX_2D_WEIGHT', 'XKF_REPAIR_D_AUX_3D_WEIGHT'):
            value = mc.get(key, 0.)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'MODEL.{key} must be a finite nonnegative number')
        super().__init__(cfg=cfg, num_classes=num_classes)
        self.repair_fp32_add = mc.get('XKF_REPAIR_FP32_ADD', False)
        self.repair_motion_target = mc.get('XKF_REPAIR_MOTION_TARGET', False)
        self.repair_d_aux_2d_weight = float(mc.get('XKF_REPAIR_D_AUX_2D_WEIGHT', 0.))
        self.repair_d_aux_3d_weight = float(mc.get('XKF_REPAIR_D_AUX_3D_WEIGHT', 0.))
        if self.repair_fp32_add:
            # The direct call below deliberately reproduces the sole xyz_mesh
            # path. Refuse combinations whose base forward applies other heads.
            if (not self.xyz_mesh or not self.predict_delta or not self.zero_event_gate
                    or self.output_dim != 51 or self.prevpos_embed or self.prev_render
                    or self.active_head or self.root_meas != 'delta' or self.finger_meas != 'delta'
                    or self.root_compose != 'add'):
                raise ValueError('XKF FP32 addition requires the bare additive XYZ 51D delta path')
        if self.repair_motion_target and (self.loss_type != 'mse_51d' or self.root_loss != 'mse_aa'):
            raise ValueError('XKF clean motion_delta uses additive 51D MSE, not a rotation-state/FK loss')

    def incremental_prediction(self, batch):
        """I prediction; no target reads, no changed XYZ association/graph."""
        if not self.repair_fp32_add:
            # The trainer rebinds xyz_recovery_study.RecoveryXYZModel by name.
            # Resolve the original parent through the stable class MRO instead.
            return super(XKFModel, self).forward_packet(batch)
        prev = batch.prev_state
        delta = self._xyz_mesh_forward(batch, prev)
        with torch.autocast(device_type=prev.device.type, enabled=False):
            empty = (batch.counts <= 0)[:, None]
            delta = torch.where(empty, torch.zeros_like(delta), delta).float()
            prediction = prev.float() + delta
            return torch.where(empty, prev.float(), prediction)

    def branches(self, batch):
        if not self.repair_fp32_add:
            return super().branches(batch)
        x_i = self.incremental_prediction(batch)
        z_d, aux = self.direct_measurement(batch)
        return x_i, z_d, aux

    def incremental_loss(self, x_i, main, batch):
        """Pure additive motion supervision, independent of incoming state error.

        motion_delta is clean_augmented_GT_current - clean_augmented_GT_previous
        and lives outside the inference packet. In particular, a self-burn change
        to main.prev_state must never alter this label.
        """
        if not self.repair_motion_target:
            return self._compute_loss(x_i, main.target, main.betas)
        if not isinstance(batch, dict) or 'motion_delta' not in batch:
            raise ValueError('XKF motion-target repair requires batch["motion_delta"] from clean GT endpoints')
        target = batch['motion_delta']
        if not torch.is_tensor(target) or target.shape != x_i.shape or not target.is_floating_point():
            raise ValueError('motion_delta must be a floating tensor matching the (B, 51) prediction')
        with torch.autocast(device_type=x_i.device.type, enabled=False):
            delta = x_i.float() - main.prev_state.float()
            return self._compute_loss(delta, target.detach().to(x_i.device).float(), main.betas)

    def _direct_loss(self, z, aux, batch):
        loss, parts = super()._direct_loss(z, aux, batch)
        if self.repair_d_aux_2d_weight == 0. and self.repair_d_aux_3d_weight == 0.:
            return loss, parts
        from semkine.xkf_repair_losses import direct_aux_losses

        extra = direct_aux_losses(self, z, aux, batch)
        loss = loss + self.lambda_t * (self.repair_d_aux_2d_weight * extra['d_aux_2d']
                                      + self.repair_d_aux_3d_weight * extra['d_aux_3d']) / self.normalizer
        return loss, {**parts, **extra}

    def training_step(self, batch, batch_nb):
        if not self.repair_motion_target:
            # Includes the all-flags-off case, preserving exact legacy semantics.
            return super().training_step(batch, batch_nb)
        if not isinstance(batch, dict) or 'main' not in batch:
            raise ValueError('XKF repair trains on clip batches (main + prefix + motion_delta)')
        main, take = self._burn_main(batch)
        x_i, z_d, aux = self.branches(main)
        y = main.target.float()
        l_i, _ = self.incremental_loss(x_i, main, batch)
        l_d, dparts = self._direct_loss(z_d, aux, main)
        rows = self._moment_rows(main, take)
        # These are actual state-estimation errors, never errors to motion_delta.
        self._update_moments(x_i.detach().float(), z_d.detach().float(), main.prev_state.float(), y, rows)
        # A perfect static motion target can have exactly zero MSE.
        log_i = l_i.float().clamp_min(1e-12).log10()
        log_d = l_d.float().clamp_min(1e-12).log10()
        loss = log_i + log_d
        k = self.gains()
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_loss_i', log_i, sync_dist=True)
        self.log('train_loss_d', log_d, sync_dist=True)
        for name, value in dparts.items():
            self.log(f'train_{name}', value, sync_dist=True)
        self.log('train_k_t', k[:3].mean(), sync_dist=True)
        self.log('train_k_root', k[3], sync_dist=True)
        self.log('train_k_finger', k[4:].mean(), sync_dist=True)
        return loss


__all__ = ['XKFRepairModel']
