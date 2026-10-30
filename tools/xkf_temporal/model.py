"""Paired two-frame D supervision with an optional true-motion objective.

Both arms retain the original joint I+D training and no-grad fused K4 burn.
The previous prefix's original zero target stays untouched for that burn.
No temporal input, parameter, buffer, or deployment filter is added to D.
"""
from __future__ import annotations

import dataclasses
import math

import torch

from tools.xkf_joint.model import JointFrozenStatisticsModel, ORIGIN_GLOBAL, ORIGIN_PARAMETER

TARGET_GLOBAL = 2800
TARGET_PARAMETER = 2600


def physical_joint_motion_mse(pred_previous, pred_main, truth_previous, truth_main):
    """Mean B×21×3 squared displacement error in metres², with detached GT.

Both predicted endpoints retain gradients. A correct nonzero movement has zero
loss; this is not a penalty on the magnitude of predicted motion.
"""
    values = (pred_previous, pred_main, truth_previous, truth_main)
    if (any(not torch.is_tensor(value) for value in values)
            or pred_previous.ndim != 3 or pred_previous.shape[1:] != (21, 3)
            or pred_previous.shape[0] == 0
            or any(value.shape != pred_previous.shape for value in values)):
        raise ValueError('matching nonempty B-by-21-by-3 physical joints required')
    if any(not value.is_floating_point() or value.device != pred_previous.device for value in values):
        raise ValueError('physical joints must be floating tensors on the same device')
    with torch.autocast(device_type=pred_previous.device.type, enabled=False):
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise FloatingPointError('nonfinite temporal joints; no sample filtering')
        predicted_delta = pred_main.float() - pred_previous.float()
        target_delta = truth_main.detach().float() - truth_previous.detach().float()
        return (predicted_delta - target_delta).square().mean()


class TemporalDModel(JointFrozenStatisticsModel):
    MODEL_KEYS = JointFrozenStatisticsModel.MODEL_KEYS | frozenset({'XKF_TEMPORAL_D_WEIGHT'})

    def __init__(self, cfg=None, num_classes=None):
        weight = (cfg or {}).get('MODEL', {}).get('XKF_TEMPORAL_D_WEIGHT', 0.)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight not in (0., 1.):
            raise ValueError('XKF_TEMPORAL_D_WEIGHT must be the declared control0 or temporal1')
        super().__init__(cfg=cfg, num_classes=num_classes)
        self.temporal_d_weight = float(weight)

    @staticmethod
    def previous_loss_packet(batch):
        """A training-label view, never a mutation of the original burn packet."""
        if not isinstance(batch, dict) or len(batch.get('prefix', ())) != 3 or 'previous_gt' not in batch:
            raise ValueError('temporal D requires original K4 and clean previous_gt')
        previous, main = batch['prefix'][-1], batch['main']
        target = batch['previous_gt']
        if (not torch.is_tensor(target) or not target.is_floating_point()
                or target.shape != main.target.shape or target.shape != (main.batch_size, 51)
                or target.device != main.target.device or not bool(torch.isfinite(target).all())):
            raise ValueError('previous_gt must be clean finite B-by-51 training labels')
        if bool(previous.target.any()):
            raise ValueError('the original prefix target must remain zero for self-burn')
        if (previous.batch_size != main.batch_size
                or not torch.equal(previous.sequence_id, main.sequence_id)
                or not bool((main.t_end_us - previous.t_end_us == 50000).all())
                or not torch.equal(main.t_start_us, previous.t_end_us)
                or not torch.equal(previous.betas, main.betas)
                or not torch.equal(previous.camera_K, main.camera_K)
                or previous.lnes is None or main.lnes is None
                or previous.lnes_window_ms is None or main.lnes_window_ms is None):
            raise ValueError('two causal same-hand/camera packets exactly50ms apart required')
        return dataclasses.replace(previous, target=target.detach().float())

    def paired_direct_loss(self, z_main, aux_main, main, z_previous, aux_previous, previous):
        """Equal original frame losses plus the declared optional motion term."""
        main = dataclasses.replace(main, target=main.target.detach().float())
        previous = dataclasses.replace(previous, target=previous.target.detach().float())
        ld_main, parts_main = self._direct_loss(z_main, aux_main, main)
        ld_previous, parts_previous = self._direct_loss(z_previous, aux_previous, previous)
        with torch.autocast(device_type=z_main.device.type, enabled=False):
            # These are physical camera-space joints, including the MANO pivot
            # and DGP translation, rather than root-centred pose coordinates.
            joints_main = aux_main['joints0'].float() + z_main[:, None, :3].float()
            joints_previous = aux_previous['joints0'].float() + z_previous[:, None, :3].float()
            with torch.no_grad():
                _, truth_main = self._fk(main.target.detach().float(), main.betas.detach().float())
                _, truth_previous = self._fk(previous.target.detach().float(), previous.betas.detach().float())
            motion_mse = physical_joint_motion_mse(joints_previous, joints_main, truth_previous, truth_main)
            contribution = .5 * self.temporal_d_weight * self.lambda_t * motion_mse / self.normalizer
            loss = .5 * (ld_main.float() + ld_previous.float()) + contribution
        parts = {name: .5 * (value + parts_previous[name]) for name, value in parts_main.items()}
        parts.update(d_main_normalized=ld_main, d_previous_normalized=ld_previous,
                     d_temporal_joint_mse=motion_mse, d_temporal_normalized=contribution)
        return loss, parts

    def training_step(self, batch, batch_idx):
        previous_loss_view = self.previous_loss_packet(batch)
        main, _ = self._burn_main(batch)
        incremental, direct, auxiliary = self.branches(main)
        # The actual D forward still sees the original zero-target prefix.
        previous_direct, previous_aux = self.direct_measurement(batch['prefix'][-1])
        li, _ = self.incremental_loss(incremental, main, batch)
        ld, parts = self.paired_direct_loss(direct, auxiliary, main,
                                          previous_direct, previous_aux, previous_loss_view)
        log_i = li.float().clamp_min(1e-12).log10()
        log_d = ld.float().clamp_min(1e-12).log10()
        loss = log_i + log_d
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('nonfinite paired temporal joint loss')
        for name, value in [('loss', loss), ('loss_i', log_i), ('loss_d', log_d), *parts.items()]:
            self.log('train_' + name, value, sync_dist=True, batch_size=main.batch_size)
        return loss

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        # The inherited class also prepares a possible full run; this experiment
        # is exclusively the fixed paired200 debug endpoint.
        checkpoint['xkf_joint_continuation'].update(target_global_step=TARGET_GLOBAL,
                                                   target_parameter_steps=TARGET_PARAMETER)
        clock = ORIGIN_PARAMETER + int(self.global_step) - ORIGIN_GLOBAL
        checkpoint['xkf_temporal_continuation'] = dict(
            formal=False, automatic_full_training=False, start_step=ORIGIN_GLOBAL,
            global_step=int(self.global_step), i_adam_step=clock, d_adam_step=clock,
            target_global_step=TARGET_GLOBAL, target_parameter_steps=TARGET_PARAMETER,
            temporal_weight=self.temporal_d_weight, two_frame_supervision_weights=[.5, .5],
            physical_motion_coefficient='.5 * temporal_weight * LAMBDA_T / NORMALIZER',
            previous_target='clean GT at last prefix endpoint; original prefix remains zero',
            inference='unchanged frame-independent D; original joint self-burn')


__all__ = ['TemporalDModel', 'physical_joint_motion_mse', 'TARGET_GLOBAL', 'TARGET_PARAMETER']
