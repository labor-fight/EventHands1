"""Joint I+D continuation with the current raw/frozen-statistics recipe.

This class retains the original fused self-burn. Stable centroid/I1000 is a
separately specified deployment adapter, never silently used during training.
No parameters or buffers are added to the original 96-parameter structure.
"""
from __future__ import annotations
import copy
import torch
from semkine.xkf_repair import XKFRepairModel

ORIGIN_GLOBAL = 2600
ORIGIN_PARAMETER = 2400
FULL_GLOBAL = 6200
RATE = .0004
POLICY = dict(gain_scale=[.5, .25, .5], count_gate='ramp1000',
              translation_reference='joint_centroid', motion_count_gate='ramp1000')


class JointFrozenStatisticsModel(XKFRepairModel):
    def __init__(self, cfg=None, num_classes=None):
        super().__init__(cfg=cfg, num_classes=num_classes)
        if not self.repair_fp32_add or not self.repair_motion_target:
            raise ValueError('joint continuation requires FP32 addition and true motion labels')
        if (self.repair_d_aux_2d_weight, self.repair_d_aux_3d_weight) != (.5, .5):
            raise ValueError('keep both D auxiliary weights at .5')
        if self.ema_decay != 0 or self.weight_decay != 0 or not self.xdl_self or self.xdl_burn_mode != 'full':
            raise ValueError('keep original Adam, disabled weight EMA and full fused self-burn')
        track, aug = self.cfg['TRACK'], self.cfg['AUG']
        if track.get('PREV_NOISE_MODE') != 'gaussian' or tuple(track.get(k) for k in ('PREV_NOISE_T', 'PREV_NOISE_R', 'PREV_NOISE_POSE')) != (.005, .05, .05):
            raise ValueError('keep original small Gaussian initialization')
        if aug.get('DOMRAND', {}).get('ENABLED') or aug.get('POLARITY_FLIP') or aug.get('PIXEL_POLARITY_SWAP'):
            raise ValueError('joint recipe requires the declared raw events')
        if not self.cfg.get('XKF_DATA', {}).get('PREFIX_LNES'):
            raise ValueError('original full prefix LNES must remain enabled')
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        self.parent_incremental_continuation = None
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.parameters()), lr=RATE)
        warmup = int(self.cfg['TRAIN'].get('WARMUP_STEPS', 500))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: float(step + 1) / warmup if step < warmup else 1.)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def _update_moments(self, *unused, **unused_kwargs):
        raise RuntimeError('joint frozen-statistics recipe must never update residual moments')

    def training_step(self, batch, batch_idx):
        main, _ = self._burn_main(batch)
        incremental, direct, auxiliary = self.branches(main)
        li, _ = self.incremental_loss(incremental, main, batch)
        ld, parts = self._direct_loss(direct, auxiliary, main)
        log_i = li.float().clamp_min(1e-12).log10()
        log_d = ld.float().clamp_min(1e-12).log10()
        loss = log_i + log_d
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('nonfinite joint loss')
        for name, value in [('loss', loss), ('loss_i', log_i), ('loss_d', log_d), *parts.items()]:
            self.log('train_' + name, value, sync_dist=True, batch_size=main.batch_size)
        return loss

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint['xkf_joint_continuation'] = {
            'origin_global_step': ORIGIN_GLOBAL, 'origin_parameter_step': ORIGIN_PARAMETER,
            'global_step': int(self.global_step),
            'i_adam_step': ORIGIN_PARAMETER + int(self.global_step) - ORIGIN_GLOBAL,
            'd_adam_step': ORIGIN_PARAMETER + int(self.global_step) - ORIGIN_GLOBAL,
            'target_global_step': FULL_GLOBAL, 'target_parameter_steps': 6000,
            'is_original_global6000_protocol': False,
            'all_bn_and_xkf_statistics_frozen': True,
            'self_burn': 'original fused no-grad K4, without external deployment policy',
            'deployment_policy': copy.deepcopy(POLICY),
            'parent_incremental_continuation': copy.deepcopy(self.parent_incremental_continuation),
        }


__all__ = ['JointFrozenStatisticsModel', 'ORIGIN_GLOBAL', 'ORIGIN_PARAMETER', 'FULL_GLOBAL', 'RATE', 'POLICY']
