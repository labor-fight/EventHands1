"""External inference-only XKF fusion candidates; no new parameters or buffers.

No policy means the exact frozen repair path. Policies are deliberately explicit
research interventions, not claims that their gains remain statistically optimal.
The count gate uses the current raw W50 count and scales only the D correction.
"""
from __future__ import annotations

import copy
import math

import torch

from semkine.lie import so3_exp, so3_log
from semkine.xkf import fuse_states, nearest_aa
from semkine.xkf_repair import XKFRepairModel


def _finite_number(value, name, lower=0., strict=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or (value <= lower if strict else value < lower)):
        raise ValueError(f'{name} must be finite and {">" if strict else ">="} {lower}')
    return float(value)


def validate_correction(spec):
    if spec is None:
        return None
    spec = copy.deepcopy(spec)
    allowed = {'kind', 'matrix', 'bias', 'scale', 'offset', 'root_left_rotation'}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError('direct correction may only contain declared target-free numeric coefficients')
    kind = spec.get('kind')
    if kind == 'translation_affine':
        matrix, bias = torch.as_tensor(spec['matrix']), torch.as_tensor(spec['bias'])
        if matrix.shape != (3, 3) or bias.shape != (3,) or not torch.isfinite(matrix).all() or not torch.isfinite(bias).all():
            raise ValueError('translation affine requires finite 3x3 matrix and length3 bias in metres')
    elif kind in ('wrist_depth_affine', 'wrist_inverse_depth_affine'):
        _finite_number(spec['scale'], 'depth scale', strict=True)
        if isinstance(spec['offset'], bool) or not isinstance(spec['offset'], (int, float)) or not math.isfinite(spec['offset']):
            raise ValueError('depth offset must be finite')
    elif kind != 'identity':
        raise ValueError(f'unsupported direct correction {kind!r}')
    if 'root_left_rotation' in spec:
        rotation = torch.as_tensor(spec['root_left_rotation'], dtype=torch.float64)
        if (rotation.shape != (3, 3) or not torch.isfinite(rotation).all()
                or not torch.allclose(rotation.T @ rotation, torch.eye(3, dtype=rotation.dtype), atol=1e-5, rtol=1e-5)
                or abs(float(torch.det(rotation)) - 1.) > 1e-5):
            raise ValueError('root_left_rotation must be a proper SO(3) matrix')
    return spec


def correct_direct(z_d, aux, spec=None):
    """Only D outputs and fixed coefficients enter: no previous state, GT or ID."""
    if spec is None:
        return z_d
    kind = spec['kind']
    result = z_d.float().clone()
    if kind == 'translation_affine':
        matrix, bias = (z_d.new_tensor(spec[key]).float() for key in ('matrix', 'bias'))
        result[:, :3] = z_d[:, :3].float() @ matrix.T + bias
    elif kind in ('wrist_depth_affine', 'wrist_inverse_depth_affine'):
        wrist0 = aux['joints0'][:, 0].float()
        wrist = z_d[:, :3].float() + wrist0
        depth = wrist[:, 2:3]
        if not bool((depth > 1e-6).all()):
            raise FloatingPointError('D wrist depth is nonpositive; calibration cannot silently clamp it')
        if kind == 'wrist_depth_affine':
            new_depth = float(spec['scale']) * depth + float(spec['offset'])
        else:
            inverse = float(spec['scale']) / depth + float(spec['offset'])
            if not bool((inverse > 1e-6).all()):
                raise FloatingPointError('calibrated inverse depth is nonpositive')
            new_depth = inverse.reciprocal()
        if not bool((new_depth > 1e-6).all()):
            raise FloatingPointError('calibrated wrist depth is nonpositive')
        result[:, :3] = wrist * (new_depth / depth) - wrist0
    if 'root_left_rotation' in spec:
        correction = z_d.new_tensor(spec['root_left_rotation']).float()
        result[:, 3:6] = nearest_aa(so3_log(correction @ so3_exp(z_d[:, 3:6].float())), z_d[:, 3:6].float())
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError('nonfinite calibrated direct state')
    return result


class FusionPolicy:
    def __init__(self, spec):
        allowed = {'id', 'gain_scale', 'count_gate', 'direct_correction', 'temperature_multiplier'}
        if not isinstance(spec, dict) or set(spec) - allowed:
            raise ValueError('unknown fusion policy key')
        self.spec = copy.deepcopy(spec)
        self.name = spec.get('id', 'external')
        if not isinstance(self.name, str) or not self.name or any(x not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for x in self.name):
            raise ValueError('candidate id must use lowercase letters, digits, underscore or hyphen')
        scales = spec.get('gain_scale', [1., 1., 1.])
        if len(scales) != 3:
            raise ValueError('gain_scale requires [translation, root, fingers]')
        self.scales = tuple(_finite_number(x, 'gain scale') for x in scales)
        self.temperature_multiplier = _finite_number(spec.get('temperature_multiplier', 1.), 'temperature multiplier', strict=True)
        self.count_gate = spec.get('count_gate')
        if self.count_gate not in (None, 'ramp1000'):
            raise ValueError('count_gate must be null or ramp1000 (latest raw W50 count)')
        self.correction = validate_correction(spec.get('direct_correction'))

    def is_identity(self):
        return self.scales == (1., 1., 1.) and self.count_gate is None and self.correction is None and self.temperature_multiplier == 1.

    def gains(self, base, counts):
        scales = base.new_tensor([self.scales[0]] * 3 + [self.scales[1]] + [self.scales[2]] * 15)
        gains = (base.float() * scales).clamp(0., 1.).expand(len(counts), -1)
        if self.count_gate == 'ramp1000':
            gains = gains * (counts.float() / 1000.).clamp(0., 1.)[:, None]
        return gains

    def fuse(self, x_i, z_d, aux, prev, counts, base_gains):
        direct = correct_direct(z_d, aux, self.correction)
        gains = self.gains(base_gains, counts)
        result = fuse_states(x_i.float(), direct.float(), gains)
        result = torch.where((counts <= 0)[:, None], prev.float(), result)
        return result, direct, gains


class XKFFusionModel(XKFRepairModel):
    """Checkpoint-compatible deployment adapter with explicitly supplied policy."""
    def __init__(self, cfg=None, num_classes=None, fusion_spec=None):
        super().__init__(cfg=cfg, num_classes=num_classes)
        self.fusion_policy = None if fusion_spec is None else FusionPolicy(fusion_spec)

    def configure_fusion(self, spec=None):
        self.fusion_policy = None if spec is None else FusionPolicy(spec)
        return self

    @torch.no_grad()
    def measure_direct(self, batch, temperature_multiplier=1.):
        multiplier = _finite_number(temperature_multiplier, 'temperature multiplier', strict=True)
        if multiplier == 1.:
            return super().direct_measurement(batch)
        # log_beta is inverse softargmax temperature; change only during this
        # inference call, then restore bit-for-bit even if inference raises.
        original = self.direct.log_beta.detach().clone()
        try:
            self.direct.log_beta.add_(math.log(multiplier))
            return super().direct_measurement(batch)
        finally:
            self.direct.log_beta.copy_(original)

    def forward_packet(self, batch):
        if self.fusion_policy is None:
            return super().forward_packet(batch)
        if self.kmode != 'fused':
            raise ValueError('an external fusion policy requires kmode=fused')
        policy = self.fusion_policy
        x_i = self.incremental_prediction(batch)
        z_d, aux = self.measure_direct(batch, policy.temperature_multiplier)
        result, _, _ = policy.fuse(x_i.detach(), z_d.detach(), aux, batch.prev_state, batch.counts, self.gains('fused'))
        return result


__all__ = ['FusionPolicy', 'XKFFusionModel', 'correct_direct', 'validate_correction']
