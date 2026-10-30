"""Opt-in evidence gating of I motion or of a whole geometric fusion update.

Only current raw W50 counts enter. No targets, sequence IDs or fitted thresholds.
The unchanged D measurement and the existing root innovation clip are retained.
"""
from __future__ import annotations
import copy
import torch
from semkine.lie import so3_exp, so3_log
from semkine.xkf import nearest_aa
from semkine.xkf_centroid import CentroidFusionPolicy, XKFCentroidModel


def interpolate_motion(previous, proposal, gate):
    """Linear t/fingers and un-clipped SO(3) root; exact zero/one endpoints."""
    previous, proposal = previous.float(), proposal.float()
    gate = gate.float().reshape(-1)
    if len(gate) != len(previous) or not bool(torch.isfinite(gate).all()) or bool(((gate < 0) | (gate > 1)).any()):
        raise ValueError('one finite [0,1] motion gate per state is required')
    if bool((gate == 1).all()):
        return proposal
    if bool((gate == 0).all()):
        return previous
    result = previous + gate[:, None] * (proposal - previous)
    rotations = so3_exp(previous[:, 3:6])
    innovation = so3_log(rotations.transpose(-1, -2) @ so3_exp(proposal[:, 3:6]))
    result[:, 3:6] = nearest_aa(so3_log(rotations @ so3_exp(gate[:, None] * innovation)), previous[:, 3:6])
    return torch.where((gate == 0)[:, None], previous,
                       torch.where((gate == 1)[:, None], proposal, result))


class MotionCountFusionPolicy(CentroidFusionPolicy):
    def __init__(self, spec):
        value = copy.deepcopy(spec)
        self.motion_count_gate = value.pop('motion_count_gate', None)
        self.update_count_gate = value.pop('update_count_gate', None)
        if self.motion_count_gate not in (None, 'ramp1000') or self.update_count_gate not in (None, 'ramp1000'):
            raise ValueError('motion/update count gate must be None or ramp1000')
        if self.motion_count_gate and self.update_count_gate:
            raise ValueError('select motion or whole-update gating, not both')
        if self.update_count_gate and value.get('count_gate') is not None:
            raise ValueError('whole-update gating requires count_gate=None to avoid double gating D')
        super().__init__(value)
        for key in ('motion_count_gate', 'update_count_gate'):
            if key in spec:
                self.spec[key] = spec[key]

    def is_identity(self):
        return not self.motion_count_gate and not self.update_count_gate and super().is_identity()

    def fuse(self, x_i, z_d, aux, prev, counts, base_gains):
        if not self.motion_count_gate and not self.update_count_gate:
            return super().fuse(x_i, z_d, aux, prev, counts, base_gains)
        if bool((counts < 0).any()) or not bool(torch.isfinite(counts).all()):
            raise ValueError('raw W50 event counts must be finite nonnegative')
        gate = (counts.float() / 1000.).clamp(0., 1.)
        motion = interpolate_motion(prev, x_i, gate) if self.motion_count_gate else x_i
        fused, direct, gains = super().fuse(motion, z_d, aux, prev, counts, base_gains)
        if self.update_count_gate:
            fused = interpolate_motion(prev, fused, gate)
        return torch.where((counts == 0)[:, None], prev.float(), fused), direct, gains


class XKFMotionGateModel(XKFCentroidModel):
    def configure_fusion(self, spec=None):
        self.fusion_policy = None if spec is None else MotionCountFusionPolicy(spec)
        return self
