"""Opt-in joint-centroid translation coupling for XKF inference.

The SO(3)/finger update and all effective gains are unchanged. Translation mixes
the two predicted 21-joint centroids, then subtracts the fused pose's centroid.
No target, camera, sequence identifier or previous state enters the geometry.
"""
from __future__ import annotations

import copy

import torch

from semkine.xkf import fuse_states
from semkine.xkf_fusion import FusionPolicy, XKFFusionModel, correct_direct


def joint_centroid_offsets(states, betas, fk):
    """FK mean of all 21 joints with state translation exactly zero (metres)."""
    if states.ndim != 2 or states.shape[1] != 51:
        raise ValueError('centroid states must have shape B x 51')
    if betas.ndim != 2 or len(betas) not in (1, len(states)):
        raise ValueError('centroid betas must have one or B rows')
    local = states.float().clone()
    local[:, :3] = 0.
    _, joints = fk(local, betas.float().expand(len(states), -1))
    if joints.shape != (len(states), 21, 3) or not bool(torch.isfinite(joints).all()):
        raise ValueError('centroid FK must produce finite B x 21 x 3 joints')
    return joints.mean(dim=1)


def fuse_joint_centroids(x_i, z_d, gains, betas, fk):
    """Return the original fused pose with geometrically coupled translation.

Kt=0/1 anchors the fused centroid to I/D even if pose gains differ. All-zero
gains preserve I exactly. All-one gains recover D up to the unchanged root
innovation clipping and normal floating point SO(3) conversion.
"""
    fused = fuse_states(x_i, z_d, gains)
    gains = gains.expand(len(x_i), -1) if gains.ndim == 1 else gains
    beta = betas.float().expand(len(x_i), -1)
    offsets = joint_centroid_offsets(torch.cat((x_i, z_d, fused)), beta.repeat(3, 1), fk)
    ci, cd, cf = offsets.chunk(3)
    kt = gains[:, :3].float()
    fused[:, :3] = (1. - kt) * (x_i[:, :3].float() + ci) + kt * (z_d[:, :3].float() + cd) - cf
    # Preserve the exact no-correction endpoint, avoiding an FK add/subtract.
    return torch.where((gains == 0).all(1, keepdim=True), x_i.float(), fused)


class CentroidFusionPolicy(FusionPolicy):
    """Explicit `translation_reference` switch; default remains original state t."""
    def __init__(self, spec):
        value = copy.deepcopy(spec)
        self.translation_reference = value.pop('translation_reference', 'state_t')
        if self.translation_reference not in ('state_t', 'joint_centroid'):
            raise ValueError('translation_reference must be state_t or joint_centroid')
        super().__init__(value)
        if 'translation_reference' in spec:
            self.spec['translation_reference'] = self.translation_reference
        self._geometry = None

    def bind_geometry(self, fk, betas):
        self._geometry = (fk, betas)
        return self

    def is_identity(self):
        return self.translation_reference == 'state_t' and super().is_identity()

    def fuse(self, x_i, z_d, aux, prev, counts, base_gains):
        if self.translation_reference == 'state_t':
            return super().fuse(x_i, z_d, aux, prev, counts, base_gains)
        direct = correct_direct(z_d, aux, self.correction)
        gains = self.gains(base_gains, counts)
        if not bool((counts > 0).any()):
            return prev.float(), direct, gains
        if self._geometry is None:
            raise RuntimeError('centroid policy requires explicit FK and ordinary packet betas')
        fk, betas = self._geometry
        result = fuse_joint_centroids(x_i.float(), direct.float(), gains, betas, fk)
        result = torch.where((counts <= 0)[:, None], prev.float(), result)
        return result, direct, gains


class XKFCentroidModel(XKFFusionModel):
    """No new tensors/checkpoint keys; disabled and state-t paths are unchanged."""
    def __init__(self, cfg=None, num_classes=None, fusion_spec=None):
        super().__init__(cfg=cfg, num_classes=num_classes)
        self.configure_fusion(fusion_spec)

    def configure_fusion(self, spec=None):
        self.fusion_policy = None if spec is None else CentroidFusionPolicy(spec)
        return self

    def forward_packet(self, batch):
        if self.fusion_policy is not None:
            self.fusion_policy.bind_geometry(self._fk, batch.betas)
        return super().forward_packet(batch)


__all__ = ['joint_centroid_offsets', 'fuse_joint_centroids', 'CentroidFusionPolicy', 'XKFCentroidModel']
