"""Batch-stable inference for explicit XKF motion-gate policies.

Combines the separately verified count-gate policy with genuine per-sample
forward calls. No padding, cached D prediction, new weights or checkpoint keys.
Single-sample deployment retains the same arithmetic as XKFMotionGateModel.
"""
from semkine.xkf_motion_gate import MotionCountFusionPolicy
from semkine.xkf_stable_geometry import XKFStableGeometryModel


class XKFStableMotionGateModel(XKFStableGeometryModel):
    def __init__(self, cfg=None, num_classes=None, fusion_spec=None,
                 stability_mode='per_sample_model'):
        super().__init__(cfg=cfg, num_classes=num_classes, fusion_spec=fusion_spec,
                         stability_mode=stability_mode)

    def configure_fusion(self, spec=None):
        self.fusion_policy = None if spec is None else MotionCountFusionPolicy(spec)
        return self


__all__ = ['XKFStableMotionGateModel']
