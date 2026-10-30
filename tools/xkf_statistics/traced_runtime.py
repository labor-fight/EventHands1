"""Inference-only trace of the same fixed paired-statistics policy."""
from semkine.xkf_paired_statistics import PairedStatisticsPolicy, XKFPairedStatisticsModel
from semkine.xkf_motion_gate import interpolate_motion


class TracedPolicy(PairedStatisticsPolicy):
    def __init__(self, spec):
        super().__init__(spec)
        self.capture = False
        self.trace = []

    def fuse(self, motion, direct, aux, previous, counts, base):
        result = super().fuse(motion, direct, aux, previous, counts, base)
        if self.capture:
            gate = (counts.float() / 1000.).clamp(0., 1.)
            self.trace.append({k: v.detach().clone() for k, v in dict(
                incoming=previous, motion=motion,
                gated_motion=interpolate_motion(previous, motion, gate),
                direct=result[1], effective_gains=result[2], motion_gate=gate).items()})
        return result


class TracedModel(XKFPairedStatisticsModel):
    def configure_fusion(self, spec=None):
        self.fusion_policy = None if spec is None else TracedPolicy(spec)
        return self
