"""Training-calibrated scalar I count ramp; existing D ramp and geometry fixed."""
from __future__ import annotations
import copy,math
import torch
from semkine.xkf_motion_gate import MotionCountFusionPolicy,interpolate_motion
from semkine.xkf_centroid import CentroidFusionPolicy
from semkine.xkf_motion_gate_runtime import XKFStableMotionGateModel

class CalibratedMotionCountPolicy(MotionCountFusionPolicy):
    def __init__(self,spec):
        value=copy.deepcopy(spec)
        threshold=value.pop('motion_count_threshold',1000.)
        if isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or threshold<=0:
            raise ValueError('motion_count_threshold must be positive finite')
        if value.get('motion_count_gate')!='ramp1000' or value.get('update_count_gate') is not None:
            raise ValueError('calibrated scalar requires I-only count gate')
        if value.get('count_gate')!='ramp1000' or value.get('gain_scale')!=[.5,.25,.5] or value.get('translation_reference')!='joint_centroid':
            raise ValueError('calibrated I policy keeps declared D1000/centroid/[.5,.25,.5]')
        if value.get('direct_correction') is not None or value.get('temperature_multiplier',1.)!=1.:
            raise ValueError('no D calibration or temperature change in scalar I fit')
        super().__init__(value)
        self.motion_count_threshold=float(threshold)
        if 'motion_count_threshold' in spec:self.spec['motion_count_threshold']=threshold
        self.capture=False;self.trace=[]

    def fuse(self,x_i,z_d,aux,prev,counts,base_gains):
        if self.motion_count_threshold==1000.:
            # Exact existing arithmetic, including endpoints and float divisions.
            result=super().fuse(x_i,z_d,aux,prev,counts,base_gains)
        else:
            if not bool(torch.isfinite(counts).all()) or bool((counts<0).any()):
                raise ValueError('raw W50 counts must be finite nonnegative')
            g=(counts.float()/self.motion_count_threshold).clamp(0.,1.)
            motion=interpolate_motion(prev,x_i,g)
            result=CentroidFusionPolicy.fuse(self,motion,z_d,aux,prev,counts,base_gains)
        if self.capture:
            g=(counts.float()/self.motion_count_threshold).clamp(0.,1.)
            self.trace.append(dict(incoming=prev.detach().clone(),motion=x_i.detach().clone(),
                gated_motion=interpolate_motion(prev,x_i,g).detach().clone(),direct=result[1].detach().clone(),
                effective_gains=result[2].detach().clone(),motion_gate=g.detach().clone()))
        return result

class XKFCalibratedMotionModel(XKFStableMotionGateModel):
    def configure_fusion(self,spec=None):
        self.fusion_policy=None if spec is None else CalibratedMotionCountPolicy(spec)
        return self
