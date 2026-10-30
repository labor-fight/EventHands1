"""Separate-pass Conv2d/Linear operation counts, not service timing.

Masked dense convolutions still execute dense operators. The optional output-
site occupancy estimate is explicitly a theoretical bound, never substituted
for the operator count or claimed as SparseExecutor hardware execution.
"""
from __future__ import annotations
import functools
import torch
from torch import nn


class NNOperations:
    def __init__(self,runner):self.runner=runner;self.rows={};self.handles=[];self.restore=None
    def row(self,name):return self.rows.setdefault((self.runner.phase_category,name),dict(calls=0,dense_macs=0,occupancy_macs=0,occupancy_calls=0))
    def __enter__(self):
        model=self.runner.model;base=getattr(model,'base',model);names={id(m):name for name,m in model.named_modules()}
        self.adapter_name=names[id(base.conv1)];self.rn_prefix=names[id(base.rn)]
        modules=dict(model.named_modules())
        for name,module in modules.items():
            if isinstance(module,(nn.Conv2d,nn.Linear)):
                def executed(m,args,out,key=name):
                    if isinstance(m,nn.Conv2d):macs=int(out.numel())*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
                    else:macs=int(out.numel())*m.in_features
                    r=self.row(key);r['calls']+=1;r['dense_macs']+=macs
                    # Only RN masked convolutions and the outer adapter need
                    # occupancy masks. Local patches/MLPs are dense as called.
                    if not isinstance(m,nn.Conv2d) or (not key.startswith(self.rn_prefix+'.') and key!=self.adapter_name):
                        r['occupancy_macs']+=macs;r['occupancy_calls']+=1
                self.handles.append(module.register_forward_hook(executed))
            if type(module).__name__=='MaskedBN2d':
                suffix=name.rsplit('.',1)[-1];prefix=name.rsplit('.',1)[0]
                conv_name=prefix+'.'+dict(bn1='conv1',bn2='conv2',down_bn='down_conv')[suffix]
                conv=modules[conv_name]
                def occupancy(m,args,out,key=conv_name,c=conv):
                    active=int(args[1].sum().item());r=self.row(key)
                    r['occupancy_macs']+=active*c.out_channels*(c.in_channels//c.groups)*c.kernel_size[0]*c.kernel_size[1]
                    r['occupancy_calls']+=1
                self.handles.append(module.register_forward_hook(occupancy))
        old=base.rn.pooled;self.restore=(base.rn,old);counter=self
        @functools.wraps(old)
        def pooled(*a,**kw):
            mask=a[1] if len(a)>1 else kw['m0'];c=base.conv1;active=int(mask.sum().item());r=counter.row(counter.adapter_name)
            r['occupancy_macs']+=active*c.out_channels*(c.in_channels//c.groups)*c.kernel_size[0]*c.kernel_size[1]
            r['occupancy_calls']+=1
            return old(*a,**kw)
        base.rn.pooled=pooled;return self
    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        if self.restore is not None:self.restore[0].pooled=self.restore[1]
        return False
    def report(self,successful_queries,*,complete=True):
        report={}
        for phase,n in successful_queries.items():
            rows={name:r for (category,name),r in self.rows.items() if category==phase}
            dense=sum(r['dense_macs'] for r in rows.values());ideal=sum(r['occupancy_macs'] for r in rows.values())
            mask_complete=all(r['calls']==r['occupancy_calls'] for r in rows.values())
            local=sum(r['dense_macs'] for name,r in rows.items() if name.startswith(('local_encoder.','local_head.')))
            report[phase]=dict(successful_outputs=n,actual_dense_operator_equivalent_MACs_total=dense,
                actual_dense_operator_equivalent_FLOPs_total=2*dense,
                actual_dense_operator_equivalent_NN_FLOPs_per_output=2*dense/n if complete and n else None,
                occupancy_output_site_theoretical_MACs_total=ideal if mask_complete else None,
                occupancy_output_site_theoretical_NN_FLOPs_per_output=2*ideal/n if complete and mask_complete and n else None,
                local_patch_encoder_and_head_FLOPs_per_output=2*local/n if complete and n else None,modules=rows)
        return dict(schema='s43_nn_executed_shape_count_v1',phases=report,
            convention='2FLOPs/MAC, actual executed Conv2d/Linear outputshapes. Dense maskedconv charges entire outputgrid includingzero sites. Direct-convolution-equivalent arithmetic, not lowlevelcuDNN/Winograd hardwareinstruction measurement.',
            exclusions='biasadds/BN/activation/pool/mask/add/compose, MANO FK/render, CPUgeometry, sampling, input/log/readout/transfer/source maintenance not inNN MAC count; their real cost is inunprofiledservicewall',
            occupancy_scope='hypothetical output-active-site pruning, all kernel taps charged. Not actualhardwareexecution, not SparseExecutor valid-inputtap count; local720patch conv/MLPs staydense. Never use occupancy value as mainexecutedFLOPs.',
            timing='separate untimed countingpass, passivehooks+maskreductions can synchronizeGPU; never use thispass for latency',complete=complete)
