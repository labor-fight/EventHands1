#!/usr/bin/env python3
"""Batch-one full-row adapter with explicit centroid policy and actual-policy MAC.

The existing row producer retains clocks, provenance, scoring and latency.
THOP retains the repository's standard-layer lower-bound convention: functional
MANO/SO(3)/scatter arithmetic is not counted. The actual configured centroid
path is executed and its three extra FK pose rows are independently audited.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'model')]
import torch
from mano_layer import ManoLayer
from semkine.xkf_centroid import CentroidFusionPolicy,XKFCentroidModel
from tools.xkf_fusion import make_row as M


@torch.no_grad()
def profile_policy_macs(cfg,evaluated_model,audit_path=None):
    from thop import profile
    from tools.xc.make_row import fixed_mac_fixture
    if evaluated_model.fusion_policy is None:
        raise ValueError('centroid row MAC must profile the actual configured policy')
    spec=copy.deepcopy(evaluated_model.fusion_policy.spec)
    model=XKFCentroidModel(cfg=copy.deepcopy(cfg)).float().eval()
    model.load_state_dict({k:v.detach().cpu() for k,v in evaluated_model.state_dict().items()},strict=True)
    model.configure_fusion(spec)
    for module in model.modules():
        if isinstance(module,ManoLayer):module.fast_chain=True
    packet,_=fixed_mac_fixture()
    packet.lnes=torch.zeros(1,model.render_h,model.render_w,2)
    packet.lnes_window_ms=torch.tensor([200])
    calls=[];original_fk=model._fk
    def audited_fk(state,beta):
        calls.append(int(len(state)))
        return original_fk(state,beta)
    model._fk=audited_fk
    # Bind a dedicated FK callback only for the candidate's centroid geometry,
    # so its extra pose rows are counted separately from I/D internal FK.
    centroid_rows=[];centroid_profiler=[];original_bind=model.fusion_policy.bind_geometry
    def bind_geometry(fk,beta):
        def centroid_fk(state,betas):
            centroid_rows.append(int(len(state)))
            # Separate descriptive matrix-MAC estimate; never add this to the
            # historical THOP table column. This CPU profiler is outside the
            # producer's latency measurement.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                        with_flops=True,record_shapes=True) as measured:
                result=fk(state,betas)
            rows=[{'operator':event.key,'calls':event.count,'flops':int(event.flops),
                   'input_shapes':event.input_shapes} for event in measured.key_averages(group_by_input_shape=True)
                  if event.flops]
            matrix_ops={'aten::mm','aten::bmm','aten::addmm','aten::baddbmm'}
            matrix_flops=sum(row['flops'] for row in rows if row['operator'] in matrix_ops)
            centroid_profiler.append({'pose_rows':int(len(state)),'matrix_flops':matrix_flops,
                                      'matrix_macs_estimate':matrix_flops/2.,'reported_nonzero_flop_operators':rows})
            return result
        return original_bind(centroid_fk,beta)
    model.fusion_policy.bind_geometry=bind_geometry
    class Wrap(torch.nn.Module):
        def __init__(self,inner):super().__init__();self.inner=inner
        def forward(self,value):return self.inner.forward_packet(value)
    value,_=profile(Wrap(model),inputs=(packet,),verbose=False)
    expected=3 if spec.get('translation_reference','state_t')=='joint_centroid' else 0
    if sum(centroid_rows)!=expected:
        raise RuntimeError(f'MAC fixture failed to execute centroid geometry: {centroid_rows}, expected {expected} pose rows')
    params=sum(p.numel() for p in model.parameters())
    audit={'policy':spec,'n_events':4096,'batch_size':1,'standard_layer_macs':float(value),
           'params':params,'all_fk_call_pose_rows':calls,'centroid_fk_call_pose_rows':centroid_rows,
           'centroid_extra_fk_pose_rows':sum(centroid_rows),'actual_checkpoint_parameters_profiled':True,
           'centroid_extra_fk_matrix_macs_estimate':sum(x['matrix_macs_estimate'] for x in centroid_profiler),
           'centroid_extra_fk_torch_profiler':centroid_profiler,
           'separate_fk_estimate_scope':'CPU torch.profiler with_flops: leaf mm/bmm/addmm/baddbmm FLOPs divided by2; only the extra centroid FK, not mixed into the comparable table MAC; excludes unreported functional arithmetic, indexing, reductions, SO3 and data movement',
           'counter_scope':'same THOP standard-layer MAC lower bound as existing rows; excludes functional MANO FK, SO3, scatter/reductions and centroid arithmetic',
           'affine_mac_accounting':'parent row producer adds9 once for a translation affine, outside THOP',
           'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if audit_path is not None:M.X.write_json(audit_path,audit)
    return float(value),params,audit


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output',required=True,type=Path)
    args,_=parser.parse_known_args()
    original_sources,original_write=M.sources,M.X.write_json
    mac_audit=None
    def sources():
        result=original_sources()
        for name in ('semkine/xkf_centroid.py','tools/xkf_fusion/make_row_centroid.py'):
            result[name]=M.E.sha(ROOT/name)
        return result
    def macs(cfg,model):
        nonlocal mac_audit
        value,params,mac_audit=profile_policy_macs(cfg,model,Path(str(args.output)+'.mac_profile.json'))
        return value,params
    def write_json(path,value):
        if Path(path).resolve()==args.output.resolve():
            if mac_audit is None:raise RuntimeError('main row cannot be saved without actual-policy MAC audit')
            value=dict(value,mac_profile_audit=mac_audit)
        return original_write(path,value)
    M.XKFFusionModel=XKFCentroidModel
    M.FusionPolicy=CentroidFusionPolicy
    M.sources=sources
    M.F.macs=macs
    M.X.write_json=write_json
    M.main()


if __name__=='__main__':main()
