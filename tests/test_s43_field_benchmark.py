"""Resource accounting tests on untrained toy output, no GT or GPU."""
import json
import numpy as np
import torch
from torch import nn
from tools.s43_revisit import s43_field_benchmark as B


class RN(nn.Module):
    def __init__(self):super().__init__();self.fc=nn.Identity()
    def pooled(self,*a):return a


class Model(nn.Module):
    prevpos_embed=False
    def __init__(self,fail=False):
        super().__init__();self.p=nn.Parameter(torch.zeros(()));self.conv1=nn.Identity();self.rn=RN();self.fail=fail
    def _render_prev(self,*a,**k):return torch.zeros(1,180,240,2)
    def forward(self,image,prev,betas,camera_K):
        if self.fail and prev[0,0]>=.0029:raise ArithmeticError('deliberate invalid geometry')
        return prev+.001+self.p
    def _fk(self,p,beta):return p[:,:3,None].transpose(1,2).expand(-1,778,-1),p[:,:3,None].transpose(1,2).expand(-1,21,-1)


def source(tmp_path):
    prefix=tmp_path/'source';n=350
    e=np.column_stack((np.full(n*2,90),np.full(n*2,80),np.repeat(np.arange(n),2),np.tile([0,1],n))).astype(np.int64)
    np.save(prefix.with_suffix('.events.npy'),e);np.save(prefix.with_suffix('.offsets.npy'),np.arange(n+1,dtype=np.int64)*2)
    B.R.dump(prefix.with_suffix('.json'),dict(stop_ms=n,K_source=[[600,0,320],[0,600,240],[0,0,1]],
        native_count=n*2,noise_count=0,events_sha256=B.R.sha(prefix.with_suffix('.events.npy')),offsets_sha256=B.R.sha(prefix.with_suffix('.offsets.npy'))))
    return prefix


def setup(monkeypatch,fail=False):
    monkeypatch.setattr(B.R,'load_model',lambda *a,**k:(Model(fail),dict(runtime_contract=dict(runtime_arm='legacy_s43',temporal_collapse=False,tensor_readout=True,state_fp32=True))))
    monkeypatch.setattr(B.R,'source_files',lambda *a,**k:[])


def test_primary_service_and_detailed_secondpass_match_and_restore_wrappers(tmp_path,monkeypatch):
    setup(monkeypatch);original=torch.as_tensor
    r=B.benchmark('canonical',source(tmp_path),tmp_path/'resource.json',engineering=True,native_s43=True)
    assert r['status']=='COMPLETE' and r['native_input_count']==700
    assert r['outputs']==5 and r['warmup_outputs']==2 and r['full_surface_outputs_including_warmup']==7
    assert r['component_profile']['prediction_and_surface_and_source_exact']
    assert r['throughput']['steady_native_events']==500
    assert r['memory']['peak_resident_numeric_state_bytes']<700000
    assert torch.as_tensor is original
    assert 'steady/steady_cycle_ms' in r['phases']
    assert 'steady/sensor_maintenance_ms' in r['component_profile']['phases']


def test_failed_workload_has_no_fullstream_throughput_and_prefix_count_excludes_drained_tail(tmp_path,monkeypatch):
    setup(monkeypatch,True)
    r=B.benchmark('canonical',source(tmp_path),tmp_path/'failed.json',engineering=True,native_s43=True)
    assert r['status']=='FAILED_PRESERVED' and r['throughput'] is None
    assert r['native_input_count']==700 and r['outputs']==1
    assert r['partial_successful_prefix_costs_only']['steady_native_events']==100
    assert r['component_profile']['prediction_and_surface_and_source_exact']
