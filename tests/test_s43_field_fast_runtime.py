"""Full readout execution parity, source/state exactness; no labels/tasks."""
import copy
import numpy as np
import pytest
import torch
from torch import nn
from tools.s43_revisit.s43_field_runtime_R1 import SharedS43RuntimeR1
from tools.s43_revisit.runtime_fastreadout import SharedS43RuntimeFastReadout
from tools.s43_revisit.s43_field_data import make_input_batch
from tools.s43_revisit.field_core import TimeFieldConfig

class Model(nn.Module):
 def __init__(self):
  super().__init__();self.p=nn.Parameter(torch.tensor(.001));self.rn=nn.Module();self.rn.fc=nn.Identity();self.bn=nn.BatchNorm1d(1)
 def forward(self,field,prev,**kw):return prev+self.rn.fc(self.bn(field.float().flatten(1).mean(1,keepdim=True))*self.p)

def batch(k):
 items=[]
 for lane in range(3):
  n=0 if k==2 else 90;lo=min(k,2)*90;t=np.sort(np.random.default_rng(k*7+lane).integers(k*50,(k+1)*50,n));e=np.column_stack((np.arange(n)%240,np.arange(n)%180,t,np.arange(n)%2)).astype(np.int64)
  items.append(dict(valid=True,padding=False,seq_id=f'fixture{lane}',reset=k==0,initial_state=np.array([0,0,.6]+[0.]*48,np.float32),
   beta_input=np.zeros(10,np.float32),K_source=np.eye(3,dtype=np.float32),run_start_ms=0,start_ms=k*50,cutoff_ms=(k+1)*50,packet_ordinal=k,source_range=(lo,lo+n),events=e))
 return make_input_batch(items,batch_id=k)

@pytest.mark.parametrize('arm,collapse',[('static',False),('static',True),('legacy_s43',False)])
def test_old_fast_full_tensor_bn_sources_and_restore_exact(arm,collapse):
 torch.set_num_threads(2);torch.manual_seed(3407);model=Model();other=copy.deepcopy(model);cfg=TimeFieldConfig(max_packet_events=3)
 old=SharedS43RuntimeR1(model,arm=arm,temporal_collapse=collapse,field_config=cfg);fast=SharedS43RuntimeFastReadout(other,arm=arm,temporal_collapse=collapse,field_config=cfg)
 for k in range(3):
  b=batch(k);p=old.inputs(b);q=fast.inputs(b)
  torch.testing.assert_close(p.field40,q.field40,rtol=0,atol=0)
  if arm=='legacy_s43':torch.testing.assert_close(p.legacy_images,q.legacy_images,rtol=0,atol=0)
  x=old.query(p);y=fast.query(q);torch.testing.assert_close(x.prediction,y.prediction,rtol=0,atol=0);old.commit(x);fast.commit(y)
  assert old.audit==fast.audit
  a=old.evidence();c=fast.evidence()
  for sid in a:
   assert a[sid]['sensor_sha256']==c[sid]['sensor_sha256'];np.testing.assert_array_equal(a[sid]['pose'],c[sid]['pose'])
 for key,val in model.state_dict().items():torch.testing.assert_close(val,other.state_dict()[key],rtol=0,atol=0)
 restored=SharedS43RuntimeFastReadout(Model(),arm=arm,temporal_collapse=collapse,field_config=cfg);restored.load_state_dict(fast.state_dict())
 assert restored.audit==fast.audit and restored.pose_diagnostics==fast.pose_diagnostics
 assert restored.state_dict()['readout_backend']=='native_ring_bins_v1'
 for sid,e in fast.evidence().items():assert e['sensor_sha256']==restored.evidence()[sid]['sensor_sha256']
 old.close();fast.close();restored.close()

def test_fast_runtime_rejects_wrong_execution_identity():
 rt=SharedS43RuntimeFastReadout(Model(),arm='static');state=rt.state_dict();state.pop('readout_backend')
 with pytest.raises(ValueError):rt.load_state_dict(state)
 rt.close()
