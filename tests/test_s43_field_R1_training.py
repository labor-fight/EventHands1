"""R1 ordered lane execution parity; synthetic packets, no task scores."""
import copy
import numpy as np
import pytest
import torch
from torch import nn
from tools.s43_revisit.s43_field_runtime import SharedS43Runtime
from tools.s43_revisit.s43_field_runtime_R1 import SharedS43RuntimeR1
from tools.s43_revisit.s43_field_data import make_input_batch
from tools.s43_revisit.field_core import TimeFieldConfig
from tools.s43_revisit import s43_field_training_R1 as T

class Model(nn.Module):
    log10_loss=True
    def __init__(self):
        super().__init__();self.p=nn.Parameter(torch.tensor(.001));self.rn=nn.Module();self.rn.fc=nn.Identity();self.bn=nn.BatchNorm1d(1)
    def forward(self,field,prev,**kwargs):
        x=field.float().flatten(1).mean(1,keepdim=True)
        return prev+self.rn.fc(self.bn(x)*self.p)

def packet(k,*,empty=False,padding=False,seqprefix='s',start=None,stop=None):
    a=k*50 if start is None else start;b=a+50 if stop is None else stop;items=[]
    for lane in range(3):
        n=0 if empty else 80;t=np.sort(np.random.default_rng(3407+lane+k).integers(a,b,n));events=np.column_stack((np.arange(n)%240,np.arange(n)%180,t,np.arange(n)%2)).astype(np.int64)
        lo=(0 if k==0 else 80*k);hi=lo+n
        # Empty-step fixture continues the actual source prefix.
        if k==2:lo=160;hi=160+n
        items.append(dict(lane=lane,valid=not(padding and lane==2),padding=padding and lane==2,seq_id=f'{seqprefix}{lane}',reset=k==0,
          initial_state=np.array([0,0,.6]+[0.]*48,np.float32),beta_input=np.zeros(10,np.float32),K_source=np.eye(3,dtype=np.float32),
          start_ms=a,cutoff_ms=b,run_start_ms=0,packet_ordinal=k,source_range=(lo,hi),events=events))
    return make_input_batch(items,batch_id=k)

def assert_state(a,b):
    assert a.audit==b.audit
    ea=a.evidence();eb=b.evidence();assert ea.keys()==eb.keys()
    for sid,x in ea.items():
        y=eb[sid];assert x['sensor_sha256']==y['sensor_sha256'];np.testing.assert_array_equal(x['pose'],y['pose'])
        assert x['source_end']==y['source_end'] and x['core_ordinal']==y['core_ordinal']

@pytest.mark.parametrize('arm,collapse',[('static',False),('static',True),('legacy_s43',False)])
def test_serial_vs_16_workers_fields_provenance_bn_and_restore(arm,collapse):
    torch.set_num_threads(2);torch.manual_seed(3407);m=Model();n=copy.deepcopy(m)
    cfg=TimeFieldConfig(max_packet_events=3)
    a=SharedS43Runtime(m,arm=arm,field_config=cfg,temporal_collapse=collapse)
    b=SharedS43RuntimeR1(n,arm=arm,field_config=cfg,temporal_collapse=collapse,runtime_workers=16)
    try:
        for k in range(3):
            batch=packet(k,empty=k==2)
            x=a.inputs(batch);y=b.inputs(batch)
            for key in ('field40','previous','beta_input','K_source'):
                torch.testing.assert_close(getattr(x,key),getattr(y,key),rtol=0,atol=0)
            if arm=='legacy_s43':torch.testing.assert_close(x.legacy_images,y.legacy_images,rtol=0,atol=0)
            p=a.query(x);q=b.query(y);torch.testing.assert_close(p.prediction,q.prediction,rtol=0,atol=0)
            a.commit(p);b.commit(q);assert_state(a,b)
        for k,v in m.state_dict().items():torch.testing.assert_close(v,n.state_dict()[k],rtol=0,atol=0)
        saved=b.state_dict();c=SharedS43RuntimeR1(Model(),arm=arm,field_config=cfg,temporal_collapse=collapse,runtime_workers=16)
        try:c.load_state_dict(saved);assert_state(b,c)
        finally:c.close()
    finally:b.close()


def test_padding_release_tail_and_reset_ordered_source_chain():
    torch.set_num_threads(2);m=Model();a=SharedS43Runtime(m,arm='static');b=SharedS43RuntimeR1(copy.deepcopy(m),arm='static')
    try:
        x=packet(0,padding=True)
        for rt in (a,b):out=rt.forward(x);rt.commit(out)
        release=dict(x.items[0],events=np.empty((0,4),np.int64),source_range=(80,80),start_ms=50,cutoff_ms=55,packet_ordinal=1,reset=False)
        fresh=packet(0,seqprefix='new',start=55,stop=105)
        items=[dict(i,run_start_ms=55) for i in fresh.items]
        x=make_input_batch(items,releases=(release,),batch_id=1)
        for rt in (a,b):out=rt.forward(x);rt.commit(out)
        assert_state(a,b);assert a.audit['tail_ms']==5 and a.audit['releases']==1
    finally:b.close()


def test_R1_constructor_and_checkpoint_identity_is_explicit(tmp_path):
    assert T.OUT.name=='s43_full_field_R1'
    for arm in T.ARMS:
        c=T.arm_contract(arm);assert c['runtime_workers']==1 and c['geometry_workers']==1
        assert c['model_variant']==('canonical_s43' if arm=='s43' else 'invalid_geometry_uniform_r1')
    bad=tmp_path/'old.ckpt';torch.save(dict(schema='s43_field_training_v1',STATE_FP32=True,RENDER_FP32=True),bad)
    with pytest.raises(ValueError):T.load_weights(bad)
    new=tmp_path/'new.ckpt';torch.save(dict(schema='s43_field_training_R1_v1',STATE_FP32=True,RENDER_FP32=True,variant='wrong'),new)
    with pytest.raises(ValueError):T.load_weights(new)
