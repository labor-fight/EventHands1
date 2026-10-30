"""Objective/output-clock and complete Adam/runtime resume, fixture data only."""
import copy,json,random
import numpy as np
import pytest
import torch
from torch import nn
from tools.s43_revisit import s43_field_training as T
from tools.s43_revisit.s43_field_data import TargetFreeRunLanes,Run,LazySupervisionReader
from tools.s43_revisit.s43_field_runtime import SharedS43Runtime


def test_mean_logphase_gradient_is_not_log_of_mean_loss():
    p=nn.Parameter(torch.tensor(1.))
    for shift in (0.,3.):((p-shift).square().log10()).backward()
    T.average_phase_gradients([p],2);actual=p.grad.clone()
    q=torch.tensor(1.,requires_grad=True);objective=torch.stack([q.square().log10(),(q-3).square().log10()]).mean()
    objective.backward();torch.testing.assert_close(actual,q.grad,rtol=0,atol=0)
    z=torch.tensor(1.,requires_grad=True);((z.square()+(z-3).square())/2).log10().backward()
    assert not torch.isclose(actual,z.grad)


def test_original_log_has_no_new_zero_clamp_and_output_scheduler_resume():
    model=type('Model',(),{'log10_loss':True})()
    assert torch.isneginf(T.original_objective(torch.tensor(0.),model))
    p=nn.Parameter(torch.tensor(0.));opt=torch.optim.Adam([p],lr=.001)
    s=T.OutputScheduler(opt,target=6144000,warmup=65536,base_lr=.001)
    assert s.prepare(256)==pytest.approx(.001/256);s.advance(256)
    state=copy.deepcopy(s.state_dict());r=T.OutputScheduler(opt,target=6144000,warmup=65536,base_lr=.001);r.load_state_dict(state)
    assert r.prepare(65536)==.001 and r.multiplier(6144000)==pytest.approx(.02)
    with pytest.raises(ValueError):r.prepare(256)


def test_five_fixed_arms_collapse_and_numerics_contract():
    assert len(T.ARMS)==5 and T.arm_contract('s43')['runtime_arm']=='legacy_s43'
    for arm in T.ARMS:
        c=T.arm_contract(arm);assert c['state_fp32'] and c['tensor_readout'] and not c['allow_external_beta']
        assert c['temporal_collapse']==arm.endswith('_collapse')
    assert T.arm_contract('field_box_collapse')['runtime_arm']=='hand_box'
    assert T.STAGES['small']['target']==32768 and T.STAGES['full']['target']==6144000


class Input:
    def __init__(self,name,root):
        self.runs=np.array([[0,325]]);self.K_cam=np.array([[600.,0,320],[0,600,240],[0,0,1]],np.float32)
    def packet(self,a,b):
        t=np.arange(a,b);e=np.column_stack((np.full(len(t),90),np.full(len(t),80),t,t%2)).astype(np.int64)
        return dict(events=e,source_range=(a,b))


class Model(nn.Module):
    arm='static';log10_loss=True;render_fp32=True
    def __init__(self):
        super().__init__();self.p=nn.Parameter(torch.tensor(0.));self.rn=nn.Module();self.rn.fc=nn.Identity()
    def forward(self,field,prev,*,beta_input,**kwargs):
        assert not beta_input.any()
        return prev+self.rn.fc(self.p+torch.randn(len(prev),1)*1e-5)+.001
    def loss(self,pred,target,*,beta_target):
        assert torch.all(beta_target==1)
        return (pred-target).square().mean()+.001,{}


class Target:
    def __init__(self,name,root):self.betas=np.ones(10,np.float32)
    def target(self,t):return np.asarray([.01,0,.6]+[0.]*48,np.float32)


def setup():
    model=Model();s=TargetFreeRunLanes('/fixture',subjects=('ch',),recordings=('ch_local',),batch_size=2,
        runs=[Run('ch_local',0,0,325)],reader_factory=Input)
    r=SharedS43Runtime(model);target=LazySupervisionReader('/fixture',reader_factory=Target)
    opt=torch.optim.Adam(model.parameters(),lr=.001);scheduler=T.OutputScheduler(opt,target=8,warmup=4,base_lr=.001)
    return model,s,r,target,opt,scheduler


def update(v):
    model,s,r,target,opt,scheduler=v;opt.zero_grad();phases=0;done=scheduler.outputs
    while phases<2:
        batch=s.peek_batch();result=r.forward(batch,supervised_limit=8-done);truth=target.read(result)
        if truth.mask.any():
            raw,_=r.loss(result,truth);T.original_objective(raw,model).backward();done+=truth.mask.sum();phases+=1
        r.commit(result);s.commit_batch(batch)
    T.average_phase_gradients(list(model.parameters()),phases);scheduler.prepare(done);opt.step();scheduler.advance(done)


def test_complete_adam_scheduler_rng_cursor_and_numeric_memory_resume(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(3407);a=setup();update(a)
    model,s,r,target,opt,sch=a
    state=copy.deepcopy(dict(model=model.state_dict(),runtime=r.state_dict(schedule=s,include_model=False),
        optimizer=opt.state_dict(),scheduler=sch.state_dict(),reads=target.reads))
    update(a);expected=model.p.detach().clone();expected_evidence=r.evidence();expected_rng=torch.get_rng_state().clone()
    b=setup();bm,bs,br,bt,bo,bsc=b;bm.load_state_dict(state['model']);bo.load_state_dict(state['optimizer']);bsc.load_state_dict(state['scheduler'])
    br.load_state_dict(state['runtime'],schedule=bs);bt.reads=state['reads'];update(b)
    torch.testing.assert_close(expected,bm.p,rtol=0,atol=0);torch.testing.assert_close(expected_rng,torch.get_rng_state(),rtol=0,atol=0)
    assert bsc.outputs==sch.outputs==8 and bt.reads==target.reads==8
    assert bs.state_dict()['state']['outputs']==br.audit['committed_outputs']
    for sid,e in expected_evidence.items():
        actual=br.evidence()[sid];assert actual['sensor_sha256']==e['sensor_sha256']
        np.testing.assert_array_equal(actual['pose'],e['pose'])
    assert all(float(x['step'])==2 for x in bo.state.values())


def test_exact_partial_budget_never_relabels_unused_rows_as_warmup():
    torch.set_num_threads(2);model,s,r,target,opt,sch=setup()
    for _ in range(2):
        b=s.peek_batch();z=r.forward(b,inference=True);r.commit(z);s.commit_batch(b)
    b=s.peek_batch();z=r.forward(b,supervised_limit=1);truth=target.read(z)
    assert truth.mask.sum()==1 and z.budget_unselected_mask.sum()==1 and not z.warmup_mask.any()
    r.commit(z);s.commit_batch(b)
    assert r.audit['supervised_outputs']==target.reads==1 and r.audit['warmup_outputs']==4
    assert r.audit['budget_unselected_outputs']==1 and r.audit['committed_outputs']==6
