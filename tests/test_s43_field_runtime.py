"""Target-free lane/source/warmup/resume engineering, no scientific fitting."""
import copy,pickle
import numpy as np
import pytest
import torch
from torch import nn
from tools.s43_revisit.s43_field_data import TargetFreeRunLanes,Run,LazySupervisionReader,make_input_batch
from tools.s43_revisit.s43_field_runtime import SharedS43Runtime
from tools.s43_revisit import s43_field_runtime as R


class InputReader:
    def __init__(self,name,root):
        self.K_cam=np.array([[600.,0,320],[0,600,240],[0,0,1]],np.float32)
        self.runs=np.array([[0,325]])
        n=2 if name=='ch_local' else 3
        self.offsets=np.arange(326)*n
        self.events=np.tile(np.array([[90,80,1]],np.int64),(325*n,1))
    def packet(self,start,stop):
        lo,hi=int(self.offsets[start]),int(self.offsets[stop])
        raw=self.events[lo:hi];times=np.repeat(np.arange(start,stop),np.diff(self.offsets[start:stop+1]))
        return dict(events=np.column_stack((raw[:,:2],times,raw[:,2])),source_range=(lo,hi))


class Model(nn.Module):
    arm='static'
    def __init__(self):
        super().__init__();self.p=nn.Parameter(torch.tensor(0.));self.input_beta=[]
        self.rn=nn.Module();self.rn.fc=nn.Identity()
    def forward(self,field,prev,*,beta_input,K_source,ref_ms,cutoff_ms,histories):
        self.input_beta.append(beta_input.detach().cpu().clone())
        return prev+self.p+field[:,0].sum((1,2))[:,None]*1e-7+.001
    def loss(self,pred,target,*,beta_target):return (pred-target).square().mean(),dict(beta=beta_target)


class Legacy(nn.Module):
    def __init__(self):
        super().__init__();self.p=nn.Parameter(torch.tensor(0.));self.seen=[]
        self.rn=nn.Module();self.rn.fc=nn.Identity()
    def forward(self,image,prev,betas,camera_K):
        self.seen.append(image.detach().clone());return prev+self.p+.001


def schedule(B=2,stop=325,finite=True):
    return TargetFreeRunLanes(root='/fixture',subjects=('ch',),recordings=('ch_local','ch_global'),batch_size=B,
        finite=finite,runs=[Run('ch_local',0,0,stop),Run('ch_global',0,0,stop)],reader_factory=InputReader)


def advance(s,r):
    batch=s.peek_batch();result=r.forward(batch,inference=True);r.commit(result);s.commit_batch(batch)
    return batch,result


def test_input_only_reader_peek_cursor_and_padding_payload_not_parsed():
    s=schedule();b=s.peek_batch();assert s.peek_batch() is b
    assert s._state['outputs']==0
    for item in b:
        assert not set(item)&{'target','betas','beta_target','lnes','previous_GT'}
        assert np.all(item['beta_input']==0)
    with pytest.raises(RuntimeError,match='unacknowledged'):s.state_dict()
    s.commit_batch(b);assert s.state_dict()['state']['outputs']==2
    class Poison:
        def __repr__(self):raise AssertionError('invalid padding payload inspected')
    p=make_input_batch([dict(valid=False,padding=True,lane=0,events=Poison(),seq_id=Poison())])
    r=SharedS43Runtime(Model());before=r.state_dict()
    z=r.forward(p);r.commit(z)
    assert not z.valid_mask.any() and not z.supervised_mask.any()
    after=r.state_dict()
    assert before['lanes']==after['lanes'] and before['audit']==after['audit']
    torch.testing.assert_close(before['torch_rng'],after['torch_rng'],rtol=0,atol=0)
    for key in before['model']:torch.testing.assert_close(before['model'][key],after['model'][key],rtol=0,atol=0)


def test_same_prefix_chunks_multiplicity_and_distinct_sequences_exact():
    sa,sb=schedule(),schedule();a,b=SharedS43Runtime(Model(),transport_chunk_ms=1),SharedS43Runtime(Model(),transport_chunk_ms=50)
    for _ in range(4):
        ba,ra=advance(sa,a);bb,rb=advance(sb,b)
        assert ba.token==bb.token
        torch.testing.assert_close(ra.prepared.field40,rb.prepared.field40,rtol=0,atol=0)
        torch.testing.assert_close(ra.prediction,rb.prediction,rtol=0,atol=0)
        assert len(a.lanes)==2 and len({x['seq_id'] for x in ba})==2
    assert a.audit['ingested_events']==b.audit['ingested_events']==1000
    assert a.audit['source_chain']==b.audit['source_chain']


def test_own_two_warmup_outputs_and_lazy_targets_only_after_all_forward_inputs():
    s=schedule();model=Model();r=SharedS43Runtime(model);opened=[]
    class Target:
        def __init__(self,name,root):opened.append(name);self.betas=np.ones(10,np.float32)
        def target(self,t):return np.full(51,t/1000.,np.float32)
    target=LazySupervisionReader('/fixture',reader_factory=Target)
    for step in range(3):
        b=s.peek_batch();prepared=r.inputs(b)
        with pytest.raises(RuntimeError,match='precede supervision'):target.read(prepared)
        pred=r.query(prepared);supervision=target.read(pred)
        assert supervision.mask.sum()==(0 if step<2 else 2)
        assert len(opened)==(0 if step<2 else 2)
        if step==2:
            loss,terms=r.loss(pred,supervision);assert loss.requires_grad
            assert torch.all(terms['beta']==1)
        r.commit(pred);s.commit_batch(b)
    assert r.audit['warmup_outputs']==4 and r.audit['supervised_outputs']==2
    assert all(not x.any() for x in model.input_beta)


def test_resume_includes_cursor_rng_model_sensor_pose_and_history_exact():
    a=schedule(finite=False);ra=SharedS43Runtime(Model(),transport_chunk_ms=7)
    for _ in range(3):advance(a,ra)
    state=copy.deepcopy(ra.state_dict(schedule=a))
    b=schedule(finite=False);rb=SharedS43Runtime(Model(),transport_chunk_ms=7);rb.load_state_dict(state,schedule=b)
    for _ in range(5):
        ba,pa=advance(a,ra);bb,pb=advance(b,rb)
        assert ba.token==bb.token
        torch.testing.assert_close(pa.prediction,pb.prediction,rtol=0,atol=0)
        torch.testing.assert_close(pa.prepared.field40,pb.prepared.field40,rtol=0,atol=0)
        assert pickle.dumps(ra.evidence())==pickle.dumps(rb.evidence())


def test_no_optimizer_boundary_reset_and_all_tail_source_consumed_once():
    s=schedule();r=SharedS43Runtime(Model())
    while not s.exhausted:advance(s,r)
    assert not r.lanes and r.audit['ingested_events']==325*5
    assert r.audit['tail_events']==25*5 and r.audit['tail_ms']==50
    assert r.audit['warmup_outputs']==4 and r.audit['supervised_outputs']==8
    assert r.audit['committed_outputs']==12 and r.audit['resets']==2
    assert s.state_dict()['state']['events']==r.audit['ingested_events']


def test_duplicate_query_is_sensor_pose_pure_and_only_latest_can_commit():
    s=schedule(B=1);r=SharedS43Runtime(Model());b=s.peek_batch();p=r.inputs(b)
    before=pickle.dumps(r.evidence());a=r.query(p);c=r.query(p)
    assert before==pickle.dumps(r.evidence())
    torch.testing.assert_close(a.prediction,c.prediction,rtol=0,atol=0)
    with pytest.raises(ValueError,match='stale'):r.commit(a)
    with pytest.raises(RuntimeError,match='complete prediction'):r.state_dict()
    r.commit(c);s.commit_batch(b)
    with pytest.raises(ValueError,match='stale'):r.commit(c)


def test_field_path_never_constructs_LNES_and_legacy_control_alone_does(monkeypatch):
    old=R._legacy_last_image;old_times=R._legacy_image_from_times
    def poison(*args):raise AssertionError('LNES constructor reached by field arm')
    monkeypatch.setattr(R,'_legacy_last_image',poison)
    monkeypatch.setattr(R,'_legacy_image_from_times',poison)
    advance(schedule(B=1),SharedS43Runtime(Model()))
    monkeypatch.setattr(R,'_legacy_last_image',old)
    monkeypatch.setattr(R,'_legacy_image_from_times',old_times)
    s=schedule(B=1);m=Legacy();r=SharedS43Runtime(m,arm='legacy_s43');b,p=advance(s,r)
    expected=old(b[0]['events'],b[0]['cutoff_ms'])
    np.testing.assert_array_equal(m.seen[0][0].numpy(),expected)
    assert r.audit['legacy_LNES_images']==1 and r.lanes[b[0]['seq_id']]['last_timestamp'].nbytes==691200
    assert p.prepared.field40.shape[0]==0  # baseline skips unused field readout/H2D


def test_future_boundary_packet_and_source_gap_reject_before_consumption():
    s=schedule(B=1);r=SharedS43Runtime(Model());b,_=advance(s,r)
    good=s.peek_batch();item=dict(good[0]);item['source_range']=(item['source_range'][0]+1,item['source_range'][1]+1)
    before=pickle.dumps(r.evidence())
    with pytest.raises(ValueError,match='continuous'):r.inputs(make_input_batch([item],batch_id=99))
    assert pickle.dumps(r.evidence())==before
    item=dict(good[0]);item['events']=item['events'].copy();item['events'][-1,2]=item['cutoff_ms']
    with pytest.raises(ValueError,match='half-open'):r.inputs(make_input_batch([item],batch_id=99))
    assert pickle.dumps(r.evidence())==before


def test_exact_supervision_limit_is_separate_from_warmup_and_padding():
    s=schedule();r=SharedS43Runtime(Model())
    for _ in range(2):advance(s,r)
    b=s.peek_batch();pred=r.forward(b,supervised_limit=1)
    assert pred.supervised_mask.tolist()==[True,False]
    assert not pred.warmup_mask.any() and pred.budget_unselected_mask.tolist()==[False,True]
    r.commit(pred);s.commit_batch(b)
    assert r.audit['warmup_outputs']==4 and r.audit['supervised_outputs']==1
    assert r.audit['budget_unselected_outputs']==1 and r.audit['committed_outputs']==6


@pytest.mark.parametrize('arm',['static','hand_box'])
def test_collapse_preserves_counts_support_and_matches_frozen_control(arm):
    from tools.s43_revisit.field_query_probe import collapse_time
    s=schedule();r=SharedS43Runtime(Model(),arm=arm,temporal_collapse=True)
    batch=s.peek_batch();prepared=r.inputs(batch)
    raw=SharedS43Runtime(Model()).inputs(batch)
    expected=np.stack([collapse_time(x.numpy()) for x in raw.field40])
    np.testing.assert_allclose(prepared.field40.numpy(),expected,rtol=2e-7,atol=3e-7)
    for a,b in ((0,16),(24,40)):
        before=torch.expm1(raw.field40[:,a:b]).reshape(2,8,2,180,240).sum(1)
        after=torch.expm1(prepared.field40[:,a:b]).reshape(2,8,2,180,240).sum(1)
        torch.testing.assert_close(before,after,rtol=2e-6,atol=3e-6)
    torch.testing.assert_close(raw.field40[:,16:24],prepared.field40[:,16:24],rtol=0,atol=0)


def test_state_fp32_instance_adapter_casts_only_network_readout():
    class B(nn.Module):
        def __init__(self):
            super().__init__();self.rn=nn.Module();self.rn.fc=nn.Linear(2,51)
    base=B();R.install_state_fp32(base)
    with torch.autocast('cpu',dtype=torch.bfloat16):out=base.rn.fc(torch.ones(2,2))
    assert out.dtype==torch.float32
    assert base.rn.fc.weight.dtype==torch.float32
    runtime=SharedS43Runtime(Model())
    assert runtime.model.render_fp32 is True
    assert runtime.state_dict()['render_fp32'] is True
    assert runtime.state_dict()['loss_autocast'] is False


def test_transport_cap_does_not_become_window_cap_and_split_same_ms_group_atomic():
    from tools.s43_revisit.field_core import TimeFieldConfig
    a,b=schedule(),schedule()
    large=SharedS43Runtime(Model())
    small=SharedS43Runtime(Model(),field_config=TimeFieldConfig(max_packet_events=3))
    for _ in range(3):
        _,x=advance(a,large);_,y=advance(b,small)
        torch.testing.assert_close(x.prepared.field40,y.prepared.field40,rtol=0,atol=0)
        torch.testing.assert_close(x.prediction,y.prediction,rtol=0,atol=0)
    assert small.audit['ingested_events']==large.audit['ingested_events']
    assert all(lane['memory']._pending_time is None for lane in small.lanes.values())


def test_extra25ms_discard_preserves_feedback_and50ms_common_prefix():
    s=schedule(B=1);original=s.peek_batch()[0];events=original['events'];lo,hi=original['source_range']
    n=int(np.searchsorted(events[:,2],25));first=dict(original,events=events[:n],cutoff_ms=25,target_ms=24,source_range=(lo,lo+n))
    runtime=SharedS43Runtime(Model());z=runtime.forward(make_input_batch([first]))
    before=pickle.dumps(runtime.evidence());runtime.discard_query(z)
    assert pickle.dumps(runtime.evidence())==before and runtime.audit['commits']==0
    second=dict(original,events=events[n:],start_ms=25,source_range=(lo+n,hi),packet_ordinal=1,reset=False)
    out=runtime.forward(make_input_batch([second],batch_id=1))
    control=SharedS43Runtime(Model()).forward(make_input_batch([original]))
    torch.testing.assert_close(out.prepared.field40,control.prepared.field40,rtol=0,atol=0)
    torch.testing.assert_close(out.prediction,control.prediction,rtol=0,atol=0)
    runtime.commit(out);assert runtime.audit['warmup_outputs']==1 and runtime.audit['supervised_outputs']==0


def test_conditional_shape_is_explicit_and_default_training_remains_beta0():
    s=schedule(B=1);item=dict(s.peek_batch()[0]);item['beta_input']=np.ones(10,np.float32)
    with pytest.raises(ValueError,match='conditional'):SharedS43Runtime(Model()).forward(make_input_batch([item]))
    runtime=SharedS43Runtime(Model(),allow_external_beta=True)
    out=runtime.forward(make_input_batch([item]));runtime.commit(out)
    assert runtime.state_dict()['allow_external_beta']
    assert torch.all(out.prepared.beta_input==1)


def test_native_collapse_only_accepts_exact_r2_block_fields():
    s=schedule(B=1);raw=SharedS43Runtime(Model()).inputs(s.peek_batch()).field40
    reference=R.temporal_collapse_field(raw)
    native=R.temporal_collapse_field(raw,native_r2=True)
    torch.testing.assert_close(reference,native,rtol=0,atol=0)
    bad=raw.clone();bad[0,0,0,1]+=.01
    with pytest.raises(ValueError,match='nearest-block'):R.temporal_collapse_field(bad,native_r2=True)
    # General nonblock callers retain the original full-image definition.
    assert torch.isfinite(R.temporal_collapse_field(bad)).all()
