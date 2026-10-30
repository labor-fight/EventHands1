"""No-fit/no-target engineering tests for the new Stage5 serving executor."""
import copy,json,pickle
import numpy as np
import pytest
import torch
from torch import nn
from tools.s43_revisit.s43_field_data import make_input_batch
from tools.s43_revisit.s43_field_runtime import SharedS43Runtime
from tools.s43_revisit.s43_field_native import NativeS43Runtime
from tools.s43_revisit.s43_field_replay import FullStreamRunner,validate_cohort,evaluation_plan
from tools.s43_revisit import s43_field_replay as R


class Reader:
    def __init__(self):
        self.K_cam=np.array([[600.,0,320],[0,600,240],[0,0,1]],np.float32)
        self.runs=np.array([[0,325],[400,449]])
        self.offsets=np.arange(451,dtype=np.int64)*2
        self.events=np.tile(np.array([[91,83,0],[92,83,1]],np.int64),(450,1))
    def packet(self,a,b):
        lo,hi=int(self.offsets[a]),int(self.offsets[b]);e=self.events[lo:hi]
        return dict(events=np.column_stack((e[:,:2],np.repeat(np.arange(a,b),2),e[:,2])),source_range=(lo,hi))


class Legacy(nn.Module):
    def __init__(self):
        super().__init__();self.p=nn.Parameter(torch.zeros(()));self.rn=nn.Module();self.rn.fc=nn.Identity();self.images=[]
    def forward(self,image,prev,betas,camera_K):
        self.images.append(image.detach().clone());return prev+.0001+image.sum((1,2,3))[:,None]*.0001+self.p


def decoder(model,result):
    p=result.prediction[0].detach().numpy().copy()
    return p,np.broadcast_to(p[:3],(778,3)).copy(),np.broadcast_to(p[:3],(21,3)).copy(),{}


CONTRACT=dict(runtime_arm='legacy_s43',temporal_collapse=False,tensor_readout=True,state_fp32=True)


def test_native_has_no_unused_field_and_matches_shared_images_and_recursive_states():
    reader=Reader();a,b=Legacy(),Legacy()
    ra=FullStreamRunner(a,CONTRACT,native_s43=True,decoder=decoder,audit_prefix=False)
    rb=FullStreamRunner(b,CONTRACT,decoder=decoder,audit_prefix=False)
    aa,aw,sa=ra.run(reader,'fixture');bb,bw,sb=rb.run(reader,'fixture')
    for x,y in zip(a.images,b.images):torch.testing.assert_close(x,y,rtol=0,atol=0)
    np.testing.assert_array_equal([r['pred'] for r in aa],[r['pred'] for r in bb])
    assert sa['native_input_count']==sb['native_input_count']==(325+49)*2
    assert sa['native_input_sha256']==sb['native_input_sha256']
    assert sa['warmup_outputs']==2 and sa['outputs']==4
    assert sa['runtime_audit']['tail_events']==(25+49)*2
    assert sa['runtime_audit']['releases']==2 and not ra.runtime.lanes


def test_fixedcommit_extra_queries_do_not_change_common_prediction_history_or_observation():
    reader=Reader();a=FullStreamRunner(Legacy(),CONTRACT,native_s43=True,decoder=decoder,hop=50)
    b=FullStreamRunner(Legacy(),CONTRACT,native_s43=True,decoder=decoder,hop=25,commit_ms=50)
    aa,_,sa=a.run(reader,'fixture');bb,_,sb=b.run(reader,'fixture')
    common={r['end_ms']:r for r in bb if r['committed']}
    for r in aa:
        other=common[r['end_ms']];np.testing.assert_array_equal(r['pred'],other['pred'])
        for key in ('sensor_sha256','history_sha256','observation_input_sha256'):assert r[key]==other[key]
    assert sa['native_input_count']==sb['native_input_count'] and sa['native_input_sha256']==sb['native_input_sha256']
    assert sb['outputs']==9 and sb['runtime_audit']['commits']==6


def test_native_invalid_packet_rejects_without_mutating_existing_observation():
    r=NativeS43Runtime(Legacy(),transport_chunk_ms=1);reader=Reader()
    runner=FullStreamRunner(r.model,CONTRACT,native_s43=True,decoder=decoder)
    item=runner._item(reader,'fixture',0,0,325,0,50,0,True)
    outcome=r.forward(make_input_batch([item]),inference=True);r.commit(outcome)
    before=pickle.dumps(r.evidence());item=runner._item(reader,'fixture',0,0,325,50,100,1,False)
    item['events'][-1,2]=100
    with pytest.raises(ValueError,match='half-open'):r.inputs(make_input_batch([item]))
    assert pickle.dumps(r.evidence())==before
    assert r.lanes[item['seq_id']]['last_timestamp'].nbytes==691200
    assert r.lanes[item['seq_id']]['memory'].nbytes()==0


def test_failed_model_preserves_partial_trace_and_consumes_remaining_source_without_hold():
    class Fail(Legacy):
        def forward(self,*args,**kwargs):
            if len(self.images)==3:raise ArithmeticError('deliberate output failure')
            return super().forward(*args,**kwargs)
    runner=FullStreamRunner(Fail(),CONTRACT,native_s43=True,decoder=decoder,audit_prefix=False)
    rows,warm,s=runner.run(Reader(),'fixture')
    assert len(rows)==1 and len(warm)==2 and s['status']=='FAILED_PRESERVED'
    assert s['native_input_count']==748 and len(s['failures'])==1 and not runner.runtime.lanes


def test_plan_is_fixed_120_traces_and_missing_artifact_fails_before_score(tmp_path):
    plan=evaluation_plan(tmp_path)
    assert len(plan['jobs'])==120 and sum(j['profile']=='primary' for j in plan['jobs'])==36
    assert len({j['out'] for j in plan['jobs']})==120
    with pytest.raises(FileNotFoundError):validate_cohort(plan)


def test_saved_cohort_checks_actual_addresses_beta_and_archive_integrity_without_targets(tmp_path,monkeypatch):
    monkeypatch.setattr(R,'MeanShapeInputReader',lambda *a:Reader())
    monkeypatch.setattr(R,'validate_plan',lambda *a,**k:True)
    runner=FullStreamRunner(Legacy(),CONTRACT,native_s43=True,decoder=decoder)
    rows,warm,s=runner.run(Reader(),'fixture');path=tmp_path/'trace'
    R.save_trace(path.with_suffix('.npz'),rows,warm,np.zeros(10,np.float32))
    R.dump(path.with_suffix('.sourcefreeze.json'),dict(formal=True,arm='canonical',recording='fixture',
        sources={},command=dict(hop=50,commit_ms=50,perturbation='none',native_s43=False,stop_ms=None,checkpoint=None,provided_beta=None)))
    summary=dict(s,engineering=False,arm='canonical',shape_condition='beta0',source_files_unchanged=True,input_guard=dict(blocked_aux_beta_attempts=0,blocked_pose_meta_attempts=0,aux_betas_accessed=False,pose_metadata_opened=False),
        prediction_sha256=R.sha(path.with_suffix('.npz')),sourcefreeze_sha256=R.sha(path.with_suffix('.sourcefreeze.json')))
    R.dump(path.with_suffix('.json'),summary)
    plan=dict(jobs=[dict(profile='primary',arm='canonical',recording='fixture',out=str(path),hop=50,commit_ms=50,known_beta=False,perturbation='none',checkpoint=None)])
    assert len(validate_cohort(plan))==1
    # Rehashing a corrupted artifact cannot excuse an omitted valid output.
    R.save_trace(path.with_suffix('.npz'),rows[:-1],warm,np.zeros(10,np.float32))
    summary.update(outputs=len(rows)-1,prediction_sha256=R.sha(path.with_suffix('.npz')));R.dump(path.with_suffix('.json'),summary)
    with pytest.raises(AssertionError,match='every declaredvalidrun'):validate_cohort(plan)


def test_projection_failure_is_never_scored_or_replaced_by_held_pose(tmp_path,monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('target score boundary crossed')
    monkeypatch.setattr(R,'read_score_targets',forbidden)
    monkeypatch.setattr(R,'all_weights_gate',lambda *a,**k:dict(checkpoints={arm:str(R.TRAIN/'train/full'/arm/'final_weights.ckpt') for arm in R.MODELS[1:]}))
    plan=evaluation_plan(tmp_path);R.dump(tmp_path/'plan.json',plan)
    with pytest.raises(FileNotFoundError):R.score(tmp_path/'plan.json',tmp_path/'report')


def test_duplicate_or_missing_plan_cannot_cross_GT_boundary(tmp_path,monkeypatch):
    for mode in ('duplicate','missing'):
        plan=evaluation_plan(tmp_path/mode)
        if mode=='duplicate':plan['jobs'][-1]=dict(plan['jobs'][0])
        else:plan['jobs'].pop()
        with pytest.raises(RuntimeError,match='exact unique'):R.validate_plan(plan)


def test_rejected_input_retains_real_prefix_and_explicitly_requires_reset():
    class Bad(Reader):
        def packet(self,a,b):
            p=super().packet(a,b)
            if b==100:p['events'][-1,2]=100
            return p
    r=FullStreamRunner(Legacy(),CONTRACT,native_s43=True,decoder=decoder,audit_prefix=False)
    rows,warm,s=r.run(Bad(),'fixture')
    assert not rows and len(warm)==1 and s['status']=='FAILED_PRESERVED'
    assert s['native_input_count']==(50+49)*2 and s['unconsumed_native_count']==275*2
    assert not s['source_fully_consumed'] and s['failures'][0]['reset_required']
    assert not r.runtime.lanes


def test_tail_rejection_keeps_preceding_predictions():
    class Bad(Reader):
        def packet(self,a,b):
            p=super().packet(a,b)
            if a==300:p['events'][-1,2]=b
            return p
    rows,warm,s=FullStreamRunner(Legacy(),CONTRACT,native_s43=True,decoder=decoder,audit_prefix=False).run(Bad(),'fixture')
    assert len(rows)==4 and len(warm)==2 and s['status']=='FAILED_PRESERVED'
    assert s['failures'][0]['phase']=='tail_consume_release' and not s['source_fully_consumed']


def test_metrics_equal_original_ET_and_keep_shape_separate_without_dataset_GT():
    from tools.s43_integration.replay import _evaluation_functions,_metrics
    rng=np.random.default_rng(3407)
    pred=np.tile(np.asarray([0,0,.6]+[0]*48,np.float32),(3,1));target=pred.copy()
    pred[:,3:]+=rng.normal(0,.02,(3,48)).astype(np.float32);pred[:,0]+=.01
    beta=np.linspace(-.1,.1,10,dtype=np.float32)
    mano=copy.deepcopy(R._mano());mano.add_mean=False
    _,old=_metrics(_evaluation_functions(),mano,pred,target,beta)
    new=R.posthoc_metrics(pred,target,beta,beta)
    for key,value in old.items():np.testing.assert_allclose(new[key],value,rtol=0,atol=1e-5)
    known=R.posthoc_metrics(target,target,beta,beta)
    assert all(np.count_nonzero(v)==0 for v in known.values())
    mean=R.posthoc_metrics(target,target,np.zeros(10,np.float32),beta)
    assert np.all(mean['mpjpe_abs_mm']>0) and np.all(mean['mpvpe_abs_mm']>0)


def test_query_mode_strata_are_saved_metadata_groups_not_GT_selection():
    trace=dict(local_query_applicable=np.array([True,True,True]),used_uniform_reacquisition=np.array([False,True,False]),
        query_used_mode=np.array(['static','fixed_full_image_uniform9x10','static']),initial_camera_invalid_count=np.array([0,2,0]))
    report=R.query_mode_groups(trace,dict(fixture_error=np.array([1.,7.,3.])))
    assert report['fallback']['n']==1 and report['normal_local']['n']==2
    assert report['by_used_mode']['static']['metrics']['fixture_error']==2
    assert report['initial_camera_invalid']['n']==1 and report['local_query_applicable_rows']==3
