"""Mean-shape input firewall, separate-shape metrics and failure preservation."""
import json
from pathlib import Path
import struct

import numpy as np
import pytest
import torch

from tools.s43_rel import mean_shape_replay as M


def source_fixture(tmp_path,*,duration=250,poison_beta=True):
    base=tmp_path/'train'/'ch_fixture';base.parent.mkdir(parents=True)
    np.save(str(base)+'_events.npy',np.tile(np.array([[1,1,0]],np.uint8),(duration,1)))
    np.save(str(base)+'_offsets.npy',np.arange(duration+1,dtype=np.int64))
    beta=np.array([object()],object) if poison_beta else np.linspace(-.4,.5,10).astype(np.float32)
    np.savez(str(base)+'_aux.npz',camera_K=np.array([[800.,0,320.],[0,800.,240.],[0,0,1.]]),
             valid_runs_ms=np.array([[0,duration]]),betas=beta)
    Path(str(base)+'.meta').write_bytes(b'must not be opened during inference')
    (tmp_path/'splits_semkine.json').write_text(json.dumps({'train':{'trials':['ch_fixture'],
        'legacy_dir':{'ch_fixture':'train'}},'val_core':{'trials':[],'legacy_dir':{}}}))
    return base


def test_inference_reader_never_loads_poison_beta_or_pose(tmp_path):
    source_fixture(tmp_path)
    with M.InferenceAccessGuard() as guard:
        reader=M.MeanShapeInputReader('ch_fixture',tmp_path)
        packet=reader.packet(0,250)
        assert np.array_equal(reader.betas,np.zeros(10,np.float32))
        assert not hasattr(reader,'target') and not hasattr(reader,'labels')
        assert packet['source_range']==(0,250)
        np.testing.assert_array_equal(packet['events'][:,2],np.arange(250))
    r=guard.receipt()
    assert r['blocked_aux_beta_attempts']==r['blocked_pose_meta_attempts']==0
    assert [x['member'] for x in r['aux_members_read']]==['camera_K','valid_runs_ms']


def test_firewall_rejects_forbidden_member_and_metadata_and_restores(tmp_path):
    base=source_fixture(tmp_path,poison_beta=False)
    with M.InferenceAccessGuard() as guard:
        with pytest.raises(RuntimeError,match='betas'):
            with np.load(str(base)+'_aux.npz',allow_pickle=False) as aux:aux['betas']
        with pytest.raises(RuntimeError,match='pose .meta'):
            Path(str(base)+'.meta').open('rb')
    assert guard.receipt()['blocked_aux_beta_attempts']==guard.receipt()['blocked_pose_meta_attempts']==1
    with np.load(str(base)+'_aux.npz',allow_pickle=False) as aux:assert aux['betas'].shape==(10,)


def test_metric_uses_mean_pred_shape_actual_truth_shape_and_distinct_alignment(monkeypatch):
    calls=[]
    def fake_fk(mano,poses,beta):
        calls.append(np.asarray(beta).copy());n=len(poses)
        if np.count_nonzero(beta)==0:
            j=np.tile(np.array([[[.5,0,0],[1.,0,0]]]),(n,1,1))
            v=np.tile(np.array([[[3.,0,0],[4.,0,0]]]),(n,1,1))
        else:
            j=np.tile(np.array([[[0.,0,0],[.5,0,0]]]),(n,1,1))
            v=np.tile(np.array([[[2.,0,0],[3.,0,0]]]),(n,1,1))
        return j,v
    monkeypatch.setattr(M,'_mano',lambda:object());monkeypatch.setattr(M,'fk',fake_fk)
    truth_beta=np.ones(10,np.float32)
    _,per=M.shape_separate_metrics(np.zeros((1,51)),np.zeros((1,51)),truth_beta)
    np.testing.assert_array_equal(calls[0],np.zeros(10));np.testing.assert_array_equal(calls[1],truth_beta)
    assert per['mpjpe_abs_mm'][0]==500.
    assert per['mpvpe_abs_mm'][0]==1000.
    assert per['mpjpe_ra_mm'][0]==0.
    assert per['mpvpe_ra_mm'][0]==0.
    assert per['mpvpe_joint_wrist_aligned_mm'][0]==500.


def test_score_refuses_GT_until_prediction_archive_exists(tmp_path):
    with M.InferenceAccessGuard() as guard:
        with pytest.raises(RuntimeError,match='persist every predicted state'):
            M.score_saved_predictions(tmp_path/'missing.npz',tmp_path/'no_GT')
    assert guard.receipt()['blocked_pose_meta_attempts']==guard.receipt()['blocked_aux_beta_attempts']==0


def test_score_phase_reads_saved_pose_with_separate_actual_beta(tmp_path):
    base=source_fixture(tmp_path,duration=2,poison_beta=False)
    pose=np.tile(M.PUBLIC_INITIAL,(2,1))
    dt=np.dtype([('data','<f8',51),('m0','u1'),('m1','u1')]);meta=np.zeros(2,dt)
    meta['data']=pose;meta['m0']=4;meta['m1']=13
    with Path(str(base)+'.meta').open('wb') as f:f.write(struct.pack('<i',51));f.write(meta.tobytes())
    predfile=tmp_path/'saved.predictions.npz'
    np.savez(predfile,pred=pose,end_ms=np.array([0,1]),inference_beta=np.zeros(10,np.float32))
    values,per,target,beta,receipt=M.score_saved_predictions(predfile,base)
    assert receipt['all_predictions_saved_before_GT']
    assert receipt['predicted_FK_beta']=='zeros10' and receipt['GT_FK_beta']=='score_only_actual_aux'
    np.testing.assert_array_equal(target,pose)
    assert np.count_nonzero(beta)>0 and values['mpjpe_abs_mm']>0 and values['mpvpe_abs_mm']>0


def test_failed_query_saves_partial_predictions_without_hold_or_GT(tmp_path,monkeypatch):
    source_fixture(tmp_path)
    marker=tmp_path/'model.marker';marker.write_text('trusted synthetic unit fixture')
    cfg={'REL':{'BASE_CHECKPOINT':str(marker),'BASE_CONFIG':str(marker)},'MEMORY':{},'GEOMETRY':{}}
    class Model(torch.nn.Module):
        arm='s43'
        def __init__(self):super().__init__();self.parameter=torch.nn.Parameter(torch.zeros(()))
        def forward(self,reps,previous,betas=None,camera_K=None):return previous.clone()
        def counts(self):return {'total':1,'base':1}
    class Canonical(torch.nn.Module):
        def forward(self,image,previous,beta,K):return previous.clone()
    Original=M.CausalRelationTracker
    class Failure(Original):
        def query(self,cutoff):
            if cutoff==250:raise ValueError('injected geometry inadmissibility, test only')
            return super().query(cutoff)
    monkeypatch.setattr(M,'load_relation_model',lambda *a,**k:(Model(),cfg))
    monkeypatch.setattr(M,'load_legacy',lambda *a,**k:(Canonical(),cfg))
    monkeypatch.setattr(M,'CausalRelationTracker',Failure)
    prefix=tmp_path/'output'/'injected'
    with pytest.raises(ValueError,match='injected geometry'):
        M.replay(marker,None,'ch_fixture',prefix,root=tmp_path,engineering_prefix_ms=250,threads=1)
    failed=json.loads(Path(str(prefix)+'_failed.json').read_text())
    assert failed['failure_phase']=='tracked_mean_shape_query' and failed['failure_cutoff_ms']==250
    assert failed['source_count']==250 and failed['outputs']==2 and failed['last_commit_ms']==200
    assert failed['no_hold_clamp_or_GT_shape_fallback'] and not failed['GT_scoring_started']
    assert failed['inference_access_guard']['blocked_aux_beta_attempts']==0
    assert failed['inference_access_guard']['blocked_pose_meta_attempts']==0
    with np.load(failed['preserved_prediction_archive']) as z:
        assert z['pred'].shape==(2,51);np.testing.assert_array_equal(z['end_ms'],[149,199])
        np.testing.assert_array_equal(z['inference_beta'],np.zeros(10))


def test_formal_gate_waits_for_both_completion_schemas(tmp_path):
    ext=tmp_path/'extension.json';post=tmp_path/'post.json';final=tmp_path/'final.json'
    ext.write_text(json.dumps({'schema':'s43_rel_mean_shape_deployment_extension_v1','recordings':['zgz_local']}))
    post.write_text(json.dumps({'complete':True}));final.write_text(json.dumps({'execution_complete':False}))
    with pytest.raises(RuntimeError,match='waits'):
        M.validate_formal('unused',None,'zgz_local',ext,post,final)
    final.write_text(json.dumps({'execution_complete':True}))
    assert M.validate_formal('unused',None,'zgz_local',ext,post,final)['extension_sha256']==M.sha(ext)
