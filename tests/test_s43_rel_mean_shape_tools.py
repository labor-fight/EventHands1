"""Public-beta0 wrapper/collector contracts; no real GT or task scores."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from tools.s43_rel import mean_shape_benchmark as B0
from tools.s43_rel import mean_shape_report as Q


def fake_source(tmp_path):
    root=tmp_path/'source';base=root/'train/ch_local';base.parent.mkdir(parents=True)
    Q.R.dump(root/'splits_semkine.json',dict(train=dict(trials=['ch_local'],legacy_dir={'ch_local':'train'}),val_core=dict(trials=[],legacy_dir={})))
    np.save(str(base)+'_events.npy',np.array([[1,2,0],[3,4,1]],np.int16))
    np.save(str(base)+'_offsets.npy',np.array([0,1,2],np.int64))
    np.savez(str(base)+'_aux.npz',camera_K=np.array([[800.,0,320],[0,800.,240],[0,0,1]]),
        valid_runs_ms=np.array([[0,2]]),betas=np.full(10,np.nan))
    Path(str(base)+'.meta').write_bytes(b'forbidden pose')
    Path(str(base)+'_tsub.npy').write_bytes(b'forbidden timestamp sidecar')
    extension=tmp_path/'extension.json';Q.R.dump(extension,{'scope':'test only'})
    return root,base,extension


def test_wrapper_uses_only_allowed_aux_members_and_restores_original_globals(tmp_path):
    root,base,extension=fake_source(tmp_path)
    old=(B0.B.ObservationReader,B0.B.source_identity,B0.B.CausalRelationTracker,B0.B.decode_surface)
    with B0.public_mean_context(extension) as (trace,guard):
        reader=B0.MeanShapeObservationReader('ch_local',root)
        assert np.array_equal(reader.betas,np.zeros(10))
        receipt=guard.receipt()
        assert {q['member'] for q in receipt['aux_members_read']}=={'camera_K','valid_runs_ms'}
        assert receipt['blocked_aux_beta_attempts']==receipt['blocked_pose_meta_attempts']==0
    assert old==(B0.B.ObservationReader,B0.B.source_identity,B0.B.CausalRelationTracker,B0.B.decode_surface)
    assert not B0._PATCH_LOCK.locked()


def test_forbidden_read_failure_restores_patch_and_blocks_beta_pose_and_tsub(tmp_path):
    root,base,extension=fake_source(tmp_path)
    original=B0.B.ObservationReader
    with B0.public_mean_context(extension) as (_,guard):
        with np.load(str(base)+'_aux.npz') as aux:
            with pytest.raises(RuntimeError):aux['betas']
        with pytest.raises(RuntimeError):Path(str(base)+'.meta').open('rb')
        with pytest.raises(RuntimeError):Path(str(base)+'_tsub.npy').open('rb')
        assert guard.receipt()['blocked_aux_beta_attempts']==1
    with pytest.raises(RuntimeError,match='intentional'):
        with B0.public_mean_context(extension):raise RuntimeError('intentional wrapper failure')
    assert B0.B.ObservationReader is original and not B0._PATCH_LOCK.locked()


def test_formal_resource_requires_both_completion_gates(tmp_path,monkeypatch):
    root=tmp_path/'run';root.mkdir();monkeypatch.setattr(B0.R,'DEFAULT',root)
    Q.R.dump(root/'protocol.json',{})
    extension=root/'extension.json';Q.R.dump(extension,dict(schema='s43_rel_mean_shape_deployment_extension_v1',frozen_main_protocol_sha256=Q.R.sha(root/'protocol.json')))
    with pytest.raises(Q.R.InvalidArtifact):B0.formal_gate('canonical',None,extension)
    Q.R.dump(root/'postprocess_complete.json',dict(complete=True))
    with pytest.raises(Q.R.InvalidArtifact):B0.formal_gate('canonical',None,extension)
    Q.R.dump(root/'finalization_complete.json',dict(execution_complete=True))
    B0.formal_gate('canonical',None,extension)


def valid_replay():
    return dict(schema=Q.SCHEMA,arm='full',recording='zgz_local',shape_condition='public_mean_beta0',inference_beta=[0.]*10,
        aux_betas_accessed_during_inference=False,pose_metadata_opened_during_inference=False,
        GT_only_after_predictions_saved=True,scoring_performed=True,
        inference_access_guard=dict(aux_betas_accessed=False,pose_metadata_opened=False,blocked_aux_beta_attempts=0,blocked_pose_meta_attempts=0),
        predicted_FK_beta='zeros10',GT_FK_beta='score_only_actual_aux',hop_ms=50,commit_schedule_ms=50,
        observation_window_ms=50,output_lag_ms=1,initialization_GT=False,full_recording=True,
        all_valid_events_consumed_once=True,source_count=3,expected_source_count=3,source_id_digest='x',expected_source_id_digest='x',
        source_files_unchanged={'source':True},outputs=2,metrics={k:1. for k in Q.R.METRICS})


@pytest.mark.parametrize('key,value',[('inference_beta',[.1]*10),('shape_condition','conditional_aux_beta'),
    ('GT_FK_beta','zeros10'),('GT_only_after_predictions_saved',False),('scoring_performed',False)])
def test_collector_refuses_conditional_or_same_shape_gt_score(key,value):
    doc=valid_replay();Q.validate_replay(doc,'full','zgz_local');doc[key]=value
    with pytest.raises(Q.R.InvalidArtifact):Q.validate_replay(doc,'full','zgz_local')


def test_mean_shape_inventory_requires_all10_replays_and5_resources(tmp_path):
    root=tmp_path/'mean_shape';extension=tmp_path/'extension.json'
    state=Q.inventory(root,extension)
    assert not state['complete_artifact_set'] and state['required_replays']==10 and state['required_resource_measurements']==5
    with pytest.raises(Q.R.IncompleteReport):Q.collect(root,extension)


def test_posthoc_gt_masks_align_by_addresses_and_empty_finger_mask_is_missing_evidence(tmp_path):
    root=tmp_path/'mean_shape';masks=tmp_path/'masks';masks.mkdir()
    for recording in Q.RECORDINGS:
        np.savez(masks/f'{recording}_nominal_motion_strata.npz',end_ms=[149,199],run_id=[0,1],
            all_outputs=[True,True],local_finger_primary=[False,False],local_index_secondary=[True,False])
        path=root/f'replays/full_{recording}_h50.npz';path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(path,end_ms=[199,149],run_id=[1,0],**{k:np.array([30.,10.]) for k in (*Q.R.METRICS,Q.WRIST_V)})
    result=Q.posthoc_motion_summary(root,arms=('full',),mask_root=masks)
    assert result['status']=='PASS' and not result['conditional_model_scores_read']
    row=result['recordings']['zgz_local']['arms']['full']['masks']
    assert row['local_finger_primary']['n']==0 and row['local_finger_primary']['metrics'] is None
    assert row['local_index_secondary']['metrics']['mpjpe_ra_mm']==10.
    path=root/'replays/full_zgz_local_h50.npz'
    np.savez(path,end_ms=[149],run_id=[0],**{k:np.array([10.]) for k in Q.R.METRICS})
    assert Q.posthoc_motion_summary(root,arms=('full',),mask_root=masks)['status']=='FAIL'


def test_complete_mean_shape_report_uses10_actual_replays_and5_beta0_resources(tmp_path):
    root=tmp_path/'mean_shape';extension=tmp_path/'extension.json'
    Q.R.dump(extension,dict(schema='s43_rel_mean_shape_deployment_extension_v1'))
    Q.R.dump(tmp_path/'postprocess_complete.json',dict(complete=True))
    Q.R.dump(tmp_path/'finalization_complete.json',dict(execution_complete=True))
    for arm in Q.ARMS:
        if arm!='canonical':
            folder=tmp_path/f'train/full_r1/{arm}';folder.mkdir(parents=True)
            (folder/'last.ckpt').write_text('resumable');(folder/'step=6000.ckpt').write_text('deploy')
            Q.R.dump(folder/'complete.json',dict(completed_steps=6000,actual_outputs=192000,checkpoint_sha256=Q.R.sha(folder/'last.ckpt')))
        for recording in Q.RECORDINGS:
            prefix=root/f'replays/{arm}_{recording}_h50';prefix.parent.mkdir(parents=True,exist_ok=True)
            saved=prefix.with_suffix('.predictions.npz');pred=np.zeros((2,51),np.float32)
            np.savez(saved,pred=pred,end_ms=[149,199],run_id=[0,0],inference_beta=np.zeros(10,np.float32))
            metrics={k:float(i+1) for i,k in enumerate((*Q.R.METRICS,Q.WRIST_V))}
            np.savez(prefix.with_suffix('.npz'),pred=pred,gt=np.ones((2,51),np.float32),end_ms=[149,199],run_id=[0,0],
                **{k:np.full(2,v,np.float32) for k,v in metrics.items()})
            doc=valid_replay();doc.update(arm=arm,recording=recording,model_arm='s43' if arm=='canonical' else arm,
                prediction_archive_sha256=Q.R.sha(saved),metrics=metrics,
                scoring_receipt=dict(all_predictions_saved_before_GT=True,predicted_beta_zero=True,
                    prediction_archive_sha256=Q.R.sha(saved),predicted_FK_beta='zeros10',GT_FK_beta='score_only_actual_aux'),
                checkpoint=None if arm=='canonical' else str((tmp_path/f'train/full_r1/{arm}/last.ckpt').resolve()),
                source_files={} if arm=='canonical' else {str((tmp_path/f'train/full_r1/{arm}/last.ckpt').resolve()):{'sha256':Q.R.sha(tmp_path/f'train/full_r1/{arm}/last.ckpt')}})
            Q.R.dump(prefix.with_suffix('.json'),doc)
        stage=dict(end_to_end_seconds=.02,pose51_service_seconds=.015,events=50,output_mesh_total_seconds=.004,
            output_mesh_context_transfer_seconds=.001,output_mano_fk_seconds=.002,output_mesh_transfer_seconds=.001,
            ingest_seconds=.001,commit_seconds=.001,geometry_seconds=.001,readout_seconds=.001,representation_seconds=.001,
            context_transfer_seconds=.001,decode_including_observation_transfer_seconds=.001,prediction_transfer_seconds=.001)
        asset=str((ROOT/'assets/mano_right.npz').resolve())
        resource=dict(schema=Q.RESOURCE_SCHEMA,base_schema='s43_relation_pipeline_benchmark_v2_surface',arm=arm,
            shape_condition='public_mean_beta0',inference_beta=[0.]*10,aux_betas_accessed=False,pose_metadata_opened=False,
            tsub_opened=False,module_globals_restored=True,extension_sha256=Q.R.sha(extension),queries=64,query_hop_ms=50,
            recording_geometry_context='ch_local',engineering_only=False,records=[dict(stage,index=i) for i in range(64)],
            output_contract='sensor_to_mesh778_joints21',GT_read=False,output_surface_shapes={'vertices_m':[778,3],'joints_m':[21,3]},
            source_identities={asset:{'sha256':Q.R.sha(asset)}},source_files_unchanged={asset:True},
            outputs_per_second_measured_queries=50.,events_per_second_measured_queries=2500.,source_events_including_warmup=3300,
            source_id_digest='same',input_kind='native quantized recording prefix',
            latency=dict(n=64,mean_ms=20.,p50_ms=20.,p95_ms=20.,p99_ms=20.),pose51_latency={'n':64},
            cuda_peak_allocated_bytes=100,cuda_peak_reserved_bytes=200,RSS_peak_bytes=300,resources={},params={'total':2867613},
            reset_and_compile_seconds=0.,load_seconds=0.,warmup_seconds=.1,latency_scope='actual surface',
            checkpoint_sha256=None if arm=='canonical' else Q.R.sha(tmp_path/f'train/full_r1/{arm}/step=6000.ckpt'))
        Q.R.dump(root/f'resources/{arm}_h50.json',resource)
        Q.R.dump(root/f'executions/resource_{arm}.json',dict(NVML_device_memory_observation=dict(
            baseline_device_used_bytes=624*2**20,sampled_peak_device_used_bytes=900*2**20,sampled_peak_minus_baseline_bytes=276*2**20,samples=100)))
    result=Q.collect(root,extension)
    assert result['execution_complete'] and not result['conditional_scores_read']
    assert result['primary']['full']['overall'][Q.WRIST_V]==5.
    assert result['resources']['full']['NVML_device_memory_observation']['samples']==100
    assert result['posthoc_motion_strata']['status']=='MISSING_MASK_ARTIFACTS'
    Q.write(result,root)
    assert len([s for s in (root/'main_table.md').read_text().splitlines() if s.startswith('|')])==7
