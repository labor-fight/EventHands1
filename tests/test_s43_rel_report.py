"""Research-report contracts: weighted errors, strict gates and real timing."""
import copy
import math
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.s43_rel import report as R


def replay(recording='zgz_local',n=3,value=10.):
    return dict(schema='s43_relation_full_recording_replay_v1',arm='full',recording=recording,
        hop_ms=50,commit_schedule_ms=50,full_recording=True,all_valid_events_consumed_once=True,
        initialization_GT=False,observation_window_ms=50,output_lag_ms=1,source_count=7,
        expected_source_count=7,source_id_digest='same',expected_source_id_digest='same',
        source_files_unchanged={'frozen':True},outputs=n,
        metrics=dict(mpjpe_ra_mm=value,mpvpe_ra_mm=value+1,mpjpe_abs_mm=value+1000,mpvpe_abs_mm=value+1001))


def test_main_metric_weighting_uses_output_counts_and_never_absolute_errors():
    m=R.aggregate_primary([replay(n=3,value=10),replay('zgz_global',1,30)])
    assert m['overall']['mpjpe_ra_mm']==15
    assert m['overall']['mpjpe_abs_mm']==1015
    assert m['local']['mpjpe_ra_mm']==10 and m['global']['mpjpe_ra_mm']==30
    row=R.headline_row('test',m,{'latency_sensor_to_mesh':{'p50_ms':13.2},'params':{'total':2867613}})
    cells=[q.strip() for q in row.split('|')[1:-1]]
    assert len(cells)==9 and cells[1]=='10.00' and cells[5]=='15.00'
    assert cells[6]=='13.20 ms' and cells[7]=='-' and cells[8]=='2.87 M'


@pytest.mark.parametrize('field,bad',[('source_count',8),('initialization_GT',True),('full_recording',False),('source_files_unchanged',{'x':False})])
def test_report_refuses_incomplete_or_noncausal_replay_even_with_good_metrics(field,bad):
    r=replay();r[field]=bad
    with pytest.raises(R.InvalidArtifact):R.validate_replay(r,'full','zgz_local',50,50)


def benchmark():
    records=[]
    for i,t in enumerate((.02,.06)):
        records.append(dict(index=i,events=10,end_to_end_seconds=t,pose51_service_seconds=t-.005,
            output_mesh_total_seconds=.004,output_mano_fk_seconds=.002,output_mesh_transfer_seconds=.001,
            output_mesh_context_transfer_seconds=.001,ingest_seconds=.002,commit_seconds=0.,geometry_seconds=0.,
            readout_seconds=0.,representation_seconds=0.,context_transfer_seconds=0.,
            decode_including_observation_transfer_seconds=0.,prediction_transfer_seconds=0.))
    return dict(schema='s43_relation_pipeline_benchmark_v2_surface',output_contract='sensor_to_mesh778_joints21',
        output_surface_shapes={'vertices_m':[778,3],'joints_m':[21,3]},GT_read=False,pose_metadata_opened=False,
        source_files_unchanged={str((ROOT/'assets/mano_right.npz').resolve()):True},
        source_identities={str((ROOT/'assets/mano_right.npz').resolve()):{'sha256':R.sha(ROOT/'assets/mano_right.npz')}},
        records=records,queries=2,query_hop_ms=50,
        latency={'n':2,'p50_ms':40,'p95_ms':58,'p99_ms':59.6},pose51_latency={'n':2},
        outputs_per_second_measured_queries=25.,events_per_second_measured_queries=250.,
        input_kind='native quantized recording prefix',source_events_including_warmup=23,source_id_digest='same',
        cuda_peak_allocated_bytes=100,cuda_peak_reserved_bytes=200,RSS_peak_bytes=300,resources={},params={'total':2867613},
        reset_and_compile_seconds=0.,load_seconds=0.,warmup_seconds=.1,latency_scope='actual sensor to mesh')


def test_deadline_fraction_uses_each_measured_full_mesh_cycle():
    summary=R.benchmark_summary(benchmark())
    assert summary['deadline_exceedance_fraction']==.5
    assert summary['deadline_exceeded_queries']==1
    assert summary['outputs_per_second']==25
    assert summary['stage_maintenance']['output_mesh_total_seconds']['total_seconds']==.008


def test_pose51_or_scaled_proxy_cannot_be_used_as_surface_latency():
    r=benchmark();r['schema']='s43_relation_pipeline_benchmark_v1'
    with pytest.raises(R.InvalidArtifact):R.benchmark_summary(r)
    r=benchmark();r['outputs_per_second_measured_queries']=100.
    with pytest.raises(R.InvalidArtifact):R.benchmark_summary(r)


def test_partial_inventory_has_no_accuracy_aggregation_and_requires_all_cases(tmp_path):
    r=R.inventory(tmp_path,tmp_path/'S39.json')
    assert not r['complete_artifact_set'] and r['missing']
    assert r['expected_paired_replays']==40 and r['expected_canonical_replays']==4
    assert not r['metrics_aggregated'] and not r['final_report_generated']
    with pytest.raises(R.IncompleteReport):R.collect_final(tmp_path,tmp_path/'S39.json')


def test_final_case_contract_has10_cases_perarm_and_separates_commit_schedule():
    cases=R.replay_cases()
    assert len(cases)==10
    assert sum(c[0]=='full_final_h50' for c in cases)==4
    assert {(q[2],q[3]) for q in cases}=={(25,25),(50,50),(100,100),(25,50)}


def test_complete_negative_study_can_be_reported_without_forcing_science_success(tmp_path):
    import json
    import numpy as np
    root=tmp_path/'artifacts';history=tmp_path/'S39.json'
    def put(path,value):R.dump(path,value)
    paths,_=R.required_paths(root,history)
    for path in paths:
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.suffix=='.json':put(path,{})
        else:path.write_text('fixed deployment weights')
    old=R.aggregate_primary([replay(n=1),replay('zgz_global',1)])
    put(history,dict(two_seed_mean={k:old[k] for k in ('local','global','overall')},latency_ms_scaled_full1p75=4.,macs_forward_packet=1e9,params_total=2800000))
    put(root/'protocol.json',dict(frozen_existing_sources={'assets/mano_right.npz':R.sha(ROOT/'assets/mano_right.npz')}))
    put(root/'audits/full_training.json',dict(complete=True,verdict='PASS',hard_failures=[],input_pairing={'common_paired_optimizer_steps':6000}))
    put(root/'audits/full_training_online.json',dict(status='PASS',reader='joint_gram',checkpoint='/full_r1/full/last.ckpt',
        checkpoint_sha256=R.sha(root/'train/full_r1/full/last.ckpt'),source_files_unchanged=True,neural_weights_and_BN_buffers_unchanged=True))
    put(root/'robustness/full_final/report.json',dict(source_files_unchanged={'reader':True},GT_input_used=False))
    put(root/'audits/spatial_order_intervention.json',dict(status='PASS',does_not_remove_all_temporal_information=True))
    put(root/'audits/canonical_diagnostic.json',dict(status='FAIL',scope='cross-device diagnostic, retained'))
    for arm in R.ARMS:
        put(root/f'train/full_r1/{arm}/complete.json',dict(completed_steps=6000,actual_outputs=192000,source_files_unchanged=True,
            checkpoint_sha256=R.sha(root/f'train/full_r1/{arm}/last.ckpt')))
        cases=[]
        for tag,recording,hop,commit in R.replay_cases():
            cases.append(dict(tag=tag,recording=recording,hop=hop,commit=None if hop==commit else commit,code=0))
            value=30. if arm=='full' else 10.
            doc=replay(recording,n=2,value=value);doc.update(arm=arm,hop_ms=hop,commit_schedule_ms=commit,
                checkpoint=str((root/f'train/full_r1/{arm}/last.ckpt').resolve()))
            prefix=root/f'eval/{tag}/{arm}_{recording}_h{hop}'
            put(prefix.with_suffix('.json'),doc)
            end=np.arange(124,300,25) if hop==25 else np.arange(149,300,50) if hop==50 else np.array([199,299])
            doc['outputs']=len(end);put(prefix.with_suffix('.json'),doc)
            np.savez(prefix.with_suffix('.npz'),end_ms=end,run_id=np.zeros(len(end),int),pred=np.zeros((len(end),51),np.float32),
                gt=np.repeat(end[:,None],51,axis=1).astype(np.float32),
                **{key:np.full(len(end),doc['metrics'][key],np.float32) for key in R.METRICS})
        put(root/f'final_pipeline_{arm}_complete.json',cases)
    put(root/'eval/canonical_final_h50/s43_queue_complete.json',[dict(recording=r,returncode=0) for r in R.RECORDINGS])
    for recording in R.RECORDINGS:
        doc=replay(recording,n=4,value=11.);doc.update(arm='s43',checkpoint=None)
        prefix=root/f'eval/canonical_final_h50/s43_{recording}_h50'
        put(prefix.with_suffix('.json'),doc)
        np.savez(prefix.with_suffix('.npz'),end_ms=np.array([149,199,249,299]),run_id=np.zeros(4,int),pred=np.zeros((4,51),np.float32))
    tasks=[]
    for arm in ('canonical',)+R.ARMS:
        for hop in ((50,) if arm=='canonical' else (25,50,100)):
            doc=benchmark();n=3200//hop;record=copy.deepcopy(doc['records'][0]);record['events']=hop
            doc.update(arm='s43' if arm=='canonical' else arm,synthetic_events_per50ms=None,recording_geometry_context='ch_local',
                queries=n,query_hop_ms=hop,records=[dict(record,index=i) for i in range(n)],
                outputs_per_second_measured_queries=50.,events_per_second_measured_queries=hop*50.,
                source_events_including_warmup=3300,checkpoint_sha256=None if arm=='canonical' else R.sha(root/f'train/full_r1/{arm}/step=6000.ckpt'))
            doc['latency']['n']=n;doc['pose51_latency']['n']=n
            doc['latency'].update(mean_ms=20.,p50_ms=20.,p95_ms=20.,p99_ms=20.)
            put(root/f'benchmark/final/actual/{arm}_h{hop}.json',doc)
            tasks.append(dict(name='benchmark_actual_canonical_h50' if arm=='canonical' else f'benchmark_actual_{arm}_h{hop}',code=0))
    for arm in R.ARMS:
        for rate in (10000,50000,150000):
            for hop in (25,50,100):
                put(root/f'benchmark/final/synthetic/{arm}_n{rate}_h{hop}_failed.json',dict(code=1,scope='failed stress is an outcome'))
                tasks.append(dict(name=f'benchmark_synthetic_{arm}_n{rate}_h{hop}',code=1))
    tasks += [dict(name=n,code=0) for n in ('robustness_final','trained_prefix_final','full_training_audit')]
    put(root/'postprocess_complete.json',dict(complete=True,tasks=tasks))
    result=R.collect_final(root,history)
    assert result['execution_complete']
    assert result['scientific_outcome']=='NO_GO_ACCURACY_MARGIN'
    assert result['primary_zgz']['full']['local']['mpjpe_abs_mm']==1030.
    assert len(result['resources']['synthetic'])==36
    assert result['diagnostics']
    assert result['common_cadence_grid_summary']['status']=='PASS'
    R.write_final(result,tmp_path/'report')
    assert (tmp_path/'report/main_table.md').exists()


def test_extra_pure_queries_compare_only_shared_commits_and_never_relax_exactness(tmp_path):
    import numpy as np
    root=tmp_path/'data'
    R.dump(root/'train/full_r1/s43/complete.json',dict(completed_steps=6000))
    for recording in R.RECORDINGS[2:]:
        a=root/f'eval/full_final_h50/s43_{recording}_h50.npz';a.parent.mkdir(parents=True,exist_ok=True)
        b=root/f'eval/full_final_h25_fixed50/s43_{recording}_h25.npz';b.parent.mkdir(parents=True,exist_ok=True)
        np.savez(a,pred=np.zeros((2,51),np.float32),end_ms=[149,199],run_id=[0,0])
        predicted=np.zeros((4,51),np.float32);predicted[[0,2]]=7.  # extra noncommit outputs can differ
        np.savez(b,pred=predicted,end_ms=[124,149,174,199],run_id=[0,0,0,0])
    result=R.audit_fixed50_purequery(root,arms=('s43',))
    assert result['status']=='PASS' and result['cases'][0]['extra_noncommit_query_outputs']==2
    changed=np.zeros((4,51),np.float32);changed[1,0]=1e-7
    b=root/'eval/full_final_h25_fixed50/s43_zgz_local_h25.npz'
    np.savez(b,pred=changed,end_ms=[124,149,174,199],run_id=[0,0,0,0])
    result=R.audit_fixed50_purequery(root,arms=('s43',),out=root/'audit.json')
    assert result['status']=='FAIL' and not result['tolerances_changed']
    assert R.read(root/'audit.json')['status']=='FAIL'


def test_common_cadence_uses_irregular_run_time_intersection_not_row_indices(tmp_path):
    import numpy as np
    tags=('full_final_h25','full_final_h50','full_final_h100','full_final_h25_fixed50')
    grids=[[(1,199),(0,149),(0,199),(1,149)],[(0,199),(1,149),(1,199)],
           [(1,199),(0,199)],[(0,174),(1,199),(0,199),(1,174)]]
    for offset,(tag,addresses) in enumerate(zip(tags,grids)):
        hop=50 if tag=='full_final_h50' else 100 if tag=='full_final_h100' else 25
        for recording in R.RECORDINGS[2:]:
            path=tmp_path/f'eval/{tag}/s43_{recording}_h{hop}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            run,end=np.asarray(addresses).T
            gt=np.zeros((len(run),51),np.float32);gt[:,0]=run;gt[:,1]=end
            value=1000*run+end+offset*10
            np.savez(path,run_id=run,end_ms=end,gt=gt,**{key:value+j for j,key in enumerate(R.METRICS)})
    result=R.common_cadence_grid_summary(tmp_path,arms=('s43',))
    assert result['status']=='PASS'
    case=result['per_arm_recording']['s43']['zgz_local']
    assert case['intersection_count']==2 and case['GT_exact']
    assert case['shared_address_first5']==[(0,199),(1,199)]
    assert case['own_grid_outputs']==dict(zip(tags,[4,3,2,4]))
    for i,tag in enumerate(tags):
        assert case['common_grid_metrics'][tag]['mpjpe_ra_mm']==699+i*10
    # Preserve the common addresses but corrupt one cadence's shared GT only.
    path=tmp_path/'eval/full_final_h100/s43_zgz_local_h100.npz'
    with np.load(path) as z:values={key:z[key].copy() for key in z.files}
    values['gt'][0,3]=1e-7;np.savez(path,**values)
    result=R.common_cadence_grid_summary(tmp_path,arms=('s43',),out=tmp_path/'common.json')
    assert result['status']=='FAIL'
    assert not result['per_arm_recording']['s43']['zgz_local']['GT_exact']
    assert R.read(tmp_path/'common.json')['status']=='FAIL'
