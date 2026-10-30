"""Artifact-only final report: strict completion, honest negative conclusions.

No model, data loader, target reconstruction, checkpoint selection or training.
Partial/dry-run modes inventory artifacts and source counts without aggregating
accuracy. Final mode requires all four completed trainings,40 final paired
replays,4 canonical replays, surface-service benchmarks and required audits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from tools.report_table import HEADER,RULE,BASELINE

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'outputs/s43_rel_20261027'
ARMS=('s43','enhanced','no_order','full')
RECORDINGS=('ycy_local','ycy_global','zgz_local','zgz_global')
METRICS=('mpjpe_ra_mm','mpvpe_ra_mm','mpjpe_abs_mm','mpvpe_abs_mm')
LABELS={'canonical':'冻结原始S43（公共初始化）','s43':'S43 配对FT6000',
        'enhanced':'S43＋增强20ch FT6000','no_order':'S43＋空间无向Z R1 FT6000',
        'full':'S43＋有向Z R1 FT6000'}
DEFAULT_HISTORY=ROOT/'outputs/semkine/trk6_s39ref_w200cfg_main_row.json'


class IncompleteReport(RuntimeError):pass
class InvalidArtifact(RuntimeError):pass


def read(path):return json.loads(Path(path).read_text())
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def require(value,message):
    if not value:raise InvalidArtifact(message)


def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
def source_guard(value):return value is True or isinstance(value,dict) and bool(value) and all(v is True for v in value.values())


def replay_cases():
    result=[('full_final_h50',r,50,50) for r in RECORDINGS]
    for hop in (25,100):result += [(f'full_final_h{hop}',r,hop,hop) for r in RECORDINGS[2:]]
    result += [('full_final_h25_fixed50',r,25,50) for r in RECORDINGS[2:]]
    return result


def required_paths(root,history):
    required=[root/'protocol.json',root/'correction_R1_prereg.json',root/'postprocess_complete.json',
        root/'audits/full_training.json',root/'audits/full_training_online.json',
        root/'audits/spatial_order_intervention.json',
        root/'audits/small_training.json',root/'robustness/full_final/report.json',
        root/'robustness/full_final/manifest_before_readouts.json',root/'robustness/full_final/sensor_prefix_audits.json',
        root/'robustness/full_final/predictions_before_scoring.npz',
        root/'mechanisms/e12/summary.json',root/'mechanisms/e12/R1/summary.json',root/'mechanisms/e3/summary.json',history,
        root/'eval/canonical_final_h50/s43_queue_complete.json']
    for arm in ARMS:
        required += [root/f'train/full_r1/{arm}/complete.json',root/f'train/full_r1/{arm}/manifest.json',
            root/f'train/full_r1/{arm}/curve.jsonl',root/f'train/full_r1/{arm}/last.ckpt',
            root/f'train/full_r1/{arm}/step=6000.ckpt',root/f'final_pipeline_{arm}_complete.json']
        for tag,recording,hop,_ in replay_cases():
            prefix=root/f'eval/{tag}/{arm}_{recording}_h{hop}'
            required += [prefix.with_suffix('.json'),prefix.with_suffix('.npz')]
        for hop in (25,50,100):required.append(root/f'benchmark/final/actual/{arm}_h{hop}.json')
    for recording in RECORDINGS:
        prefix=root/f'eval/canonical_final_h50/s43_{recording}_h50'
        required += [prefix.with_suffix('.json'),prefix.with_suffix('.npz')]
    required.append(root/'benchmark/final/actual/canonical_h50.json')
    alternatives=[]
    for arm in ARMS:
        for rate in (10000,50000,150000):
            for hop in (25,50,100):
                prefix=root/f'benchmark/final/synthetic/{arm}_n{rate}_h{hop}'
                alternatives.append((prefix.with_suffix('.json'),prefix.with_name(prefix.name+'_failed').with_suffix('.json')))
    return required,alternatives


def inventory(root,history):
    required,alternatives=required_paths(root,history)
    missing=[str(p) for p in required if not p.exists()]
    missing += [' OR '.join(map(str,pair)) for pair in alternatives if not any(p.exists() for p in pair)]
    available=[]
    # Metadata only, including completed subsets: do not calculate partial zgz
    # accuracy or infer final scientific status from an unfinished arm.
    for arm in ARMS:
        for tag,recording,hop,_ in replay_cases():
            path=root/f'eval/{tag}/{arm}_{recording}_h{hop}.json'
            if path.exists():
                r=read(path)
                available.append(dict(arm=arm,tag=tag,recording=recording,source_count=r.get('source_count'),
                    expected_source_count=r.get('expected_source_count'),outputs=r.get('outputs'),
                    source_count_matches=r.get('source_count')==r.get('expected_source_count'),
                    source_digest_matches=r.get('source_id_digest')==r.get('expected_source_id_digest')))
    return dict(schema='s43_relation_report_completion_v1',mode='inventory',
        complete_artifact_set=not missing,missing=missing,required_single_paths=len(required),
        required_synthetic_outcomes=len(alternatives),present_paired_replays=len(available),
        expected_paired_replays=40,expected_canonical_replays=4,available_replay_metadata=available,
        metrics_aggregated=False,final_report_generated=False)


def validate_replay(r,arm,recording,hop,commit):
    require(r.get('schema')=='s43_relation_full_recording_replay_v1','replay schema mismatch')
    require((r.get('arm'),r.get('recording'),r.get('hop_ms'),r.get('commit_schedule_ms'))==(arm,recording,hop,commit),'replay identity/schedule mismatch')
    require(r.get('full_recording') is True and r.get('all_valid_events_consumed_once') is True,'replay is not complete once-only source traversal')
    require(r.get('initialization_GT') is False and r.get('observation_window_ms')==50 and r.get('output_lag_ms')==1,'replay causal initialization/W50/lag mismatch')
    require(r.get('source_count')==r.get('expected_source_count'),'source count mismatch')
    require(bool(r.get('source_id_digest')) and r.get('source_id_digest')==r.get('expected_source_id_digest'),'source digest mismatch')
    require(source_guard(r.get('source_files_unchanged')),'replay source guard failed')
    require(isinstance(r.get('outputs'),int) and r['outputs']>0,'replay has no scored outputs')
    for key in METRICS:require(finite(r.get('metrics',{}).get(key)) and r['metrics'][key]>=0,'nonfinite/negative replay metric '+key)
    return r


def weighted_metrics(reports):
    require(bool(reports),'empty metric aggregation')
    n=sum(r['outputs'] for r in reports)
    require(n>0,'zero outputs in weighted aggregation')
    return dict(n_outputs=n,**{key:sum(r['outputs']*r['metrics'][key] for r in reports)/n for key in METRICS})


def aggregate_primary(reports):
    local=[r for r in reports if r['recording'].endswith('_local')]
    global_=[r for r in reports if r['recording'].endswith('_global')]
    return {'local':weighted_metrics(local),'global':weighted_metrics(global_),'overall':weighted_metrics(reports),
        'weighting':'outputs/frame-weighted; frozen root_align subtracts index0 of each array: joint0 for MPJPE, vertex0 for MPVPE; absolute metrics retained separately'}


def check_grid(path,outputs):
    import numpy as np
    with np.load(path,allow_pickle=False) as z:
        end=z['end_ms'].copy();run=z['run_id'].copy()
        require(end.shape==(outputs,) and run.shape==(outputs,),'replay target-address array length mismatch')
        require(np.all(np.isfinite(end)) and np.all(np.isfinite(run)),'nonfinite output addresses')
    return end,run


def audit_fixed50_purequery(root,arms=ARMS,out=None):
    """Compare only shared commit addresses; extra noncommit predictions differ.

    No targets or error arrays are opened. Failure is recorded at exact array
    equality, without introducing a numerical tolerance or rescoring accuracy.
    """
    import numpy as np
    cases=[]
    for arm in arms:
        receipt=read(root/f'train/full_r1/{arm}/complete.json')
        require(receipt.get('completed_steps')==6000,'purequery check requires a completed final arm')
        for recording in RECORDINGS[2:]:
            nominal=root/f'eval/full_final_h50/{arm}_{recording}_h50.npz'
            extra=root/f'eval/full_final_h25_fixed50/{arm}_{recording}_h25.npz'
            with np.load(nominal,allow_pickle=False) as z:
                original=z['pred'].copy();end=z['end_ms'].copy();run=z['run_id'].copy()
            with np.load(extra,allow_pickle=False) as z:
                queried=z['pred'].copy();end25=z['end_ms'].copy();run25=z['run_id'].copy()
            addresses=list(zip(run.tolist(),end.tolist()));addresses25=list(zip(run25.tolist(),end25.tolist()))
            unique=len(set(addresses))==len(addresses) and len(set(addresses25))==len(addresses25)
            index={key:i for i,key in enumerate(addresses25)}
            matched=[(i,index[key]) for i,key in enumerate(addresses) if key in index]
            subset=len(matched)==len(addresses)
            left=original[[i for i,_ in matched]];right=queried[[j for _,j in matched]]
            valid=original.shape==(len(end),51) and queried.shape==(len(end25),51) and np.isfinite(original).all() and np.isfinite(queried).all()
            exact=bool(valid and subset and unique and np.array_equal(left,right))
            max_abs=float(np.max(np.abs(left-right))) if valid and left.size else None
            cases.append(dict(arm=arm,recording=recording,status='PASS' if exact else 'FAIL',
                purequery_shared_commit_predictions_array_exact=exact,nominal_outputs=len(addresses),
                fixed50_query25_outputs=len(addresses25),matched_shared_commit_outputs=len(matched),
                unmatched_nominal_outputs=len(addresses)-len(matched),
                extra_noncommit_query_outputs=len(addresses25)-len(matched),unique_addresses=unique,finite_pose51_shapes=bool(valid),
                max_abs_pose51_difference_at_shared_commits=max_abs,nominal_npz=str(nominal),query25_npz=str(extra),
                nominal_npz_sha256=sha(nominal),query25_npz_sha256=sha(extra)))
    result=dict(schema='s43_rel_fixed50_purequery_exact_audit_v1',
        status='PASS' if all(q['status']=='PASS' for q in cases) else 'FAIL',cases=cases,
        definition='Match end_ms/run_id shared50ms commits. Require np.array_equal(pred). Additional25ms noncommit outputs are not compared to nonexistent50ms outputs.',
        tolerances_changed=False,GT_arrays_opened=False,accuracy_metrics_computed=False)
    if out is not None:dump(out,result)
    return result


def common_cadence_grid_summary(root,arms=ARMS,recordings=RECORDINGS[2:],out=None):
    """Summarize saved errors at the four cadence grids' address intersection.

    Address selection uses only (run_id,end_ms), never GT, error, motion or
    confidence. Shared GT must be array-exact. This is an attribution summary,
    without FK reruns, new thresholds or replacement of full-own-grid results.
    """
    import numpy as np
    tags=('full_final_h25','full_final_h50','full_final_h100','full_final_h25_fixed50')
    by_arm={};aggregate={};all_cases=[]
    for arm in arms:
        by_arm[arm]={};aggregate[arm]={}
        for recording in recordings:
            streams={}
            for tag in tags:
                hop=50 if tag=='full_final_h50' else 100 if tag=='full_final_h100' else 25
                path=root/f'eval/{tag}/{arm}_{recording}_h{hop}.npz'
                with np.load(path,allow_pickle=False) as z:
                    end=z['end_ms'].copy();run=z['run_id'].copy();gt=z['gt'].copy()
                    errors={key:z[key].copy() for key in METRICS}
                n=len(end)
                require(run.shape==(n,) and gt.shape==(n,51),'common-cadence address/GT shapes invalid')
                addresses=list(zip(run.tolist(),end.tolist()))
                require(len(set(addresses))==n,'duplicate cadence output address')
                require(all(a.shape==(n,) and np.isfinite(a).all() and (a>=0).all() for a in errors.values()),'saved per-frame error vector invalid')
                streams[tag]=dict(path=path,addresses=addresses,index={key:i for i,key in enumerate(addresses)},
                    gt=gt,errors=errors,n=n)
            shared=sorted(set.intersection(*(set(q['addresses']) for q in streams.values())))
            n=len(shared);indices={tag:np.asarray([q['index'][key] for key in shared],np.int64) for tag,q in streams.items()}
            shared_gt={tag:q['gt'][indices[tag]] for tag,q in streams.items()}
            reference_gt=shared_gt['full_final_h100']
            exact={tag:bool(np.array_equal(gt,reference_gt)) for tag,gt in shared_gt.items()}
            gt_exact=all(exact.values()) if n else None
            metrics={}
            if n and gt_exact:
                metrics={tag:dict(n_outputs=n,**{key:float(np.mean(q['errors'][key][indices[tag]],dtype=np.float64)) for key in METRICS})
                    for tag,q in streams.items()}
            status='PASS' if n and gt_exact else 'FAIL' if n else 'NO_COMMON_OUTPUTS'
            case=dict(status=status,intersection_count=n,GT_exact=gt_exact,GT_exact_by_cadence=exact if n else {},
                own_grid_outputs={tag:q['n'] for tag,q in streams.items()},common_grid_metrics=metrics,
                shared_address_sha256=hashlib.sha256(np.asarray(shared,dtype='<i8').reshape(-1,2).tobytes()).hexdigest(),
                shared_address_first5=shared[:5],shared_address_last5=shared[-5:],
                source_npz={tag:dict(path=str(q['path']),sha256=sha(q['path'])) for tag,q in streams.items()})
            by_arm[arm][recording]=case;all_cases.append(case)
        for tag in tags:
            rows=[dict(recording=recording,outputs=case['intersection_count'],metrics=case['common_grid_metrics'][tag])
                for recording,case in by_arm[arm].items() if case['status']=='PASS']
            aggregate[arm][tag]={domain:weighted_metrics(group) if group else None for domain,group in
                [('local',[r for r in rows if r['recording'].endswith('_local')]),
                 ('global',[r for r in rows if r['recording'].endswith('_global')]),('overall',rows)]}
    status='FAIL' if any(c['status']=='FAIL' for c in all_cases) else 'PASS_WITH_MISSING_GRID_EVIDENCE' if any(c['status']=='NO_COMMON_OUTPUTS' for c in all_cases) else 'PASS'
    result=dict(schema='s43_rel_common_cadence_grid_summary_v1',status=status,per_arm_recording=by_arm,
        aggregate_common_grid=aggregate,cadence_tags=list(tags),source_sha256=sha(__file__),
        selection='Exact set intersection of(run_id,end_ms) across all four grids; per-run100ms common output addresses, without GT/error filtering.',
        arrays_opened=['gt','end_ms','run_id']+list(METRICS),prediction_or_FK_recomputed=False,
        metric_arithmetic='float64 mean of the saved per-frame mm error arrays, weighted by common-frame counts for local/global aggregation',
        GT_used_for_selection=False,tolerances_changed=False,new_scientific_decision_gate=False,
        GT_mismatch_is_integrity_failure=True,original_own_grid_cadences_retained=True,
        interpretation='Descriptive same-target-time comparison. Commit/feedback schedules still differ; this is not a new independent test set or a causal isolation of every cadence effect.')
    if out is not None:dump(out,result)
    return result


def benchmark_summary(r):
    import numpy as np
    require(r.get('schema')=='s43_relation_pipeline_benchmark_v2_surface','primary resources require surface-complete v2 benchmark')
    require(r.get('output_contract')=='sensor_to_mesh778_joints21','resource artifact stops before final MANO surface')
    require(r.get('GT_read') is False,'resource benchmark read a target')
    require(r.get('pose_metadata_opened') is False,'resource benchmark opened pose metadata')
    require(source_guard(r.get('source_files_unchanged')),'resource benchmark source guard failed')
    identities=r.get('source_identities',{});asset=str((ROOT/'assets/mano_right.npz').resolve())
    require(identities.get(asset,{}).get('sha256')==sha(asset),'resource benchmark MANO asset identity mismatch')
    require(not any(str(path).endswith('.meta') for path in identities),'target metadata was included in resource source identities')
    require(r.get('output_surface_shapes')=={'vertices_m':[778,3],'joints_m':[21,3]},'surface output shapes mismatch')
    records=r.get('records',[]);n=r.get('queries');hop=r.get('query_hop_ms')
    require(n==len(records) and n>0 and hop in (25,50,100),'benchmark query count/cadence invalid')
    required=('end_to_end_seconds','pose51_service_seconds','output_mesh_context_transfer_seconds',
        'output_mano_fk_seconds','output_mesh_transfer_seconds','output_mesh_total_seconds',
        'ingest_seconds','commit_seconds','geometry_seconds','readout_seconds','representation_seconds',
        'context_transfer_seconds','decode_including_observation_transfer_seconds','prediction_transfer_seconds')
    for q in records:
        require(all(finite(q.get(k)) and q[k]>=0 for k in required),'missing/nonfinite surface stage cost')
        require(q['end_to_end_seconds']+1e-9>=q['pose51_service_seconds']+q['output_mesh_total_seconds'],'surface wall excludes a measured service component')
    seconds=[q['end_to_end_seconds'] for q in records]
    total=sum(seconds);source=sum(q['events'] for q in records)
    require(total>0,'zero measured service time')
    require(source<=r['source_events_including_warmup'],'measured-query events exceed complete benchmark source')
    require(math.isclose(r['outputs_per_second_measured_queries'],n/total,rel_tol=1e-8),'outputs/sec does not use actual mesh service wall')
    require(math.isclose(r['events_per_second_measured_queries'],source/total,rel_tol=1e-8),'events/sec does not use actual mesh service wall')
    require(r.get('latency',{}).get('n')==n and r.get('pose51_latency',{}).get('n')==n,'latency summary query counts mismatch')
    def latency(values,recorded):
        v=np.asarray(values,np.float64)*1000
        measured=dict(n=n,mean_ms=float(v.mean()),p50_ms=float(np.percentile(v,50)),
            p95_ms=float(np.percentile(v,95)),p99_ms=float(np.percentile(v,99)))
        for key,value in recorded.items():
            if key in measured:require(math.isclose(value,measured[key],rel_tol=1e-6,abs_tol=1e-6),'latency summary differs from measured records: '+key)
        return measured
    mesh_latency=latency(seconds,r['latency'])
    pose_latency=latency([q['pose51_service_seconds'] for q in records],r['pose51_latency'])
    stage_keys=sorted({key for q in records for key,v in q.items() if key.endswith('_seconds') and finite(v)})
    stages={key:dict(total_seconds=sum(q.get(key,0.) for q in records),
        mean_ms=1000*sum(q.get(key,0.) for q in records)/n) for key in stage_keys}
    exceeded=sum(t>hop/1000 for t in seconds)
    return dict(input_kind=r['input_kind'],query_hop_ms=hop,queries=n,latency_sensor_to_mesh=mesh_latency,
        latency_pose51=pose_latency,deadline_exceeded_queries=exceeded,deadline_exceedance_fraction=exceeded/n,
        device=r.get('device'),CUDA_VISIBLE_DEVICES=r.get('CUDA_VISIBLE_DEVICES'),seed=r.get('seed'),
        recording_geometry_context=r.get('recording_geometry_context'),start_ms=r.get('start_ms'),cutoff_ms=r.get('cutoff_ms'),
        GT_read=False,pose_metadata_opened=False,source_identities=identities,source_files_unchanged=r['source_files_unchanged'],
        deadline_definition='strict actual end_to_end_seconds > query_hop_ms/1000; physical sensor waiting excluded',
        events_per_second=r['events_per_second_measured_queries'],outputs_per_second=r['outputs_per_second_measured_queries'],
        source_events_measured_queries=source,source_events_including_warmup=r['source_events_including_warmup'],
        source_id_digest=r['source_id_digest'],stage_maintenance=stages,
        stage_accounting='pose51_service and output_mesh_total are parent totals; do not add them again to their child stages.',
        cuda_peak_allocated_bytes=r['cuda_peak_allocated_bytes'],cuda_peak_reserved_bytes=r['cuda_peak_reserved_bytes'],
        RSS_peak_bytes=r['RSS_peak_bytes'],resources=r['resources'],params=r['params'],
        reset_and_compile_seconds=r['reset_and_compile_seconds'],load_seconds=r['load_seconds'],warmup_seconds=r['warmup_seconds'],
        scope=r['latency_scope'],checkpoint_sha256=r.get('checkpoint_sha256'),deployment_sha256=r.get('deployment_sha256'),
        benchmark_sha256=r.get('benchmark_sha256'),replay_sha256=r.get('replay_sha256'))


def headline_row(label,metrics,resource):
    latency=resource['latency_sensor_to_mesh']['p50_ms'];params=resource['params']['total']
    return (f"| {label} | {metrics['local']['mpjpe_ra_mm']:.2f} | {metrics['global']['mpjpe_ra_mm']:.2f} "
        f"| {metrics['local']['mpvpe_ra_mm']:.2f} | {metrics['global']['mpvpe_ra_mm']:.2f} "
        f"| {metrics['overall']['mpjpe_ra_mm']:.2f} | {latency:.2f} ms | - | {params/1e6:.2f} M |")


def historical_row(history):
    m=history['two_seed_mean'];lat=history.get('latency_ms_scaled_full1p75');macs=history.get('macs_forward_packet')
    return (f"| S39 历史上下文（w200/S39栈，协议不同） | {m['local']['mpjpe_ra_mm']:.2f} | {m['global']['mpjpe_ra_mm']:.2f} "
        f"| {m['local']['mpvpe_ra_mm']:.2f} | {m['global']['mpvpe_ra_mm']:.2f} | {m['overall']['mpjpe_ra_mm']:.2f} "
        f"| {lat:.2f} ms | {macs/1e9:.3f} G | {history['params_total']/1e6:.2f} M |")


def validate_postprocess(root):
    post=read(root/'postprocess_complete.json')
    require(post.get('complete') is True,'postprocess is incomplete')
    tasks={r['name']:r for r in post.get('tasks',[])}
    names=[f'benchmark_actual_{arm}_h{h}' for arm in ARMS for h in (25,50,100)]
    names += ['benchmark_actual_canonical_h50','robustness_final','trained_prefix_final','full_training_audit']
    names += [f'benchmark_synthetic_{arm}_n{n}_h{h}' for arm in ARMS for n in (10000,50000,150000) for h in (25,50,100)]
    require(set(names)<=tasks.keys(),'postprocess lacks mandatory task execution receipts')
    require(all(tasks[n].get('code')==0 for n in names if not n.startswith('benchmark_synthetic_')),'mandatory actual service/audit task failed')
    training=read(root/'audits/full_training.json')
    require(training.get('complete') is True and training.get('verdict','').startswith('PASS') and not training.get('hard_failures'),'full training audit did not pass')
    require(training.get('input_pairing',{}).get('common_paired_optimizer_steps')==6000,'full source-pairing audit incomplete')
    online=read(root/'audits/full_training_online.json')
    require(online.get('status')=='PASS' and online.get('reader')=='joint_gram','trained training/online prefix audit did not pass')
    require(online.get('source_files_unchanged') is True and online.get('neural_weights_and_BN_buffers_unchanged') is True,'trained-prefix audit changed weights or sources')
    require('/full_r1/full/' in str(online.get('checkpoint','')),'online parity audit is not for final full checkpoint')
    robustness=read(root/'robustness/full_final/report.json')
    require(source_guard(robustness.get('source_files_unchanged')) and robustness.get('GT_input_used') is False,'robustness source/input guard failed')
    return dict(postprocess=post,training=training,training_online=online,robustness=robustness)


def collect_final(root,history_path):
    import numpy as np
    state=inventory(root,history_path)
    if state['missing']:raise IncompleteReport(str(len(state['missing']))+' required artifact(s)/outcome(s) are absent')
    audits=validate_postprocess(root)
    frozen_asset=read(root/'protocol.json').get('frozen_existing_sources',{}).get('assets/mano_right.npz')
    require(frozen_asset==sha(ROOT/'assets/mano_right.npz'),'current MANO asset differs from preregistered identity')
    spatial=read(root/'audits/spatial_order_intervention.json')
    require(spatial.get('status')=='PASS' and spatial.get('does_not_remove_all_temporal_information') is True,'spatial-order control audit incomplete')
    audits['spatial_order_intervention']=spatial
    replay_data={};source_by_recording={};grids={};training={}
    for arm in ARMS:
        receipt=read(root/f'train/full_r1/{arm}/complete.json')
        require(receipt.get('completed_steps')==6000 and receipt.get('actual_outputs')==192000 and receipt.get('source_files_unchanged') is True,'training completion invalid: '+arm)
        training[arm]=receipt
        pipeline=read(root/f'final_pipeline_{arm}_complete.json')
        expected={(t,r,h,None if c==h else c) for t,r,h,c in replay_cases()}
        actual={(q['tag'],q['recording'],q['hop'],q.get('commit')) for q in pipeline}
        require(len(pipeline)==10 and actual==expected and all(q.get('code')==0 for q in pipeline),'final pipeline does not certify exact10 successful replays: '+arm)
        replay_data[arm]={}
        for tag,recording,hop,commit in replay_cases():
            path=root/f'eval/{tag}/{arm}_{recording}_h{hop}.json'
            r=validate_replay(read(path),arm,recording,hop,commit)
            require(r.get('checkpoint') and Path(r['checkpoint']).resolve()==(root/f'train/full_r1/{arm}/last.ckpt').resolve(),'replay checkpoint is not fixed final last: '+str(path))
            grid=check_grid(path.with_suffix('.npz'),r['outputs']);key=(tag,recording)
            if key in grids:require(all(np.array_equal(a,b) for a,b in zip(grids[key],grid)),'paired replay output addresses differ')
            else:grids[key]=grid
            source=(r['source_count'],r['source_id_digest'])
            if recording in source_by_recording:require(source_by_recording[recording]==source,'full source traversal differs across arms/cadences')
            else:source_by_recording[recording]=source
            replay_data[arm][tag+'/'+recording]=r
    require(audits['training_online'].get('checkpoint_sha256')==training['full'].get('checkpoint_sha256'),'training/online audit is not bound to the final full checkpoint')
    purequery=audit_fixed50_purequery(root,out=root/'audits/final_fixed50_purequery.json')
    require(purequery['status']=='PASS','shared50ms commit predictions changed after extra uncommitted25ms queries; preserved exact FAIL audit')
    audits['fixed50_purequery']=purequery
    common_grid=common_cadence_grid_summary(root,out=root/'audits/final_common_cadence_grid.json')
    require(common_grid['status']!='FAIL','shared cadence addresses have mismatching GT; preserved exact FAIL summary')
    canonical={}
    queue=read(root/'eval/canonical_final_h50/s43_queue_complete.json')
    require(len(queue)==4 and {q['recording'] for q in queue}==set(RECORDINGS) and all(q.get('returncode')==0 for q in queue),'canonical queue is not four successful full recordings')
    for recording in RECORDINGS:
        path=root/f'eval/canonical_final_h50/s43_{recording}_h50.json'
        r=validate_replay(read(path),'s43',recording,50,50)
        require(r.get('checkpoint') is None,'frozen canonical unexpectedly has an FT checkpoint')
        require(source_by_recording[recording]==(r['source_count'],r['source_id_digest']),'canonical source differs from paired controls')
        grid=check_grid(path.with_suffix('.npz'),r['outputs'])
        require(all(np.array_equal(a,b) for a,b in zip(grids['full_final_h50',recording],grid)),'canonical and paired target grids differ')
        canonical[recording]=r
    resources={'actual':{},'synthetic':{}}
    actual_raw={}
    for label in ('canonical',)+ARMS:
        resources['actual'][label]={}
        for hop in ((50,) if label=='canonical' else (25,50,100)):
            path=root/f'benchmark/final/actual/{label}_h{hop}.json'
            raw=read(path);require(raw.get('synthetic_events_per50ms') is None,'actual resource row is synthetic stress')
            require(raw.get('arm')==('s43' if label=='canonical' else label),'benchmark arm mismatch')
            require(raw.get('query_hop_ms')==hop,'benchmark hop mismatch')
            require(raw.get('queries',0)*hop==3200 and raw.get('recording_geometry_context')=='ch_local','actual resource row is not the fixed3.2s ch prefix')
            require((raw.get('checkpoint_sha256') is None)==(label=='canonical'),'benchmark canonical/FT checkpoint role mismatch')
            if label!='canonical':
                require(raw['checkpoint_sha256']==sha(root/f'train/full_r1/{label}/step=6000.ckpt'),'resource benchmark is not the fixed final deployment weights')
            resources['actual'][label][str(hop)]=dict(benchmark_summary(raw),source_path=str(path),source_sha256=sha(path))
            actual_raw[label,hop]=raw
    actual_sources={(q['source_events_including_warmup'],q['source_id_digest']) for q in actual_raw.values()}
    require(len(actual_sources)==1,'actual resource benchmark prefixes differ across arms/hops')
    for arm in ARMS:
        for rate in (10000,50000,150000):
            for hop in (25,50,100):
                prefix=root/f'benchmark/final/synthetic/{arm}_n{rate}_h{hop}'
                path=prefix.with_suffix('.json');failure=prefix.with_name(prefix.name+'_failed').with_suffix('.json')
                require(not (path.exists() and failure.exists()),'ambiguous success/failure synthetic outcome')
                key=f'{arm}/n{rate}/h{hop}'
                if path.exists():
                    raw=read(path);require(raw.get('synthetic_events_per50ms')==rate and raw.get('arm')==arm and raw.get('query_hop_ms')==hop,'synthetic workload identity mismatch')
                    require(raw.get('queries',0)*hop==400,'synthetic resource row is not the fixed400ms stress workload')
                    resources['synthetic'][key]=dict(status='completed',**benchmark_summary(raw),source_path=str(path))
                else:
                    fail=read(failure);require(fail.get('code') not in (None,0),'synthetic failure lacks nonzero execution receipt')
                    resources['synthetic'][key]=dict(status='failed_workload',execution=fail,source_path=str(failure))
    primary={'canonical':aggregate_primary([canonical[r] for r in RECORDINGS[2:]])}
    select={'canonical':aggregate_primary([canonical[r] for r in RECORDINGS[:2]])}
    cadences={}
    for arm in ARMS:
        primary[arm]=aggregate_primary([replay_data[arm]['full_final_h50/'+r] for r in RECORDINGS[2:]])
        select[arm]=aggregate_primary([replay_data[arm]['full_final_h50/'+r] for r in RECORDINGS[:2]])
        cadences[arm]={tag:aggregate_primary([replay_data[arm][tag+'/'+r] for r in RECORDINGS[2:]])
            for tag in ('full_final_h25','full_final_h50','full_final_h100','full_final_h25_fixed50')}
    decisions={}
    for arm in ('enhanced','no_order','full'):
        decisions[arm]={}
        for domain in ('local','global'):
            baseline=primary['s43'][domain]['mpjpe_ra_mm'];value=primary[arm][domain]['mpjpe_ra_mm'];margin=max(.5,.05*baseline)
            decisions[arm][domain]=dict(paired_baseline_mm=baseline,arm_mm=value,delta_mm=value-baseline,noninferiority_margin_mm=margin,noninferior=value<=baseline+margin)
    mechanisms={name:read(root/path) for name,path in [('R0_E12','mechanisms/e12/summary.json'),('R1_E12','mechanisms/e12/R1/summary.json'),('E3','mechanisms/e3/summary.json')]}
    diagnostic={}
    for path in sorted((root/'audits').glob('*canonical*.json')):
        doc=read(path)
        if isinstance(doc,dict) and ('status' in doc or 'verdict' in doc):diagnostic[str(path)]=dict(source_sha256=sha(path),artifact=doc)
    outcome='NONINFERIOR_DESCRIPTIVE' if all(q['noninferior'] for q in decisions['full'].values()) else 'NO_GO_ACCURACY_MARGIN'
    failed_stress=sum(q['status']=='failed_workload' for q in resources['synthetic'].values())
    result=dict(schema='s43_relation_final_results_v1',execution_complete=True,scientific_outcome=outcome,
        scientific_outcome_definition='Only the preregistered paired RA-MPJPE noninferiority test; not overall validation of mechanism, service capacity or novelty.',
        source_sha256=sha(__file__),created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        protocol_sha256=sha(root/'protocol.json'),correction_R1_sha256=sha(root/'correction_R1_prereg.json'),
        primary_protocol='seed3407, fixed final6000, complete valid streams, W50/raw, public initial state + two common frozen canonical W50 warmups, hop50/commit50,1ms retrospective output',
        main_table_metrics='Frozen root_align(x)=x-x[..., :1,:]: MPJPE aligned at joint0/wrist, MPVPE independently at vertex0. Overall RA frame/output-weighted across zgz local/global; absolute metrics retained.',
        main_table_resources='actual synchronized sensor-to-MANO778vertices/21joints CPU output p50 on fixed ch_local3.2s measured prefix; physical waiting/serialization excluded; nominal wrapper parameters; FLOPs unmeasured.',
        primary_zgz=primary,training_recording_ycy=select,SELECT_ycy=select,
        legacy_key_roles={'SELECT_ycy':'Legacy key only: ycy/ylf selected during small-stage, then included among all9 full-training subjects. Final ycy scores are training-recording diagnostics, not heldout SELECT.'},
        test_independence={'heldout_subjects':1,'heldout_subject':'zgz','recordings':2,'seeds':1},
        cadences=cadences,paired_noninferiority=decisions,resources=resources,
        common_cadence_grid_summary=common_grid,
        resource_evidence={'synthetic_completed_workloads':36-failed_stress,'synthetic_failed_workloads':failed_stress,
            'full_actual_h50_deadline_exceedance_fraction':resources['actual']['full']['50']['deadline_exceedance_fraction'],
            'memory_scope':'Measured process includes the retained frozen canonical warmup model plus the evaluated wrapper and workspaces. Params cell counts the evaluated nominal wrapper only.'},
        replays=replay_data,canonical_replays=canonical,training=training,mechanisms=mechanisms,
        audits=audits,diagnostics=diagnostic,
        source_accounting=dict(per_recording={k:dict(rows=v[0],source_id_digest=v[1]) for k,v in source_by_recording.items()},
            unique_native_rows_across_four_recordings=sum(v[0] for v in source_by_recording.values()),
            paired_replay_runs=40,canonical_replay_runs=4,
            interpretation='Controls/cadences intentionally reconsume the same source. Replays/windows are not independent recordings or subjects.'),
        historical_S39=dict(path=str(history_path),source_sha256=sha(history_path),artifact=read(history_path),
            protocol='historical w200/S39-stack and historical initialization/filtering/cost scaling; separate context, no direct paired inference'),
        limitations=['One seed, zgz remains project development/test identity; no inferential multi-subject claim.',
            'Final ycy recordings are in full-training data (all9subjects); their former small-stageSELECT role does not make final scores heldout.',
            'Existing recordings are v2e provenance, not a verified real-event-sensor test.',
            'Original negative mechanisms and R1 adaptation limits remain; favorable final accuracy does not establish a novel two-point statistic.',
            'Full-recording replay latency ends at pose51; scoring MANO FK was offline. Primary service latency comes only from new v2_surface benchmark.',
            'Canonical cross-CPU/GPU diagnostic discrepancies are disclosed, without presuming ordinary floating-point error. Four-arm same-backend comparisons are distinct.',
            '192000 FT outputs/arm are3.125% of historical6144000 examples, not a fresh historical-budget retrain.',
            'No measured whole-pipeline FLOPs; no scaled legacy latency is substituted for surface service.',
            'A negative or conditional scientific outcome is a completed study, not a missing artifact.'])
    return result


def write_final(result,out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError('preserve prior final results: '+str(out/'results.json'))
    lines=[HEADER,RULE,BASELINE]
    for key in ('canonical',)+ARMS:
        lines.append(headline_row(LABELS[key],result['primary_zgz'][key],result['resources']['actual'][key]['50']))
    notes=['主表沿冻结root_align(x)=x−x[..., :1,:]、单位mm：MPJPE减joint0腕坐标，MPVPE独立减vertex0坐标，不把后者称为减关节腕坐标。local/global来自zgz两个完整录制，RA总列按输出帧数加权。',
        '首行是已发表背景值，协议和成本口径不同；其余五行为本轮公共初始化/W50/raw单seed结果。',
        'Latency为ch_local固定3.2s实际前缀的sensor-to-mesh服务P50，包含末端MANO FK和778顶点/21关节CPU传输；不含物理等待与序列化。',
        'Params为实际wrapper名义总参数，含disabled支路；active/结构非零参数另见JSON。FLOPs未实测，填-。',
        '以下历史S39只作上下文，其初始化/过滤/步长与缩放成本协议不同，不参与配对非劣判定。']
    history=[HEADER,RULE,historical_row(result['historical_S39']['artifact'])]
    table='\n'.join(lines)+'\n\n'+'\n\n'.join(notes)+'\n\n'+'\n'.join(history)+'\n'
    (out/'main_table.md').write_text(table)
    resources=['| 服务臂/间隔 | P50/P95/P99 ms | 超时次数/查询数 | 超时比例 | outputs/s | events/s | CUDA allocated/reserved MiB |',
        '|---|---|---|---|---|---|---|']
    for arm,by_hop in result['resources']['actual'].items():
        for hop,r in by_hop.items():
            q=r['latency_sensor_to_mesh']
            resources.append(f"| {arm}/{hop}ms | {q['p50_ms']:.2f}/{q['p95_ms']:.2f}/{q['p99_ms']:.2f} | {r['deadline_exceeded_queries']}/{r['queries']} | {r['deadline_exceedance_fraction']:.1%} | {r['outputs_per_second']:.2f} | {r['events_per_second']:.0f} | {r['cuda_peak_allocated_bytes']/2**20:.1f}/{r['cuda_peak_reserved_bytes']/2**20:.1f} |")
    text=table+f"\n执行状态：完整。配对RA-MPJPE非劣判定：`{result['scientific_outcome']}`；机制、服务能力和新统计主张分别看证据。\n\n"
    text+='\n'.join(resources)+'\n\n'
    text+='超时按每次实际sensor-to-mesh服务耗时严格大于该查询间隔计算；actual与10k/50k/150k synthetic负载分列。维护父计时与子计时不可重复相加。\n\n'
    text+='显存为实际测量进程，包含评估wrapper、仍驻留的冻结canonical预热模型和workspace；Params列仅计评估wrapper，二者含义不同。\n\n'
    text+='完整stream回放计时截止51D姿态，末端FK仅用于离线评分；主表采用postprocess新增完整mesh服务测量，两种口径均保存在results.json。\n\n'
    text+='最终ycy是训练录制诊断：full使用全部9位原训练受试者，包含小阶段SELECT2。zgz是一位留出受试者的两条录制，不能把录制、窗口或重复查询当独立受试者。\n\n'
    text+='空间无向Z仅平均反向空间边，保留正滞后、极性槽与τ/γ信息；不是删除所有时间信息。共享50ms提交网格上的额外25ms纯查询须预测逐位一致，独立audit见results.json。\n\n'
    text+='四种cadence除保留各自全网格均值，还按(run_id,end_ms)四路交集报告相同时刻的已保存误差；共享GT逐位核对。该补充不按GT/误差筛样、不重算FK、不改门，反馈日程仍有区别。\n\n'
    for path,diagnostic in result['diagnostics'].items():
        doc=diagnostic['artifact'];status=doc.get('status',doc.get('verdict'))
        text+=f"诊断 `{Path(path).name}`：`{status}`。{doc.get('primary_finding',doc.get('scope',''))}\n\n"
    for arm in ('s43','full'):
        q=result['primary_zgz'][arm]
        text+=(f"{LABELS[arm]}绝对误差（local/global，mm）：MPJPE {q['local']['mpjpe_abs_mm']:.2f}/{q['global']['mpjpe_abs_mm']:.2f}，"
            f"MPVPE {q['local']['mpvpe_abs_mm']:.2f}/{q['global']['mpvpe_abs_mm']:.2f}。\n\n")
    text+='\n\n'.join(result['limitations'])+'\n\n'
    text+='原始绝对MPJPE/MPVPE、全部25/50/100ms及固定50ms反馈结果、实际source计数、峰值内存和逐阶段维护账见results.json。\n'
    (out/'report.md').write_text(text)
    dump(out/'resources.json',result['resources']);dump(out/'results.json',result)
    dump(out/'completion.json',dict(execution_complete=True,final_report_generated=True,scientific_outcome=result['scientific_outcome']))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--out-dir',type=Path)
    parser.add_argument('--mode',choices=('final','partial','dry-run'),default='final')
    parser.add_argument('--history-s39',type=Path,default=DEFAULT_HISTORY)
    args=parser.parse_args();out=args.out_dir or args.root/'report'
    status=inventory(args.root,args.history_s39)
    if args.mode!='final':
        status['mode']=args.mode;dump(out/'incomplete.json',status)
        print(json.dumps(dict(mode=args.mode,complete_artifact_set=status['complete_artifact_set'],missing=len(status['missing']),metrics_aggregated=False)))
        return 0
    if status['missing']:
        status['mode']='final_refused';dump(out/'incomplete.json',status)
        print(json.dumps(dict(status='INCOMPLETE',missing=len(status['missing']),report_generated=False)))
        return 2
    try:result=collect_final(args.root,args.history_s39)
    except (IncompleteReport,InvalidArtifact) as error:
        dump(out/'invalid.json',dict(execution_complete=False,final_report_generated=False,error=str(error)))
        print(json.dumps(dict(status='INVALID_OR_INCOMPLETE',error=str(error),report_generated=False)));return 2
    write_final(result,out)
    print(json.dumps(dict(status='COMPLETE',scientific_outcome=result['scientific_outcome'],out_dir=str(out))))
    return 0


if __name__=='__main__':sys.exit(main())
