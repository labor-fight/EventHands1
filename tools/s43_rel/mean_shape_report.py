"""Strict artifact-only public-beta0 deployment report, separate from main R1.

No conditional-beta score is read, rescaled, or used to estimate these results.
Accuracy comes exclusively from10 complete public-beta0 replays; resources
come exclusively from5 public-beta0 sensor-to-surface service measurements.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np

from tools.report_table import HEADER,RULE
from . import report as R

ARMS=('canonical',)+R.ARMS
RECORDINGS=('zgz_local','zgz_global')
SCHEMA='s43_relation_mean_shape_full_recording_replay_v1'
RESOURCE_SCHEMA='s43_relation_mean_shape_benchmark_v1'
WRIST_V='mpvpe_joint_wrist_aligned_mm'


def inventory(root,extension):
    required=[extension,root.parent/'postprocess_complete.json',root.parent/'finalization_complete.json']
    for arm in ARMS:
        for recording in RECORDINGS:
            prefix=root/f'replays/{arm}_{recording}_h50'
            required += [prefix.with_suffix('.json'),prefix.with_suffix('.npz'),prefix.with_suffix('.predictions.npz')]
        required.append(root/f'resources/{arm}_h50.json')
    missing=[str(path) for path in required if not path.exists()]
    return dict(schema='s43_relation_mean_shape_report_inventory_v1',complete_artifact_set=not missing,
        required_replays=10,required_resource_measurements=5,missing=missing,
        accuracy_aggregated=False,conditional_scores_read=False,final_report_generated=False)


def validate_replay(doc,arm,recording):
    R.require(doc.get('schema')==SCHEMA,'mean-shape replay schema mismatch')
    R.require(doc.get('arm')==arm and doc.get('recording')==recording,'mean-shape replay identity mismatch')
    R.require(doc.get('shape_condition')=='public_mean_beta0','conditional shape score cannot enter public-beta0 report')
    R.require(np.array_equal(np.asarray(doc.get('inference_beta')),np.zeros(10)),'inference beta is not exactly public zeros10')
    for key in ('aux_betas_accessed_during_inference','pose_metadata_opened_during_inference'):
        R.require(doc.get(key) is False,'forbidden inference input flag '+key)
    R.require(doc.get('GT_only_after_predictions_saved') is True,'GT was not gated after persisted prediction')
    R.require(doc.get('scoring_performed') is True,'engineering/no-score trace is not a full deployment evaluation')
    access=doc.get('inference_access_guard',{})
    R.require(access.get('aux_betas_accessed') is False and access.get('pose_metadata_opened') is False and
        access.get('blocked_aux_beta_attempts')==0 and access.get('blocked_pose_meta_attempts')==0,'inference access firewall receipt invalid')
    R.require(doc.get('predicted_FK_beta')=='zeros10' and doc.get('GT_FK_beta')=='score_only_actual_aux','predicted/GT scoring shape contract invalid')
    R.require(doc.get('hop_ms')==50 and doc.get('commit_schedule_ms')==50 and doc.get('observation_window_ms')==50 and doc.get('output_lag_ms')==1,'mean-shape W50/commit/lag schedule mismatch')
    R.require(doc.get('initialization_GT') is False and doc.get('full_recording') is True,'mean-shape replay is not complete public-initialized stream')
    R.require(doc.get('all_valid_events_consumed_once') is True and doc.get('source_count')==doc.get('expected_source_count'),'mean-shape source count guard failed')
    R.require(doc.get('source_id_digest') and doc.get('source_id_digest')==doc.get('expected_source_id_digest'),'mean-shape source digest guard failed')
    R.require(R.source_guard(doc.get('source_files_unchanged')),'mean-shape replay executable/source guard failed')
    R.require(isinstance(doc.get('outputs'),int) and doc['outputs']>0,'mean-shape replay has no outputs')
    R.require(all(R.finite(doc.get('metrics',{}).get(k)) and doc['metrics'][k]>=0 for k in R.METRICS),'mean-shape replay metrics invalid')
    return doc


def posthoc_motion_summary(root,arms=ARMS,mask_root=None):
    """Reuse ALL frozen target-only motion masks, matched by exact addresses.

    Only saved beta0 per-frame errors enter the averages. No conditional-beta
    model score, new FK, threshold, GT query input or favorable-mask selection.
    """
    mask_root=Path(mask_root or root.parent/'robustness/full_final')
    paths={recording:mask_root/f'{recording}_nominal_motion_strata.npz' for recording in RECORDINGS}
    missing=[str(path) for path in paths.values() if not path.exists()]
    if missing:return dict(status='MISSING_MASK_ARTIFACTS',missing=missing,GT_used_for_query=False,conditional_model_scores_read=False)
    result={};failed=False
    for recording,path in paths.items():
        with np.load(path,allow_pickle=False) as data:
            mask_end=data['end_ms'].copy();mask_run=data['run_id'].copy()
            masks={key:data[key].copy() for key in data.files if data[key].dtype.kind=='b' and data[key].shape==(len(mask_end),)}
        R.require(bool(masks),'motion-mask artifact contains no saved boolean masks')
        addresses=list(zip(mask_run.tolist(),mask_end.tolist()));mask_set=set(addresses)
        R.require(len(set(addresses))==len(addresses),'duplicate saved motion-mask address')
        result[recording]=dict(mask_source=str(path),mask_source_sha256=R.sha(path),
            mask_outputs=len(addresses),mask_names=list(masks),arms={})
        for arm in arms:
            beta0=root/f'replays/{arm}_{recording}_h50.npz'
            with np.load(beta0,allow_pickle=False) as data:
                end=data['end_ms'].copy();run=data['run_id'].copy()
                errors={key:data[key].copy() for key in (*R.METRICS,WRIST_V) if key in data.files}
            own=list(zip(run.tolist(),end.tolist()));index={address:i for i,address in enumerate(own)}
            exact=len(index)==len(own) and set(index)==mask_set
            row=dict(address_set_exact=exact,mean_shape_outputs=len(own),
                matched_outputs=sum(address in index for address in addresses),
                unmatched_mask_outputs=sum(address not in index for address in addresses),
                unmatched_mean_shape_outputs=sum(address not in mask_set for address in own),
                mean_shape_source=str(beta0),mean_shape_sha256=R.sha(beta0),masks={})
            if not exact:
                row['status']='FAIL';failed=True
            else:
                R.require(all(v.shape==(len(own),) and np.isfinite(v).all() for v in errors.values()),'beta0 per-frame motion-strata error invalid')
                aligned=np.asarray([index[address] for address in addresses],np.int64)
                for name,mask in masks.items():
                    selected=aligned[mask];n=len(selected)
                    row['masks'][name]=dict(n=n,evidence_status='available' if n else 'no samples; evidence missing',
                        metrics={key:float(np.mean(values[selected],dtype=np.float64)) for key,values in errors.items()} if n else None)
                row['status']='PASS'
            result[recording]['arms'][arm]=row
    return dict(schema='s43_rel_mean_shape_posthoc_motion_summary_v1',status='FAIL' if failed else 'PASS',recordings=result,
        mask_policy='All stored boolean masks retained; definitions/thresholds unchanged from main posthoc GT-only artifact.',
        mask_builder_source_sha256=R.sha(R.ROOT/'tools/s43_rel/robustness.py'),
        GT_used_for_query=False,GT_used_only_for_posthoc_masks=True,conditional_model_scores_read=False,
        new_FK_or_threshold=False,no_samples_policy='Report missing evidence for every empty mask, including local_finger_primary; never substitute a favorable stratum.')


def collect(root,extension):
    state=inventory(root,extension)
    if state['missing']:raise R.IncompleteReport('public-beta0 deployment extension has missing artifacts')
    ext=R.read(extension)
    R.require(ext.get('schema')=='s43_rel_mean_shape_deployment_extension_v1','extension identity invalid')
    R.require(R.read(root.parent/'postprocess_complete.json').get('complete') is True,'conditional main postprocess must finish before extension formal resources')
    R.require(R.read(root.parent/'finalization_complete.json').get('execution_complete') is True,'main finalization must finish before public-mean extension summary')
    replays={};resources={};primary={};reference_grid={};source={}
    for arm in ARMS:
        replays[arm]={}
        for recording in RECORDINGS:
            prefix=root/f'replays/{arm}_{recording}_h50'
            doc=validate_replay(R.read(prefix.with_suffix('.json')),arm,recording)
            expected_arm='s43' if arm=='canonical' else arm
            R.require(doc.get('model_arm')==expected_arm,'public-mean replay model role mismatch')
            if arm=='canonical':R.require(doc.get('checkpoint') is None,'canonical public-mean replay was finetuned')
            else:
                checkpoint=root.parent/f'train/full_r1/{arm}/last.ckpt'
                R.require(doc.get('checkpoint') and Path(doc['checkpoint']).resolve()==checkpoint.resolve(),'mean-shape accuracy must use fixed final resumable checkpoint')
                complete=R.read(checkpoint.with_name('complete.json'))
                R.require(complete.get('completed_steps')==6000 and complete.get('actual_outputs')==192000,'mean-shape accuracy model not fully trained')
                R.require(doc.get('source_files',{}).get(str(checkpoint.resolve()),{}).get('sha256')==complete.get('checkpoint_sha256'),'mean-shape accuracy checkpoint identity mismatch')
            prediction_path=prefix.with_suffix('.predictions.npz')
            R.require(doc.get('prediction_archive_sha256')==R.sha(prediction_path),'pre-scoring prediction archive identity mismatch')
            scoring=doc.get('scoring_receipt',{})
            R.require(scoring.get('all_predictions_saved_before_GT') is True and scoring.get('predicted_beta_zero') is True,'score phase did not verify prediction persistence/public shape')
            R.require(scoring.get('prediction_archive_sha256')==R.sha(prediction_path),'score phase used another prediction archive')
            R.require(scoring.get('predicted_FK_beta')=='zeros10' and scoring.get('GT_FK_beta')=='score_only_actual_aux','score phase shape contract differs')
            with np.load(prefix.with_suffix('.npz'),allow_pickle=False) as data:
                end=data['end_ms'].copy();run=data['run_id'].copy();pred=data['pred'].copy();gt=data['gt'].copy()
                R.require(pred.shape==gt.shape==(doc['outputs'],51),'mean-shape prediction/GT arrays invalid')
                R.require(end.shape==run.shape==(doc['outputs'],),'mean-shape target-address arrays invalid')
                per={k:data[k].copy() for k in R.METRICS}
            with np.load(prefix.with_suffix('.predictions.npz'),allow_pickle=False) as saved:
                R.require('gt' not in saved.files,'pre-scoring prediction artifact contains GT')
                R.require(np.array_equal(saved['inference_beta'],np.zeros(10)),'pre-scoring artifact inference shape invalid')
                R.require(np.array_equal(saved['pred'],pred),'pre-scoring and scored prediction states differ')
                if 'end_ms' in saved.files:R.require(np.array_equal(saved['end_ms'],end),'pre-scoring output addresses differ')
            for key,values in per.items():
                R.require(values.shape==(doc['outputs'],) and np.isfinite(values).all(),'per-frame mean-shape score invalid')
                R.require(math.isclose(float(values.mean()),doc['metrics'][key],rel_tol=2e-6,abs_tol=1e-4),'JSON/per-frame mean-shape metric mismatch')
            grid=(end,run,gt)
            if recording in reference_grid:
                R.require(all(np.array_equal(a,b) for a,b in zip(reference_grid[recording],grid)),'public-beta0 arms have unmatched GT/scoring addresses')
            else:reference_grid[recording]=grid
            identity=(doc['source_count'],doc['source_id_digest'])
            if recording in source:R.require(source[recording]==identity,'public-beta0 source traversal differs across arms')
            else:source[recording]=identity
            replays[arm][recording]=doc
        raw=R.read(root/f'resources/{arm}_h50.json')
        R.require(raw.get('schema')==RESOURCE_SCHEMA and raw.get('arm')==arm,'public-beta0 resource schema/arm invalid')
        R.require(raw.get('shape_condition')=='public_mean_beta0' and np.array_equal(np.asarray(raw.get('inference_beta')),np.zeros(10)),'public-beta0 resource shape contract invalid')
        R.require(raw.get('aux_betas_accessed') is False and raw.get('pose_metadata_opened') is False,'public-beta0 resource read forbidden inputs')
        R.require(raw.get('tsub_opened') is False and raw.get('module_globals_restored') is True,'mean-shape benchmark wrapper guard/restore incomplete')
        R.require(raw.get('extension_sha256')==R.sha(extension),'mean-shape resource extension identity mismatch')
        if arm=='canonical':R.require(raw.get('checkpoint_sha256') is None,'canonical beta0 resource used finetuned weights')
        else:R.require(raw.get('checkpoint_sha256')==R.sha(root.parent/f'train/full_r1/{arm}/step=6000.ckpt'),'beta0 resource did not use final deployment weights')
        R.require(raw.get('queries')==64 and raw.get('query_hop_ms')==50 and raw.get('recording_geometry_context')=='ch_local','public-beta0 resource workload is not fixed ch3.2s/hop50')
        R.require(raw.get('engineering_only') is False,'CPU engineering smoke is not a formal resource measurement')
        adapted=dict(raw,schema=raw.get('base_schema'))
        resources[arm]=dict(R.benchmark_summary(adapted),source_path=str(root/f'resources/{arm}_h50.json'))
        execution=root/f'executions/resource_{arm}.json'
        resources[arm]['NVML_device_memory_observation']=R.read(execution).get('NVML_device_memory_observation') if execution.exists() else None
        resources[arm]['NVML_scope']='Read-only device-wide20Hz observer: baseline, sampled peak and sampled increment; includes device reserved/driver/context memory, not an exact per-process peak. Torch allocator peaks remain separate.'
        primary[arm]=R.aggregate_primary(list(replays[arm].values()))
        for domain,rows in (('local',[replays[arm]['zgz_local']]),('global',[replays[arm]['zgz_global']]),('overall',list(replays[arm].values()))):
            if all(R.finite(q['metrics'].get(WRIST_V)) for q in rows):
                primary[arm][domain][WRIST_V]=sum(q['outputs']*q['metrics'][WRIST_V] for q in rows)/sum(q['outputs'] for q in rows)
    R.require(len({(r['source_events_including_warmup'],r['source_id_digest']) for r in resources.values()})==1,'mean-shape resource prefixes differ')
    motion=posthoc_motion_summary(root)
    R.dump(root/'posthoc_motion_alignment.json',motion)
    R.require(motion['status']!='FAIL','mean-shape posthoc motion-mask address mismatch; preserved FAIL artifact')
    return dict(schema='s43_relation_mean_shape_results_v1',execution_complete=True,extension_sha256=R.sha(extension),
        aggregator_sha256=R.sha(__file__),shape_condition='public_mean_beta0',inference_beta=[0.]*10,
        conditional_scores_read=False,models_retrained=False,new_risk_thresholds=False,
        scoring='Predicted FK/mesh uses public zero beta; target FK/mesh uses true aux shape only after predictions saved.',
        alignment='Frozen original metrics: joint0 for RA-MPJPE, independently vertex0 for RA-MPVPE; absolute errors and optional joint-wrist-alignedMPVPE retained.',
        primary=primary,resources=resources,replays=replays,posthoc_motion_strata=motion,
        source_accounting={rec:dict(rows=n,source_id_digest=h) for rec,(n,h) in source.items()},
        limits=['Same fixed trained models; public mean-shape inference is an input-condition diagnostic, not shape-free training or real-sensor validation.',
            'One heldout subject zgz, two recordings, one seed; no newly registered statistical improvement gate.',
            'No conditional-beta score was used to estimate the mean-shape accuracy or resource values.'])


def write(result,root):
    if (root/'results.json').exists():raise FileExistsError('preserve scored public-beta0 report')
    lines=[HEADER,RULE]
    for arm in ARMS:
        label=R.LABELS[arm]+' β=0'
        lines.append(R.headline_row(label,result['primary'][arm],result['resources'][arm]))
    notes=['公共平均形状β=0部署输入验证，与给定真实来源β的条件诊断分开。没有重训或按新结果选择参数。',
        '表中MPJPE按joint0对齐、MPVPE按vertex0对齐；绝对误差与关节腕点对齐MPVPE补充见下文/JSON。',
        '预测mesh用β=0；GT mesh仅在预测保存后用真实GT来源β解码。没有把同一个β错用于两边。',
        'Latency为固定ch_local3.2s前缀的实际sensor-to-mesh P50，含末端MANO和CPU顶点传输，FLOPs未测。']
    text='\n'.join(lines)+'\n\n'+'\n\n'.join(notes)+'\n\n'
    for arm in ARMS:
        q=result['primary'][arm]
        text+=(f"{R.LABELS[arm]}绝对误差local/global（mm）：MPJPE {q['local']['mpjpe_abs_mm']:.2f}/{q['global']['mpjpe_abs_mm']:.2f}，"
            f"MPVPE {q['local']['mpvpe_abs_mm']:.2f}/{q['global']['mpvpe_abs_mm']:.2f}。\n\n")
        if WRIST_V in q['overall']:text+=f"关节腕点对齐MPVPE总体：{q['overall'][WRIST_V]:.2f} mm。\n\n"
        r=result['resources'][arm];nvml=r.get('NVML_device_memory_observation')
        text+=f"Torch显存allocated/reserved峰值：{r['cuda_peak_allocated_bytes']/2**20:.1f}/{r['cuda_peak_reserved_bytes']/2**20:.1f} MiB。\n\n"
        if nvml and all(isinstance(nvml.get(k),(int,float)) for k in ('baseline_device_used_bytes','sampled_peak_device_used_bytes','sampled_peak_minus_baseline_bytes')):
            text+=(f"20Hz设备级NVML基线/采样峰值/增量：{nvml['baseline_device_used_bytes']/2**20:.1f}/"
                f"{nvml['sampled_peak_device_used_bytes']/2**20:.1f}/{nvml['sampled_peak_minus_baseline_bytes']/2**20:.1f} MiB；"
                "包含reserved/driver/context，是设备级采样观测，不是精确process peak。\n\n")
    strata=result.get('posthoc_motion_strata',{})
    for recording,source in strata.get('recordings',{}).items():
        for arm in ('s43','full'):
            primary=source['arms'][arm].get('masks',{}).get('local_finger_primary')
            if primary is None:continue
            if not primary['n']:text+=f"{recording}/{R.LABELS[arm]}局部手指主分层：0样本，缺证据。\n\n"
            else:text+=f"{recording}/{R.LABELS[arm]}局部手指主分层：{primary['n']}个固定GT事后标记样本，RA-MPJPE {primary['metrics']['mpjpe_ra_mm']:.2f} mm；全部预定mask结果与来源hash保存在JSON。\n\n"
    root.mkdir(parents=True,exist_ok=True)
    (root/'main_table.md').write_text('\n'.join(lines)+'\n')
    (root/'report.md').write_text(text)
    R.dump(root/'results.json',result)
    R.dump(root/'completion.json',dict(execution_complete=True,resource_measurements=5,full_replays=10,conditional_scores_read=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=R.DEFAULT/'mean_shape')
    p.add_argument('--extension',type=Path,default=R.DEFAULT/'mean_shape_deployment_extension.json')
    p.add_argument('--partial',action='store_true');args=p.parse_args()
    state=inventory(args.root,args.extension)
    if args.partial or state['missing']:
        R.dump(args.root/'incomplete.json',state)
        print({'status':'INCOMPLETE','missing':len(state['missing']),'accuracy_aggregated':False})
        return 0 if args.partial else 2
    try:result=collect(args.root,args.extension)
    except (R.IncompleteReport,R.InvalidArtifact) as error:
        R.dump(args.root/'invalid.json',dict(execution_complete=False,error=str(error)))
        print({'status':'INVALID','error':str(error)});return 2
    write(result,args.root);print({'status':'COMPLETE','root':str(args.root)});return 0


if __name__=='__main__':sys.exit(main())
