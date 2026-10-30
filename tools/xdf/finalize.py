#!/usr/bin/env python3
"""Independent bookkeeping and artifact integrity checks, no GPU calls."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.xdf.angle_probe import write,sha
OUT=ROOT/'outputs/xdf'
def read(p):return json.loads((OUT/p).read_text())


def main():
    prior=read('preregistration_v1.json');assert sha(OUT/'preregistration_v1.md')==prior['snapshot_sha256']
    preserved=[];updated_docs=[]
    for name,digest in prior['source_sha256'].items():
        if name in ['docs/S37_XYZ_ASSOC_DIAGNOSIS.md','docs/EXPERIMENT_RECORDS.md','docs/网络结构分析.md']:updated_docs.append(name);continue
        assert sha(ROOT/name)==digest,name;preserved.append(name)
    current=(ROOT/'docs/S37_XYZ_ASSOC_DIAGNOSIS.md').read_text()
    archived=(ROOT/'outputs/s37_doc_merge_20261014/S37_XYZ_ASSOC_DIAGNOSIS.md').read_text()
    anchor='<a id="inventory-detail"></a>'
    assert current[current.index(anchor):]==archived[archived.index(anchor):]
    for p in (ROOT/'tools/xdf').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    interface=read('interface/summary.json');angles=read('angles/extraction.json');pipeline=read('pipeline/result.json')
    assert all(v==0 for v in interface['numeric_max_abs'].values())
    assert angles['max_delta_replay_diff']==0 and angles['max_wrist_preservation_error_mm']<=.01
    assert read('interface/rollout/summary.json')['audit']['q1_e1_max_state_difference']==0
    assert len(pipeline['records'])==24 and len(pipeline['failures'])==6
    # Measure repeat agreement; never silently assume bitwise identity.
    repeat_audit=[]
    for name in ['zgz_local','zgz_global']:
        for mode in ['Q1','F_s1701','L0_s1701','L1_s1701']:
            arrays=[np.load(OUT/'pipeline'/f'{mode}_r{i}_{name}.npz')['prediction51'] for i in range(3)]
            assert all(a.shape==arrays[0].shape for a in arrays)
            diffs=[float(np.max(np.abs(a.astype(float)-arrays[0].astype(float)))) for a in arrays[1:]]
            failure=next((r['failure'] for r in pipeline['records'] if r['mode']==mode and r['sequence']==name and r['repeat']==0),None)
            if failure is not None:assert all(d==0 for d in diffs)
            repeat_audit.append({'sequence':name,'mode':mode,'finite_prefix_n':len(arrays[0]),'max_state_diff':max(diffs),'failure':failure,
                'last_finite_translation_m':arrays[0][-1,:3].astype(float).tolist(),
                'last_finite_max_abs_state':float(np.abs(arrays[0][-1].astype(float)).max())})
    exact=all(r['max_state_diff']==0 for r in repeat_audit)
    write(OUT/'pipeline/independent_audit.json',{'repeat_predictions':repeat_audit,'source_sha256':sha(__file__),'all_repeat_predictions_bitwise_equal':exact,
          'scope':'No additional inference; failed L0/L1 finite prefixes exact across3 passes; Q1local repeat3 differs by max1.12e-6, explicitly retained. No tolerance substituted for a scientific gate.'})
    costs={
        'interface_and_rollout':{'measured_seconds':read('interface/budget_audit.json')['measured_GPU_wall_seconds'],'method':'main-entry through serialization; excludes interpreter/NumPy/tool launch before main'},
        'angles':{'measured_seconds':sum(read('angles/'+p)['wall_seconds'] for p in ['extract.wall.json','fit_1701.wall.json','fit_1702.wall.json']),'method':'/usr/bin/time full Python process'},
        'depth':{'measured_seconds':read('depth/execution_receipt.json')['gpu_worker_inclusive_seconds'],'method':'/proc process start through final output'},
        'objective':{'measured_seconds':read('objective/final_audit.json')['budget']['allocated_gpu_hours']*3600,'method':'worker receipts; interpreter/import/CUDA setup unmeasured; separate reserve below'},
        'pipeline':{'measured_seconds':read('pipeline/run.wall.json')['wall_seconds'],'method':'/usr/bin/time full Python process, including failures and warmups'},
    }
    total=sum(v['measured_seconds'] for v in costs.values())/3600
    by_study={'XDF-1':sum(costs[k]['measured_seconds'] for k in ['interface_and_rollout','angles','pipeline'])/3600,
              'XDF-2':costs['depth']['measured_seconds']/3600,'XDF-3':costs['objective']['measured_seconds']/3600}
    assert by_study['XDF-1']<=2 and by_study['XDF-2']<=1 and by_study['XDF-3']+.05<=2
    evidence=['interface/summary.json','interface/rollout/summary.json','interface/state_tz_partial_by_domain.json','interface/objective_independent_review.json',
        'interface/supplemental_guardrails.json','angles/protocol.json','angles/summary.json','angles/aa_seam_audit.json','angles/gradient_strata_protocol.json','angles/gradient_strata_result.json',
        'depth/native_population.json','depth/native_overlap_details.json','depth/support.json','depth/flow_result.json','depth/cpu_final_audit.json',
        'objective/protocol.json','objective/result.json','objective/final_audit.json','objective/supplemental_guardrails.json','pipeline/result.json','pipeline/independent_audit.json']
    calls={'interface_capture_and_feedback':30720,'angle_capture':11520,'actual_augmentation_flow':10000,
           'objective_own_burn_and_cache':16856,'objective_clean_cache':2590,'objective_short_feedback':13440,'pipeline_including_warmups_and_failed_calls':pipeline['calls_including_warmup']}
    summary={'study':'XDF-20261014','status':'completed_fast_diagnostics_with_scientific_failures_and_preregistered_support_stop',
       'preregistration_sha256':prior['snapshot_sha256'],'preexecution_manifest_preserved':True,'physical_gpus_used':[2,3,4,5,6,7],
       'full_encoder_retraining':False,'production_adoption':False,'diagnostic_heads':{'translation':10,'angle':16,'objective_including_repeats':6,'total':32},
       'complete_model_forward_calls':calls,'total_complete_model_forward_calls':sum(calls.values()),
       'forward_count_scope':'model/data-flow calls including own burn, failed calls, and profile warmups; excludes head-only optimizer/evaluation calls',
       'measured_gpu_hours':total,'measured_costs':costs,'measured_by_study_gpu_hours':by_study,
       'separate_unmeasured_setup_reserve_gpu_hours':.05,'measured_plus_reserve_gpu_hours':total+.05,
       'cost_limits':{'XDF-1':2,'XDF-2':1,'XDF-3':2,'total':5},
       'cost_caveats':['0.05 is an allowance, not a measurement or exact total','shared interface/angle cache charged once','all shared pipeline cost conservatively charged to XDF-1','CPU-only statistics/audits not GPU time'],
       'outcomes':{'legacy_content_gate':'PASS','angle_content_gate_root':'FAIL','angle_content_gate_finger':'FAIL',
           'native_depth_matched_fit':'NOT_RUN_SUPPORT_GATE_FAIL','z_weight_bias_gate':'FAIL','local_own_new_recovery':'0/24 for all new deployable heads',
           'long_local_L0_L1_seed1701':'nonfinite at query803/839 on all3 repeats','long_F_seed1701':'finite but severely divergent; local/global full-run wrist means4918.46/12787.44mm',
           'overall_software_cost_guardrail':'NOT_ESTABLISHED; local Q1 unstable, Fglobal unstable, L0/L1local incomplete'},
       'unresolved':['candidate-source support and learned ranking','H3-to-LBS-to-slot positive localization','causal depth coverage intervention','root/finger reliable clean and own update','stationary jitter:0/24 eligible clips','end-to-end physical sensor latency','two full-model seeds and matched repeat'],
       'evidence_sha256':{p:sha(OUT/p) for p in evidence},'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tools/xdf').glob('*.py'))},
       'historical_source_files_unchanged':preserved,'updated_documentation':updated_docs,'canonical_report':'docs/S37_XYZ_ASSOC_DIAGNOSIS.md#xdf-results'}
    write(OUT/'execution_summary.json',summary)
    missing=[];links=0
    for name in updated_docs:
        doc=ROOT/name
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
            target=target.strip('<>')
            if target.startswith(('http:','https:','mailto:')):continue
            path=unquote(target.split('#')[0]);file=doc if not path else (doc.parent/path).resolve()
            if not file.exists():missing.append({'doc':name,'target':target})
            links+=1
    historical_missing=[];new_missing=[]
    for item in missing:
        before=ROOT/'outputs/s37_doc_merge_20261014/references_before'/Path(item['doc']).name
        if before.exists() and item['target'] in before.read_text():historical_missing.append(item)
        else:new_missing.append(item)
    write(OUT/'documentation_link_audit.json',{'checked':links,'historical_missing':historical_missing,'new_missing':new_missing,
        'historical_reference':'outputs/s37_doc_merge_20261014/references_before; no invented or deleted historical artifacts'})
    assert not new_missing,new_missing
    write(OUT/'execution_verification.json',{'status':'PASS','prereg_snapshot_intact':True,'historical_sources_intact_n':len(preserved),
       'historical_51_entry_appendix_unchanged':True,'local_doc_links_checked':links,'new_missing_links':new_missing,'historical_missing_links_n':len(historical_missing),'new_python_AST_valid':True,
       'pipeline_all_repeat_predictions_exact':exact,'pipeline_max_repeat_state_difference':max(r['max_state_diff'] for r in repeat_audit),
       'expected_scientific_failures_retained':6,'execution_summary_sha256':sha(OUT/'execution_summary.json')})
    print(json.dumps({'measured_gpu_hours':total,'plus_unmeasured_reserve':total+.05,'calls':sum(calls.values()),'local_links':links,'historical_sources':len(preserved)},ensure_ascii=False))


if __name__=='__main__':main()
