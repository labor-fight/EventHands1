#!/usr/bin/env python3
"""Independent XDL raw completeness, finite-data and descriptive result audit.

No inference/training, checkpoint selection or new significance/repeat claims.
Main rows are only consumed from the existing producer; this tool writes no table.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import datetime
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'model'))
from tools.xdl import diagnostics as D
from tools.xdl import geometry as G
from tools.xdl.prepare import sha,write_json
OLD=('xa_c0_s3407','xa_c1_s3407','xa_c0_s3408','xa_c1_s3408','xa_c1r_s3407')
ARMS=('f0','q0','f1','q1','n1')
ERROR_FIELDS=('wrist_mm','mpjpe_abs_mm','mpjpe_ra_mm','mpvpe_abs_mm','mpvpe_ra_mm','root_deg','finger_deg')
SCOPE='single-seed exploratory; no repeat reliability gate or inferential CI/p'
CONTRASTS={'query_teacher':('f0','q0'),'query_self':('f1','q1'),'self_fixed':('f0','f1'),
           'self_query':('q0','q1'),'event_content':('n1','q1'),'self_vs_null':('n1','f1'),'joint_utility':('f0','q1')}


def describe_required(values):
    values=list(values);defined=[v for v in values if v is not None]
    valid=all(np.isfinite(v) for v in defined)
    result={'n':len(values),'n_defined':len(defined),'n_undefined':len(values)-len(defined),'finite':valid,'mean':None,'median':None}
    if valid and defined:result.update(mean=float(np.mean(defined)),median=float(np.median(defined)))
    return result


def invalid_numbers(value,path='$'):
    if isinstance(value,np.ndarray):
        if np.issubdtype(value.dtype,np.number):return [f'{path}{tuple(map(int,idx))}' for idx in np.argwhere(~np.isfinite(value))]
        value=value.tolist()
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float):return [path] if not math.isfinite(value) else []
    if isinstance(value,dict):return [p for k,v in value.items() for p in invalid_numbers(v,f'{path}.{k}')]
    if isinstance(value,(list,tuple)):return [p for i,v in enumerate(value) for p in invalid_numbers(v,f'{path}[{i}]')]
    return []


def finite_records(rows,fields):
    invalid=[];flags=[]
    for i,row in enumerate(rows):
        for field in fields:
            if field not in row or row[field] is None or (isinstance(row[field],dict) and not row[field]):invalid.append(f'rows[{i}].{field}:missing_required_field');continue
            invalid+=invalid_numbers(row[field],f'rows[{i}].{field}')
            if field in ('endpoint','errors'):
                invalid += [f'rows[{i}].{field}.{k}:null_required_observation' for k,v in row[field].items() if v is None]
        pred=row.get('prediction',row.get('predictions'))
        if pred is not None and bool(row.get('finite',False))!=bool(np.isfinite(pred).all()):flags.append(i)
    return {'passed':not invalid and not flags,'n_records':len(rows),'invalid_paths':invalid,'incorrect_finite_flags':flags}


def key_audit(expected,observed):
    counts=Counter(observed);wanted=set(expected);got=set(counts)
    return {'passed':len(expected)==len(wanted) and wanted==got and all(v==1 for v in counts.values()),
            'expected':len(expected),'observed':len(observed),'missing':[list(k) for k in sorted(wanted-got)],
            'extra':[list(k) for k in sorted(got-wanted)],'duplicated':[{'key':list(k),'count':v} for k,v in counts.items() if v!=1]}


def audit_bank_keys(bank,rows):
    expected=[(seq,p['packet_id'],s['id']) for seq,meta in bank['sequences'].items() for p in meta['bank'] for s in p['specs']]
    return key_audit(expected,[(r['seq'],r['packet_id'],r['spec']['id']) for r in rows])


def audit_trajectory(row,times):
    pred=np.asarray(row.get('predictions'));wanted=np.asarray(times,np.int64)
    checks={'shape':pred.shape==(len(wanted),51),'clock':np.array_equal(row.get('times_ms'),wanted),
            'finite':bool(np.isfinite(pred).all()),'finite_flag':bool(row.get('finite',False))==bool(np.isfinite(pred).all()),
            'error_fields':set(row.get('errors',{}))==set(ERROR_FIELDS),'error_lengths':all(v is not None and len(v)==len(wanted) for v in row.get('errors',{}).values()),
            'initial_first_state':pred.shape==(len(wanted),51) and 'initial' in row and np.array_equal(row['initial'],pred[0],equal_nan=True),
            'error_numeric':not invalid_numbers(row.get('errors',{}))}
    return {'passed':all(checks.values()),'checks':checks,'invalid_paths':invalid_numbers(pred,'predictions')+invalid_numbers(row.get('errors',{}),'errors')}


def verify_file(path,expected):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(path)
    actual=sha(path)
    if actual!=expected:raise ValueError(f'file identity changed: {path}: expected{expected}, got{actual}')
    return {'path':str(path),'sha256':actual,'bytes':path.stat().st_size}


def audit_labels(bank,rows):
    bad=[]
    for i,row in enumerate(rows):
        seq=row.get('seq');meta=bank['sequences'].get(seq)
        if meta is None or row.get('cohort')!=meta['cohort'] or row.get('subject',seq.split('_')[0])!=seq.split('_')[0]:bad.append(i)
    return {'passed':not bad,'mismatched_row_indices':bad,'counts_by_cohort':dict(Counter(r.get('cohort','missing') for r in rows)),
            'sequences_by_cohort':{c:len({r['seq'] for r in rows if r.get('cohort')==c}) for c in ('seen','zgz')}}


@lru_cache(maxsize=20)
def checkpoint_step(path,expected_sha):
    verify_file(path,expected_sha)
    saved=torch.load(path,map_location='cpu')
    return int(saved.get('global_step',-1))


@lru_cache(maxsize=4)
def common_reference(out):
    return D.load_reference(out)[0]


@lru_cache(maxsize=8)
def mano_mean(path):
    return np.asarray(np.load(path,allow_pickle=False)['hands_mean'],np.float32)


def expected_initials(bank,run,out,cfg,section):
    reference=common_reference(str(out));mean=mano_mean(str(Path(cfg['MANO']['NPZ']).resolve()));expected={};specs={};truth_cache={}
    for seq,meta in bank['sequences'].items():
        if section!='bank' and ((section=='motion' and meta['cohort']!='zgz') or (section=='recovery' and not run.startswith('xdl_') and meta['cohort']!='zgz')):continue
        data=D.load_data(cfg,seq,meta)
        if section=='bank':items=[(p['packet_id'],p['previous_end_ms'],p['specs'],p) for p in meta['bank']]
        elif section=='recovery':
            items=[]
            for c in meta['clips']:
                first=meta['rows'][c['end_indices'][0]];items.append((c['clip_id'],first['previous_end_ms'],G.recovery_specs(c['clip_id']),first))
        else:
            items=[]
            for rid,(a,b) in enumerate(meta['valid_runs_ms']):
                rows=[r for r in meta['rows'] if r['run_id']==rid]
                if rows:items.append((rid,a,G.recovery_specs(f'{seq}:run{rid}',small_only=True),rows[0]))
        for identifier,time,ss,first in items:
            clean=data['gt'][time].copy();natural=D.reference_lookup(reference,seq,first)
            for spec in ss:
                key=(seq,identifier,spec['id']);expected[key]=G.perturb_state(clean,spec,mean,natural=natural);specs[key]=spec
            if section=='motion' and len([r for r in meta['rows'] if r['run_id']==identifier])<20:
                key=(seq,identifier,'natural');expected[key]=natural;specs[key]={'block':'natural','id':'natural'}
    return expected,specs


def initial_pair_audit(bank,run,out,cfg,section,data):
    expected,specs=expected_initials(bank,run,out,cfg,section);bad=[];fingerprints={}
    def key(row):return (row['seq'],row['packet_id'] if section=='bank' else (row['clip_id'] if section=='recovery' else row['run_id']),row['spec']['id'])
    for collection in (['rows','controls','empty'] if section=='bank' else ['rows','short_natural']):
        for i,row in enumerate(data.get(collection,[])):
            if collection=='empty':kk=(row['seq'],row['packet_id'],bank['sequences'][row['seq']]['bank'][0]['specs'][0]['id'])
            else:kk=key(row)
            observed=np.asarray(row.get('initial'));value=expected.get(kk)
            match=value is not None and observed.shape==(51,) and np.array_equal(observed,value,equal_nan=True)
            payload=True if collection=='empty' else row['spec']==specs.get(kk)
            if not match or not payload:bad.append({'collection':collection,'index':i,'key':list(kk),'initial_matches':match,'spec_payload_matches':payload})
            if collection=='rows' and observed.shape==(51,):fingerprints['|'.join(map(str,kk))]=hashlib.sha256(observed.astype(np.float32).tobytes()).hexdigest()
    joined=json.dumps(fingerprints,sort_keys=True).encode()
    return {'passed':not bad,'mismatches':bad,'actual_initial_map_sha256':hashlib.sha256(joined).hexdigest(),'n_common_initials':len(fingerprints),
            'verified_against':'GT incoming labels and immutable common C0 reference, exact physical spec including mean once; no candidate endpoint selection'}


def expected_trajectories(bank,section,legacy):
    expected={};queries=0
    for seq,meta in bank['sequences'].items():
        if section=='recovery':
            if legacy and meta['cohort']!='zgz':continue
            for clip in meta['clips']:
                rows=[meta['rows'][i] for i in clip['end_indices']];times=[rows[0]['previous_end_ms']]+[r['end_ms'] for r in rows]
                for spec in G.recovery_specs(clip['clip_id']):expected[(seq,clip['clip_id'],spec['id'])]=times;queries+=len(rows)
        else:
            if meta['cohort']!='zgz':continue
            for rid,(a,b) in enumerate(meta['valid_runs_ms']):
                rows=[r for r in meta['rows'] if r['run_id']==rid]
                if not rows:continue
                times=[a]+[r['end_ms'] for r in rows]
                for spec in G.recovery_specs(f'{seq}:run{rid}',small_only=True):expected[(seq,rid,spec['id'])]=times;queries+=len(rows)
    return expected,queries


def metric_profiles(section,data):
    """Retain blocks/scales and seen subjects, without treating packets as subjects."""
    groups=defaultdict(list)
    for row in data['rows']:
        cohort=row['cohort'];seq=row['seq'];spec=row['spec'];scale=spec.get('scale','none')
        labels=[(cohort,seq),(cohort,'all')]
        if cohort=='seen':labels.append((cohort,row['subject']))
        for c,label in labels:groups[(c,label,spec['block'],scale)].append(row)
    profiles={};flat={}
    for group,rows in groups.items():
        item={'n_records':len(rows),'n_sequences':len({r['seq'] for r in rows}),'n_subjects':len({r.get('subject',r['seq'].split('_')[0]) for r in rows}),
              'n_nonfinite':sum(not r['finite'] for r in rows),'statistics':{}}
        raw=defaultdict(list)
        for r in rows:
            if section=='bank':
                for k,v in r['endpoint'].items():raw['endpoint.'+k].append(v)
                for block in ('translation','root','finger'):
                    pair=r['paired'][block];raw[f'rho.{block}'].append(pair['rho'])
                    if block=='translation':
                        raw['direction.translation.gain'].append(pair['gain']);raw['direction.translation.cosine'].append(pair['cosine'])
                    else:
                        for side in ('left','right'):
                            for metric in ('gain','cosine'):raw[f'direction.{block}.{side}.{metric}'].append(pair[side][metric])
            else:
                for k,v in r['auc'].items():raw['auc.'+k].append(v if r['finite'] else float('nan'))
                for k,v in r['rho_auc'].items():raw['rho_auc.'+k].append(v if r['finite'] else float('nan'))
                for block in ('absolute','root_relative','root','finger'):
                    for metric in ('gain','gain_bias','cosine','lag_ms'):
                        raw[f'motion.{block}.{metric}'].append(r['motion'][block][metric] if r['finite'] else float('nan'))
        for metric,values in raw.items():
            stats=describe_required(values);item['statistics'][metric]=stats
            flat['|'.join((section,)+group+(metric,))]=stats['mean']
            if metric.startswith('rho.'):
                defined=[v for v in values if v is not None]
                stats['fraction_lt1']=float(np.mean(np.asarray(defined)<1)) if defined and stats['finite'] else None
                stats['physical_policy_met']=stats['median'] is not None and stats['median']<=.90 and stats['fraction_lt1']>=.70
        if section!='bank':
            item.update(initially_satisfied=sum(r['recovery']['initially_satisfied'] for r in rows),sustained_recovered=sum(r['recovery']['hit_step'] is not None for r in rows),
                        right_censored=sum(r['recovery']['right_censored'] for r in rows))
            item['half_recovery']={b:{'defined':sum(bool(r['half_recovery'][b]['defined']) for r in rows),
                'hits':sum(r['half_recovery'][b].get('hit_step') is not None for r in rows),'censored':sum(bool(r['half_recovery'][b].get('right_censored')) for r in rows)} for b in ('translation','root','finger')}
        profiles['|'.join(group)]=item
    # Seen subject means preserve nine-subject descriptive aggregation, not CI.
    subject_stats=defaultdict(list)
    for group,item in profiles.items():
        c,label,block,scale=group.split('|')
        if c=='seen' and label in ('ch','lfz','lpc','lr','ly','lyh','lyq','ycy','ylf'):
            for metric,stat in item['statistics'].items():subject_stats[(block,scale,metric)].append(stat['mean'])
    for (block,scale,metric),values in subject_stats.items():flat['|'.join((section,'seen','subject_equal',block,scale,metric))]=describe_required(values)['mean']
    return profiles,flat


def audit_section(bank,banksha,run,section,out):
    directory=Path(out)/'diagnostics'/run
    original=directory/f'{section}.json';preferred=directory/f'{section}.reanalysis.json'
    path=preferred if preferred.exists() else original
    if not path.exists():return {'status':'INCOMPLETE','reason':'missing section artifact','path':str(path)},{}
    try:
        result=json.loads(path.read_text());raw_info=verify_file(result['raw']['path'],result['raw']['sha256']);raw=D.load_raw(raw_info['path']);data=raw['data'];metadata=raw['metadata']
        checks={'bank_identity':result['bank_sha256']==banksha,'raw_bank_identity':metadata['bank_sha256']==banksha,
                'raw_checkpoint_identity':metadata['identity']==result['identity'],'exact_step6000':result['identity']['global_step']==6000,
                'formal_serial':result.get('prediction_batch_size')==1 and metadata.get('prediction_batch_size')==1,
                'complete_protocol':not result.get('partial_smoke',True) and not data.get('partial_smoke',True),
                'section_identity':result['section']==section and metadata['section']==section}
        files=[verify_file(result['identity']['checkpoint'],result['identity']['checkpoint_sha256']),verify_file(result['identity']['config'],result['identity']['config_sha256'])]
        refs=[verify_file(r['path'],r['sha256']) for r in result['references']]
        if path==preferred:
            verify_file(result['original_file']['path'],result['original_file']['sha256'])
            checks['reanalysis_unchanged']=all(result['reanalysis']['verification'].values())
        checks['checkpoint_payload_step']=checkpoint_step(result['identity']['checkpoint'],result['identity']['checkpoint_sha256'])==6000
        checks['checkpoint_run_filename']=Path(result['identity']['checkpoint']).name==f'{run}-step=6000.ckpt'
        labels=audit_labels(bank,data['rows']);checks['cohort_subject_labels']=labels['passed']
        cfg=D.load_config(result['identity']['config']);initial_audit=initial_pair_audit(bank,run,out,cfg,section,data);checks['actual_common_initials_and_specs']=initial_audit['passed']
        legacy=not run.startswith('xdl_')
        if section=='bank':
            expected_queries=sum(len(p['specs']) for m in bank['sequences'].values() for p in m['bank'])
            keys=audit_bank_keys(bank,data['rows']);numeric=finite_records(data['rows'],('prediction','endpoint','paired'))
            cp=[(s,p['packet_id'],sp['id'],mode) for s,m in bank['sequences'].items() if m['cohort']=='zgz' for p in m['bank'] for sp in p['control_specs'] for mode in ('swap','mirror')]
            controls=key_audit(cp,[(r['seq'],r['packet_id'],r['spec']['id'],r['mode']) for r in data['controls']])
            ee=[(s,p['packet_id']) for s,m in bank['sequences'].items() if m['cohort']=='zgz' for p in m['bank']]
            empty=key_audit(ee,[(r['seq'],r['packet_id']) for r in data['empty']]);empty_hold=all(r['exact_hold'] for r in data['empty'])
            checks.update(probe_ids_complete=keys['passed'],control_ids_complete=controls['passed'],empty_ids_complete=empty['passed'],empty_hold=empty_hold,mandatory_numeric=numeric['passed'])
            expected_queries+=len(cp)+len(ee)
            # Exact E/clock and independently reconstructed common physical input states.
            rows_by_packet={(s,p['packet_id']):p for s,m in bank['sequences'].items() for p in m['bank']}
            input_bad=[]
            for r in data['rows']:
                p=rows_by_packet.get((r['seq'],r['packet_id']))
                if p is None:continue
                required_clock=('end_ms','previous_end_ms','state_interval_ms','query_start_offset_ms')
                if r['event_sha256']!=p['events_sha256'] or set(r.get('clock',{}))!=set(required_clock) or any(r.get('clock',{}).get(k)!=p[k] for k in required_clock):input_bad.append(r['packet_id'])
            checks['E_and_clock']=not input_bad
            checks['endpoint_field_sets']=all(set(r.get('endpoint',{}))==set(ERROR_FIELDS) and np.asarray(r.get('prediction')).shape==(51,) for r in data['rows'])
            summary=D.summarize_bank(data)
            ancillary={'controls':controls,'empty':empty,'control_numeric':finite_records(data['controls'],('prediction','endpoint')),'input_mismatches':input_bad}
        else:
            expected,expected_queries=expected_trajectories(bank,section,legacy)
            if section=='recovery':
                clocks={(s,c['clip_id']):{k:[m['rows'][i][k] for i in c['end_indices']] for k in ('state_interval_ms','query_start_offset_ms')} for s,m in bank['sequences'].items() for c in m['clips']}
                checks['query_clock_arrays']=all(r.get('clock')==clocks.get((r['seq'],r['clip_id'])) for r in data['rows'])
            if section=='recovery':observed=[(r['seq'],r['clip_id'],r['spec']['id']) for r in data['rows']]
            else:observed=[(r['seq'],r['run_id'],r['spec']['id']) for r in data['rows']]
            keys=key_audit(list(expected),observed);trajectory_checks=[]
            for key,row in zip(observed,data['rows']):trajectory_checks.append({'key':list(key),**audit_trajectory(row,expected.get(key,[]))})
            numeric=finite_records(data['rows'],('predictions','errors'))
            checks.update(trajectory_ids_complete=keys['passed'],all_states_clock_and_numeric=all(r['passed'] for r in trajectory_checks),mandatory_numeric=numeric['passed'])
            ancillary={'trajectory_failures':[r for r in trajectory_checks if not r['passed']]}
            if section=='motion':
                short_expected={(s,r['run_id']):[m['rows'][r['end_indices'][0]]['previous_end_ms']]+[m['rows'][i]['end_ms'] for i in r['end_indices']]
                    for s,m in bank['sequences'].items() if m['cohort']=='zgz' for r in m['short_runs']}
                short_keys=key_audit(list(short_expected),[(r['seq'],r['run_id']) for r in data['short_natural']])
                sa=[audit_trajectory(r,short_expected.get((r['seq'],r['run_id']),[])) for r in data['short_natural']]
                checks.update(all_short_natural=short_keys['passed'],short_states_clock_and_numeric=all(r['passed'] for r in sa))
                expected_queries+=sum(len(t)-1 for t in short_expected.values());ancillary['short_natural']={'keys':short_keys,'trajectories':sa,'rows':len(data['short_natural']),
                    'auc':{k:describe_required(r['auc'][k] if r['finite'] else float('nan') for r in data['short_natural']) for k in ('wrist_mm','mpjpe_abs_mm','mpjpe_ra_mm','root_deg','finger_deg')}}
            summary=D.summarize_trajectories(data)
        checks['summary_recomputed']=G.sanitize(summary)==result['summary']
        if checks['mandatory_numeric'] and all(r.get('finite',True) for r in data.get('short_natural',[])):
            checks['query_denominator']=data['queries_attempted']==expected_queries
        else:checks['query_denominator']=data['queries_attempted']<=expected_queries
        profiles,metrics=metric_profiles(section,data)
        return {'status':'COMPLETE' if all(checks.values()) else 'FAIL','checks':checks,'artifact':{'path':str(path),'sha256':sha(path)},'raw':raw_info,'identity':result['identity'],
                'references':refs,'identity_files':files,'actual_initials':initial_audit,'cohort_counts':labels,'keys':keys,'numeric':numeric,'ancillary':ancillary,'expected_queries':expected_queries,'observed_queries':data['queries_attempted'],
                'summary_recomputed':summary,'profiles':profiles},metrics
    except Exception as e:return {'status':'FAIL','reason':type(e).__name__,'message':str(e),'artifact':{'path':str(path),'sha256':sha(path)}},{}


def normalized_config(cfg,paired=True):
    result=copy.deepcopy(cfg);result.pop('_config_path',None)
    if paired:
        for key in ('XDL_QUERY','XDL_SELF','XDL_OBSERVATION_NULL'):result.get('MODEL',{}).pop(key,None)
    for key in ('OUTPUT_DIR','RUN_NAME'):result.get('TRAIN',{}).pop(key,None)
    result.get('EVAL',{}).pop('OUTPUT_DIR',None)
    return result


def config_mode_checks(run,cfg):
    arm=run.split('_')[1];query,self_state,null={'f0':('fixed',False,False),'q0':('mano',False,False),'f1':('fixed',True,False),'q1':('mano',True,False),'n1':('mano',True,True)}[arm]
    model=cfg['MODEL'];train=cfg['TRAIN'];data=cfg['DATA']
    return {'seed3407':cfg['SEED']==3407,'study':model.get('XDL_STUDY') is True,'query':model.get('XDL_QUERY')==query,
        'self':model.get('XDL_SELF') is self_state,'null':model.get('XDL_OBSERVATION_NULL') is null,'C0_geometry':model.get('XYZ_GEOM') is False,
        'burn_full':model.get('XDL_BURN_MODE')=='full','W50':data['WINDOW_MIN']==data['WINDOW_MAX']==50,'speed_disabled':cfg['AUG']['SPEED_AUG'] is False,
        '6000':train['MAX_STEPS']==6000,'EBS1024':train['DEVICES']==2 and train['BATCH_SIZE_PER_GPU']==512 and train['ACCUMULATE_GRAD_BATCHES']==1}


def audit_training(bank,run,identity,out):
    directory=Path(identity['checkpoint']).parent;meta_path=directory/'training_metadata.json';finish_path=directory/'train_finish.json'
    if not meta_path.exists() or not finish_path.exists():return {'status':'INCOMPLETE','reason':'training completion metadata not yet available'}
    try:
        meta=json.loads(meta_path.read_text());finish=json.loads(finish_path.read_text());cfg=D.load_config(identity['config'])
        freeze_path=Path(out)/'frozen_manifest.json';freeze=json.loads(freeze_path.read_text());freeze_hash=sha(freeze_path)
        checks=config_mode_checks(run,cfg);checks.update(run_name=meta['run_name']==run,formal=meta['preflight_not_formal'] is False and finish['preflight_not_formal'] is False,
            fresh=meta['start_step']==0 and meta['resumed_from'] is None,finished=finish['status']=='completed_exact_step6000' and finish['global_step']==6000,
            ckpt=finish['checkpoint_sha256']==identity['checkpoint_sha256'],resolved_config=finish['config_sha256']==identity['config_sha256'],
            source_config=sha(meta['config_path'])==meta['config_sha256'],frozen=meta['freeze_manifest_sha256']==freeze_hash,
            split=meta['split_sha256']==bank['split']['sha256'],train_sequences=meta['train_sequences']==[s for s,m in bank['sequences'].items() if m['cohort']=='seen'],
            EBS=meta['effective_batch']==1024 and meta['devices']==2 and meta['batch_size_per_gpu']==512 and meta['accumulate_grad_batches']==1,
            coverage=meta['coverage']['window_ms']==50 and meta['coverage']['clip_k']==4 and meta['coverage']['sequences_with_legal_clips']==72,
            main_samples=meta['main_samples']==6000*1024,paired_counter=meta['paired_main_stream'] is True,
            no_failure_artifacts=not any(directory.glob('nonfinite*.json')))
        entries=freeze['source_sha256'];checks['recorded_source_freeze']=all(entries.get(p)==h for p,h in meta.get('source_sha256',{}).items()) and bool(meta.get('source_sha256'))
        original_cfg=D.load_config(meta['config_path']);checks['resolved_config_equivalence']=normalized_config(original_cfg,False)==normalized_config(cfg,False)
        checks['source_config_modes']=all(config_mode_checks(run,original_cfg).values())
        saved=torch.load(identity['checkpoint'],map_location='cpu');hp=saved.get('hyper_parameters',{});hp=hp.get('cfg',hp)
        checks['checkpoint_hparams']=normalized_config(hp,False)==normalized_config(cfg,False)
        checks['source_config_frozen']=freeze['source_sha256'].get(str(Path(meta['config_path']).resolve().relative_to(REPO.resolve())))==meta['config_sha256']
        exposures=[];streams=[];files=[{'path':str(meta_path),'sha256':sha(meta_path)},{'path':str(finish_path),'sha256':sha(finish_path)},{'path':str(freeze_path),'sha256':freeze_hash}]
        for rank in (0,1):
            ep=directory/f'actual_exposure_rank{rank}.json';sp=directory/f'pair_stream_rank{rank}.json'
            exposure=json.loads(ep.read_text());stream=json.loads(sp.read_text());exposures.append(exposure);streams.append(stream)
            checks[f'rank{rank}_exposure']=exposure['main_supervised_rows']==6000*512 and exposure['final_global_step']==6000 and exposure['preflight_not_formal'] is False
            if run.split('_')[1] in ('f0','q0'):checks[f'rank{rank}_teacher_no_burn']=exposure['burn_model_rows']==0 and exposure['self_main_rows']==0
            checks[f'rank{rank}_streams']=len(stream)==3 and [r['optimizer_step'] for r in stream]==[0,1,2] and all(r['rank']==rank and r['main_rows']==512 for r in stream)
            files+=[{'path':str(ep),'sha256':sha(ep)},{'path':str(sp),'sha256':sha(sp)}]
        stream_projection=[[{k:r[k] for k in ('optimizer_step','rank','global_rows','sample_index_sha256','main_sha256','prefix_sha256','main_rows','main_events')} for r in stream] for stream in streams]
        return {'status':'COMPLETE' if all(checks.values()) else 'FAIL','checks':checks,'files':files,'exposures':exposures,
            'normalized_config_sha256':hashlib.sha256(json.dumps(normalized_config(cfg),sort_keys=True).encode()).hexdigest(),
            'paired_first3_each_rank_sha256':hashlib.sha256(json.dumps(stream_projection,sort_keys=True).encode()).hexdigest(),
            'stream_scope':'first three actual main/prefix packets on each rank; remaining data flow is fixed source/counter protocol, not a claimed6000-step full-event hash measurement'}
    except Exception as e:return {'status':'FAIL','reason':type(e).__name__,'message':str(e)}


def audit_main_row(run,identity):
    path=REPO/f'outputs/semkine/{run}_main_row.json'
    if not path.exists():return {'status':'UNMEASURED','path':str(path)},{}
    try:
        row=json.loads(path.read_text());seed=run.rsplit('s',1)[-1];value=row['per_seed'][seed]
        checks={'checkpoint':value.get('checkpoint_sha256',row.get('provenance',{}).get('checkpoint_sha256'))==identity['checkpoint_sha256'],
                'exact_step':value['step']==6000,'frames':value['n_frames']==2590,'finite':not invalid_numbers(value),
                'no_finite_subset':row.get('finite_subset_reduction') is False}
        if run.startswith('xdl_'):
            protocol=row.get('protocol',{});checks.update(config_sha=row.get('provenance',{}).get('config_sha256')==identity['config_sha256'],seed_protocol=protocol.get('seeds')==[3407],single_seed=protocol.get('n_seeds')==1,W50=protocol.get('window_ms')==protocol.get('step_ms')==50,optimizer_step=protocol.get('exact_optimizer_step')==6000)
        metrics={f'main|zgz|{group}|RA-MPJPE':value[group]['mpjpe_ra_mm'] for group in ('overall','local','global')}
        return {'status':'COMPLETE' if all(checks.values()) else 'FAIL','checks':checks,'path':str(path),'sha256':sha(path)},metrics
    except Exception as e:return {'status':'FAIL','reason':str(e),'path':str(path)},{}


def audit_run(bank,banksha,run,out):
    sections={};metrics={}
    for section in ('bank','recovery','motion'):
        audit,values=audit_section(bank,banksha,run,section,out);sections[section]=audit;metrics.update(values)
    statuses=[v['status'] for v in sections.values()]
    state='FAIL' if 'FAIL' in statuses else ('COMPLETE' if all(s=='COMPLETE' for s in statuses) else 'INCOMPLETE')
    identity=next((v['identity'] for v in sections.values() if 'identity' in v),None)
    training=audit_training(bank,run,identity,out) if run.startswith('xdl_') and identity else {'status':'NOT_APPLICABLE' if not run.startswith('xdl_') else 'INCOMPLETE'}
    main,mainmetrics=audit_main_row(run,identity) if identity else ({'status':'UNMEASURED'}, {})
    if main['status']=='COMPLETE':metrics.update(mainmetrics)
    if run.startswith('xdl_') and state=='COMPLETE' and any(v['status']!='COMPLETE' for v in (main,training)):state='FAIL' if any(v['status']=='FAIL' for v in (main,training)) else 'INCOMPLETE'
    return {'run':run,'status':state,'identity':identity,'sections':sections,'training':training,'main_row':main,'metrics':metrics,'science_scope':SCOPE if run.startswith('xdl_') else 'given frozen legacy weights; descriptive capacity profile'}


def practical_margin(key):
    if 'rho' in key:return .05
    if key.endswith('_deg'):return 1.
    if key.endswith('_mm') or key.endswith('RA-MPJPE'):return 1.1
    return None


def build_contrasts(arms):
    result={}
    for label,(base,candidate) in CONTRASTS.items():
        item={'base':base,'candidate':candidate,'evidence_scope':SCOPE,'status':'UNAVAILABLE','values':{}}
        if all(a in arms and arms[a]['status']=='COMPLETE' for a in (base,candidate)):
            pairing={}
            if all('sections' in arms[a] and 'training' in arms[a] for a in (base,candidate)):
                pairing={'actual_initials':all(arms[base]['sections'][s]['actual_initials']['actual_initial_map_sha256']==arms[candidate]['sections'][s]['actual_initials']['actual_initial_map_sha256'] for s in ('bank','recovery','motion')),'same_common_configs':arms[base]['training']['normalized_config_sha256']==arms[candidate]['training']['normalized_config_sha256'],'same_first3_main_stream':arms[base]['training']['paired_first3_each_rank_sha256']==arms[candidate]['training']['paired_first3_each_rank_sha256']}
            item['pairing_checks']=pairing
            if pairing and not all(pairing.values()):item['reason']='paired input/training identity mismatch';result[label]=item;continue
            item['status']='DESCRIPTIVE'
            for key in sorted(set(arms[base]['metrics'])|set(arms[candidate]['metrics'])):
                b=arms[base]['metrics'].get(key);c=arms[candidate]['metrics'].get(key);finite=b is not None and c is not None and math.isfinite(b) and math.isfinite(c)
                direction=1 if key.endswith('cosine') or key.endswith('.gain') else -1
                difference=(c-b)*direction if finite else None;margin=practical_margin(key)
                item['values'][key]={'base':b,'candidate':c,'improvement':difference,'practical_margin':margin,
                    'exceeds_margin':None if difference is None or margin is None else difference>margin,
                    'metric_direction':'candidate-base descriptive only' if key.endswith('.gain') else ('higher_is_better' if direction==1 else 'lower_is_better')}
        result[label]=item
    interaction={'base':'2x2','candidate':'q1','evidence_scope':SCOPE,'status':'UNAVAILABLE','values':{}}
    if all(a in arms and arms[a]['status']=='COMPLETE' for a in ('f0','q0','f1','q1')):
        paired_labels=('query_teacher','query_self','self_fixed','self_query')
        if any(result[k]['status']!='DESCRIPTIVE' for k in paired_labels):interaction['reason']='four-cell pairing mismatch';result['interaction']=interaction;return result
        interaction['status']='DESCRIPTIVE'
        keys=set.intersection(*(set(arms[a]['metrics']) for a in ('f0','q0','f1','q1')))
        for key in sorted(keys):
            v={a:arms[a]['metrics'][key] for a in ('f0','q0','f1','q1')};good=all(x is not None and math.isfinite(x) for x in v.values())
            interaction['values'][key]={'cells':v,'interaction':v['q0']+v['f1']-v['f0']-v['q1'] if good else None,
                'scope':'positive superadditivity only for error/rho measures; no synergy gate without utility/events/recovery/motion evidence'}
    result['interaction']=interaction;return result


def narrative(payload):
    old=payload['legacy_runs'];new=payload['new_runs'];lines=[f"XDL 原始结果审计：旧权重完成 {sum(v['status']=='COMPLETE' for v in old.values())}/5；新五臂完成 {sum(v['status']=='COMPLETE' for v in new.values())}/5。"]
    lines.append('审计核对固定事件/初态IDs、21状态与真实时间、空包、所有短段、原始文件/权重/配置SHA及数值失败；原始非有限值保留，不对有限子集补平均。')
    for name,run in old.items():
        lines.append(f"{name}：{run['status']}。")
        if run['status']!='COMPLETE':continue
        bank=run['sections']['bank']['profiles'];recovery=run['sections']['recovery']['profiles']
        for seq in ('zgz_global','zgz_local'):
            bits=[]
            for block in ('translation','root','finger'):
                item=bank.get(f'zgz|{seq}|{block}|large',{}).get('statistics',{}).get(f'rho.{block}',{})
                if item:bits.append(f"{block} large ρ中位 {item['median']:.3f}" if item['median'] is not None else f'{block} large ρ未定义/失败')
            natural=recovery.get(f'zgz|{seq}|natural|none',{}).get('statistics',{}).get('auc.mpjpe_ra_mm',{})
            if natural and natural['mean'] is not None:bits.append(f"natural真实时间RA-AUC {natural['mean']:.2f} mm")
            lines.append(f"  {seq}："+'；'.join(bits)+'。')
    lines.append('ρ<1只表示同E下前态响应收缩，不能证明GT终点准确或动态恢复；natural局部AUC差仅是表现描述，不能单独证明根因。')
    lines.append('旧模型表现可定位错误前态敏感性和闭环边界，不能据此识别新银行、查询中心或训练状态来源的因果收益。')
    if not all(v['status']=='COMPLETE' for v in new.values()):lines.append('新五臂尚未全部通过完整性审计；不作Q/F、自身前态或事件必要性结论。')
    else:lines.append('新五臂固定比较全部保留；seed3407仅为单次训练的探索性信号，不声明显著、跨seed稳定或原δ/τ复现门通过。ρ、实际终点、事件内容、恢复和运动护栏须联判。')
    lines.append('批量FK/关联舍入放大的工程数值预检单列，停止的bulk产物不进入本科学结果。主结果九列表另由make_row/report_table生成；本叙述用于预注册ledger诊断区。')
    return '\n'.join(lines)+'\n'


def audit_job(args):
    bank,banksha,run,out=args
    return audit_run(bank,banksha,run,out)


def finalize(out=REPO/'outputs/xdl',workers=5,legacy_only=False):
    out=Path(out);bankpath=out/'packet_bank.json';bank=json.loads(bankpath.read_text());banksha=sha(bankpath)
    runs=list(OLD)+([] if legacy_only else [f'xdl_{a}_s3407' for a in ARMS])
    with ProcessPoolExecutor(max_workers=workers) as pool:audits=dict(zip(runs,pool.map(audit_job,[(bank,banksha,run,out) for run in runs])))
    old={r:audits[r] for r in OLD};new={a:audits.get(f'xdl_{a}_s3407',{'run':f'xdl_{a}_s3407','status':'INCOMPLETE','reason':'not evaluated in legacy-only pass'}) for a in ARMS}
    payload={'study':G.STUDY,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'implementer':'/root/xd_diag_impl','bank':{'path':str(bankpath),'sha256':banksha,'counts':bank['counts']},
        'source':[{'path':str(REPO/p),'sha256':sha(REPO/p)} for p in ('tools/xdl/finalize.py','tools/xdl/diagnostics.py','tools/xdl/geometry.py','tools/xdl/prepare.py')],
        'legacy_runs':old,'new_runs':new,'contrasts':build_contrasts(new),'single_seed_scope':SCOPE,'engineering_bulk_diagnostics_excluded':str(out/'runtime_numeric_batching_20261013'),
        'new_stage_complete':all(v['status']=='COMPLETE' for v in new.values()),'legacy_stage_complete':all(v['status']=='COMPLETE' for v in old.values()),
        'inference_statistics':'none; no CI/p, no repeat noise estimate, no invented missing δ=0','main_table':'not generated here; tools/xdl/make_row.py + tools/report_table.py',
        'denominator_policy':'all frozen queries/initials/runs; never finite subset; seen subject means descriptive with nine subjects, zgz is one subject/two sequences'}
    invalid=[];clean=G.sanitize(payload,invalid=invalid);clean['invalid_numeric_paths']=invalid
    write_json(out/'results_audit.json',clean)
    (out/'diagnostic_analysis.txt').write_text(narrative(payload))
    print(json.dumps({'results_audit':str(out/'results_audit.json'),'legacy_complete':payload['legacy_stage_complete'],'new_complete':payload['new_stage_complete'],
        'status_counts':dict(Counter(v['status'] for v in audits.values()))},ensure_ascii=False),flush=True)
    return payload

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(REPO/'outputs/xdl'));p.add_argument('--workers',type=int,default=5);p.add_argument('--legacy-only',action='store_true')
    args=p.parse_args();finalize(args.out,args.workers,args.legacy_only)
