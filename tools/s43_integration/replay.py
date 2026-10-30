"""Complete-recording CPU replay through the shared S43 representation lifecycle.

The default public initialization uses two independent frozen-canonical S43
warmup windows per valid run, then the requested integrated model. Every native
source event in every valid run is consumed, including warmup and short tails.
Only complete post-warmup 50ms windows are scored. No --maxsteps shortcut exists.

This is the explicitly documented public-initialization protocol, not a new
canonical 6000-step main row. --init standard is the old GT-assisted initial
condition and is always identified separately.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from tools.s43_br.data import DATA_ROOT, PUBLIC_INITIAL_STATE, SequenceReader
from .model import (DEFAULT_CFG,DEFAULT_CKPT,ROOT,S43IntegratedModel,
                    load_integrated,load_legacy)
from .tracker import S43StreamTracker


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def _json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,(np.integer,np.floating)):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def _write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    temporary=Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=_json)+'\n')
    os.replace(temporary,path)


def _evaluation_functions():
    """Reuse the unchanged evaluator definitions without its model-name import.

    eval_track.py imports ``from model import MNISTModel`` at module scope,
    conflicting with callers that already imported a model namespace package.
    Compile its exact three function definitions, retaining their original
    bodies, into a private namespace instead of mutating global module aliases.
    Their source-file SHA is recorded. No metric implementation is rewritten.
    """
    pose_name='_s43_integration_eval_pose_repr'
    if pose_name not in sys.modules:
        spec=importlib.util.spec_from_file_location(pose_name,ROOT/'model/pose_repr.py')
        module=importlib.util.module_from_spec(spec);sys.modules[pose_name]=module
        spec.loader.exec_module(module)
    file=ROOT/'semkine/eval_track.py'
    names={'mano_fk','root_align','sample_init_noise'}
    nodes=[node for node in ast.parse(file.read_text()).body
           if isinstance(node,ast.FunctionDef) and node.name in names]
    if {n.name for n in nodes}!=names:raise RuntimeError('frozen evaluator metric contract changed')
    scope={'np':np,'torch':torch,'decode_to_mano_inputs':sys.modules[pose_name].decode_to_mano_inputs,
           '__name__':'_s43_integration_frozen_evaluator_functions'}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),scope)
    return {name:scope[name] for name in names}


def _source_snapshot(config,checkpoint,reader,public,base_checkpoint=DEFAULT_CKPT):
    paths={ROOT/'tools/s43_integration/replay.py',ROOT/'tools/s43_integration/model.py',
           ROOT/'tools/s43_integration/tracker.py',ROOT/'tools/s43_integration/memory.py',
           ROOT/'tools/s43_integration/fast_geometry.py',ROOT/'tools/s43_integration/raster.cpp',
           ROOT/'tools/s43_br/carrier.py',ROOT/'tools/s43_br/data.py',
           ROOT/'model/model.py',ROOT/'model/mano_layer.py',ROOT/'model/pose_repr.py',
           ROOT/'semkine/eval_track.py',ROOT/'assets/mano_right.npz',Path(config),
           reader.root/'splits_semkine.json',Path(base_checkpoint)}
    if checkpoint:paths.add(Path(checkpoint))
    if public:paths.update((DEFAULT_CKPT,DEFAULT_CFG))
    for suffix in ('_events.npy','_offsets.npy','_tsub.npy','_aux.npz','.meta'):
        paths.add(Path(str(reader.base)+suffix))
    return {str(p.resolve()):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(paths)}


def _expected_digest(reader,runs):
    digest=hashlib.sha256();count=0
    for a,b in runs:
        lo,hi=int(reader.offsets[a]),int(reader.offsets[b]);count+=hi-lo
        for index in range(lo,hi,8192):
            digest.update(np.arange(index,min(index+8192,hi),dtype=np.int64).tobytes())
    return count,digest.hexdigest()


def _bank(tracker,sid):
    return tracker.bank if tracker.states[sid]['supported'] else tracker.unsupported_bank


def _consume_ms(tracker,sid,reader,start,end,account):
    """Native 1ms packets; the same ingest core used by training constructors."""
    wall_start=time.perf_counter()
    for ms in range(int(start),int(end)):
        packet=reader.window(ms,window_ms=1)
        events=packet['events'];events[:,0]+=ms*1000
        count=tracker.ingest(sid,events,packet['source_ids'],watermark_us=(ms+1)*1000)
        if count!=len(events):raise AssertionError('a fresh native replay packet was dropped or deduplicated')
        account['source_count']+=count
        account['source_digest'].update(packet['source_ids'].astype(np.int64,copy=False).tobytes())
        account['native_ms_packets']+=1
    return time.perf_counter()-wall_start


def _query(tracker,sid,cutoff,duplicate):
    hand=tracker.states[sid]
    before_pose=hand['pose'].copy();before_version=hand['version']
    before_digest=_bank(tracker,sid).digest(sid) if duplicate else None
    started=time.perf_counter();result=tracker.query(sid,int(cutoff));seconds=time.perf_counter()-started
    check=None
    if duplicate:
        first=result['state'].copy()
        started=time.perf_counter();result=tracker.query(sid,int(cutoff));seconds+=time.perf_counter()-started
        difference=float(np.max(np.abs(first-result['state'])))
        unchanged=(np.array_equal(before_pose,hand['pose']) and before_version==hand['version']
                   and before_digest==_bank(tracker,sid).digest(sid))
        if difference!=0. or not unchanged:raise AssertionError('duplicate query changed evidence/state/prediction')
        check={'cutoff_us':int(cutoff),'prediction_max_abs_diff':difference,
               'hand_pose_unchanged':True,'version_unchanged':True,'representation_digest_unchanged':True}
    return result,seconds,check


def _metrics(functions,mano,predictions,targets,betas):
    if not len(predictions):return {},{}
    with torch.inference_mode():
        b=torch.tensor(betas,dtype=torch.float32).reshape(1,10)
        pj,pv=functions['mano_fk'](mano,np.asarray(predictions,np.float32),b,torch.device('cpu'))
        gj,gv=functions['mano_fk'](mano,np.asarray(targets,np.float32),b,torch.device('cpu'))
    align=functions['root_align']
    per={
      'mpjpe_abs_mm':np.linalg.norm(pj-gj,axis=-1).mean(-1)*1000,
      'mpvpe_abs_mm':np.linalg.norm(pv-gv,axis=-1).mean(-1)*1000,
      'mpjpe_ra_mm':np.linalg.norm(align(pj)-align(gj),axis=-1).mean(-1)*1000,
      'mpvpe_ra_mm':np.linalg.norm(align(pv)-align(gv),axis=-1).mean(-1)*1000}
    if not all(np.isfinite(values).all() for values in per.values()):raise ArithmeticError('nonfinite replay metrics')
    return {key:float(value.mean()) for key,value in per.items()},per


def replay(config,checkpoint,seq,out,*,threads=1,init='public',duplicate_query=False,root=DATA_ROOT):
    os.environ['CUDA_VISIBLE_DEVICES']='';torch.set_num_threads(int(threads))
    prefix=Path(out);json_path=Path(str(prefix)+'.json');npz_path=Path(str(prefix)+'.npz')
    failed_path=Path(str(prefix)+'_failed.json')
    if any(p.exists() for p in (json_path,npz_path,failed_path)):raise FileExistsError(prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    wrapper,cfg=load_integrated(config,checkpoint)
    if int(cfg.get('EVAL',{}).get('WINDOW_MS',50))!=50:
        raise ValueError('this integration replay fixes W50; use the original entry for other protocols')
    reader=SequenceReader(seq,Path(root));runs=np.asarray(reader.runs,np.int64).reshape(-1,2)
    if np.any(runs[:,0]<0) or np.any(runs[:,1]<runs[:,0]) or np.any(runs[:,1]>=len(reader.offsets)):
        raise ValueError('invalid run metadata')
    if len(runs)>1 and np.any(runs[1:,0]<runs[:-1,1]):
        raise ValueError('valid runs overlap or are not chronological; refusing duplicate source replay')
    expected_count,expected_digest=_expected_digest(reader,runs)
    source_before=_source_snapshot(config,checkpoint,reader,init=='public',
                 cfg.get('INTEGRATION',{}).get('BASE_CHECKPOINT',DEFAULT_CKPT))
    functions=_evaluation_functions()
    canonical=None
    if init=='public':
        old,_=load_legacy(DEFAULT_CKPT,DEFAULT_CFG)
        canonical=S43IntegratedModel(old,mode='legacy').eval()
    tracker=S43StreamTracker(wrapper,max_sequences=1,unsupported_policy='hold')
    warm_tracker=S43StreamTracker(canonical,max_sequences=1) if canonical is not None else None
    account={'source_count':0,'source_digest':hashlib.sha256(),'native_ms_packets':0}
    predictions=[];ends=[];run_ids=[];native_counts=[];unsupported=[];statuses=[]
    duplicates=[];run_records=[];memories=[]
    costs={'ingest_wall_seconds':0.,'query_wall_seconds':0.,'commit_wall_seconds':0.,
           'warmup_wall_seconds':0.,'tail_wall_seconds':0.,'scoring_seconds':0.}
    epoch_costs={key:0. for key in ['reset_seconds','geometry_seconds','template_seconds',
                                  'local_ingest_seconds','phase_advance_seconds','readout_seconds']}
    rng=np.random.default_rng(0);started=time.perf_counter()
    try:
        for run_id,(a0,b0) in enumerate(runs):
            a,b=int(a0),int(b0);run_started=time.perf_counter();cursor=a
            before_count=account['source_count'];before_predictions=len(predictions)
            run={'run_id':run_id,'start_ms':a,'stop_ms':b,'warmup_windows':0,
                 'warmup_events':0,'tail_ms':0,'tail_events':0,
                 'expected_native_events':int(reader.offsets[b]-reader.offsets[a])}
            sid=seq
            if init=='public':
                history=PUBLIC_INITIAL_STATE.copy();history_time=a*1000-1000
                if a<b:
                    before_warm=time.perf_counter()
                    warm_tracker.reset(sid,a*1000,history,reader.betas,reader.K_cam,
                                       segment_id=run_id,state_time_us=history_time)
                    for warm_index in range(2):
                        if b-cursor<50:break
                        before_events=account['source_count']
                        _consume_ms(warm_tracker,sid,reader,cursor,cursor+50,account)
                        result,_,_=_query(warm_tracker,sid,(cursor+50)*1000,False)
                        memory=_bank(warm_tracker,sid).memory_bytes(sid)
                        memory['tracker_role']='canonical_warmup';memories.append(memory)
                        history=result['state'].copy();history_time=int(result['target_time_us'])
                        cursor+=50;run['warmup_windows']+=1
                        run['warmup_events']+=account['source_count']-before_events
                        warm_tracker.commit(sid,result,begin_next=warm_index==0 and b>cursor)
                    # A short run has only warmup and its unscored final prefix.
                    if run['warmup_windows']<2:
                        if cursor<b:
                            before_events=account['source_count']
                            _consume_ms(warm_tracker,sid,reader,cursor,b,account)
                            warm_tracker.representation(sid,b*1000)
                            memory=_bank(warm_tracker,sid).memory_bytes(sid)
                            memory['tracker_role']='canonical_warmup_tail';memories.append(memory)
                            run['tail_ms']=b-cursor;run['tail_events']=account['source_count']-before_events
                            cursor=b
                        warm_tracker.release(sid)
                        costs['warmup_wall_seconds']+=time.perf_counter()-before_warm
                        run['actual_native_events']=account['source_count']-before_count
                        run['scored_packets']=0;run['seconds']=time.perf_counter()-run_started
                        if run['actual_native_events']!=run['expected_native_events']:raise AssertionError('short-run source count mismatch')
                        run_records.append(run);continue
                    warm_tracker.release(sid)
                    costs['warmup_wall_seconds']+=time.perf_counter()-before_warm
            elif init=='standard':
                # The only GT entering an initialization is explicitly this
                # old evaluator protocol, never the deployable/public branch.
                if b-a>=50:
                    history=reader.target(a)+functions['sample_init_noise'](cfg,rng,1.)
                    history_time=a*1000
                else:
                    # Original ET skips short runs before drawing RNG noise.
                    # Still consume unscored native events without GT/noise.
                    history=PUBLIC_INITIAL_STATE.copy();history_time=a*1000-1000
            else:raise ValueError('init must be public or standard')
            if cursor<b:
                reset_t=time.perf_counter()
                tracker.reset(sid,cursor*1000,history,reader.betas,reader.K_cam,
                              segment_id=run_id,state_time_us=history_time)
                initial_reset_seconds=time.perf_counter()-reset_t
                run['initial_reset_wall_seconds']=initial_reset_seconds
                while b-cursor>=50:
                    begin_count=account['source_count']
                    costs['ingest_wall_seconds']+=_consume_ms(tracker,sid,reader,cursor,cursor+50,account)
                    count=account['source_count']-begin_count
                    packet_index=len(predictions)
                    duplicate=bool(duplicate_query and (packet_index==0 or (packet_index+1)%100==0))
                    result,query_seconds,check=_query(tracker,sid,(cursor+50)*1000,duplicate)
                    costs['query_wall_seconds']+=query_seconds
                    if check is not None:duplicates.append(check)
                    meta=result['representation_meta']
                    for key in epoch_costs:epoch_costs[key]+=float(meta.get(key,0.))
                    is_unsupported=meta.get('status','ok')!='ok'
                    unsupported.append(is_unsupported);statuses.append(str(meta.get('unsupported_reason',meta.get('status','ok'))))
                    predictions.append(result['state'].copy());ends.append(cursor+49);run_ids.append(run_id);native_counts.append(count)
                    memory=_bank(tracker,sid).memory_bytes(sid)
                    memory['tracker_role']='main';memories.append(memory)
                    cursor+=50
                    before_commit=time.perf_counter()
                    tracker.commit(sid,result,begin_next=cursor<b)
                    costs['commit_wall_seconds']+=time.perf_counter()-before_commit
                    if len(predictions)%100==0:
                        print(json.dumps({'seq':seq,'mode':wrapper.mode,'scored_packets':len(predictions),
                                          'run_id':run_id,'cursor_ms':cursor,'consumed_events':account['source_count'],
                                          'unsupported':int(sum(unsupported)),
                                          'elapsed_seconds':time.perf_counter()-started}),flush=True)
                if cursor<b:
                    tail_start=time.perf_counter();before_events=account['source_count']
                    _consume_ms(tracker,sid,reader,cursor,b,account)
                    rep=tracker.representation(sid,b*1000)
                    for key in epoch_costs:epoch_costs[key]+=float(rep['meta'].get(key,0.))
                    memory=_bank(tracker,sid).memory_bytes(sid)
                    memory['tracker_role']='main_tail';memories.append(memory)
                    run['tail_ms']=b-cursor;run['tail_events']=account['source_count']-before_events
                    costs['tail_wall_seconds']+=time.perf_counter()-tail_start;cursor=b
                tracker.release(sid)
            run['actual_native_events']=account['source_count']-before_count
            run['scored_packets']=len(predictions)-before_predictions
            run['seconds']=time.perf_counter()-run_started
            if run['actual_native_events']!=run['expected_native_events']:raise AssertionError('run source count mismatch')
            run_records.append(run)
        actual_digest=account['source_digest'].hexdigest()
        if account['source_count']!=expected_count or actual_digest!=expected_digest:
            raise AssertionError('complete-recording source count or exact ordered-ID digest mismatch')
        if tracker.states or tracker.bank.states or tracker.unsupported_bank.states:
            raise AssertionError('final sequence state was not released')
        # No target is read in public mode until the complete forward replay.
        score_start=time.perf_counter()
        pred=np.asarray(predictions,dtype=np.float32).reshape(-1,51)
        target=np.stack([reader.target(end) for end in ends]) if ends else np.empty((0,51),np.float32)
        metrics,per=_metrics(functions,wrapper.base.mano,pred,target,reader.betas)
        costs['scoring_seconds']=time.perf_counter()-score_start
        np.savez_compressed(npz_path,pred=pred,gt=target,end_ms=np.asarray(ends,np.int64),
              run_id=np.asarray(run_ids,np.int64),n_events=np.asarray(native_counts,np.int64),
              unsupported_hold=np.asarray(unsupported,bool),status=np.asarray(statuses,str),**per)
        unchanged={path:sha(path)==info['sha256'] for path,info in source_before.items()}
        if not all(unchanged.values()):raise RuntimeError('an executed source/input changed during replay')
        domain=seq.rsplit('_',1)[-1]
        report={'schema':'s43_integration_complete_replay_v1',
          'at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'sequence':seq,'domain':domain,'mode':wrapper.mode,'config':str(config),
          'checkpoint':None if checkpoint is None else str(checkpoint),'init':init,
          'initialization_role':'label-free public + 2 canonical W50 warmup per run' if init=='public' else
                                'explicit GT-assisted old evaluator initialization',
          'initialization_GT_used':init=='standard','scoring_GT_only_after_public_replay':init=='public',
          'full_recording':True,'valid_runs_total':len(runs),'runs_processed':len(run_records),
          'runs':run_records,'scored_packets':len(pred),'hop_ms':50,'window_ms':50,
          'event_cutoff_target_offset_us':1000,'all_valid_native_events_consumed':True,
          'expected_native_events':expected_count,'actual_native_events':account['source_count'],
          'total_native_events_in_file':len(reader.events),
          'events_outside_valid_runs_not_scored':int(len(reader.events)-expected_count),
          'expected_ordered_source_digest':expected_digest,'actual_ordered_source_digest':actual_digest,
          'native_ms_packets':account['native_ms_packets'],
          'unsupported_hold_packets':int(sum(unsupported)),
          'unsupported_hold_fraction':float(np.mean(unsupported)) if unsupported else 0.,
          'duplicate_query_checks':duplicates,'metrics':metrics,
          'metric_implementation':'unchanged semkine.eval_track.mano_fk/root_align definitions',
          'metrics_are_canonical_main_row':False,'trained_steps_not_6000':True,
          'timing':costs,'epoch_component_seconds':epoch_costs,
          'tracker_totals':tracker.timing,'canonical_warmup_tracker_totals':None if warm_tracker is None else warm_tracker.timing,
          'total_wall_seconds':time.perf_counter()-started,
          'timing_scope':['CPU replay, not sensor-to-mesh latency',
             'commit wall time includes next-epoch reset; do not sum nested component totals',
             'epoch component sums describe scored/main-tail epochs, warmup reported separately',
             'metric decoding is separate from forward replay'],
          'peak_accounted_memory':max(memories,key=lambda m:m['total_accounted_bytes']) if memories else None,
          'peak_memory_scope':'all canonical warmup and main/tail epoch-end snapshots; not RSS or temporary workspace',
          'final_live_sequences':len(tracker.states),'source_files_before':source_before,
          'source_files_unchanged':unchanged,'python':sys.executable,'torch':torch.__version__,
          'numpy':np.__version__,'threads':int(threads),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
          'npz':str(npz_path),'npz_sha256':sha(npz_path)}
        _write(json_path,report)
        print(json.dumps({'report':str(json_path),'full_recording':True,'mode':wrapper.mode,
                          'scored_packets':len(pred),'actual_events':account['source_count'],
                          'unsupported_hold_packets':int(sum(unsupported)),
                          'seconds':report['total_wall_seconds']}),flush=True)
        return report
    except Exception as error:
        if not failed_path.exists():
            _write(failed_path,{'full_recording':False,'error_type':type(error).__name__,'error':str(error),
              'sequence':seq,'mode':wrapper.mode,'consumed_native_events':account['source_count'],
              'expected_native_events':expected_count,'completed_runs':run_records,'scored_packets':len(predictions),
              'seconds':time.perf_counter()-started,'source_files_before':source_before})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--seq',required=True)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--init',choices=['public','standard'],default='public')
    parser.add_argument('--duplicate-query',action='store_true')
    parser.add_argument('--root',type=Path,default=DATA_ROOT)
    args=parser.parse_args()
    if args.threads<1:raise SystemExit('threads must be positive')
    replay(args.config,args.checkpoint,args.seq,args.out,threads=args.threads,
           init=args.init,duplicate_query=args.duplicate_query,root=args.root)


if __name__=='__main__':main()
