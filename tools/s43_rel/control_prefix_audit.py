"""Trained four-arm B2/W50 versus independent B1/native-ms parity audit.

No optimizer, target pose or task-risk calculation is used.  Each comparison
conditions on exactly the same previous FP32 prediction; the batched result
is explicitly committed to both builders after measuring B2/B1 differences.
This controls floating-point feedback divergence rather than scoring it.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from tools.s43_br.data import PUBLIC_INITIAL_STATE,SequenceReader
from .core import RelConfig
from .data import ContinuousRunLanes,last_lnes
from .geometry import GeometryConfig
from .model import ARMS,load_relation_model
from .replay import CausalRelationTracker,duplicate_query_check
from .training import RelationRuntime,_source_files

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/s43_rel_20261027'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def digest(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a);h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()


def compare(a,b,rtol=3e-6,atol=2e-6):
    if torch.is_tensor(a):a=a.detach().cpu().numpy()
    if torch.is_tensor(b):b=b.detach().cpu().numpy()
    if a is None or b is None:
        return {'both_absent':a is None and b is None,'pass':a is None and b is None}
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape:return {'shape_training':list(a.shape),'shape_online':list(b.shape),'pass':False}
    finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
    diff=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return {'shape':list(a.shape),'bitwise_equal':bool(np.array_equal(a,b)),
            'max_abs':float(diff.max(initial=0.)),
            'relative_L2':float(np.linalg.norm(diff)/max(np.linalg.norm(b.astype(np.float64)),1e-30)),
            'rtol':rtol,'atol':atol,'finite':finite,
            'pass':bool(finite and np.allclose(a,b,rtol=rtol,atol=atol))}


def weight_digest(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():h.update(key.encode());h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def packet_item(reader,start,stop,sid,ordinal,reset):
    raw=ContinuousRunLanes._packet(reader,start,stop)
    return dict(raw,seq_id=sid,packet_ordinal=ordinal,reset=reset,released_seq_id=None,
        released_tail=None,run_start_ms=0,start_ms=start,cutoff_ms=stop,
        initial_state=PUBLIC_INITIAL_STATE.copy(),betas=reader.betas.copy(),K_cam=reader.K_cam.copy(),
        lnes=last_lnes(raw['events'],start,stop))


def field_compare(training,online,key):
    present=(key in training,key in online)
    if not any(present):return {'present_training':False,'present_online':False,'pass':True,
                                'meaning':'branch-disabled field absent in both; not synthesized as fake zeros'}
    return dict(compare(training.get(key),online.get(key)),present_training=present[0],present_online=present[1])


def audit_arm(directory,arm,readers,require_final=False):
    checkpoint=Path(directory)/arm/'last.ckpt';config=checkpoint.with_name('config_resolved.json')
    complete=json.loads(checkpoint.with_name('complete.json').read_text())
    if complete.get('checkpoint_sha256')!=sha(checkpoint):raise RuntimeError('completion/checkpoint hash mismatch')
    if complete.get('completed_steps') not in (120,6000):raise RuntimeError('only completed small or final checkpoint allowed')
    if require_final and not (complete.get('stage')=='full' and complete.get('completed_steps')==6000):
        raise RuntimeError('final audit cannot use incomplete or small weights')
    model,cfg=load_relation_model(config,checkpoint,device='cpu');model.eval()
    initial_weights=weight_digest(model)
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise RuntimeError('FP32 model required')
    runtime=RelationRuntime(model,RelConfig(**cfg.get('MEMORY',{})),GeometryConfig(**cfg.get('GEOMETRY',{})),prep_workers=2)
    trackers={name:CausalRelationTracker(model,runtime.rel_config,runtime.geometry_config) for name in readers}
    for name,r in readers.items():trackers[name].reset(name,0,PUBLIC_INITIAL_STATE,r.betas,r.K_cam)
    report={'arm':arm,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
            'completed_steps':complete['completed_steps'],'stage':complete['stage'],
            'config_sha256':sha(config),'reader':model.reader,
            'branches':{'sensor_active':arm!='s43','typed_active':arm in ('full','no_order'),
                'sensor_weight_norm':float(model.sensor_adapter.weight.norm().detach()),
                'typed_output_weight_norm':float(model.typed_adapter.output.weight.norm().detach())},
            'steps':[],'duplicate_queries':[]}
    all_pass=True
    try:
        with torch.inference_mode():
            for step,stop in enumerate((50,100,150,200)):
                start=stop-50
                items=[packet_item(r,start,stop,name,step,step==0) for name,r in readers.items()]
                for name,r in readers.items():
                    for ms in range(start,stop):
                        raw=ContinuousRunLanes._packet(r,ms,ms+1)
                        trackers[name].ingest(raw['events'],watermark_ms=ms+1,packet_ordinal=ms,
                                              source_range=raw['source_range'])
                reps,previous,beta,K,snapshots=runtime.inputs(items)
                prediction=model(reps,previous,beta,K)
                row={'cutoff_ms':stop,'recordings':{}}
                for i,(name,r) in enumerate(readers.items()):
                    tracker=trackers[name]
                    rep,snapshot,_=tracker.representation(stop)
                    fields={key:field_compare(reps[i],rep,key) for key in ('lnes','sensor','f','a','lam')}
                    fields['previous']=compare(previous[i],tracker.pose,rtol=0,atol=0)
                    state={'snapshot_A':compare(snapshots[i].A,snapshot.A)}
                    if arm in ('full','no_order'):
                        state['snapshot_Z']=compare(snapshots[i].Z,snapshot.Z)
                        state['snapshot_edge_ids']=compare(snapshots[i].edge_ids,snapshot.edge_ids,rtol=0,atol=0)
                    if arm=='enhanced':
                        state['enhanced_EMA_numeric_state']=compare(runtime.memories[name].A,tracker.memory.A)
                        state['enhanced_EMA_cutoff_equal']=runtime.memories[name].cutoff_ms==tracker.memory.cutoff_ms==stop
                        channel_groups={'last':(0,2),'first':(2,4),'count':(4,6),'bins4':(6,14),'EMA':(14,20)}
                        fields['enhanced_channel_groups']={k:compare(reps[i]['sensor'][...,a:b],rep['sensor'][...,a:b])
                                                          for k,(a,b) in channel_groups.items()}
                    result=tracker.query(stop)
                    pred_compare=compare(prediction[i],result['state'],rtol=1e-5,atol=2e-6)
                    pred_compare['strict_absolute_2e6_pass']=pred_compare['max_abs']<=2e-6
                    clock={'delta_commit_ms_training':float(reps[i]['delta_commit_ms']),
                           'delta_commit_ms_online':float(rep['delta_commit_ms']),
                           'output_lag_ms_training':float(reps[i]['output_lag_ms']),
                           'output_lag_ms_online':float(rep['output_lag_ms']),
                           'training_semantic_cutoff_ms':reps[i]['meta']['cutoff_ms'],
                           'online_semantic_cutoff_ms':rep['meta']['cutoff_ms'],
                           'online_watermark_ms':tracker.watermark_ms,'target_ms_online':result['target_ms'],
                           'last_commit_training':runtime.last_cutoffs[name],'last_commit_online':tracker.last_commit_ms,
                           'missing_training':bool(reps[i]['missing']),'missing_online':bool(rep['missing']),
                           'same_seq_id':reps[i]['meta']['seq_id']==rep['meta']['seq_id']}
                    clock['pass']=bool(clock['delta_commit_ms_training']==clock['delta_commit_ms_online']==50 and
                        clock['output_lag_ms_training']==clock['output_lag_ms_online']==1 and
                        clock['training_semantic_cutoff_ms']==clock['online_semantic_cutoff_ms']==clock['online_watermark_ms']==stop and
                        clock['target_ms_online']==stop-1 and clock['last_commit_training']==clock['last_commit_online']==start and
                        clock['missing_training']==clock['missing_online'] and clock['same_seq_id'])
                    branch_activity={'lnes_nonzero':bool(np.any(rep['lnes'])),
                        'sensor_nonzero':bool(np.any(rep['sensor'])),
                        'f_nonzero':bool(np.any(rep.get('f',np.empty(0)))),
                        'lambda_nonzero':bool(np.any(rep.get('lam',np.empty(0))))}
                    checks=[value['pass'] for key,value in fields.items() if key!='enhanced_channel_groups']
                    checks += [v['pass'] for v in fields.get('enhanced_channel_groups',{}).values()]
                    checks += [v['pass'] if isinstance(v,dict) else bool(v) for v in state.values()]
                    checks += [pred_compare['pass'],pred_compare['strict_absolute_2e6_pass'],clock['pass']]
                    case_pass=all(checks);all_pass &= case_pass
                    row['recordings'][name]={'pass':case_pass,'fields':fields,'state':state,'prediction':pred_compare,
                        'semantic_clocks':clock,'branch_activity':branch_activity,
                        'packet_metadata_intentionally_different':{'training_ordinal':reps[i]['meta']['packet_ordinal'],
                            'online_ordinal':rep['meta']['packet_ordinal'],
                            'training_source_range':reps[i]['meta']['source_range'],
                            'online_source_range':rep['meta']['source_range'],
                            'reason':'50ms submitted packet ordinal versus native-ms ordinal; online meta source_range is W50 read-support placeholder, not a new source submission.'}}
                    if step in (0,3):
                        _,pure=duplicate_query_check(tracker,stop)
                        report['duplicate_queries'].append(dict(recording=name,**pure))
                runtime.commit(items,prediction,snapshots)
                for i,name in enumerate(readers):trackers[name].canonical_commit(stop,prediction[i].cpu().numpy())
                report['steps'].append(row)
        report['model_weights_and_BN_unchanged']=weight_digest(model)==initial_weights
        report['pass']=bool(all_pass and report['model_weights_and_BN_unchanged'])
    finally:
        if runtime._pool:runtime._pool.shutdown()
        for tracker in trackers.values():tracker.release()
    del runtime,trackers,model;gc.collect()
    return report


def main(directory,arms,out,require_final=False):
    torch.set_num_threads(1)
    sources=_source_files();sources[str(ROOT/'tools/s43_rel/replay.py')]=sha(ROOT/'tools/s43_rel/replay.py')
    readers={name:SequenceReader(name) for name in ('ch_local','lfz_global')}
    for reader in readers.values():
        if not np.any((reader.runs[:,0]==0)&(reader.runs[:,1]>=200)):raise RuntimeError('required first200ms invalid')
    report={'schema':'s43_rel_control_prefix_parity_v1','tool_sha256':sha(__file__),
        'checkpoint_dir':str(directory),'dtype':'FP32, autocast disabled','device':'CPU','torch':torch.__version__,
        'numpy':np.__version__,'python':sys.version,'sources':sources,
        'source_scope':'22 frozen training dependencies plus independently read replay source',
        'input_scope':'ch_local and lfz_global first complete200ms, B2/W50 training versus independent B1/native-ms online',
        'feedback_control':'Common detached B2 prediction explicitly committed to both builders after measuring prediction differences; no GT history.',
        'tolerances':{'representation_rtol':3e-6,'representation_atol':2e-6,
            'prediction_rtol':1e-5,'prediction_atol':2e-6,'prediction_additional_strict_abs_max':2e-6},
        'results':[],'GT_targets_read':False,'zgz_accessed':False,'optimizer_used':False,
        'recordings':{name:{'events':int(r.offsets[200]-r.offsets[0]),
            'source_rows':[int(r.offsets[0]),int(r.offsets[200])],
            'event_sha256':digest(r.events[int(r.offsets[0]):int(r.offsets[200])])} for name,r in readers.items()}}
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    for arm in arms:
        result=audit_arm(directory,arm,readers,require_final)
        report['results'].append(result)
        # Persist each completed arm; tool failures must not erase evidence.
        out.write_text(json.dumps(dict(report,status='IN_PROGRESS'),indent=2,allow_nan=False)+'\n')
        print(json.dumps({'arm':arm,'pass':result['pass'],'completed_steps':result['completed_steps']}),flush=True)
    report['source_files_unchanged']=all(sha(p)==h for p,h in sources.items())
    report['status']='PASS' if report['source_files_unchanged'] and all(r['pass'] for r in report['results']) else 'MISMATCH'
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'out':str(out)}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-dir',type=Path,default=OUT/'train/small_r1')
    p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS));p.add_argument('--require-final',action='store_true')
    p.add_argument('--out',type=Path,default=OUT/'audits/control_prefix_small_r1.json')
    a=p.parse_args();main(a.checkpoint_dir,a.arms,a.out,a.require_final)
