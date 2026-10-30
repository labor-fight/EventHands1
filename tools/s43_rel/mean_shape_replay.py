"""Public-mean-shape deployment replay with strictly separate GT scoring.

Inference reads only native events/offsets, external camera K and valid-run
intervals.  All ten inference betas are zero, including canonical warmup and
query geometry.  No dataset pose reader is instantiated.  Predictions are
saved before the separate scoring phase opens pose metadata or actual shape.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import struct
import sys
import time
import traceback

import numpy as np
import torch

from tools.s43_integration.model import load_legacy
from .core import RelConfig
from .geometry import GeometryConfig
from .model import load_relation_model
from .replay import CausalRelationTracker,gpu_sync,summarize_latencies,expected_source,validate_device
from .data import last_lnes
from .training import _source_files

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/s43_rel_20261027'
DATA_ROOT=Path('/data1/lyq/code/mesh/EventHands/data/hand_data51')
PUBLIC_INITIAL=np.array([0.,0.,.6]+[0.]*48,np.float32)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,ensure_ascii=False)+'\n')


def identity(path):
    path=Path(path)
    return {'sha256':sha(path),'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}


class InferenceAccessGuard:
    """Runtime firewall for forbidden score-only data during inference.

    NumPy member access is restored on exit.  Python audit hooks cannot be
    unregistered; the hook becomes inert on exit.  The guard does not alter
    permitted payloads, numerical kernels, model code or checkpoint contents.
    """
    def __init__(self):
        self.active=False
        self.blocked_aux_beta_attempts=0
        self.blocked_pose_meta_attempts=0
        self.aux_members_read=[]

    def _open_hook(self,event,args):
        if not self.active or event!='open' or not args:return
        path=args[0]
        if isinstance(path,(str,bytes,os.PathLike)) and os.fsdecode(path).endswith('.meta'):
            self.blocked_pose_meta_attempts+=1
            raise RuntimeError('inference firewall forbids opening pose .meta')

    def __enter__(self):
        if self.active:raise RuntimeError('guard already active')
        self.active=True
        self._npz_class=np.lib.npyio.NpzFile
        self._original_getitem=self._npz_class.__getitem__
        original=self._original_getitem;guard=self
        def guarded_getitem(archive,key):
            if guard.active and key=='betas':
                guard.blocked_aux_beta_attempts+=1
                raise RuntimeError('inference firewall forbids reading NPZ betas field')
            path=getattr(getattr(archive,'zip',None),'filename',None)
            if guard.active and path is not None and str(path).endswith('_aux.npz'):
                guard.aux_members_read.append({'path':str(path),'member':str(key)})
            return original(archive,key)
        self._npz_class.__getitem__=guarded_getitem
        sys.addaudithook(self._open_hook)
        return self

    def __exit__(self,typ,value,traceback):
        self.active=False
        self._npz_class.__getitem__=self._original_getitem
        return False

    def receipt(self):
        return {'active':self.active,'aux_betas_accessed':False,'pose_metadata_opened':False,
                'blocked_aux_beta_attempts':self.blocked_aux_beta_attempts,
                'blocked_pose_meta_attempts':self.blocked_pose_meta_attempts,
                'aux_members_read':list(self.aux_members_read)}


class MeanShapeInputReader:
    """Inference-only source adapter; no target accessor or pose mapping."""
    def __init__(self,recording,root=DATA_ROOT,*,base_override=None):
        self.root,self.recording=Path(root),str(recording)
        if base_override is None:
            manifest=json.loads((self.root/'splits_semkine.json').read_text())
            locations={seq:info['legacy_dir'][seq] for split in ('train','val_core')
                       for info in (manifest[split],) for seq in info['trials']}
            self.base=self.root/locations[self.recording]/self.recording
        else:self.base=Path(base_override)
        self.events=np.load(str(self.base)+'_events.npy',mmap_mode='r',allow_pickle=False)
        self.offsets=np.load(str(self.base)+'_offsets.npy',mmap_mode='r',allow_pickle=False)
        # Deliberately do not inspect or retrieve the betas member.
        with np.load(str(self.base)+'_aux.npz',allow_pickle=False) as aux:
            self.K_cam=np.asarray(aux['camera_K'],np.float32).reshape(3,3).copy()
            self.runs=np.asarray(aux['valid_runs_ms'],np.int64).reshape(-1,2).copy()
        self.betas=np.zeros(10,np.float32)
        if self.events.ndim!=2 or self.events.shape[1]!=3 or self.events.dtype.kind not in 'iu':
            raise ValueError('native events must be integer [x,y,stored_polarity]')
        if self.offsets.ndim!=1 or self.offsets.dtype.kind not in 'iu' or self.offsets[0]!=0:
            raise ValueError('native integer offsets required')
        if np.any(np.diff(self.offsets)<0) or int(self.offsets[-1])!=len(self.events):
            raise ValueError('event/offset identity mismatch')
        if (not np.isfinite(self.K_cam).all() or self.K_cam[0,0]<=0 or self.K_cam[1,1]<=0 or
            not np.allclose(self.K_cam[2],[0,0,1])):raise ValueError('finite external K required')
        if (np.any(self.runs[:,0]<0) or np.any(self.runs[:,1]<self.runs[:,0]) or
            np.any(self.runs[:,1]>=len(self.offsets)) or
            (len(self.runs)>1 and np.any(self.runs[1:,0]<self.runs[:-1,1]))):
            raise ValueError('valid-run intervals overlap or exceed source')

    def packet(self,start,stop):
        start,stop=int(start),int(stop)
        if not 0<=start<=stop<len(self.offsets):raise ValueError('packet outside source coverage')
        lo,hi=int(self.offsets[start]),int(self.offsets[stop])
        native=np.asarray(self.events[lo:hi],np.int64)
        times=np.repeat(np.arange(start,stop,dtype=np.int64),np.diff(np.asarray(self.offsets[start:stop+1],np.int64)))
        events=np.empty((len(native),4),np.int64)
        events[:,:2],events[:,2],events[:,3]=native[:,:2],times,native[:,2]
        if len(events) and (np.any(events[:,:2]<0) or np.any(events[:,0]>=240) or
            np.any(events[:,1]>=180) or np.any((events[:,3]!=0)&(events[:,3]!=1))):
            raise ValueError('native pixel/polarity outside declared sensor domain')
        return {'events':events,'source_range':(lo,hi)}

    def input_paths(self):
        return [Path(str(self.base)+suffix) for suffix in ('_events.npy','_offsets.npy','_aux.npz')]


def validate_formal(config,checkpoint,recording,extension,postprocess,finalization):
    for gate,key in ((postprocess,'complete'),(finalization,'execution_complete')):
        state=json.loads(Path(gate).read_text())
        if state.get(key) is not True:raise RuntimeError('mean-shape formal work waits for completed postprocess and finalization')
    ext=json.loads(Path(extension).read_text())
    if ext.get('schema')!='s43_rel_mean_shape_deployment_extension_v1' or recording not in ext['recordings']:
        raise RuntimeError('frozen mean-shape extension or recording scope mismatch')
    if checkpoint is not None:
        ckpt=Path(checkpoint);done=json.loads(ckpt.with_name('complete.json').read_text())
        if done.get('stage')!='full' or done.get('completed_steps')!=6000 or done.get('checkpoint_sha256')!=sha(ckpt):
            raise RuntimeError('fixed completed6000 checkpoint required')
    return {'extension':str(extension),'extension_sha256':sha(extension),
            'postprocess_sha256':sha(postprocess),'finalization_sha256':sha(finalization)}


def _mano():
    name='_s43_rel_mean_shape_score_only_mano'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,ROOT/'model/mano_layer.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name].ManoLayer(ROOT/'assets/mano_right.npz',add_mean=True).cpu().eval()


@torch.inference_mode()
def fk(mano,poses,beta):
    poses=np.asarray(poses,np.float32).reshape(-1,51);joints=[];vertices=[]
    for start in range(0,len(poses),256):
        p=torch.from_numpy(poses[start:start+256]);b=torch.from_numpy(np.asarray(beta,np.float32)[None]).expand(len(p),-1)
        v,j=mano(b,p[:,3:6],p[:,6:],p[:,:3]);joints.append(j.numpy());vertices.append(v.numpy())
    if not len(poses):return np.empty((0,21,3),np.float32),np.empty((0,778,3),np.float32)
    return np.concatenate(joints),np.concatenate(vertices)


def shape_separate_metrics(pred,gt,gt_beta):
    """Four original J0/V0 metrics plus true joint-wrist-aligned MPVPE."""
    mano=_mano();pj,pv=fk(mano,pred,np.zeros(10,np.float32));gj,gv=fk(mano,gt,gt_beta)
    norm=lambda x:np.linalg.norm(x,axis=-1).mean(-1)*1000.
    per={'mpjpe_abs_mm':norm(pj-gj),'mpvpe_abs_mm':norm(pv-gv),
         'mpjpe_ra_mm':norm((pj-pj[:,:1])-(gj-gj[:,:1])),
         'mpvpe_ra_mm':norm((pv-pv[:,:1])-(gv-gv[:,:1])),
         'mpvpe_joint_wrist_aligned_mm':norm((pv-pj[:,:1])-(gv-gj[:,:1]))}
    if not all(np.isfinite(v).all() for v in per.values()):raise ArithmeticError('nonfinite separate-shape score')
    return {k:float(v.mean()) for k,v in per.items()} if len(pred) else {},per


def score_saved_predictions(prediction_path,base):
    prediction_path,base=Path(prediction_path),Path(base)
    if not prediction_path.exists():raise RuntimeError('persist every predicted state before score-only GT access')
    prediction_hash=sha(prediction_path)
    with np.load(prediction_path,allow_pickle=False) as z:
        pred=z['pred'].copy();ends=z['end_ms'].copy()
        if not np.array_equal(z['inference_beta'],np.zeros(10,np.float32)):
            raise RuntimeError('predicted FK must use public mean shape')
    first_GT_access_ns=time.time_ns()
    # First actual-shape member and first pose-metadata open in this process.
    with np.load(str(base)+'_aux.npz',allow_pickle=False) as aux:
        gt_beta=np.asarray(aux['betas'],np.float32).reshape(10).copy()
    meta=Path(str(base)+'.meta')
    with meta.open('rb') as f:n,=struct.unpack('<i',f.read(4))
    if n!=51:raise ValueError('GT score-only metadata must be 51D')
    dtype=np.dtype([('data','<f8',51),('m0','u1'),('m1','u1')])
    if (meta.stat().st_size-4)%dtype.itemsize:raise ValueError('truncated score-only metadata')
    source=np.memmap(meta,dtype=dtype,mode='r',offset=4)
    selected=source[ends]
    if np.any(selected['m0']!=4) or np.any(selected['m1']!=13):raise ValueError('invalid score-only GT marker')
    gt=np.asarray(selected['data'],np.float32).copy()
    if not np.isfinite(gt_beta).all() or not np.isfinite(gt).all():raise ValueError('nonfinite score-only targets')
    metrics,per=shape_separate_metrics(pred,gt,gt_beta)
    receipt={'prediction_archive':str(prediction_path),'prediction_archive_sha256':prediction_hash,
        'prediction_archive_mtime_ns':prediction_path.stat().st_mtime_ns,'first_GT_access_ns':first_GT_access_ns,
        'all_predictions_saved_before_GT':first_GT_access_ns>=prediction_path.stat().st_mtime_ns,
        'predicted_FK_beta':'zeros10','GT_FK_beta':'score_only_actual_aux',
        'predicted_beta_zero':True,'GT_beta_sha256':hashlib.sha256(gt_beta.tobytes()).hexdigest(),
        'metric_alignment':{'mpjpe_ra_mm':'joint0/wrist','mpvpe_ra_mm':'vertex0 legacy',
                            'mpvpe_joint_wrist_aligned_mm':'true joint0/wrist supplementary'},
        'score_sources':{str(meta):identity(meta),str(Path(str(base)+'_aux.npz')):identity(str(base)+'_aux.npz')}}
    if not receipt['all_predictions_saved_before_GT']:raise RuntimeError('GT/prediction phase ordering violated')
    return metrics,per,gt,gt_beta,receipt


def replay(config,checkpoint,recording,out,*,device='cpu',threads=2,root=DATA_ROOT,
           extension=OUT/'mean_shape_deployment_extension.json',postprocess=OUT/'postprocess_complete.json',
           finalization=OUT/'finalization_complete.json',engineering_prefix_ms=None):
    validate_device(device);torch.set_num_threads(int(threads))
    prefix=Path(out);prediction_path=Path(str(prefix)+'.predictions.npz')
    if any(Path(str(prefix)+suffix).exists() for suffix in ('.json','.npz','.predictions.npz','_failed.json')):
        raise FileExistsError(prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    if engineering_prefix_ms is None:
        gates=validate_formal(config,checkpoint,recording,extension,postprocess,finalization)
    else:
        if recording.startswith('zgz') or int(engineering_prefix_ms)<1:raise ValueError('engineering uses only non-report prefixes, never zgz scores')
        gates={'engineering_only':True,'formal_gates_bypassed_for_no_score_prefix':True}
    account={'count':0,'ms_packets':0,'digest':hashlib.sha256(),'payload_digest':hashlib.sha256()}
    guard=InferenceAccessGuard();predictions=[];ends=[];run_ids=[];counts=[];latencies=[];run_records=[]
    costs={k:0. for k in ('reset_compile_seconds','ingest_seconds','geometry_seconds','readout_seconds',
        'representation_seconds','context_transfer_seconds','decode_including_observation_transfer_seconds',
        'prediction_transfer_seconds','commit_seconds','warmup_seconds','tail_seconds')}
    memories=[];tracker=None;started=time.perf_counter()
    phase='initialization';active_run=None;active_cutoff=None
    scoring_started=False;inference_complete=False
    try:
        with guard:
            reader=MeanShapeInputReader(recording,root)
            model,cfg=load_relation_model(config,checkpoint,device);model.eval()
            label='canonical' if checkpoint is None else model.arm
            if checkpoint is None and model.arm!='s43':raise ValueError('canonical requires the S43 control config')
            canonical,_=load_legacy(cfg['REL']['BASE_CHECKPOINT'],cfg['REL']['BASE_CONFIG'],device);canonical.eval()
            tracker=CausalRelationTracker(model,RelConfig(**cfg.get('MEMORY',{})),GeometryConfig(**cfg.get('GEOMETRY',{})))
            runs=reader.runs.copy()
            if engineering_prefix_ms is not None:
                a,b=map(int,runs[0]);runs=np.array([[a,min(b,a+int(engineering_prefix_ms))]],np.int64)
            expected_count,expected_digest=expected_source(reader,runs)
            inputs=[*reader.input_paths(),Path(root)/'splits_semkine.json',Path(config),Path(__file__),
                    Path(__file__).with_name('replay.py'),Path(cfg['REL']['BASE_CHECKPOINT']),Path(cfg['REL']['BASE_CONFIG'])]
            if checkpoint is not None:inputs.append(Path(checkpoint))
            sources={str(p.resolve()):identity(p) for p in inputs}
            for path,h in _source_files().items():sources[path]={'sha256':h,'bytes':Path(path).stat().st_size,'mtime_ns':Path(path).stat().st_mtime_ns}
            load_seconds=time.perf_counter()-started
            if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
            stream_started=time.perf_counter()

            def consume(start,stop,ordinal):
                nonlocal phase,active_cutoff
                for ms in range(start,stop):
                    phase='native_ms_ingest';active_cutoff=ms+1
                    raw=reader.packet(ms,ms+1)
                    tracker.ingest(raw['events'],watermark_ms=ms+1,packet_ordinal=ordinal,source_range=raw['source_range'])
                    lo,hi=raw['source_range'];account['digest'].update(np.arange(lo,hi,dtype=np.int64).tobytes())
                    account['payload_digest'].update(raw['events'].tobytes());account['count']+=hi-lo;account['ms_packets']+=1;ordinal+=1
                return ordinal

            with torch.inference_mode():
                for run,(a,b) in enumerate(runs):
                    active_run=run
                    a,b=int(a),int(b);before_count=account['count'];before_outputs=len(predictions)
                    phase='valid_run_reset';active_cutoff=a
                    t=time.perf_counter();tracker.reset(f'{recording}:beta0:run{run}',a,PUBLIC_INITIAL,reader.betas,reader.K_cam)
                    costs['reset_compile_seconds']+=time.perf_counter()-t
                    cursor,ordinal,warmups=a,0,0;t=time.perf_counter()
                    for _ in range(2):
                        if b-cursor<50:break
                        ordinal=consume(cursor,cursor+50,ordinal);cursor+=50
                        phase='canonical_mean_shape_warmup';active_cutoff=cursor
                        observed=tracker.window.read(cursor);image=last_lnes(observed,cursor-50,cursor)
                        prediction=canonical(torch.as_tensor(image[None],device=device),torch.as_tensor(tracker.pose[None],device=device),
                            torch.as_tensor(reader.betas[None],device=device),torch.as_tensor(reader.K_cam[None],device=device))
                        gpu_sync(device)
                        warm_state=prediction[0].float().cpu().numpy()
                        if warm_state.shape!=(51,) or not np.isfinite(warm_state).all():
                            raise ArithmeticError('nonfinite canonical beta0 warmup; no fallback')
                        tracker.canonical_commit(cursor,warm_state);warmups+=1
                    costs['warmup_seconds']+=time.perf_counter()-t
                    if warmups==2:
                        while b-cursor>=50:
                            gpu_sync(device);cycle=time.perf_counter();before_query_count=account['count']
                            t=time.perf_counter();ordinal=consume(cursor,cursor+50,ordinal);cursor+=50
                            costs['ingest_seconds']+=time.perf_counter()-t
                            phase='tracked_mean_shape_query';active_cutoff=cursor
                            result=tracker.query(cursor)
                            for key,v in result['costs'].items():costs[key]+=v
                            phase='tracked_mean_shape_commit'
                            t=time.perf_counter();tracker.commit(result);gpu_sync(device);costs['commit_seconds']+=time.perf_counter()-t
                            latencies.append(time.perf_counter()-cycle);predictions.append(result['state']);ends.append(cursor-1)
                            run_ids.append(run);counts.append(account['count']-before_query_count);memories.append(tracker.memory_stats())
                            if len(predictions)%100==0:print(json.dumps({'arm':label,'recording':recording,'outputs':len(predictions),'source_count':account['count']}),flush=True)
                    tail_ms=b-cursor;t=time.perf_counter();before_tail=account['count']
                    if cursor<b:ordinal=consume(cursor,b,ordinal)
                    costs['tail_seconds']+=time.perf_counter()-t;tail_events=account['count']-before_tail
                    memories.append(tracker.memory_stats());tracker.release()
                    consumed=account['count']-before_count
                    if consumed!=int(reader.offsets[b]-reader.offsets[a]):raise AssertionError('valid-run source consumption mismatch')
                    run_records.append({'run_id':run,'start_ms':a,'stop_ms':b,'warmup_windows':warmups,
                        'outputs':len(predictions)-before_outputs,'source_count':consumed,'tail_ms':tail_ms,'tail_events':tail_events})
            gpu_sync(device);stream_seconds=time.perf_counter()-stream_started
            if account['count']!=expected_count or account['digest'].hexdigest()!=expected_digest:
                raise AssertionError('complete source count/digest mismatch')
            if not np.array_equal(reader.betas,np.zeros(10,np.float32)):raise AssertionError('inference shape changed')
            phase='persist_all_inference_predictions'
            pred=np.asarray(predictions,np.float32).reshape(-1,51)
            np.savez_compressed(prediction_path,pred=pred,end_ms=np.asarray(ends,np.int64),run_id=np.asarray(run_ids,np.int64),
                n_new_events=np.asarray(counts,np.int64),latency_seconds=np.asarray(latencies),inference_beta=np.zeros(10,np.float32))
            prediction_saved_ns=time.time_ns();prediction_sha=sha(prediction_path)
            access=guard.receipt()
            if access['blocked_aux_beta_attempts'] or access['blocked_pose_meta_attempts']:raise RuntimeError('forbidden inference GT access attempted')
            unchanged={p:sha(p)==v['sha256'] for p,v in sources.items()}
            if not all(unchanged.values()):raise RuntimeError('inference source changed')
            inference_complete=True
        # Access guard is now inactive, and every predicted state is already on disk.
        infer_access=guard.receipt()
        metrics,per,gt,gt_beta,score_receipt={},{},np.empty((0,51),np.float32),None,None
        scoring_seconds=0.
        if engineering_prefix_ms is None:
            phase='score_only_after_prediction_save';scoring_started=True
            t=time.perf_counter();metrics,per,gt,gt_beta,score_receipt=score_saved_predictions(prediction_path,reader.base)
            scoring_seconds=time.perf_counter()-t
        np.savez_compressed(prefix.with_suffix('.npz'),pred=pred,gt=gt,end_ms=np.asarray(ends,np.int64),
            run_id=np.asarray(run_ids,np.int64),n_new_events=np.asarray(counts,np.int64),
            latency_seconds=np.asarray(latencies),inference_beta=np.zeros(10,np.float32),
            **({'score_only_GT_beta':gt_beta} if gt_beta is not None else {}),**per)
        report={'schema':'s43_relation_mean_shape_full_recording_replay_v1','recording':recording,'arm':label,'model_arm':model.arm,
            'shape_condition':'public_mean_beta0','config':str(config),'checkpoint':None if checkpoint is None else str(checkpoint),
            'inference_beta':[0.]*10,'aux_betas_accessed_during_inference':False,'pose_metadata_opened_during_inference':False,
            'GT_only_after_predictions_saved':True,'predicted_FK_beta':'zeros10','GT_FK_beta':'score_only_actual_aux',
            'inference_access_guard':infer_access,'input_reader':'MeanShapeInputReader; no pose dataset reader',
            'scoring_performed':engineering_prefix_ms is None,'scoring_receipt':score_receipt,
            'prediction_archive':str(prediction_path),'prediction_archive_sha256':prediction_sha,'prediction_saved_ns':prediction_saved_ns,
            'device':device,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'hop_ms':50,'observation_window_ms':50,
            'output_lag_ms':1,'commit_schedule_ms':50,'initialization':'public state + two shared frozen canonical beta0 W50 predictions',
            'initialization_GT':False,'target_role':'scoring only after all predictions persisted',
            'full_recording':engineering_prefix_ms is None,'engineering_prefix_ms':engineering_prefix_ms,
            'native_ms_packets':account['ms_packets'],'source_count':account['count'],'expected_source_count':expected_count,
            'source_id_digest':account['digest'].hexdigest(),'expected_source_id_digest':expected_digest,
            'source_payload_digest':account['payload_digest'].hexdigest(),'all_valid_events_consumed_once':True,
            'outputs':len(pred),'metrics':metrics,'runs':run_records,'params':model.counts(),'costs':costs,
            'load_seconds':load_seconds,'stream_seconds':stream_seconds,'scoring_seconds':scoring_seconds,
            'pipeline_latency':summarize_latencies(latencies),
            'events_per_second_including_warmup_and_tails':expected_count/max(stream_seconds,1e-12),
            'outputs_per_second_including_warmup_and_tails':len(pred)/max(stream_seconds,1e-12),
            'RSS_peak_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
            'cuda_memory':{'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)} if str(device).startswith('cuda') else {},
            'peak_resident_observation_bytes':max((m['resident_bytes'] for m in memories),default=0),
            'peak_shared_topology_bytes':max((m.get('shared_topology_bytes',0) for m in memories),default=0),
            'peak_raw_window_bytes':max((m['raw_window_bytes'] for m in memories),default=0),
            'source_files':sources,'source_files_unchanged':unchanged,'extension_gates':gates,
            'scientific_scope':'public-mean-shape deployment; known-shape conditional results are separate diagnostics; no shape-free accuracy claim',
            'latency_scope':'synchronized replay service; excludes physical arrival waiting, score-only FK and source hash/verification',
            'argv':sys.argv}
        write(prefix.with_suffix('.json'),report)
        return report
    except BaseException as error:
        preserved_prediction=prediction_path if prediction_path.exists() else Path(str(prefix)+'.partial_predictions.npz')
        if not preserved_prediction.exists():
            np.savez_compressed(preserved_prediction,pred=np.asarray(predictions,np.float32).reshape(-1,51),
                end_ms=np.asarray(ends,np.int64),run_id=np.asarray(run_ids,np.int64),
                n_new_events=np.asarray(counts,np.int64),inference_beta=np.zeros(10,np.float32),
                inference_complete=np.array(False))
        pose_value=None if tracker is None else getattr(tracker,'pose',None)
        last_pose=None if pose_value is None or not np.isfinite(pose_value).all() else pose_value.tolist()
        write(Path(str(prefix)+'_failed.json'),{'error':repr(error),'source_count':account['count'],
            'outputs':len(predictions),'inference_access_guard':guard.receipt(),'GT_scoring_started':scoring_started,
            'full_recording':False,'full_inference_completed':inference_complete,'failure_phase':phase,
            'failure_run_id':active_run,'failure_cutoff_ms':active_cutoff,
            'last_committed_state':last_pose,'last_commit_ms':None if tracker is None else getattr(tracker,'last_commit_ms',None),
            'source_id_digest_consumed_so_far':account['digest'].hexdigest(),
            'preserved_prediction_archive':str(preserved_prediction),'preserved_prediction_archive_sha256':sha(preserved_prediction),
            'no_hold_clamp_or_GT_shape_fallback':True,'traceback':traceback.format_exc()})
        raise


def self_test(out):
    """Independent scoring fixture; does not read any recording target data."""
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    # Same pose, different known synthetic shapes must not score as identical.
    p=np.tile(PUBLIC_INITIAL,(2,1));beta=np.linspace(-.4,.5,10).astype(np.float32)
    metrics,per=shape_separate_metrics(p,p,beta)
    if metrics['mpvpe_abs_mm']<=0 or metrics['mpjpe_abs_mm']<=0:raise AssertionError('cross-shape scoring incorrectly reuses the same beta')
    zero,_=shape_separate_metrics(p,p,np.zeros(10,np.float32))
    if any(v!=0 for v in zero.values()):raise AssertionError('identical pose and mean-shape scoring identity failed')
    guard=InferenceAccessGuard()
    with guard:
        with np.load(ROOT/'assets/mano_right.npz',allow_pickle=False) as asset:asset['hands_mean']
        try:
            with np.load(ROOT/'assets/mano_right.npz',allow_pickle=False) as asset:asset['betas']
        except RuntimeError:pass
        else:raise AssertionError('beta member firewall inactive')
        try:(out.parent/'forbidden_fixture.meta').open('rb')
        except RuntimeError:pass
        else:raise AssertionError('pose metadata firewall inactive')
    receipt=guard.receipt()
    if receipt['blocked_aux_beta_attempts']!=1 or receipt['blocked_pose_meta_attempts']!=1:raise AssertionError('firewall accounting failed')
    result={'schema':'mean_shape_scoring_fixture_v1','status':'PASS','source_sha256':sha(__file__),
        'separate_shape_fixture_positive':True,'identical_shape_fixture_exact_zero':True,
        'guard_unit_test':receipt,'recorded_GT_read':False,'zgz_accessed':False,'task_scores_computed':False,
        'metric_keys':list(per),'shape_scoring_definition':'pred beta0; GT explicitly supplied synthetic different beta'}
    write(out,result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--checkpoint');p.add_argument('--recording')
    p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--root',default=str(DATA_ROOT));p.add_argument('--extension',default=str(OUT/'mean_shape_deployment_extension.json'))
    p.add_argument('--postprocess-complete',default=str(OUT/'postprocess_complete.json'))
    p.add_argument('--finalization-complete',default=str(OUT/'finalization_complete.json'))
    p.add_argument('--engineering-prefix-ms',type=int);p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:result=self_test(a.out)
    else:
        if not a.config or not a.recording:p.error('--config and --recording are required')
        checkpoint=None if a.checkpoint in (None,'None','none') else a.checkpoint
        result=replay(a.config,checkpoint,a.recording,a.out,device=a.device,threads=a.threads,root=a.root,
            extension=a.extension,postprocess=a.postprocess_complete,finalization=a.finalization_complete,
            engineering_prefix_ms=a.engineering_prefix_ms)
    print(json.dumps({'schema':result['schema'],'out':a.out,'outputs':result.get('outputs'),'scoring_performed':result.get('scoring_performed',False)}),flush=True)


if __name__=='__main__':main()
