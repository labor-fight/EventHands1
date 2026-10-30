"""Target-free experimental TimeField -> MANO51 streaming client.

No SequenceReader, target/meta/aux access, LNES, old sparse mask/gate or old
prev_mlp. Atomic checkpoints are runnable engineering, not full-S43 training.
"""
from __future__ import annotations
import argparse,json,os,resource,time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from .field_core import TimeFieldConfig,TimeFieldMemory,integer
from .field_query import FieldQueryBuilder,FieldQueryConfig
from .field_mano_decoder import FieldManoDecoder,ASSET
from .newfield_mano_probe import IdentityFeatureReader
from . import field_query_probe as P
ROOT=P.ROOT;PUBLIC_INITIAL=np.asarray([0.,0.,.6]+[0.]*48,np.float32)
COMPLETE_ARMS=('context_only','global_only','hand_box','static','static_temporal_collapse','anatomical')
sha=P.sha;dump=P.dump


def readonly(x):
    a=np.array(x,copy=True);a.flags.writeable=False;return a

def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)

def validate_device(device):P.old.validate_device(str(device))


class CostSummary:
    """Bounded timing reservoir; exact counts/sums/max and small-run quantiles."""
    def __init__(self,capacity=4096):self.capacity=capacity;self.values={};self.rng=np.random.default_rng(3407)
    def add(self,key,ms):
        item=self.values.setdefault(key,dict(n=0,total_ms=0.,max_ms=0.,samples=[]));item['n']+=1;item['total_ms']+=float(ms);item['max_ms']=max(item['max_ms'],float(ms))
        if len(item['samples'])<self.capacity:item['samples'].append(float(ms))
        else:
            i=int(self.rng.integers(item['n']))
            if i<self.capacity:item['samples'][i]=float(ms)
    def report(self):
        result={}
        for k,v in self.values.items():
            a=np.asarray(v['samples']);result[k]={q:v[q] for q in ('n','total_ms','max_ms')}
            result[k].update(mean_ms=v['total_ms']/v['n'],p50_ms=float(np.percentile(a,50)),p95_ms=float(np.percentile(a,95)),p99_ms=float(np.percentile(a,99)),percentiles_exact=v['n']<=self.capacity)
        return dict(phases=result,reservoir_capacity=self.capacity,timing_state_bound='perphase4096floats; exact totals/count/max, reservoirpercentiles aftercapacity')


class MMapEventStream:
    """External integer Nx4 source; no O(T) timestamp/normalized-event copy."""
    def __init__(self,path,time_unit='ms',start_ms=None,stop_ms=None,max_packet_events=262144):
        if time_unit not in ('ms','us'):raise ValueError('time_unit ms or us')
        self.path=Path(path);self.raw=np.load(self.path,mmap_mode='r',allow_pickle=False)
        if self.raw.ndim!=2 or self.raw.shape[1]!=4 or not np.issubdtype(self.raw.dtype,np.integer):raise ValueError('events.npy must be integer Nx4[x,y,t,p]')
        self.factor=1 if time_unit=='ms' else 1000;self.time_unit=time_unit;self.cap=integer(max_packet_events,'packetcap')
        if self.cap<=0:raise ValueError('positivepacketcap required')
        self.times=self.raw[:,2]  # strided mmap view, not a full timestamp copy
        before=None
        for lo in range(0,len(self.raw),65536):
            a=np.asarray(self.raw[lo:lo+65536]);t=a[:,2]
            if len(a) and (np.any(a[:,:3]<0) or np.any(a[:,0]>=240) or np.any(a[:,1]>=180) or np.any(a[:,2]>np.iinfo(np.int64).max) or np.any((a[:,3]!=0)&(a[:,3]!=1)) or np.any(t[1:]<t[:-1]) or (before is not None and t[0]<before)):raise ValueError('event range/polarity/raw timestamp order rejected before stream reset')
            if len(t):before=int(t[-1])
        first=int(self.times[0])//self.factor if len(self.raw) else 0;last=int(self.times[-1])//self.factor+1 if len(self.raw) else first
        self.start=first if start_ms is None else integer(start_ms,'start_ms');self.stop=last if stop_ms is None else integer(stop_ms,'stop_ms')
        if self.start<0 or self.start>np.iinfo(np.int64).max-50 or self.stop>np.iinfo(np.int64).max or self.start>first or self.stop<last or self.stop<self.start:raise ValueError('declared observation boundary/end would discard native input or overflowclock')
        self.cursor=0;self.next_ms=self.start
    def peek_ms(self,ms):
        if ms!=self.next_ms or ms>=self.stop:raise ValueError('consecutive1ms observation packets required')
        hi=int(np.searchsorted(self.times,(ms+1)*self.factor,side='left'))
        if hi-self.cursor>self.cap:raise ValueError('native-ms packet exceeds declared transportcap; reject without clipping or cursor advance')
        a=np.asarray(self.raw[self.cursor:hi],np.int64).copy()
        if self.factor!=1:a[:,2]//=self.factor
        return a,(self.cursor,hi)
    def commit_ms(self,ms,hi):
        if ms!=self.next_ms or not self.cursor<=hi<=len(self.raw):raise ValueError('stale sourcepacket commit')
        self.cursor=int(hi);self.next_ms=ms+1
    def close(self):
        if isinstance(self.raw,np.memmap):self.raw._mmap.close()


def calibration(path,shape='mean0'):
    x=json.loads(Path(path).read_text());has_source='K_source' in x;has_event='K_event' in x
    if has_source==has_event:raise ValueError('exactly one K_source640x480 or K_event240x180 required')
    given=np.asarray(x['K_source'] if has_source else x['K_event'],np.float64)
    if given.shape!=(3,3) or not np.isfinite(given).all() or min(given[0,0],given[1,1])<=0 or not np.allclose(given[2],[0,0,1],rtol=0,atol=1e-12):raise ValueError('finite pinhole3x3 calibration required')
    source=given.copy();event=given.copy()
    if has_source:event[:2]*=.375
    else:source[:2]/=.375
    if shape not in ('mean0','provided'):raise ValueError('shape mode')
    beta=np.zeros(10,np.float32) if shape=='mean0' else np.asarray(x['betas'],np.float32)
    if beta.shape!=(10,) or not np.isfinite(beta).all():raise ValueError('explicit providedbetas10 required')
    return source,event,beta,dict(shape_condition='beta0' if shape=='mean0' else 'explicit_external_beta_conditional',K_contract='source640→event240*.375 once' if has_source else 'provided eventK kept; contextsource /.375 once',boundary_ms=x.get('boundary_ms'),stop_ms=x.get('stop_ms'))


@dataclass(frozen=True)
class SurfaceQuery:
    seq_id:object
    cutoff_ms:int
    pose_version:int
    observation_version:int
    state51:np.ndarray
    vertices_m:np.ndarray
    joints_m:np.ndarray
    raw51:np.ndarray
    timings_ms:dict
    _owner:object


class FieldManoOnline:
    """One sequence per instance; sensor state, query and commit stay separate."""
    def __init__(self,model,stats,K_source,beta=None,*,K_event=None,device='cpu',metadata=None):
        validate_device(device);self.device=str(device);self.model=model.to(device).eval();self.model.requires_grad_(False)
        if model.reader!='mano51' or model.arm not in COMPLETE_ARMS:raise ValueError('online supports only mano51 completefield checkpoints; quota uses a different sparse observation')
        self.arm=model.arm;self.decoder=FieldManoDecoder().to(device);self.beta=readonly(np.zeros(10,np.float32) if beta is None else np.asarray(beta,np.float32))
        self.K_source=readonly(np.asarray(K_source,np.float64));ev=np.array(self.K_source,copy=True);ev[:2]*=.375
        self.K_event=readonly(ev if K_event is None else np.asarray(K_event,np.float64))
        if self.beta.shape!=(10,) or not np.isfinite(self.beta).all() or self.K_source.shape!=(3,3) or self.K_event.shape!=(3,3) or not np.isfinite(self.K_source).all() or not np.isfinite(self.K_event).all():raise ValueError('finite beta/calibration required')
        if not np.allclose(self.K_event,ev,atol=1e-6,rtol=1e-7):raise ValueError('Kevent does not match the declared singlescale camera')
        expected={'global_scale':40,'local_scale':5,'context_mean':70,'context_std':70};self.stats={}
        for key,n in expected.items():
            a=np.asarray(stats[key],np.float32)
            if a.shape!=(n,) or not np.isfinite(a).all() or (key!='context_mean' and np.any(a<=0)):raise ValueError('invalid checkpoint normalization '+key)
            self.stats[key]=readonly(a)
        self.gs=torch.as_tensor(self.stats['global_scale'].copy(),device=device)[None,:,None,None]
        self.ls=torch.as_tensor(self.stats['local_scale'].copy(),device=device)[None,None,None,:,None,None]
        self.metadata={} if metadata is None else dict(metadata);self.cost=CostSummary();self.memory=TimeFieldMemory(TimeFieldConfig(resolution=2));self.reset('default',0)
    def reset(self,seq_id,boundary_ms=0,initial_state=None):
        q=np.asarray(PUBLIC_INITIAL if initial_state is None else initial_state,np.float32)
        if q.shape!=(51,) or not np.isfinite(q).all():raise ValueError('finite external/publicinitial51 required')
        self.memory.reset(seq_id,boundary_ms);self.seq_id=seq_id;self.pose=readonly(q);self.last_commit=int(boundary_ms);self.pose_version=0
        self.history=[(self.last_commit,self.pose)];self._owner=object();self.released=False;self.cost=CostSummary();return self
    def ingest(self,events,watermark_ms,packet_ordinal,*,source_range,seq_id=None,valid=True):
        started=time.perf_counter();self.memory.ingest(events,watermark_ms,packet_ordinal,source_range=source_range,seq_id=seq_id,valid=valid)
        ms=(time.perf_counter()-started)*1000;self.cost.add('sensor_ingest',ms);return ms
    def query(self,cutoff_ms=None):
        if self.released:raise RuntimeError('released online state')
        T=self.memory.watermark_ms if cutoff_ms is None else integer(cutoff_ms,'cutoff_ms');timing={};sync(self.device);begin=time.perf_counter()
        t=time.perf_counter();snap=self.memory.readout(T,seq_id=self.seq_id);grid=snap.to_grid40ch()
        if self.arm=='static_temporal_collapse':grid=P.collapse_time(grid)
        context=P.old.condition70(self.pose,self.beta,self.K_source);context=np.clip((context-self.stats['context_mean'])/self.stats['context_std'],-8,8).astype(np.float32)
        timing['observation_readout_and_conditioning']=(time.perf_counter()-t)*1000
        t=time.perf_counter();coordinates=None
        if self.model.use_local:
            mode='static' if self.arm=='static_temporal_collapse' else self.arm
            query=FieldQueryBuilder(self.pose,self.beta,self.K_event,self.last_commit,history=tuple(self.history),config=FieldQueryConfig(field_resolution=2)).build(T,mode=mode)
            coordinates={k:getattr(query,k) for k in ('grid','valid','in_view','full_interp_support')}
        timing['geometry']=(time.perf_counter()-t)*1000
        t=time.perf_counter();g=torch.as_tensor(grid,device=self.device);c=torch.as_tensor(context[None],device=self.device)
        anchor=torch.as_tensor(self.pose.copy()[None],device=self.device);beta=torch.as_tensor(self.beta.copy()[None],device=self.device)
        if coordinates is not None:coordinates={k:v.to(self.device) for k,v in coordinates.items()}
        sync(self.device);timing['H2D']=(time.perf_counter()-t)*1000
        with torch.inference_mode():
            t=time.perf_counter()
            local=P.sample_patches(g,coordinates,2)[None] if coordinates is not None else g.new_zeros((1,8,90,5,7,7))
            ng=(g[None]/self.gs).clamp(-8,8);nl=(local/self.ls).clamp(-8,8);sync(self.device);timing['sampling_and_normalization']=(time.perf_counter()-t)*1000
            t=time.perf_counter();raw=self.model(ng,nl,c);sync(self.device);timing['network']=(time.perf_counter()-t)*1000
            if not bool(torch.isfinite(raw).all()):raise ArithmeticError('nonfinite model output; no emptygate/hold/clip fallback')
            t=time.perf_counter();decoded=self.decoder(anchor,beta,raw);sync(self.device);timing['full_MANO_surface_decode']=(time.perf_counter()-t)*1000
            t=time.perf_counter();state=decoded.commit_pose51()[0].cpu().numpy();vertices=decoded.vertices[0].cpu().numpy();joints=decoded.joints[0].cpu().numpy();raw_array=raw[0].cpu().numpy();sync(self.device)
            timing['surface_D2H_and_chart']=(time.perf_counter()-t)*1000
        if not all(np.isfinite(a).all() for a in (state,vertices,joints,raw_array)):raise ArithmeticError('nonfinite MANOsurface; preserve partialtrace')
        timing['query_service']=(time.perf_counter()-begin)*1000
        for k,v in timing.items():self.cost.add(k,v)
        return SurfaceQuery(self.seq_id,T,self.pose_version,self.memory.version,readonly(state),readonly(vertices),readonly(joints),readonly(raw_array),timing,self._owner)
    def commit(self,result):
        started=time.perf_counter()
        if result._owner is not self._owner or result.seq_id!=self.seq_id or result.pose_version!=self.pose_version or result.observation_version!=self.memory.version or result.cutoff_ms!=self.memory.watermark_ms or result.cutoff_ms<=self.last_commit:raise ValueError('stale/crosslane/duplicate commit')
        self.pose=readonly(result.state51);self.last_commit=result.cutoff_ms;self.pose_version+=1;self.history.append((self.last_commit,self.pose))
        lower=self.last_commit-50;keep=0
        for i,(t,_) in enumerate(self.history):
            if t<=lower:keep=i
        self.history=self.history[keep:]
        if len(self.history)>52:raise AssertionError('native-ms bounded posehistory contract violated')
        ms=(time.perf_counter()-started)*1000;self.cost.add('commit',ms);return ms
    def resident(self):
        return dict(field_numeric_bytes=self.memory.nbytes(),pose_history_array_bytes=sum(q.nbytes for _,q in self.history),pose_history_knots=len(self.history),pose_history_cap=52,
          model_parameter_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters()),model_buffer_bytes=sum(v.numel()*v.element_size() for v in self.model.buffers()),decoder_buffer_bytes=sum(v.numel()*v.element_size() for v in self.decoder.buffers()),
          exclusions='inputmmap/callerpacket, snapshots/grid/geometry/sampling/outputs/Python/RSS/cudareserved separate')
    def release(self):self.memory.release();self.released=True


def load_checkpoint(checkpoint,serving_contract,K_source,beta,*,K_event=None,device='cpu'):
    validate_device(device);contract=json.loads(Path(serving_contract).read_text())
    if contract.get('schema')!='s43_field_identity_serving_contract_v1':raise ValueError('pure serving contract required, never training sourcefreeze')
    for file,identity in contract['runtime_code_and_asset_identities'].items():
        path=Path(file).resolve()
        # A copied repository may keep identical code bytes at a new root.
        if not path.is_relative_to(ROOT):
            if '/tools/' in str(path):path=ROOT/'tools'/str(path).split('/tools/',1)[1]
            elif str(path).endswith('/assets/mano_right.npz'):path=ASSET
        if path!=ASSET.resolve() and (not path.resolve().is_relative_to(ROOT/'tools') or path.suffix not in ('.py','.cpp')):raise ValueError('serving identity may contain only runtimecode/asset, never data/targets')
        if sha(path)!=identity:raise RuntimeError('serving runtime identity changed: '+str(path))
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ckpt.get('schema')!='s43_field_identity_mano_head_v1' or ckpt.get('reader')!='mano51' or ckpt.get('arm') not in COMPLETE_ARMS:raise ValueError('expected mano51 completefield checkpoint; joint63/quota rejected')
    if ckpt.get('source_freeze_sha256')!=contract['training_freeze_sha256']:raise RuntimeError('checkpoint/serving training identity mismatch')
    expected=ckpt['inference_input_contract']
    if expected['global_shape']!=[40,180,240] or expected['local_shape']!=[8,90,5,7,7] or expected['context_width']!=70 or expected['empty_gate'] or expected['action_clip'] or ckpt['output_target_mean_added'] or ckpt['per_axis_target_std']:raise ValueError('checkpoint input/action contract mismatch')
    if ckpt['network_source_sha256']!=sha(ROOT/'tools/s43_revisit/newfield_mano_probe.py') or ckpt['decoder_metadata']['asset_sha256']!=sha(ASSET) or ckpt['decoder_metadata']['source_sha256']!=sha(ROOT/'tools/s43_revisit/field_mano_decoder.py'):raise RuntimeError('decoder/network/asset identity mismatch')
    model=IdentityFeatureReader(ckpt['arm'],'mano51');model.load_state_dict(ckpt['model'],strict=True)
    stats={k:ckpt[k] for k in ('global_scale','local_scale','context_mean','context_std')}
    meta=dict(checkpoint_sha256=sha(checkpoint),serving_contract_sha256=sha(serving_contract),training_freeze_sha256=contract['training_freeze_sha256'],arm=ckpt['arm'],reader='mano51',steps=ckpt['steps'],regime=ckpt['regime'],fold=ckpt['fold'],scope='experimental atomic conditional weights; not sufficiently trained S43 or deploymentaccuracy',runtime_identities=contract['runtime_code_and_asset_identities'])
    return FieldManoOnline(model,stats,K_source,beta,K_event=K_event,device=device,metadata=meta)


def run(events,calibration_path,checkpoint,serving_contract,out,*,time_unit='ms',shape='mean0',hop=50,device='cpu',initial_state=None):
    if hop not in (25,50,100):raise ValueError('hop25/50/100')
    out=Path(out);summary_path=Path(str(out)+'.summary.json');faces_path=Path(str(out)+'.faces.npy')
    if out.exists() or summary_path.exists() or faces_path.exists():raise FileExistsError('fresh output required')
    out.parent.mkdir(parents=True,exist_ok=True);cold=time.perf_counter();Ksource,Kevent,beta,condition=calibration(calibration_path,shape)
    stream=MMapEventStream(events,time_unit,condition['boundary_ms'],condition['stop_ms'])
    try:engine=load_checkpoint(checkpoint,serving_contract,Ksource,beta,K_event=Kevent,device=device)
    except Exception:stream.close();raise
    engine.reset('external_stream',stream.start,initial_state);identities={str(Path(p).resolve()):sha(p) for p in (events,calibration_path,checkpoint,serving_contract,__file__)}
    metadata=dict(engine.metadata);metadata['initial_pose51']=engine.pose.tolist();np.save(faces_path,engine.decoder.faces.cpu().numpy())
    cold_seconds=time.perf_counter()-cold;warm_end=stream.start+100;next_query=stream.start+50;warm=main=0;cost=CostSummary();failed=None
    if device!='cpu':torch.cuda.reset_peak_memory_stats()
    before=time.perf_counter();logging_ms=0.
    try:
      with out.open('w') as log:
        for ms in range(stream.start,stream.stop):
            t=time.perf_counter();packet,span=stream.peek_ms(ms);cost.add('native_source_pack',(time.perf_counter()-t)*1000)
            engine.ingest(packet,ms+1,ms-stream.start,source_range=span,seq_id='external_stream');stream.commit_ms(ms,span[1]);T=ms+1
            if failed is not None or T!=next_query:continue
            try:
                result=engine.query(T);commit_ms=engine.commit(result)
            except Exception as exc:
                failed=dict(type=type(exc).__name__,message=str(exc),at_cutoff_ms=T,policy='No predictionhold/clamp/gate; remainingvalidnativeevents consumed without furtherprediction')
                continue
            if T<=warm_end:
                warm+=1;next_query=T+50 if T<warm_end else T+hop
            else:
                main+=1;next_query=T+hop;t=time.perf_counter()
                log.write(json.dumps(dict(cutoff_ms=T,t_query_ms=T-1,state51=result.state51.tolist(),vertices_m=result.vertices_m.tolist(),joints_m=result.joints_m.tolist(),units='metres',pose_version=engine.pose_version,shape_condition=condition['shape_condition'],timings_ms=dict(result.timings_ms,commit=commit_ms)),allow_nan=False)+'\n');log.flush();logging_ms+=(time.perf_counter()-t)*1000
        if stream.cursor!=len(stream.raw) or engine.memory.stats()['ingested_events']!=len(stream.raw):raise RuntimeError('native input not consumed exactly once')
      process_wall=time.perf_counter()-before;source_count=len(stream.raw);resident=engine.resident();sensor=engine.memory.stats();report=dict(status='FAILED' if failed else 'COMPLETE',failure=failed,checkpoint=metadata,condition=condition,public_initial_used=initial_state is None,hop_ms=hop,own_warmup_ms=100,warmup_forwards=warm,main_surface_outputs=main,native_events=source_count,native_source_rows_consumed=stream.cursor,all_tail_events_consumed=True,native_time_policy='integer1ms;us floored perpacket, raw order validated,no intrabin sortinginformation',cold_load_preflight_hash_seconds=cold_seconds,processing_wall_seconds=process_wall,output_serialization_write_ms=logging_ms,service_wall_excluding_output_logging_ms=process_wall*1000-logging_ms,costs=dict(native=cost.report(),engine=engine.cost.report()),resident=resident,sensor=sensor,RSS_max_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),surface_payload_bytes=9588,state_plus_surface_payload_bytes=9792,source_identities=identities,sources_unchanged={p:sha(p)==h for p,h in identities.items()},GT_data_or_labels_accessed=False,lnes_oldmask_gate_prevMLP_used=False,beta0_does_not_establish_shape_generalization=True,output_logs='O(T)on disk; no retainedtrajectory inengine',timing_scope='Inputmmap accesses/nativepacking,ingest,observationreadout,geometry,H2D,sampling,net,full778v21jFK,D2H/chart,commit included; outputJSON/serialization separate. Prehash maywarmOScache; not coldIO or finalformalbenchmark')
      if device!='cpu':report['torch_cuda_peaks']=dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved(),excludes_driver_context=True)
      report['topology_faces']=dict(path=str(faces_path.resolve()),sha256=sha(faces_path),shape=list(engine.decoder.faces.shape))
      report['previous_pose_condition']='same trained70D anchor/beta/K context; no legacyS43prev_mlp branch'
      report['execution']=dict(device=str(device),torch_version=torch.__version__,CPU_threads=torch.get_num_threads(),
        OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'),no_autocast=True,
        TF32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),TF32_cudnn=bool(torch.backends.cudnn.allow_tf32),
        input_staging_scope='CPUtensorwrapping only,no H2D' if device=='cpu' else 'synchronized H2D',
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
      service_seconds=report['service_wall_excluding_output_logging_ms']/1000
      report['throughput']=dict(events_per_service_second=source_count/service_seconds if service_seconds else None,
        main_outputs_per_service_second=main/service_seconds if service_seconds else None,
        warmup_included=True,hardware_sensor_realtime_claim=False)
      if not all(report['sources_unchanged'].values()):report['status']='FAILED';report['failure']=dict(type='SourceIdentityChanged')
      dump(summary_path,report)
    finally:engine.release();stream.close()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--events',required=True);p.add_argument('--calibration',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--serving-contract',required=True);p.add_argument('--out',required=True);p.add_argument('--time-unit',choices=('ms','us'),default='ms');p.add_argument('--shape',choices=('mean0','provided'),default='mean0');p.add_argument('--hop',type=int,choices=(25,50,100),default=50);p.add_argument('--device',default='cpu');p.add_argument('--initial-state');args=p.parse_args()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    initial=None if args.initial_state is None else np.asarray(json.loads(Path(args.initial_state).read_text()),np.float32)
    try:report=run(args.events,args.calibration,args.checkpoint,args.serving_contract,args.out,time_unit=args.time_unit,shape=args.shape,hop=args.hop,device=args.device,initial_state=initial)
    except Exception as exc:
        dump(args.out+'.failure.json',dict(status='FAILED',type=type(exc).__name__,message=str(exc),partial_trace=args.out,no_hold_or_clip=True,GT_data_or_labels_accessed=False));raise
    print(json.dumps(dict(status=report['status'],native_events=report['native_events'],outputs=report['main_surface_outputs'],summary=args.out+'.summary.json')))
    if report['status']!='COMPLETE':raise SystemExit(1)


if __name__=='__main__':main()
