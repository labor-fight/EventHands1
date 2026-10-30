"""Standalone full-S43 field client: only caller events, calibration and weights.

This is distinct from the small identity-head field_mano_online prototype.
Serving never opens dataset manifests, pose targets, auxiliary shape or GT.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_info
from . import s43_field_replay as R


class TimestampOffsets:
    """searchsorted index over a strided mmap timestamp view, no O(T) copy."""
    def __init__(self,timestamps,factor):self.t=timestamps;self.factor=factor
    def __getitem__(self,t):
        if not np.isscalar(t):raise TypeError('scalar native-ms timestamp required')
        return int(np.searchsorted(self.t,int(t)*self.factor,side='left'))


def calibration(path,shape='mean0'):
    d=R.read(path);source='K_source' in d;event='K_event' in d
    if source==event:raise ValueError('exactly one K_source640x480 or K_event240x180 required')
    K=np.asarray(d['K_source' if source else 'K_event'],np.float64)
    if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1],rtol=0,atol=1e-12):raise ValueError('finite pinhole camera matrix required')
    K=K.copy()
    if event:K[:2]/=.375
    beta=np.zeros(10,np.float32) if shape=='mean0' else np.asarray(d['beta_input'],np.float32)
    if shape not in ('mean0','provided') or beta.shape!=(10,) or not np.isfinite(beta).all():raise ValueError('explicit externalbeta10 required for providedshape')
    return K.astype(np.float32),beta,d


class NpyInputReader:
    """Caller integer Nx4 stream with ms/us input and explicit run boundaries.

    Only one logical input packet is materialized; us timestamps are floored
    in that packet. The declared cap rejects before the runtime sees input,
    never clips events. Fixed core/history state is independent of stream T.
    """
    def __init__(self,events,calibration_path,*,time_unit='ms',shape='mean0',max_logical_events=1048576):
        if time_unit not in ('ms','us'):raise ValueError('time_unit ms or us')
        self.path=Path(events);self.calibration_path=Path(calibration_path)
        self.K_cam,self.beta,self.calibration=calibration(calibration_path,shape)
        self.events=np.load(self.path,mmap_mode='r',allow_pickle=False);self.factor=1 if time_unit=='ms' else 1000
        self.time_unit=time_unit;self.max_logical_events=int(max_logical_events)
        if self.max_logical_events<1:raise ValueError('positive logicalpacketcap required')
        if self.events.ndim!=2 or self.events.shape[1]!=4 or self.events.dtype.kind not in 'iu':raise ValueError('events.npy must contain integer Nx4[x,y,t,p0/1]')
        self.timestamps=self.events[:,2];last=None
        for lo in range(0,len(self.events),65536):
            e=np.asarray(self.events[lo:lo+65536]);t=e[:,2]
            if len(e) and (np.any(e[:,:3]<0) or np.any(e[:,0]>=240) or np.any(e[:,1]>=180) or
                np.any(e[:,2]>np.iinfo(np.int64).max) or np.any((e[:,3]!=0)&(e[:,3]!=1)) or
                np.any(t[1:]<t[:-1]) or (last is not None and t[0]<last)):raise ValueError('invalid sensor coordinates/polarity or unordered raw timestamps')
            if len(t):last=int(t[-1])
        first=int(self.timestamps[0])//self.factor if len(self.events) else 0
        stop=int(self.timestamps[-1])//self.factor+1 if len(self.events) else first
        runs=self.calibration.get('valid_runs_ms',[[self.calibration.get('boundary_ms',first),self.calibration.get('stop_ms',stop)]])
        runs=np.asarray(runs)
        if (runs.ndim!=2 or runs.shape[1]!=2 or runs.dtype.kind not in 'iu' or np.any(runs<0) or
            np.any(runs>np.iinfo(np.int64).max-100) or np.any(runs[:,1]<runs[:,0]) or
            (len(runs)>1 and np.any(runs[1:,0]<runs[:-1,1]))):raise ValueError('ordered nonoverlapping integer valid_runs_ms required')
        self.runs=runs.astype(np.int64);self.offsets=TimestampOffsets(self.timestamps,self.factor)
        count=sum(self.offsets[int(b)]-self.offsets[int(a)] for a,b in self.runs)
        if count!=len(self.events):raise ValueError('declared validruns would discard native input; provide only runsource events')
    def packet(self,start,stop):
        lo,hi=self.offsets[start],self.offsets[stop]
        if hi-lo>self.max_logical_events:raise ValueError('logical inputpacket exceeds declared cap; reject/reset, never crop')
        e=np.asarray(self.events[lo:hi],np.int64)
        if self.factor!=1:e=e.copy();e[:,2]//=self.factor
        return dict(events=e,source_range=(lo,hi))
    def input_paths(self):return [self.path,self.calibration_path]
    def close(self):self.events._mmap.close()


def run(events,calibration_path,out,*,checkpoint=None,arm=None,device='cpu',threads=2,time_unit='ms',shape='mean0',
        hop=50,commit_ms=None,max_logical_events=1048576):
    out=Path(out);summary_path=out.with_suffix('.summary.json')
    if out.exists() or summary_path.exists():raise FileExistsError('fresh streamoutput identity required')
    R.configure(device,threads);reader=NpyInputReader(events,calibration_path,time_unit=time_unit,shape=shape,max_logical_events=max_logical_events)
    if arm is None:
        if checkpoint is None:raise ValueError('provide trainedcheckpoint or explicitly --arm canonical')
        header=torch.load(checkpoint,map_location='cpu');arm=header.get('arm');del header
    model,payload=R.load_model(arm,checkpoint,device)
    base=getattr(model,'base',model)
    files=R.source_files(reader,checkpoint,payload)+[Path(__file__).resolve()]
    files=sorted(set(p.resolve() for p in files));before={str(p):R.sha(p) for p in files}
    out.parent.mkdir(parents=True,exist_ok=True)
    R.dump(out.with_suffix('.sourcefreeze.json'),dict(created_ns=time.time_ns(),schema='s43_full_field_external_client_source_v1',sources=before,
        checkpoint=None if checkpoint is None else str(checkpoint),arm=arm,shape=shape,lag_ms=1,query_ms=hop,commit_ms=commit_ms or hop))
    runner=R.FullStreamRunner(model,payload['runtime_contract'],native_s43=arm in ('canonical','s43'),beta=reader.beta,
        hop=hop,commit_ms=commit_ms,audit_prefix=False)
    phases={};output_serialization_ms=0.
    def add(key,ms):phases.setdefault(runner.phase_category+'/'+key,[]).append(float(ms))
    runner.phase=add;surface_outputs=0
    try:
        with out.open('x') as f:
            def emit(row,warm):
                nonlocal output_serialization_ms,surface_outputs
                start=time.perf_counter();record={k:v for k,v in row.items() if k not in ('pred','vertices_m','joints_m')}
                record.update(warmup=warm,state51=row['pred'].tolist(),vertices_m=row['vertices_m'].tolist(),joints_m=row['joints_m'].tolist())
                f.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n');surface_outputs+=1
                output_serialization_ms+=(time.perf_counter()-start)*1000
            rows,warm,summary=runner.run(reader,'external_events',on_output=emit)
        after={str(p):R.sha(p) for p in files}
        if before!=after:raise RuntimeError('external source/serving identity changed')
        mano=getattr(base,'mano',None);faces=getattr(mano,'f',getattr(mano,'faces',None))
        if faces is not None:
            if torch.is_tensor(faces):faces=faces.detach().cpu().numpy()
            np.save(out.with_suffix('.faces.npy'),np.asarray(faces))
        summary.update(schema='s43_full_field_external_client_v1',arm=arm,checkpoint=str(checkpoint),shape_condition='beta0' if shape=='mean0' else 'explicit_external_shape_condition',
            checkpoint_supervised_outputs=payload.get('supervised_outputs'),checkpoint_variant=payload.get('model_variant',payload.get('variant')),
            checkpoint_schema=payload.get('schema','canonical_reference'),execution_variant=payload.get('execution_variant','canonical_reference'),
            artifact_status=payload.get('serving_status'),
            environment=dict(torch_threads=torch.get_num_threads(),actual_threadpools=threadpool_info(),thread_note='actual loadedpools; no BLAS1 claim from OPENBLAS env'),
            time_unit=time_unit,quantization='rawms preserved' if time_unit=='ms' else 'floor(rawus/1000) perpacket; intrabinorder not a feature',
            K_contract='K_source640 supplied' if 'K_source' in reader.calibration else 'providedKevent240 / .375 tosource once; model * .375 exactlyonce',
            GT_input=False if shape=='mean0' else 'explicitexternalshape condition; origin callerprovided',GT_pose_input=False,
            automatic_auxiliary_shape_read=False,external_shape_used=shape=='provided',
            dataset_reader_instantiated=False,source_files_unchanged=True,full_surface_outputs_including_warmup=surface_outputs,
            output_JSONL=str(out),output_JSONL_sha256=R.sha(out),output_serialization_ms=output_serialization_ms,
            output_storage='O(T) JSONL separate from boundedcore/history; in-memorypose/timing logs O(numberqueries), not sensorhistory',
            max_logical_packet_events=max_logical_events,logical_input_scratch_bound_bytes=max_logical_events*32,
            costs={k:dict(n=len(v),mean_ms=float(np.mean(v)),p50_ms=float(np.percentile(v,50)),p95_ms=float(np.percentile(v,95)),p99_ms=float(np.percentile(v,99))) for k,v in phases.items()},
            timing_scope='nativepack/digest/ingest/readout/H2D/query/fullmeshD2H/commit includes inputmmap; outputserialization/load/pre/posthash outsidecycle; cachedsource engineering, not formal isolated resources',
            sourcefreeze_sha256=R.sha(out.with_suffix('.sourcefreeze.json')),scientific_scope='runnable serving of supplied checkpoint; no GT accuracy assessed, not a claim of adequate training or representation advantage')
        R.dump(summary_path,summary);return summary
    except BaseException as exc:
        R.dump(out.with_suffix('.failure.json'),dict(status='FAILED',error_type=type(exc).__name__,error=str(exc),partial_output=str(out),no_hold_or_clamp=True));raise
    finally:
        runner.close();reader.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--events',required=True);p.add_argument('--calibration',dest='calibration_path',required=True);p.add_argument('--out',required=True)
    p.add_argument('--checkpoint');p.add_argument('--arm',choices=R.MODELS);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--time-unit',choices=('ms','us'),default='ms');p.add_argument('--shape',choices=('mean0','provided'),default='mean0')
    p.add_argument('--hop',type=int,choices=(25,50,100),default=50);p.add_argument('--commit-ms',type=int);p.add_argument('--max-logical-events',type=int,default=1048576)
    run(**vars(p.parse_args()))


if __name__=='__main__':main()
