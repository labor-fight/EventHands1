"""Source-frozen E1/E2 diagnostics for native witness episodes.

No legacy LNES/A/Z feature is read. Existing labels are 63D FK proxies, not
an end-to-end MANO model. Observation features are persisted before targets
are opened. Every sampler shares a reader, initialization and training run.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/s43_revisit_20261028'
PROTOCOL=OUT/'stage2_protocol.json'
PROTOCOL_SHA='a7f227609a738e92c3968a904d852ff88385d4917bfdc710edd0b14a1372b041'
SAMPLERS=('uniform','recency','density','quota','generic_temporal','witness','field2','fullfine')
ARMS=SAMPLERS+('context_only','coverage_only')
ERRORS=('none','root_x10mm','root_yaw0.1')
SEED=3407


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as s:
        for b in iter(lambda:s.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')


def dependencies():
    names=['tools/s43_revisit/witness_probe.py','tools/s43_revisit/witness_core.py',
      'tools/s43_revisit/witness_geometry.py','tools/s43_revisit/witness_geometry_runtime.py',
      'tools/s43_revisit/__init__.py','tools/s43_rel/__init__.py','tools/s43_rel/geometry.py',
      'tools/s43_rel/geometry_cpu.cpp','tools/s43_rel/mechanisms.py',
      'tools/s43_br/data.py','assets/mano_right.npz']
    return {str((ROOT/n).resolve()):sha(ROOT/n) for n in names}


def frozen_inputs():
    """Hash actual observation/score files without decoding any scientific targets."""
    files={PROTOCOL,OUT/'stage2_engineering_R1_prereg.json',OUT/'stage2_pre_task_clock_amendment.json',
      OUT/'witness_probe_engineering/implementation_extension.json',
      OUT/'witness_synthetic_v1/manifest.json',OUT/'witness_synthetic_v1/sourcefreeze.json',
      ROOT/'outputs/s43_rel_20261027/mechanisms/e12/existing_features.npz',
      ROOT/'outputs/s43_rel_20261027/mechanisms/e12/existing_features.manifest.json'}
    manifest=json.loads((OUT/'witness_synthetic_v1/manifest.json').read_text())
    for row in manifest['cases']:
        for name,key in [('observations','observation_sha256'),('targets','target_sha256')]:
            if sha(row[name])!=row[key]:raise RuntimeError('generator source identity changed: '+row[name])
            files.add(Path(row[name]))
        files.add(Path(row['manifest']))
    from tools.s43_br.data import DATA_ROOT,recording_dirs
    files.add(DATA_ROOT/'splits_semkine.json');dirs=recording_dirs(DATA_ROOT)
    old=json.loads((ROOT/'outputs/s43_rel_20261027/mechanisms/e12/existing_features.manifest.json').read_text())
    for row in old['samples']:
        base=DATA_ROOT/dirs[row['recording']]/row['recording']
        files.update((Path(str(base)+'_events.npy'),Path(str(base)+'_offsets.npy')))
    return {str(p.resolve()):sha(p) for p in sorted(files)}


def require_gate(freeze):
    if sha(PROTOCOL)!=PROTOCOL_SHA:raise RuntimeError('Stage2 protocol identity changed')
    p=Path(freeze);x=json.loads(p.read_text())
    if x.get('schema')!='s43_witness_actual_sourcefreeze_v1' or x.get('protocol_sha256')!=PROTOCOL_SHA:
        raise RuntimeError('actual Stage2 source freeze required before task extraction/fitting')
    if not x.get('engineering_passed'):raise RuntimeError('engineering gate not passed')
    current=dependencies()
    if x.get('dependencies')!=current:raise RuntimeError('actual implementation no longer matches source freeze')
    for name,identity in {**x.get('sources',{}),**x.get('engineering',{})}.items():
        if sha(name)!=identity:raise RuntimeError('frozen input identity changed: '+name)
    return x


def sourcefreeze(path,engineering,commands):
    """Called only after actual engineering receipts exist, never authorizes science."""
    if sha(PROTOCOL)!=PROTOCOL_SHA:raise RuntimeError('wrong protocol')
    if Path(path).exists():raise FileExistsError(path)
    checks={}
    for name in engineering:
        x=json.loads(Path(name).read_text())
        if not (x.get('status')=='PASS' or x.get('passed') is True or x.get('unit_tests_passed') is True):raise RuntimeError('nonpassing engineering receipt: '+str(name))
        checks[str(Path(name).resolve())]=sha(name)
    if not checks:raise RuntimeError('actual engineering receipts required')
    sources=frozen_inputs()
    dump(path,dict(schema='s43_witness_actual_sourcefreeze_v1',created_utc=datetime.now(timezone.utc).isoformat(),
      protocol_sha256=PROTOCOL_SHA,dependencies=dependencies(),sources=sources,engineering_passed=True,
      engineering_passed_scope='Functional/source/shape/clock checks permit E1/E2 finite diagnostic; no resource/real-time gate or novelty pass implied',
      engineering=checks,commands=commands,device_contract='CNN all sampler arms share one FP32 cuda:0 backend with CUDA_VISIBLE_DEVICES=4; TF32 disabled, no autocast, CPU8 prep/RBF; CPU reproducer supported',
      scores_read=False))


def implementation_manifest(path):
    """Pre-score implementation detail, independent from formal result files."""
    dump(path,dict(schema='s43_witness_probe_implementation_v1',created_utc=datetime.now(timezone.utc).isoformat(),
      protocol_sha256=PROTOCOL_SHA,probe_source_sha256=sha(__file__),arms=ARMS,selection_errors=ERRORS,
      input='core grid40(log_counts=True), float32; counts already log1p, coverage fraction; no duplicate log',
      same_byte_field='Same<=6MiB budget domain, not equal resident allocation:50x90x120x2 uint32 ring4,320,400B; F nearest-upsampled2px count density(count/4) before log; coverage at2px, not nativefine truth; Q complete4px count; unused budget is a control limitation',
      fullfine='50x180x240x2 uint32 information positive control; larger than main memory budget',
      cnn=dict(stem='Conv40->8 k3,s1,p1;SiLU;AvgPool4;Conv8->16 k3,p1;SiLU;AdaptivePool6x8',
        readout='flatten768 + same70D context ->128 SiLU ->128 SiLU ->10class or63joint-delta',
        steps=2000,batch=32,lr=.001,optimizer='Adam',seed=SEED,CPU_threads=8,
        shared='one model mixed over all10 arms and3 selection errors; corrected common query context; frozen for E2',
        precision='float32, no autocast; actual backend/TF32 flags saved',
        cache='training may preload unique observation grids on chosen device; offline training dataset cache is separate from online observation state'),
      rbf=dict(input='fixed30x30 spatial average to6x8x40=1920 + same70D context',ridge=.01,
        bandwidth='positive FIT pair squared-distance median; exact mixed FIT kernel',
        normalization='FIT uncentered RMSfloor1/clip8; context FIT mean/std and dimension scale',
        caveat='fixed pooling can remove distinctions; not a raw-information upper-bound test; N^2 offline memory and model library reported'),
      existing='288 old sample addresses; conditional aux-beta-derived shared anchor; only context/prior/target joints reused, not old feature arrays',
      synthetic='80 physical sequences,4 orientation/root contexts leave-one-context; clean/background kept paired; input builder never receives finger/sign',
      primary_query_ms=[50],optional_secondary_query_ms=[10,20,30,40,75,100],
      target_times=dict(synthetic='cutoff endpoint T, event timestamps denote intervals[t,t+1)',existing='old endpoint T-1; native offsets[x,y,p], no sub-ms reconstruction'),
      target_role='synthetic10way and active-finger FK; existing63D joint-delta proxy; not full51D MANO system',
      no_learned_task_saliency=True,novelty_limit='DEVO-style learnedtask sampler absent; no novelty certification',
      no_task_scores_read=True))


def inject_anchor(anchor,error):
    from .witness_geometry import right_retract
    q=np.asarray(anchor,np.float64).copy()
    if error=='root_x10mm':q[0]+=.01
    elif error=='root_yaw0.1':q=right_retract(q,0,[0.,1.,0.],.1)
    elif error!='none':raise ValueError(error)
    return q


def condition70(anchor,beta,K):
    return np.concatenate((np.asarray(anchor).reshape(51),np.asarray(beta).reshape(10),np.asarray(K).reshape(9))).astype(np.float32)


def native_events(events,start,stop):
    a=np.asarray(events)
    if a.ndim!=2 or a.shape[1]!=4 or not np.issubdtype(a.dtype,np.integer):raise ValueError('native integer Nx4 required')
    a=np.asarray(a,np.int64)
    if len(a) and (np.any(np.diff(a[:,2])<0) or np.any(a[:,2]<start) or np.any(a[:,2]>=stop)
      or np.any(a[:,0]<0) or np.any(a[:,0]>=240) or np.any(a[:,1]<0) or np.any(a[:,1]>=180)
      or np.any((a[:,3]!=0)&(a[:,3]!=1))):raise ValueError('native event range/order/polarity')
    return a


class DenseFieldMemory:
    """Bounded complete count baseline, native1px or spatial2px resolution."""
    def __init__(self,resolution,start_ms):
        if resolution not in (1,2):raise ValueError('field resolution')
        self.r=resolution;self.start=self.watermark=int(start_ms)
        self.count=np.zeros((50,180//resolution,240//resolution,2),np.uint32)
        self.times=np.full(50,-1,np.int64);self.events=0
    def ingest(self,packet,ms):
        if ms!=self.watermark:raise ValueError('field requires consecutive complete native-ms packets')
        ev=native_events(packet,ms,ms+1)
        row=np.zeros(self.count.shape[1:],np.uint64)
        if len(ev):np.add.at(row,(ev[:,1]//self.r,ev[:,0]//self.r,ev[:,3]),1)
        if np.any(row>np.iinfo(np.uint32).max):raise OverflowError('field uint32 overflow, no advance')
        self.count[ms%50]=row.astype(np.uint32);self.times[ms%50]=ms
        self.events+=len(ev);self.watermark=ms+1
    def grid(self,T):
        if T!=self.watermark:raise ValueError('field current prefix only')
        fine=np.zeros((8,180//self.r,240//self.r,2),np.uint64);exposure=np.zeros(8,np.int64)
        for t in range(T-50,T):
            if t>=self.start and self.times[t%50]==t:
                b=(t-(T-50))*8//50;fine[b]+=self.count[t%50];exposure[b]+=1
        widths=np.bincount(np.arange(50)*8//50,minlength=8)
        c=4//self.r;coarse=fine.reshape(8,45,c,60,c,2).sum((2,4))
        density=fine.astype(np.float32)/(self.r*self.r)
        density=np.repeat(np.repeat(density,self.r,1),self.r,2)
        coverage=np.broadcast_to((exposure/widths)[:,None,None],(8,180,240)).copy()
        coarse=np.repeat(np.repeat(coarse.astype(np.float32),4,1),4,2)
        x=np.concatenate((np.log1p(density).transpose(0,3,1,2).reshape(16,180,240),coverage,
          np.log1p(coarse).transpose(0,3,1,2).reshape(16,180,240)),0).astype(np.float32)
        return x,dict(resolution_px=self.r,resident_bytes=self.count.nbytes+self.times.nbytes,
          source_count=self.events,complete_at_declared_resolution=True,native_fine_observed=self.r==1)


def packet_iter(events,start,stop,source_start=0):
    events=native_events(events,start,stop);cursor=0
    for ms in range(int(start),int(stop)):
        hi=int(np.searchsorted(events[:,2],ms+1,side='left'))
        yield ms,events[cursor:hi],(int(source_start+cursor),int(source_start+hi))
        cursor=hi
    if cursor!=len(events):raise ValueError('native event coverage incomplete')


def build_case_features(events,anchor,beta,K_source,start,stop,out_dir,case_id,queries,source_start=0):
    """No targets/actions accepted by signature. Corrected query context is original anchor."""
    from .witness_core import EpisodeConfig,EpisodeMemory
    from .witness_geometry_runtime import CacheBoundMotionPolicy as MotionPolicy
    K=np.asarray(K_source,np.float64).copy();K[:2]*=.375
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    common=condition70(anchor,beta,K_source);rows=[];normal_files={}
    for error in ERRORS:
      selection_anchor=inject_anchor(anchor,error)
      for arm in SAMPLERS:
        # These samplers do not depend on selection anchor. Reuse the same actual content file.
        independent=arm in ('uniform','recency','density','generic_temporal','field2','fullfine')
        if independent and error!='none':
            for T in queries:
                row=dict(normal_files[(arm,T)]);row.update(selection_error=error)
                rows.append(row)
            continue
        started=time.perf_counter()
        if arm in ('field2','fullfine'):
            mem=DenseFieldMemory(2 if arm=='field2' else 1,start);policy=None
        else:
            policy=MotionPolicy(selection_anchor,beta,K,start,mode=arm) if arm in ('witness','quota','generic_temporal') else None
            mem=EpisodeMemory(EpisodeConfig(policy=arm),policy=policy);mem.reset(case_id,boundary_ms=start,policy=policy)
        ordinal=0;local=[]
        for ms,packet,source_range in packet_iter(events,start,stop,source_start):
            if isinstance(mem,DenseFieldMemory):mem.ingest(packet,ms)
            else:mem.ingest(packet,watermark_ms=ms+1,packet_ordinal=ordinal,source_range=source_range)
            ordinal+=1;T=ms+1
            if T not in queries:continue
            raw_saved={}
            if isinstance(mem,DenseFieldMemory):grid,receipt=mem.grid(T)
            else:
                snap=mem.readout(T);before=snap.content_sha256();grid=snap.to_grid40ch()
                # Query context restoration must not reselect or commit a memory.
                after=mem.readout(T).content_sha256()
                if before!=after:raise RuntimeError('readout mutated observation content')
                raw_saved=dict(raw_blocks=snap.blocks,raw_tile_xy=snap.tile_xy,raw_epoch_ms=snap.epoch_start_ms,
                  raw_observed_start_ms=snap.observed_start_ms,raw_observed_end_ms=snap.observed_end_ms,
                  raw_source_events=snap.source_events,raw_owner=snap.owner,raw_priority=snap.priority,
                  raw_sign=snap.sign,raw_selection_kind=snap.selection_kind,raw_partial=snap.partial_counts)
                receipt=dict(snapshot_metadata=snap.metadata,content_sha256=before,dictionary_sha256=snap.dictionary_sha256(),
                  resident_bytes=mem.nbytes(),costs=mem.stats(),policy=mem.policy.receipt() if mem.policy else None)
            path=out_dir/f'{arm}_{error}_t{T}.npz'
            np.savez_compressed(path,grid=np.asarray(grid,np.float32),**raw_saved)
            row=dict(case_id=case_id,arm=arm,selection_error=error,cutoff_ms=T,feature_file=str(path.resolve()),
              feature_mode='all',feature_sha256=sha(path),context=common.tolist(),receipt=receipt,
              preprocessing_seconds=time.perf_counter()-started,target_not_read=True)
            rows.append(row);local.append(row)
            if error=='none':normal_files[(arm,T)]=row
        if isinstance(mem,DenseFieldMemory):final_count=mem.events
        else:final_count=mem.readout(stop).metadata['source_ingested']
        if int(final_count)!=len(events):raise RuntimeError('source consumption count mismatch')
        for r in local:r.update(full_input_events=len(events),consumed_to_ms=stop)
      witness=[r for r in rows if r['arm']=='witness' and r['selection_error']==error]
      for item in witness:
        for name,mode in [('context_only','zero'),('coverage_only','coverage')]:
            row=dict(item);row.update(arm=name,feature_mode=mode);rows.append(row)
    return rows


def synthetic_inputs():
    manifest_path=OUT/'witness_synthetic_v1/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    for row in manifest['cases']:
        path=Path(row['observations'])
        if sha(path)!=row['observation_sha256'] or sha(row['targets'])!=row['target_sha256']:
            raise RuntimeError('synthetic generator identity changed: '+str(row['id']))
        with np.load(path,allow_pickle=False) as z:
            # Actions/target file are not opened in this phase.
            yield dict(case_id=str(row['id']),domain='synthetic',observation_file=str(path),
              events=np.asarray(z['events'],np.int64),anchor=np.asarray(z['public_initial51'],np.float64),
              beta=np.asarray(z['beta'],np.float64),K=np.asarray(z['K_source'],np.float64),
              start=int(z['boundary_ms']),stop=int(z['stop_ms']),target_file=str(row['targets']),
              source_manifest=str(row['manifest']))


def existing_inputs():
    cache=ROOT/'outputs/s43_rel_20261027/mechanisms/e12/existing_features.npz'
    from tools.s43_br.data import DATA_ROOT,recording_dirs
    # Only allowed, pre-existing input/partition arrays are materialized here.
    with np.load(cache,allow_pickle=False) as z:
        context=z['context'].copy();ids=z['sample_id'].astype(str);rec=z['recording'].astype(str)
        parts=z['partition'].astype(str)
    dirs=recording_dirs(DATA_ROOT)
    original=json.loads(cache.with_suffix('.manifest.json').read_text())
    expected={r['sample_id']:r for r in original['samples']}
    if len(expected)!=288 or len(ids)!=288:raise RuntimeError('old288 sample-address contract changed')
    for index,(sample,recording,part,c) in enumerate(zip(ids,rec,parts,context)):
        end=int(sample.rsplit('/',1)[1]);start=end-49;stop=end+1
        base=DATA_ROOT/dirs[recording]/recording
        native=np.load(str(base)+'_events.npy',mmap_mode='r');offsets=np.load(str(base)+'_offsets.npy',mmap_mode='r')
        lo,hi=int(offsets[start]),int(offsets[stop]);raw=np.asarray(native[lo:hi],np.int64)
        times=np.repeat(np.arange(start,stop,dtype=np.int64),np.diff(np.asarray(offsets[start:stop+1],np.int64)))
        events=np.column_stack((raw[:,:2],times,raw[:,2])).astype(np.int64)
        relative=events.copy();relative[:,2]-=start
        identity=hashlib.sha256(relative.tobytes()).hexdigest();old=expected[str(sample)]
        if identity!=old['source_sha256'] or len(events)!=old['source_events'] or str(part)!=old['partition']:
            raise RuntimeError('old native window identity/count/partition mismatch: '+str(sample))
        yield dict(case_id=f'existing_{index:03d}',domain='existing',sample_id=sample,recording=recording,
          partition=part,old_index=index,events=events,anchor=c[:51].astype(np.float64),beta=c[51:61].astype(np.float64),
          K=(c[61:70].reshape(3,3)*640.).astype(np.float64),start=start,stop=stop,source_row_range=[lo,hi],
          old_source_parity=dict(relative_ms_int64_sha256=identity,source_events=len(events),partition=str(part),passed=True),
          source_files=[str(base)+'_events.npy',str(base)+'_offsets.npy'])


def verify_existing_sources(out):
    """All288 windows pass label-free old native identity before any extraction."""
    rows=[]
    for case in existing_inputs():rows.append(dict(sample_id=case['sample_id'],**case['old_source_parity']))
    if len(rows)!=288 or not all(r['passed'] for r in rows):raise RuntimeError('existing288 identity gate failed')
    result=dict(schema='s43_witness_existing_native_identity_v1',status='PASS',samples=len(rows),rows=rows,
      scientific_targets_opened=False,task_scores_read=False,created_utc=datetime.now(timezone.utc).isoformat())
    dump(out,result);return result


def extract(domain,dest,freeze,queries=(50,)):
    require_gate(freeze)
    dest=Path(dest)
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True);implementation_manifest(dest/'implementation_manifest.json')
    if domain=='existing':verify_existing_sources(dest/'existing_native_identity.json')
    before=dependencies();started=time.perf_counter();rows=[];cases=[];identities={}
    iterator=synthetic_inputs() if domain=='synthetic' else existing_inputs()
    for case in iterator:
        caseid=case['case_id'];start=case['start'];stop=case['stop']
        if domain=='existing':actual_queries=(stop,)
        else:
            actual_queries=tuple(start+int(t) for t in queries)
            if any(t not in (10,20,30,40,50,75,100) for t in queries):raise ValueError('query outside preregistered addresses')
        result=build_case_features(case['events'],case['anchor'],case['beta'],case['K'],start,stop,
          dest/'observations'/caseid,caseid,actual_queries,source_start=case.get('source_row_range',[0])[0])
        # Seal every observation before attaching labels in a separate phase below.
        for row in result:row['domain']=domain
        rows.extend(result)
        summary={k:v for k,v in case.items() if k!='events'}
        summary['native_events']=len(case['events']);summary['native_xytp_sha256']=hashlib.sha256(case['events'].tobytes()).hexdigest()
        cases.append(summary)
        for file in case.get('source_files',[case.get('observation_file')]):
            if file and file not in identities:identities[file]=sha(file)
        print(json.dumps(dict(case=caseid,observations_persisted=True,rows=len(result),native_events=len(case['events'])),ensure_ascii=False),flush=True)
    dump(dest/'observations_complete.json',dict(rows=len(rows),cases=len(cases),source_files=identities,
      created_utc=datetime.now(timezone.utc).isoformat(),targets_opened=False,dependencies=before))
    # No target file/target array has been accessed by any selection/encoding call.
    labels={};case_map={c['case_id']:c for c in cases}
    if domain=='existing':
        cache=ROOT/'outputs/s43_rel_20261027/mechanisms/e12/existing_features.npz'
        with np.load(cache,allow_pickle=False) as z:prior=z['prior_joints'].copy();truth=z['target_joints'].copy()
        for c in cases:
            i=c['old_index'];labels[c['case_id']]=dict(prior=prior[i].tolist(),target=truth[i].tolist(),
              partition=c['partition'],recording=c['recording'],fold=-1,target_time_ms=c['stop']-1)
    else:
        for c in cases:
            # Conditions are first parsed now, after the full observation collection is sealed.
            manifest=json.loads(Path(c['source_manifest']).read_text());condition=manifest['condition']
            fold=int(condition['root_y_rad']!=0)*2+int(condition['root_dx_m']!=0)
            with np.load(c['target_file'],allow_pickle=False) as z:
                labels[c['case_id']]=dict(finger=int(z['finger']),sign=int(z['sign']),target51=z['target51'].tolist(),
                  target_joints=z['joints_m'].tolist(),fold=fold,moving_background=bool(condition['moving_background']),
                  public_anchor=c['anchor'].tolist(),beta=c['beta'].tolist())
    for row in rows:
        label=labels[row['case_id']]
        row['fold']=label['fold']
        if domain=='existing':row.update(partition=label['partition'],recording=label['recording'])
    dump(dest/'rows.json',rows);dump(dest/'labels.json',labels)
    if dependencies()!=before:raise RuntimeError('source changed during extraction')
    changed={name:sha(name)==identity for name,identity in identities.items()}
    if not all(changed.values()):raise RuntimeError('native source changed during extraction')
    dump(dest/'complete.json',dict(schema='s43_witness_extraction_v1',domain=domain,cases=len(cases),rows=len(rows),
      observation_then_target=True,source_freeze_sha256=sha(freeze),dependencies=before,input_files=identities,
      source_files_unchanged=changed,seconds=time.perf_counter()-started,queries=list(queries),
      condition='legacy conditional beta-assisted anchor diagnostic' if domain=='existing' else 'known public beta0 synthetic boundary',
      learned_task_sampler_present=False))


class GridStore:
    """Bounded CPU decompression cache; model preloading is separately measured."""
    def __init__(self,capacity=32):self.cache=OrderedDict();self.capacity=capacity
    def get(self,row):
        p=row['feature_file']
        if p not in self.cache:
            with np.load(p,allow_pickle=False) as z:self.cache[p]=np.asarray(z['grid'],np.float32)
            if len(self.cache)>self.capacity:self.cache.popitem(last=False)
        else:self.cache.move_to_end(p)
        x=self.cache[p]
        if row['feature_mode']=='zero':return np.zeros_like(x)
        if row['feature_mode']=='coverage':
            out=np.zeros_like(x);out[16:24]=x[16:24];return out
        return x


def pooled(x):
    return x.reshape(40,6,30,8,30).mean((2,4)).reshape(-1)


class RBFReader:
    def fit(self,x,c,y):
        self.scale=np.maximum(np.sqrt(np.mean(x.astype(np.float64)**2,axis=0)),1.)
        self.cm=c.mean(0);self.cs=c.std(0);n=int(np.sum(self.cs>=1e-6));self.cs[self.cs<1e-6]=1.
        self.context_scale=np.sqrt(max(1,n));self.library=self.transform(x,c)
        d=self.distance(self.library,self.library);v=d[np.triu_indices(len(d),1)];v=v[v>1e-12]
        self.bandwidth=max(float(np.median(v)) if len(v) else 1.,1e-6)
        K=np.exp(-d/self.bandwidth);K.flat[::len(K)+1]+=.01
        self.coef=np.linalg.solve(K,y.astype(np.float64))
        self.training_kernel_bytes=int(K.nbytes+d.nbytes)
        return self
    def transform(self,x,c):
        xx=np.clip(x/self.scale,-8,8).astype(np.float64)
        for a,b in [(0,16*48),(16*48,24*48),(24*48,40*48)]:xx[:,a:b]*=np.sqrt((1/3)/(b-a))
        cc=np.clip((c-self.cm)/self.cs,-8,8)/self.context_scale
        return np.concatenate((xx,cc),1)
    @staticmethod
    def distance(a,b):return np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)[None]-2*a@b.T,0)
    def predict(self,x,c):
        return np.exp(-self.distance(self.transform(x,c),self.library)/self.bandwidth)@self.coef
    def save(self,path):np.savez_compressed(path,**self.__dict__)
    def accounting(self):
        return dict(parameters=int(self.coef.size),library_rows=len(self.library),
          model_array_bytes=sum(v.nbytes for v in self.__dict__.values() if isinstance(v,np.ndarray)),
          offline_training_kernel_bytes=self.training_kernel_bytes)


def make_cnn(output):
    import torch
    class Reader(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=torch.nn.Sequential(torch.nn.Conv2d(40,8,3,padding=1),torch.nn.SiLU(),torch.nn.AvgPool2d(4),
              torch.nn.Conv2d(8,16,3,padding=1),torch.nn.SiLU(),torch.nn.AdaptiveAvgPool2d((6,8)))
            self.head=torch.nn.Sequential(torch.nn.Linear(768+70,128),torch.nn.SiLU(),torch.nn.Linear(128,128),
              torch.nn.SiLU(),torch.nn.Linear(128,output))
        def forward(self,x,c):return self.head(torch.cat((self.encoder(x).flatten(1),c),1))
    return Reader()


def validate_device(device):
    if str(device)=='cpu':return
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    physical=[x.strip() for x in visible.split(',') if x.strip()]
    if not device.startswith('cuda') or not physical or any(x not in ('4','5','6','7') for x in physical):
        raise RuntimeError('CNN CUDA requires explicit CUDA_VISIBLE_DEVICES physical4–7 only')


def target_arrays(rows,labels,domain):
    if domain=='existing':
        y=np.asarray([(np.asarray(labels[r['case_id']]['target'])-np.asarray(labels[r['case_id']]['prior'])).reshape(63)*1000 for r in rows],np.float32)
        return y
    classes=np.asarray([labels[r['case_id']]['finger']*2+int(labels[r['case_id']]['sign']<0) for r in rows],np.int64)
    return np.eye(10,dtype=np.float32)[classes]


def fit_cnn(rows,fit_mask,y,store,device,steps=2000,batch=32):
    import torch
    validate_device(device);torch.set_num_threads(8);torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if device!='cpu':torch.cuda.manual_seed_all(SEED)
    model=make_cnn(y.shape[1]).to(device);rng=np.random.default_rng(SEED)
    # Same pixel encoder is shared across every arm/error version. Preload only unique contents.
    def key(r):return ('shared_zero','zero') if r['feature_mode']=='zero' else (r['feature_file'],r['feature_mode'])
    keys=list(dict.fromkeys(key(r) for r in rows));where={k:i for i,k in enumerate(keys)}
    first={}
    for r in rows:first.setdefault(key(r),r)
    bank=torch.empty((len(keys),40,180,240),dtype=torch.float32,device=device)
    for i,k in enumerate(keys):bank[i]=torch.as_tensor(store.get(first[k]),device=device)
    ix=np.asarray([where[key(r)] for r in rows],np.int64)
    train=np.flatnonzero(fit_mask);weights=np.bincount(ix[train],minlength=len(keys)).astype(np.float64)
    channel=np.zeros(40,np.float64)
    for i,w in enumerate(weights):
        if w:channel+=w*bank[i].double().square().mean((1,2)).cpu().numpy()
    scale=np.maximum(np.sqrt(channel/len(train)),1.).astype(np.float32)
    bank.div_(torch.as_tensor(scale[None,:,None,None],device=device));bank.clamp_(-8,8)
    context=np.asarray([r['context'] for r in rows],np.float32);cm=context[train].mean(0);cs=context[train].std(0);cs[cs<1e-6]=1.
    context=np.clip((context-cm)/cs,-8,8).astype(np.float32)
    ct=torch.as_tensor(context,device=device);yt=torch.as_tensor(y,device=device);iit=torch.as_tensor(ix,device=device)
    ym=y[train].mean(0);ys=np.maximum(y[train].std(0),1.)
    classification=y.shape[1]==10
    opt=torch.optim.Adam(model.parameters(),lr=.001);curve=[];started=time.perf_counter()
    for step in range(steps):
        ids=rng.choice(train,batch,replace=True);ti=torch.as_tensor(ids,device=device)
        opt.zero_grad(set_to_none=True);pred=model(bank[iit[ti]],ct[ti])
        if classification:loss=torch.nn.functional.cross_entropy(pred,yt[ti].argmax(1))
        else:loss=torch.nn.functional.mse_loss(pred,(yt[ti]-torch.as_tensor(ym,device=device))/torch.as_tensor(ys,device=device))
        loss.backward();opt.step()
        if step%100==0 or step==steps-1:
            curve.append([step+1,float(loss.detach().cpu())])
            print(json.dumps(dict(cnn_step=step+1,FIT_loss=curve[-1][1],SELECT_not_scored=True)),flush=True)
    if device!='cpu':torch.cuda.synchronize()
    elapsed=time.perf_counter()-started;predictions=[];model.eval()
    with torch.inference_mode():
        for a in range(0,len(rows),batch):
            pred=model(bank[iit[a:a+batch]],ct[a:a+batch])
            if not classification:pred=pred*torch.as_tensor(ys,device=device)+torch.as_tensor(ym,device=device)
            predictions.append(pred.float().cpu().numpy())
    result=np.concatenate(predictions)
    state=dict(model=model.state_dict(),channel_scale=scale,context_mean=cm,context_std=cs,target_mean=ym,target_std=ys,
      output=y.shape[1],seed=SEED,steps=steps,precision='float32',device=device)
    accounting=dict(parameters=sum(p.numel() for p in model.parameters()),steps=steps,batch=batch,seconds=elapsed,
      unique_training_dataset_grid_cache_bytes=bank.numel()*bank.element_size(),curve=curve,
      device=device,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),torch_version=torch.__version__,
      allow_tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),allow_tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
      no_autocast=True,cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
      cudnn_deterministic=bool(torch.backends.cudnn.deterministic))
    del bank,model
    if device!='cpu':torch.cuda.empty_cache()
    return result,state,accounting


def metrics(pred,rows,labels,domain):
    if domain=='existing':
        prior=np.asarray([labels[r['case_id']]['prior'] for r in rows]);target=np.asarray([labels[r['case_id']]['target'] for r in rows])
        p=prior+pred.reshape(-1,21,3)/1000.
        absolute=np.linalg.norm(p-target,axis=-1).mean(-1)*1000
        ra=np.linalg.norm((p-p[:,:1])-(target-target[:,:1]),axis=-1).mean(-1)*1000
        root=np.linalg.norm(p[:,0]-target[:,0],axis=-1)*1000
        result=dict(n=len(rows),abs_mm=float(absolute.mean()),J0_mm=float(ra.mean()),root_mm=float(root.mean()))
        rec=np.asarray([r['recording'] for r in rows]);result['per_recording']={str(k):dict(n=int((rec==k).sum()),
          abs_mm=float(absolute[rec==k].mean()),J0_mm=float(ra[rec==k].mean())) for k in np.unique(rec)}
        return result
    from .witness_geometry import public_flexion_axes,motion_endpoints,right_retract,fk_joints,FINGER_JOINTS
    classes=pred.argmax(1);correct=[];finger_errors=[];ra=[];absolute=[];root=[];cache={}
    for cl,row in zip(classes,rows):
        label=labels[row['case_id']];true_class=label['finger']*2+int(label['sign']<0);correct.append(cl==true_class)
        public=np.asarray(label['public_anchor']);beta=np.asarray(label['beta']);T=int(row['cutoff_ms'])
        key=(public.tobytes(),beta.tobytes())
        if key not in cache:cache[key]=motion_endpoints(public,beta)[1]
        finger=int(cl)//2;direction=1 if int(cl)%2==0 else -1;axis=public_flexion_axes()[finger]
        angle=cache[key]['angles_rad'][finger][0 if direction==1 else 1]
        # Fixed legal candidates are generated from public boundary, never current target.
        state=right_retract(public,axis['mcp_mano'],np.eye(3)[axis['axis']]*axis['sign']*direction,min(T/50.,1.)*angle)
        p=fk_joints(state,beta);target=np.asarray(label['target_joints'][T]);ids=FINGER_JOINTS[label['finger']]
        pr=p-p[:1];tr=target-target[:1]
        finger_errors.append(np.linalg.norm(pr[ids]-tr[ids],axis=-1).mean()*1000)
        ra.append(np.linalg.norm(pr-tr,axis=-1).mean()*1000);absolute.append(np.linalg.norm(p-target,axis=-1).mean()*1000)
        root.append(np.linalg.norm(p[0]-target[0])*1000)
    return dict(n=len(rows),mode_accuracy=float(np.mean(correct)),chance=.1,active_finger_J0_mm=float(np.mean(finger_errors)),
      J0_mm=float(np.mean(ra)),abs_mm=float(np.mean(absolute)),root_mm=float(np.mean(root)),
      caveat='finite10candidate information diagnostic; candidate decoder does not estimate unknown root displacement or general51D MANO')


def grouped_metrics(pred,rows,labels,domain,fit_mask,test_mask):
    result={}
    for split,mask in [('FIT',fit_mask),('SELECT',test_mask)]:
      for arm in ARMS:
       for error in ERRORS:
        for T in sorted({r['cutoff_ms'] for r in rows}):
            chosen=np.asarray([bool(ok) and r['arm']==arm and r['selection_error']==error and r['cutoff_ms']==T
              for ok,r in zip(mask,rows)])
            if not chosen.any():continue
            # Existing cutoffs are recording-global and differ across rows. Aggregate separately below.
            if domain=='existing':continue
            rr=[r for r,keep in zip(rows,chosen) if keep]
            result[f'{split}/{arm}/{error}/t{T}']=metrics(pred[chosen],rr,labels,domain)
            bg=np.asarray([labels[r['case_id']]['moving_background'] for r in rows])
            result[f'{split}/{arm}/{error}/t{T}']['by_background']={}
            for present in (False,True):
                selected=chosen&(bg==present)
                if selected.any():result[f'{split}/{arm}/{error}/t{T}']['by_background'][str(present)]=metrics(pred[selected],[r for r,k in zip(rows,selected) if k],labels,domain)
        if domain=='existing':
            chosen=np.asarray([bool(ok) and r['arm']==arm and r['selection_error']==error for ok,r in zip(mask,rows)])
            if chosen.any():result[f'{split}/{arm}/{error}']=metrics(pred[chosen],[r for r,k in zip(rows,chosen) if k],labels,domain)
    return result


def development_gate(all_results):
    """Registered error-wise comparison; attribution/feasibility require explicit review."""
    tests={};simple=('uniform','recency','density','quota','generic_temporal','field2')
    for error in ERRORS:
        heads={}
        for kind,report in all_results.items():
            rows=report['metrics'];b=rows[f'SELECT/witness/{error}']
            baseline=min(simple,key=lambda arm:rows[f'SELECT/{arm}/{error}']['J0_mm']);base=rows[f'SELECT/{baseline}/{error}']
            margin=max(.25,.02*base['J0_mm']);gain=base['J0_mm']-b['J0_mm']
            deltas={rec:base['per_recording'][rec]['J0_mm']-v['J0_mm'] for rec,v in b['per_recording'].items()}
            heads[kind]=dict(strongest_simple=baseline,J0_gain_mm=gain,required_gain_mm=margin,
              per_recording_gain_mm=deltas,task_comparison_passed=bool(gain>=margin and sum(v>0 for v in deltas.values())>=3 and min(deltas.values())>=-.5),
              B_minus_coverage_J0_mm=b['J0_mm']-rows[f'SELECT/coverage_only/{error}']['J0_mm'],
              B_minus_context_J0_mm=b['J0_mm']-rows[f'SELECT/context_only/{error}']['J0_mm'])
        tests[error]=dict(readers=heads,both_reader_task_comparisons_passed=all(h['task_comparison_passed'] for h in heads.values()))
    return dict(comparisons=tests,positive_control_and_coverage_attribution_review_required=True,
      no_automatic_training_authorization=True,no_novelty_certification=True,
      learned_task_sampler_absent=True,rule='ExistingSELECT J0 conservative development rule; positive-control feasibility, synthetic and coverage/context evidence separately reviewed')


def fit(domain,features,dest,freeze,device='cpu'):
    gate=require_gate(freeze);validate_device(device)
    features=Path(features);dest=Path(dest)
    if dest.exists():raise FileExistsError(dest)
    complete=json.loads((features/'complete.json').read_text())
    if complete['source_freeze_sha256']!=sha(freeze):raise RuntimeError('feature/sourcefreeze identity mismatch')
    for name,identity in complete['dependencies'].items():
        if sha(name)!=identity:raise RuntimeError('feature implementation identity changed')
    rows=json.loads((features/'rows.json').read_text());labels=json.loads((features/'labels.json').read_text())
    for row in {r['feature_file']:r for r in rows}.values():
        if sha(row['feature_file'])!=row['feature_sha256']:raise RuntimeError('feature content changed')
    dest.mkdir(parents=True);implementation_manifest(dest/'implementation_manifest.json')
    store=GridStore();y=target_arrays(rows,labels,domain);context=np.asarray([r['context'] for r in rows],np.float32)
    x=np.stack([pooled(store.get(r)) for r in rows]);folds=range(4) if domain=='synthetic' else (-1,)
    results={};pending=[];oof={name:np.zeros_like(y) for name in ('cnn','rbf')}
    import torch
    from threadpoolctl import threadpool_limits
    for fold in folds:
        train=np.asarray([r['fold']!=fold if domain=='synthetic' else r['partition']=='FIT7' for r in rows])
        test=~train
        prefix=dest/f'fold{fold}';prefix.mkdir()
        pred,state,account=fit_cnn(rows,train,y,store,device)
        torch.save(state,prefix/'cnn.pt');np.savez_compressed(prefix/'cnn_predictions.npz',prediction=pred,fit=train,select=test)
        pending.append(('cnn',fold,pred,train,test,account))
        oof['cnn'][test]=pred[test]
        started=time.perf_counter()
        with threadpool_limits(limits=8):
            head=RBFReader().fit(x[train],context[train],y[train]);p=np.concatenate([head.predict(x[a:a+128],context[a:a+128]) for a in range(0,len(x),128)])
        head.save(prefix/'rbf.npz');np.savez_compressed(prefix/'rbf_predictions.npz',prediction=p,fit=train,select=test)
        pending.append(('rbf',fold,p,train,test,dict(head.accounting(),seconds=time.perf_counter()-started)))
        oof['rbf'][test]=p[test]
        print(json.dumps(dict(fold=fold,all_predictions_persisted=True,task_metrics_not_yet_computed=True)),flush=True)
    # All fixed reader/fold outputs are persisted before any SELECT metric is computed.
    dump(dest/'predictions_complete.json',dict(folds=list(folds),readers=['cnn','rbf'],created_utc=datetime.now(timezone.utc).isoformat()))
    for name,fold,p,train,test,account in pending:
        results[f'{name}/fold{fold}']=dict(accounting=account,metrics=grouped_metrics(p,rows,labels,domain,train,test))
    report=dict(schema='s43_witness_probe_results_v1',domain=domain,protocol_sha256=PROTOCOL_SHA,
      source_freeze_sha256=sha(freeze),feature_complete_sha256=sha(features/'complete.json'),results=results,
      cases=complete['cases'],input_views=len(rows),not_independent_samples='sampler/error/query views reuse same source; synthetic4contexts, existing9subjects18recordings',
      input='LNES-free grid40 plus public/conditional anchor; no sampler ID/action/target in encoder input',
      missing_strong_control='No DEVO-style learnedtask-score sampler, cannot certify novelty',
      full_frontend_or_deployment_claim=False)
    if domain=='synthetic':
        report['out_of_fold']={name:grouped_metrics(pred,rows,labels,domain,np.zeros(len(rows),bool),np.ones(len(rows),bool)) for name,pred in oof.items()}
        np.savez_compressed(dest/'out_of_fold_predictions.npz',**oof)
    if domain=='existing':report['development_gate']=development_gate({k.split('/')[0]:v for k,v in results.items()})
    dump(dest/'results.json',report);require_gate(freeze)
    return report


def engineering(out):
    """No physical task data or scientific targets: shapes, guards and source transport."""
    import torch
    torch.set_num_threads(8);torch.manual_seed(SEED)
    x=torch.randn(2,40,180,240);c=torch.zeros(2,70)
    model=make_cnn(63);p=model(x,c);assert p.shape==(2,63)
    p.square().mean().backward();assert any(a.grad is not None and torch.isfinite(a.grad).all() for a in model.parameters())
    empty=np.empty((0,4),np.int64);field=DenseFieldMemory(2,0)
    for t in range(50):field.ingest(empty,t)
    grid,meta=field.grid(50);assert grid.shape==(40,180,240) and np.all(grid[16:24]==1)
    assert np.all(grid[:16]==0) and meta['resident_bytes']<6*(1<<20)
    base=np.asarray([0,0,.6]+[0.]*48,np.float64);err=inject_anchor(base,'root_x10mm')
    assert np.array_equal(base[1:],err[1:]) and abs(err[0]-.01)<1e-12
    result=dict(schema='s43_witness_probe_engineering_v1',status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),
      source_sha256=sha(__file__),steps_run=0,scientific_data_opened=False,task_scores_read=False,
      cnn_parameters=sum(p.numel() for p in model.parameters()),native_local_nonlinearity_before_pool=True,
      input_shape=[40,180,240],outputs=[10,63],source_transport_and_target_signature_checked=True,
      field_empty_observed_not_missing=True,actual_steps_for_formal_cnn=2000,source_compatibility_notes='Existing context stores K/640 plus2constant clocks; K explicitly restored x640, head uses common70 observed fields only')
    dump(out,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('manifest');a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('engineering');a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('verify-existing');a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('freeze');a.add_argument('--out',type=Path,required=True);a.add_argument('--engineering',type=Path,nargs='+',required=True);a.add_argument('--commands',nargs='+',required=True)
    a=sub.add_parser('extract');a.add_argument('--domain',choices=('synthetic','existing'),required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--source-freeze',type=Path,required=True);a.add_argument('--queries',type=int,nargs='+',default=[50])
    a=sub.add_parser('fit');a.add_argument('--domain',choices=('synthetic','existing'),required=True);a.add_argument('--features',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--source-freeze',type=Path,required=True);a.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.command=='manifest':implementation_manifest(args.out)
    elif args.command=='engineering':engineering(args.out)
    elif args.command=='verify-existing':verify_existing_sources(args.out)
    elif args.command=='freeze':sourcefreeze(args.out,args.engineering,args.commands)
    elif args.command=='extract':extract(args.domain,args.out,args.source_freeze,tuple(args.queries))
    else:fit(args.domain,args.features,args.out,args.source_freeze,args.device)


if __name__=='__main__':main()
