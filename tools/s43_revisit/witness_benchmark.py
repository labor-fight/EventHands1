"""Target-free native-prefix representation cost; no estimator or mesh output.

Each case runs in a fresh CPU process. Public fixed pose/beta0 policies refresh
on the declared commit clock. These are scripted engineering conditions, never
pose accuracy, closed-loop tracking or complete sensor-to-mesh latency.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/s43_revisit_20261028/witness_resources'
PUBLIC=np.array([0.,0.,.6]+[0.]*48,np.float64)
CASES=[(arm,hop,hop) for arm in ('witness','quota','uniform','field2') for hop in (25,50,100)]+[('witness',25,50)]
SOURCE_NAMES=(
    'tools/s43_revisit/witness_benchmark.py','tools/s43_revisit/witness_core.py',
    'tools/s43_revisit/witness_geometry.py','tools/s43_revisit/witness_geometry_runtime.py',
    'tools/s43_revisit/witness_probe.py','tools/s43_rel/mean_shape_replay.py',
    'assets/mano_right.npz','outputs/s43_revisit_20261028/stage2_protocol.json',
    'outputs/s43_revisit_20261028/stage2_engineering_R1_prereg.json',
    'outputs/s43_revisit_20261028/stage2_pre_task_clock_amendment.json',
    'outputs/s43_revisit_20261028/witness_probe_engineering/actual_sourcefreeze.json')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item() if isinstance(a,np.generic) else str(a))+'\n')


def rss():
    with Path('/proc/self/status').open() as stream:
        for line in stream:
            if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    return None


def python_bytes(obj,seen=None):
    """Reachable object/header estimate, excluding ndarray payloads, not RSS."""
    seen=set() if seen is None else seen
    if id(obj) in seen:return 0
    seen.add(id(obj));size=sys.getsizeof(obj)
    if isinstance(obj,np.ndarray):return max(0,size-(obj.nbytes if obj.flags.owndata else 0))
    if isinstance(obj,dict):return size+sum(python_bytes(k,seen)+python_bytes(v,seen) for k,v in obj.items())
    if isinstance(obj,(list,tuple,set)):return size+sum(python_bytes(v,seen) for v in obj)
    if hasattr(obj,'__dict__') and not isinstance(obj,type):return size+python_bytes(vars(obj),seen)
    return size


def distribution(values,hop=None):
    a=np.asarray(values,np.float64)
    result=dict(n=len(a),mean_ms=float(a.mean()*1000),P50_ms=float(np.percentile(a,50)*1000),
                P95_ms=float(np.percentile(a,95)*1000),P99_ms=float(np.percentile(a,99)*1000))
    if hop is not None:
        result.update(exceeds_hop=int(np.sum(a>hop/1000.)),exceedance_fraction=float(np.mean(a>hop/1000.)))
    return result


def key(arm,hop,commit):return f'{arm}_h{hop}_commit{commit}'


def worker(freeze,arm,hop,commit,out):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only explicit empty CUDA_VISIBLE_DEVICES required')
    f=json.loads(Path(freeze).read_text())
    if (arm,hop,commit) not in [tuple(x) for x in f['cases']]:raise ValueError('case outside frozen engineering scope')
    for path,identity in {**f['sources'],**f['inputs']}.items():
        if sha(path)!=identity:raise RuntimeError('engineering source/input identity changed:'+path)
    import torch
    torch.set_num_threads(f['threads'])
    from threadpoolctl import threadpool_limits
    from tools.s43_rel.mean_shape_replay import MeanShapeInputReader,InferenceAccessGuard
    from .witness_core import EpisodeConfig,EpisodeMemory
    from .witness_geometry_runtime import CacheBoundMotionPolicy
    from .witness_probe import DenseFieldMemory
    started=time.perf_counter();records=[]
    with threadpool_limits(limits=f['threads']),InferenceAccessGuard() as guard:
        t=time.perf_counter();reader=MeanShapeInputReader(f['recording']);load_seconds=time.perf_counter()-t
        start=f['start_ms'];warm_end=start+f['warmup_ms'];stop=start+f['duration_ms']
        K=reader.K_cam.astype(np.float64).copy();K[:2]*=.375
        beta=np.zeros(10,np.float64)

        def policy(T):
            return CacheBoundMotionPolicy(PUBLIC,beta,K,T,mode=arm) if arm in ('witness','quota') else None

        t=time.perf_counter();initial=policy(start)
        if arm=='field2':memory=DenseFieldMemory(2,start)
        else:memory=EpisodeMemory(EpisodeConfig(policy=arm),policy=initial).reset('native_resource',start,policy=initial)
        reset_seconds=time.perf_counter()-t
        count=ordinal=0;digest=hashlib.sha256();max_packet=0;max_packet_events=0
        maximum_state=0;maximum_python=0;numeric_transaction=0;max_rss_service=0
        last_commit=start

        def consume(a,b):
            nonlocal count,ordinal,max_packet,max_packet_events
            pack_seconds=ingest_seconds=0.
            for ms in range(a,b):
                t=time.perf_counter();packet=reader.packet(ms,ms+1)
                lo,hi=packet['source_range'];digest.update(np.arange(lo,hi,dtype=np.int64).tobytes())
                pack_seconds+=time.perf_counter()-t
                t=time.perf_counter()
                if arm=='field2':memory.ingest(packet['events'],ms)
                else:memory.ingest(packet['events'],ms+1,ordinal,source_range=(lo,hi))
                ingest_seconds+=time.perf_counter()-t
                ordinal+=1;count+=len(packet['events'])
                max_packet=max(max_packet,packet['events'].nbytes);max_packet_events=max(max_packet_events,len(packet['events']))
            return pack_seconds,ingest_seconds

        def query(T):
            if arm=='field2':
                grid,receipt=memory.grid(T);return grid,None,receipt
            snap=memory.readout(T);return snap.to_grid40ch(),snap,None

        def commit_policy(T,snap):
            nonlocal last_commit
            t=time.perf_counter();new=policy(T);construction=time.perf_counter()-t
            t=time.perf_counter()
            if arm!='field2':memory.commit(new,T,expected_version=snap.version if snap is not None else None)
            elapsed=time.perf_counter()-t;last_commit=T
            return construction,elapsed

        # Two scripted50ms warm cycles prime the same memory and policy path.
        t=time.perf_counter();cursor=start
        for T in range(start+50,warm_end+1,50):
            consume(cursor,T);_,s,_=query(T);commit_policy(T,s);cursor=T
        warmup_seconds=time.perf_counter()-t
        if cursor!=warm_end:raise ValueError('warmup must be a multiple of50ms')
        source_at_warm=count;rss_before_service=rss();common={};duplicate=None
        for T in range(warm_end+hop,stop+1,hop):
            cycle=time.perf_counter();n_before=count
            before_stats=memory.stats() if arm!='field2' else None
            packing,ingestion=consume(cursor,T);cursor=T
            t=time.perf_counter();grid,snap,field_meta=query(T);reading=time.perf_counter()-t
            precommit_state = snap.metadata['resident_bytes'] if snap is not None else field_meta['resident_bytes']
            construction=committing=0.
            committed=(T-start)%commit==0
            if committed:construction,committing=commit_policy(T,snap)
            wall=time.perf_counter()-cycle
            # Auditing/hashing/profiling are outside primitive service timing.
            audit=time.perf_counter()
            grid_sha=hashlib.sha256(np.ascontiguousarray(grid).tobytes()).hexdigest()
            if snap is not None:
                dict_sha=snap.dictionary_sha256();content_sha=snap.content_sha256()
                state=memory.nbytes();statistics=memory.stats()
                selection_delta=statistics['selection_seconds']-before_stats['selection_seconds']
                numeric=statistics['numeric_state_bytes']
                policies={id(p):p for p in (memory.policy,memory._epoch_policy) if p is not None}
                receipts=[p.receipt() for p in policies.values()]
            else:
                dict_sha=None;content_sha=grid_sha;state=field_meta['resident_bytes'];numeric=state
                statistics=field_meta;selection_delta=0.;receipts=[]
            py=python_bytes(memory);maximum_state=max(maximum_state,state,precommit_state);maximum_python=max(maximum_python,py)
            numeric_transaction=max(numeric_transaction,numeric if arm!='field2' else 0)
            max_rss_service=max(max_rss_service,rss() or 0)
            if (T-warm_end)%50==0:common[str(T)]=dict(grid_sha256=grid_sha,dictionary_sha256=dict_sha,content_sha256=content_sha)
            if duplicate is None:
                version=None if snap is None else memory.version
                again,again_snap,_=query(T)
                duplicate=dict(grid_bitwise_equal=bool(np.array_equal(grid,again)),
                    dictionary_equal=snap is None or snap.dictionary_sha256()==again_snap.dictionary_sha256(),
                    version_unchanged=snap is None or memory.version==version,
                    note='query performed after optionalcommit; count content unchanged, query metadata poseversion may change')
                if not all(duplicate[k] for k in ('grid_bitwise_equal','dictionary_equal','version_unchanged')):
                    raise RuntimeError('pure query failed')
            records.append(dict(cutoff_ms=T,new_events=count-n_before,source_packet_seconds=packing,
                ingest_seconds=ingestion,ingest_selection_seconds_child=selection_delta,readout_grid_seconds=reading,
                policy_construction_seconds=construction,commit_seconds=committing,cycle_seconds=wall,
                unassigned_python_loop_seconds=wall-packing-ingestion-reading-construction-committing,
                committed=committed,resident_numeric_policy_bytes=max(state,precommit_state),
                precommit_numeric_policy_bytes=precommit_state,postcommit_numeric_policy_bytes=state,
                python_object_header_estimate_bytes=py,
                source_packet_max_bytes=max_packet,policy_receipts=receipts,statistics=statistics,
                audit_seconds=time.perf_counter()-audit))
        expected=int(reader.offsets[stop])-int(reader.offsets[start])
        if count!=expected:raise RuntimeError('native source count mismatch')
        receipt_guard=guard.receipt()
    receipt_guard=guard.receipt()
    if receipt_guard['blocked_aux_beta_attempts'] or receipt_guard['blocked_pose_meta_attempts']:
        raise RuntimeError('forbidden inference access attempted')
    unchanged={p:sha(p)==v for p,v in {**f['sources'],**f['inputs']}.items()}
    if not all(unchanged.values()):raise RuntimeError('source changed during worker')
    parts=('source_packet_seconds','ingest_seconds','readout_grid_seconds','policy_construction_seconds','commit_seconds')
    result=dict(schema='s43_witness_native_representation_benchmark_v1',execution_complete=True,
        recording=f['recording'],sampler=arm,query_hop_ms=hop,commit_hop_ms=commit,
        sourcefreeze_sha256=sha(freeze),start_ms=start,stop_ms=stop,warmup_ms=f['warmup_ms'],
        actual_source_events=count,expected_source_events=expected,source_events_warmup=source_at_warm,
        source_events_service=count-source_at_warm,source_id_digest=digest.hexdigest(),source_files_unchanged=unchanged,
        script_pose=PUBLIC.tolist(),inference_beta='publiczeros10',K_event=K.tolist(),K_scale_applied_once=.375,
        estimator_run=False,mesh_output_run=False,GT_or_target_opened=False,inference_access_guard=receipt_guard,
        output_contract='native events to grid40 + scripted context policy refresh; representation engineering only',
        load_reader_seconds=load_seconds,reset_initial_policy_seconds=reset_seconds,warmup_seconds=warmup_seconds,
        worker_core_seconds=time.perf_counter()-started,latency=distribution([r['cycle_seconds'] for r in records],hop),
        components={p:distribution([r[p] for r in records]) for p in parts},
        child_note='ingest_selection is inside ingest and includes scalar scorer/kernel refresh; do not add it again',
        events_per_second_service=(count-source_at_warm)/sum(r['cycle_seconds'] for r in records),
        queries_per_second_service=len(records)/sum(r['cycle_seconds'] for r in records),
        maximum_numeric_policy_state_bytes=maximum_state,maximum_python_object_header_estimate_bytes=maximum_python,
        RSS_before_service_bytes=rss_before_service,RSS_sampled_after_cycles_peak_bytes=max_rss_service,
        RSS_process_lifetime_peak_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
        memory_scope='nbytes excludesPythonheaders/shared static geometry; Python recursive object/header estimate excludesarraypayload; RSS includesstartup/mmap/library and boundedscratch',
        explicit_scratch_bounds=dict(max_native_packet_payload_bytes=max_packet,max_native_packet_events=max_packet_events,
            source_adapter_explicit_array_upper_bytes=max_packet_events*64,
            transaction_numeric_clone_bytes=numeric_transaction,
            new_tile_payload_upper_bytes=23*30*10*8*8*2*4 if arm!='field2' else 0,
            selection_repack_bytes=128*10*8*8*2*4 if arm!='field2' else 0,
            caveat='These are separately bounded explicit buffers, not totalallocatorpeak. NumPyunique/sort/Python,queryraster and LAPACK/sharedlibrary scratch not certified by summing them; lifetimeRSS measured separately.'),
        latency_scope='1ms input mmap/packing/source-IDdigest + ingest/seal/selection/kernel + readout/grid + scripted policyconstruction/commit. No physicalwait/network/mesh/serialization/sourcefullhash/audit.',
        cached_input_note='fullinputSHA validation precedeswarmup and can warm filepages; no coldIO claim',
        pure_query_check=duplicate,common_50ms_prefix=common,records=records)
    dump(out,result)
    print(json.dumps(dict(case=key(arm,hop,commit),queries=len(records),P50_ms=result['latency']['P50_ms'],state_bytes=maximum_state),ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--source-freeze',type=Path)
    p.add_argument('--sampler',choices=('witness','quota','uniform','field2'))
    p.add_argument('--hop',type=int,choices=(25,50,100))
    p.add_argument('--commit-hop',type=int,choices=(25,50,100))
    args=p.parse_args()
    if args.worker:return worker(args.source_freeze,args.sampler,args.hop,args.commit_hop,args.out)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('run with CUDA_VISIBLE_DEVICES empty')
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    from tools.s43_rel.mean_shape_replay import MeanShapeInputReader,InferenceAccessGuard
    with InferenceAccessGuard() as guard:
        reader=MeanShapeInputReader('ch_local')
        eligible=[(int(a),int(b)) for a,b in reader.runs if b-a>=3300]
        if not eligible:raise RuntimeError('no3300ms valid native prefix')
        start=eligible[0][0]
        inputs={str(x.resolve()):sha(x) for x in reader.input_paths()}
        inputs[str((reader.root/'splits_semkine.json').resolve())]=sha(reader.root/'splits_semkine.json')
    sources={str((ROOT/name).resolve()):sha(ROOT/name) for name in SOURCE_NAMES}
    for name in SOURCE_NAMES:
        destination=args.out/'sources'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
        if sha(destination)!=sources[str((ROOT/name).resolve())]:raise RuntimeError('source snapshot race')
    commands=[]
    for arm,hop,commit in CASES:
        commands.append([sys.executable,'-m','tools.s43_revisit.witness_benchmark','--worker',
            '--source-freeze',str((args.out/'sourcefreeze.json').resolve()),'--sampler',arm,
            '--hop',str(hop),'--commit-hop',str(commit),'--out',str((args.out/(key(arm,hop,commit)+'.json')).resolve())])
    freeze=args.out/'sourcefreeze.json'
    dump(freeze,dict(schema='s43_witness_representation_resource_sourcefreeze_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        scope='target-free scripted native ch_local3300ms,100warm+3200service;13CPUcases, no estimator or mesh',
        recording='ch_local',start_ms=start,duration_ms=3300,warmup_ms=100,threads=args.threads,
        cases=CASES,sources=sources,inputs=inputs,actual_parent_argv=[sys.executable,*sys.argv],worker_commands=commands,
        environment={'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':str(args.threads),'MKL_NUM_THREADS':str(args.threads),'OMP_NUM_THREADS':str(args.threads)},
        parent_input_guard=guard.receipt(),scientific_tasks_or_scores=False))
    execution=[]
    for cmd in commands:
        started=time.perf_counter();env=os.environ.copy();env.update(json.loads(freeze.read_text())['environment'])
        r=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True)
        case=Path(cmd[-1]).stem
        (args.out/(case+'.log')).write_text(r.stdout+r.stderr)
        execution.append(dict(case=case,command=cmd,returncode=r.returncode,process_wall_seconds=time.perf_counter()-started))
        dump(args.out/'executions.json',execution)
        if r.returncode:
            dump(args.out/'incomplete.json',dict(case=case,returncode=r.returncode,sourcefreeze_sha256=sha(freeze)))
            raise RuntimeError('resource worker failed; preserve logfile:'+case)
        print(r.stdout.strip(),flush=True)
    cases={key(a,h,c):json.loads((args.out/(key(a,h,c)+'.json')).read_text()) for a,h,c in CASES}
    base=cases[key('witness',50,50)]['common_50ms_prefix']
    extra=cases[key('witness',25,50)]['common_50ms_prefix']
    shared=sorted(set(base)&set(extra),key=int)
    pure=dict(shared_prefixes=len(shared),all_grid_dictionary_content_exact=all(base[t]==extra[t] for t in shared),
              metadata='sameprefix,policycommit50; additionalquery25 does notselect/commit')
    if not pure['all_grid_dictionary_content_exact']:raise RuntimeError('fixedcommit purequery sharedprefix mismatch')
    dump(args.out/'summary.json',dict(schema='s43_witness_native_resource_summary_v1',execution_complete=True,
        sourcefreeze_sha256=sha(freeze),scope='representationonly, scriptedpublicpose/beta0; no accuracy or network/mesh realtime claim',
        cases={k:{x:v[x] for x in ('latency','components','maximum_numeric_policy_state_bytes','maximum_python_object_header_estimate_bytes',
            'RSS_process_lifetime_peak_bytes','actual_source_events','events_per_second_service','queries_per_second_service')} for k,v in cases.items()},
        purequery_fixed50=pure,executions=execution))


if __name__=='__main__':main()
