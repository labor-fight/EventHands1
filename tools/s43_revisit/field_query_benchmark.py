"""A target-free representation resources on the frozen ch3300ms prefix.

No estimator, training, target, score or output mesh. All cases are isolated
CPU workers; public fixed pose commits and pure extra queries are explicit.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np

from .witness_benchmark import sha,dump,rss,python_bytes,distribution

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/s43_revisit_20261028'
PUBLIC=np.asarray([0.,0.,.6]+[0.]*48,np.float64)
MODES=('field_core','hand_box','static','generic_trajectory','anatomical')
CASES=[(mode,hop,hop) for mode in MODES for hop in (25,50,100)]+[('anatomical',25,50)]
SOURCES=('tools/s43_revisit/field_query_benchmark.py','tools/s43_revisit/field_query.py',
    'tools/s43_revisit/field_core.py','tools/s43_revisit/witness_geometry.py',
    'tools/s43_revisit/field_query_probe.py',
    'tools/s43_revisit/witness_benchmark.py','tools/s43_rel/geometry.py',
    'tools/s43_rel/mean_shape_replay.py','assets/mano_right.npz',
    'outputs/s43_revisit_20261028/stage3_field_query_protocol.json',
    'outputs/s43_revisit_20261028/field_query_geometry_engineering/receipt.json',
    'tests/test_s43_field_query_benchmark.py',
    'outputs/s43_revisit_20261028/field_resource_engineering/receipt.json',
    'outputs/s43_revisit_20261028/field_engineering/native_1000ms.json',
    'outputs/s43_revisit_20261028/witness_resources_v2/sourcefreeze.json')


def case_key(mode,hop,commit):return f'{mode}_h{hop}_commit{commit}'


def numeric_arrays(obj,seen=None):
    seen=set() if seen is None else seen
    if id(obj) in seen:return 0
    seen.add(id(obj))
    if isinstance(obj,np.ndarray):return obj.nbytes
    if isinstance(obj,dict):return sum(numeric_arrays(v,seen) for v in obj.values())
    if isinstance(obj,(tuple,list)):return sum(numeric_arrays(v,seen) for v in obj)
    if hasattr(obj,'__dict__') and not isinstance(obj,type):return numeric_arrays(vars(obj),seen)
    return 0


def bounded_history(history,T):
    """Keep one bracket before T-50 and later real scripted commit knots."""
    items=list(history);first=next((i for i,(t,_) in enumerate(items) if t>=T-50),len(items))
    return items[max(0,first-1):]


def worker(freeze,mode,hop,commit,out):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('requires explicit CPU-only CUDA_VISIBLE_DEVICES empty')
    f=json.loads(Path(freeze).read_text());identities={**f['sources'],**f['inputs']}
    if (mode,hop,commit) not in [tuple(x) for x in f['cases']]:raise ValueError('case outside frozen resource scope')
    for p,v in identities.items():
        if sha(p)!=v:raise RuntimeError('frozen source/input changed:'+p)
    import torch
    from threadpoolctl import threadpool_limits
    from .field_core import TimeFieldMemory
    from .field_query import FieldQueryBuilder
    from . import witness_geometry as G
    from tools.s43_rel.mean_shape_replay import MeanShapeInputReader,InferenceAccessGuard
    torch.set_num_threads(f['threads']);start=f['start_ms'];stop=start+3300;warm_end=start+100
    count=ordinal=0;records=[];digest=hashlib.sha256();phase='initial';cursor=start
    try:
        with threadpool_limits(limits=f['threads']),InferenceAccessGuard() as guard:
            t=time.perf_counter();reader=MeanShapeInputReader('ch_local');load=time.perf_counter()-t
            K=np.asarray(reader.K_cam,np.float64).copy();K[:2]*=.375;beta=np.zeros(10)
            t=time.perf_counter();memory=TimeFieldMemory(seq_id='field_resource',boundary_ms=start)
            builder=None if mode=='field_core' else FieldQueryBuilder(PUBLIC,beta,K,start)
            reset=time.perf_counter()-t;max_packet=max_packet_events=0;last_commit=start

            def consume(a,b):
                nonlocal count,ordinal,max_packet,max_packet_events,phase,cursor
                packing=ingestion=0.
                for ms in range(a,b):
                    phase='source_packet';t=time.perf_counter();p=reader.packet(ms,ms+1)
                    lo,hi=p['source_range'];digest.update(np.arange(lo,hi,dtype=np.int64).tobytes());packing+=time.perf_counter()-t
                    phase='ingest';t=time.perf_counter();memory.ingest(p['events'],ms+1,ordinal,source_range=(lo,hi));ingestion+=time.perf_counter()-t
                    ordinal+=1;count+=len(p['events']);cursor=ms+1
                    max_packet=max(max_packet,p['events'].nbytes);max_packet_events=max(max_packet_events,len(p['events']))
                return packing,ingestion

            def query(T):
                nonlocal phase
                phase='field_readout_grid';t=time.perf_counter();snap=memory.readout(T);grid=snap.to_grid40ch();reading=time.perf_counter()-t
                prepare=geometry=sampling=0.;q=local=None;cold=False
                if builder is not None:
                    phase='local_cell_prepare';t=time.perf_counter()
                    fine=grid[:16].reshape(8,2,180,240)[:,:,::2,::2]
                    coverage=grid[16:24,None,::2,::2]
                    cell=torch.tensor(np.concatenate((fine,coverage),axis=1));prepare=time.perf_counter()-t
                    phase='query_coords';cold=builder._geo is None or (mode in ('anatomical','generic_trajectory') and builder._endpoint_states is None)
                    t=time.perf_counter();q=builder.build(T,mode=mode);geometry=time.perf_counter()-t
                    phase='cell_sample_flags';t=time.perf_counter();samples=q.sample(cell)
                    view=q.in_view.reshape(8,5,3,3,2,7,7).permute(1,2,3,4,0,5,6).unsqueeze(-1).float()
                    support=q.full_interp_support.reshape(8,5,3,3,2,7,7).permute(1,2,3,4,0,5,6).unsqueeze(-1).float()
                    local=torch.cat((samples,view,support),-1);sampling=time.perf_counter()-t
                return snap,grid,q,local,dict(field_readout_grid_seconds=reading,local_cell_prepare_seconds=prepare,
                    querycoords_seconds=geometry,sample_flags_seconds=sampling),cold

            def commit_public(T):
                nonlocal builder,last_commit,phase
                phase='public_commit';t=time.perf_counter()
                if builder is not None:builder=FieldQueryBuilder(PUBLIC,beta,K,T,history=bounded_history(builder.history,T))
                last_commit=T
                return time.perf_counter()-t

            t=time.perf_counter();pure=None
            for T in (start+50,start+100):
                consume(cursor,T);snap,grid,q,local,_,_=query(T)
                if pure is None:
                    v=memory.version;before=snap.content_sha256();hist=None if builder is None else tuple(t for t,_ in builder.history)
                    s2,g2,q2,l2,_,_=query(T)
                    pure=dict(field_grid_exact=bool(np.array_equal(grid,g2)),field_content_exact=before==s2.content_sha256(),version_unchanged=v==memory.version,
                        query_grid_exact=q is None or bool(torch.equal(q.grid,q2.grid)),local_samples_exact=local is None or bool(torch.equal(local,l2)),
                        deployed_history_unchanged=builder is None or hist==tuple(t for t,_ in builder.history),when='warmup beforecommit; no duplicatequery cost in service')
                    if not all(v for k,v in pure.items() if k!='when'):raise RuntimeError('pure query failed')
                commit_public(T)
            warm_seconds=time.perf_counter()-t;warm_count=count;rss_start=rss();max_rss=0;max_numeric=0;max_python=0
            common={};scratch={};loadavg_before=list(os.getloadavg())
            for T in range(warm_end+hop,stop+1,hop):
                cycle=time.perf_counter();n_before=count
                packing,ingestion=consume(cursor,T)
                snap,grid,q,local,parts,cold=query(T)
                audit=time.perf_counter();pre_numeric=memory.nbytes()+numeric_arrays(builder);pre_history=0 if builder is None else len(builder.history)
                audit_pre=time.perf_counter()-audit
                committed=(T-start)%commit==0;committing=commit_public(T) if committed else 0.
                wall=time.perf_counter()-cycle-audit_pre
                # Digests, recursive object accounting and RSS outside service.
                audit=time.perf_counter();numeric=memory.nbytes()+numeric_arrays(builder);py=python_bytes(dict(memory=memory,builder=builder))
                max_numeric=max(max_numeric,numeric,pre_numeric);max_python=max(max_python,py);max_rss=max(max_rss,rss() or 0)
                entry=dict(field_grid_sha256=hashlib.sha256(grid.tobytes()).hexdigest(),field_content_sha256=snap.content_sha256())
                if q is not None:
                    entry.update(query_grid_sha256=hashlib.sha256(q.grid.numpy().tobytes()).hexdigest(),
                        local5_sha256=hashlib.sha256(local.numpy().tobytes()).hexdigest(),query_ref_ms=q.meta['ref_ms'])
                if (T-warm_end)%50==0:common[str(T)]=entry
                scratch=dict(snapshot_array_bytes=sum(x.nbytes for x in (snap.counts,snap.times_ms,snap.observed,snap.source_range_span,snap.source_events_per_ms)),
                    global_grid40_bytes=grid.nbytes,query_tensor_bytes=0 if q is None else q.meta['coordinate_tensor_bytes'],
                    local5_float32_bytes=0 if local is None else local.numel()*local.element_size(),
                    local_cell3_float32_bytes=0 if q is None else 8*3*90*120*4,
                    caveat='Outputs and named scratch,not summed allocator peak; numpy reductions/grid_sample intermediates and object bookkeeping separate')
                measured=packing+ingestion+sum(parts.values())+committing
                records.append(dict(cutoff_ms=T,new_events=count-n_before,source_packet_seconds=packing,ingest_seconds=ingestion,
                    **parts,commit_seconds=committing,cycle_seconds=wall,unassigned_loop_seconds=wall-measured,
                    committed=committed,query_cache_cold=cold,last_commit_after_ms=last_commit,query_history_knots=pre_history,
                    precommit_numeric_bytes=pre_numeric,postcommit_numeric_bytes=numeric,python_header_estimate_bytes=py,
                    audit_seconds=audit_pre+time.perf_counter()-audit))
            expected=int(reader.offsets[stop])-int(reader.offsets[start]);assert count==expected==f['expected_source_events']
            assert digest.hexdigest()==f['expected_source_id_digest']
            access=guard.receipt()
        access=guard.receipt();assert not access['aux_betas_accessed'] and not access['pose_metadata_opened']
        assert not access['blocked_aux_beta_attempts'] and not access['blocked_pose_meta_attempts']
        unchanged={p:sha(p)==v for p,v in identities.items()};assert all(unchanged.values())
        components=('source_packet_seconds','ingest_seconds','field_readout_grid_seconds','local_cell_prepare_seconds','querycoords_seconds','sample_flags_seconds','commit_seconds')
        constants=sum(v.nbytes for v in G._joint_constants().values()) if G._joint_constants.cache_info().currsize else 0
        shape_entries=G._shape_terms.cache_info().currsize
        result=dict(schema='s43_field_query_native_resource_v1',execution_complete=True,mode=mode,query_hop_ms=hop,commit_hop_ms=commit,
            sourcefreeze_sha256=sha(freeze),actual_source_events=count,expected_source_events=expected,source_events_warmup=warm_count,source_events_service=count-warm_count,
            source_id_digest=digest.hexdigest(),source_files_unchanged=unchanged,recording='ch_local',start_ms=start,stop_ms=stop,warmup_ms=100,threads=f['threads'],
            public_pose=PUBLIC.tolist(),inference_beta='zeros10',K_event=K.tolist(),estimator_run=False,mesh_output_run=False,GT_or_target_opened=False,
            access_guard=access,load_seconds=load,reset_seconds=reset,warmup_seconds=warm_seconds,latency=distribution([r['cycle_seconds'] for r in records],hop),
            components={k:distribution([r[k] for r in records]) for k in components},
            service_total_seconds=sum(r['cycle_seconds'] for r in records),events_per_second_service=(count-warm_count)/sum(r['cycle_seconds'] for r in records),
            numeric_state_max_bytes=max_numeric,field_core_numeric_bytes=memory.nbytes(),python_object_header_estimate_max_bytes=max_python,
            shared_joint_constants_array_bytes=constants,shared_shape_cache_entries=shape_entries,shared_shape_cache_entry_bytes=504,
            external_public_pose_beta_K_bytes=560,RSS_before_service_bytes=rss_start,RSS_sampled_service_peak_bytes=max_rss,
            RSS_process_lifetime_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,loadavg_before=loadavg_before,loadavg_after=list(os.getloadavg()),
            explicit_output_scratch=scratch,max_packet_payload_bytes=max_packet,max_packet_events=max_packet_events,
            scope='representation only:1ms nativepack+IDdigest/ingest/readout40/cellprep/querycoords/bilinearsample+flags/scriptedcommit;noCNN/head/mesh/wait/serialization/audit',
            lazy_geometry='Each commit makes newbuilder; action calibration is lazy and its refresh cost counted in next querycoords; purequeries reuse only computationalcache',
            history='bounded real scripted constant-publicpose commit knots; interpolated past, legalfuturehypotheses;no target state',
            memory_scope='Ownedfield/builder numeric and recursivePythonheaders separately; sharedglobals and callercondition separate; snapshot/grid/tokens/scratch and RSS not statebytes',
            comparison='Oldfield2 used different DenseFieldMemory implementation; exactgrid parity is checked, cost delta is implementation+addedquery,not a new event statistic or task gain',
            pure_query_check=pure,common_50ms_prefix=common,records=records)
        dump(out,result);print(json.dumps(dict(case=case_key(mode,hop,commit),P50_ms=result['latency']['P50_ms'],numeric=max_numeric)),flush=True)
    except Exception as e:
        dump(out,dict(schema='s43_field_query_native_resource_failure_v1',execution_complete=False,error=repr(e),traceback=traceback.format_exc(),
            phase=phase,consumed_events=count,consumed_to_ms=cursor,source_id_digest=digest.hexdigest(),records=records,sourcefreeze_sha256=sha(freeze)))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=BASE/'field_query_resources_v1');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--worker',action='store_true');p.add_argument('--source-freeze',type=Path);p.add_argument('--mode',choices=MODES);p.add_argument('--hop',type=int);p.add_argument('--commit-hop',type=int)
    args=p.parse_args()
    if args.worker:return worker(args.source_freeze,args.mode,args.hop,args.commit_hop,args.out)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or args.threads!=2:raise RuntimeError('freeze domain is explicitCPU2')
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    engineering=json.loads((BASE/'field_resource_engineering/receipt.json').read_text())
    if not engineering.get('unit_tests_passed') or engineering.get('benchmark_source_sha256')!=sha(__file__):
        raise RuntimeError('resource engineering receipt must bind this actual source before run')
    from tools.s43_rel.mean_shape_replay import MeanShapeInputReader,InferenceAccessGuard
    old=json.loads((BASE/'witness_resources_v2/field2_h50_commit50.json').read_text())
    with InferenceAccessGuard() as guard:
        r=MeanShapeInputReader('ch_local');eligible=[(int(a),int(b)) for a,b in r.runs if b-a>=3300];assert eligible
        start=eligible[0][0];assert start==old['start_ms']
        expected=int(r.offsets[start+3300])-int(r.offsets[start]);assert expected==old['actual_source_events']==247738
        inputs={str(x.resolve()):sha(x) for x in r.input_paths()};inputs[str(r.root/'splits_semkine.json')]=sha(r.root/'splits_semkine.json')
    sources={str(ROOT/n):sha(ROOT/n) for n in SOURCES}
    for n in SOURCES:
        dest=args.out/'sources'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest);assert sha(dest)==sources[str(ROOT/n)]
    commands=[[sys.executable,'-m','tools.s43_revisit.field_query_benchmark','--worker','--source-freeze',str((args.out/'sourcefreeze.json').resolve()),
        '--mode',mode,'--hop',str(hop),'--commit-hop',str(commit),'--out',str((args.out/(case_key(mode,hop,commit)+'.json')).resolve())] for mode,hop,commit in CASES]
    freeze=args.out/'sourcefreeze.json'
    dump(freeze,dict(schema='s43_field_query_resource_sourcefreeze_v1',created_utc=datetime.now(timezone.utc).isoformat(),scientific_tasks_or_scores=False,
        scope='targetfree same ch3300ms247738events aswitness_resources_v2;100warm+3200service,16freshCPU2processes',cases=CASES,start_ms=start,expected_source_events=expected,
        expected_source_id_digest=old['source_id_digest'],threads=2,sources=sources,inputs=inputs,commands=commands,parent_access_guard=guard.receipt(),
        environment=dict(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
        input_note='Opaque sourceSHA includesauxfile bytes,only cameraK/validruns members semantically read; no beta/pose/targetaccess. SHAvalidation can warm filepages; no coldIO claim.'))
    execution=[]
    for cmd in commands:
        started=time.perf_counter();env=os.environ.copy();env.update(json.loads(freeze.read_text())['environment'])
        result=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True);case=Path(cmd[-1]).stem
        (args.out/(case+'.log')).write_text(result.stdout+result.stderr);execution.append(dict(case=case,returncode=result.returncode,wall_seconds=time.perf_counter()-started,command=cmd))
        dump(args.out/'executions.json',execution);print(json.dumps(execution[-1]),flush=True)
    values={case_key(m,h,c):json.loads((args.out/(case_key(m,h,c)+'.json')).read_text()) for m,h,c in CASES}
    successful={k:v for k,v in values.items() if v['execution_complete']}
    pure={}
    a=successful.get('anatomical_h50_commit50');b=successful.get('anatomical_h25_commit50')
    if a and b:
        keys=set(a['common_50ms_prefix'])&set(b['common_50ms_prefix']);pure=dict(prefixes=len(keys),all_fields_exact=all(a['common_50ms_prefix'][k]==b['common_50ms_prefix'][k] for k in keys))
    old_compare={}
    for hop in (25,50,100):
        previous=json.loads((BASE/f'witness_resources_v2/field2_h{hop}_commit{hop}.json').read_text());new=successful.get(case_key('field_core',hop,hop))
        if new:
            keys=set(previous['common_50ms_prefix'])&set(new['common_50ms_prefix'])
            old_compare[str(hop)]=dict(common_prefixes=len(keys),grid_bitwise=all(previous['common_50ms_prefix'][k]['grid_sha256']==new['common_50ms_prefix'][k]['field_grid_sha256'] for k in keys),
                old_latency=previous['latency'],new_latency=new['latency'],old_numeric_bytes=previous['maximum_numeric_policy_state_bytes'],new_numeric_bytes=new['numeric_state_max_bytes'])
    contracts=len(successful)==16 and pure.get('all_fields_exact',False) and all(v['grid_bitwise'] for v in old_compare.values())
    dump(args.out/'summary.json',dict(schema='s43_field_query_resource_summary_v1',sourcefreeze_sha256=sha(freeze),contracts_pass=contracts,
        execution_complete=len(successful)==16,failed=[k for k,v in values.items() if not v['execution_complete']],
        scope='scripted publicpose/beta0 representationonly; no estimator/mesh or accuracy',fixed_commit50_extraquery25=pure,old_field2_implementation_comparison=old_compare,
        cases={k:{name:v[name] for name in ('latency','components','numeric_state_max_bytes','python_object_header_estimate_max_bytes','RSS_sampled_service_peak_bytes','RSS_process_lifetime_peak_bytes','actual_source_events')} for k,v in successful.items()}))
    if not contracts:raise RuntimeError('resource matrix incomplete/parityfailed; all artifacts preserved')


if __name__=='__main__':main()
