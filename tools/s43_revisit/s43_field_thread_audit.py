"""Independent CPU-thread execution audit; never changes training processes.

Fixed geometry/runtime worker1, NumPy BLAS1, source/model/conditions. Compare
Torch CPU8->1->8->1, then2; no optimization or scientific target/score reads.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from .s43_field_training_R1 import build_model

ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def plans_identity(plans):
    h=hashlib.sha256();metas=[]
    for plan in plans:
        for name in ('grid','valid','in_view','full_interp_support','centers_px','depth_valid','bin_edges_ms','bin_times_ms','history_supported'):
            h.update(name.encode());h.update(getattr(plan.query,name).numpy().tobytes())
        h.update(plan.association_flag.numpy().tobytes())
        # Timing is deliberately not a semantic condition of the query.
        metas.append({k:v for k,v in plan.metadata.items() if k!='build_seconds'})
    return dict(grid_flags_sha256=h.hexdigest(),semantic_metadata_sha256=hashlib.sha256(json.dumps(metas,sort_keys=True).encode()).hexdigest())


def conditions(B):
    p=torch.zeros(B,51);p[:,2]=.6
    # Public fixed conditions, not current learned/GT poses. Half invalid
    # exercises exactly the same total-query policy across thread settings.
    p[::4,2]=-.2;p[1::4,0]=100
    p[2::4,3]=.1
    beta=torch.zeros(B,10);K=torch.tensor([[600.,0,320],[0,600,240],[0,0,1.]])[None].repeat(B,1,1)
    return p,beta,K,np.zeros(B,np.int64),np.full(B,50,np.int64),[()]*B


def cpu_audit(model,repetitions):
    from .field_query import _tensor
    report={};micro={}
    for B in (8,64):
        args=conditions(B);first=None;rows=[]
        for label,threads in (('8a',8),('1a',1),('8b',8),('1b',1),('2',2)):
            torch.set_num_threads(threads)
            model.prepare_queries_cpu(*args) # source/constants/allocator warmup
            wall=[];process=[]
            for _ in range(repetitions):
                rng=torch.get_rng_state().clone();start=time.perf_counter();cp=time.process_time()
                plans=model.prepare_queries_cpu(*args)
                process.append((time.process_time()-cp)*1000);wall.append((time.perf_counter()-start)*1000)
                assert torch.equal(torch.get_rng_state(),rng)
                identity=plans_identity(plans)
                if first is None:first=identity
                assert identity==first
            rows.append(dict(label=label,torch_CPU_threads=threads,wall_ms=wall,process_CPU_ms=process,
                wall_median_ms=float(np.median(wall)),CPU_to_wall_ratio=float(np.sum(process)/np.sum(wall)),
                exact_grid_flags_metadata_source=True,RNG_unchanged=True))
        eight=np.concatenate([r['wall_ms'] for r in rows if r['torch_CPU_threads']==8])
        one=np.concatenate([r['wall_ms'] for r in rows if r['torch_CPU_threads']==1])
        report[str(B)]=dict(rows=rows,identity=first,median8_ms=float(np.median(eight)),median1_ms=float(np.median(one)),
            ratio8_over1=float(np.median(eight)/np.median(one)))
    # Mechanism-only small-copy/reduction companion, after the uninstrumented
    # geometry comparison. These are not substituted for the measured query.
    array=np.linspace(-1,1,8*90*49*2,dtype=np.float64).reshape(8,90,49,2)
    flags=np.ones((8,90,49),bool)
    for threads in (8,1,2):
        torch.set_num_threads(threads);times={}
        for name,fn in (('coordinate_tensor_copy70k',lambda:_tensor(array,torch.float32)),
                        ('bool_tensor_copy35k',lambda:_tensor(flags,torch.bool)),
                        ('bool_sum35k',lambda:torch.as_tensor(flags).sum())):
            for _ in range(3):fn()
            samples=[]
            for _ in range(20):
                start=time.perf_counter();fn();samples.append((time.perf_counter()-start)*1000)
            times[name]=dict(median_ms=float(np.median(samples)),all_ms=samples)
        micro[str(threads)]=times
    return report,micro


def gpu_audit():
    """Optional sameB64 complete inputs/query, same checkpoint/public prefix."""
    from tools.s43_rel.mean_shape_replay import MeanShapeInputReader,InferenceAccessGuard
    from .s43_field_runtime_R1 import SharedS43RuntimeR1
    from .s43_field_data import make_input_batch
    model,contract=build_model('field_static','cuda:0');model.eval();model.requires_grad_(False)
    B=64;p,b,K,refs,T,hist=conditions(B)
    before={k:tensor_hash(v) for k,v in model.state_dict().items()};reference=None;rows=[]
    # Retain existing query plans for after-clock identity checks only. This
    # adds no kernel, sync, RNG or mathematical work within geometry/NN.
    original_prepare=model.prepare_queries_cpu;captured=[]
    def capture(*a,**kw):
        value=original_prepare(*a,**kw);captured[:]=value;return value
    model.prepare_queries_cpu=capture
    with InferenceAccessGuard() as guard:
        reader=MeanShapeInputReader('ch_local')
        def run(threads):
            torch.set_num_threads(threads)
            runtime=SharedS43RuntimeR1(model,arm=contract['runtime_arm'],runtime_workers=1,
                temporal_collapse=False,tensor_readout=True,state_fp32=True)
            phases=[];predictions=[]
            for ordinal in range(3):
                left,right=ordinal*50,(ordinal+1)*50;packet=reader.packet(left,right)
                items=[dict(lane=i,valid=True,padding=False,seq_id=f'thread-audit:{i}',reset=ordinal==0,recording='ch_local',run_id=0,
                    run_start_ms=0,run_stop_ms=150,start_ms=left,cutoff_ms=right,target_ms=right-1,packet_ordinal=ordinal,
                    initial_state=p[i].numpy(),beta_input=np.zeros(10,np.float32),K_source=reader.K_cam.copy(),**packet) for i in range(B)]
                batch=make_input_batch(items);torch.cuda.synchronize();start=time.perf_counter()
                formed=runtime.inputs(batch);torch.cuda.synchronize();middle=time.perf_counter()
                out=runtime.query(formed,inference=True);torch.cuda.synchronize();finish=time.perf_counter()
                identities=dict(field40=tensor_hash(formed.field40),source_chain=runtime.audit['source_chain'],
                    batch_token=batch.token,previous=tensor_hash(formed.previous),grid=plans_identity(captured))
                predictions.append(out.prediction.detach().cpu().clone())
                assert int(out.warmup_mask.sum())==(B if ordinal<2 else 0)
                assert int(out.supervised_mask.sum())==(0 if ordinal<2 else B)
                runtime.commit(out)
                phases.append(dict(role='warmup' if ordinal<2 else 'steady',cutoff_ms=right,
                    input_ms=(middle-start)*1000,query_ms=(finish-middle)*1000,total_ms=(finish-start)*1000,identity=identities))
            runtime.close()
            return dict(torch_CPU_threads=threads,phases=phases,ownwarmup_outputs=runtime.audit['warmup_outputs'],
                        steady_outputs=runtime.audit['supervised_outputs']),torch.stack(predictions)
        run(8) # GPU code/allocator warmup, excluded
        for label,threads in (('8a',8),('1a',1),('8b',8),('1b',1),('2',2)):
            rng=torch.get_rng_state().clone();row,pred=run(threads);row['label']=label
            identities=[p['identity'] for p in row['phases']]
            if reference is None:reference=(identities,pred)
            assert identities==reference[0]
            maxdiff=float((pred-reference[1]).abs().max());assert maxdiff<=2e-6
            assert torch.equal(torch.get_rng_state(),rng)
            row.update(prediction_max_abs_vs_first=maxdiff,source_field_exact=True,RNG_unchanged=True);rows.append(row)
    assert before=={k:tensor_hash(v) for k,v in model.state_dict().items()}
    return dict(rows=rows,guard=guard.receipt(),model_buffers_and_weights_exact=True,
        declared_prediction_tolerance=2e-6,scope='fixed real ch0:150ms B64 field_static, two own50mswarmups+one steady query; no optimizer/targets; sharedhost only')


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--out',type=Path,required=True);a.add_argument('--gpu',action='store_true');a.add_argument('--gpu-only',action='store_true');a.add_argument('--repetitions',type=int,default=2)
    args=a.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    if os.environ.get('OPENBLAS_NUM_THREADS')!='1':raise RuntimeError('fixed NumPyBLAS1 required')
    names=['tools/s43_revisit/s43_field_thread_audit.py','tools/s43_revisit/s43_field_model_reacquire.py',
        'tools/s43_revisit/s43_field_training_R1.py','tools/s43_revisit/s43_field_runtime_R1.py',
        'tools/s43_revisit/field_query.py','tools/s43_revisit/witness_geometry.py']
    identities={str(ROOT/n):sha(ROOT/n) for n in names}
    if args.gpu_only and not args.gpu:raise ValueError('gpu-only requiresgpu')
    torch.set_num_threads(8);cpu={};micro={}
    if not args.gpu_only:
        model,_=build_model('field_static','cpu');model.eval()
        initial={k:tensor_hash(v) for k,v in model.state_dict().items()}
        cpu,micro=cpu_audit(model,args.repetitions)
        assert initial=={k:tensor_hash(v) for k,v in model.state_dict().items()}
    gpu=gpu_audit() if args.gpu else None
    assert all(sha(p)==h for p,h in identities.items())
    out=dict(schema='s43_field_cpu_thread_execution_audit_v1',status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),
        sources=identities,geometry_workers=1,runtime_workers=1,NumPy_BLAS_threads=1,CPU_query=cpu,micro=micro,GPU=gpu,
        model_weights_buffers_unchanged=True,running_sources_or_processes_changed=False,
        optimizer_steps=0,GT_targets_read=False,scientific_scores_read=False,shared_host=True,
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        limitation='CPU execution configuration only; timings do not establish task accuracy or formal service latency')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS',CPU={B:{k:r[k] for k in ('median8_ms','median1_ms','ratio8_over1')} for B,r in cpu.items()},GPU=gpu,out=str(args.out))))


if __name__=='__main__':main()
