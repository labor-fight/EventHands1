"""Target-free shared-host primitive audit; no optimizer or task scoring."""
from __future__ import annotations
import argparse,gc,hashlib,json,os,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from .field_core import TimeFieldMemory
from .field_tensor import tensorize as old
from .field_tensor_fast import tensorize as snapshot_fast,tensorize_memories,aggregate_memories
from .s43_field_thread_audit import conditions,tensor_hash
from tools.s43_rel.mean_shape_replay import MeanShapeInputReader,InferenceAccessGuard

ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def memory_hash(m):
    s=m.state_dict();h=hashlib.sha256(repr(s['scalars']).encode())
    for a in s['arrays'].values():h.update(a.tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--model',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    if os.environ.get('OPENBLAS_NUM_THREADS')!='1':raise ValueError('NumPyBLAS1 required')
    files=['field_tensor_fast.py','field_tensor_fast_engineering.py','field_core.py','field_tensor.py',
           's43_field_model_reacquire.py','s43_field_runtime_R1.py','s43_field_training_R1.py','s43_field_model.py']
    source={str(ROOT/'tools/s43_revisit'/x):sha(ROOT/'tools/s43_revisit'/x) for x in files}
    freeze=dict(schema='s43_field_fast_tensor_engineering_sourcefreeze_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        code=source,device=a.device,CPU_threads=8,NumPy_BLAS_threads=1,task_scores_read=False,optimizer_steps=0,
        workload='actual ch_local0:150ms, independentB64 completed50-ms rings; public beta0/initialconditions; old-fast-old-fast')
    write(a.out/'sourcefreeze.json',freeze)
    sync=(lambda:torch.cuda.synchronize()) if torch.device(a.device).type=='cuda' else lambda:None
    with InferenceAccessGuard() as guard:
        reader=MeanShapeInputReader('ch_local');packets=[reader.packet(i,i+50) for i in (0,50,100)]
        source_bytes=hashlib.sha256(b''.join(x['events'].tobytes() for x in packets)).hexdigest()
        memories=[TimeFieldMemory(seq_id='fast-engineering:'+str(i)) for i in range(64)]
        for m in memories:
            for j,packet in enumerate(packets):m.ingest(packet['events'],(j+1)*50,j,source_range=packet['source_range'])
        before=[memory_hash(m) for m in memories];rng=torch.get_rng_state().clone()
        def run(mode,ms):
            if mode=='old':return old([m.readout() for m in ms],a.device)
            if mode=='snapshot_fast':return snapshot_fast([m.readout() for m in ms],a.device)
            return tensorize_memories(ms,device=a.device)
        rows=[];identities={};reference=None
        # One warmup per implementation, excluded; no scientific weights fit.
        for mode in ('old','snapshot_fast','memory_direct'):
            value=run(mode,memories);sync();del value
        for label,mode in (('old_a','old'),('direct_a','memory_direct'),('old_b','old'),('direct_b','memory_direct'),('snapshot_fast','snapshot_fast')):
            sync();start=time.perf_counter();cp=time.process_time();f=run(mode,memories);sync()
            wall=(time.perf_counter()-start)*1000;CPU=(time.process_time()-cp)*1000
            identity={n:tensor_hash(getattr(f,n)) for n in ('global40','cell3','observed_support')}
            if reference is None:reference=identity
            assert identity==reference
            rows.append(dict(label=label,mode=mode,wall_ms=wall,process_CPU_ms=CPU,exact=True,
                count_payload_bytes=f.count_payload_bytes,grid_bytes=f.reference_grid_bytes))
            identities[label]=identity;del f
        model_rows=[]
        if a.model:
            from .s43_field_training_R1 import build_model
            model,_=build_model('field_static',a.device);model.eval();model.requires_grad_(False)
            model_before={k:tensor_hash(v) for k,v in model.state_dict().items()}
            prev,beta,_,refs,_,hist=conditions(64);K=torch.from_numpy(reader.K_cam)[None].repeat(64,1,1)
            # Currentstate is a public fixed condition, not target or learned
            # feedback; this isolated forward checks equality, not tracking.
            prev=prev.to(a.device);beta=beta.to(a.device);K=K.to(a.device)
            refs=np.full(64,100,np.int64);cutoffs=np.full(64,150,np.int64)
            pred_ref=None
            with torch.inference_mode():
                f=run('memory_direct',memories)
                model(f.global40,prev,beta_input=beta,K_source=K,ref_ms=refs,cutoff_ms=cutoffs,histories=hist);sync();del f
                for label,mode in (('old_a','old'),('direct_a','memory_direct'),('old_b','old'),('direct_b','memory_direct')):
                    sync();start=time.perf_counter();f=run(mode,memories);sync();middle=time.perf_counter()
                    pred=model(f.global40,prev,beta_input=beta,K_source=K,ref_ms=refs,cutoff_ms=cutoffs,histories=hist);sync();finish=time.perf_counter()
                    pred=pred.detach().cpu()
                    if pred_ref is None:pred_ref=pred.clone()
                    diff=float((pred-pred_ref).abs().max());assert diff==0 and torch.isfinite(pred).all()
                    model_rows.append(dict(label=label,mode=mode,readout_ms=(middle-start)*1000,
                        query_ms=(finish-middle)*1000,total_ms=(finish-start)*1000,maxpred_vs_first=diff,
                        prediction_sha256=tensor_hash(pred)))
                    del f,pred
            assert model_before=={k:tensor_hash(v) for k,v in model.state_dict().items()}
            model.close_geometry_workers();del model
        assert before==[memory_hash(m) for m in memories]
        # Models are freshly initialized with a deliberate fixed seed; record
        # RNG preservation of readout separately from model construction.
        readout_rng_unchanged=torch.equal(rng,torch.get_rng_state()) if not a.model else None
        reader_identity=dict(recording='ch_local',native_range_ms=[0,150],events_per_lane=sum(len(x['events']) for x in packets),
            source_ranges=[list(x['source_range']) for x in packets],native_xytp_sha256=source_bytes,
            source_files={str(x):dict(bytes=x.stat().st_size,mtime_ns=x.stat().st_mtime_ns) for x in reader.input_paths()})
    assert all(sha(path)==digest for path,digest in source.items())
    result=dict(schema='s43_field_fast_tensor_engineering_v1',status='PASS',sourcefreeze_sha256=sha(a.out/'sourcefreeze.json'),
        scope='B64 integer readout+Torch readout; optional same public-context R1 forward; NOT complete ingest/service/training latency',
        shared_host=True,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),reader=reader_identity,
        rows=rows,identities=identities,model_rows=model_rows,inference_access_guard=guard.receipt(),
        state_source_unchanged=True,readout_RNG_unchanged=readout_rng_unchanged,frozen_sources_unchanged=True,
        task_scores_read=False,optimizer_steps=0,limitations='No ownfeedback stream or formal throughput claim; sourcechecks and memoryexport outside clocks; originalreadout audit counters increment only by old paths')
    write(a.out/'result.json',result);print(json.dumps(dict(status='PASS',rows=rows,model_rows=model_rows)))


if __name__=='__main__':main()
