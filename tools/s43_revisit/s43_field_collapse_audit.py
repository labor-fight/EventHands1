"""Target-free engineering of r2 native-cell conservative time collapse.

Compare the current full-grid reference with an explicit block-constant r2
candidate. This tool neither fits nor reads task scores. GPU timings are
isolated primitives on a shared host, not a deployment benchmark.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime,timezone
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch

from .field_core import TimeFieldMemory
from .field_tensor import tensorize
from .s43_field_runtime import temporal_collapse_field
from tools.s43_rel.mean_shape_replay import MeanShapeInputReader,InferenceAccessGuard

ROOT=Path(__file__).resolve().parents[2]


def assert_r2_blocks(field):
    if field.dtype!=torch.float32 or field.ndim!=4 or field.shape[1:]!=(40,180,240):
        raise ValueError('r2 candidate requires float32 [B,40,180,240]')
    for a,b,r in ((0,16,2),(24,40,4)):
        values=field[:,a:b]
        expanded=values[:,:,::r,::r][:,:,:,None,:,None].expand(-1,-1,-1,r,-1,r)
        if not torch.equal(values.reshape(-1,16,180//r,r,240//r,r),expanded):
            raise ValueError('r2 F2px/Q4px nearest-block contract violated; use full-grid reference')


def native_candidate(field):
    """Same exact formula on F2px/Q4px cells, strict full block validation."""
    assert_r2_blocks(field)
    output=field.clone()
    widths=torch.tensor([7,6,6,6,7,6,6,6],device=field.device,dtype=torch.float64)
    for a,b,r in ((0,16,2),(24,40,4)):
        native=field[:,a:b,::r,::r].double().reshape(-1,8,2,180//r,240//r)
        total=torch.expm1(native).sum(1,keepdim=True)
        transformed=torch.log1p(total*widths[None,:,None,None,None]/50.)
        output[:,a:b]=transformed.flatten(1,2).to(field.dtype).repeat_interleave(r,-2).repeat_interleave(r,-1)
    return output


def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _write(p,r):Path(p).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def inputs():
    with InferenceAccessGuard() as guard:
        reader=MeanShapeInputReader('ch_local');memory=TimeFieldMemory(seq_id='collapse_engineering')
        snapshots=[];names=[];count=0
        for ms in range(200):
            p=reader.packet(ms,ms+1);count+=len(p['events'])
            memory.ingest(p['events'],ms+1,ms,source_range=p['source_range'])
            if ms+1 in (7,50,100,150,200):snapshots.append(memory.readout(ms+1));names.append('real_partial7' if ms==6 else f'real_{ms+1}')
        empty=TimeFieldMemory(seq_id='empty')
        empty.ingest(np.empty((0,4),np.int64),50,0,source_range=(0,0))
        e=empty.readout(50);snapshots.append(e);names.append('empty_observed50')
        high=e.counts.copy();high[0,10,11,0]=2**30+1;high[49,10,11,0]=2**32-1;high[25,10,11,1]=2**29+3
        snapshots.append(replace(e,counts=high));names.append('fabricated_high_valid_uint32')
    return snapshots,names,count,guard.receipt()


def compare(field):
    reference=temporal_collapse_field(field);candidate=native_candidate(field)
    installed=temporal_collapse_field(field,native_r2=True)
    assert torch.equal(installed,candidate)
    torch.testing.assert_close(candidate,reference,atol=3e-7,rtol=2e-7)
    assert torch.equal(candidate[:,16:24],field[:,16:24])
    assert_r2_blocks(candidate)
    masses={}
    for a,b,label in ((0,16,'F'),(24,40,'Q')):
        before=torch.expm1(field[:,a:b].double()).reshape(-1,8,2,180,240).sum(1)
        after=torch.expm1(candidate[:,a:b].double()).reshape(-1,8,2,180,240).sum(1)
        relative=((after-before).abs()/before.abs().clamp(min=1)).amax()
        if float(relative)>3e-6:raise AssertionError('float32-output count conservation exceeded3e-6')
        masses[label]=dict(max_abs=float((after-before).abs().amax()),max_relative=float(relative),
            definition='per native-image pixel/polarity sum of expm1(float32 logfield); output quantization included')
    return dict(reference_max_abs=float((candidate-reference).abs().amax()),reference_bitwise=torch.equal(candidate,reference),
        runtime_native_matches_candidate_bitwise=True,
        coverage_bitwise=True,output_blocks_bitwise=True,mass=masses)


def timing(field,repetitions):
    rows={}
    for name,fn in (('full_grid_reference',temporal_collapse_field),
                    ('native_cells_guarded',lambda x:temporal_collapse_field(x,native_r2=True))):
        for _ in range(3):y=fn(field)
        torch.cuda.synchronize();del y
        torch.cuda.reset_peak_memory_stats();wall=[];gpu=[]
        for _ in range(repetitions):
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize();t=time.perf_counter();begin.record();value=fn(field);end.record();torch.cuda.synchronize()
            wall.append((time.perf_counter()-t)*1000);gpu.append(begin.elapsed_time(end));del value
        rows[name]=dict(wall_ms=wall,cuda_event_ms=gpu,wall_median_ms=float(np.median(wall)),
            cuda_median_ms=float(np.median(gpu)),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
    rows['wall_median_ratio_reference_over_candidate']=rows['full_grid_reference']['wall_median_ms']/rows['native_cells_guarded']['wall_median_ms']
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu',action='store_true');p.add_argument('--repetitions',type=int,default=7)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    source={str(Path(__file__).resolve()):_sha(__file__),str(ROOT/'tools/s43_revisit/s43_field_runtime.py'):_sha(ROOT/'tools/s43_revisit/s43_field_runtime.py')}
    snapshots,names,count,access=inputs();cpu=tensorize(snapshots).global40
    checks={name:compare(cpu[i:i+1]) for i,name in enumerate(names)}
    random=torch.rand(1,40,180,240)
    try:native_candidate(random)
    except ValueError:random_rejected=True
    else:raise AssertionError('nonblock random field silently accepted')
    gpu_checks={};times={}
    if args.gpu:
        for i,name in enumerate(names):gpu_checks[name]=compare(tensorize([snapshots[i]],'cuda:0').global40)
        for B in (8,64):
            selected=[snapshots[i%len(snapshots)] for i in range(B)]
            field=tensorize(selected,'cuda:0').global40
            times[str(B)]=timing(field,args.repetitions)
            del field;torch.cuda.empty_cache()
    assert all(_sha(path)==value for path,value in source.items())
    result=dict(schema='s43_field_collapse_primitive_engineering_v1',status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=source,source_events=count,input_guard=access,CPU=checks,GPU=gpu_checks,timing=times,
        random_nonblock_rejected=random_rejected,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        shared_host=True,optimizer_steps=0,task_scores_read=False,
        cost_scope='collapse only incl strict block checks+output clone+nearest expansion; excludes core/render/CNN/H2D/whole pipeline',
        mass_note='Exact conservative formula; float32 input/output logs have intrinsic count roundoff, highcount stress reports it explicitly',
        conclusion_scope='finite engineering fixtures and primitive timings; not a formal runtime speed or scientific-effect claim')
    _write(args.out,result)
    print(json.dumps(dict(status='PASS',CPU_cases=len(checks),GPU_cases=len(gpu_checks),timing=times,out=str(args.out))))


if __name__=='__main__':main()
