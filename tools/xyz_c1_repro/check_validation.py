#!/usr/bin/env python3
"""Verify short validation insertion without hiding its expected CPU-RNG draw."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from compare import differences

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference',required=True)
    ap.add_argument('--validation',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();ref=Path(args.reference).resolve();val=Path(args.validation).resolve()
    errors=[];cpu_rng_differences={};validation=[];checked_files={}
    for rank in range(2):
        cpu_rng_differences[rank]=[]
        for path in(ref/f'indices_rank{rank}.i32',val/f'indices_rank{rank}.i32'):
            checked_files[str(path)]=sha(path)
        a=np.fromfile(ref/f'indices_rank{rank}.i32',dtype=np.int32)
        b=np.fromfile(val/f'indices_rank{rank}.i32',dtype=np.int32)
        if a.size!=50*512 or not np.array_equal(a,b):errors.append(f'indices/rank{rank}')
        for step in range(51):
            pa=ref/f'state_rank{rank}_step{step:06}.json';pb=val/pa.name
            x,y=read(pa),read(pb);checked_files[str(pa)]=sha(pa);checked_files[str(pb)]=sha(pb)
            if x['torch_cpu_rng']!=y['torch_cpu_rng']:cpu_rng_differences[rank].append(step)
            for field in('model','optimizers','schedulers','torch_cuda_rng','input_stream_sha256','batches','step'):
                errors.extend(differences(x[field],y[field],f'rank{rank}/step{step}/{field}'))
        for directory in(ref,val):
            p=directory/f'batches_rank{rank}.jsonl';checked_files[str(p)]=sha(p)
        x=[json.loads(l)for l in(ref/f'batches_rank{rank}.jsonl').read_text().splitlines()]
        y=[json.loads(l)for l in(val/f'batches_rank{rank}.jsonl').read_text().splitlines()]
        if len(x)!=50 or len(y)!=50:errors.append(f'journal_denominator/rank{rank}')
        errors.extend(differences(x,y,f'complete_train_journal/rank{rank}'))
        for step in(25,50):
            p=val/f'validation_rank{rank}_step{step:06}.json';r=read(p);checked_files[str(p)]=sha(p)
            if(r['status'],r['step'],r['batches'],r['samples_with_original_distributed_padding'],r['model_state_unchanged'])!=('PASS',step,126,64094,True):
                errors.append(f'full_validation/rank{rank}/step{step}')
            validation.append(r)
        c=read(val/f'completion_rank{rank}.json')
        if c['status']!='COMPLETE'or c['step']!=50 or c['batches']!=50:errors.append(f'completion/rank{rank}')
        # State snapshots are emitted at train-batch-end, before validation. The
        # first extra validation iterator draw appears in step26's CPU RNG.
        if cpu_rng_differences[rank]!=list(range(26,51)):errors.append(f'unexpected_cpu_rng_difference_steps/rank{rank}')
    report={'status':'PASS'if not errors else'FAIL','reference':str(ref),'validation_run':str(val),
        'scope':'Fresh50 steps only; complete validation at25/50 and original training returns26..50. No optimizer updates in validation.',
        'training_comparison':'All initial and50 model/Adam/scheduler/CUDA-RNG/input states, all50 full raw batches/predictions/gradients/objectives and actual512-index tapes exactly equal.',
        'validation_receipts':validation,'cpu_rng_difference_steps':cpu_rng_differences,
        'cpu_rng_explanation':'Each validation iterator consumes original global CPU RNG for its worker base seed. This expected draw starts after25; epoch0 RandomSampler permutation was already materialized. Formal runs all use the same original500 validation cadence; the epoch-boundary effect must remain part of full6000 comparison.',
        'errors':errors,'checked_files_sha256':checked_files,'comparator_sha256':sha(__file__)}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'errors':errors,'cpu_rng_difference_steps':cpu_rng_differences}))
    return 0 if not errors else 1

if __name__=='__main__':raise SystemExit(main())
