#!/usr/bin/env python3
"""Compare a shorter guard/resource smoke with a frozen longer run's prefix."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from compare import differences

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference',required=True)
    ap.add_argument('--smoke',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();ref=Path(args.reference).resolve();smoke=Path(args.smoke).resolve()
    errors=[];files={};runtimes=[];completed=[]
    for rank in range(2):
        c=read(smoke/f'completion_rank{rank}.json');steps=c['step'];completed.append(c)
        if c['status']!='COMPLETE'or c['batches']!=steps or steps not in(20,50):errors.append(f'budget/rank{rank}')
        x=np.fromfile(ref/f'indices_rank{rank}.i32',dtype=np.int32).reshape(-1,512)[:steps]
        y=np.fromfile(smoke/f'indices_rank{rank}.i32',dtype=np.int32).reshape(-1,512)
        if x.shape!=(steps,512)or not np.array_equal(x,y):errors.append(f'indices/rank{rank}')
        for step in range(steps+1):
            a=ref/f'state_rank{rank}_step{step:06}.json';b=smoke/a.name
            errors.extend(differences(read(a),read(b),f'complete_state/rank{rank}/step{step}'))
            files[str(a)]=sha(a);files[str(b)]=sha(b)
        x=[json.loads(l)for l in(ref/f'batches_rank{rank}.jsonl').read_text().splitlines()][:steps]
        y=[json.loads(l)for l in(smoke/f'batches_rank{rank}.jsonl').read_text().splitlines()]
        if len(x)!=steps or len(y)!=steps:errors.append(f'journal_denominator/rank{rank}')
        errors.extend(differences(x,y,f'complete_journal/rank{rank}'))
        a=read(ref/f'effective_runtime_rank{rank}.json');b=read(smoke/f'effective_runtime_rank{rank}.json')
        for key in('actual_math_flags','sampler_policy','original_entry_sha256','torch','lightning','cuda','cudnn','nccl','gpu','precision','world_size'):
            errors.extend(differences(a[key],b[key],f'runtime/{key}'))
        loader=b['actual_train_loader']
        if loader['num_workers']!=12 or loader['prefetch_factor']!=1 or loader['batch_size']!=512:
            errors.append(f'actual_smoke_resource/rank{rank}')
        if not(loader['underlying_generator']is None and loader['loader_generator']is None):errors.append(f'actual_generator/rank{rank}')
        if b['effective_seed']!=3407 or b['effective_max_steps']!=steps or b['effective_accumulate_grad_batches']!=1 or b['epoch_batches']!=4437:
            errors.append(f'effective_runtime/rank{rank}')
        runtimes.append(b)
        for directory in(ref,smoke):
            for name in(f'indices_rank{rank}.i32',f'batches_rank{rank}.jsonl',f'effective_runtime_rank{rank}.json'):
                files[str(directory/name)]=sha(directory/name)
    report={'status':'PASS'if not errors else'FAIL','reference':str(ref),'smoke':str(smoke),
        'scope':'Final wrapper guard and workers12/prefetch1 smoke agrees exactly with frozen v2 workers2/prefetch1 prefix; no6000-step claim.',
        'exact_fields':'Every initial and20 complete model/Adam/schedule/CPU+CUDA RNG/rolling-input state, every actual512-index/fullraw/prediction/allgrad/objective hash.',
        'resource_boundary':'workers and maxstep/output identity differ intentionally; raw inputs and numerical trajectory do not.',
        'actual_runtime':runtimes,'completions':completed,'errors':errors,'checked_files_sha256':files,'comparator_sha256':sha(__file__)}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'errors':errors[:20],'error_count':len(errors)}))
    return 0 if not errors else 1

if __name__=='__main__':raise SystemExit(main())
