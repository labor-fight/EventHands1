#!/usr/bin/env python3
"""Compare complete saved deterministic preflight trajectories, never scores."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import sys
import numpy as np
import torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def differences(a,b,prefix=''):
    if torch.is_tensor(a)and torch.is_tensor(b):
        return []if a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)else[prefix]
    if type(a)is not type(b):return[prefix+':type']
    if isinstance(a,dict):
        if a.keys()!=b.keys():return[prefix+':keys']
        return [x for k in a for x in differences(a[k],b[k],prefix+'/'+str(k))]
    if isinstance(a,(list,tuple)):
        if len(a)!=len(b):return[prefix+':length']
        return[x for i,(x,y)in enumerate(zip(a,b))for x in differences(x,y,prefix+'/'+str(i))]
    return []if a==b else[prefix]

def compare(paths,output):
    errors=[];reports={};source_hashes={};allstates={};allbatches={};allcheckpoints={}
    for directory in paths:
        path=Path(directory).resolve();tag=path.name
        completions=[read(path/f'completion_rank{rank}.json')for rank in range(2)]
        steps=completions[0]['step']
        if steps not in(20,50)or any(r['step']!=steps or r['batches']!=steps or r['status']!='COMPLETE'for r in completions):
            errors.append(tag+':incomplete_budget')
        runtimes=[read(path/f'effective_runtime_rank{rank}.json')for rank in range(2)]
        for rank,runtime in enumerate(runtimes):
            flags=runtime['actual_math_flags'];loader=runtime['actual_train_loader']
            if not flags['deterministic_algorithms']or flags['warn_only']or flags['cudnn_benchmark']:errors.append(tag+':flags')
            if loader['batch_size']!=512 or runtime['world_size']!=2 or loader['dataset_samples']!=4544300:errors.append(tag+':data_worldsize')
            if(runtime['sampler_policy']=='legacy_none')!=(loader['underlying_generator']is None):errors.append(tag+':actual_generator')
        state={};batches={};tapes={}
        for rank in range(2):
            batches[rank]=[json.loads(line)for line in(path/f'batches_rank{rank}.jsonl').read_text().splitlines()]
            if len(batches[rank])!=steps:errors.append(tag+':batch_denominator')
            tape=np.fromfile(path/f'indices_rank{rank}.i32',dtype=np.int32).reshape(-1,512);tapes[rank]=tape
            if tape.shape!=(steps,512):errors.append(tag+':index_tape_shape')
            for i,row in enumerate(batches[rank]):
                if row['step']!=i+1 or row['batch_size']!=512:errors.append(tag+':batch_step_identity')
                if row['indices_sha256']!=hashlib.sha256(tape[i].tobytes()).hexdigest():errors.append(tag+':index_hash')
                if not row.get('full_packet_sha256')or not row.get('prediction_sha256')or not row.get('gradients'):
                    errors.append(tag+':missing_full_preflight_hashes')
            state[rank]={step:read(path/f'state_rank{rank}_step{step:06}.json')for step in range(steps+1)}
        if len(np.unique(np.concatenate([x.reshape(-1)for x in tapes.values()])))!=2*steps*512:
            errors.append(tag+':duplicate_train_indices_in_first_epoch')
        for step in range(steps+1):
            for field in('model','optimizers','schedulers'):
                if state[0][step][field]!=state[1][step][field]:errors.append(tag+':ranks_disagree:'+str(step)+':'+field)
        allstates[tag]=state;allbatches[tag]=batches
        ckpt=torch.load(path/'repro_final.ckpt',map_location='cpu',weights_only=False)
        allcheckpoints[tag]=ckpt
        if ckpt['global_step']!=steps:errors.append(tag+':checkpoint_step')
        reports[tag]={'directory':str(path),'steps':steps,'completions':completions,
            'actual_runtime':runtimes,'tape_shape_each_rank':[steps,512],
            'full_raw_batches_hashed_each_rank':steps,'initial_state':state[0][0]['model'],
            'final_state':state[0][steps]['model']}
        for file in path.glob('*.json'):source_hashes[str(file)]=sha(file)
        for file in path.glob('*.jsonl'):source_hashes[str(file)]=sha(file)
        for file in path.glob('*.i32'):source_hashes[str(file)]=sha(file)
        source_hashes[str(path/'repro_final.ckpt')]=sha(path/'repro_final.ckpt')
    tags=list(reports);reference=tags[0];comparisons={}
    for tag in tags[1:]:
        diff=[]
        diff+=differences(allstates[reference],allstates[tag],'all_step_state')
        diff+=differences(allbatches[reference],allbatches[tag],'all_step_batches')
        for field in('state_dict','optimizer_states','lr_schedulers','loops','global_step','epoch'):
            diff+=differences(allcheckpoints[reference][field],allcheckpoints[tag][field],'checkpoint/'+field)
        a,b=reports[reference]['actual_runtime'][0],reports[tag]['actual_runtime'][0]
        for field in('actual_math_flags','environment','sampler_policy','wrapper_sha256','original_entry_sha256','torch','lightning','cuda','cudnn','nccl','gpu','world_size','precision'):
            diff+=differences(a[field],b[field],'runtime/'+field)
        # Physical pair and output/config identity are intentionally different.
        comparisons[tag]={'reference':reference,'equal':not diff,'different_paths':diff}
        errors.extend(tag+':'+x for x in diff)
    report={'status':'PASS'if not errors else'FAIL','scope':'Strict engineering comparison of fresh20/50-step runs; no accuracy selection and no claim of full6000 reproduction yet.',
        'runs':reports,'comparisons':comparisons,'errors':errors,'files_sha256':source_hashes,
        'comparator_sha256':sha(__file__),'historical_boundary':'Does not restore missing September weights or prove an accuracy improvement.'}
    Path(output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'runs':tags,'error_count':len(errors),'errors':errors[:20],'receipt':str(output)}))
    return 0 if not errors else 1

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('runs',nargs='+');ap.add_argument('--output',required=True)
    args=ap.parse_args();sys.exit(compare(args.runs,args.output))
