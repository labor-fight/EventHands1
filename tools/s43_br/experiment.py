"""Small B-R experiments, separate from canonical S43 training/evaluation.

Run with the EventHandsTrain Python, CUDA_VISIBLE_DEVICES='', from repo root.
Every output is an opt-in finite-candidate diagnostic, never a main-row alias.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

from . import geometry as G
from .cache_data import load_cache
from .data import counts_from_events
from .readouts import KernelReadout, train_adapted
from .representations import (ARMS, SENSOR, evidence, pixel_features,
                              probabilities, shared_context, temporal_intervention)

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/s43_br_20261027'
H,W=24,32


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(o):
    if isinstance(o,np.ndarray): return o.tolist()
    if isinstance(o,(np.integer,np.floating)): return o.item()
    if isinstance(o,Path): return str(o)
    raise TypeError(type(o).__name__)


def write_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=jsonable)+'\n')


def freeze_stage(name):
    protocol=json.loads((OUT/'protocol.json').read_text())
    if sha(ROOT/protocol['preregistration'])!=protocol['prereg_sha256']:
        raise RuntimeError('preregistration changed')
    path=OUT/'sources'/f'{name}.json'
    write_json(path,{'stage':name,'at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                     'python':sys.executable,'cuda':os.environ.get('CUDA_VISIBLE_DEVICES'),
                     'files':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tools/s43_br').glob('*.py'))},
                     'frozen_production':{str(p):sha(ROOT/p) for p in [
                         Path('model/model.py'),Path('model/mano_layer.py'),Path('semkine/events.py'),
                         Path('semkine/dataset.py'),Path('semkine/eval_track.py'),
                         Path('tools/report_table.py'),Path('assets/mano_right.npz')]}})


def _worker_init():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(1)


def construct(paths, counts, history, candidates, seed, betas, K_cam):
    """No target argument: everything below is a causal representation."""
    t0=time.perf_counter(); lnes,enhanced=pixel_features(counts)
    pixel_seconds=time.perf_counter()-t0
    base=evidence(paths,counts,need_remixed=True)
    shuffled=temporal_intervention(counts,seed)
    time_evidence=evidence(paths,shuffled)
    context=shared_context(history,candidates,paths,betas,K_cam)
    return {'lnes':lnes.ravel(),'enhanced':enhanced.ravel(),'br':base['relative'],
            'context':context,'br_time':time_evidence['relative'],
            'br_geometry':base['relative'][[2,0,1]],
            'analytic':base['posterior'],'analytic_time':probabilities(time_evidence['likelihood']),
            'analytic_geometry':base['posterior'][[2,0,1]],
            'analytic_remixed':probabilities(base['remixed']),
            'count_events':int(np.minimum(counts,SENSOR['cap']).sum()),
            'time_changed_bins':int(np.count_nonzero(shuffled-counts)),
            'pixel_seconds':pixel_seconds,'advance_seconds':base['advance_seconds'],
            'query_seconds':base['query_seconds'],
            'bytes':base['memory']['total_accounted_bytes']}


def _synthetic_job(sample):
    result=construct(sample['paths'],sample['counts'],sample['history'],
                     sample['candidate_states'],int(hashlib.sha256(sample['sample_id'].encode()).hexdigest()[:8],16),
                     sample['betas'],sample['K_cam'])
    # True candidate/target are first used AFTER construction, only for scoring.
    result.update(label=int(sample['target_class']),candidate_joints=sample['candidate_joints'],
                  target_joints=sample['target_joints'],
                  group=str(sample['context_id']),split='validation' if sample['split']=='val' else sample['split'],
                  domain='synthetic',axis=sample['axis'])
    return result


def _existing_job(task):
    sample,axis,revision=task
    axis_name='tx' if axis=='root_x' else 'index_y'
    amplitudes=(-.015,0.,.015) if axis=='root_x' else (-.25,0.,.25)
    states=G.candidate_paths(sample['prev_state'],amplitudes=amplitudes,axis=axis_name)
    K=sample['K_cam'].copy(); K[:2]*=.375
    t0=time.perf_counter()
    paths=G.render_paths(states,sample['betas'],K,image_size=(H,W),texture_seed=None)
    render_seconds=time.perf_counter()-t0
    counts=counts_from_events(sample['events'],(H,W),cap=SENSOR['cap'])
    result=construct(paths,counts,sample['prev_state'],states,
                     int(sample['end_ms'])+sum(map(ord,sample['recording'])),sample['betas'],K)
    # Last target is 1ms before the observation cutoff under the old convention.
    candidates=G.fk_joints(states[:,49],sample['betas'])
    target=G.fk_joints(sample['target'],sample['betas'])
    errors=np.linalg.norm(candidates-target,axis=-1).mean(-1)
    result.update(label=int(errors.argmin()),candidate_joints=candidates,
                  target_joints=target,group=sample['recording'],
                  split='train' if sample['partition']=='FIT7' else
                        'validation' if sample['partition']=='SELECT' else 'test',
                  domain=sample['domain'],axis=axis,render_seconds=render_seconds,
                  GT_input=False,history_source=sample['history_source'])
    return result


def pack(results,path):
    arrays={k:np.stack([r[k] for r in results]) for k in [
        'lnes','enhanced','br','context','br_time','br_geometry','analytic',
        'analytic_time','analytic_geometry','analytic_remixed','candidate_joints','target_joints']}
    for key in ['label','group','split','domain','axis']:
        arrays[key]=np.asarray([r[key] for r in results])
    for key in ['count_events','time_changed_bins','pixel_seconds','advance_seconds','query_seconds','bytes','render_seconds']:
        arrays[key]=np.asarray([r.get(key,0) for r in results])
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    np.savez_compressed(path,**arrays)
    write_json(path.with_suffix('.json'),{'samples':len(results),'sha256':sha(path),
               'keys':{k:list(v.shape) for k,v in arrays.items()},'GT_input':False,
               'protocol':str(OUT/'protocol.json'),'source_stage':sys.argv,
               'scope':'finite candidate diagnostic; not recurrent S43 main-row'})


def load_features(path):
    with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in z.files}


def metrics(probs,labels,candidates,target):
    probs=np.clip(np.asarray(probs,float),1e-12,1.)
    probs=probs/probs.sum(1,keepdims=True)
    choice=probs.argmax(1); picked=candidates[np.arange(len(choice)),choice]
    errors=np.linalg.norm(picked-target,axis=-1).mean(-1)*1000
    picked_ra=picked-picked[:,:1]; target_ra=target-target[:,:1]
    ra=np.linalg.norm(picked_ra-target_ra,axis=-1).mean(-1)*1000
    oracle=np.linalg.norm(candidates-target[:,None],axis=-1).mean(-1).min(-1)*1000
    return {'accuracy':float(np.mean(choice==labels)),
            'nll':float(-np.log(probs[np.arange(len(labels)),labels]).mean()),
            'brier':float(((probs-np.eye(3)[labels])**2).sum(1).mean()),
            'entropy':float(-(probs*np.log(probs)).sum(1).mean()),
            'absolute_joint_mm':float(errors.mean()),'root_aligned_joint_mm':float(ra.mean()),
            'oracle_candidate_mm':float(oracle.mean()),'n':len(labels),
            'per_sample_absolute_mm':errors,'per_sample_accuracy':(choice==labels).astype(float),
            'class_histogram':np.bincount(labels,minlength=3).tolist()}


def grouped_difference(a,b,groups,seed=3407):
    """Difference a-b; recording/context, never overlapping windows, is unit."""
    unique=np.unique(groups)
    values=np.array([np.mean(np.asarray(a)[groups==g]-np.asarray(b)[groups==g]) for g in unique])
    rng=np.random.default_rng(seed)
    draws=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(1)
    return {'mean':float(values.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),
            'independent_groups':len(values),'unit':'context' if 'context_' in str(unique[0]) else 'recording'}


def evaluate(path,out,*,steps=400,train_path=None,eval_split='validation'):
    data=load_features(path)
    train=data if train_path is None else load_features(train_path)
    trainmask=train['split']=='train'
    evalmask=(data['split']!='train') if train_path is None else np.ones(len(data['label']),bool)
    if eval_split is not None:
        evalmask &= data['split']==eval_split
    if not trainmask.any() or not evalmask.any(): raise ValueError('missing split')
    contexts=train['context'][trainmask]; labels=train['label'][trainmask]
    y=data['label'][evalmask]; c=data['context'][evalmask]
    cj=data['candidate_joints'][evalmask]; tj=data['target_joints'][evalmask]
    report={'source':str(path),'train_source':str(train_path or path),'steps':steps,
            'train_samples':int(trainmask.sum()),'eval_samples':len(y),
            'GT_input':False,'scope':'heldout finite-candidate task, not full recurrent tracking',
            'common':{},'adapted':{},'analytic':{},'interventions':{},'contrasts':{},
            'readout_receipts':{},'prior_controls':{}}
    common_predictions={}; adapted_predictions={}
    eval_slices={name:np.flatnonzero(data['split'][evalmask]==name)
                 for name in np.unique(data['split'][evalmask])}
    context_model=KernelReadout().fit(np.zeros((len(labels),1)),contexts,labels)
    context_probs=context_model.predict(np.zeros((len(y),1)),c)
    majority=(np.bincount(labels,minlength=3)+1.)/(len(labels)+3.)
    for split,ix in eval_slices.items():
        report['prior_controls'][split]={
            'context_only':metrics(context_probs[ix],y[ix],cj[ix],tj[ix]),
            'class_prior_only':metrics(np.broadcast_to(majority,(len(ix),3)),y[ix],cj[ix],tj[ix])}
    for arm in ARMS:
        x=train[arm][trainmask]; ev=data[arm][evalmask]
        model=KernelReadout().fit(x,contexts,labels)
        probs=model.predict(ev,c); common_predictions[arm]=probs
        report['common'][arm]={split:metrics(probs[ix],y[ix],cj[ix],tj[ix]) for split,ix in eval_slices.items()}
        all_ev=[ev]
        if arm=='br': all_ev.extend([data['br_time'][evalmask],data['br_geometry'][evalmask]])
        adapted,receipt=train_adapted(x,contexts,labels,np.concatenate(all_ev),
            np.concatenate([c]*len(all_ev)),arm=arm,steps=steps)
        adapted_predictions[arm]=adapted[:len(y)]
        report['adapted'][arm]={split:metrics(adapted[:len(y)][ix],y[ix],cj[ix],tj[ix]) for split,ix in eval_slices.items()}
        report['readout_receipts'][arm]={'kernel_train_accuracy':model.train_accuracy,
                'kernel_fit_seconds':model.fit_seconds,'kernel_bandwidth':model.bandwidth,'adapted':receipt}
        if arm=='br':
            for index,intervention in enumerate(['time','geometry']):
                changed=data[f'br_{intervention}'][evalmask]
                cp=model.predict(changed,c); ap=adapted[(index+1)*len(y):(index+2)*len(y)]
                report['interventions'][intervention]={
                    'common':{s:metrics(cp[ix],y[ix],cj[ix],tj[ix]) for s,ix in eval_slices.items()},
                    'adapted':{s:metrics(ap[ix],y[ix],cj[ix],tj[ix]) for s,ix in eval_slices.items()}}
    for kind in ['analytic','analytic_time','analytic_geometry','analytic_remixed']:
        probs=data[kind][evalmask]
        report['analytic'][kind]={s:metrics(probs[ix],y[ix],cj[ix],tj[ix]) for s,ix in eval_slices.items()}
    for split,ix in eval_slices.items():
        g=data['group'][evalmask][ix]
        report['contrasts'][split]={}
        for kind,predictions in [('common',common_predictions),('adapted',adapted_predictions)]:
            a=metrics(predictions['br'][ix],y[ix],cj[ix],tj[ix])
            b=metrics(predictions['enhanced'][ix],y[ix],cj[ix],tj[ix])
            report['contrasts'][split][kind]={
                'accuracy_br_minus_enhanced':grouped_difference(a['per_sample_accuracy'],b['per_sample_accuracy'],g),
                'absolute_mm_br_minus_enhanced':grouped_difference(a['per_sample_absolute_mm'],b['per_sample_absolute_mm'],g)}
    report['costs']={k:{'median':float(np.median(data[k])), 'p95':float(np.quantile(data[k],.95))}
                     for k in ['pixel_seconds','advance_seconds','query_seconds','bytes','render_seconds']}
    report['time_intervention']={'changed_bin_entries':int(data['time_changed_bins'].sum()),
                                'protected_lnes_enhanced_exact':True}
    write_json(out,report)
    print(json.dumps({'report':str(out),'train':int(trainmask.sum()),'eval':len(y)},ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('features'); s.add_argument('--kind',choices=['synthetic','existing'],required=True)
    s.add_argument('--source',type=Path,required=True); s.add_argument('--out',type=Path,required=True)
    s.add_argument('--axis',choices=['root_x','index_y'],default='root_x'); s.add_argument('--workers',type=int,default=6)
    s=sub.add_parser('evaluate'); s.add_argument('--source',type=Path,required=True)
    s.add_argument('--out',type=Path,required=True); s.add_argument('--steps',type=int,default=400)
    s.add_argument('--train-source',type=Path)
    s.add_argument('--eval-split',choices=['validation','test'],default='validation')
    args=p.parse_args(); os.environ['CUDA_VISIBLE_DEVICES']=''
    freeze_stage(args.out.stem)
    if args.command=='features':
        if args.kind=='synthetic':
            from .synthetic import load_synthetic
            samples=load_synthetic(args.source,axis=args.axis)
            jobs=samples; func=_synthetic_job
        else:
            samples=load_cache(args.source); jobs=[(s,args.axis,0) for s in samples]; func=_existing_job
        t0=time.perf_counter()
        with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=_worker_init) as pool:
            results=[]
            for i,r in enumerate(pool.map(func,jobs,chunksize=1)):
                results.append(r)
                if i%12==0: print(f'{args.kind}/{args.axis}: {i+1}/{len(jobs)} ({time.perf_counter()-t0:.1f}s)',flush=True)
        pack(results,args.out)
        print(f'complete {args.out} in {time.perf_counter()-t0:.1f}s',flush=True)
    else:
        evaluate(args.source,args.out,steps=args.steps,train_path=args.train_source,eval_split=args.eval_split)


if __name__=='__main__': main()
