"""The one registered correction: unverified polarity and score calibration.

All orientation/temperature fitting uses FIT labels only, as supervised readout
calibration. No GT texture/candidate enters the constructor. This does not turn
the assumed radiance provider into a validated physical camera likelihood.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np

from . import geometry as G
from .cache_data import load_cache
from .carrier import GridKernel,EvidenceMemory,logsumexp
from .data import counts_from_events
from .experiment import (OUT,freeze_stage,write_json,load_features,sha,_worker_init)
from .representations import SENSOR,probabilities,temporal_intervention


def dual_evidence(paths,counts):
    # With symmetric thresholds and initial phase, negating model brightness
    # is exactly a whole-epoch ON/OFF interpretation reversal. It preserves
    # all spatial geometry and the observed stored-polarity data.
    dual=np.concatenate([paths,-paths],axis=1)
    kernel=GridKernel(dual,polarity_convention='stored mapping unknown; two global interpretations',**SENSOR)
    memory=EvidenceMemory(kernel); memory.counts[:]=np.minimum(counts,kernel.cap)
    mixed=np.zeros((kernel.K,2)); t0=time.perf_counter()
    for b in range(kernel.B):
        old=memory.ell.copy(); memory.advance((b+1)*1000,watermark_us=(b+1)*1000)
        delta=memory.ell-old
        for orient in range(2):
            mixed[:,orient]+=logsumexp(delta[:,orient*3:(orient+1)*3]-np.log(3),axis=-1)
    elapsed=time.perf_counter()-t0
    ll=np.stack([logsumexp(memory.ell[:,i*3:(i+1)*3]-np.log(3),axis=-1) for i in range(2)],-1)
    return ll,mixed,elapsed,memory.memory_bytes()['total_accounted_bytes']


def _job(task):
    sample,axis=task
    axis_name='tx' if axis=='root_x' else 'index_y'
    amp=(-.015,0,.015) if axis=='root_x' else (-.25,0,.25)
    states=G.candidate_paths(sample['prev_state'],amplitudes=amp,axis=axis_name)
    camera=sample['K_cam'].copy(); camera[:2]*=.375
    t0=time.perf_counter()
    paths=G.render_paths(states,sample['betas'],camera,image_size=(24,32),texture_seed=None)
    render=time.perf_counter()-t0
    counts=counts_from_events(sample['events'],(24,32),cap=8)
    ll,remix,elapsed,mem=dual_evidence(paths,counts)
    seed=int(sample['end_ms'])+sum(map(ord,sample['recording']))
    lt,_,_,_=dual_evidence(paths,temporal_intervention(counts,seed))
    return {'ll':ll,'time':lt,'remix':remix,'seconds':elapsed,'bytes':mem,'render':render}


def make_components(cache,axis,out,workers):
    freeze_stage(out.stem)
    samples=load_cache(cache)
    with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn'),initializer=_worker_init) as pool:
        results=[]
        for i,r in enumerate(pool.map(_job,[(s,axis) for s in samples])):
            results.append(r)
            if i%16==0: print(f'{axis}: {i+1}/{len(samples)}',flush=True)
    if out.exists(): raise FileExistsError(out)
    np.savez_compressed(out,**{k:np.stack([r[k] for r in results]) for k in results[0]})
    write_json(out.with_suffix('.json'),{'source':str(cache),'axis':axis,'samples':len(samples),
               'sha256':sha(out),'GT_input':False,'M':6,'mapping':'global within epoch',
               'model_role':'assumed-model evidence, subsequent FIT calibration is discriminative'})


def mix(ll,weight):
    prior=np.array([1-weight,weight],float)
    lp=np.full(2,-np.inf); np.log(prior,out=lp,where=prior>0)
    return logsumexp(ll+lp,axis=-1)


def fit_calibration(feature_paths,component_paths,out):
    freeze_stage(out.stem); inputs=[]
    for fp,cp in zip(feature_paths,component_paths):
        f=load_features(fp); c=load_features(cp); fit=f['split']=='train'
        inputs.append((c['ll'][fit],f['label'][fit]))
    trials=[]
    for weight in [0.,.25,.5,.75,1.]:
        for temperature in [1.,10.,100.,1000.,10000.,100000.]:
            losses=[]
            for ll,y in inputs:
                prob=probabilities(mix(ll,weight)/temperature)
                losses.extend(-np.log(np.clip(prob[np.arange(len(y)),y],1e-12,1)))
            trials.append({'flip_prior':weight,'temperature':temperature,'FIT_nll':float(np.mean(losses))})
    selected=min(trials,key=lambda r:r['FIT_nll'])
    write_json(out,{'selected':selected,'trials':trials,'fit_samples':sum(len(y) for _,y in inputs),
               'fit_axes':len(inputs),'SELECT_or_zgz_used':False,'correction_round':1,
               'scope':'supervised finite-class score calibration, not sensor calibration',
               'features':[{'path':str(p),'sha256':sha(p)} for p in feature_paths],
               'components':[{'path':str(p),'sha256':sha(p)} for p in component_paths]})


def transform(feature,components,calibration,out):
    import json
    freeze_stage(out.stem)
    data=load_features(feature); comp=load_features(components)
    params=json.loads(calibration.read_text())['selected']; w=params['flip_prior']; t=params['temperature']
    original=mix(comp['ll'],w)/t; changed=mix(comp['time'],w)/t
    data['br']=(original-original.max(-1,keepdims=True)).astype(np.float32)
    data['br_time']=(changed-changed.max(-1,keepdims=True)).astype(np.float32)
    data['br_geometry']=data['br'][:,[2,0,1]]
    data['analytic']=probabilities(original); data['analytic_time']=probabilities(changed)
    data['analytic_geometry']=data['analytic'][:,[2,0,1]]
    data['analytic_remixed']=probabilities(mix(comp['remix'],w)/t)
    data['advance_seconds']=comp['seconds']; data['bytes']=comp['bytes']; data['render_seconds']=comp['render']
    # Kernel/MLP standardization will cancel a global temperature, correctly.
    # Only physical mapping changes their discriminating evidence. Analytic
    # probabilities separately show confidence calibration, without a fake
    # claim that temperature recovers missing state information.
    if out.exists(): raise FileExistsError(out)
    np.savez_compressed(out,**data)
    write_json(out.with_suffix('.json'),{'source':str(feature),'components':str(components),
               'calibration':str(calibration),'calibration_sha256':sha(calibration),
               'sha256':sha(out),'parameters':params,'GT_input':False,
               'no_test_selection':True,'round':1,'simple_arms_unchanged':True})


def main():
    p=argparse.ArgumentParser(description=__doc__); s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('components'); q.add_argument('--cache',type=Path,required=True)
    q.add_argument('--axis',choices=['root_x','index_y'],required=True); q.add_argument('--out',type=Path,required=True)
    q.add_argument('--workers',type=int,default=6)
    q=s.add_parser('fit'); q.add_argument('--features',nargs='+',type=Path,required=True)
    q.add_argument('--components',nargs='+',type=Path,required=True); q.add_argument('--out',type=Path,required=True)
    q=s.add_parser('transform'); q.add_argument('--features',type=Path,required=True)
    q.add_argument('--components',type=Path,required=True); q.add_argument('--calibration',type=Path,required=True)
    q.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.command=='components': make_components(a.cache,a.axis,a.out,a.workers)
    elif a.command=='fit': fit_calibration(a.features,a.components,a.out)
    else: transform(a.features,a.components,a.calibration,a.out)


if __name__=='__main__': main()
