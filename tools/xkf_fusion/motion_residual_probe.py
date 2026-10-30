#!/usr/bin/env python3
"""Offline true-motion versus I-increment count-bin audit; never fits a policy."""
from __future__ import annotations
from collections import Counter
import argparse,json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'model')]
from tools.xkf_fusion.assess import Assessor,CONDITIONS,R2,sha,write

EDGES=(0,50,100,250,500,1000,2000,4000,float('inf'))

def score(pred,true,gate,w):
    p,t=np.asarray(pred,np.float64).reshape(len(w),-1),np.asarray(true,np.float64).reshape(len(w),-1)
    w=w/w.sum();avg=lambda x:float(np.sum(w*x));dot=np.sum(p*t,1);p2=np.sum(p*p,1);t2=np.sum(t*t,1)
    optimum=avg(dot)/max(avg(p2),1e-20)
    return {'true_motion_RMS':avg(t2)**.5,'I_motion_RMS':avg(p2)**.5,
            'raw_I_delta_RMSE':avg(np.sum((p-t)**2,1))**.5,
            'ramped_I_delta_RMSE':avg(np.sum((gate[:,None]*p-t)**2,1))**.5,
            'hold_delta_RMSE':avg(t2)**.5,'oracle_scalar_unclipped':optimum,'oracle_scalar_clipped':float(np.clip(optimum,0,1)),
            'raw_motion_projection_fraction':avg(dot)/max(avg(t2),1e-20),
            'gated_motion_projection_fraction':avg(gate*dot)/max(avg(t2),1e-20),
            'raw_bias':np.sum(w[:,None]*(p-t),0).tolist(),'gated_bias':np.sum(w[:,None]*(gate[:,None]*p-t),0).tolist()}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);study=ROOT/'outputs/xkf/fusion_20261015_r3';base=study/'direct_finetune/motion_gate'
    assessed=json.loads((base/'assessment.json').read_text());assert assessed['status']=='ASSESSED' and not assessed['validation_errors']
    declaration={'kind':'OFFLINE_TRUE_MOTION_RESIDUAL_BY_W50','candidate':'motion_ramp_centroid','seeds':[3407,3408,3409],
        'count_edges':[0,50,100,250,500,1000,2000,4000,'infinity'],'conditions':list(CONDITIONS),'all112_clips_preserved':True,
        'true_motion':'GT current versus previousGT; previousGT is clip clean initialization then truth[t-1], never current predicted prev',
        'predicted_motion':'saved un-gated I proposal versus that candidate own previous output',
        'weighting':'inverse duplicate packet+condition multiplicity in every reported cell',
        'oracle_scope':'diagnostic scalar from developmentGT only; no policy/threshold export, no closed-loop counterfactual claim',
        'source_sha256':{str(Path(__file__).resolve()):sha(__file__),str((base/'assessment.json').resolve()):sha(base/'assessment.json')}}
    write(out/'protocol.json',declaration);checker=Assessor(study,base,True);checker.setup();torch=checker.torch
    def fk(states,beta):
        result=[]
        with torch.no_grad():
            for begin in range(0,len(states),256):
                result.append(checker.fk_model._fk(torch.from_numpy(states[begin:begin+256]),torch.from_numpy(beta[begin:begin+256]))[1].numpy())
        return np.concatenate(result)
    result={'status':'OFFLINE_ONLY','protocol_sha256':sha(out/'protocol.json'),'seeds':{}}
    for seed in (3407,3408,3409):
        directory=base/f's{seed}';protocol=json.loads((directory/'protocol.json').read_text());mi=protocol['modes'].index('motion_ramp_centroid');ii=protocol['motion_modes'].index('motion_ramp_centroid')
        parts={k:[] for k in ('prev','proposal','gt_prev','truth','beta','packet','sequence','condition','short','count')}
        for index,clip in enumerate(protocol['clips']):
            path=directory/'raw'/f'{index:03d}_{clip["cohort"]}_run{clip["run_id"]}.npz';assert sha(path)==assessed['input_sha256'][str(path.resolve())]
            with np.load(R2/f'evaluations/repair_s{seed}_step2200_direct/{clip["sequence"]}_W200.npz') as z:beta=z['beta']
            with np.load(path) as z:
                n=len(z['truth']);prev=np.concatenate([z['initial'][None],z['predictions'][:-1,:,mi]],0)
                gtprev=np.concatenate([z['clean'][None],z['truth'][:-1]],0)
                parts['prev'].append(prev.reshape(-1,51));parts['proposal'].append(z['motion_predictions'][:,:,ii].reshape(-1,51))
                parts['gt_prev'].append(np.repeat(gtprev[:,None],9,1).reshape(-1,51));parts['truth'].append(np.repeat(z['truth'][:,None],9,1).reshape(-1,51))
                parts['beta'].append(np.broadcast_to(beta,(n*9,beta.size)).copy());parts['packet'].append(np.repeat(z['packet_ids'],9))
                parts['sequence'].append(np.repeat(clip['sequence'],n*9));parts['condition'].append(np.tile(np.asarray(CONDITIONS),n))
                parts['short'].append(np.repeat(clip['censored'],n*9));parts['count'].append(np.repeat(z['n_raw_events'],9))
        a={k:np.concatenate(v) for k,v in parts.items()};assert len(a['prev'])==8118
        jp,ji,jgp,jg=(fk(a[k],a['beta']) for k in ('prev','proposal','gt_prev','truth'))
        pr,ir,gpr,gr=(Rotation.from_rotvec(a[k][:,3:6]) for k in ('prev','proposal','gt_prev','truth'))
        deltas={'translation_mm':((a['proposal'][:,:3]-a['prev'][:,:3])*1000,(a['truth'][:,:3]-a['gt_prev'][:,:3])*1000),
                'depth_mm':((a['proposal'][:,2:3]-a['prev'][:,2:3])*1000,(a['truth'][:,2:3]-a['gt_prev'][:,2:3])*1000),
                'root_world_deg':((ir*pr.inv()).as_rotvec()*180/np.pi,(gr*gpr.inv()).as_rotvec()*180/np.pi),
                'centroid_mm':((ji.mean(1)-jp.mean(1))*1000,(jg.mean(1)-jgp.mean(1))*1000),
                'all21_joints_mm':((ji-jp)*1000,(jg-jgp)*1000)}
        np.savez_compressed(out/f's{seed}_motion.npz',**a,**{f'{prefix}_{key}':v for key,pair in deltas.items() for prefix,v in zip(('pred','true'),pair)})
        stats={}
        for sequence in ('pooled','zgz_local','zgz_global'):
            for condition in ('all_conditions',*CONDITIONS):
                for low,high in zip(EDGES[:-1],EDGES[1:]):
                    mask=(a['count']>=low)&(a['count']<high)
                    if sequence!='pooled':mask &= a['sequence']==sequence
                    if condition!='all_conditions':mask &= a['condition']==condition
                    if not mask.any():continue
                    keys=list(zip(a['packet'][mask],a['condition'][mask]));counts=Counter(keys);w=np.array([1/counts[k] for k in keys]);g=np.clip(a['count'][mask]/1000.,0,1)
                    stats[f'{sequence}/{condition}/{low}:{high}']={'rows':int(mask.sum()),'unique_packet_conditions':len(counts),
                        'gate_mean':float(np.average(g,weights=w)),'count_mean':float(np.average(a['count'][mask],weights=w)),
                        'short_rows':int(a['short'][mask].sum()),'metrics':{k:score(p[mask],t[mask],g,w) for k,(p,t) in deltas.items()}}
        result['seeds'][str(seed)]={'raw_sha256':sha(out/f's{seed}_motion.npz'),'statistics':stats};write(out/'result.json',result)
        print(json.dumps({'seed':seed,'rows':len(a['prev']),'cells':len(stats)}),flush=True)
    result['all_three_seeds_complete']=True;write(out/'result.json',result)

if __name__=='__main__':main()
