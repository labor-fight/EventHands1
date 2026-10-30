#!/usr/bin/env python3
"""Offline residual audit only; never exports a deployment gain or runs a new loop."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'model')]
from tools.xkf_fusion.assess import Assessor,CONDITIONS,R2,sha,write

CANDIDATE='uncalibrated_centroid_t05_r025_f05_ramp'

def moments(ei,ed,g,weights):
    ei,ed=np.asarray(ei,np.float64).reshape(len(g),-1),np.asarray(ed,np.float64).reshape(len(g),-1)
    weights=np.asarray(weights,np.float64);weights=weights/weights.sum()
    avg=lambda v:float(np.sum(weights*v))
    vi,vd,c=avg(np.sum(ei*ei,1)),avg(np.sum(ed*ed,1)),avg(np.sum(ei*ed,1))
    innovation=ed-ei;den=avg(np.sum(innovation**2,1))
    k=(vi-c)/max(den,1e-20)
    gate_den=avg(g*g*np.sum(innovation**2,1));kg=-avg(g*np.sum(ei*innovation,1))/max(gate_den,1e-20)
    bias_i=np.sum(weights[:,None]*ei,0);bias_d=np.sum(weights[:,None]*ed,0)
    return dict(V_I=vi,V_D=vd,C_ID=c,I_RMSE=vi**.5,D_RMSE=vd**.5,
                I_bias=bias_i.tolist(),D_bias=bias_d.tolist(),
                I_bias_squared=float(bias_i@bias_i),D_bias_squared=float(bias_d@bias_d),
                oracle_diagnostic_K_unclipped=k,oracle_diagnostic_K_clipped=float(np.clip(k,0,1)),
                oracle_diagnostic_baseK_with_ramp_unclipped=kg,
                oracle_diagnostic_baseK_with_ramp_clipped=float(np.clip(kg,0,1)),
                normalized_uncentered_cross=c/max((vi*vd)**.5,1e-20))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    study=ROOT/'outputs/xkf/fusion_20261015_r3';base=study/'direct_finetune'
    assessed_path=base/'assessment_raw.json';assessed=json.loads(assessed_path.read_text())
    if assessed['status']!='ASSESSED' or assessed['validation_errors'] or assessed['missing']:raise ValueError('requires complete raw assessment')
    declaration={'kind':'OFFLINE_RESIDUAL_DIAGNOSIS_NO_CLOSED_LOOP','candidate':CANDIDATE,
        'seeds':[3407,3408,3409],'clips':112,'conditions':list(CONDITIONS),'all_rows_preserved':True,
        'weighting':'within every condition/cohort/scope, inverse multiplicity of physical packet_id; all-condition pools equal each condition',
        'coefficient_scope':'GT-derived diagnostic optima only; no exported policy; no closed-loop claim',
        'rigid_metric':'common scalar over centroid plus world root tangent projected through the current predicted I centered21-joint geometry, first-order only',
        'formula':'K=-E[eI dot (eD-eI)]/E[||eD-eI||^2]; ramp-baseK=-E[g eI dot innovation]/E[g^2||innovation||^2]',
        'training_control':'same new2400 train2304 D versus its checkpoint frozen2200 EMA VD; populations differ, not a causal ratio',
        'source_sha256':{str(Path(__file__).resolve()):sha(__file__),str(assessed_path):sha(assessed_path)}}
    write(out/'protocol.json',declaration)
    checker=Assessor(study,base/'evaluations/raw',True);checker.setup();torch=checker.torch
    def fk(states,beta):
        values=[]
        with torch.no_grad():
            for begin in range(0,len(states),256):
                x=torch.from_numpy(states[begin:begin+256]).float()
                b=torch.from_numpy(beta[begin:begin+256]).float()
                values.append(checker.fk_model._fk(x,b)[1].numpy())
        return np.concatenate(values)
    result={'protocol_sha256':sha(out/'protocol.json'),'status':'OFFLINE_ONLY','seeds':{}}
    for seed in (3407,3408,3409):
        directory=base/'evaluations/raw'/f's{seed}'
        protocol=json.loads((directory/'protocol.json').read_text());summary=json.loads((directory/'summary.json').read_text())
        ci=protocol['modes'][6:].index(CANDIDATE);mi=protocol['motion_modes'].index(CANDIDATE)
        parts={k:[] for k in ('i','d','y','beta','packet','cohort','condition','short','gate','current_gain')}
        for index,clip in enumerate(protocol['clips']):
            path=directory/'raw'/f'{index:03d}_{clip["cohort"]}_run{clip["run_id"]}.npz'
            if sha(path)!=assessed['input_sha256'][str(path.resolve())]:raise ValueError('assessed raw changed')
            beta_path=R2/f'evaluations/repair_s{seed}_step2200_direct/{clip["sequence"]}_W200.npz'
            with np.load(beta_path) as data:beta=data['beta'].copy()
            with np.load(path,allow_pickle=False) as data:
                n=len(data['truth']);parts['i'].append(data['motion_predictions'][:,:,mi].reshape(-1,51))
                parts['d'].append(np.repeat(data['corrected_direct'][:,ci,None,:],9,1).reshape(-1,51))
                parts['y'].append(np.repeat(data['truth'][:,None,:],9,1).reshape(-1,51))
                parts['beta'].append(np.broadcast_to(beta,(n*9,len(beta.reshape(-1)))).copy())
                parts['packet'].append(np.repeat(data['packet_ids'],9));parts['cohort'].append(np.repeat(clip['cohort'],n*9))
                parts['condition'].append(np.tile(np.asarray(CONDITIONS),n));parts['short'].append(np.repeat(clip['censored'],n*9))
                parts['gate'].append(np.repeat(np.clip(data['n_raw_events']/1000.,0,1),9))
                parts['current_gain'].append(np.repeat(data['effective_gains'][:,ci,None,:],9,1).reshape(-1,19))
        a={k:np.concatenate(v) for k,v in parts.items()};assert len(a['i'])==902*9
        ji,jd,jy=(fk(a[k],a['beta']) for k in ('i','d','y'))
        ri,rd,ry=(Rotation.from_rotvec(a[k][:,3:6]) for k in ('i','d','y'))
        eri,erd=(ri*ry.inv()).as_rotvec(),(rd*ry.inv()).as_rotvec()
        cent_i,cent_d,cent_y=(j.mean(1) for j in (ji,jd,jy))
        pc=ji-cent_i[:,None,:]
        rigid_i=cent_i[:,None,:]-cent_y[:,None,:]+np.cross(eri[:,None,:],pc)
        rigid_d=cent_d[:,None,:]-cent_y[:,None,:]+np.cross(erd[:,None,:],pc)
        errors={
            **{f'state_t_{axis}_mm':((a['i'][:,n:n+1]-a['y'][:,n:n+1])*1000,(a['d'][:,n:n+1]-a['y'][:,n:n+1])*1000) for n,axis in enumerate('xyz')},
            **{f'centroid_{axis}_mm':((cent_i[:,n:n+1]-cent_y[:,n:n+1])*1000,(cent_d[:,n:n+1]-cent_y[:,n:n+1])*1000) for n,axis in enumerate('xyz')},
            'centroid_xyz_mm':((cent_i-cent_y)*1000,(cent_d-cent_y)*1000),
            'root_world_rad':(eri,erd),'rigid_joint_first_order_mm':(rigid_i*1000,rigid_d*1000)}
        np.savez_compressed(out/f's{seed}_residuals.npz',**a,**{f'{kind}_{name}':v for name,pair in errors.items() for kind,v in zip(('ei','ed'),pair)})
        stats={}
        for cohort in ('pooled_unique',*sorted(set(a['cohort']))):
            for condition in ('all_conditions',*CONDITIONS):
                for scope in ('all_observed','full_horizon','short_only'):
                    mask=np.ones(len(a['i']),bool)
                    if cohort!='pooled_unique':mask &= a['cohort']==cohort
                    if condition!='all_conditions':mask &= a['condition']==condition
                    if scope!='all_observed':mask &= a['short'] if scope=='short_only' else ~a['short']
                    if not mask.any():continue
                    keys=list(zip(a['packet'][mask],a['condition'][mask]));counts=Counter(keys)
                    weights=np.array([1/counts[key] for key in keys]);gate=a['gate'][mask]
                    stats[f'{cohort}/{condition}/{scope}']={'rows':int(mask.sum()),'unique_packet_conditions':len(counts),
                        'effective_gain_mean':np.average(a['current_gain'][mask,:4],axis=0,weights=weights).tolist(),
                        'gate_mean':float(np.average(gate,weights=weights)),
                        'moments':{name:moments(ei[mask],ed[mask],gate,weights) for name,(ei,ed) in errors.items()}}
        checkpoint=torch.load(summary['identity']['checkpoint'],map_location='cpu',weights_only=False)
        old=checkpoint['state_dict']['xkf_moments'].numpy();train_path=base/f'calibration_raw_s{seed}/train_predictions.npz'
        from semkine.xkf import group_residuals,group_moments
        with np.load(train_path) as data:
            err=group_residuals(torch.from_numpy(data['direct_state']),torch.from_numpy(data['truth']))
            vd=group_moments(err,err).numpy()
        result['seeds'][str(seed)]={'base_gains_frozen2200':summary['gains_fused'],'nominal_policy_gain_saturatedW50':(np.asarray(summary['gains_fused'])*np.asarray([.5]*3+[.25]+[.5]*15)).tolist(),
            'old_training_moments':old.tolist(),'new_raw_train2304_V_D':vd.tolist(),'old_VD_div_new_raw2304_VD':(old[1]/np.maximum(vd,1e-20)).tolist(),
            'train_predictions_sha256':sha(train_path),'raw_residuals_sha256':sha(out/f's{seed}_residuals.npz'),'statistics':stats}
        write(out/'result.json',result);print(json.dumps({'completed_seed':seed,'rows':len(a['i']),'cells':len(stats)}),flush=True)
    result['all_three_seeds_complete']=True;write(out/'result.json',result)

if __name__=='__main__':main()
