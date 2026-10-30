#!/usr/bin/env python3
"""Score all frozen angle heads with separate signs, cross-block and depth strata."""
from pathlib import Path
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.xdf.angle_probe import OUT,load_arrays,angle_errors,write,sha


def main():
    arrays,meta,mean,sources,protocol=load_arrays()
    wrong=np.concatenate([np.load(f)['wrong_prediction51'] for d in (ROOT/'outputs/xdf/interface/cache',OUT/'cache') for f in sorted(d.glob('*.npz'))])
    assert wrong.shape==arrays['prediction51'].shape
    split=np.array([r['split'] for r in meta]);seq=np.array([r['sequence'] for r in meta]);angle=np.array([r['angle_cache'] for r in meta]);dz=np.array([r['dz_mm'] for r in meta]);block=np.array([r.get('block','clean') for r in meta]);sign=np.array([r.get('sign',0) for r in meta])
    groups={k:split==k for k in ('fit','check')};groups.update({k:seq==k for k in ('zgz_local','zgz_global')})
    for who in sorted(set(n.split('_')[0] for n in seq if not n.startswith('zgz'))):groups['check_'+who]=(split=='check')&np.array([n.startswith(who+'_') for n in seq])
    masks={}
    for name,g in groups.items():
        for z in (0,-50,50):
            masks[f'{name}/clean_z{z}']=g&~angle&(dz==z)
            for b in ('root','finger'):
                masks[f'{name}/{b}_both_z{z}']=g&angle&(block==b)&(dz==z)
                for s in (-1,1):masks[f'{name}/{b}_{s:+d}_z{z}']=g&angle&(block==b)&(sign==s)&(dz==z)
    results={};gates=[]
    for b in ('root','finger'):
        hold=angle_errors(arrays['incoming51'],arrays['target51'],mean,b)
        predictions={'hold':arrays['incoming51'],'q1':arrays['prediction51'],'q1_wrong':wrong}
        for seed in (1701,1702):
            for interface in ('F','B','state','wrongFull'):
                p=OUT/'heads'/f'{b}_{interface}_s{seed}.predictions.npz'
                predictions[f'{interface}_s{seed}']=np.load(p)['prediction51'];sources[str(p.relative_to(ROOT))]=sha(p)
        for name,pred in predictions.items():
            err=angle_errors(pred,arrays['target51'],mean,b);scores={}
            for group,mask in masks.items():
                assert mask.any(),group
                scores[group]={'n':int(mask.sum()),'mean_deg':float(err[mask].mean()),'p95_sample_mean_deg':float(np.percentile(err[mask].mean(1),95)),
                               'per_joint_deg':err[mask].mean(0).tolist(),'better_than_hold_n':int((err[mask].mean(1)<hold[mask].mean(1)).sum())}
            results[f'{b}/{name}']=scores
        group=f'check/{b}_both_z0'
        for seed in (1701,1702):
            value=results[f'{b}/F_s{seed}'][group]['mean_deg']
            for control in ('state','wrongFull'):
                ref=results[f'{b}/{control}_s{seed}'][group]['mean_deg']
                gates.append({'block':b,'seed':seed,'control':control,'F_deg':value,'reference_deg':ref,'improvement_deg':ref-value,
                              'relative_improvement':1-value/ref,'pass':bool(ref-value>=1 and value<=.9*ref)})
    write(OUT/'summary.json',{'protocol_sha256':sha(OUT/'protocol.json'),'source_sha256':sha(__file__),'sources':sources,'n':len(meta),'gates':gates,
            'block_gate_pass':{b:all(g['pass'] for g in gates if g['block']==b) for b in ('root','finger')},'metrics':results,
            'interpretation':'SO3 evaluation, correct mean once; signs and cross-block effects separated. Negative probe is not proof of information absence or an upstream bottleneck.'})
    print(json.dumps({'gates':gates},ensure_ascii=False))


if __name__=='__main__':main()
