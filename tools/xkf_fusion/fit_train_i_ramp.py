#!/usr/bin/env python3
"""Train-only one-parameter I ramp calibration; never reads development outputs."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'model')]
BASE=ROOT/'outputs/xkf/fusion_20261015_r3/direct_finetune'
SEEDS=(3407,3408,3409)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    temp=Path(path).with_suffix(Path(path).suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def require(value,message):
    if not value:raise ValueError(message)

def analytic_scale(count,p2,cross):
    """Exact piecewise quadratic solve in 1/N, including N=infinity diagnosis."""
    count,p2,cross=(np.asarray(x,np.float64) for x in (count,p2,cross))
    require(count.ndim==p2.ndim==cross.ndim==1 and count.shape==p2.shape==cross.shape and len(count)>0,'proxy dimensions')
    require(np.isfinite(np.stack([count,p2,cross])).all() and (count>=0).all() and (p2>=0).all(),'invalid proxy')
    order=np.argsort(count);c,p,b=count[order],p2[order],cross[order]
    A=np.r_[0.,np.cumsum(c*c*p)];B=np.r_[0.,np.cumsum(c*b)];C=np.r_[0.,np.cumsum(p-2*b)]
    bounds=np.r_[1.,np.unique(c[c>1]),np.inf];candidates=[]
    for low,high in zip(bounds[:-1],bounds[1:]):
        index=np.searchsorted(c,low,side='right');aa,bb=A[index],B[index];cc=C[-1]-C[index]
        optimum=aa/bb if bb>0 else np.inf
        n=float(np.clip(optimum,low,high))
        loss=cc if math.isinf(n) else aa/(n*n)-2*bb/n+cc
        candidates.append((float(loss)/len(c),n))
        candidates.append(((aa/(low*low)-2*bb/low+cc)/len(c),float(low)))
    # count0 has g0; its predicted delta should be zero, but do not assume it.
    zero_loss=0.
    candidates.append((zero_loss,np.inf))
    loss,n=min(candidates,key=lambda item:(item[0],item[1]))
    return {'threshold':None if math.isinf(n) else n,'proxy_loss_minus_true_energy':float(loss),'identified_finite':math.isfinite(n)}

def interpolate(previous,proposal,g):
    """Independent float64 scipy SO3; exact boundary states, no added clip."""
    p,q=previous.reshape(-1,51),proposal.reshape(-1,51);g=np.asarray(g,np.float64).reshape(-1)
    result=p.astype(np.float64)+g[:,None]*(q.astype(np.float64)-p)
    rp,rq=Rotation.from_rotvec(p[:,3:6]),Rotation.from_rotvec(q[:,3:6])
    r=(rp*Rotation.from_rotvec(g[:,None]*(rp.inv()*rq).as_rotvec())).as_rotvec()
    norm=np.maximum(np.linalg.norm(r,axis=1,keepdims=True),1e-9);alt=r*(1-2*np.pi/norm)
    choose=(np.linalg.norm(alt-p[:,3:6],axis=1)<np.linalg.norm(r-p[:,3:6],axis=1))&(norm[:,0]>1e-6)
    result[:,3:6]=np.where(choose[:,None],alt,r)
    return np.where((g==0)[:,None],p,np.where((g==1)[:,None],q,result)).astype(np.float32).reshape(previous.shape)

def paired_evidence(before,after):
    d=np.asarray(before,np.float64)-np.asarray(after,np.float64)
    return {'improvement':float(d.mean()),'standard_error':float(d.std(ddof=1)/np.sqrt(len(d))),
            'beyond_one_SE':bool(d.mean()>d.std(ddof=1)/np.sqrt(len(d)))}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=BASE/'train_i_ramp_calibration');args=parser.parse_args();out=args.output.resolve()
    require((out/'prereg.json').is_file(),'pre-registration required before reading predictions')
    require(not (out/'result.json').exists(),'refuse overwriting fitted evidence')
    prereg=read(out/'prereg.json');require(prereg['read_development_data'] is False and prereg['rows_per_seed']==2304 and prereg['conditions']==10,'prereg contract differs')
    import torch,yaml
    from mano_layer import ManoLayer
    from pose_repr import decode_to_mano_inputs
    from tools.xkf_fusion.train_i_motion_probe import reduce,rotations
    from tools.xkf_repair.assess import assert_close
    torch.set_num_threads(4);inputs={};data={};protocols={};identities={};verification={}
    code_paths=['tools/xkf_fusion/fit_train_i_ramp.py','tools/xkf_fusion/train_i_motion_probe.py','model/mano_layer.py','model/pose_repr.py']
    source={name:sha(ROOT/name) for name in code_paths}
    for seed in SEEDS:
        directory=BASE/f'train_i_motion_s{seed}';protocol=read(directory/'protocol.json');summary=read(directory/'result.json')
        require(summary['status']=='COMPLETE' and not summary['partial_smoke'] and summary['sample_count']==2304 and summary['condition_count']==10,'incomplete train probe')
        require(summary['protocol_sha256']==sha(directory/'protocol.json') and summary['raw_sha256']==sha(directory/'predictions.npz'),'train producer hash differs')
        require(summary['D_forward_calls']==0 and summary['actual_I_forward_calls']==23040 and summary['all_named_buffers_unchanged'],'wrong I-only execution')
        require(summary['identity']==protocol['identity'] and summary['identity']['seed']==seed and summary['identity']['global_step']==2400,'wrong training endpoint')
        for key in ('config','checkpoint'):
            require(sha(summary['identity'][key])==summary['identity'][key+'_sha256'],'current training endpoint artifact differs '+key)
        clocks=summary['identity']['training_clocks']
        require(clocks['I_parameter_Adam_step']==2200 and clocks['D_parameter_Adam_step']==clocks['scheduler_and_global_step']==2400,'training clocks differ')
        for name,digest in protocol['source_sha256'].items():require(sha(ROOT/name if not Path(name).is_absolute() else name)==digest,'train source changed '+name)
        for name,digest in protocol['input_sha256'].items():require(sha(name)==digest,'train input changed '+name)
        for name in ('protocol.json','result.json','predictions.npz'):inputs[str((directory/name).resolve())]=sha(directory/name)
        with np.load(directory/'predictions.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
        require(a['incoming'].shape==a['prediction'].shape==(2304,10,51) and a['truth'].shape==a['previous_truth'].shape==(2304,51),'train rows/conditions wrong')
        for name,value in a.items():
            if value.dtype.kind in 'fc':require(np.isfinite(value).all(),'nonfinite train '+name)
        require(np.array_equal(a['motion_delta'],a['truth']-a['previous_truth']) and np.array_equal(a['predicted_motion'],a['prediction']-a['incoming']),'motion target mixed with initial error')
        require(np.array_equal(a['motion_error'],a['predicted_motion']-a['motion_delta'][:,None]),'motion residual differs')
        rr=rotations(a)
        for key,value in zip(('predicted_root_world_delta','true_root_world_delta','root_motion_geodesic_error_deg'),rr):require(np.array_equal(a[key],value),'root reducer differs')
        assert_close(summary['summaries'],reduce(a),'training raw reducers')
        require(len(set(a['sequence']))==72 and not any(s.startswith('zgz_') for s in a['sequence']),'development sequence prohibited')
        require(a['sequence'].tolist()==[r['sequence'] for r in protocol['rows']] and a['end_ms'].tolist()==[r['end_ms'] for r in protocol['rows']],'canonical row order differs')
        if data:
            for key in ('incoming','previous_truth','truth','beta','camera_K','short_count','sequence','end_ms','event_sha256','actual_label_delta_t_s'):
                require(np.array_equal(a[key],data[SEEDS[0]][key]),'seed input pairing differs '+key)
        data[seed]=a;protocols[seed]=protocol;identities[seed]=summary['identity'];verification[str(seed)]={'all2304x10_preserved':True,'noD_and_target0_source_audited':True,'raw_reducers_recomputed':True}
    config=yaml.safe_load(Path(identities[SEEDS[0]]['config']).read_text());mano=ManoLayer(config['MANO']['NPZ'],add_mean=False).float().eval();mano.fast_chain=True
    source[str(Path(config['MANO']['NPZ']).resolve())]=sha(config['MANO']['NPZ'])
    def fk(states,betas):
        shape=states.shape[:-1];x=states.reshape(-1,51);b=np.broadcast_to(betas[:,None,:],(*shape,10)).reshape(-1,10) if len(shape)==2 else betas
        result=[]
        with torch.no_grad():
            for begin in range(0,len(x),256):
                state=torch.from_numpy(x[begin:begin+256]).float();beta=torch.from_numpy(b[begin:begin+256].copy()).float()
                decoded=decode_to_mano_inputs(state,config['MODEL']['POSE_REPR'],mano.hands_components,mano.hands_mean)
                result.append(mano(beta,decoded['global_orient'],decoded['local_full_aa'],decoded['transl'])[1].numpy())
        return np.concatenate(result).reshape(*shape,21,3)
    first=data[SEEDS[0]];counts=first['short_count'].astype(float);sequence=first['sequence'];person=np.array([s.split('_')[0] for s in sequence]);require(len(set(person))==9,'not nine training identities')
    full=np.array([r['stratum']=='full200' for r in protocols[SEEDS[0]]['rows']]);require(full.sum()==2016 and (~full).sum()==288,'eligibility strata differ')
    gt_delta=(fk(first['truth'],first['beta'])-fk(first['previous_truth'],first['beta']))*1000
    gt_energy=np.mean(np.sum(gt_delta.astype(float)**2,axis=-1),axis=-1);speed=np.sqrt(gt_energy)/first['actual_label_delta_t_s']
    old_joints={};proxy={};ideal_reference={}
    for seed in SEEDS:
        a=data[seed];old_joints[seed]=fk(a['incoming'],a['beta']);pred_delta=(fk(a['prediction'],a['beta'])-old_joints[seed])*1000
        ideal_states=a['incoming']+a['motion_delta'][:,None]
        ideal_delta=(fk(ideal_states,a['beta'])-old_joints[seed])*1000
        ideal_error=np.mean(np.sum((ideal_delta-gt_delta[:,None]).astype(float)**2,-1),-1)
        model_to_ideal=np.mean(np.sum((pred_delta-ideal_delta).astype(float)**2,-1),-1)
        ideal_reference[str(seed)]={}
        for group,columns in {'clean':[0],'small_gaussian':[9],'eight_large_perturbations':list(range(1,9)),'all10':list(range(10))}.items():
            ideal_reference[str(seed)][group]={'ideal_clean51delta_vs_physical_true_motion_RMSE_mm':float(np.sqrt(ideal_error[:,columns].mean())),
                'actual_I_vs_ideal_clean51delta_flow_RMSE_mm':float(np.sqrt(model_to_ideal[:,columns].mean())),
                'ideal_physical_flow_axis_bias_mm':(ideal_delta[:,columns]-gt_delta[:,None]).mean((0,1,2)).tolist()}
        proxy[seed]=(np.mean(np.sum(pred_delta.astype(float)**2,axis=-1),axis=(1,2)),np.mean(np.sum(pred_delta*gt_delta[:,None],axis=-1),axis=(1,2)))
    fit_source={'prereg_sha256':sha(out/'prereg.json'),'source_sha256':source,'training_source_sha256':inputs,'input_verification':verification}
    write(out/'fit_input_manifest.json',fit_source)
    folds={};thresholds={};high={};fits={}
    for scheme,groups in [('identity',person),('trial',sequence)]:
        folds[scheme]=sorted(set(groups));thresholds[scheme]={family:{seed:np.zeros(2304) for seed in SEEDS} for family in ('shared','per_seed')};high[scheme]=np.zeros(2304,bool);fits[scheme]=[]
        for group in folds[scheme]:
            test=groups==group;train=~test;cut=float(np.quantile(speed[train],.75));high[scheme][test]=speed[test]>=cut
            p=np.mean([proxy[s][0][train] for s in SEEDS],axis=0);b=np.mean([proxy[s][1][train] for s in SEEDS],axis=0)
            common=analytic_scale(counts[train],p,b);each={s:analytic_scale(counts[train],proxy[s][0][train],proxy[s][1][train]) for s in SEEDS}
            fits[scheme].append({'heldout':str(group),'train_rows':int(train.sum()),'test_rows':int(test.sum()),'high_motion_fit_q75_mm_s':cut,'shared':common,'per_seed':each})
            for seed in SEEDS:
                thresholds[scheme]['shared'][seed][test]=common['threshold'] if common['threshold'] is not None else np.inf
                thresholds[scheme]['per_seed'][seed][test]=each[seed]['threshold'] if each[seed]['threshold'] is not None else np.inf
    scores={};raw_oof={}
    for seed in SEEDS:
        a=data[seed];raw_oof[seed]={};scores[seed]={}
        policies={'fixed1000':np.full(2304,1000.)}
        policies.update({f'{scheme}_{family}':thresholds[scheme][family][seed] for scheme in folds for family in ('shared','per_seed')})
        for name,ns in policies.items():
            gates=np.minimum(counts/ns,1);states=interpolate(a['incoming'],a['prediction'],np.repeat(gates,10))
            delta=(fk(states,a['beta'])-old_joints[seed])*1000
            error=delta-gt_delta[:,None];mse=np.mean(np.sum(error.astype(float)**2,axis=-1),axis=(1,2));dot=np.mean(np.sum(delta*gt_delta[:,None],axis=-1),axis=(1,2))
            scores[seed][name]={'row_mse':mse,'row_dot':dot,
                'condition_mse':np.mean(np.sum(error.astype(float)**2,-1),-1),
                'condition_dot':np.mean(np.sum(delta*gt_delta[:,None],-1),-1)}
            raw_oof[seed][name+'_delta_mm']=delta;raw_oof[seed][name+'_threshold']=ns
        np.savez_compressed(out/f'oof_s{seed}.npz',**raw_oof[seed],truth_delta_mm=gt_delta,sequence=sequence,identity=person,full200=full,**{f'{k}_high_motion':v for k,v in high.items()})
        print(json.dumps({'scored_seed':seed,'rows':2304,'families':5}),flush=True)
    def summarize(seed,name,mask):
        require(mask.any(),'empty validation stratum');s=scores[seed][name]
        return {'rows':int(mask.sum()),'motion_MSE_mm2':float(s['row_mse'][mask].mean()),'motion_RMSE_mm':float(np.sqrt(s['row_mse'][mask].mean())),
                'motion_projection_fraction':float(s['row_dot'][mask].sum()/max(gt_energy[mask].sum(),1e-20))}
    metrics={};eligible={};family_identity_losses={}
    for family in ('shared','per_seed'):
        cells={};allguards=True;finite=True
        for scheme in folds:
            cells[scheme]={}
            for seed in SEEDS:
                name=f'{scheme}_{family}';masks={'all':np.ones(2304,bool),'original_K4_main_eligible':full,'cold_truncated':~full,'high_motion':high[scheme]}
                cell={k:{'candidate':summarize(seed,name,v),'fixed1000':summarize(seed,'fixed1000',v)} for k,v in masks.items()}
                cell['condition_strata']={}
                for group,columns in {'clean':[0],'small_gaussian':[9],'eight_large_perturbations':list(range(1,9))}.items():
                    cell['condition_strata'][group]={}
                    for label,policy in [('candidate',name),('fixed1000','fixed1000')]:
                        part=scores[seed][policy]
                        cell['condition_strata'][group][label]={
                            'motion_RMSE_mm':float(np.sqrt(part['condition_mse'][:,columns].mean())),
                            'motion_projection_fraction':float(part['condition_dot'][:,columns].mean(1).sum()/max(gt_energy.sum(),1e-20)),
                            'high_motion_RMSE_mm':float(np.sqrt(part['condition_mse'][high[scheme]][:,columns].mean()))}
                overall=cell['all'];h=cell['high_motion'];bp=h['fixed1000']['motion_projection_fraction']
                guards={'overall_MSE_noninferior':overall['candidate']['motion_MSE_mm2']<=overall['fixed1000']['motion_MSE_mm2'],
                        'high_motion_RMSE':h['candidate']['motion_RMSE_mm']<=1.05*h['fixed1000']['motion_RMSE_mm'],
                        'high_motion_projection':bp>0 and h['candidate']['motion_projection_fraction']>=.95*min(bp,1.)}
                cell['guards']=guards;cells[scheme][str(seed)]=cell;allguards &= all(guards.values());finite &= bool(np.isfinite(thresholds[scheme][family][seed]).all())
        baseline=[];candidate=[]
        for group in folds['identity']:
            mask=person==group;baseline.append(np.mean([scores[s]['fixed1000']['row_mse'][mask].mean() for s in SEEDS]));candidate.append(np.mean([scores[s]['identity_'+family]['row_mse'][mask].mean() for s in SEEDS]))
        ev=paired_evidence(baseline,candidate);family_identity_losses[family]=candidate
        eligible[family]=bool(allguards and finite and ev['beyond_one_SE']);metrics[family]={'fold_validation':cells,'identity_paired_improvement':ev,'all_folds_finite':finite,'eligible':eligible[family]}
    superiority=paired_evidence(family_identity_losses['shared'],family_identity_losses['per_seed'])
    selected='shared' if eligible['shared'] else 'per_seed' if eligible['per_seed'] else 'fixed1000'
    if eligible['shared'] and eligible['per_seed'] and superiority['beyond_one_SE']:selected='per_seed'
    final_common=analytic_scale(counts,np.mean([proxy[s][0] for s in SEEDS],0),np.mean([proxy[s][1] for s in SEEDS],0))
    final_each={s:analytic_scale(counts,*proxy[s]) for s in SEEDS}
    final_n={s:1000. if selected=='fixed1000' else final_common['threshold'] if selected=='shared' else final_each[s]['threshold'] for s in SEEDS}
    accepted=selected!='fixed1000' and all(v is not None and np.isfinite(v) for v in final_n.values())
    result={'kind':'TRAIN_ONLY_I_RAMP_FIT','status':'ACCEPTED_FOR_DEV_EVALUATION' if accepted else 'TRAIN_EVIDENCE_NOT_SUPPORTIVE','formal':False,'automatic_full_training':False,
            **fit_source,'selected_family':selected,'accepted_for_dev':bool(accepted),'thresholds':final_n,'metrics':metrics,'per_seed_vs_shared_paired_improvement':superiority,
            'fold_fits':fits,'ideal_clean51delta_geometry_reference':ideal_reference,
            'ideal_reference_scope':'FK(incoming + GTclean_delta)-FK(incoming) need not equal physical GT jointflow under large initial pose perturbations; this ideal reference is diagnostic only and does not change the declared fit target or weights',
            'all_training_refit':{'shared':final_common,'per_seed':final_each},'oof_sha256':{str(out/f'oof_s{s}.npz'):sha(out/f'oof_s{s}.npz') for s in SEEDS},
            'scope':'train calibration model selection only; mandatory fresh deployment own-loop; no development data read'}
    require(all(sha(path)==digest for path,digest in inputs.items()),'training inputs changed during fit')
    write(out/'result.json',result)
    if accepted:
        for seed in SEEDS:
            write(out/f'runtime_s{seed}.json',{'kind':'motion_count_ramp','threshold':final_n[seed],'seed':seed,'checkpoint_sha256':identities[seed]['checkpoint_sha256'],
                'fit_path':str(out/'result.json'),'fit_sha256':sha(out/'result.json'),'prereg_path':str(out/'prereg.json'),'prereg_sha256':sha(out/'prereg.json'),
                'training_source_sha256':inputs,'selection_family':selected,'accepted_for_dev':True})
    print(json.dumps({'status':result['status'],'selected':selected,'thresholds':final_n}),flush=True)

if __name__=='__main__':main()
