#!/usr/bin/env python3
"""Independent fixed-policy raw audit for parent2400 and I-only2600 endpoints.

All 112 clips and nine conditions are retained. This certifies data, geometry and
unchanged additional debug gates, not formal G1-G4/full-training readiness.
"""
from __future__ import annotations
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'model')]
from tools.xkf_fusion.assess import Assessor,R2,CONDITIONS,DIRECTIONAL,angle,directional_reduce,near,require,assert_close,write,sha
from tools.xkf_fusion.assess_motion_gate import independent_motion

BASE=ROOT/'outputs/xkf/fusion_20261015_r3/direct_finetune'
POLICY={'gain_scale':[.5,.25,.5],'count_gate':'ramp1000','translation_reference':'joint_centroid','motion_count_gate':'ramp1000','id':'fixed1000','motion_count_threshold':1000.}
MODES=('fixed1000','hold')


def independent_pose_fusion(a,b,k):
    """Independent scipy SO(3), original60degree root clip, linear other groups."""
    av,bv=a.numpy().astype(float),b.numpy().astype(float);kv=k.numpy().astype(float)
    result=av.copy();result[:,:3]+=kv[:,:3]*(bv[:,:3]-av[:,:3]);result[:,6:]+=np.repeat(kv[:,4:],3,axis=1)*(bv[:,6:]-av[:,6:])
    ra=Rotation.from_rotvec(av[:,3:6]);rb=Rotation.from_rotvec(bv[:,3:6]);innovation=(ra.inv()*rb).as_rotvec()
    innovation*=np.minimum(1,np.radians(60)/np.maximum(np.linalg.norm(innovation,axis=1,keepdims=True),1e-9))
    roots=(ra*Rotation.from_rotvec(kv[:,3:4]*innovation)).as_rotvec();norm=np.linalg.norm(roots,axis=1,keepdims=True)
    alt=roots*(1-2*np.pi/np.maximum(norm,1e-9));use=(np.linalg.norm(alt-av[:,3:6],axis=1)<np.linalg.norm(roots-av[:,3:6],axis=1))&(norm[:,0]>1e-6)
    roots=np.where(use[:,None],alt,roots);result[:,3:6]=np.where(kv[:,3:4]>0,roots,av[:,3:6])
    return a.new_tensor(result)


class ContinuationAssessor(Assessor):
    def endpoint(self,identity,summary,arm,seed):
        torch=self.torch;path=self.path(identity['checkpoint']);cfg=self.read_yaml(identity['config'])
        require(self.digest(path)==identity['checkpoint_sha256'] and self.digest(identity['config'])==identity['config_sha256'],'checkpoint/config changed')
        saved=torch.load(path,map_location='cpu',weights_only=False);finish=self.read(path.parent/'train_finish.json');metadata=self.read(path.parent/'training_metadata.json')
        step=2400 if arm=='parent' else 2600;istep=2200 if arm=='parent' else 2400
        require(identity['seed']==seed==saved['hyper_parameters']['SEED']==cfg['SEED'] and identity['global_step']==saved['global_step']==step,'seed/global checkpoint identity differs')
        require(finish['status']=='DEBUG200_COMPLETE' and finish['global_step']==step and finish['checkpoint_sha256']==self.digest(path),'endpoint incomplete/changed')
        require(metadata['arm']==('raw' if arm=='parent' else arm),'wrong training arm')
        require(metadata['formal'] is False and metadata['updates']==200,'wrong experiment scope')
        for name,digest in finish['source_sha256'].items():require(self.digest(name)==digest,'training source changed: '+name)
        order=self.read(path.parent/'resume_audit_rank0.json')['parameter_order']
        require(order==self.read(path.parent/'resume_audit_rank1.json')['parameter_order'] and len(order)==len(set(order))==96,'named96 Adam parameter order differs')
        require(sum(n.startswith('direct.') for n in order)==60,'D parameter count differs')
        state=saved['state_dict'];opt=saved['optimizer_states'][0];ids=opt['param_groups'][0]['params']
        require(len(saved['optimizer_states'])==len(opt['param_groups'])==1 and len(ids)==len(opt['state'])==96,'incomplete original Adam')
        digest=hashlib.sha256()
        for name,key in zip(order,ids):
            tensor=state[name];digest.update(name.encode());digest.update(tensor.contiguous().numpy().tobytes());slot=opt['state'][key]
            require(int(slot['step'])==(2400 if name.startswith('direct.') else istep),'wrong Adam parameter clock '+name)
            for field in ('exp_avg','exp_avg_sq'):require(slot[field].shape==tensor.shape and bool(torch.isfinite(slot[field]).all()),'invalid Adam moment '+name)
        require(digest.hexdigest()==summary['all_parameters_unchanged_sha256'],'runtime parameter digest differs from actual checkpoint')
        require(all(v['last_epoch']==step for v in saved['lr_schedulers']),'scheduler clock differs')
        for name,value in state.items():require(not value.is_floating_point() or bool(torch.isfinite(value).all()),'checkpoint nonfinite '+name)
        require(summary['all_buffers_unchanged'] is True,'producer buffer assertion failed')
        changes=[]
        if arm=='parent':
            clocks=saved['xkf_direct_continuation'];require(clocks['i_adam_step']==2200 and clocks['d_adam_step']==2400 and clocks['all_bn_and_xkf_statistics_frozen'],'parent mixed clock declaration differs')
            require(identity['training_clocks']['I_parameter_Adam_step']==2200 and identity['training_clocks']['D_parameter_Adam_step']==2400,'parent runtime clocks differ')
            priorpath=R2/f'repair_s{seed}/xkf_repair_s{seed}-step=2200.ckpt';prior=torch.load(priorpath,map_location='cpu',weights_only=False)
            frozen=lambda name:not name.startswith('direct.') or name not in order
        else:
            clocks=saved['xkf_incremental_continuation'];require(clocks['start_step']==2400 and clocks['i_adam_step']==clocks['d_adam_step']==2400 and clocks['all_bn_and_xkf_statistics_frozen'],'I-only clock declaration differs')
            require(identity['training_clocks']['i_adam_step']==identity['training_clocks']['d_adam_step']==2400,'runtime continuation clocks differ')
            priorpath=BASE/f'raw_s{seed}/xkf_donly_raw_s{seed}-step=2400.ckpt';prior=torch.load(priorpath,map_location='cpu',weights_only=False)
            require(Path(metadata['resumed_from']).resolve()==priorpath.resolve(),'wrong raw2400 parent')
            require(clocks['parent_direct_continuation']==prior['xkf_direct_continuation'],'parent clock lineage differs')
            frozen=lambda name:name.startswith('direct.') or name not in order
        require(self.digest(priorpath)==metadata['parent_checkpoint_sha256']==finish['parent_checkpoint_sha256'],'parent checkpoint lineage differs')
        require(set(state)==set(prior['state_dict']),'checkpoint structure changed')
        prioropt=prior['optimizer_states'][0];require(ids==prioropt['param_groups'][0]['params'],'Adam slot order changed')
        for name,value in state.items():
            same=torch.equal(value,prior['state_dict'][name]);require(not frozen(name) or same,'frozen parameter/buffer changed '+name)
            if not same:changes.append(name)
        require(bool(changes),'no trainable parameter updated')
        for name,key in zip(order,ids):
            if not frozen(name):continue
            require(set(opt['state'][key])==set(prioropt['state'][key]),'frozen Adam structure changed')
            for field,value in opt['state'][key].items():
                old=prioropt['state'][key][field];require(torch.equal(value,old) if torch.is_tensor(value) else value==old,'frozen Adam changed '+name+'/'+field)
        if arm!='parent':
            for rank in (0,1):
                audit=self.read(path.parent/f'freeze_audit_rank{rank}.json');exposure=self.read(path.parent/f'actual_exposure_rank{rank}.json')
                require(audit['status']=='PASS' and audit['global_step']==2600 and audit['i_adam_step']==audit['d_adam_step']==2400,'rank freeze audit differs')
                require(exposure['final_global_step']==2600 and exposure['main_supervised_rows']==102400 and exposure['self_main_rows']>0 and exposure['burn_model_rows']>0,'rank actual training exposure incomplete')
        vi,vd,v0,cid,c0d=state['xkf_moments'];gains=((vi-cid)/(vi+vd-2*cid).clamp_min(1e-12)).clamp(0,1).numpy()
        return gains,{'global_step':step,'I_Adam_step':istep,'D_Adam_step':2400,'named_Adam_slots':96,'changed_parameters':changes,'frozen_tensors_and_Adam_bitwise_parent':True,'runtime_parameters_recomputed_from_checkpoint':True,'runtime_buffers_scope':'producer before/after named-buffer torch.equal assertion plus hashed source audit; no independent runtime snapshot'}

    def replay(self,raw,old,clip,beta,base):
        t=self.torch;n=clip['observed_steps'];shapes={'predictions':(n,9,2,51),'metrics':(n,9,2,len(self.metrics)),'physical_wrist_error_xyz_mm':(n,9,2,3),'state_t_error_xyz_mm':(n,9,2,3),'truth':(n,51),'initial':(9,51),'clean':(51,),'packet_ids':(n,),'end_ms':(n,),'lnes_window_ms':(n,),'n_raw_events':(n,),'motion_predictions':(n,9,1,51),'gated_motion_predictions':(n,9,1,51),'corrected_direct':(n,9,1,51),'effective_gains':(n,9,1,19),'motion_gate':(n,9,1)}
        require(set(raw)==set(shapes),'raw field set changed')
        for key,shape in shapes.items():
            require(raw[key].shape==shape,'raw dimensions wrong '+key)
            if raw[key].dtype.kind in 'fc':require(np.isfinite(raw[key]).all(),'nonfinite raw '+key)
        for key in ('initial','clean','truth','packet_ids','end_ms','lnes_window_ms','n_raw_events'):require(np.array_equal(raw[key],old[key]),'paired input changed '+key)
        pred=raw['predictions'];require(np.array_equal(pred[:,:,1],np.broadcast_to(raw['initial'],(n,9,51))),'hold changed')
        previous=np.concatenate([raw['initial'][None],pred[:-1,:,0]],0)
        gate=np.minimum(1.,raw['n_raw_events'].astype(np.float32)/1000.);require((gate>=0).all(),'negative counts')
        near(raw['motion_gate'][:,:,0],np.broadcast_to(gate[:,None],(n,9)),'I gate from raw counts',atol=2e-8,rtol=1e-7)
        gain=np.clip(base*np.array([.5]*3+[.25]+[.5]*15,dtype=np.float32),0,1)[None,:]*gate[:,None]
        near(raw['effective_gains'][:,:,0],np.broadcast_to(gain[:,None],(n,9,19)),'D gains from frozen residuals and W50 counts',atol=1e-7,rtol=1e-6)
        incoming=t.from_numpy(previous.reshape(-1,51));motion=t.from_numpy(raw['motion_predictions'][:,:,0].reshape(-1,51));gated=independent_motion(incoming,motion,np.repeat(gate,9))
        near(raw['gated_motion_predictions'][:,:,0],gated.numpy().reshape(n,9,51),'independent I gate interpolation',atol=2e-6,rtol=2e-6)
        direct=t.from_numpy(raw['corrected_direct'][:,:,0].reshape(-1,51));k=t.from_numpy(raw['effective_gains'][:,:,0].reshape(-1,19));fused=independent_pose_fusion(gated,direct,k)
        states=t.cat([gated,direct,fused]).clone();states[:,:3]=0.
        with t.no_grad():_,joints=self.fk_model._fk(states,t.from_numpy(beta).float().expand(len(states),-1))
        ci,cd,cf=joints.mean(1).chunk(3);fused[:,:3]=(1-k[:,:3])*(gated[:,:3]+ci)+k[:,:3]*(direct[:,:3]+cd)-cf
        fused=t.where((k==0).all(1,keepdim=True),gated,fused).numpy().reshape(n,9,51);fused[raw['n_raw_events']==0]=previous[raw['n_raw_events']==0]
        near(pred[:,:,0],fused,'independent motion+root+centroid algebra',atol=3e-6,rtol=3e-6)
        for step in np.flatnonzero(raw['n_raw_events']==0):require(np.array_equal(pred[step,:,0],previous[step]),'empty packet not exact previous')
        # D is recomputed for each real B1 call but must remain previous-state free.
        near(raw['corrected_direct'],np.broadcast_to(raw['corrected_direct'][:,:1],(n,9,1,51)),'D previous-state independence across nine conditions',atol=0,rtol=0)
        flat=pred.reshape(-1,51);truth=np.broadcast_to(raw['truth'][:,None,None],(n,9,2,51)).reshape(-1,51).copy();values=[];wrists=[];translations=[]
        with t.no_grad():
            for start in range(0,len(flat),256):
                v,w,s=self.frame_metrics(self.fk_model,t.from_numpy(flat[start:start+256]),t.from_numpy(truth[start:start+256]),t.from_numpy(beta));values.append(v.numpy());wrists.append(w.numpy());translations.append(s.numpy())
        values=np.concatenate(values).reshape(raw['metrics'].shape);near(raw['metrics'],values,'all17 metrics independent FK',atol=.01,rtol=5e-5)
        near(raw['physical_wrist_error_xyz_mm'],np.concatenate(wrists).reshape(n,9,2,3),'physical wrist FK',atol=.002,rtol=2e-5)
        near(raw['state_t_error_xyz_mm'],np.concatenate(translations).reshape(n,9,2,3),'state t errors',atol=.0002,rtol=1e-6)
        return float(np.abs(values-raw['metrics']).max())

    def seed(self,arm,seed):
        directory=self.rollout_root/arm/f's{seed}';summary=self.read(directory/'summary.json');protocol=self.read(directory/'protocol.json')
        require(summary['protocol_sha256']==self.digest(directory/'protocol.json'),'protocol changed')
        require(summary['kind']==protocol['kind']=='XKF_FIXED_I_CONTINUATION_ACTUAL_FORWARD' and summary['arm']==protocol['arm']==arm,'wrong evaluation kind/arm')
        require(summary['execution_complete'] is True and not summary['partial_smoke'] and not protocol['partial_smoke'] and summary['clip_count']==112,'incomplete/partial loop')
        require(summary['actual_uncached_single_sample_forward_calls']==902*9 and protocol['stability_mode']=='per_sample_model' and protocol['direct_cache'] is False,'not actual single-sample full loop')
        require(summary['identity']==protocol['identity'] and summary['source_sha256']==protocol['source_sha256'] and summary['input_sha256']==protocol['input_sha256'],'summary identities differ')
        require(summary['modes']==protocol['modes']==list(MODES) and summary['metrics']==protocol['metrics']==list(self.metrics),'modes/metrics differ')
        require(protocol['conditions']==self.old_schedule['conditions'] and protocol['clips']==self.old_schedule['clips'] and protocol['excluded']==self.old_schedule['excluded'] and protocol['bank_sha256']==self.old_schedule['bank_sha256'],'canonical112 schedule/conditions changed')
        require(protocol['policies']==[POLICY],'fixed policy changed')
        for name,digest in {**protocol['source_sha256'],**protocol['input_sha256']}.items():require(self.digest(name)==digest,'evaluation source/input changed '+name)
        declaration=BASE/'incremental_finetune/evaluation_declaration.json';require(self.digest(declaration)==protocol['declaration_sha256'],'predeclaration changed')
        declared=self.read(declaration)
        for name,digest in declared['source_sha256'].items():require(self.digest(name)==digest,'declared adapter changed '+name)
        gains,endpoint=self.endpoint(summary['identity'],summary,arm,seed)
        rawfiles=list((directory/'raw').glob('*.npz'));require(len(rawfiles)==len(summary['raw_sha256'])==112,'missing/extra raw clips')
        records=[];directions=[];loops={};counts=defaultdict(lambda:{'clips':0,'frames':0});fkmax=0.;highmotion=[];parentparity=[]
        for index,clip in enumerate(protocol['clips']):
            name=f'{index:03d}_{clip["cohort"]}_run{clip["run_id"]}.npz';path=directory/'raw'/name;require(self.digest(path)==summary['raw_sha256'][name],'raw hash changed')
            with np.load(path,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
            oldpath=R2/f'evaluations/repair_s{seed}_step2200_rollout/raw'/name;require(self.digest(oldpath)==self.parent['input_sha256'][str(oldpath.resolve())],'old packet source changed')
            with np.load(oldpath,allow_pickle=False) as z:old={k:z[k] for k in z.files}
            betapath=R2/f'evaluations/repair_s{seed}_step2200_direct/{clip["sequence"]}_W200.npz';self.digest(betapath)
            if clip['sequence'] not in self.cache:
                with np.load(betapath,allow_pickle=False) as z:self.cache[clip['sequence']]=z['beta'].copy()
            beta=self.cache[clip['sequence']];fkmax=max(fkmax,self.replay(raw,old,clip,beta,gains))
            if arm!='parent':
                parentpath=self.rollout_root/'parent'/f's{seed}'/'raw'/name;self.digest(parentpath)
                with np.load(parentpath,allow_pickle=False) as z:
                    for key in ('initial','clean','truth','packet_ids','end_ms','lnes_window_ms','n_raw_events'):require(np.array_equal(raw[key],z[key]),'parent-continuation input changed')
                    require(np.array_equal(raw['corrected_direct'],z['corrected_direct']),'frozen D actual B1 output changed')
                    parentparity.append({'clip_id':clip['clip_id'],'D_all_conditions_bitwise_parent':True})
            for scope in ('all_observed','short_only' if clip['censored'] else 'full_horizon'):counts[(clip['cohort'],scope)]['clips']+=1;counts[(clip['cohort'],scope)]['frames']+=clip['observed_steps']
            for ci,condition in enumerate(CONDITIONS):
                for mi,mode in enumerate(MODES):
                    v=raw['metrics'][:,ci,mi];red=self.reducers['summarize_values'];records.append(dict(clip,condition=condition,mode=mode,mean=red(v),first=red(v[:1]),endpoint=red(v[-1:]),lnes_first_ms=int(raw['lnes_window_ms'][0])))
                if condition in DIRECTIONAL:
                    pred=raw['predictions'][-1]
                    if condition.startswith('z_'):
                        target=float(raw['physical_wrist_error_xyz_mm'][-1,ci,0,2]);held=float(raw['physical_wrist_error_xyz_mm'][-1,ci,1,2]);residue=float(raw['physical_wrist_error_xyz_mm'][-1,ci,0,2]-raw['physical_wrist_error_xyz_mm'][-1,0,0,2]);initial=float((raw['initial'][ci,2]-raw['initial'][0,2])*1000)
                    else:
                        target=float(angle(raw['truth'][-1,3:6],pred[ci,0,3:6]));held=float(angle(raw['truth'][-1,3:6],pred[ci,1,3:6]));residue=float(angle(pred[0,0,3:6],pred[ci,0,3:6]));initial=float(angle(raw['initial'][0,3:6],raw['initial'][ci,3:6]))
                    directions.append(dict(clip,condition=condition,mode='fixed1000',target_error=target,hold_target_error=held,perturbation_residue=residue,initial_perturbation=initial))
            if clip['cohort'].endswith('_bank') and clip['sequence'] not in loops:
                ab=float(raw['metrics'][:,7,0,0].mean());ha=float(raw['metrics'][:,7,1,0].mean());wr=float(np.linalg.norm(raw['physical_wrist_error_xyz_mm'][-1,7,0]));hw=float(np.linalg.norm(raw['physical_wrist_error_xyz_mm'][-1,7,1]))
                loops[clip['sequence']]={'clip_id':clip['clip_id'],'mean_abs':ab,'hold_abs':ha,'endpoint_wrist':wr,'hold_endpoint_wrist':hw,'mean_abs_pass':ab<=ha,'endpoint_wrist_pass':wr<hw}
            if clip['cohort']=='global_bank':
                gt=np.concatenate([raw['clean'][None],raw['truth']]);ts=np.r_[clip['initial_gt_ms'],raw['end_ms']]
                with self.torch.no_grad():_,j=self.fk_model._fk(self.torch.from_numpy(gt),self.torch.from_numpy(beta).expand(len(gt),-1))
                speed=np.sqrt(np.square(np.diff(j.numpy(),axis=0)).sum(2).mean(1))*1e6/np.diff(ts)
                for i in range(len(speed)):highmotion.append({'packet_id':str(raw['packet_ids'][i]),'speed_mm_s':float(speed[i]),'clean_abs_mm':float(raw['metrics'][i,0,0,0]),'clean_ra_mm':float(raw['metrics'][i,0,0,1])})
        perclip=directory/'per_clip.jsonl';self.digest(perclip);assert_close([json.loads(x) for x in perclip.read_text().splitlines()],records,'perclip raw reducers')
        aggregate=self.reducers['aggregate'](records);assert_close(summary['aggregate'],aggregate,'all aggregate raw reducers')
        index={(r['cohort'],r['condition'],r['mode'],r['scope']):r for r in aggregate};control=self.read(R2/f'evaluations/control_s{seed}_step2200_rollout/summary.json');ctl={(r['cohort'],r['condition'],r['mode'],r['scope']):r for r in control['aggregate']};guards={}
        for cohort,scope in [('local_bank','all_observed'),('local_cold','full_horizon')]:
            value=index[(cohort,'clean','fixed1000',scope)]['pooled_frames']['abs_mpjpe_mm'];limit=index[(cohort,'clean','hold',scope)]['pooled_frames']['abs_mpjpe_mm'];guards[cohort]={'value':value,'limit':limit,'pass':value<=limit}
        value=index[('global_bank','clean','fixed1000','all_observed')]['pooled_frames']['abs_mpjpe_mm'];limit=ctl[('global_bank','clean','fused','all_observed')]['pooled_frames']['abs_mpjpe_mm']*1.1;guards['global_bank']={'value':value,'limit':limit,'pass':value<=limit}
        dr=[directional_reduce(directions,c,cond,'fixed1000',scope,.9,.9) for c in ('local_bank','local_cold','local_warm3') for cond in DIRECTIONAL for scope in ('all_observed','full_horizon','short_only')];dr=[v for v in dr if v is not None];full=[v for v in dr if v['used_for_gate']];require(len(full)==18,'directional rows incomplete')
        unique={r['packet_id']:r for r in highmotion};threshold=float(np.quantile([r['speed_mm_s'] for r in unique.values()],.75));high=[r for r in unique.values() if r['speed_mm_s']>=threshold]
        return {'identity':summary['identity'],'endpoint_audit':endpoint,'clean_guards':guards,'clean_pass':all(v['pass'] for v in guards.values()),'directional':dr,'directional_pass':all(v['pass'] for v in full),'original_candidate_loop_checks':loops,'aggregate':aggregate,'counts':[{'cohort':k[0],'scope':k[1],**v} for k,v in sorted(counts.items())],'global_high_motion':{'scope':'unique global_bank packets, clean; truth-speed topquartile, descriptive paired guard','threshold_mm_s':threshold,'selected_packet_ids':[r['packet_id'] for r in high],'packet_count':len(high),'clean_abs_mm':float(np.mean([r['clean_abs_mm'] for r in high])),'clean_ra_mm':float(np.mean([r['clean_ra_mm'] for r in high]))},'D_parent_parity':parentparity,'original_parent_readiness':self.read(R2/f'evaluations/repair_s{seed}_step2200.json')['readiness'],'candidate_original_readiness':{'status':'NOT_CERTIFIED','reason':'original G1-G4 and matching full-sequence candidate evidence are separate; minimal direction gates do not replace them'},'metric_max_cpu_gpu_difference':fkmax,'full_raw_geometry_and_gate_replay':True,'scope':'actual112x9; original readiness and formal G1-G4 remain separate, never inferred from minimal direction gates'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arms',nargs='+',choices=('parent','raw','aug'),default=['parent','raw','aug']);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    study=ROOT/'outputs/xkf/fusion_20261015_r3';a=ContinuationAssessor(study,BASE/'incremental_finetune/evaluations',True);setup=a.attempt('setup',a.setup)
    result={'kind':'INDEPENDENT_FIXED_I_CONTINUATION_ASSESSMENT','formal_gate_assessed':False,'automatic_full_training':False,'setup':setup,'requested_arms':args.arms,'arms':{},'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha(__file__)}}
    if setup:
        for arm in args.arms:
            result['arms'][arm]={}
            for seed in (3407,3408,3409):result['arms'][arm][str(seed)]=a.attempt(f'{arm}/{seed}',lambda arm=arm,seed=seed:a.seed(arm,seed))
    complete=len(result['arms'])==len(args.arms) and all(len(v)==3 and all(v.values()) for v in result['arms'].values())
    result.update(status='INVALID' if a.errors else 'INCOMPLETE' if a.missing or not complete else 'ASSESSED',validation_errors=a.errors,missing=a.missing,input_sha256=a.inputs)
    # Recheck the exact inputs at export, including code and prior raw predictions.
    for name,digest in a.inputs.items():
        if sha(name)!=digest:result['validation_errors'].append({'scope':'final_input_hash','path':name});result['status']='INVALID'
    write(args.output,result);print(json.dumps({'status':result['status'],'errors':result['validation_errors'],'missing':result['missing']}),flush=True)
    return 2 if result['status']=='INVALID' else 0
if __name__=='__main__':raise SystemExit(main())
