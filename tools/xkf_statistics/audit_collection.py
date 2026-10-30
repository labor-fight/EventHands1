#!/usr/bin/env python3
"""CPU independent raw geometry, trajectory and fold audit; never fit a policy."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'model')]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def near(a,b,name,atol=3e-6,rtol=3e-6):
    require(np.shape(a)==np.shape(b) and np.allclose(a,b,atol=atol,rtol=rtol),name)
def typed_sha(value):
    a=np.ascontiguousarray(value);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    import torch,yaml
    from mano_layer import ManoLayer
    from pose_repr import decode_to_mano_inputs
    from tools.xkf_fusion.assess_i_continuation import independent_pose_fusion
    from tools.xkf_fusion.assess_motion_gate import independent_motion
    torch.set_num_threads(4);inputs={};results={}
    def read(path):
        path=Path(path);inputs[str(path.resolve())]=sha(path);return json.loads(path.read_text())
    declaration=read(args.root/'declaration_v2.json');canonical=read(declaration['rows_path']);indices=np.array(declaration['fit_row_indices']);rows=[canonical[i] for i in indices]
    cfg=yaml.safe_load(Path(declaration['models']['3407']['config']).read_text());inputs[str(Path(cfg['MANO']['NPZ']).resolve())]=sha(cfg['MANO']['NPZ'])
    mano=ManoLayer(cfg['MANO']['NPZ'],add_mean=False).float().eval();mano.fast_chain=True
    def fk(states,beta):
        values=[]
        with torch.no_grad():
            for start in range(0,len(states),256):
                x=torch.from_numpy(np.asarray(states[start:start+256],np.float32));d=decode_to_mano_inputs(x,cfg['MODEL']['POSE_REPR'],mano.hands_components,mano.hands_mean)
                values.append(mano(torch.from_numpy(np.asarray(beta[start:start+256],np.float32)),d['global_orient'],d['local_full_aa'],d['transl'])[1].numpy())
        return np.concatenate(values)
    refpath=ROOT/'outputs/xkf/fusion_20261015_r3/direct_finetune/trajectory1049_analysis/canonical2304_motion_gt.npz';inputs[str(refpath)]=sha(refpath)
    with np.load(refpath,allow_pickle=False) as z:reference={k:z[k] for k in z.files}
    common=None
    for seed in (3407,3408,3409):
        directory=args.root/f'collection_s{seed}';summary=read(directory/'summary.json');protocol=read(directory/'protocol.json');result=read(directory/'result.json');folds=read(args.root/f'folds_s{seed}.json')
        require(summary['rows']==2016 and summary['actual_forward_calls']==8064 and not summary['partial_smoke'] and summary['all_buffers_unchanged'],'incomplete collection')
        require(summary['declaration_sha256']==sha(args.root/'declaration_v2.json') and summary['raw_sha256']==sha(directory/'residuals.npz'),'collection binding differs')
        require(result['protocol_sha256']==sha(directory/'protocol.json') and summary['result_sha256']==sha(directory/'result.json'),'raw protocol/result binding differs')
        for obj in (summary,folds):
            for group in ('source_sha256','input_sha256'):
                for name,digest in obj[group].items():
                    path=Path(name);path=path if path.is_absolute() else ROOT/path;require(sha(path)==digest,'source/input hash differs');inputs[str(path.resolve())]=digest
        rawpath=directory/'residuals.npz';inputs[str(rawpath.resolve())]=sha(rawpath)
        with np.load(rawpath,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        require(all(v.shape[0]==2016 for v in a.values()),'missing raw rows')
        for key,v in a.items():
            if v.dtype.kind in 'fc':require(np.isfinite(v).all(),'nonfinite raw '+key)
        require(np.array_equal(a['row_indices'],indices) and a['sequence'].tolist()==[r['sequence'] for r in rows],'canonical population differs')
        for name in ('truth','previous_truth','sequence','end_ms'):
            require(np.array_equal(a[name],reference[name][indices]),'independent GT reference differs '+name)
        require(np.array_equal(a['trajectory_short_count'][:,-1],reference['short_count'][indices]),'independent main count differs')
        paired=('truth','previous_truth','sequence','end_ms','beta','camera_K','initial','initial_truth','gaussian_draw','trajectory_truth','trajectory_previous_truth','trajectory_short_count','trajectory_end_ms','trajectory_event_sha256','trajectory_lnes_sha256')
        if common is None:common={k:a[k].copy() for k in paired}
        else:
            for k in paired:require(np.array_equal(common[k],a[k]),'seed input changed '+k)
        near(a['trajectory_incoming'][:,1:],a['trajectory_fused'][:,:-1],'own-loop links',0,0)
        near(a['trajectory_incoming'][:,0],a['initial'],'initial forward',0,0);near(a['initial'],a['initial_truth']+a['gaussian_draw'],'initial Gaussian',0,0)
        near(a['trajectory_end_ms'],a['end_ms'][:,None]+np.array([-150,-100,-50,0]),'four times',0,0)
        near(a['trajectory_actual_label_dt_s'],np.broadcast_to([.049,.05,.05,.05],(2016,4)),'49/50ms label gaps',0,0)
        gate=np.minimum(a['trajectory_short_count'].astype(np.float32)/1000,1);near(a['trajectory_motion_gate'],gate,'I count gates',2e-8,1e-7);near(a['gate'],a['trajectory_motion_gate'][:,-1],'main gates',0,0);near(a['gate'],a['gates'],'gate aliases',0,0)
        near(a['trajectory_effective_gains'],gate[:,:,None]*np.array(summary['baseline_effective_gains'],np.float32),'effective baseline gains',1e-7,1e-6)
        inc=torch.from_numpy(a['trajectory_incoming'].reshape(-1,51));motion=torch.from_numpy(a['trajectory_motion'].reshape(-1,51));gated=independent_motion(inc,motion,gate.reshape(-1));near(gated.numpy(),a['trajectory_gated_motion'].reshape(-1,51),'independent I SO3/linear gate')
        direct=torch.from_numpy(a['trajectory_direct'].reshape(-1,51));k=torch.from_numpy(a['trajectory_effective_gains'].reshape(-1,19));fused=independent_pose_fusion(gated,direct,k)
        beta=np.repeat(a['beta'],4,axis=0);states=torch.cat([gated,direct,fused]).numpy();states[:,:3]=0;centroids=fk(states,np.tile(beta,(3,1))).mean(1);ci,cd,cf=np.split(centroids,3)
        fused[:,:3]=torch.from_numpy((1-k[:,:3].numpy())*(gated[:,:3].numpy()+ci)+k[:,:3].numpy()*(direct[:,:3].numpy()+cd)-cf)
        ff=fused.numpy();allzero=(k.numpy()==0).all(1);ff[allzero]=gated.numpy()[allzero];empty=a['trajectory_short_count'].reshape(-1)==0;ff[empty]=inc.numpy()[empty]
        near(ff,a['trajectory_fused'].reshape(-1,51),'independent 8064 centroid/root fusions')
        for key,state in [('truth_joints',a['truth']),('previous_truth_joints',a['previous_truth']),('gated_i_joints',a['trajectory_gated_motion'][:,-1]),('direct_joints',a['trajectory_direct'][:,-1])]:near(fk(state,a['beta']),a[key],'independent FK '+key)
        gt=a['truth'];rg=Rotation.from_rotvec(gt[:,3:6]);
        for key,state,jkey in [('e_i',a['trajectory_gated_motion'][:,-1],'gated_i_joints'),('e_d',a['trajectory_direct'][:,-1],'direct_joints')]:
            e=state.astype(float)-gt.astype(float);e[:,:3]=a[jkey].astype(float).mean(1)-a['truth_joints'].astype(float).mean(1);e[:,3:6]=(rg.inv()*Rotation.from_rotvec(state[:,3:6])).as_rotvec();near(e,a[key],'independent physical/tangent residual '+key,1e-12,1e-12)
        speed=np.linalg.norm((a['truth_joints'].astype(float)-a['previous_truth_joints'].astype(float))/.05,axis=2).mean(1)*1000;near(speed,a['gt_joint_speed_mm_s'],'physical mean joint speed',1e-12,1e-12)
        checkpoint=Path(summary['identity']['checkpoint']);saved=torch.load(checkpoint,map_location='cpu',weights_only=False);order=read(checkpoint.parent/'resume_audit_rank0.json')['parameter_order'];h=hashlib.sha256()
        for name in order:h.update(name.encode());h.update(saved['state_dict'][name].contiguous().numpy().tobytes())
        require(h.hexdigest()==summary['all_parameters_unchanged_sha256'],'actual runtime params differ from checkpoint')
        buffer_audit=read(directory/'model_state_audit.json');present=0
        for name,digest in buffer_audit['named_buffer_sha256'].items():
            if name in saved['state_dict']:require(typed_sha(saved['state_dict'][name].numpy())==digest,'checkpoint buffer differs '+name);present+=1
        groups=[(0,1),(1,2),(2,3),(3,6)]+[(i,i+3) for i in range(6,51,3)];identities=np.array([s.split('_')[0] for s in a['sequence']]);families=np.array([s.split('_v')[0] for s in a['sequence']]);checked=0
        candidates=[('all',folds['all_train'],np.ones(2016,bool))]+[(name,v,identities!=name) for name,v in folds['identity'].items()]+[(name,v,families!=name) for name,v in folds['motion_family'].items()]
        require(len(candidates)==28,'fold coverage differs')
        for name,fold,mask in candidates:
            require(fold['train_indices']==indices[mask].tolist() and fold['excluded_indices']==indices[~mask].tolist(),'fold leakage or omitted rows')
            require(fold['high_motion_threshold_mm_s']==float(np.quantile(speed[mask],.75)),'fit-side high-motion threshold differs')
            coeff=[]
            for gi,(start,stop) in enumerate(groups):
                ei,ed=a['e_i'][mask,start:stop],a['e_d'][mask,start:stop];g=a['gate'][mask].astype(float);diff=ei-ed;num=np.mean(g*np.sum(ei*diff,1));den=np.mean(g*g*np.sum(diff*diff,1));coeff.append(summary['baseline_effective_gains'][gi] if den<=declaration['fitting']['denominator_floor'] else float(np.clip(num/den,0,1)))
            near(np.array(coeff),np.array(fold['coefficients']),'independent numpy coefficients',2e-12,2e-12);checked+=1
        require(set(folds['trial'])==set(a['sequence']),'missing version lookup')
        for sequence,value in folds['trial'].items():require(value==folds['motion_family'][sequence.split('_v')[0]],'sequence lookup not family fit')
        results[str(seed)]={'status':'PASS','rows':2016,'independent_fusion_replays':8064,'physical_residuals_and_speed_recomputed':True,'folds_recomputed':checked,'whole_motion_family_exclusion':True,'named_checkpoint_buffers_verified':present,'runtime_buffer_total':len(buffer_audit['named_buffer_sha256']),'runtime_buffer_scope':'producer before/after torch.equal plus per-buffer hashes, independent checkpoint comparison where persisted'}
        print(json.dumps({'verified':seed}),flush=True)
    for name,digest in inputs.items():require(sha(name)==digest,'audit input changed')
    out=dict(kind='INDEPENDENT_TRAIN_PAIRED_COLLECTION_AND_FIT_AUDIT',status='PASS',seeds=results,input_sha256=inputs,source_sha256={str(Path(__file__).resolve()):sha(__file__)},formal=False,accepted_for_dev=False)
    args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'PASS'}))

if __name__=='__main__':main()
