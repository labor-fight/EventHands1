#!/usr/bin/env python3
"""Independent all-row geometry and unchanged-gate audit of actual forward1/9."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'model')]
from tools.xkf_fusion.assess import Assessor,R2,CONDITIONS,DIRECTIONAL,angle,directional_reduce,near,require,assert_close,write,sha

MODES=('reference27','off_forward1','off_forward9','stable_forward9','hold')

class ActualAssessor(Assessor):
    def seed(self,seed):
        base=self.study/'direct_finetune';directory=base/f'actual_forward_raw_s{seed}'
        summary=self.read(directory/'summary.json');protocol=self.read(directory/'protocol.json')
        reference_dir=base/f'evaluations/raw/s{seed}';reference=self.read(reference_dir/'summary.json')
        source_protocol=self.read(reference_dir/'protocol.json');assessed=self.read(base/'assessment_raw.json')
        require(assessed['status']=='ASSESSED' and not assessed['missing'] and not assessed['validation_errors'],'reference assessment invalid')
        require(summary['protocol_sha256']==self.digest(directory/'protocol.json'),'actual protocol hash changed')
        require(protocol['independent_assessment_sha256']==self.digest(base/'assessment_raw.json'),'reference assessment hash changed')
        require(protocol['reference_summary_sha256']==self.digest(reference_dir/'summary.json') and protocol['reference_protocol_sha256']==self.digest(reference_dir/'protocol.json'),'reference manifests changed')
        for name,digest in protocol['source_sha256'].items():require(self.digest(name)==digest,'actual inference source changed: '+name)
        for name,digest in protocol['assessed_reference_input_sha256'].items():
            require(self.digest(name)==digest==assessed['input_sha256'][str(self.path(name).resolve())],'reference input identity differs')
        require(len(protocol['assessed_reference_input_sha256'])==114,'reference input set incomplete')
        require(protocol['clips']==source_protocol['clips']==self.old_schedule['clips'],'112 schedule differs')
        require(summary['clip_count']==112 and not summary['partial_smoke'] and not protocol['partial_smoke'],'partial actual forward')
        require(summary['modes']==protocol['modes']==list(MODES),'actual modes incomplete')
        require(summary['metrics']==protocol['metrics']==list(self.metrics),'metrics differ')
        require(protocol['conditions']==[list(v) for v in self.reducers['CONDITIONS']],'conditions differ')
        require(summary['identity']==protocol['identity'] and summary['identity']['seed']==seed and summary['identity']['global_step']==2400,'actual identity differs')
        for key in ('checkpoint_sha256','config_sha256'):
            require(summary['identity'][key]==reference['identity'][key],'reference checkpoint/config mismatch')
        require(self.digest(summary['identity']['checkpoint'])==summary['identity']['checkpoint_sha256'],'checkpoint changed')
        require(self.digest(summary['identity']['config'])==summary['identity']['config_sha256'],'config changed')
        require(summary['candidate_id']==protocol['candidate_id'],'candidate mismatch')
        name=summary['candidate_id'];refmi=reference['modes'].index(name);holdmi=reference['modes'].index('hold')
        require(name in assessed['seeds'][str(seed)]['candidates'],'unassessed reference candidate')
        require(len(list((directory/'raw').glob('*.npz')))==112 and len(summary['input_raw_sha256'])==112,'raw file count differs')
        records,directions,loops,parity=[],[],{},[];fkmax=0.
        for index,clip in enumerate(protocol['clips']):
            filename=f'{index:03d}_{clip["cohort"]}_run{clip["run_id"]}.npz'
            path=directory/'raw'/filename;oldpath=reference_dir/'raw'/filename;self.digest(path)
            require(self.digest(oldpath)==assessed['input_sha256'][str(oldpath.resolve())],'reference raw changed')
            actual_ref_hash={str(self.path(k).resolve()):v for k,v in summary['input_raw_sha256'].items()}
            require(actual_ref_hash[str(oldpath.resolve())]==self.digest(oldpath),'recorded actual input hash differs')
            with np.load(path,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
            with np.load(oldpath,allow_pickle=False) as z:old={k:z[k] for k in z.files}
            n=clip['observed_steps'];shapes={'predictions':(n,9,5,51),'metrics':(n,9,5,len(self.metrics)),
                'physical_wrist_error_xyz_mm':(n,9,5,3),'state_t_error_xyz_mm':(n,9,5,3),'initial':(9,51),
                'truth':(n,51),'clean':(51,),'packet_ids':(n,),'end_ms':(n,),'lnes_window_ms':(n,),'n_raw_events':(n,)}
            require(set(raw)==set(shapes),'actual raw field set differs')
            for key,shape in shapes.items():
                require(raw[key].shape==shape,'actual raw shape differs: '+key)
                if raw[key].dtype.kind in 'fc':require(np.isfinite(raw[key]).all(),'nonfinite actual raw: '+key)
            for key in ('initial','truth','clean','packet_ids','end_ms','lnes_window_ms','n_raw_events'):
                require(np.array_equal(raw[key],old[key]),'actual input changed: '+key)
            require(np.array_equal(raw['predictions'][:,:,0],old['predictions'][:,:,refmi]),'reference27 predictions changed')
            require(np.array_equal(raw['predictions'][:,:,-1],old['predictions'][:,:,holdmi]),'hold predictions changed')
            previous=np.concatenate([np.broadcast_to(raw['initial'][:,None],(9,5,51))[None],raw['predictions'][:-1]],0)
            for t in np.flatnonzero(raw['n_raw_events']==0):require(np.array_equal(raw['predictions'][t],previous[t]),'empty packet fails exact hold')
            item={'clip_id':clip['clip_id'],'stable9_bitwise_equal_actual1':bool(np.array_equal(raw['predictions'][:,:,3],raw['predictions'][:,:,1])),
                  'off9_bitwise_equal_actual1':bool(np.array_equal(raw['predictions'][:,:,2],raw['predictions'][:,:,1])),
                  'reference27_bitwise_equal_actual1':bool(np.array_equal(raw['predictions'][:,:,0],raw['predictions'][:,:,1]))}
            parity.append(item)
            beta_path=R2/f'evaluations/repair_s{seed}_step2200_direct/{clip["sequence"]}_W200.npz'
            if clip['sequence'] not in self.cache:
                self.digest(beta_path)
                with np.load(beta_path) as z:self.cache[clip['sequence']]=z['beta'].copy()
            beta=self.cache[clip['sequence']];p=raw['predictions'].reshape(-1,51)
            truth=np.broadcast_to(raw['truth'][:,None,None],(n,9,5,51)).reshape(-1,51).copy();values=[];wrist=[];states=[]
            with self.torch.no_grad():
                for begin in range(0,len(p),256):
                    v,w,s=self.frame_metrics(self.fk_model,self.torch.from_numpy(p[begin:begin+256]),self.torch.from_numpy(truth[begin:begin+256]),self.torch.from_numpy(beta))
                    values.append(v.numpy());wrist.append(w.numpy());states.append(s.numpy())
            values=np.concatenate(values).reshape(raw['metrics'].shape)
            near(raw['metrics'],values,'actual metric CPU-FK replay',atol=.01,rtol=5e-5)
            near(raw['physical_wrist_error_xyz_mm'],np.concatenate(wrist).reshape(n,9,5,3),'actual wrist CPU-FK replay',atol=.002,rtol=2e-5)
            near(raw['state_t_error_xyz_mm'],np.concatenate(states).reshape(n,9,5,3),'actual translation replay',atol=.0002,rtol=1e-6)
            fkmax=max(fkmax,float(np.abs(values-raw['metrics']).max()))
            for ci,condition in enumerate(CONDITIONS):
                for mi,mode in enumerate(MODES):
                    v=raw['metrics'][:,ci,mi];reduce=self.reducers['summarize_values']
                    records.append(dict(clip,condition=condition,mode=mode,mean=reduce(v),first=reduce(v[:1]),endpoint=reduce(v[-1:]),lnes_first_ms=int(raw['lnes_window_ms'][0])))
                    if condition in DIRECTIONAL and mode!='hold':
                        pred=raw['predictions'][-1]
                        if condition.startswith('z_'):
                            target=float(raw['physical_wrist_error_xyz_mm'][-1,ci,mi,2]);held=float(raw['physical_wrist_error_xyz_mm'][-1,ci,-1,2])
                            residue=float(raw['physical_wrist_error_xyz_mm'][-1,ci,mi,2]-raw['physical_wrist_error_xyz_mm'][-1,0,mi,2]);initial=float((raw['initial'][ci,2]-raw['initial'][0,2])*1000)
                        else:
                            target=float(angle(raw['truth'][-1,3:6],pred[ci,mi,3:6]));held=float(angle(raw['truth'][-1,3:6],pred[ci,-1,3:6]))
                            residue=float(angle(pred[0,mi,3:6],pred[ci,mi,3:6]));initial=float(angle(raw['initial'][0,3:6],raw['initial'][ci,3:6]))
                        directions.append(dict(clip,condition=condition,mode=mode,target_error=target,hold_target_error=held,perturbation_residue=residue,initial_perturbation=initial))
            if clip['cohort'].endswith('_bank') and clip['sequence'] not in loops:
                loops[clip['sequence']]={}
                for mi,mode in enumerate(MODES[:-1]):
                    ab=float(raw['metrics'][:,7,mi,0].mean());ha=float(raw['metrics'][:,7,-1,0].mean())
                    wr=float(np.linalg.norm(raw['physical_wrist_error_xyz_mm'][-1,7,mi]));hw=float(np.linalg.norm(raw['physical_wrist_error_xyz_mm'][-1,7,-1]))
                    loops[clip['sequence']][mode]={'clip_id':clip['clip_id'],'mean_abs':ab,'hold_abs':ha,'endpoint_wrist':wr,'hold_endpoint_wrist':hw,'mean_abs_pass':ab<=ha,'endpoint_wrist_pass':wr<hw}
        aggregate=self.reducers['aggregate'](records);assert_close(summary['aggregate'],aggregate,'actual aggregate raw replay')
        require(summary['all_stable9_bitwise_equal_actual_batch1']==all(r['stable9_bitwise_equal_actual1'] for r in parity),'producer parity summary differs')
        require(len(summary['parity'])==112 and [r['clip_id'] for r in summary['parity']]==[r['clip_id'] for r in parity],'producer parity rows incomplete')
        for a,b in zip(summary['parity'],parity):
            for mode,key in [('stable_forward9','stable9_bitwise_equal_actual1'),('off_forward9','off9_bitwise_equal_actual1'),('reference27','reference27_bitwise_equal_actual1')]:
                require(a[mode]['bitwise_equal']==b[key],'producer parity flag differs')
        ix={(r['cohort'],r['condition'],r['mode'],r['scope']):r for r in aggregate}
        control=self.read(R2/f'evaluations/control_s{seed}_step2200_rollout/summary.json')
        ctl={(r['cohort'],r['condition'],r['mode'],r['scope']):r for r in control['aggregate']};gates={}
        for mode in MODES[:-1]:
            guards={}
            for cohort,scope in [('local_bank','all_observed'),('local_cold','full_horizon')]:
                v=ix[(cohort,'clean',mode,scope)]['pooled_frames']['abs_mpjpe_mm'];limit=ix[(cohort,'clean','hold',scope)]['pooled_frames']['abs_mpjpe_mm']
                guards[cohort]={'value':v,'limit':limit,'pass':v<=limit}
            v=ix[('global_bank','clean',mode,'all_observed')]['pooled_frames']['abs_mpjpe_mm'];limit=ctl[('global_bank','clean','fused','all_observed')]['pooled_frames']['abs_mpjpe_mm']*1.1
            guards['global_bank']={'value':v,'limit':limit,'pass':v<=limit}
            dr=[directional_reduce(directions,c,cond,mode,scope,.9,.9) for c in ('local_bank','local_cold','local_warm3') for cond in DIRECTIONAL for scope in ('all_observed','full_horizon','short_only')]
            dr=[v for v in dr if v is not None];full=[v for v in dr if v['used_for_gate']];require(len(full)==18,'directional rows incomplete')
            gates[mode]={'clean_guards':guards,'clean_pass':all(v['pass'] for v in guards.values()),'directional_pass':all(v['pass'] for v in full),'directional':dr,'original_candidate_loop_checks':{seq:r[mode] for seq,r in loops.items()}}
        return {'identity':summary['identity'],'candidate_id':name,'parity':parity,'all_stable9_bitwise_equal_actual1':all(r['stable9_bitwise_equal_actual1'] for r in parity),
                'metric_max_cpu_gpu_difference':fkmax,'gates':gates,'aggregate':aggregate,'raw_geometry_replayed':True,
                'scope':'original thresholds; reference27 descriptive; actual batch1 anchors deployment; no formal G1-G4 certification'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    study=ROOT/'outputs/xkf/fusion_20261015_r3';a=ActualAssessor(study,study,True)
    setup=a.attempt('setup',a.setup);result={'kind':'INDEPENDENT_ACTUAL_FORWARD_GEOMETRY_ASSESSMENT','setup':setup,'seeds':{},'formal_gate_assessed':False,'automatic_full_training':False,'source_sha256':sha(__file__)}
    if setup:
        for seed in (3407,3408,3409):result['seeds'][str(seed)]=a.attempt(str(seed),lambda s=seed:a.seed(s))
    result.update(status='INVALID' if a.errors else 'INCOMPLETE' if a.missing or len(result['seeds'])!=3 or not all(result['seeds'].values()) else 'ASSESSED',validation_errors=a.errors,missing=a.missing,input_sha256=a.inputs)
    write(args.output,result);print(json.dumps({'status':result['status'],'errors':a.errors,'missing':a.missing}),flush=True)
    return 2 if a.errors else 0

if __name__=='__main__':raise SystemExit(main())
