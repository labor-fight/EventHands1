#!/usr/bin/env python3
"""Independent exact2400 D-only raw audit, preserving all original debug gates.

The seed reducer is an explicit copy of the frozen2200 assessor, with only
checkpoint/kind identity adapted. No fake2200 identity or BN protocol is used.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'model')]
from tools.xkf_fusion.assess import R2,CONDITIONS,DIRECTIONAL,angle,directional_reduce,assert_close,near,require,sha,write
from tools.xkf_fusion.assess_centroid import CentroidAssessor
from tools.xkf_fusion.evaluate_tuned_centroid import validate_spec,DECLARATION


class TunedCentroidAssessor(CentroidAssessor):
    def endpoint_identity(self,identity,protocol,summary,seed):
        require(identity['seed']==seed and identity['global_step']==2400,'wrong D-only seed/step identity')
        validate_spec(protocol['spec'],self.read(DECLARATION))
        path=self.path(identity['checkpoint']);finish=self.read(path.parent/'train_finish.json')
        require(finish['status']=='DEBUG200_COMPLETE' and finish['global_step']==2400 and finish['freeze_audit']=='PASS','D-only endpoint not complete')
        require(self.digest(path)==identity['checkpoint_sha256']==finish['checkpoint_sha256'],'endpoint checkpoint hash differs')
        require(self.digest(identity['config'])==identity['config_sha256'],'endpoint config changed')
        parent=self.parent['arms']['repair'][str(seed)]['training']
        require(finish['parent_checkpoint_sha256']==parent['checkpoint_sha256'],'different continuation parent')
        for name,digest in finish['source_sha256'].items():
            require(self.digest(name)==digest,'D-only training source changed: '+name)
        torch=self.torch
        saved=torch.load(path,map_location='cpu',weights_only=False)
        parent_path=R2/f'repair_s{seed}/xkf_repair_s{seed}-step=2200.ckpt'
        require(self.digest(parent_path)==parent['checkpoint_sha256'],'parent checkpoint changed')
        previous=torch.load(parent_path,map_location='cpu',weights_only=False)
        require(int(saved['global_step'])==2400 and int(previous['global_step'])==2200,'checkpoint clocks differ')
        clocks=saved['xkf_direct_continuation']
        require(clocks['i_adam_step']==2200 and clocks['d_adam_step']==2400 and clocks['all_bn_and_xkf_statistics_frozen'],'D-only clock metadata differs')
        current,old=saved['state_dict'],previous['state_dict']
        require(current.keys()==old.keys(),'D-only changed checkpoint structure')
        changes=[]
        for name,value in current.items():
            require(not value.is_floating_point() or bool(torch.isfinite(value).all()),'nonfinite checkpoint tensor')
            changed=not torch.equal(value,old[name])
            frozen=(not name.startswith('direct.') or name.endswith(('running_mean','running_var','num_batches_tracked')))
            require(not(frozen and changed),'frozen I/BN/EMA tensor changed: '+name)
            if changed:changes.append(name)
        require(bool(changes),'new D checkpoint has no changed parameter')
        order=self.read(path.parent/'resume_audit_rank0.json')['parameter_order']
        require(order==self.read(path.parent/'resume_audit_rank1.json')['parameter_order'] and len(order)==len(set(order))==96,'named Adam order incomplete')
        require(sum(name.startswith('direct.') for name in order)==60,'D parameter count changed')
        optimizer,old_optimizer=saved['optimizer_states'][0],previous['optimizer_states'][0]
        ids=optimizer['param_groups'][0]['params']
        require(len(optimizer['param_groups'])==1 and ids==old_optimizer['param_groups'][0]['params'] and len(ids)==96,'Adam slot identity/order changed')
        for name,key in zip(order,ids):
            state,old_state=optimizer['state'][key],old_optimizer['state'][key]
            require(state['exp_avg'].shape==current[name].shape,'named Adam moment shape differs')
            require(int(state['step'])==(2400 if name.startswith('direct.') else 2200),'actual Adam clock differs: '+name)
            if not name.startswith('direct.'):
                require(state.keys()==old_state.keys(),'I Adam structure changed')
                for field,value in state.items():
                    require(torch.equal(value,old_state[field]) if torch.is_tensor(value) else value==old_state[field],'I Adam moment changed: '+name+'/'+field)
        require(all(s['last_epoch']==2400 for s in saved['lr_schedulers']),'scheduler clock differs')
        parent_summary=self.read(R2/f'evaluations/repair_s{seed}_step2200_rollout/summary.json')
        near(summary['gains_fused'],parent_summary['gains_fused'],'frozen EMA gains',atol=0,rtol=0)
        del saved,previous
        return {'I_BN_EMA_bitwise_parent':True,'I_Adam_bitwise_parent':True,'all96_named_Adam_clocks_verified':True,
                'changed_direct_tensors':changes,'global_D_step':2400,'I_step':2200}

    def calibration(self,descriptor,seed,clips):
        if descriptor is None:return None
        require(self.digest(descriptor['path'])==descriptor['sha256'],'new affine descriptor changed')
        if descriptor['sha256'] in self.calibrations:return self.calibrations[descriptor['sha256']]
        path=self.path(descriptor['path']);doc=self.read(path)
        require(doc['seed']==seed and doc['fit_rows']==2304 and doc['partial_smoke'] is False,'partial/wrong new affine')
        require(doc['identity']['global_step']==2400 and doc['checkpoint_sha256']==self.active_identity['checkpoint_sha256'],'old/different D affine checkpoint')
        require(doc['identity']['config_sha256']==self.active_identity['config_sha256'],'new affine config differs')
        declaration=self.read(path.parent/'protocol.json')
        require(self.digest(path.parent/'protocol.json')==doc['protocol_sha256'],'new fit protocol changed')
        require(declaration['partial_smoke'] is False and declaration['fit_rows']==2304 and declaration['ridge']==.001,'new fit partial/changed')
        require(declaration['identity']==doc['identity'],'fit/export identity differs')
        for name,digest in declaration['source_sha256'].items():require(self.digest(name)==digest,'new fit source changed: '+name)
        manifest=ROOT/'data/hand_data51/splits_semkine.json'
        require(self.digest(manifest)==declaration['split_manifest_sha256'],'canonical split changed')
        names=self.read(manifest)['train']['trials']
        require(names==declaration['canonical_train_sequences'] and len(names)==72,'new fit noncanonical sequences')
        rawpath=path.parent/'train_predictions.npz'
        require(self.digest(rawpath)==doc['fit_source_sha256'],'new fit predictions changed')
        with np.load(rawpath,allow_pickle=False) as raw:
            require(raw['direct_state'].shape==raw['truth'].shape==(2304,51),'new fit rows incomplete')
            require(set(raw['sequence'])==set(names) and not any(x.startswith('zgz_') for x in raw['sequence']),'zgz leaked into fit')
            require(raw['sequence'].tolist()==[r['sequence'] for r in declaration['rows']] and raw['end_ms'].tolist()==[r['end_ms'] for r in declaration['rows']],'new fit row identities differ')
            x=raw['direct_state'][:,:3];residual=raw['truth'][:,:3]-x
            mean,scale=x.mean(0),np.maximum(x.std(0),1e-8);z=(x-mean)/scale;bias=residual.mean(0)
            weights=np.linalg.solve(z.T@z+len(x)*.001*np.eye(3),z.T@(residual-bias));delta=(weights/scale[:,None]).T
        near(doc['spec']['matrix'],np.eye(3)+delta,'new affine refit matrix',atol=1e-10,rtol=1e-10)
        near(doc['spec']['bias'],bias-delta@mean,'new affine refit bias',atol=1e-10,rtol=1e-10)
        result=dict(doc,clip_specs={c['clip_id']:doc['spec'] for c in clips});self.calibrations[descriptor['sha256']]=result
        return result

    def raw_verify(self,raw,old,protocol,clip,clip_index,beta):
        require('bn_calibration_by_seed' not in protocol['spec'],'new D cannot use BN adaptation')
        old={key:value.copy() for key,value in old.items()}
        for mode in ('d','fused','d_zero','fused_keep_t'):
            index=protocol['modes'].index(mode);old['predictions'][:,:,index]=raw['predictions'][:,:,index]
        return super().raw_verify(raw,old,protocol,clip,clip_index,beta)

    def seed(self, seed):
        directory=self.rollout_root/f's{seed}'
        summary=self.read(directory/'summary.json'); protocol=self.read(directory/'protocol.json')
        require(summary['protocol_sha256']==self.digest(directory/'protocol.json'),'run protocol hash mismatch')
        for key in ('source_sha256','spec_sha256','modes','metrics'):
            require(summary[key]==protocol[key],f'summary/protocol {key} mismatch')
        require(summary['kind']=='XKF_D_ONLY_CONTINUATION_FUSION_EXPLORATORY' and summary['partial'] is False and summary['clip_count']==112,'partial/wrong rollout')
        require(summary['formal_gate_assessed'] is False and summary['automatic_full_training'] is False
                and summary['finite_subset_reduction'] is False and summary['temperature_parameter_restored'] is True,'scope or restored parameter contract changed')
        require(protocol['clips']==self.old_schedule['clips'] and protocol['excluded']==self.old_schedule['excluded'],'112-clip schedule changed')
        require(protocol['conditions']==self.old_schedule['conditions'] and protocol['metrics']==list(self.metrics),'conditions/metrics changed')
        require(protocol['bank_sha256']==self.old_schedule['bank_sha256'],'bank changed')
        require(self.digest(protocol['spec_path'])==protocol['spec_sha256'],'candidate specification changed')
        spec=self.read(protocol['spec_path']);require(spec==protocol['spec'],'candidate spec payload differs')
        names=[x['id'] for x in spec['candidates']];modes=protocol['modes']
        require(modes==list(self.reducers['MODES'])+names and len(set(modes))==len(modes),'candidate/original modes incomplete')
        require(protocol['motion_modes']==['i','fused','fused_keep_t']+names,'motion mode order mismatch')
        require(protocol['seed']==seed and protocol['checkpoint_step']==2400,'seed/step mismatch')
        for name,digest in protocol['source_sha256'].items():
            require(self.digest(name)==digest,f'run source changed: {name}')
        identity=summary['identity']
        endpoint_audit=self.endpoint_identity(identity,protocol,summary,seed)
        self.active_identity=identity
        require(self.digest(identity['checkpoint'])==identity['checkpoint_sha256'] and self.digest(identity['config'])==identity['config_sha256'],'checkpoint/config changed')
        require(protocol['checkpoint_sha256']==identity['checkpoint_sha256'] and protocol['config_sha256']==identity['config_sha256'],'declared identity changed')
        require(len(protocol['calibration_sources'])==len(names),'calibration source count mismatch')
        for i,candidate in enumerate(spec['candidates']):
            doc=self.calibration(protocol['calibration_sources'][i],seed,protocol['clips'])
            for clip in protocol['clips']:
                resolved=protocol['resolved_clip_policies'][clip['clip_id']][i]
                expected={k:v for k,v in candidate.items() if k not in ('direct_correction_by_seed','direct_correction_file')}
                if doc is not None:expected['direct_correction']=doc['clip_specs'][clip['clip_id']]
                require(resolved==expected,'resolved clip calibration/policy changed')
        protocol=dict(protocol,base_gains=summary['gains_fused'])
        require(len(summary['parity'])==112 and [r['clip_id'] for r in summary['parity']]==[c['clip_id'] for c in protocol['clips']],'per-clip parity incomplete')
        rawfiles=list((directory/'raw').glob('*.npz')); require(len(rawfiles)==112,'missing/extra raw clips')
        records,directional,original_loops=[],[],{}
        fkmax=0.;counts=defaultdict(lambda:{'clips':0,'frames':0})
        for index,clip in enumerate(protocol['clips']):
            filename=f'{index:03d}_{clip["cohort"]}_run{clip["run_id"]}.npz'
            path=directory/'raw'/filename;self.digest(path)
            oldpath=R2/f'evaluations/repair_s{seed}_step2200_rollout/raw'/filename
            require(self.digest(oldpath)==self.parent['input_sha256'][str(oldpath.resolve())],'parent raw changed since assessment')
            with np.load(path,allow_pickle=False) as source:raw={k:source[k] for k in source.files}
            with np.load(oldpath,allow_pickle=False) as source:old={k:source[k] for k in source.files}
            directpath=R2/f'evaluations/repair_s{seed}_step2200_direct/{clip["sequence"]}_W200.npz'
            if clip['sequence'] not in self.cache:
                self.digest(directpath)
                with np.load(directpath,allow_pickle=False) as source:self.cache[clip['sequence']]=source['beta'].copy()
            fkmax=max(fkmax,self.raw_verify(raw,old,protocol,clip,index,self.cache[clip['sequence']]))
            for scope in ('all_observed','short_only' if clip['censored'] else 'full_horizon'):
                counts[(clip['cohort'],scope)]['clips']+=1;counts[(clip['cohort'],scope)]['frames']+=clip['observed_steps']
            for ci,condition in enumerate(CONDITIONS):
                for mi,mode in enumerate(modes):
                    values=raw['metrics'][:,ci,mi];summarize=self.reducers['summarize_values']
                    records.append(dict(clip,condition=condition,mode=mode,mean=summarize(values),first=summarize(values[:1]),endpoint=summarize(values[-1:]),lnes_first_ms=int(raw['lnes_window_ms'][0])))
                    if condition in DIRECTIONAL and mode in names:
                        hold=modes.index('hold');pred=raw['predictions'][-1]
                        if condition.startswith('z_'):
                            target=float(raw['physical_wrist_error_xyz_mm'][-1,ci,mi,2]);held=float(raw['physical_wrist_error_xyz_mm'][-1,ci,hold,2])
                            residue=float(raw['physical_wrist_error_xyz_mm'][-1,ci,mi,2]-raw['physical_wrist_error_xyz_mm'][-1,0,mi,2])
                            initial=float((raw['initial'][ci,2]-raw['initial'][0,2])*1000)
                        else:
                            target=float(angle(raw['truth'][-1,3:6],pred[ci,mi,3:6]));held=float(angle(raw['truth'][-1,3:6],pred[ci,hold,3:6]))
                            residue=float(angle(pred[0,mi,3:6],pred[ci,mi,3:6]));initial=float(angle(raw['initial'][0,3:6],raw['initial'][ci,3:6]))
                        directional.append(dict(clip,condition=condition,mode=mode,target_error=target,hold_target_error=held,perturbation_residue=residue,initial_perturbation=initial))
            if clip['cohort'].endswith('_bank') and clip['sequence'] not in original_loops:
                original_loops[clip['sequence']]={}
                for mode in names:
                    mi=modes.index(mode);hold=modes.index('hold');ci=7
                    candidate_abs=float(raw['metrics'][:,ci,mi,0].mean());hold_abs=float(raw['metrics'][:,ci,hold,0].mean())
                    wrist=float(np.linalg.norm(raw['physical_wrist_error_xyz_mm'][-1,ci,mi]));hold_wrist=float(np.linalg.norm(raw['physical_wrist_error_xyz_mm'][-1,ci,hold]))
                    original_loops[clip['sequence']][mode]={'clip_id':clip['clip_id'],'condition':CONDITIONS[ci],
                        'candidate_abs':candidate_abs,'hold_abs':hold_abs,'candidate_endpoint_wrist':wrist,'hold_endpoint_wrist':hold_wrist,
                        'mean_abs_pass':candidate_abs<=hold_abs,'endpoint_wrist_pass':wrist<hold_wrist}
        perclip=directory/'per_clip.jsonl';self.digest(perclip)
        assert_close([json.loads(x) for x in perclip.read_text().splitlines()],records,'perclip/raw reducers')
        aggregate=self.reducers['aggregate'](records);assert_close(summary['aggregate'],aggregate,'aggregate/raw reducers')
        short=self.reducers['aggregate']([r for r in records if r['censored']])
        for row in short:row['scope']='short_only'
        index={(r['cohort'],r['condition'],r['mode'],r['scope']):r for r in aggregate}
        control=self.read(R2/f'evaluations/control_s{seed}_step2200_rollout/summary.json')
        ctl={(r['cohort'],r['condition'],r['mode'],r['scope']):r for r in control['aggregate']}
        original=self.read(R2/f'evaluations/repair_s{seed}_step2200.json')
        candidates={}
        for name in names:
            guards={}
            for cohort,scope in (('local_bank','all_observed'),('local_cold','full_horizon')):
                value=index[(cohort,'clean',name,scope)]['pooled_frames']['abs_mpjpe_mm'];limit=index[(cohort,'clean','hold',scope)]['pooled_frames']['abs_mpjpe_mm']
                guards[cohort]={'value':value,'hold':limit,'pass':value<=limit}
            value=index[('global_bank','clean',name,'all_observed')]['pooled_frames']['abs_mpjpe_mm'];reference=ctl[('global_bank','clean','fused','all_observed')]['pooled_frames']['abs_mpjpe_mm']
            guards['global_bank']={'value':value,'control':reference,'limit':reference*1.1,'pass':value<=reference*1.1}
            directions=[directional_reduce(directional,cohort,condition,name,scope,.9,.9)
                        for cohort in ('local_bank','local_cold','local_warm3') for condition in DIRECTIONAL
                        for scope in ('all_observed','full_horizon','short_only')]
            directions=[v for v in directions if v is not None]
            full=[v for v in directions if v['used_for_gate']];require(len(full)==18,'directional gate cells missing')
            candidates[name]={'clean_guards':guards,'clean_pass':all(x['pass'] for x in guards.values()),
                'directional':directions,'directional_pass':all(x['pass'] for x in full),
                'directional_failed_cells':[{'cohort':x['cohort'],'condition':x['condition'],'target_fraction_of_hold':x['target_fraction_of_hold'],
                                            'residue_fraction_of_initial':x['residue_fraction_of_initial']} for x in full if not x['pass']],
                'original_candidate_loop_checks':{seq:v[name] for seq,v in original_loops.items()},
                'candidate_original_readiness':{'status':'INCOMPLETE','reason':'retained parent criteria are unchanged; no matching candidate initial/check-cohort evidence supplied',
                                               'criteria':original['readiness']['criteria']}}
            candidates[name]['additional_debug_status']='PASS' if candidates[name]['clean_pass'] and candidates[name]['directional_pass'] else 'FAIL'
        return {'identity':identity,'endpoint_freeze_audit':endpoint_audit,'candidates':candidates,'original_parent_readiness':original['readiness'],
                'raw_verification':{'complete':True,'clips':112,'conditions':9,'modes':modes,'all_metrics_cpu_fk_replayed':self.fk,
                                    'metric_max_cpu_gpu_difference':fkmax,'fusion_algebra_replayed':True,
                                    'neural_i_inference_replayed':False,'own_loop_input_contract':'source audit plus existing nonlinear recurrence unit test; each candidate uses own previous state'},
                'counts':[{'cohort':k[0],'scope':k[1],**v} for k,v in sorted(counts.items())],
                'aggregate':aggregate,'short_only_aggregate':short}

    def run(self):
        result=super().run()
        result['kind']='XKF_DONLY2400_CENTROID_RAW_ASSESSMENT'
        result['tuned_assessor_sha256']=sha(__file__)
        result['clock_scope']='D/global2400; I/BN/EMA frozen at parent2200'
        for seed in result['seeds'].values():
            if seed is None:continue
            seed['raw_verification']['fusion_algebra_replayed']='independent centroid FK or state-t algebra for every candidate'
            seed['raw_verification']['baseline_parity']='old2200 inputs/I/hold unchanged; D-based modes intentionally changed by D continuation'
            for row in seed['candidates'].values():
                row['candidate_original_readiness']['known_loop_failure']=any(not x['mean_abs_pass'] or not x['endpoint_wrist_pass'] for x in row['original_candidate_loop_checks'].values())
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=TunedCentroidAssessor(ROOT/'outputs/xkf/fusion_20261015_r3',args.rollout_dir.resolve(),True).run()
    write(args.output,result)
    print(json.dumps({'status':result['status'],'errors':result['validation_errors'],'missing':result['missing'],'candidates':result['all_three_seed_candidates']}),flush=True)
    return 2 if result['status']=='INVALID' else 0


if __name__=='__main__':raise SystemExit(main())
