#!/usr/bin/env python3
"""Describe frozen training validation flow failure by time/domain/motion.

No fit, changed gate, policy selection, development read, or network forward.
Separates transient endpoint correction from continued temporal residuals.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'model')]
import numpy as np
import torch
from tools.xkf_statistics import collect as C

SEEDS=(3407,3408,3409)
MODES=('baseline','identity_cv','family_cv')


def require(value,message):
    if not value:raise ValueError(message)


def aggregate(rows):
    require(rows,'empty aggregation')
    n=len(rows);energy=lambda key:float(sum(r[key] for r in rows))
    t=energy('true_energy');p=energy('pred_energy');dot=energy('pred_true_dot');a=dot/t if t else None
    ce,pe,cross=energy('current_error_energy'),energy('previous_error_energy'),energy('current_previous_dot')
    den=n*21.;flow=energy('flow_error_energy')
    independent=energy('replayed_flow_error_energy')
    perclip=defaultdict(list)
    for r in rows:perclip[r['clip']].append(r['flow_error_energy']/21.)
    return dict(frames=n,clips=len(perclip),joint_flow_rmse_mm=float(np.sqrt(flow/den)),
        projection=a,global_parallel_amplitude_error_rms_mm=None if a is None else float(np.sqrt((a-1)**2*t/den)),
        global_true_flow_orthogonal_residual_rms_mm=None if a is None else float(np.sqrt(max(0,p-dot*dot/t)/den)),
        current_endpoint_rmse_mm=float(np.sqrt(ce/den)),previous_endpoint_rmse_mm=float(np.sqrt(pe/den)),
        current_previous_error_normalized_dot=None if ce*pe==0 else cross/np.sqrt(ce*pe),
        current_error_energy_per_joint_mm2=ce/den,previous_error_energy_per_joint_mm2=pe/den,
        twice_current_previous_dot_per_joint_mm2=2*cross/den,
        replayed_flow_error_mse_mm2=independent/den,
        error_difference_identity_residual_mm2=(ce+pe-2*cross-independent)/den,
        actual_saved_vs_CPU_replayed_flow_mse_delta_mm2=(flow-independent)/den,
        current_abs_mpjpe_mm=float(np.mean([r['abs_mpjpe_mm'] for r in rows])),
        mean_true_speed_mm_s=float(np.mean([r['speed'] for r in rows])),
        mean_raw_count=float(np.mean([r['count'] for r in rows])),
        per_clip_flow_mse_mm2={k:float(np.mean(v)) for k,v in perclip.items()})


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'refusing overwrite');args.output.mkdir(parents=True)
    torch.set_num_threads(4);root=args.root.resolve();inputs={};sources={**C.sources(),str(Path(__file__).resolve()):C.sha(__file__)}
    def read(path):
        path=Path(path).resolve();digest=C.sha(path);require(str(path) not in inputs or inputs[str(path)]==digest,'changed input')
        inputs[str(path)]=digest;return json.loads(path.read_text())
    assessment=read(root/'assessment/result.json')
    require(assessment['status']=='REJECTED_TRAIN_VALIDATION' and len(assessment['verification'])==3,'complete rejected trial required')
    require(all(f['check']=='clean_high_motion_guard' for f in assessment['failures']),'unexpected rejection')
    declaration=C.load_declaration(Path(assessment['declaration_path']));inputs[assessment['declaration_path']]=assessment['declaration_sha256']
    schedule=read(declaration['validation_schedule_path'])['clips'];require(len(schedule)==72,'fixed72 required')
    cells=[];frame_rows=[];seed_verification={}
    for seed in SEEDS:
        model,cfg,identity=C.load_bound_model(declaration,seed,torch.device('cpu'));before=C.snapshot_model(model)
        folds=read(root/f'folds_s{seed}.json');summary=read(root/f'validation_s{seed}_shard0/summary.json')
        require(summary['identity']==identity and summary['status']=='COMPLETE_CONDITION_SHARD','wrong endpoint or incomplete shard')
        for key in ('config','checkpoint'):inputs[identity[key]]=identity[key+'_sha256']
        records=[];max_error=0.
        for ci,clip in enumerate(schedule):
            path=root/f'validation_s{seed}_shard0/raw/{ci:03d}_{clip["sequence"]}.npz';digest=C.sha(path)
            require(digest==summary['raw_sha256'][path.name]==assessment['input_sha256'][str(path)],'raw does not match independently audited input')
            inputs[str(path)]=digest
            with np.load(path,allow_pickle=False) as loaded:raw={k:loaded[k] for k in loaded.files}
            require(raw['condition_indices'][0]==0,'clean condition missing')
            pred=raw['predictions'][:,0];truth=raw['truth'];n=len(pred);beta=torch.from_numpy(raw['beta'].copy())
            def joints(states):
                flat=np.asarray(states).reshape(-1,51)
                return model._fk(torch.tensor(flat,dtype=torch.float32),beta[None].expand(len(flat),-1))[1].numpy().astype(float)
            jp=joints(pred).reshape(n,3,21,3);jt=joints(truth);jprev=joints(raw['previous_truth'])
            jin=joints(raw['incoming'][:,0]).reshape(n,3,21,3)
            current=(jp-jt[:,None])*1000;previous=(jin-jprev[:,None])*1000
            require(np.max(np.abs(previous[0]))<.0001,'clean initial not true previous pose')
            error=raw['joint_motion_error_mm'][:,0].astype(float)
            replay=current-previous;difference=float(np.max(np.abs(error-replay)));max_error=max(max_error,difference)
            require(difference<.005,'actual residual does not equal endpoint error difference')
            for ti in range(n):
                true=raw['true_joint_motion_mm'][ti].astype(float)
                for mi,mode in enumerate(MODES):
                    predicted=raw['joint_motion_mm'][ti,0,mi].astype(float);e=current[ti,mi];ep=previous[ti,mi]
                    records.append(dict(seed=seed,clip=clip['sequence'],identity=clip['training_identity'],
                        domain='local' if '_local' in clip['sequence'] else 'global',step=ti,mode=mode,
                        speed=float(raw['true_joint_speed_mm_s'][ti]),count=int(raw['n_raw_events'][ti]),
                        flow_error_energy=float(np.sum(error[ti,mi]**2)),replayed_flow_error_energy=float(np.sum(replay[ti,mi]**2)),
                        current_error_energy=float(np.sum(e*e)),previous_error_energy=float(np.sum(ep*ep)),
                        current_previous_dot=float(np.sum(e*ep)),true_energy=float(np.sum(true*true)),
                        pred_energy=float(np.sum(predicted*predicted)),pred_true_dot=float(np.sum(predicted*true)),
                        abs_mpjpe_mm=float(raw['metrics'][ti,0,mi,0]),
                        identity_high=bool(raw['true_joint_speed_mm_s'][ti]>=folds['identity'][clip['training_identity']]['high_motion_threshold_mm_s']),
                        family_high=bool(raw['true_joint_speed_mm_s'][ti]>=folds['trial'][clip['sequence']]['high_motion_threshold_mm_s'])))
        for scheme in MODES[1:]:
            for phase in ('all','prefix012','step0','step1','step2','main3to9','step3','step4','step5','step6','step7','step8','step9'):
                for domain in ('all','local','global'):
                    for motion in ('all','high','low'):
                        common=[r for r in records if (domain=='all' or r['domain']==domain)
                            and (phase=='all' or phase=='prefix012' and r['step']<3 or phase=='main3to9' and r['step']>=3
                                 or phase.startswith('step') and r['step']==int(phase[4:]))
                            and (motion=='all' or r['identity_high' if scheme=='identity_cv' else 'family_high']==(motion=='high'))]
                        paired={mode:[r for r in common if r['mode']==mode] for mode in ('baseline',scheme)}
                        if not paired['baseline']:
                            cells.append(dict(seed=seed,scheme=scheme,phase=phase,domain=domain,motion=motion,frames=0));continue
                        baseline,candidate=[aggregate(paired[mode]) for mode in ('baseline',scheme)]
                        require(baseline['frames']==candidate['frames'] and baseline['per_clip_flow_mse_mm2'].keys()==candidate['per_clip_flow_mse_mm2'].keys(),'paired denominator differs')
                        cells.append(dict(seed=seed,scheme=scheme,phase=phase,domain=domain,motion=motion,frames=baseline['frames'],
                            baseline=baseline,candidate=candidate,flow_rmse_ratio=candidate['joint_flow_rmse_mm']/baseline['joint_flow_rmse_mm'],
                            flow_mse_increase_mm2=candidate['joint_flow_rmse_mm']**2-baseline['joint_flow_rmse_mm']**2,
                            clips_worse=sum(candidate['per_clip_flow_mse_mm2'][c]>v for c,v in baseline['per_clip_flow_mse_mm2'].items()),
                            endpoint_mpjpe_change_mm=candidate['current_abs_mpjpe_mm']-baseline['current_abs_mpjpe_mm']))
        seed_verification[str(seed)]=dict(clips=72,observed_steps=690,max_saved_vs_replayed_flow_residual_mm=max_error,
            model_immutability=C.verify_model(model,before))
        frame_rows.extend(records);print(json.dumps(dict(seed=seed,frames=len(records),CPU_replay_maxdiff_mm=max_error)),flush=True)
    # Original six numerical guard cells must be reproduced exactly by raw data.
    for original in assessment['gates']['high_motion']:
        cell=next(x for x in cells if x['seed']==original['seed'] and x['scheme']==original['scheme'] and x['phase']=='all' and x['domain']=='all' and x['motion']=='high')
        for mode in ('baseline','candidate'):
            require(np.isclose(cell[mode]['joint_flow_rmse_mm'],original[mode]['joint_motion_rmse_mm'],atol=1e-7,rtol=1e-7)
                and np.isclose(cell[mode]['projection'],original[mode]['projection'],atol=1e-10,rtol=1e-10),'original guard reproduction differs')
    for path,digest in {**inputs,**sources}.items():require(C.sha(ROOT/path)==digest,'source/input changed at end')
    result=dict(kind='DESCRIPTIVE_FROZEN_VALIDATION_FLOW_STRATIFICATION',status='COMPLETE',formal=False,new_acceptance_gate=False,
        no_fitting=True,no_neural_forward=True,no_development_data_loaded=True,no_policy_selection=True,
        source_sha256=sources,input_sha256=inputs,verification=seed_verification,
        interpretation='All ten conditions stay in the original assessment. This mechanism diagnostic uses clean initial condition only. '
          'High/low uses each held-out fold original fit-side q75. No thresholds or gains are fitted. '
          'Squared flow error equals squared current endpoint error plus squared prior endpoint error minus their twice inner product. '
          'This is a fused-trajectory mechanism description, not identification of D noise.',cells=cells)
    frames=args.output/'frame_sufficient_statistics.json';frames.write_text(json.dumps(frame_rows,separators=(',',':'),allow_nan=False)+'\n')
    result['frame_sufficient_statistics']=dict(path=str(frames.resolve()),sha256=C.sha(frames))
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='COMPLETE',cells=len(cells))),flush=True)


if __name__=='__main__':main()
