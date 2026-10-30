#!/usr/bin/env python3
"""Independent training-only raw/geometry comparison for I2600 paired debug.

No development predictions, fitting, coefficient export or readiness inference.
Parent2304 and each new arm preserve2016 eligible/288cold and all ten initial states.
"""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'model')]
from tools.xkf_fusion.assess_i_continuation import ContinuationAssessor,BASE
from tools.xkf_fusion.assess import require,near,write,sha
from tools.xkf_repair.assess import extracted_functions,assert_close
from tools.xkf_fusion.train_i_motion_probe import CONDITIONS,COUNT_BINS,SPEED_BINS,GAUSSIAN_SEED,SMALL_STD,digest_arrays,reduce,rotations


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arms',nargs='+',choices=('parent','raw','aug'),default=['parent','raw','aug']);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 import torch,yaml
 from mano_layer import ManoLayer
 from pose_repr import decode_to_mano_inputs
 from semkine.lie import so3_exp,so3_log
 torch.set_num_threads(4);args.output.mkdir(parents=True,exist_ok=True)
 audit=ContinuationAssessor(BASE.parent,BASE/'incremental_finetune/evaluations',True);audit.torch=torch
 records=[];identities={};verification={};raw_hashes={};common=None;geometry=None;failed=[];missing=[]
 config_path=BASE/'raw_s3407/config_resolved.yaml';audit.digest(config_path);cfg=yaml.safe_load(config_path.read_text());audit.digest(cfg['MANO']['NPZ'])
 mano=ManoLayer(cfg['MANO']['NPZ'],add_mean=False).float().eval();mano.fast_chain=True
 class FK:
  def _fk(self,state,beta):
   decoded=decode_to_mano_inputs(state,cfg['MODEL']['POSE_REPR'],mano.hands_components,mano.hands_mean)
   return mano(beta,decoded['global_orient'],decoded['local_full_aa'],decoded['transl'])
 fkmodel=FK();frame_metrics=extracted_functions(ROOT/'tools/xkf/recovery_probe.py',{'frame_metrics'},{'torch':torch,'math':math,'so3_exp':so3_exp,'so3_log':so3_log})['frame_metrics']
 def fk(states,betas):
  shape=states.shape[:-1];x=states.reshape(-1,51);b=np.broadcast_to(betas[:,None,:],(*shape,10)).reshape(-1,10) if len(shape)==2 else betas;values=[]
  with torch.no_grad():
   for begin in range(0,len(x),256):values.append(fkmodel._fk(torch.from_numpy(x[begin:begin+256]),torch.from_numpy(b[begin:begin+256].copy()))[1].numpy())
  return np.concatenate(values).reshape(*shape,21,3)
 source={str(Path(__file__).relative_to(ROOT)):sha(__file__),**{name:sha(ROOT/name) for name in ('tools/xkf_fusion/train_i_motion_probe.py','tools/xkf_fusion/assess_i_continuation.py','model/mano_layer.py','model/pose_repr.py')}}
 for arm in args.arms:
  for seed in (3407,3408,3409):
   label=f'{arm}/{seed}'
   try:
    directory=BASE/f'train_i_motion_s{seed}' if arm=='parent' else BASE/'incremental_finetune/motion_probes'/arm/f's{seed}'
    protocol=audit.read(directory/'protocol.json');summary=audit.read(directory/'result.json');rawpath=directory/'predictions.npz'
    require(summary['status']=='COMPLETE' and not summary['partial_smoke'] and summary['sample_count']==2304 and summary['condition_count']==10,'partial/incomplete motion evidence')
    require(summary['protocol_sha256']==audit.digest(directory/'protocol.json') and summary['raw_sha256']==audit.digest(rawpath),'protocol/raw hash differs')
    require(summary['D_forward_calls']==0 and summary['actual_I_forward_calls']==23040 and summary['all_named_buffers_unchanged'],'wrong I-only calls/buffer contract')
    require(summary['conditions']==protocol['conditions']==list(CONDITIONS) and summary['metrics']==protocol['metrics'],'conditions/metrics differ')
    require(protocol['actual_inference_batch']==1 and protocol['stability_mode']=='per_sample_i' and protocol['no_fitting_or_gate_selection'] is True,'wrong forward/fitting scope')
    require(summary['identity']==protocol['identity'],'endpoint identity differs')
    for name,digest in {**protocol['source_sha256'],**protocol['input_sha256']}.items():require(audit.digest(name)==digest,'source/input changed '+name)
    if arm!='parent':
     require(protocol['kind']=='CANONICAL_I_CONTINUATION_MOTION_DIAGNOSIS' and protocol['arm']==summary['arm']==arm,'wrong continued motion kind/arm')
     declaration=BASE/'incremental_finetune/motion_probe_declaration.json';require(audit.digest(declaration)==protocol['declaration_sha256'],'motion declaration changed')
     for name,digest in audit.read(declaration)['source_sha256'].items():require(audit.digest(name)==digest,'declared source differs')
    # Reuse the separately reviewed named-Adam/tensor checker, never its dev setup.
    runtime=dict(summary,all_buffers_unchanged=summary['all_named_buffers_unchanged'])
    _,endpoint=audit.endpoint(summary['identity'],runtime,arm,seed)
    with np.load(rawpath,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    shapes={'incoming':(2304,10,51),'prediction':(2304,10,51),'previous_truth':(2304,51),'truth':(2304,51),'motion_delta':(2304,51),'endpoint_metrics':(2304,10,17),'hold_endpoint_metrics':(2304,10,17),'beta':(2304,10),'camera_K':(2304,3,3),'short_count':(2304,),'actual_label_delta_t_s':(2304,),'event_sha256':(2304,),'sequence':(2304,),'end_ms':(2304,),'previous_ms':(2304,),'category':(2304,),'small_gaussian_draw':(2304,51),'predicted_motion':(2304,10,51),'motion_error':(2304,10,51),'predicted_root_world_delta':(2304,10,3),'true_root_world_delta':(2304,3),'root_motion_geodesic_error_deg':(2304,10)}
    require(set(a)==set(shapes),'raw fields differ')
    for name,shape in shapes.items():
     require(a[name].shape==shape,'missing row/condition '+name)
     if a[name].dtype.kind in 'fc':require(np.isfinite(a[name]).all(),'nonfinite raw '+name)
    require(a['sequence'].tolist()==[r['sequence'] for r in protocol['rows']] and a['end_ms'].tolist()==[r['end_ms'] for r in protocol['rows']],'canonical row order differs')
    seqs=protocol['canonical_train_sequences'];require(len(seqs)==72 and set(a['sequence'])==set(seqs) and not any(x.startswith('zgz_') for x in seqs),'noncanonical training sequences')
    require(len({x.split('_')[0] for x in seqs})==9,'not nine identities')
    full=np.array([r['stratum']=='full200' for r in protocol['rows']]);require(full.sum()==2016 and (~full).sum()==288,'main/cold strata differ')
    require(np.array_equal(a['motion_delta'],a['truth']-a['previous_truth']) and np.array_equal(a['predicted_motion'],a['prediction']-a['incoming']),'motion delta mixed with initial correction')
    require(np.array_equal(a['motion_error'],a['predicted_motion']-a['motion_delta'][:,None]),'motion error differs')
    for key,value in zip(('predicted_root_world_delta','true_root_world_delta','root_motion_geodesic_error_deg'),rotations(a)):require(np.array_equal(a[key],value),'world-left root reducer differs '+key)
    assert_close(summary['summaries'],reduce(a),'raw producer summaries')
    require(summary['input_feature_sha256']==digest_arrays([a[k] for k in ('event_sha256','beta','camera_K','previous_truth','small_gaussian_draw')]),'feature digest differs')
    small=(np.random.default_rng(GAUSSIAN_SEED).normal(size=(2304,51))*SMALL_STD).astype(np.float32);require(np.array_equal(small,a['small_gaussian_draw']),'fixed Gaussian changed')
    require(np.array_equal(a['incoming'][:,0],a['previous_truth']) and np.array_equal(a['incoming'][:,9],a['previous_truth']+small),'clean/small initialization differs')
    reference=BASE/'trajectory1049_analysis/canonical2304_motion_gt.npz';audit.digest(reference)
    with np.load(reference,allow_pickle=False) as ref:
     for key,key2 in [('previous_truth','previous_truth'),('truth','truth'),('motion_delta','motion_delta'),('short_count','short_count'),('actual_label_delta_t_s','delta_t_s'),('sequence','sequence'),('end_ms','end_ms')]:require(np.array_equal(a[key],ref[key2]),'independent GT/domain differs '+key)
    pairedkeys=('incoming','truth','previous_truth','beta','camera_K','short_count','sequence','end_ms','previous_ms','category','event_sha256','actual_label_delta_t_s','small_gaussian_draw')
    if common is None:
     common={k:a[k].copy() for k in pairedkeys};incoming_j=fk(a['incoming'],a['beta']);truth_j=fk(a['truth'],a['beta']);previous_j=fk(a['previous_truth'],a['beta']);ideal_j=fk(a['incoming']+a['motion_delta'][:,None],a['beta']);gt_delta=(truth_j-previous_j)*1000
     speed=np.sqrt(np.mean(np.sum(gt_delta.astype(float)**2,-1),-1))/a['actual_label_delta_t_s'];high_cut=float(np.quantile(speed[full],.75));geometry=(incoming_j,truth_j,ideal_j,gt_delta,speed,high_cut)
    else:
     for key in pairedkeys:require(np.array_equal(a[key],common[key]),'arm/seed training inputs differ '+key)
    incoming_j,truth_j,ideal_j,gt_delta,speed,high_cut=geometry;pred_j=fk(a['prediction'],a['beta']);pred_delta=(pred_j-incoming_j)*1000;ideal_delta=(ideal_j-incoming_j)*1000
    physical_mse=np.mean(np.sum((pred_delta-gt_delta[:,None]).astype(float)**2,-1),-1);ideal_mse=np.mean(np.sum((pred_delta-ideal_delta).astype(float)**2,-1),-1);geometry_mse=np.mean(np.sum((ideal_delta-gt_delta[:,None]).astype(float)**2,-1),-1);dot=np.mean(np.sum(pred_delta*gt_delta[:,None],-1),-1);gtenergy=np.mean(np.sum(gt_delta.astype(float)**2,-1),-1)
    # Check physical endpoint geometry independently; wrist-relative measures retain original17 reducer validation above.
    abs_error=np.linalg.norm(pred_j-truth_j[:,None],axis=-1).mean(-1)*1000;near(a['endpoint_metrics'][:,:,0],abs_error,'all predicted endpoint joint FK',atol=.002,rtol=2e-5)
    held=np.linalg.norm(incoming_j-truth_j[:,None],axis=-1).mean(-1)*1000;near(a['hold_endpoint_metrics'][:,:,0],held,'all held endpoint joint FK',atol=.002,rtol=2e-5)
    derived=dict(physical_motion_MSE_mm2=physical_mse,ideal_clean_delta_motion_MSE_mm2=ideal_mse,ideal_geometry_floor_MSE_mm2=geometry_mse,physical_motion_projection_dot=dot,truth_motion_energy_mm2=gtenergy,truth_joint_speed_mm_s=speed,full200=full,sequence=a['sequence'],end_ms=a['end_ms'])
    outpath=args.output/f'{arm}_s{seed}_motion_scores.npz';np.savez_compressed(outpath,**derived);raw_hashes[str(outpath)]=sha(outpath)
    masks={'all':np.ones(2304,bool),'high_joint_motion_main_q75':speed>=high_cut}
    for lo,hi in COUNT_BINS:masks[f'count_{lo}_{hi}']=(a['short_count']>=lo)&(True if hi is None else a['short_count']<hi)
    t_speed=np.linalg.norm(a['motion_delta'][:,:3],axis=1)*1000/a['actual_label_delta_t_s']
    for lo,hi in SPEED_BINS:masks[f't_speed_{lo}_{hi}']=(t_speed>=lo)&(True if hi is None else t_speed<hi)
    for stratum,sm in [('all2304',np.ones(2304,bool)),('main_eligible2016',full),('cold288',~full)]:
     for category in ('all','local','global'):
      for binname,bm in masks.items():
       mask=sm&bm&(True if category=='all' else a['category']==category)
       for condition,columns in [('clean',[0]),('small_gaussian',[9]),('eight_large_equal',list(range(1,9))),('all10_equal',list(range(10)))]:
        row={'arm':arm,'seed':seed,'stratum':stratum,'category':category,'bin':binname,'initial_group':condition,'packets':int(mask.sum()),'conditions':len(columns)}
        if not mask.any():records.append(dict(row,metrics=None));continue
        error=a['motion_error'][mask][:,columns].astype(float);truth=a['motion_delta'][mask].astype(float)
        metrics={'pure_motion_t_RMSE_mm':float(np.sqrt(np.mean(np.sum(error[...,:3]**2,-1)))*1000),'zero_motion_t_RMSE_mm':float(np.sqrt(np.mean(np.sum(truth[:,:3]**2,-1)))*1000),'root_world_motion_RMSE_deg':float(np.sqrt(np.mean(a['root_motion_geodesic_error_deg'][mask][:,columns].astype(float)**2))),'finger_clean_delta_RMSE_rad':float(np.sqrt(np.mean(error[...,6:]**2))),'physical_joint_motion_RMSE_mm':float(np.sqrt(physical_mse[mask][:,columns].mean())),'vs_ideal_clean_delta_joint_motion_RMSE_mm':float(np.sqrt(ideal_mse[mask][:,columns].mean())),'ideal_clean_delta_geometry_floor_RMSE_mm':float(np.sqrt(geometry_mse[mask][:,columns].mean())),'joint_motion_projection_fraction':float(dot[mask][:,columns].mean(1).sum()/max(gtenergy[mask].sum(),1e-20)),'endpoint_abs_mm':float(abs_error[mask][:,columns].mean()),'hold_endpoint_abs_mm':float(held[mask][:,columns].mean())}
        records.append(dict(row,metrics=metrics))
    identities[label]=summary['identity'];verification[label]={'endpoint':endpoint,'all2304x10_complete':True,'main_eligible':2016,'cold':288,'D_calls':0,'I_calls':23040,'raw_reducers_and_joint_endpoint_geometry_recomputed':True,'high_motion_threshold_mm_s':high_cut};print(json.dumps({'verified':label}),flush=True)
   except FileNotFoundError as e:missing.append({'scope':label,'path':str(e)})
   except (ValueError,AssertionError,KeyError,IndexError,RuntimeError,TypeError) as e:failed.append({'scope':label,'error':type(e).__name__,'message':str(e)})
 indexed={(r['arm'],r['seed'],r['stratum'],r['category'],r['bin'],r['initial_group']):r for r in records};comparisons=[]
 if not failed and not missing and set(args.arms)=={'parent','raw','aug'}:
  for key,r in indexed.items():
   if key[0]!='raw' or r['metrics'] is None:continue
   parent=indexed[('parent',*key[1:])];aug=indexed[('aug',*key[1:])];require(r['packets']==parent['packets']==aug['packets'],'paired denominators differ')
   comparisons.append({k:v for k,v in r.items() if k not in ('arm','metrics')}|{'metrics':{m:{'parent':parent['metrics'][m],'raw':r['metrics'][m],'aug':aug['metrics'][m],'raw_minus_parent':r['metrics'][m]-parent['metrics'][m],'aug_minus_parent':aug['metrics'][m]-parent['metrics'][m],'raw_minus_aug':r['metrics'][m]-aug['metrics'][m]} for m in r['metrics']}})
 for name,digest in audit.inputs.items():
  if sha(name)!=digest:failed.append({'scope':'final_input_hash','path':name})
 for name,digest in source.items():require(sha(ROOT/name)==digest,'analysis source changed')
 result={'kind':'INDEPENDENT_CANONICAL_I_CONTINUATION_MOTION_COMPARISON','status':'INVALID' if failed else 'INCOMPLETE' if missing or len(verification)!=3*len(args.arms) else 'ASSESSED','formal':False,'automatic_full_training':False,'read_development_predictions':False,'no_fitting_or_policy_selection':True,'source_sha256':source,'input_sha256':audit.inputs,'derived_raw_sha256':raw_hashes,'verification':verification,'identities':identities,'records':records,'paired_comparisons':comparisons,'validation_errors':failed,'missing':missing,'scope':'single-step from declared GTprevious perturbations, training identities only; clean puremotion fitting and large-initial geometry separated; no own-loop or new-identity-generalization claim'}
 write(args.output/'result.json',result);print(json.dumps({'status':result['status'],'errors':failed,'missing':missing}),flush=True)
 return 2 if failed else 0
if __name__=='__main__':raise SystemExit(main())
