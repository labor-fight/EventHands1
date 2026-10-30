"""Fixed training-only, held-out, genuine B1 closed-loop statistics validation.

No fitting, model update, dev access or release decision occurs here. A condition
shard retains every clip/policy and is explicitly incomplete until both shards
are independently merged. The first three prefix steps are retained, not hidden.
"""
from __future__ import annotations
import argparse,hashlib,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'model')]
import numpy as np
import torch
from semkine.xkf_paired_statistics import PairedStatisticsPolicy,coefficient_vector,fit_paired_gains
from semkine.xkf_motion_gate import MotionCountFusionPolicy,interpolate_motion
from tools.xkf import recovery_probe as R
from tools.xkf_fusion.bn_probe import parameter_digest

MODES=('baseline','identity_cv','family_cv')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
 if not value:raise ValueError(message)
def write(path,value):
 temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def array_sha(value):
 a=np.ascontiguousarray(value);h=hashlib.sha256(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes());return h.hexdigest()
def resolved(path):
 path=Path(path);return path.resolve() if path.is_absolute() else (ROOT/path).resolve()
def verified_hashes(values):
 result={}
 for path,digest in values.items():
  path=resolved(path);require(sha(path)==digest,'bound input/source changed: '+str(path));result[str(path)]=digest
 return result
def load_folds(path,declaration,declaration_sha,seed,rows):
 value=json.loads(path.read_text());require(value['kind']=='PAIRED_STATISTICS_CROSSFIT_CANDIDATES' and value['status']=='FITTED_NOT_ACCEPTED' and value['accepted_for_dev'] is False and value['automatic_full_training'] is False,'unexpected fit lifecycle')
 require(value['seed']==seed and value['declaration_sha256']==declaration_sha,'fold declaration/seed mismatch')
 bound=verified_hashes(value['source_sha256'])|verified_hashes(value['input_sha256'])
 collection=resolved(value['collection_summary_path']);require(sha(collection)==value['collection_summary_sha256'],'collection summary changed')
 summary=json.loads(collection.read_text());require(summary['status']=='COMPLETE' and summary['partial_smoke'] is False and summary['rows']==2016 and summary['actual_forward_calls']==8064 and summary['all_buffers_unchanged'] is True,'partial/incomplete collection cannot fit validation policy')
 require(summary['seed']==seed and summary['declaration_sha256']==declaration_sha,'collection seed/declaration mismatch')
 model=declaration['models'][str(seed)];identity=summary['identity']
 require(identity['seed']==seed and identity['global_step']==2600 and all(identity[k+'_sha256']==model[k+'_sha256'] for k in ('checkpoint','config')),'collection endpoint differs')
 bound.update(verified_hashes(summary['source_sha256']));bound.update(verified_hashes(summary['input_sha256']))
 rawpath=resolved(summary['raw_path']);require(sha(rawpath)==summary['raw_sha256']==value['collection_raw_sha256'] and rawpath==resolved(value['collection_raw_path']),'collection raw changed')
 bound[str(rawpath)]=sha(rawpath)
 resultpath=resolved(summary['result_path']);require(sha(resultpath)==summary['result_sha256'],'collection result changed');bound[str(resultpath)]=sha(resultpath)
 result=json.loads(resultpath.read_text());require(result['raw_sha256']==summary['raw_sha256'] and result['partial_smoke'] is False and result['no_development_data_loaded'] is True and result['all_training_data_file_stats_unchanged'] is True,'collection scope mismatch')
 with np.load(rawpath,allow_pickle=False) as loaded:raw={k:loaded[k] for k in ('row_indices','sequence','e_i','e_d','gate','gt_joint_speed_mm_s')}
 indices=np.asarray(declaration['fit_row_indices']);require(np.array_equal(raw['row_indices'],indices) and np.array_equal(raw['sequence'],[rows[i]['sequence'] for i in indices]),'collection canonical row order differs')
 require(raw['e_i'].shape==raw['e_d'].shape==(2016,51) and raw['gate'].shape==raw['gt_joint_speed_mm_s'].shape==(2016,),'collection shapes differ')
 require(all(np.isfinite(raw[k]).all() for k in ('e_i','e_d','gate','gt_joint_speed_mm_s')) and (raw['gt_joint_speed_mm_s']>=0).all(),'invalid collection values')
 fit=set(declaration['fit_row_indices']);require(len(fit)==2016,'requires all2016 training anchors')
 identities={r['sequence'].split('_')[0] for r in rows};trials={r['sequence'] for r in rows};families={s.split('_v')[0] for s in trials}
 require(len(identities)==9 and len(trials)==72 and len(families)==18,'canonical train hierarchy differs')
 require(set(value['identity'])==identities and set(value['trial'])==trials,'fold coverage incomplete')
 coefficient_vector(value['baseline_effective_gains']);require(value['baseline_effective_gains']==summary['baseline_effective_gains'],'fit baseline changed')
 require(set(value['motion_family'])==families and value['identity_folds']==9 and value['motion_family_folds']==18 and value['sequence_lookup_aliases']==72,'fold denominator changed')
 verified_fits={}
 for scheme,folds in (('identity',value['identity']),('trial',value['trial'])):
  for name,fold in folds.items():
   require(fold['index_space']=='canonical2304','fold index convention differs')
   family=name.split('_v')[0]
   excluded={i for i in fit if (rows[i]['sequence'].split('_')[0]==name if scheme=='identity' else rows[i]['sequence'].split('_v')[0]==family)}
   require(len(fold['train_indices'])==len(set(fold['train_indices'])) and len(fold['excluded_indices'])==len(set(fold['excluded_indices'])),'duplicate fold rows')
   require(set(fold['excluded_indices'])==excluded and set(fold['train_indices'])==fit-excluded,'whole held-out population was not excluded')
   coefficient_vector(fold['coefficients']);threshold=fold['high_motion_threshold_mm_s'];require(isinstance(threshold,(float,int)) and math.isfinite(threshold) and threshold>=0,'invalid fit-side motion threshold')
   require(fold['fold_family']==(name if scheme=='identity' else family),'wrong held-out motion family')
   fitkey=(scheme,name if scheme=='identity' else family)
   if fitkey not in verified_fits:
    mask=np.isin(indices,fold['train_indices']);recomputed=fit_paired_gains(raw['e_i'][mask],raw['e_d'][mask],raw['gate'][mask],value['baseline_effective_gains'],**declaration['fitting'])
    require(np.allclose(recomputed['coefficients'],fold['coefficients'],rtol=1e-12,atol=1e-12) and np.array_equal(np.asarray(recomputed['coefficients'],np.float32),np.asarray(fold['coefficients'],np.float32)) and float(np.quantile(raw['gt_joint_speed_mm_s'][mask],.75))==threshold,'held-out coefficients or high-motion threshold differ from actual fit rows')
    verified_fits[fitkey]=True
   if scheme=='trial':require(fold==value['motion_family'][family],'trial alias differs from whole-family fit')
 for family in families:
  aliases=[value['trial'][name] for name in trials if name.split('_v')[0]==family]
  require(len(aliases)==4,'motion family must retain all four views')
  for other in aliases[1:]:
   require(all(other[k]==aliases[0][k] for k in ('coefficients','train_indices','excluded_indices','high_motion_threshold_mm_s')),'family aliases use different fit policies')
 return value,{**bound,str(path.resolve()):sha(path),str(collection.resolve()):sha(collection)}
def instrument(model,coefficients):
 spec=dict(id='paired_statistics',gain_scale=[1.,1.,1.],count_gate='ramp1000',motion_count_gate='ramp1000',translation_reference='joint_centroid',fixed_effective_gains=list(coefficient_vector(coefficients)))
 model.fusion_policy=PairedStatisticsPolicy(spec);policy=model.fusion_policy;original=policy.fuse;trace=[]
 def fuse(xi,zd,aux,previous,counts,base):
  result,direct,gains=original(xi,zd,aux,previous,counts,base)
  gate=(counts.float()/1000.).clamp(0.,1.)
  trace.append({k:v.detach().clone() for k,v in dict(incoming=previous,motion=xi,gated_motion=interpolate_motion(previous,xi,gate),direct=direct,effective_gains=gains,motion_gate=gate).items()})
  return result,direct,gains
 policy.fuse=fuse
 return trace
def clip_records(clip,raw,condition_indices,conditions,folds):
 records=[];steps=len(raw['end_ms']);is_full=steps==10
 for window,mask in (('all',np.ones(steps,bool)),('main_onward',np.arange(steps)>=3)):
  require(mask.any(),'main frame missing from fixed clip')
  for ci,condition_index in enumerate(condition_indices):
   for mi,mode in enumerate(MODES):
    metric=raw['metrics'][mask,ci,mi].astype(float);motion_error=raw['joint_motion_error_mm'][mask,ci,mi].astype(float)
    predicted=raw['joint_motion_mm'][mask,ci,mi].astype(float);truth=raw['true_joint_motion_mm'][mask].astype(float)
    energy=float(np.sum(truth**2));projection=None if energy==0 else float(np.sum(predicted*truth)/energy)
    item=dict(clip_id=clip['sequence'],training_identity=clip['training_identity'],family=clip['sequence'].split('_v')[0],category='local' if '_local' in clip['sequence'] else 'global',condition=conditions[condition_index],condition_index=condition_index,mode=mode,window=window,observed_steps=steps,scored_steps=int(mask.sum()),full10=is_full,censored=not is_full,
     metrics={name:float(metric[:,i].mean()) for i,name in enumerate(R.METRICS)},joint_motion_mse_mm2=float(np.mean(np.sum(motion_error**2,-1))),joint_motion_rmse_mm=float(np.sqrt(np.mean(np.sum(motion_error**2,-1)))),joint_motion_projection=projection,joint_jitter_mm_per_step=float(np.linalg.norm(predicted,axis=-1).mean()),gt_joint_move_mm_per_step=float(np.linalg.norm(truth,axis=-1).mean()),high_motion={})
    for scheme,fold in folds.items():
     selected=mask&(raw['true_joint_speed_mm_s']>=fold['high_motion_threshold_mm_s']);n=int(selected.sum())
     if n:
      e=raw['joint_motion_error_mm'][selected,ci,mi].astype(float);p=raw['joint_motion_mm'][selected,ci,mi].astype(float);t=raw['true_joint_motion_mm'][selected].astype(float);den=float(np.sum(t*t))
      item['high_motion'][scheme]=dict(n=n,threshold_mm_s=fold['high_motion_threshold_mm_s'],joint_motion_mse_mm2=float(np.mean(np.sum(e*e,-1))),joint_motion_projection=None if den==0 else float(np.sum(p*t)/den))
     else:item['high_motion'][scheme]=dict(n=0,threshold_mm_s=fold['high_motion_threshold_mm_s'],joint_motion_mse_mm2=None,joint_motion_projection=None)
    records.append(item)
 return records

@torch.no_grad()
def main():
 from tools.xkf_statistics import collect as C
 parser=argparse.ArgumentParser(description=__doc__)
 for key in ('declaration','fold-specs','output'):parser.add_argument('--'+key,type=Path,required=True)
 parser.add_argument('--seed',type=int,required=True,choices=(3407,3408,3409));parser.add_argument('--device',default='cuda:0')
 parser.add_argument('--condition-shard',type=int,choices=(0,1));parser.add_argument('--limit-clips',type=int,default=0)
 args=parser.parse_args();require(args.limit_clips>=0,'negative clip limit');require(not args.output.exists(),'refusing overwrite')
 declaration=C.load_declaration(args.declaration);require(declaration['status']=='ACTIVE_TRAIN_ONLY_POLICY_TRIAL' and declaration.get('read_development_data') is False,'inactive/non-training declaration')
 torch.set_num_threads(4);device=torch.device(args.device);model,cfg,identity=C.load_bound_model(declaration,args.seed,device)
 dataset,rows,fit_indices,draws=C.training_inputs(cfg,declaration)
 require(list(fit_indices)==declaration['fit_row_indices'],'fit index order differs')
 schedulepath=Path(declaration['validation_schedule_path']);require(sha(schedulepath)==declaration['validation_schedule_sha256'],'schedule changed')
 schedule=json.loads(schedulepath.read_text());clips=schedule['clips'];require(len(clips)==72 and sum(len(c['planned_end_ms']) for c in clips)==690,'full fixed72/690 schedule required')
 require(sum(len(c['planned_end_ms'])<10 for c in clips)==5,'short clip denominator changed')
 conditions=declaration['conditions'];require(conditions==[x[0] for x in R.CONDITIONS]+['small_gaussian'],'conditions differ')
 condition_indices=list(range(10)) if args.condition_shard is None else list(range(args.condition_shard*5,args.condition_shard*5+5))
 selected=clips[:args.limit_clips] if args.limit_clips else clips
 folds,inputs=load_folds(args.fold_specs,declaration,sha(args.declaration),args.seed,rows)
 inputs.update({str(args.declaration.resolve()):sha(args.declaration),str(schedulepath):sha(schedulepath),**declaration['input_sha256']})
 for entry in declaration['models'].values():inputs.update({entry['config']:entry['config_sha256'],entry['checkpoint']:entry['checkpoint_sha256']})
 inputs=verified_hashes(inputs)
 sources=verified_hashes({**C.sources(),**declaration['source_sha256'],str(Path(__file__).resolve()):sha(__file__),str(ROOT/'semkine/xkf_paired_statistics.py'):sha(ROOT/'semkine/xkf_paired_statistics.py')})
 parameters=parameter_digest(model);buffers={name:value.detach().cpu().clone() for name,value in model.named_buffers()};require(len(buffers)==65,'named buffer coverage changed')
 baseline=np.asarray(folds['baseline_effective_gains'],np.float32)
 effective=(model.gains().float()*model.gains().new_tensor([.5,.5,.5,.25]+[.5]*15)).detach().cpu().numpy()
 require(np.array_equal(baseline,effective),'baseline effective coefficients do not match frozen moments/scales')
 args.output.mkdir(parents=True);(args.output/'raw').mkdir()
 protocol=dict(kind='TRAIN_ONLY_PAIRED_STATISTICS_ACTUAL_OWN_LOOP',formal=False,automatic_full_training=False,identity=identity,source_sha256=sources,input_sha256=inputs,declaration_sha256=sha(args.declaration),fold_specs_sha256=sha(args.fold_specs),conditions=conditions,condition_indices=condition_indices,partial_shard=args.condition_shard is not None,condition_shard=args.condition_shard,partial_smoke=bool(args.limit_clips),clips=selected,modes=MODES,metrics=R.METRICS,inference_batch_size=1,uncached_D=True,coefficients_fixed_per_clip=True,primary='small_gaussian, main-onward step>=3; clip mean then identity mean; merger computes nine-identity paired SE after all three seeds',all_full_short_retained=True)
 write(args.output/'protocol.json',protocol);all_records=[];rawhashes={};actual_calls=0;Dcalls=0;started=time.monotonic();parity=None;data_signatures={};domain_hashes={};handles={}
 def count_D(*unused):
  nonlocal Dcalls
  Dcalls+=1
 hook=model.direct.register_forward_pre_hook(count_D)
 for clip_index,clip in enumerate(selected):
  anchor_index=clip['canonical2304_row_index'];anchor=rows[anchor_index]
  require(anchor==clip['canonical_anchor'] and anchor_index in fit_indices,'canonical anchor changed')
  require(not clip['sequence'].startswith('zgz') and anchor['sequence']==clip['sequence'] and clip['training_identity']==clip['sequence'].split('_')[0],'non-training sequence/identity')
  handle=dataset._handle(anchor['sequence_id']);beta=torch.tensor(handle.betas.copy(),device=device).float()[None];camera=torch.tensor(handle.camera_K.copy(),device=device).float()[None]
  handles[handle.seq]=handle;data_signatures[handle.seq]=C.data_file_signature(handle);domain_hashes[handle.seq]=C.array_sha(handle.pos51,handle.betas,handle.camera_K,handle.valid_runs)
  initial_ms=clip['declared_initial_gt_ms'];ends=clip['planned_end_ms'];require(initial_ms==anchor['end_ms']-199 and ends==list(range(anchor['end_ms']-150,min(anchor['end_ms']+301,anchor['run_stop_ms']),50)),'fixed clip clocks changed')
  require(len(ends)==clip['observed_steps_from_frozen_valid_run_metadata'],'censored length changed')
  clean=torch.from_numpy(handle.pos51[initial_ms].copy()).to(device).float();initial10=torch.cat((R.initial_states(clean),(clean+torch.as_tensor(draws[anchor_index],device=device))[None]))
  initial=initial10[condition_indices];previous=initial[:,None].expand(-1,3,-1).clone();previous_truth=clean[None]
  fold_for_clip={'identity_cv':folds['identity'][clip['training_identity']],'family_cv':folds['trial'][clip['sequence']]}
  coefficients=[folds['baseline_effective_gains'],fold_for_clip['identity_cv']['coefficients'],fold_for_clip['family_cv']['coefficients']]
  raw={key:[] for key in ('incoming','predictions','truth','previous_truth','metrics','motion_predictions','gated_motion_predictions','direct_predictions','effective_gains','motion_gate','joint_motion_mm','joint_motion_error_mm','true_joint_motion_mm','true_joint_speed_mm_s','n_raw_events','lnes_window_ms','end_ms','previous_gt_ms','input_sha256','raw_event_sha256','lnes_sha256')}
  C.reset(model)
  for ti,end in enumerate(ends):
   packet,observation=C.packet_for_end(handle,end,anchor['run_start_ms'],previous[:1,0],beta,camera)
   require(packet.batch_size==1 and not bool(packet.target.any()),'inference target or batch violation')
   if parity is None:
    model.fusion_policy=MotionCountFusionPolicy(declaration['baseline_policy']);C.reset(model);old=model.forward_packet(packet)
    instrument(model,coefficients[0]);C.reset(model);new=model.forward_packet(packet)
    require(torch.equal(old,new),'original baseline and explicit coefficients differ on real B1')
    parity=dict(bitwise_equal=True,device=str(device),full_forward_calls=2);Dcalls=0
   traces=[];outputs=[];incoming=previous.clone()
   for mi,coefficient in enumerate(coefficients):
    trace=instrument(model,coefficient);out=[]
    for ci in range(len(condition_indices)):
     packet.prev_state=previous[ci:ci+1,mi];C.reset(model);out.append(model.forward_packet(packet));actual_calls+=1
    require(len(trace)==len(condition_indices) and all(v['motion'].shape==(1,51) for v in trace),'trace is not true B1')
    require(all(torch.equal(trace[ci]['incoming'],previous[ci:ci+1,mi]) for ci in range(len(condition_indices))),'candidate used another trajectory incoming state')
    traces.append({key:torch.cat([v[key] for v in trace]) for key in trace[0]});outputs.append(torch.cat(out))
   previous=torch.stack(outputs,1)
   # This is the first access to the current label; every candidate already ran.
   truth=torch.from_numpy(handle.pos51[end].copy()).to(device).float()[None]
   metric,_,_=R.frame_metrics(model,previous.reshape(-1,51),truth,beta)
   _,pred_j=model._fk(previous.reshape(-1,51),beta.expand(len(condition_indices)*3,-1));_,incoming_j=model._fk(incoming.reshape(-1,51),beta.expand(len(condition_indices)*3,-1))
   _,true_j=model._fk(truth,beta);_,old_true_j=model._fk(previous_truth,beta)
   motion=(pred_j-incoming_j).reshape(len(condition_indices),3,21,3)*1000;true_motion=(true_j-old_true_j)[0]*1000
   prevms=initial_ms if ti==0 else ends[ti-1];speed=true_motion.norm(dim=-1).mean()/((end-prevms)/1000.)
   values=dict(incoming=incoming,predictions=previous,truth=truth[0],previous_truth=previous_truth[0],metrics=metric.reshape(len(condition_indices),3,-1),joint_motion_mm=motion,joint_motion_error_mm=motion-true_motion,true_joint_motion_mm=true_motion,true_joint_speed_mm_s=speed)
   for key,tracekey in (('motion_predictions','motion'),('gated_motion_predictions','gated_motion'),('direct_predictions','direct'),('effective_gains','effective_gains'),('motion_gate','motion_gate')):values[key]=torch.stack([t[tracekey] for t in traces],1)
   for key,value in values.items():require(bool(torch.isfinite(value).all()),'nonfinite full trajectory '+key);raw[key].append(value.cpu().numpy())
   for key,value in dict(n_raw_events=int(packet.counts[0]),lnes_window_ms=float(packet.lnes_window_ms[0]),end_ms=end,previous_gt_ms=prevms,raw_event_sha256=observation['event_sha256'],lnes_sha256=observation['lnes_sha256'],input_sha256=hashlib.sha256((json.dumps(observation,sort_keys=True)+array_sha(packet.events.cpu().numpy())+array_sha(packet.lnes.cpu().numpy())+array_sha(packet.camera_K.cpu().numpy())+array_sha(packet.betas.cpu().numpy())).encode()).hexdigest()).items():raw[key].append(value)
   if int(packet.counts[0])==0:require(torch.equal(previous,incoming),'zero-count packet must hold every policy')
   previous_truth=truth
  raw={key:np.stack(value) for key,value in raw.items()};raw.update(initial=initial.cpu().numpy(),clean=clean.cpu().numpy(),beta=handle.betas.copy(),camera_K=handle.camera_K.copy(),condition_indices=np.array(condition_indices),coefficients=np.array(coefficients),gaussian_draw=np.asarray(draws[anchor_index]))
  path=args.output/'raw'/f'{clip_index:03d}_{clip["sequence"]}.npz';np.savez_compressed(path,**raw);rawhashes[path.name]=sha(path)
  all_records.extend(clip_records(clip,raw,condition_indices,conditions,fold_for_clip));write(args.output/'progress.json',dict(completed_clips=clip_index+1,total_clips=len(selected),actual_full_B1_calls=actual_calls,elapsed_seconds=time.monotonic()-started))
  if (clip_index+1)%8==0:print(json.dumps(dict(seed=args.seed,shard=args.condition_shard,clips=clip_index+1)),flush=True)
 hook.remove();require(actual_calls==Dcalls==sum(len(c['planned_end_ms']) for c in selected)*len(condition_indices)*3,'actual forward count incomplete or D cached')
 require(parameter_digest(model)==parameters and all(torch.equal(value.cpu(),buffers[name]) for name,value in model.named_buffers()),'model parameters/buffers changed')
 for name,handle in handles.items():require(C.data_file_signature(handle)==data_signatures[name] and C.array_sha(handle.pos51,handle.betas,handle.camera_K,handle.valid_runs)==domain_hashes[name],'training data/context changed: '+name)
 for path,digest in {**inputs,**sources}.items():require(sha(path)==digest,'input/source changed: '+path)
 with (args.output/'per_clip.jsonl').open('w') as stream:
  for record in all_records:stream.write(json.dumps(record,allow_nan=False)+'\n')
 write(args.output/'summary.json',dict(kind=protocol['kind'],status='COMPLETE_SMOKE' if args.limit_clips else ('COMPLETE_CONDITION_SHARD' if args.condition_shard is not None else 'COMPLETE'),formal=False,automatic_full_training=False,identity=identity,declaration_sha256=sha(args.declaration),fold_specs_sha256=sha(args.fold_specs),protocol_sha256=sha(args.output/'protocol.json'),condition_indices=condition_indices,partial_shard=args.condition_shard is not None,partial_smoke=bool(args.limit_clips),clips=len(selected),observed_steps_per_condition=sum(len(c['planned_end_ms']) for c in selected),modes=MODES,actual_full_B1_calls=actual_calls,actual_D_forward_calls=Dcalls,baseline_real_B1_parity=parity,source_sha256=sources,input_sha256=inputs,raw_sha256=rawhashes,all_parameters_unchanged_sha256=parameters,all65_named_buffers_unchanged=True,named_buffer_sha256={name:C.array_sha(value.numpy()) for name,value in buffers.items()},training_domain_sha256=domain_hashes,data_file_stat_identity=data_signatures,all_training_data_file_stats_unchanged=True,no_development_data_loaded=True,finite_subset_reduction=False,per_clip_sha256=sha(args.output/'per_clip.jsonl'),elapsed_seconds=time.monotonic()-started,acceptance='not evaluated by a shard; complete10 conditions and all three seeds required'))
if __name__=='__main__':main()
