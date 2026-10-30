#!/usr/bin/env python3
"""Read-only matched-population train D/I residual audit; never exports gains."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'model')]
BASE=ROOT/'outputs/xkf/fusion_20261015_r3/direct_finetune'
OUT=BASE/'incremental_finetune/paired_training_gain_audit'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def residual(x,y):
 e=(x-y).astype(float);e[:,3:6]=(Rotation.from_rotvec(y[:,3:6]).inv()*Rotation.from_rotvec(x[:,3:6])).as_rotvec();return e

def cross(a,b):
 v=a*b;return np.concatenate([v[:,:3],v[:,3:6].sum(1,keepdims=True),v[:,6:].reshape(-1,15,3).sum(2)],1)

def moment(a,b,mask):return cross(a[mask],b[mask]).mean(0)
def main():
 import torch
 from tools.xkf_fusion.fit_train_i_ramp import interpolate
 torch.set_num_threads(4);pre=json.loads((OUT/'prereg.json').read_text());require(pre['read_development_predictions'] is False and pre['coefficient_export'] is False,'wrong scope')
 inputs={str(OUT/'prereg.json'):sha(OUT/'prereg.json')};rows=[];checks={};raws={}
 for seed in (3407,3408,3409):
  paths={'old':ROOT/f'outputs/xkf/fusion_20261015_r3/bn/train_calibration_s{seed}','new':BASE/f'calibration_raw_s{seed}','I':BASE/f'train_i_motion_s{seed}'};data={};protocol={}
  for key,path in paths.items():
   names={'old':'train_predictions_alpha0.0.npz','new':'train_predictions.npz','I':'predictions.npz'}
   for name in ('protocol.json','result.json',names[key]):inputs[str(path/name)]=sha(path/name)
   protocol[key]=json.loads((path/'protocol.json').read_text())
   with np.load(path/names[key],allow_pickle=False) as z:data[key]={k:z[k].copy() for k in z.files}
   for file,digest in protocol[key]['source_sha256'].items():require(sha(ROOT/file if not Path(file).is_absolute() else file)==digest,'changed source '+file)
   identity=protocol[key]['identity']
   for name in ('config','checkpoint'):require(sha(identity[name])==identity[name+'_sha256'],'changed model '+key+'/'+name)
  require(protocol['old']['identity']['global_step']==2200 and protocol['new']['identity']['global_step']==protocol['I']['identity']['global_step']==2400,'wrong checkpoint clocks')
  require(protocol['old']['rows']==protocol['new']['rows']==protocol['I']['rows'],'row/window selection mismatch')
  for key in ('sequence','end_ms','truth'):
   require(np.array_equal(data['old'][key],data['new'][key]) and np.array_equal(data['new'][key],data['I'][key]),'unpaired sample '+key)
  a=data['I'];n=len(a['truth']);require(n==2304 and a['prediction'].shape==(2304,10,51),'incomplete sample')
  full=np.array([r['stratum']=='full200' for r in protocol['I']['rows']]);require(full.sum()==2016,'eligibility differs')
  saved=torch.load(protocol['I']['identity']['checkpoint'],map_location='cpu',weights_only=False);ema=saved['state_dict']['xkf_moments'].numpy();oldg=np.clip((ema[0]-ema[3])/np.maximum(ema[0]+ema[1]-2*ema[3],1e-12),0,1)
  eo=residual(data['old']['direct_state'],a['truth']);en=residual(data['new']['direct_state'],a['truth'])
  for key in ('old','new'):
   runtime='translation_affine_alpha0.0_runtime.json' if key=='old' else 'translation_affine_runtime.json';p=paths[key]/runtime;inputs[str(p)]=sha(p);d=json.loads(p.read_text());require(d['fit_source_sha256']==inputs[str(paths[key]/('train_predictions_alpha0.0.npz' if key=='old' else 'train_predictions.npz'))],'D raw fit source unbound')
  gate=np.minimum(a['short_count']/1000.,1.);gated=interpolate(a['incoming'],a['prediction'],np.repeat(gate,10))
  errors={name:residual(value.reshape(-1,51),np.repeat(a['truth'],10,axis=0)).reshape(n,10,51) for name,value in [('I_ungated',a['prediction']),('I_ramp1000',gated)]}
  checks[str(seed)]={'rows':n,'main_eligible':int(full.sum()),'cold':int((~full).sum()),'old_D_step':2200,'new_D_step':2400,'unchanged_I_step':2200,'same_row_truth_window':True,'D_batch_context_scope':'both raw D calibration batches; current I is realB1; no exact deployment covariance claim'}
  for stratum,mask in [('main_full200',full),('cold288',~full),('all2304',np.ones(n,bool))]:
   for category in ('all','local','global'):
    m=mask if category=='all' else mask&(a['category']==category)
    if not m.any():continue
    vo,vn=moment(eo,eo,m),moment(en,en,m)
    for group,columns in [('clean',[0]),('small_gaussian',[9]),('eight_large_equal',list(range(1,9))),('all10_equal',list(range(10)))]:
     for iname,ie in errors.items():
      ei=ie[:,columns].reshape(-1,51);m2=np.repeat(m,len(columns));edold=np.repeat(eo,len(columns),axis=0);ednew=np.repeat(en,len(columns),axis=0)
      vi=moment(ei,ei,m2);co=moment(ei,edold,m2);cn=moment(ei,ednew,m2)
      ko=np.clip((vi-co)/np.maximum(vi+vo-2*co,1e-12),0,1);kn=np.clip((vi-cn)/np.maximum(vi+vn-2*cn,1e-12),0,1)
      rows.append({'seed':seed,'stratum':stratum,'category':category,'initial_group':group,'I_stage':iname,'packets':int(m.sum()),'conditions':len(columns),'V_I':vi.tolist(),'V_D_old2200_same_rows':vo.tolist(),'V_D_new2400_same_rows':vn.tolist(),'C_ID_old':co.tolist(),'C_ID_new':cn.tolist(),'D_MSE_new_over_old':(vn/np.maximum(vo,1e-20)).tolist(),'K_diagnostic_old_D':ko.tolist(),'K_diagnostic_new_D':kn.tolist(),'frozen_training_EMA_K':oldg.tolist(),'frozen_EMA_VD':ema[1].tolist(),'hybrid_oldEMA_VI_CID_newVD_negative_determinant_groups':np.flatnonzero(ema[0]*vn-ema[3]**2<0).tolist()})
  raws[str(seed)]={'old_D_residual':eo,'new_D_residual':en,'I_endpoint_residual':errors['I_ungated'],'I1000_endpoint_residual':errors['I_ramp1000'],'full200':full,'category':a['category'],'sequence':a['sequence'],'end_ms':a['end_ms']}
 for seed,raw in raws.items():np.savez_compressed(OUT/f's{seed}_residuals.npz',**raw)
 require(all(sha(path)==digest for path,digest in inputs.items()),'input changed')
 result={'kind':pre['kind'],'status':'COMPLETE','formal':False,'coefficient_export':False,'training_or_policy_changed':False,'prereg_sha256':sha(OUT/'prereg.json'),'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},'input_sha256':inputs,'verification':checks,'records':rows,'raw_sha256':{str(OUT/f's{s}_residuals.npz'):sha(OUT/f's{s}_residuals.npz') for s in raws},'limits':['Old EMA used augmented self-burn rows; these canonical raw rows with declared initial states are a different population. Comparing old/new D here is paired; replacing original EMA from it is not justified.','K-star is a one-step group tangent linear residual diagnostic, not nonlinear centroid closed-loop optimum; all initial-state groups retained separately.','VI/VD/CID must be refreshed together on final deployed input population; mixing old moments with new VD can violate covariance consistency.']}
 write(OUT/'result.json',result);print(json.dumps({'status':'COMPLETE','records':len(rows)}))
if __name__=='__main__':main()
