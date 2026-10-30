import json,sys
from pathlib import Path
import numpy as np
sys.path[:0]=['.','model']
from tools.xkf_fusion.assess import angle,directional_reduce,DIRECTIONAL
B=Path('outputs/xkf/fusion_20261015_r3/direct_finetune/motion_gate');O={}
for seed in (3407,3408,3409):
 p=B/f's{seed}';pr=json.load(open(p/'protocol.json'));su=json.load(open(p/'summary.json'));modes=pr['modes'];names=modes[6:];recs=[];original={}
 for fi,clip in enumerate(pr['clips']):
  z=np.load(p/'raw'/f'{fi:03d}_{clip["cohort"]}_run{clip["run_id"]}.npz');pred=z['predictions'][-1];hold=modes.index('hold')
  for ci,c in enumerate([v[0] for v in pr['conditions']]):
   if c not in DIRECTIONAL:continue
   for mode in names:
    mi=modes.index(mode)
    if c.startswith('z_'):
     target=float(z['physical_wrist_error_xyz_mm'][-1,ci,mi,2]);held=float(z['physical_wrist_error_xyz_mm'][-1,ci,hold,2]);residue=float(z['physical_wrist_error_xyz_mm'][-1,ci,mi,2]-z['physical_wrist_error_xyz_mm'][-1,0,mi,2]);initial=float((z['initial'][ci,2]-z['initial'][0,2])*1000)
    else:
     target=float(angle(z['truth'][-1,3:6],pred[ci,mi,3:6]));held=float(angle(z['truth'][-1,3:6],pred[ci,hold,3:6]));residue=float(angle(pred[0,mi,3:6],pred[ci,mi,3:6]));initial=float(angle(z['initial'][0,3:6],z['initial'][ci,3:6]))
    recs.append(dict(clip,condition=c,mode=mode,target_error=target,hold_target_error=held,perturbation_residue=residue,initial_perturbation=initial))
  if fi in (64,88):
   original[clip['sequence']]={}
   for mode in names:
    mi=modes.index(mode);a=float(z['metrics'][:,7,mi,0].mean());h=float(z['metrics'][:,7,hold,0].mean());w=float(np.linalg.norm(z['physical_wrist_error_xyz_mm'][-1,7,mi]));wh=float(np.linalg.norm(z['physical_wrist_error_xyz_mm'][-1,7,hold]));original[clip['sequence']][mode]=dict(mean_joint=a,hold_mean_joint=h,endpoint_wrist=w,hold_endpoint_wrist=wh,mean_pass=a<=h,endpoint_pass=w<wh)
 ix={(v['cohort'],v['condition'],v['mode'],v['scope']):v for v in su['aggregate']};ctl=json.load(open(f'outputs/xkf/repair_20261015_r2/evaluations/control_s{seed}_step2200_rollout/summary.json'));ctlix={(v['cohort'],v['condition'],v['mode'],v['scope']):v for v in ctl['aggregate']}
 O[str(seed)]={}
 for mode in names:
  dirs=[directional_reduce(recs,co,c,mode,sc,.9,.9) for co in ('local_bank','local_cold','local_warm3') for c in DIRECTIONAL for sc in ('all_observed','full_horizon','short_only')];dirs=[v for v in dirs if v]
  guards=[]
  for co,sc in [('local_bank','all_observed'),('local_cold','full_horizon'),('global_bank','all_observed')]:
   val=ix[(co,'clean',mode,sc)]['pooled_frames']['abs_mpjpe_mm'];limit=ctlix[(co,'clean','fused',sc)]['pooled_frames']['abs_mpjpe_mm']*1.1 if co=='global_bank' else ix[(co,'clean','hold',sc)]['pooled_frames']['abs_mpjpe_mm'];guards.append(dict(cohort=co,value=val,limit=limit,passed=val<=limit))
  O[str(seed)][mode]=dict(clean_guards=guards,clean_pass=all(v['passed'] for v in guards),directions=dirs,directional_pass=all(v['pass'] for v in dirs if v['used_for_gate']),directional_failures=[v for v in dirs if v['used_for_gate'] and not v['pass']],original={co:v[mode] for co,v in original.items()})
  print(seed,mode,'clean',O[str(seed)][mode]['clean_pass'],'directions',O[str(seed)][mode]['directional_pass'],O[str(seed)][mode]['directional_failures'])
(B/'quickcheck.json').write_text(json.dumps(dict(scope='Producer-side check; independent raw/FK assessment pending; original criteria unchanged, no formal/full claim',seeds=O),indent=2)+'\n')
