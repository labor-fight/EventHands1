"""Independent CPU reducers for completed canonical I-motion probes; never fits a policy."""
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/xkf/fusion_20261015_r3/direct_finetune'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=BASE/'train_i_motion_independent_analysis';out.mkdir(exist_ok=True)
 reference=np.load(BASE/'trajectory1049_analysis/canonical2304_motion_gt.npz')
 records=[];sources={str(Path(__file__).relative_to(ROOT)):sha(__file__)};counts={}
 for seed in (3407,3408,3409):
  folder=BASE/f'train_i_motion_s{seed}';loaded=np.load(folder/'predictions.npz');a={key:loaded[key] for key in loaded.files};p=json.loads((folder/'protocol.json').read_text());result=json.loads((folder/'result.json').read_text())
  for name in ('predictions.npz','protocol.json','result.json'):sources[str((folder/name).relative_to(ROOT))]=sha(folder/name)
  assert result['status']=='COMPLETE' and not result['partial_smoke'] and result['D_forward_calls']==0
  assert result['sample_count']==2304 and result['condition_count']==10
  for key in ('previous_truth','truth','motion_delta','short_count','sequence','end_ms'):assert np.array_equal(a[key],reference[key]),key
  assert np.array_equal(a['prediction']-a['incoming'],a['predicted_motion'])
  assert np.array_equal(a['predicted_motion']-a['motion_delta'][:,None],a['motion_error'])
  eligible=np.array([row['stratum']=='full200' for row in p['rows']]);assert eligible.sum()==2016
  speed=np.linalg.norm(a['motion_delta'][:,:3],axis=-1)*1000/a['actual_label_delta_t_s']
  n=a['short_count'];counts[str(seed)]={'eligible':int(eligible.sum()),'cold':int((~eligible).sum()),'under50_eligible':int(((n<50)&eligible).sum()),'under50_cold':int(((n<50)&~eligible).sum())}
  bins={'all':np.ones(len(n),bool)}
  for lo,hi in ((0,50),(50,1000),(1000,2000),(2000,4000),(4000,5000),(5000,np.inf)):bins[f'count_{lo}_{hi}']=(n>=lo)&(n<hi)
  for lo,hi in ((0,20),(20,100),(100,300),(300,np.inf)):bins[f't_speed_mm_s_{lo}_{hi}']=(speed>=lo)&(speed<hi)
  for stratum,fullmask in {'all':np.ones(len(n),bool),'main_eligible2016':eligible,'cold288':~eligible}.items():
   for category in ('local','global'):
    for binname,binmask in bins.items():
     mask=fullmask&binmask&(a['category']==category)
     if not mask.any():continue
     delta=a['motion_delta'][mask].astype(float);zero_t=float(np.sqrt(np.mean(np.sum(delta[:,:3]**2,-1)))*1000)
     for ci,condition in enumerate(result['conditions']):
      err=a['motion_error'][mask,ci].astype(float);pred=a['predicted_motion'][mask,ci].astype(float)
      weighted=[30000*float(np.mean(err[:,:3]**2)),60*float(np.mean(err[:,3:6]**2)),450*float(np.mean(err[:,6:]**2))]
      records.append({'seed':seed,'stratum':stratum,'category':category,'bin':binname,'condition':condition,'n':int(mask.sum()),
       'motion_t_RMSE_mm':float(np.sqrt(np.mean(np.sum(err[:,:3]**2,-1)))*1000),'zero_motion_t_RMSE_mm':zero_t,
       'motion_t_axis_RMSE_mm':(np.sqrt(np.mean(err[:,:3]**2,0))*1000).tolist(),'motion_t_axis_bias_mm':(np.mean(err[:,:3],0)*1000).tolist(),
       'root_world_motion_RMSE_deg':float(np.sqrt(np.mean(a['root_motion_geodesic_error_deg'][mask,ci].astype(float)**2))),
       'zero_root_motion_RMSE_deg':float(np.sqrt(np.mean(reference['root_motion_deg'][mask]**2))),
       'weighted_MSE_terms_t_root_fingers_before_div51':weighted,'finger_weighted_loss_fraction':weighted[2]/sum(weighted),
       'endpoint_abs_mpjpe_mm':float(a['endpoint_metrics'][mask,ci,0].mean()),'hold_endpoint_abs_mpjpe_mm':float(a['hold_endpoint_metrics'][mask,ci,0].mean()),
       'translation_projection_energy_ratio':float(np.sum(pred[:,:3]*delta[:,:3])/np.sum(delta[:,:3]**2))})
 summary={'kind':'INDEPENDENT_TRAIN_I_MOTION_REDUCTION','formal':False,'scope':'CPU scoring only; no dev data, no fitting, no policy selection; endpoint errors at perturbed incoming are not motion-fit errors','source_sha256':sources,'counts':counts,'records':records,
  'conclusion':'Local clean true motion fitting is deficient even among main-eligible high-count packets. Large incoming errors add harm but are not the sole cause. Global fast translation is useful and must be preserved. Count shrink alone failed training held-out high-motion fidelity. This supports paired raw-vs-original-augmentation I-only continuation with unchanged objective and noise/selfburn; does not establish the outcome.',
  'loss_note':'Original local finger weighted term dominates; recorded as mechanism, not changed in the paired200 experiment.',
  'training_coverage':'2304 independent train-identity calibration packets include 288 truncated cold packets not original K4 main; 2016 main-eligible separated. Under50 evidence is entirely cold, so canonical sample cannot certify this deployed sparse regime.'}
 (out/'result.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'output':str(out/'result.json'),'records':len(records),'status':'COMPLETE'}))
if __name__=='__main__':main()
