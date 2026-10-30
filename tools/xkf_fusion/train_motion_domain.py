#!/usr/bin/env python3
"""CPU-only GT motion-domain audit of the already frozen canonical2304 rows."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'model')]
import numpy as np,torch,yaml
from scipy.spatial.transform import Rotation
from tools.xkf_fusion import bn_probe as B
BASE=ROOT/'outputs/xkf/fusion_20261015_r3/direct_finetune';OUT=BASE/'trajectory1049_analysis'
def main():
 cfg=yaml.safe_load((BASE/'raw_s3409/config_resolved.yaml').read_text());ds=B.calibration_dataset(cfg);rows=B.calibration_rows(ds)
 previous=[];truth=[];counts=[];dt=[];roots=[];joint=[];cat=[];camera_flow=[]
 for row in rows:
  h=ds._handle(row['sequence_id']);end=row['end_ms'];start=max(row['run_start_ms'],end-49);prev=max(row['run_start_ms'],end-50)
  a=h.pos51[prev].astype(np.float32);b=h.pos51[end].astype(np.float32);previous.append(a);truth.append(b);dt.append((end-prev)/1000);cat.append(h.category)
  ev=ds._raw_events(h,end,end-start+1);counts.append(len(ev[0]));roots.append(np.degrees((Rotation.from_rotvec(a[3:6]).inv()*Rotation.from_rotvec(b[3:6])).magnitude()))
  K=h.camera_K
  def proj(t):
   x=t+h.j0;px=K@x;return px[:2]/px[2]
  camera_flow.append(proj(b[:3])-proj(a[:3]))
 previous=np.stack(previous);truth=np.stack(truth);counts=np.array(counts);change=(truth[:,:3]-previous[:,:3])*1000;root=np.array(roots)
 frozen=np.load(BASE/'calibration_raw_s3409/train_predictions.npz')
 assert np.array_equal(truth,frozen['truth']);assert np.array_equal(np.array([r['sequence'] for r in rows]),frozen['sequence']);assert np.array_equal(np.array([r['end_ms'] for r in rows]),frozen['end_ms'])
 summaries={}
 for label,mask in [('all',np.ones(len(rows),bool)),('local',np.array(['local' in r['sequence'] for r in rows])),('global',np.array(['global' in r['sequence'] for r in rows]))]:
  assert mask.any();x=change[mask];c=counts[mask];summaries[label]=dict(n=int(mask.sum()),mean_delta_xyz_mm=x.mean(0).tolist(),rms_delta_xyz_mm=np.sqrt((x*x).mean(0)).tolist(),median_motion_mm=float(np.median(np.linalg.norm(x,axis=1))),p90_motion_mm=float(np.percentile(np.linalg.norm(x,axis=1),90)),mean_root_deg=float(root[mask].mean()),count_under50=int((c<50).sum()),count_under1000=int((c<1000).sum()),positive_x_fraction=float((x[:,0]>0).mean()))
 np.savez_compressed(OUT/'canonical2304_motion_gt.npz',previous_truth=previous,truth=truth,motion_delta=truth-previous,short_count=counts,delta_t_s=np.array(dt),root_motion_deg=root,wrist_camera_flow_px=np.array(camera_flow),sequence=frozen['sequence'],end_ms=frozen['end_ms'])
 result=dict(scope='CPU labels only, no neural inference, no fitting/training. Frozen canonical9/72×32 rows and truth byte-equal existing D calibration.',config_noise=cfg['TRACK'],sample_count=len(rows),subject_count=len({r['sequence'].split('_')[0] for r in rows}),sequence_count=len({r['sequence'] for r in rows}),categories=list(map(str,set(cat))),summaries=summaries,proposed_probe='Fixed same2304 raw W50 packets, true previous GT clean + deterministic small gaussian(5mm,.05rad,.05rad) + six camera z±50mm/rx±20deg/ry±20deg. Nine conditions including original combined is optional diagnostic, not tune. I proposals scored against identical trueGT_current−trueGT_previous. No optimizer/D forward; D buffers/params unchanged. Report all rows split local/global/count; require actuallarge-vssmall degradation, clean absolute error, and parent R2 trained objective before I-only repair.')
 (OUT/'train_motion_domain.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
