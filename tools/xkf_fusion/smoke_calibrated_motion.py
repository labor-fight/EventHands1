"""CPU actual batch1/9 forward parity for calibrated motion ramp N=1000."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'model')]
import torch,numpy as np
from semkine.xkf_calibrated_motion_gate import XKFCalibratedMotionModel,CalibratedMotionCountPolicy
from semkine.xkf_motion_gate import MotionCountFusionPolicy
from tools.xkf_fusion.evaluate_calibrated_motion import BASE_POLICY
from tools.xkf_fusion import evaluate_tuned as Q
@torch.no_grad()
def main():
 torch.set_num_threads(4);Q.RE.setup();Q.E.XKFModel=XKFCalibratedMotionModel
 run=ROOT/'outputs/xkf/fusion_20261015_r3/direct_finetune/raw_s3407';model,cfg,identity=Q.E.load_model(run/'config_resolved.yaml',run/'xkf_donly_raw_s3407-step=2400.ckpt',torch.device('cpu'))
 for m in model.modules():
  if isinstance(m,Q.ManoLayer):m.fast_chain=True
 bank,_=Q.R.D.load_bank();clips,_=Q.R.schedule(bank,10);clip=clips[0];meta=bank['sequences'][clip['sequence']];data=Q.R.D.load_data(cfg,clip['sequence'],meta);cache=Q.PacketCache(model,data,torch.device('cpu'))
 initial=Q.R.initial_states(torch.from_numpy(data['gt'][clip['initial_gt_ms']]).float());row=meta['rows'][clip['row_indices'][0]]
 policy=dict(BASE_POLICY,id='smoke');outputs={}
 for label,cls,n,batch in [('old_batch1',MotionCountFusionPolicy,None,1),('new_batch1',CalibratedMotionCountPolicy,1000.,1),('new_batch9',CalibratedMotionCountPolicy,1000.,9)]:
  model.fusion_policy=cls(policy if n is None else dict(policy,motion_count_threshold=n));pred=[]
  for i in range(0,9,batch):
   packet,values=cache.packet(row,initial[i:i+batch],[]);assert not values;Q.R.D.reset(model);pred.append(model.forward_packet(packet))
  outputs[label]=torch.cat(pred);assert not cache.measurements
 assert torch.equal(outputs['old_batch1'],outputs['new_batch1']);assert torch.equal(outputs['old_batch1'],outputs['new_batch9'])
 assert bool(torch.isfinite(outputs['new_batch1']).all())
 result=dict(status='PASS',scope='one known real nonempty raw packet x9 declared initial conditions, full I+D+centroid no D cache; CPU exact batch1 baseline and stable9; not dev effectiveness',identity=identity,old_new_batch1_bitwise=True,new_batch1_batch9_bitwise=True,actual_policy=policy,threshold=1000.)
 out=ROOT/'outputs/xkf/fusion_20261015_r3/direct_finetune/calibrated_motion_runtime_smoke.json';out.write_text(json.dumps(result,indent=2)+'\n');print(result['status'])
if __name__=='__main__':main()
