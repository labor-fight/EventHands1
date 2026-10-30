"""Fixed-recorded-I-delta ablations, deliberately not fresh own-loop inference."""
import sys,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'model')]
import torch,numpy as np,yaml
from mano_layer import ManoLayer
from pose_repr import decode_to_mano_inputs
from semkine.xkf_centroid import fuse_joint_centroids
from semkine.xkf_motion_gate import interpolate_motion
from semkine.lie import so3_exp,so3_log
B=ROOT/'outputs/xkf/fusion_20261015_r3/direct_finetune';torch.set_num_threads(4)
cfg=yaml.safe_load((B/'raw_s3407/config_resolved.yaml').read_text());mano=ManoLayer(cfg['MANO']['NPZ'],add_mean=False).float().eval();mano.fast_chain=True
beta=torch.from_numpy(np.load(B/'calibration_raw_s3407/zgz_local_W200.npz')['beta'])
def fk(x,b):
 d=decode_to_mano_inputs(x,cfg['MODEL']['POSE_REPR'],mano.hands_components,mano.hands_mean);return mano(b.expand(len(x),-1),d['global_orient'],d['local_full_aa'],d['transl'])
@torch.no_grad()
def main():
 result={}
 for seed in (3407,3408,3409):
  p=B/f'motion_gate/s{seed}';pr=json.load(open(p/'protocol.json'));z=np.load(p/'raw/064_local_bank_run0.npz');m=pr['modes'].index('motion_ramp_centroid');mi=pr['motion_modes'].index('motion_ramp_centroid');ci=m-6
  pred=torch.from_numpy(z['predictions'][:,:,m]);initial=torch.from_numpy(z['initial']);prev=torch.cat((initial[None],pred[:-1]));xi=torch.from_numpy(z['motion_predictions'][:,:,mi]);delta=xi-prev;direct=torch.from_numpy(z['corrected_direct'][:,ci]);ks=torch.from_numpy(z['effective_gains'][:,ci]);truth=torch.from_numpy(z['truth']);g=torch.tensor(np.minimum(z['n_raw_events']/1000,1),dtype=torch.float32)
  _,jg=fk(truth,beta);result[str(seed)]={}
  for change in ('replay','zero_translation_delta','zero_root_delta','zero_finger_delta','zero_rigid_delta','zero_all_delta'):
   pp=initial.clone();trace=[]
   for t in range(len(truth)):
    dd=delta[t].clone()
    if change in ('zero_translation_delta','zero_rigid_delta','zero_all_delta'):dd[:,:3]=0
    if change in ('zero_root_delta','zero_rigid_delta','zero_all_delta'):dd[:,3:6]=0
    if change in ('zero_finger_delta','zero_all_delta'):dd[:,6:]=0
    proposal=interpolate_motion(pp,pp+dd,g[t].expand(9));pp=fuse_joint_centroids(proposal,direct[t].expand(9,-1),ks[t],beta,fk);trace.append(pp)
   trace=torch.stack(trace);_,jp=fk(trace.reshape(-1,51),beta);e=(jp.reshape(10,9,21,3)-jg[:,None])*1000
   j=e.norm(dim=-1).mean(-1);c=e.mean(2).norm(dim=-1)
   result[str(seed)][change]={pr['conditions'][k][0]:dict(mean_joint=float(j[:,k].mean()),end_joint=float(j[-1,k]),mean_centroid=float(c[:,k].mean()),end_centroid=float(c[-1,k])) for k in range(9)}
   print(seed,change,result[str(seed)][change]['z_plus50_rx_plus20'])
 (B/'trajectory1049_analysis/group_counterfactual.json').write_text(json.dumps(dict(scope='Mechanism-only fixed recorded I deltas from actual motion-gated loop; group suppression is causal surgery, not proposed frozen-state deployment or own-loop',results=result),indent=2)+'\n')
if __name__=='__main__':main()
