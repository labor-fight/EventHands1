#!/usr/bin/env python3
"""CPU mechanism audit of saved own-loops; counterfactuals never rerun I association."""
from pathlib import Path
import sys,json,csv,hashlib,math
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'model')]
import numpy as np
import torch,yaml
from mano_layer import ManoLayer
from pose_repr import decode_to_mano_inputs
from semkine.lie import so3_exp,so3_log
from semkine.xkf import fuse_states,nearest_aa
from semkine.xkf_centroid import fuse_joint_centroids
B=ROOT/'outputs/xkf/fusion_20261015_r3/direct_finetune'
OUT=B/'trajectory1049_analysis'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def interp(a,b,g):
    out=a+g[:,None]*(b-a)
    root=nearest_aa(so3_log(so3_exp(a[:,3:6])@so3_exp(g[:,None]*so3_log(so3_exp(a[:,3:6]).transpose(-1,-2)@so3_exp(b[:,3:6])))),a[:,3:6])
    out[:,3:6]=root
    return torch.where((g==0)[:,None],a,torch.where((g==1)[:,None],b,out))

@torch.no_grad()
def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    cfg=yaml.safe_load((B/'raw_s3407/config_resolved.yaml').read_text())
    mano=ManoLayer(cfg['MANO']['NPZ'],add_mean=False).float().eval();mano.fast_chain=True
    def fk(x,beta):
        d=decode_to_mano_inputs(x,cfg['MODEL']['POSE_REPR'],mano.hands_components,mano.hands_mean)
        return mano(beta.expand(len(x),-1),d['global_orient'],d['local_full_aa'],d['transl'])
    with np.load(B/'calibration_raw_s3407/zgz_local_W200.npz') as z:beta=torch.from_numpy(z['beta'])
    def stats(x,y):
        _,j=fk(x,beta);_,t=fk(y,beta);e=(j-t)*1000;w=e[:,0];r=e-w[:,None];c=e.mean(1);center=e-c[:,None]
        return dict(joint=e.norm(dim=-1).mean(-1),ra=r.norm(dim=-1).mean(-1),t=(x[:,:3]-y[:,:3]).norm(dim=-1)*1000,
            tx=(x[:,0]-y[:,0])*1000,ty=(x[:,1]-y[:,1])*1000,tz=(x[:,2]-y[:,2])*1000,
            root=so3_log(so3_exp(y[:,3:6]).transpose(-1,-2)@so3_exp(x[:,3:6])).norm(dim=-1)*180/math.pi,
            centroid=c.norm(dim=-1),cx=c[:,0],cy=c[:,1],cz=c[:,2],
            abs_mse=e.square().sum(-1).mean(-1),wrist_mse=w.square().sum(-1),relative_mse=r.square().sum(-1).mean(-1),cross=2*(w*r.mean(1)).sum(-1),
            centroid_mse=c.square().sum(-1),centered_mse=center.square().sum(-1).mean(-1))
    rows=[];summaries={};bins=[];sources={}
    for seed in (3407,3408,3409):
        p=B/f'evaluations/raw/s{seed}';pr=json.loads((p/'protocol.json').read_text());sources[str(p/'protocol.json')]=digest(p/'protocol.json')
        data=np.load(p/'raw/064_local_bank_run0.npz');sources[str(p/'raw/064_local_bank_run0.npz')]=digest(p/'raw/064_local_bank_run0.npz')
        truth=torch.from_numpy(data['truth']);pred=torch.from_numpy(data['predictions']);ini=torch.from_numpy(data['initial']);prev=torch.cat((ini[None,:,None].expand(1,9,12,51),pred[:-1]),0)
        motion=torch.from_numpy(data['motion_predictions']);ds=torch.from_numpy(data['corrected_direct']);gains=torch.from_numpy(data['effective_gains']);g=torch.from_numpy(np.minimum(data['n_raw_events']/1000,1)).float()
        summary={'windows':data['lnes_window_ms'].tolist(),'counts':data['n_raw_events'].tolist(),'modes':{},'counterfactual_scope':'fixed recorded I deltas replayed, NO fresh association/own-loop; cannot assess gates','counterfactual':{}}
        for m,name in enumerate(pr['modes']):
            state=pred[:,:,m];s=stats(state.reshape(-1,51),truth[:,None].expand(-1,9,-1).reshape(-1,51));ss={k:v.reshape(10,9) for k,v in s.items()}
            summary['modes'][name]={pr['conditions'][c][0]:{k:{'mean':float(v[:,c].mean()),'end':float(v[-1,c]),'first6':float(v[:6,c].mean()),'last4':float(v[6:,c].mean())} for k,v in ss.items()} for c in range(9)}
            for c in range(9):
                for f in range(10):
                    rows.append(dict(seed=seed,mode=name,condition=pr['conditions'][c][0],source='actual_own_loop',end_ms=int(data['end_ms'][f]),count=int(data['n_raw_events'][f]),**{k:float(v[f,c]) for k,v in ss.items()}))
        for ci,name in enumerate(pr['modes'][6:]):
            mi=pr['motion_modes'].index(name);xi=motion[:,:,mi];incoming=prev[:,:,ci+6]
            for label,st in [('incoming',incoming),('I_proposal',xi),('D_used',ds[:,ci,None].expand(-1,9,-1))]:
                s=stats(st.reshape(-1,51),truth[:,None].expand(-1,9,-1).reshape(-1,51));ss={k:v.reshape(10,9) for k,v in s.items()}
                for c in range(9):
                    for f in range(10):rows.append(dict(seed=seed,mode=name,condition=pr['conditions'][c][0],source=label,end_ms=int(data['end_ms'][f]),count=int(data['n_raw_events'][f]),**{k:float(v[f,c]) for k,v in ss.items()}))
            use_centroid='centroid' in name
            def fusion(a,d,k):return fuse_joint_centroids(a,d,k,beta,fk) if use_centroid else fuse_states(a,d,k)
            reconstructed=torch.stack([fusion(xi[f],ds[f,ci].expand(9,-1),gains[f,ci]) for f in range(10)])
            if (reconstructed-pred[:,:,ci+6]).abs().max()>2e-5:raise ValueError('fusion replay differs')
            summary['counterfactual'][name]={}
            for change in ('fixed_delta_replay','motion_ramp1000','whole_update_ramp1000','no_I_delta','root_sync_depth','root_unscaled'):
                pp=ini.clone();trace=[]
                for f in range(10):
                    delta=xi[f]-incoming[f];ip=pp+delta
                    if change=='motion_ramp1000':ip=interp(pp,ip,g[f].expand(9))
                    if change=='no_I_delta':ip=pp
                    k=gains[f,ci].clone()
                    if change=='root_sync_depth':k[3]=k[2]
                    if change=='root_unscaled' and ci:k[3]=(k[3]*4).clamp_max(1)
                    if change=='whole_update_ramp1000':
                        k=k/g[f].clamp_min(1e-9);out=interp(pp,fusion(ip,ds[f,ci].expand(9,-1),k),g[f].expand(9))
                    else:out=fusion(ip,ds[f,ci].expand(9,-1),k)
                    trace.append(out);pp=out
                trace=torch.stack(trace);s=stats(trace.reshape(-1,51),truth[:,None].expand(-1,9,-1).reshape(-1,51));ss={k:v.reshape(10,9) for k,v in s.items()}
                summary['counterfactual'][name][change]={pr['conditions'][c][0]:{k:{'mean':float(v[:,c].mean()),'end':float(v[-1,c])} for k,v in ss.items()} for c in range(9)}
        # All112 exact original I-only paths; equal saved occurrence weighting. Packet dedup not possible for wrong incoming.
        for fi,clip in enumerate(pr['clips']):
            f=p/'raw'/f'{fi:03d}_{clip["cohort"]}_run{clip["run_id"]}.npz';z=np.load(f)
            pp=torch.from_numpy(np.concatenate((z['initial'][None,:,None,:].repeat(12,axis=2),z['predictions'][:-1]),axis=0)[:,:,0]);ip=torch.from_numpy(z['motion_predictions'][:,:,0]);tr=torch.from_numpy(z['truth']);tp=torch.cat((torch.from_numpy(z['clean'])[None],tr[:-1]));n=len(tr)
            dt=(ip[:,:,:3]-pp[:,:,:3]).norm(dim=-1)*1000
            dr=so3_log(so3_exp(pp[:,:,3:6]).transpose(-1,-2)@so3_exp(ip[:,:,3:6])).norm(dim=-1)*180/math.pi
            deltaerr=((ip[:,:,:3]-pp[:,:,:3])-(tr[:,:3]-tp[:,:3])[:,None]).norm(dim=-1)*1000
            for j in range(n):
                count=int(z['n_raw_events'][j]);bucket='0' if count==0 else '1-49' if count<50 else '50-249' if count<250 else '250-999' if count<1000 else '1000+'
                for c in (0,7):bins.append(dict(seed=seed,cohort=clip['cohort'],condition=pr['conditions'][c][0],bucket=bucket,count=count,delta_t_mm=float(dt[j,c]),delta_root_deg=float(dr[j,c]),delta_t_error_mm=float(deltaerr[j,c]),actual_motion_t_mm=float((tr[j,:3]-tp[j,:3]).norm()*1000)))
        summaries[str(seed)]=summary
        print('completed',seed,flush=True)
    with (OUT/'per_frame.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (OUT/'all112_I_deltas.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(bins[0]));w.writeheader();w.writerows(bins)
    (OUT/'result.json').write_text(json.dumps(dict(kind='EXPLORATORY_SAVED_TRAJECTORY_MECHANISM',formal=False,own_loop_counterfactual=False,source_sha256=sources,script_sha256=digest(__file__),seeds=summaries),indent=2)+'\n')
if __name__=='__main__':main()
