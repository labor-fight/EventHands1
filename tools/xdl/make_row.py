"""Exact-step6000 raw-W50 rows, with own-state latency and retained failures."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'model'))
import numpy as np
import torch
from config import load_config
from mano_layer import ManoLayer
from model import MNISTModel
from semkine import eval_track as ET
from semkine.dataset import sequences_for_split,splits_manifest
from tools import make_s36_row as C

METRICS=C.METRICS

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sanitize_json(value,path='$',invalid=None):
    invalid=[] if invalid is None else invalid
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,dict):return {str(k):sanitize_json(v,f'{path}.{k}',invalid) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [sanitize_json(v,f'{path}[{i}]',invalid) for i,v in enumerate(value)]
    if isinstance(value,float) and not math.isfinite(value):invalid.append(path);return None
    return value

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    invalid=[];value=sanitize_json(value,invalid=invalid)
    if isinstance(value,dict):value.update(numeric_valid=not invalid,invalid_numeric_paths=invalid)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def exact_checkpoint(directory):
    matches=[p for p in sorted(Path(directory).glob('*step=*.ckpt'))
             if re.search(r'step=(\d+)',p.name) and int(re.search(r'step=(\d+)',p.name).group(1))==6000]
    if len(matches)!=1:raise RuntimeError(f'exact unique step6000 required: {matches}')
    checkpoint=torch.load(matches[0],map_location='cpu')
    if int(checkpoint.get('global_step',-1))!=6000:raise RuntimeError('checkpoint filename differs from optimizer global_step')
    return matches[0]

def accuracy_aggregate(rows):
    if {r['seq'] for r in rows}!={'zgz_global','zgz_local'} or len(rows)!=2:
        raise RuntimeError('both complete zgz sequences required')
    if sum(r['n_frames'] for r in rows)!=2590:raise RuntimeError('complete2590 query denominator required')
    for r in rows:
        if any(not math.isfinite(float(r[k])) for k in METRICS):raise FloatingPointError('nonfinite accuracy; retain raw result and do not average finite subset')
    result={}
    for group in ['overall','local','global']:
        chosen=rows if group=='overall' else [r for r in rows if ('_local' in r['seq'])==(group=='local')]
        n=sum(r['n_frames'] for r in chosen)
        result[group]={k:sum(r[k]*r['n_frames'] for r in chosen)/n for k in METRICS}
    result.update(n_frames=2590,sequences=[r['seq'] for r in rows],per_seq={r['seq']:{k:r[k] for k in METRICS} for r in rows})
    return result

def model_class(cfg):
    if cfg.get('MODEL',{}).get('XDL_STUDY',False):
        from semkine.xyz_recovery_study import RecoveryXYZModel
        return RecoveryXYZModel
    return MNISTModel

def load_model(cfg,checkpoint,device):
    m=model_class(cfg).load_from_checkpoint(str(checkpoint),cfg=cfg,map_location='cpu',strict=True)
    return m.to(device).eval()

@torch.no_grad()
def latency_packets(m,cfg,device,n=600):
    root=Path(cfg['DATA']['ROOT']);split=splits_manifest(cfg)
    seq,d=next((s,d) for s,d in sequences_for_split(root,'train',split) if s=='lyq_local')
    events,offsets,aux,gt=ET.load_sequence(root,d,seq)
    sub=root/d/f'{seq}_tsub.npy';sub=np.load(sub,mmap_mode='r') if sub.exists() else None
    beta=torch.tensor(np.array(aux['betas']),device=device,dtype=torch.float32).view(1,-1)
    camera=torch.tensor(np.array(aux['camera_K']),device=device,dtype=torch.float32).view(1,3,3)
    m.set_hand_context(beta,camera)
    packets=[];rng=np.random.default_rng(0)
    for a,b in np.asarray(aux['valid_runs_ms'],np.int64).reshape(-1,2):
        if hasattr(m,'reset_recovery_context'):m.reset_recovery_context()
        prev=torch.tensor(gt[a].copy()+ET.sample_init_noise(cfg,rng,1.),device=device).view(1,-1).float()
        for end in np.arange(a+49,b,50,dtype=np.int64):
            ev=ET._window_events(events,offsets,sub,int(end),50)
            packet=ET.make_eval_packet(ev,prev.detach().clone(),beta,camera,50,device)
            pred=m.forward_packet(packet)
            if not bool(torch.isfinite(pred).all()):raise FloatingPointError('latency staging own-loop nonfinite')
            packets.append(packet);prev=pred.detach().clone()
            if len(packets)==n:return packets
    raise RuntimeError(f'require {n} chronological own-loop timing packets; got {len(packets)}')

@torch.no_grad()
def measure_latency(m,cfg,device):
    packets=latency_packets(m,cfg,device)
    for p in packets[:100]:m.forward_packet(p)
    torch.cuda.synchronize(device);means=[]
    for _ in range(3):
        start=time.perf_counter()
        for p in packets:m.forward_packet(p)
        torch.cuda.synchronize(device);means.append((time.perf_counter()-start)*1000/len(packets))
    return min(means),{'packets':600,'warm':100,'passes':3,'pass_mean_ms':means,'context':'staged from own rawW50 predictions; cold model path; no GT-held prior','includes':'forward_packet FK/association/bank/query/readout; excludes untimed staging and sensor pipeline'}

def measure_macs(cfg):
    from thop import profile
    from tools.xc.make_row import fixed_mac_fixture
    m=model_class(cfg)(cfg=cfg).float().eval()
    p,_=fixed_mac_fixture()
    class Wrap(torch.nn.Module):
        def __init__(self,inner):super().__init__();self.inner=inner
        def forward(self,packet):return self.inner.forward_packet(packet)
    macs,_=profile(Wrap(m),inputs=(p,),verbose=False)
    params=sum(v.numel() for v in m.parameters())
    heads=sum(v.numel() for k,v in m.named_parameters() if k.startswith(('xyz.fc1.','xyz.fc2.','xyz.slot_proj.')))
    return float(macs),params,heads

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--config');ap.add_argument('--output');ap.add_argument('--device',default='cuda')
    args=ap.parse_args();directory=REPO/'outputs/semkine'/args.run
    cfg=load_config(args.config or directory/'config_resolved.yaml')
    checkpoint=exact_checkpoint(directory);device=torch.device(args.device)
    output=Path(args.output or REPO/f'outputs/semkine/{args.run}_main_row.json')
    mano=ManoLayer(cfg['MANO']['NPZ'],add_mean=False).to(device).eval();m=load_model(cfg,checkpoint,device)
    root=Path(cfg['DATA']['ROOT']);seqs=sequences_for_split(root,'val_core',splits_manifest(cfg));rng=np.random.default_rng(0)
    raw=[]
    for seq,d in seqs:
        if hasattr(m,'reset_recovery_context'):m.reset_recovery_context()
        result=ET.track_sequence(m,mano,cfg,root,d,seq,50,device,rng,1.,1.,None,window_ms=50)
        if result is None:raise RuntimeError(f'empty whole sequence result: {seq}')
        raw.append(result)
        write_json(str(output)+'.raw_accuracy.json',{'run':args.run,'checkpoint_sha256':sha(checkpoint),'checkpoint':str(checkpoint),'raw_sequences':raw,'whole_sequence_protocol':'original rawW50/rng0 noisy initialization','complete':len(raw)==2,'finite_subset_reduction':False})
    aggregate=accuracy_aggregate(raw);aggregate.update(step=6000,ckpt=str(checkpoint),checkpoint_sha256=sha(checkpoint))
    latency,audit=measure_latency(m,cfg,device);del m,mano;torch.cuda.empty_cache()
    # Dense Full anchor stays the original model class, with no study adapters.
    anchor=C.latency_anchor(device)
    macs,params,headparams=measure_macs(cfg)
    seed=int(cfg['SEED'])
    row={'per_seed':{str(seed):aggregate},'two_seed_mean':{k:aggregate[k] for k in ['overall','local','global']},
         'latency_ms_scaled_full1p75':latency/anchor*C.FULL_PUBLISHED_MS,'latency_ms_raw':latency,
         'anchor_full_raw_ms':anchor,'macs_forward_packet':macs,'params_total':params,
         'params_frontend':params-headparams,'macs_frontend':macs,'macs_heads':None,
         'macs_forward_packet_note':'thop standard-layer lower bound only; excludes native FK/ray/scatter/topk/softmax/pooling arithmetic; all nominal parameters include null event modules',
         'latency_context_audit':audit,'protocol':{'split':'val_core','seeds':[seed],'n_seeds':1,'step_ms':50,'window_ms':50,'exact_optimizer_step':6000,'model_class':model_class(cfg).__name__,'scope':'single-seed exploratory; no checkpoint selection'},
         'finite_subset_reduction':False,'numeric_status':'PASS',
         'provenance':{'config_sha256':sha(args.config or directory/'config_resolved.yaml'),'checkpoint_sha256':sha(checkpoint),'producer_sha256':sha(__file__)}}
    write_json(output,row);print(json.dumps({'output':str(output),'step':6000,'numeric_status':'PASS','run':args.run}),flush=True)

if __name__=='__main__':main()
