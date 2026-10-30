#!/usr/bin/env python3
"""XDF-1 physical angle perturbations; frozen Q1, explicit target-free calls."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'model'))
OUT = ROOT/'outputs/xdf/angles'
RUN = ROOT/'outputs/semkine/xdl_q1_s3407'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def device():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '4': raise RuntimeError('physical GPU4 only')
    import torch
    torch.set_num_threads(4);torch.manual_seed(3407)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    return torch.device('cuda:0')


def prepare():
    from tools.xde.e2_probes import fixed_midpoints
    source=ROOT/'outputs/xde/e2/protocol.json';p=json.loads(source.read_text())
    selected=[]
    for name in sorted({x['sequence'] for x in p['selected_packets']}):
        rows=[x for x in p['selected_packets'] if x['sequence']==name]
        for split,n in ([('zgz',24)] if name.startswith('zgz_') else [('fit',4),('check',2)]):
            selected.extend(fixed_midpoints([r for r in rows if r['split']==split],n))
    assert len(selected)==480
    states=[{'block':b,'sign':s,'dz_mm':z} for b in ('root','finger') for s in (-1,1) for z in (0.,-50.,50.)]
    value={'study':'XDF-1-angle-interfaces-v1','source_protocol_sha256':sha(source),
           'parent_preregistration_sha256':sha(ROOT/'outputs/xdf/preregistration_v1.md'),
           'selected':selected,'states':states,'rows':5760,'forwards':11520,
           'checkpoint':p['checkpoint'],'checkpoint_sha256':p['checkpoint_sha256'],
           'perturbation':'XDL recovery_specs physical20deg per-packet axis; all15 finger joints; wrist preserved before adding declared z; true next endpoint targets',
           'targets':'all model packets contain zero target; GT used to construct declared previous perturbation and score only',
           'capture':'fc1 prehook appended after _recovery_readout; 707 dimensions; same-state wrong_content as E2',
           'fit':'original E2 full interface cache + angle cache, fit only. 707-256-ReLU-blockdim, targets AA residual divided by0.35rad, AdamW lr.001 wd.0001 batch256 fixed300epochs; seeds1701/1702. common real-fit stats then masks.',
           'fit_interfaces':['F','B','state','wrongFull'],'fit_blocks':['root','finger'],
           'fit_validation':'check score only, no epoch/width/head selection; original Q1 backbone saw training identities',
           'gate':'Each block on check own-block perturbed angle, dz0: F vs state and wrongFull both seeds >=10% and >=1degree; hold separately',
           'training_for_objective':'all angle fit rows included once per epoch alongside original E2fit and frozen Q1own fit; not outcome-selected',
           'physical_gpu':4,'gpu_hours_limit_including_fit':1.0,'batch_source':1,
           'script_sha256_at_freeze':sha(__file__)}
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());value['script_sha256_at_freeze']=old['script_sha256_at_freeze'];assert old==value
    else:write(path,value)
    print(json.dumps({'protocol_sha256':sha(path),'packets':480,'rows':5760}),flush=True)


def extract():
    import torch
    from tools.xdl import diagnostics as D, geometry as G
    from tools.xde.e2_probes import wrong_events
    dev=device();p=json.loads((OUT/'protocol.json').read_text());ph=sha(OUT/'protocol.json')
    started=time.monotonic();model,cfg,identity=D.load_model(str(RUN),dev)
    assert identity['checkpoint_sha256']==p['checkpoint_sha256']
    model.xyz_keep_aux=True
    for par in model.parameters():par.requires_grad_(False)
    bank=json.loads((ROOT/'outputs/xdl/packet_bank.json').read_text())
    mean=model.mano.hands_mean.detach().cpu().numpy().reshape(45)
    captured={}
    hook=model.xyz.fc1.register_forward_pre_hook(lambda _m,inputs:captured.update(full=inputs[0].detach().clone()))
    count=0;max_replay=0.;max_wrist=0.;reports=[]
    names=sorted({r['sequence'] for r in p['selected']})
    for name in names:
        path=OUT/'cache'/f'{name}.npz'
        if path.exists():
            r=json.loads(path.with_suffix('.json').read_text());assert r['protocol_sha256']==ph and r['cache_sha256']==sha(path);continue
        data=D.load_data(cfg,name,bank['sequences'][name]);beta,K=D.context(data,dev);model.set_hand_context(beta,K);D.reset(model)
        arrays={k:[] for k in ('full707','wrong_full707','incoming51','target51','prediction51','wrong_prediction51')};meta=[]
        def wrist(state):
            with torch.no_grad():
                _,j=model._fk(torch.tensor(state[None],dtype=torch.float32,device=dev),beta)
                return j[0,0].cpu().numpy().astype(np.float64)
        for item in [r for r in p['selected'] if r['sequence']==name]:
            row=item['row'];ev=D.raw_events(data,row);wrong=wrong_events(ev,D.raw_events(data,item['donor_row']))
            if not len(ev):raise RuntimeError('unexpected empty source requires separate retained amendment')
            initial=data['gt'][row['previous_end_ms']].copy();truth=data['gt'][row['end_ms']].copy()
            for spec in p['states']:
                gspec={'block':spec['block'],'scale':'large','sign':spec['sign'],'vector_id':row['packet_id']}
                incoming=G.perturb_state(initial,gspec,mean,wrist_fn=wrist)
                max_wrist=max(max_wrist,float(np.linalg.norm(wrist(incoming)-wrist(initial))*1000))
                incoming[2]+=spec['dz_mm']/1000.
                for prefix,events in (('',ev),('wrong_',wrong)):
                    captured.clear();packet=D.make_packet(events,incoming[None],beta,K,dev,row)
                    assert not bool(packet.target.any())
                    with torch.no_grad():
                        pred=model.forward_packet(packet)
                        x=captured['full'];assert x.shape==(1,707)
                        h=torch.relu(torch.nn.functional.linear(x,model.xyz.fc1.weight,model.xyz.fc1.bias))
                        delta=torch.nn.functional.linear(h,model.xyz.fc2.weight,model.xyz.fc2.bias)
                        diff=float((delta-model.xyz_last['delta']).abs().max());max_replay=max(max_replay,diff)
                        assert diff<=1e-6
                        assert float((pred-(packet.prev_state+delta)).abs().max())<=1e-6
                    a=x.cpu().numpy()[0];pr=pred.cpu().numpy()[0]
                    assert np.isfinite(a).all() and np.isfinite(pr).all()
                    arrays[prefix+'full707'].append(a);arrays[prefix+'prediction51'].append(pr)
                    count+=1
                assert np.array_equal(arrays['full707'][-1][608:],arrays['wrong_full707'][-1][608:])
                arrays['incoming51'].append(incoming);arrays['target51'].append(truth)
                meta.append({'sequence':name,'split':item['split'],'segment':row['segment'],'packet_id':row['packet_id'],
                             'end_ms':row['end_ms'],'previous_end_ms':row['previous_end_ms'],**spec,
                             'event_sha256':hashlib.sha256(ev.tobytes()).hexdigest()})
                if time.monotonic()-started>3000:raise TimeoutError('angle extraction budget')
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**{k:np.stack(v) for k,v in arrays.items()},hands_mean=mean)
        write(path.with_suffix('.json'),{'protocol_sha256':ph,'cache_sha256':sha(path),'rows':meta,'identity':identity,'script_sha256':sha(__file__)})
        reports.append({'sequence':name,'rows':len(meta)});
        print(json.dumps({'sequence':name,'rows':len(meta),'calls':count,'elapsed_s':time.monotonic()-started}),flush=True)
    hook.remove();torch.cuda.synchronize()
    write(OUT/'extraction.json',{'status':'complete','calls_new':count,'wall_seconds':time.monotonic()-started,'physical_gpu':4,'max_delta_replay_diff':max_replay,
          'max_wrist_preservation_error_mm':max_wrist,'protocol_sha256':ph,'script_sha256':sha(__file__),'completed_this_call':reports})


def load_arrays():
    p=json.loads((OUT/'protocol.json').read_text());full={k:[] for k in ('full707','wrong_full707','incoming51','target51','prediction51')};meta=[];mean=None;sources={}
    for directory,angle in ((ROOT/'outputs/xdf/interface/cache',False),(OUT/'cache',True)):
        for path in sorted(directory.glob('*.npz')):
            a=json.loads(path.with_suffix('.json').read_text());sources[str(path.relative_to(ROOT))]=sha(path)
            assert a['cache_sha256']==sources[str(path.relative_to(ROOT))]
            with np.load(path) as d:
                for key in full:full[key].append(d[key])
                if angle:
                    if mean is None:mean=d['hands_mean']
                    else:assert np.array_equal(mean,d['hands_mean'])
            for row in a['rows']:meta.append(dict(row,angle_cache=angle))
    assert len(meta)==7680+5760 and mean is not None
    return {k:np.concatenate(v) for k,v in full.items()},meta,mean,sources,p


def angle_errors(pred,target,mean,block):
    from scipy.spatial.transform import Rotation
    columns=slice(3,6) if block=='root' else slice(6,51)
    add=0. if block=='root' else mean
    a=(pred[:,columns]+add).reshape(-1,3);b=(target[:,columns]+add).reshape(-1,3)
    error=(Rotation.from_rotvec(a).inv()*Rotation.from_rotvec(b)).magnitude()*180/np.pi
    return error.reshape(len(pred),-1)


def fit(seed):
    import torch
    from torch import nn
    dev=device();started=time.monotonic();data,meta,mean,sources,p=load_arrays()
    split=np.array([r['split'] for r in meta]);fit_mask=split=='fit';seq=np.array([r['sequence'] for r in meta])
    labels=np.array([r.get('block','clean') for r in meta]);dz=np.array([r['dz_mm'] for r in meta]);is_angle=np.array([r['angle_cache'] for r in meta])
    center=data['full707'][fit_mask].mean(0);scale=np.maximum(data['full707'][fit_mask].std(0),.001)
    truth=data['target51'];incoming=data['incoming51'];reports=[]
    for block,cols in [('root',slice(3,6)),('finger',slice(6,51))]:
        target=(truth[:,cols]-incoming[:,cols]).astype(np.float32)/.35
        for interface in p['fit_interfaces']:
            torch.manual_seed(seed)
            x=(data['wrong_full707'] if interface=='wrongFull' else data['full707']).copy();x=(x-center)/scale
            if interface=='B':x[:,:544]=0
            elif interface=='state':x[:,:608]=0
            xt=torch.tensor(x,device=dev);yt=torch.tensor(target,device=dev);idx=torch.tensor(np.flatnonzero(fit_mask),device=dev)
            head=nn.Sequential(nn.Linear(707,256),nn.ReLU(),nn.Linear(256,target.shape[1])).to(dev)
            optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
            for epoch in range(300):
                order=idx[torch.randperm(len(idx),device=dev)]
                for indices in order.split(256):
                    optimizer.zero_grad(set_to_none=True);loss=(head(xt[indices])-yt[indices]).square().mean();loss.backward();optimizer.step()
                if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite head fit')
            with torch.no_grad():delta=torch.cat([head(batch) for batch in xt.split(1024)]).cpu().numpy()*.35
            prediction=data['prediction51'].copy();prediction[:,cols]=incoming[:,cols]+delta
            errors=angle_errors(prediction,truth,mean,block);hold=angle_errors(incoming,truth,mean,block);orig=angle_errors(data['prediction51'],truth,mean,block)
            groups={k:split==k for k in ('fit','check')}
            groups.update({k:seq==k for k in ('zgz_local','zgz_global')})
            scores={}
            for name,group in groups.items():
                for label,mask in [('clean',~is_angle&(dz==0)),('angle_z0',is_angle&(labels==block)&(dz==0)),('angle_z-50',is_angle&(labels==block)&(dz==-50)),('angle_z+50',is_angle&(labels==block)&(dz==50))]:
                    m=group&mask
                    scores[name+'/'+label]={'n':int(m.sum()),'angle_mae_deg':float(errors[m].mean()),'hold_deg':float(hold[m].mean()),'q1_deg':float(orig[m].mean()),
                        'per_joint_deg':errors[m].mean(0).tolist(),'better_than_hold_count':int((errors[m].mean(1)<hold[m].mean(1)).sum())}
            path=OUT/'heads'/f'{block}_{interface}_s{seed}.pt';path.parent.mkdir(exist_ok=True)
            torch.save({'state_dict':head.cpu().state_dict(),'mean':center,'std':scale,'block':block,'interface':interface,'seed':seed,'protocol_sha256':sha(OUT/'protocol.json')},path)
            np.savez_compressed(path.with_suffix('.predictions.npz'),prediction51=prediction,angle_error_deg=errors)
            record={'block':block,'interface':interface,'seed':seed,'metrics':scores,'head_sha256':sha(path)}
            write(path.with_suffix('.json'),record);reports.append(record)
            print(json.dumps({'block':block,'interface':interface,'seed':seed,'check_angle_z0':scores['check/angle_z0']}),flush=True)
            if time.monotonic()-started>3000:raise TimeoutError('angle fit budget')
    write(OUT/f'fit_s{seed}.json',{'reports':reports,'wall_seconds':time.monotonic()-started,'physical_gpu':4,'source_sha256':sources,'script_sha256':sha(__file__)})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','extract','fit']);parser.add_argument('--seed',type=int,default=1701)
    args=parser.parse_args()
    if args.phase=='prepare':prepare()
    elif args.phase=='extract':extract()
    else:fit(args.seed)
