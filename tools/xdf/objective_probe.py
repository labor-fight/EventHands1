#!/usr/bin/env python3
"""XDF-3: frozen Q1 full707, paired final-head z risk weights (1 versus 2).

All inference is batch one; all training states come from frozen Q1. This tool
only writes outputs/xdf/objective and never alters the production model.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'model')]
OUT = ROOT/'outputs/xdf/objective'
RUN = ROOT/'outputs/semkine/xdl_q1_s3407'
JOBS = [(w,s,r) for s,r in [(1701,0),(1702,0),(1701,1)] for w in (1,2)]

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def rel(p):
    try:return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:return str(Path(p).resolve())

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)

def read(p):return json.loads(Path(p).read_text())

def ident(p):return {'path':rel(p),'sha256':sha(p)}

def protocol():return read(OUT/'protocol.json')

def midpoint(items,n):
    n=min(n,len(items))
    return [items[int((i+.5)*len(items)/n)] for i in range(n)] if n else []

def device():
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('5','7'):
        raise RuntimeError('require explicit physical CUDA_VISIBLE_DEVICES=5 or7')
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    return torch.device('cuda:0')

def prepare():
    bank=read(ROOT/'outputs/xdl/packet_bank.json');e2=read(ROOT/'outputs/xde/e2/protocol.json')
    train=sorted({r['sequence'] for r in e2['selected_packets'] if r['split']=='fit'})
    clips=[];denom={}
    for seq in train:
        meta=bank['sequences'][seq];groups={}
        for row in meta['rows']:
            key=(row['run_id'],(row['event_start_ms']-meta['valid_runs_ms'][row['run_id']][0])//5000)
            groups.setdefault(key,[]).append(row)
        fitkeys=[k for i,k in enumerate(sorted(groups)) if i%4!=3]
        eligible=[]
        for key in fitkeys:
            rows=groups[key]
            if len(rows)<21:continue
            offset=1+(len(rows)-21)//2
            selected=rows[offset:offset+20]
            assert len(selected)==20 and all(r['run_id']==key[0] for r in selected)
            eligible.append({'sequence':seq,'segment':f'{seq}:{key[0]}:{key[1]}',
                            'clip_id':f'{seq}:{key[0]}:{key[1]}:own20',
                            'burn_indices':[r['index'] for r in rows[:offset]],
                            'end_indices':[r['index'] for r in selected],
                            'initial_time_ms':rows[0]['previous_end_ms']})
        selected=midpoint(eligible,4);clips+=selected
        denom[seq]={'eligible_fit_blocks':len(eligible),'selected_clips':len(selected)}
    sources=[ROOT/'outputs/xdl/packet_bank.json',ROOT/'outputs/xde/e2/protocol.json',
             ROOT/'outputs/xde/e1/protocol.json',RUN/'config_resolved.yaml',next(RUN.glob('*-step=6000.ckpt')),
             ROOT/'model/model.py',ROOT/'semkine/xyz_recovery_study.py',ROOT/'semkine/xyz_mesh.py',
             ROOT/'tools/xdl/diagnostics.py',ROOT/'tools/xdl/geometry.py']
    p={'study':'XDF-3-fixed-interface-z-risk-20261014','created_before_own_inference':True,
       'physical_gpus':[5,7],'cpu_threads_each':4,'max_gpu_hours':2.,'numeric':'pure fp32;matmul and cudnn TF32 disabled;no autocast',
       'source_forward_batch':1,'model':'original Q1 seed3407 step6000;all except finalfc1/fc2 frozen',
       'training_mix':{'interface':'all XDF1 interface split=fit, original E2 clean/z-50/z+50: expected5184',
                       'angles':'all parent angles split=fit:root+/-20,finger+/-20 crossed z0/-50/+50: expected3456',
                       'own':'all selected frozen-Q1 own fit states: up to5760',
                       'sampling':'concatenate each row once per epoch, same seed CPU torch randperm shared across weight arms;no online new-head training states',
                       'no_normalization':True,'no_augmentation':True},
       'own_initialization':'GT at first preceding time of each selected fit5s block ONCE as declared conventional initialization;frozen Q1 burns to chronological midpoint clip;all cached incoming states are Q1 outputs;GT afterward only supervision',
       'own_selection':'up to4 chronological midpoint-ranked eligible fit blocks per training sequence;20 midpoint queries after >=1 burn;no fold/valid crossing;no outcomes used',
       'own_clips':clips,'own_denominators':denom,'expected_own_rows':len(clips)*20,
       'expected_own_burn_calls':sum(len(c['burn_indices']) for c in clips),
       'loss':{'formula':'log10((450*mean(e_fingerAA^2)+60*mean(e_rootAA^2)+30000*mean(ex^2+ey^2+w*ez^2)/3)/51)',
               'units':'translation meters;angles residual axis-angle radians;pred=incoming+delta;freshAdam lr1e-4',
               'z_weights':[1,2],'epochs':100,'batch_size':512,'selection':'fixed final epoch100, no best selector'},
       'jobs':[{'job':i,'z_weight':w,'seed':s,'repeat':r} for i,(w,s,r) in enumerate(JOBS)],
       'gradient_audit':'same first512 rows of complete fixed fit concatenation,before/after;weighted individual x/y/z/root/finger contributions to shared fc1 weight+bias;common log derivative recorded separately',
       'evaluation':{'fullclean':['zgz_local','zgz_global'],'fullclean_local_expected':1204,
                     'strata':['dataset','split','state block','dz_mm','subject'],
                     'rollout':'E1 frozen24clips x20 eachsequence;own and clean initial;originalQ1,hold,sixheads;AA addition unchanged',
                     'gate':'both seeds L1 wrist >=10mm better than matchedL0, absmeanbiasz at leasthalved;mean gain > max(10mm,2*abs(Crepeat-Coriginal));only bias gate,additional protection needed'},
       'inputs':[ident(p) for p in sources],'executed_prepare_source':ident(__file__)}
    path=OUT/'protocol.json'
    if path.exists():raise RuntimeError('protocol already frozen; do not overwrite')
    save(path,p);print(json.dumps({'protocol':rel(path),'own_clips':len(clips),'own_rows':p['expected_own_rows'],'burn':p['expected_own_burn_calls']}),flush=True)

def capture_full(model):
    cap={}
    def hook(_m,inputs):
        if inputs[0].shape[-1]!=707:raise RuntimeError('capture must run after recovery707 hook')
        cap['full707']=inputs[0].detach()
    return cap,model.xyz.fc1.register_forward_pre_hook(hook)

def infer(D,model,cap,ev,state,beta,camera,dev,row):
    import torch
    cap.clear();packet=D.make_packet(ev,np.asarray(state)[None],beta,camera,dev,row)
    assert not packet.target.any()
    with torch.no_grad():out=model.forward_packet(packet)
    pred=out[0].cpu().numpy()
    feature=cap['full707'][0].cpu().numpy() if len(ev) else np.zeros(707,np.float32)
    if not np.isfinite(pred).all() or not np.isfinite(feature).all():raise FloatingPointError('nonfinite frozenQ1 inference')
    return pred,feature

def own_cache(shard):
    import torch
    from tools.xdl import diagnostics as D
    dev=device();start=time.perf_counter();p=protocol();ph=sha(OUT/'protocol.json')
    model,cfg,identity=D.load_model(str(RUN),dev);cap,handle=capture_full(model)
    bank=read(ROOT/'outputs/xdl/packet_bank.json');names=sorted({c['sequence'] for c in p['own_clips']})[shard::2]
    calls=0;newrows=0
    for seq in names:
        path=OUT/'own_cache'/f'{seq}.npz'
        if path.exists():
            old=read(path.with_suffix('.json'));assert old['protocol_sha256']==ph and old['cache_sha256']==sha(path);continue
        data=D.load_data(cfg,seq,bank['sequences'][seq]);beta,camera=D.context(data,dev);model.set_hand_context(beta,camera)
        arrays={k:[] for k in ['full707','incoming51','target51','prediction51']};rows=[]
        for clip in [c for c in p['own_clips'] if c['sequence']==seq]:
            state=data['gt'][clip['initial_time_ms']].copy();D.reset(model)
            for idx in clip['burn_indices']+clip['end_indices']:
                row=bank['sequences'][seq]['rows'][idx];ev=D.raw_events(data,row)
                pred,feature=infer(D,model,cap,ev,state,beta,camera,dev,row);calls+=1
                if idx in clip['end_indices']:
                    arrays['full707'].append(feature.copy());arrays['incoming51'].append(state.copy())
                    arrays['target51'].append(data['gt'][row['end_ms']].copy());arrays['prediction51'].append(pred.copy())
                    rows.append({'sequence':seq,'split':'fit','segment':clip['segment'],'clip_id':clip['clip_id'],
                                 'block':'own','dz_mm':None,'index':idx,'end_ms':row['end_ms'],
                                 'previous_end_ms':row['previous_end_ms'],'raw_count':len(ev),
                                 'event_sha256':hashlib.sha256(ev.tobytes()).hexdigest()})
                state=pred
                if calls==100:print(json.dumps({'phase':'own_speed100','seconds':time.perf_counter()-start,'estimated_all_shard_seconds':(time.perf_counter()-start)*(p['expected_own_rows']+p['expected_own_burn_calls'])/200}),flush=True)
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**{k:np.stack(v) for k,v in arrays.items()})
        save(path.with_suffix('.json'),{'rows':rows,'protocol_sha256':ph,'source_sha256':sha(__file__),'cache_sha256':sha(path),'identity':identity,'failures':0})
        newrows+=len(rows);print(json.dumps({'phase':'own','shard':shard,'seq':seq,'calls':calls,'rows':newrows,'seconds':time.perf_counter()-start}),flush=True)
    handle.remove();save(OUT/f'own_shard{shard}.json',{'calls':calls,'rows':newrows,'wall_seconds':time.perf_counter()-start,'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'protocol_sha256':ph,'source_sha256':sha(__file__)})

def read_cache_dir(directory,fit_only=False):
    arrays={k:[] for k in ('full707','incoming51','target51','prediction51')};metadata=[];sources=[]
    for path in sorted(Path(directory).glob('*.npz')):
        audit=read(path.with_suffix('.json'));rows=audit['rows']
        if audit.get('cache_sha256') != sha(path):raise RuntimeError(f'cache SHA changed: {path}')
        with np.load(path) as z:
            assert z['full707'].shape==(len(rows),707)
            selected=np.array([r.get('split')=='fit' or not fit_only for r in rows],bool)
            for k in arrays:arrays[k].append(z[k][selected])
        for r in np.array(rows,dtype=object)[selected]:
            r=dict(r);r.setdefault('sequence',r.get('seq',path.stem));r.setdefault('block','clean' if r.get('dz_mm',0)==0 else 'translation');r['source_group']=Path(directory).parent.name;metadata.append(r)
        sources.extend([ident(path),ident(path.with_suffix('.json'))])
    if not metadata:raise RuntimeError(f'empty cache {directory}')
    return {k:np.concatenate(v) for k,v in arrays.items()},metadata,sources

def fit_data():
    arrays={k:[] for k in ('full707','incoming51','target51','prediction51')};meta=[];ids=[];counts={}
    for name,path,expected in [('interface',ROOT/'outputs/xdf/interface/cache',5184),('angles',ROOT/'outputs/xdf/angles/cache',3456),('own',OUT/'own_cache',protocol()['expected_own_rows'])]:
        a,m,s=read_cache_dir(path,True)
        if len(m)!=expected:raise RuntimeError(f'{name} not complete: {len(m)} != {expected}')
        for k in arrays:arrays[k].append(a[k])
        meta+=m;ids+=s;counts[name]=len(m)
    a={k:np.concatenate(v) for k,v in arrays.items()}
    if not all(np.isfinite(v).all() for v in a.values()):raise FloatingPointError('nonfinite train cache')
    return a,meta,ids,counts

def make_head(dev,state=None):
    import torch
    from torch import nn
    checkpoint=next(RUN.glob('*-step=6000.ckpt'))
    sd=torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'] if state is None else state
    w=sd['xyz.fc1.weight'].shape[0] if 'xyz.fc1.weight' in sd else sd['0.weight'].shape[0]
    head=nn.Sequential(nn.Linear(707,w),nn.ReLU(),nn.Linear(w,51)).to(dev)
    if 'xyz.fc1.weight' in sd:head.load_state_dict({f'{i}.{k}':sd[f'xyz.fc{j}.{k}'] for i,j in [(0,1),(2,2)] for k in ['weight','bias']})
    else:head.load_state_dict(sd)
    return head

def loss_parts(pred,target,zweight):
    err=(pred-target).float();xyz=err[:,:3].square().mean(0)
    parts={'x':xyz[0]*30000/3/51,'y':xyz[1]*30000/3/51,'z':xyz[2]*zweight*30000/3/51,
           'root':err[:,3:6].square().mean()*60/51,'finger':err[:,6:].square().mean()*450/51}
    return parts

def gradient_audit(head,x,initial,target,weight):
    import torch
    pred=initial+head(x);parts=loss_parts(pred,target,weight);total=sum(parts.values())
    grads={};params=[head[0].weight,head[0].bias]
    for k,v in parts.items():grads[k]=torch.cat([z.flatten() for z in torch.autograd.grad(v,params,retain_graph=True)])
    norms={k:float(v.norm()) for k,v in grads.items()}
    cos={f'{a}/{b}':float(torch.nn.functional.cosine_similarity(grads[a][None],grads[b][None])) for a,b in [('z','root'),('z','finger'),('root','finger'),('z','x'),('z','y')]}
    return {'n':len(x),'unlogged_loss':float(total.detach()),'log_common_derivative':float((1/(total.detach()*np.log(10))).cpu()),
            'weighted_prelog_block_loss':{k:float(v.detach()) for k,v in parts.items()},'fc1_gradient_norm':norms,'fc1_gradient_cosine':cos}

def tag(job):
    w,s,r=JOBS[job];return f'z{w}_s{s}_r{r}'

def fit(job):
    import torch
    dev=device();start=time.perf_counter();a,meta,sources,counts=fit_data();w,seed,repeat=JOBS[job]
    torch.manual_seed(seed);head=make_head(dev);optimizer=torch.optim.Adam(head.parameters(),lr=1e-4)
    x=torch.as_tensor(a['full707'],device=dev);initial=torch.as_tensor(a['incoming51'],device=dev);target=torch.as_tensor(a['target51'],device=dev)
    audit_indices=torch.arange(min(512,len(x)),device=dev);before=gradient_audit(head,x[audit_indices],initial[audit_indices],target[audit_indices],w)
    rng=torch.Generator(device='cpu');rng.manual_seed(seed);orders=hashlib.sha256();curve=[];steps=0
    for epoch in range(100):
        order=torch.randperm(len(x),generator=rng);orders.update(order.numpy().tobytes());losses=[]
        for idx in order.split(512):
            idx=idx.to(dev);optimizer.zero_grad(set_to_none=True);pred=initial[idx]+head(x[idx]);unlogged=sum(loss_parts(pred,target[idx],w).values())
            loss=unlogged.log10()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite trainloss')
            loss.backward();optimizer.step();steps+=1;losses.append(float(loss.detach()))
        curve.append({'epoch':epoch+1,'mean_batch_log10_loss':float(np.mean(losses))})
        if (epoch+1)%20==0:print(json.dumps({'phase':'fit','head':tag(job),'epoch':epoch+1,'loss':curve[-1],'seconds':time.perf_counter()-start}),flush=True)
    after=gradient_audit(head,x[audit_indices],initial[audit_indices],target[audit_indices],w)
    if not all(torch.isfinite(v).all() for v in head.state_dict().values()):raise FloatingPointError('nonfinite fitted head')
    path=OUT/'heads'/f'{tag(job)}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in head.state_dict().items()},'job':job,'z_weight':w,'seed':seed,'repeat':repeat,'protocol_sha256':sha(OUT/'protocol.json'),'training_sources':sources,'counts':counts},path)
    save(path.with_suffix('.json'),{'head':tag(job),'job':job,'z_weight':w,'seed':seed,'repeat':repeat,'cache_counts':counts,'training_rows':len(x),'steps':steps,'epochs':100,'order_sha256':orders.hexdigest(),'gradient_before':before,'gradient_after':after,'curve':curve,'head_sha256':sha(path),'sources':sources,'protocol_sha256':sha(OUT/'protocol.json'),'source_sha256':sha(__file__),'wall_seconds':time.perf_counter()-start,'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'failures':0})

def load_heads(dev):
    import torch
    result={'q1':make_head(dev).eval()}
    for job in range(6):
        path=OUT/'heads'/f'{tag(job)}.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['protocol_sha256']==sha(OUT/'protocol.json');result[tag(job)]=make_head(dev,saved['state_dict']).eval()
    return result

def clean_cache(sequence):
    from tools.xdl import diagnostics as D
    dev=device();start=time.perf_counter();model,cfg,identity=D.load_model(str(RUN),dev);cap,handle=capture_full(model)
    bank=read(ROOT/'outputs/xdl/packet_bank.json');data=D.load_data(cfg,sequence,bank['sequences'][sequence]);beta,camera=D.context(data,dev);model.set_hand_context(beta,camera)
    arrays={k:[] for k in ['full707','incoming51','target51','prediction51']};meta=[]
    for row in bank['sequences'][sequence]['rows']:
        incoming=data['gt'][row['previous_end_ms']].copy();ev=D.raw_events(data,row)
        pred,feature=infer(D,model,cap,ev,incoming,beta,camera,dev,row)
        for k,v in [('full707',feature),('incoming51',incoming),('target51',data['gt'][row['end_ms']]),('prediction51',pred)]:arrays[k].append(v.copy())
        meta.append({'sequence':sequence,'split':'zgz','block':'clean','dz_mm':0.,'end_ms':row['end_ms'],'previous_end_ms':row['previous_end_ms'],'raw_count':len(ev),'index':row['index']})
    path=OUT/'clean_cache'/f'{sequence}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**{k:np.stack(v) for k,v in arrays.items()},betas=beta.cpu().numpy()[0],hands_mean=model.mano.hands_mean.cpu().numpy())
    handle.remove();save(path.with_suffix('.json'),{'rows':meta,'cache_sha256':sha(path),'identity':identity,'source_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),'wall_seconds':time.perf_counter()-start,'calls':len(meta),'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'failures':0})

def scalar_stats(v):
    v=np.asarray(v,float);good=np.isfinite(v)
    return {'mean':float(v[good].mean()) if good.any() else None,'p95':float(np.percentile(v[good],95)) if good.any() else None,'defined':int(good.sum()),'undefined':int((~good).sum())}

def score_states(D,mano,pred,initial,target,beta,dev,mean):
    errors,_,_=D.endpoint(mano,pred,target,beta,dev,mean);delta=(pred[:,:3].astype(float)-target[:,:3])*1000.;hold=np.linalg.norm((initial[:,:3]-target[:,:3])*1000,axis=1)
    return {'n':len(pred),'failures':int((~np.isfinite(pred).all(1)).sum()),'metrics':{k:scalar_stats(v) for k,v in errors.items()},
            'signed_xyz_mm':delta.mean(0).tolist(),'z_mae_mm':float(np.abs(delta[:,2]).mean()),'hold_wrist_mm':float(hold.mean()),'better_hold':int((np.linalg.norm(delta,axis=1)<hold).sum())}

def evaluate():
    import torch
    from tools.xdl import diagnostics as D
    from config import load_config
    dev=device();start=time.perf_counter();heads=load_heads(dev);cfg=load_config(RUN/'config_resolved.yaml');mano=D.ManoLayer(cfg['MANO']['NPZ'],add_mean=False).to(dev).eval();mano.fast_chain=True
    results={};predictions={};maxdiff={}
    for seq in ['zgz_local','zgz_global']:
        path=OUT/'clean_cache'/f'{seq}.npz';z=np.load(path);x=torch.as_tensor(z['full707'],device=dev);initial=z['incoming51'];target=z['target51'];beta=torch.as_tensor(z['betas'][None],device=dev);mean=z['hands_mean']
        results[seq]={}
        for name,head in heads.items():
            with torch.no_grad():pred=initial+torch.cat([head(xx) for xx in x.split(512)]).cpu().numpy()
            if name=='q1':maxdiff[seq]=float(np.max(np.abs(pred-z['prediction51'])))
            results[seq][name]=score_states(D,mano,pred,initial,target,beta,dev,mean);predictions[f'{seq}/{name}']=pred
        print(json.dumps({'phase':'evalclean','seq':seq,'n':len(x),'q1_replay_maxdiff':maxdiff[seq]}),flush=True)
    path=OUT/'eval_clean.npz';np.savez_compressed(path,**predictions)
    save(OUT/'eval_clean.json',{'metrics':results,'q1_head_batched_replay_maxdiff':maxdiff,'raw':ident(path),'source_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),'wall_seconds':time.perf_counter()-start,'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES']})

def strata():
    """Read existing cached states; no source-model inference or fitting."""
    import torch
    from tools.xdl import geometry as G
    from tools.xdl import diagnostics as D
    from config import load_config
    dev=device();start=time.perf_counter();heads=load_heads(dev);cfg=load_config(RUN/'config_resolved.yaml')
    mano=D.ManoLayer(cfg['MANO']['NPZ'],add_mean=False);mean=mano.hands_mean.cpu().numpy()
    result={'groups':{},'sources':[],'scope':'same frozen incoming for all heads;own rows are fitting states,not independent recovery'}
    for source,directory in [('interface',ROOT/'outputs/xdf/interface/cache'),('angles',ROOT/'outputs/xdf/angles/cache'),('own',OUT/'own_cache')]:
        a,rows,ids=read_cache_dir(directory);result['sources']+=ids
        x=torch.as_tensor(a['full707'],device=dev);initial=a['incoming51'];target=a['target51']
        gr,gf=G.rotations(target,mean);ir,ifl=G.rotations(initial,mean)
        grouping={}
        for i,r in enumerate(rows):
            seq=r['sequence'];kind='local' if 'local' in seq else 'global';subject=seq.split('_')[0]
            key=f'{source}/{r["split"]}/{kind}/{r.get("block")}/z={r.get("dz_mm")}'
            grouping.setdefault(key,[]).append(i)
            grouping.setdefault(key+'/subject='+subject,[]).append(i)
        saved={}
        for mode,h in heads.items():
            with torch.no_grad():pred=initial+torch.cat([h(xx) for xx in x.split(512)]).cpu().numpy()
            saved[mode]=pred;pr,pf=G.rotations(pred,mean)
            err=(pred[:,:3].astype(float)-target[:,:3])*1000.;wrist=np.linalg.norm(err,axis=1)
            root=G.geodesic_deg(pr,gr);finger=G.geodesic_deg(pf,gf).mean(-1)
            hold=np.linalg.norm((initial[:,:3]-target[:,:3])*1000,axis=1)
            for key,inds in grouping.items():
                ii=np.array(inds);v={'n':len(ii),'signed_xyz_mm':err[ii].mean(0).tolist(),'z_mae_mm':float(np.abs(err[ii,2]).mean()),
                    'wrist_mm':scalar_stats(wrist[ii]),'root_deg':scalar_stats(root[ii]),'finger_deg':scalar_stats(finger[ii]),
                    'hold_wrist_mm':float(hold[ii].mean()),'better_hold':int((wrist[ii]<hold[ii]).sum())}
                result['groups'].setdefault(key,{})[mode]=v
        path=OUT/f'strata_predictions_{source}.npz';np.savez_compressed(path,**saved)
        save(path.with_suffix('.json'),{'rows':rows,'raw':ident(path),'source_group':source})
    result.update(source_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json'),wall_seconds=time.perf_counter()-start,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'])
    save(OUT/'strata.json',result)

def rollout(sequence):
    import torch
    from tools.xdl import diagnostics as D
    dev=device();start=time.perf_counter();model,cfg,identity=D.load_model(str(RUN),dev);heads=load_heads(dev)
    mano=D.ManoLayer(cfg['MANO']['NPZ'],add_mean=False).to(dev).eval();mano.fast_chain=True;mean=mano.hands_mean.cpu().numpy()
    bank=read(ROOT/'outputs/xdl/packet_bank.json');data=D.load_data(cfg,sequence,bank['sequences'][sequence]);beta,camera=D.context(data,dev);model.set_hand_context(beta,camera)
    clips=read(ROOT/'outputs/xde/e1/protocol.json')['own_selection']['selected'][sequence]
    modes=['q1','hold']+[tag(i) for i in range(6)];records=[];calls=0;failures=[]
    original_fc1={k:v.detach().clone() for k,v in model.xyz.fc1.state_dict().items()};original_fc2={k:v.detach().clone() for k,v in model.xyz.fc2.state_dict().items()}
    for clip in clips:
        rows=[bank['sequences'][sequence]['rows'][i] for i in clip['end_indices']];raw=[D.raw_events(data,r) for r in rows];times=np.array([clip['initial_time_ms']]+[r['end_ms'] for r in rows]);truth=data['gt'][times]
        for initialization in ['own','clean']:
            initial=np.asarray(clip['initial'] if initialization=='own' else truth[0],np.float32);predictions=[]
            for mode in modes:
                if mode!='hold':
                    h=heads[mode];model.xyz.fc1.load_state_dict(h[0].state_dict());model.xyz.fc2.load_state_dict(h[2].state_dict())
                D.reset(model);states=[initial.copy()];failed=False
                for ev,row in zip(raw,rows):
                    if mode=='hold':states.append(initial.copy());continue
                    if failed:states.append(np.full(51,np.nan,np.float32));continue
                    try:
                        packet=D.make_packet(ev,states[-1][None],beta,camera,dev,row);assert not packet.target.any()
                        with torch.no_grad():value=model.forward_packet(packet)[0].cpu().numpy()
                        calls+=1
                        if not np.isfinite(value).all():raise FloatingPointError('nonfinite rollout')
                        states.append(value)
                    except Exception as error:
                        failures.append({'clip':clip['clip_id'],'initialization':initialization,'mode':mode,'step':len(states),'error':repr(error)});states.append(np.full(51,np.nan,np.float32));failed=True
                predictions.append(np.stack(states))
            predictions=np.stack(predictions);specs=[{'id':m,'block':m} for m in modes]
            with torch.no_grad():scored=D.score_trajectories(mano,predictions,truth,times,specs,np.stack([initial]*len(modes)),beta,dev,mean)
            for r in scored:r.update(sequence=sequence,clip_id=clip['clip_id'],initialization=initialization,truth=truth);records.append(r)
        print(json.dumps({'phase':'rollout','seq':sequence,'clip':clip['clip_id'],'calls':calls,'seconds':time.perf_counter()-start}),flush=True)
    model.xyz.fc1.load_state_dict(original_fc1);model.xyz.fc2.load_state_dict(original_fc2)
    path=OUT/f'rollout_{sequence}.pt.gz'
    with gzip.open(path,'wb',compresslevel=2) as f:torch.save({'records':records,'failures':failures,'protocol_sha256':sha(OUT/'protocol.json')},f)
    result={'metrics':{},'failures':failures,'calls':calls,'wall_seconds':time.perf_counter()-start,'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'source_sha256':sha(__file__),'raw':ident(path)}
    for initialization in ['own','clean']:
        for mode in modes:
            rs=[r for r in records if r['initialization']==initialization and r['spec']['id']==mode];out={'n_clips':len(rs),'auc':{k:scalar_stats([r['auc'][k] for r in rs]) for k in rs[0]['auc']},'motion':{},'roughness':{}}
            for space in ['absolute','root_relative','root','finger']:
                out['motion'][space]={k:scalar_stats([r['motion'][space][k] if r['motion'][space][k] is not None else np.nan for r in rs]) for k in ['gain','gain_bias','cosine','lag_ms']}
            for order,label in [(1,'velocity'),(2,'acceleration'),(3,'jerk')]:
                vals=[]
                for r in rs:
                    pd=np.diff(r['predictions'][:,:3].astype(float)*1000,n=order,axis=0)/.05**order;gt=np.diff(r['truth'][:,:3].astype(float)*1000,n=order,axis=0)/.05**order;vals.append(np.linalg.norm(pd-gt,axis=-1).mean())
                out['roughness'][f'{label}_error']=scalar_stats(vals)
            initialgood=sum(r['recovery']['initially_satisfied'] for r in rs)
            out['recovery']={'initially_satisfied':initialgood,'eligible_new_recovery':len(rs)-initialgood,'new_sustained_recovery':sum(not r['recovery']['initially_satisfied'] and r['recovery']['hit_step'] is not None for r in rs)}
            result['metrics'][f'{initialization}/{mode}']=out
    save(OUT/f'rollout_{sequence}.json',result)

def summarize():
    e=read(OUT/'eval_clean.json');local=e['metrics']['zgz_local'];gates=[]
    for seed in [1701,1702]:
        a=local[f'z1_s{seed}_r0'];b=local[f'z2_s{seed}_r0'];gain=a['metrics']['wrist_mm']['mean']-b['metrics']['wrist_mm']['mean'];ratio=abs(b['signed_xyz_mm'][2])/max(abs(a['signed_xyz_mm'][2]),1e-12)
        gates.append({'seed':seed,'wrist_gain_mm':gain,'abs_z_bias_ratio':ratio,'bias_gate_pass':gain>=10 and ratio<=.5})
    c0=local['z1_s1701_r0']['metrics']['wrist_mm']['mean']-local['z2_s1701_r0']['metrics']['wrist_mm']['mean'];cr=local['z1_s1701_r1']['metrics']['wrist_mm']['mean']-local['z2_s1701_r1']['metrics']['wrist_mm']['mean'];threshold=max(10,2*abs(cr-c0));mean_gain=float(np.mean([g['wrist_gain_mm'] for g in gates]))
    receipts=[]
    for pattern in ['own_shard*.json','clean_cache/*.json','heads/*.json','eval_clean.json','strata.json','rollout_*.json']:
        for path in sorted(OUT.glob(pattern)):
            v=read(path)
            if 'wall_seconds' in v:receipts.append({'path':rel(path),'seconds':v['wall_seconds'],'physical_gpu':v.get('physical_gpu')})
    result={'gate_by_seed':gates,'repeat_original_effect_mm':c0,'repeat_effect_mm':cr,'repeat_threshold_mm':threshold,'mean_gain_mm':mean_gain,'bias_gate_pass':all(g['bias_gate_pass'] for g in gates) and mean_gain>threshold,'scope':'frozen feature final-head optimization only;not a full model train or deployment candidate;negative does not rule out other objectives','budget':{'receipts':receipts,'allocated_gpu_hours':sum(r['seconds'] for r in receipts)/3600,'maximum_gpu_hours':2.},'protocol':ident(OUT/'protocol.json'),'source':ident(__file__)}
    save(OUT/'result.json',result);print(json.dumps(result),flush=True)

def self_test():
    import torch
    torch.set_num_threads(4)
    pred=torch.randn(7,51,requires_grad=True);target=torch.randn(7,51)
    l0=sum(loss_parts(pred,target,1).values());expected=(450*(pred[:,6:]-target[:,6:]).square().mean()+60*(pred[:,3:6]-target[:,3:6]).square().mean()+30000*(pred[:,:3]-target[:,:3]).square().mean())/51
    assert torch.allclose(l0,expected)
    l1=sum(loss_parts(pred,target,2).values());assert torch.allclose(l1-l0,30000/3/51*(pred[:,2]-target[:,2]).square().mean())
    save(OUT/'self_test.json',{'passed':True,'checks':['legacyweighted51Dloss_equals_formula','zweight2denominator3','noGPUneeded'],'source':ident(__file__)})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','self-test','own-cache','fit','clean-cache','evaluate','strata','rollout','summarize']);ap.add_argument('--shard',type=int);ap.add_argument('--job',type=int);ap.add_argument('--sequence',choices=['zgz_local','zgz_global']);args=ap.parse_args()
    try:
        if args.command=='prepare':prepare()
        elif args.command=='self-test':self_test()
        elif args.command=='own-cache':own_cache(args.shard)
        elif args.command=='fit':fit(args.job)
        elif args.command=='clean-cache':clean_cache(args.sequence)
        elif args.command=='evaluate':evaluate()
        elif args.command=='strata':strata()
        elif args.command=='rollout':rollout(args.sequence)
        else:summarize()
    except Exception as error:
        save(OUT/'failures'/f'{args.command}_sh{args.shard}_j{args.job}_{args.sequence}.json',{'error':repr(error),'traceback':traceback.format_exc(),'source':ident(__file__),'arguments':vars(args)});raise
