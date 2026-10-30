#!/usr/bin/env python3
"""XDF-2: native support, matched exposure heads, actual augmentation paths.

Writes stay in outputs/xdf/depth, except the explicitly assigned CPU angle
review artifact outputs/xdf/angles/aa_seam_audit.json. GPU requires GPU6.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'model')]
OUT=ROOT/'outputs/xdf/depth'
E2=ROOT/'outputs/xde/e2'
RUN=ROOT/'outputs/semkine/xdl_q1_s3407'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)


def stats(values):
    a=np.asarray(values,np.float64).reshape(-1);good=a[np.isfinite(a)]
    return {'n':len(a),'finite':len(good),'mean':float(good.mean()) if len(good) else None,
            **{k:float(np.quantile(good,q)) if len(good) else None for k,q in [('min',0),('p50',.5),('p80',.8),('p95',.95),('p99',.99),('max',1)]}}


def read_labels(path):
    with Path(path).open('rb') as f:
        assert struct.unpack('<i',f.read(4))[0]==51
        a=np.fromfile(f,dtype=np.dtype([('data',np.float64,51),('m0',np.uint8),('m1',np.uint8)]))
    assert (a['m0']==4).all() and (a['m1']==13).all()
    return a['data'].astype(np.float32)


def root_joint(beta):
    with np.load(ROOT/'assets/mano_right.npz') as a:
        shaped=a['v_template'].astype(np.float64)+np.einsum('l,mkl->mk',np.asarray(beta,np.float64),a['shapedirs'].astype(np.float64))
        return (a['J_regressor'].astype(np.float64)@shaped)[0]


def prepare():
    sources=['outputs/xdf/preregistration_v1.md','outputs/xde/e2/protocol.json','outputs/xde/e3_preflight/samples.json',
             'outputs/xde/e3_preflight/geometry_arrays.npz','outputs/xde/e3_preflight/actual_wrist_depth_coverage.json',
             'tools/xdl/data.py','semkine/domrand.py','semkine/dataset.py','semkine/xyz_mesh.py',
             'semkine/xyz_recovery_study.py','semkine/xyz_observation.py','assets/mano_right.npz',
             'outputs/semkine/xdl_q1_s3407/config_resolved.yaml','outputs/semkine/xdl_q1_s3407/xdl_q1_s3407-step=6000.ckpt']
    p={'study':'XDF-2-native-support-and-actual-streams-20261014','frozen_before_new_analysis':True,
       'sources':{v:sha(ROOT/v) for v in sources},'script_at_freeze_sha256':sha(__file__),
       'native_population':'all canonical72 sequences, legal RawRecoveryDataset K4W50 main endpoints a+199..b-1; previous=end-50, initial=end-199; actual wrist=t+J0(beta)',
       'support':'E2 unique clean packets;fit T80 actual target wristz;fit median rawcount and true50mswristmotion bins;subject xlocal/global x2x2',
       'matching':'withinfitcell sort(sequence,end_ms), N=5floor(min(nlow,5nhigh)/5);D0 firstNlow;D1 first.8N of same low plus first.2N high;all3dz replicas;no duplicates',
       'check_gate':'highdepth check within fit cells N>0: >=3 training subjects and >=20 unique base-sequence/run/5s-blocks (v2/v3/v4 collapsed conservatively);else no forced fit',
       'fit':{'input':707,'head':'707->256 ReLU->3','seeds':[1701,1702],'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'batch_size':256,'epochs':300,'selection':'fixed final epoch','normalization':'common D0 low-fit mean/std, floor.001;targets xyzmm/50','loss':'xyz mean squared endpoint delta'},
       'flow':{'counter':'same1000 E3 sample_index values;actual RawRecoveryDataset.__getitem__ bothmodes, commonnamedRNG',
               'paths':['teacher_noisy_main','three_Q1_burn_then_main'],'forwards':1000*2*5,'batch_size':1,'precision':'float32 eval; frozen originalQ1',
               'physical_residual':'reuse frozen E3 noiseless same-time target projection arrays only; no noisy/selfstate in physicalapproximation residual',
               'parts':'offline all-face zbuffer on focal transformed mainGT endpoint, dominant LBS facepart; same sampled integer event pixels; endpoint geometric pseudo-source only, not per-event true source',
               'quantities':['event-and-stride equality','K/GTprev/noise/target','association vertex mass totalvariation including unmatched','top16 member fraction','pseudo-part candidate coverage and retained mass','bank queryweights/matrix','state99/full707','predicted-state error to each transformedtarget'],
               'raw_save':'onecompressednpz per counter, all four shared event packets, bothmode prev/target/K/noises, main teacher/burn features, sparseassociations,queryweights,outputstates,indices,pseudo-part; reproducible source hashes'},
       'gates':{'relative':.10,'absolute_mm':2.,'lowdepth_max_worsening_mm':1.,'negative_bias_min_mm':2.,'absolute_bias_halved':True},
       'limits':{'physical_gpu':6,'gpu_hours':1.,'cpu_hours':2.,'cpu_workers':16,'blas_threads':1},
       'limitations':['frozen backbone saw training identities and depth range','depth sampling policy changes actual pose and occlusion diversity','focal/depth is semantic total difference,not coverage-only causality','flow no retraining,no source-quality claim,no replacement of old5percentgate']}
    path=OUT/'protocol.json'
    if path.exists():raise RuntimeError('already frozen')
    save('protocol.json',p);(OUT/'protocol.sha256').write_text(sha(path)+'\n')
    print(json.dumps({'frozen':sha(path)}),flush=True)


def native_worker(item):
    name,meta=item
    source=lambda ext:ROOT/next(s['path'] for s in meta['sources'] if s['path'].endswith(ext))
    ap=source('_aux.npz');gp=source('.meta');aux=np.load(ap);gt=read_labels(gp);j0=root_joint(aux['betas'])
    index=np.concatenate([np.arange(a+199,b,dtype=np.int64) for a,b in meta['valid_runs_ms'] if b>a+199])
    depths=np.stack([gt[index-off,2].astype(np.float64)*1000+j0[2]*1000 for off in (0,50,199)],-1)
    return name,{'n_legal_endpoints':len(index),'J0_xyz_mm':j0*1000,'source_sha256':{str(ap):sha(ap),str(gp):sha(gp)},
                 'target':stats(depths[:,0]),'previous':stats(depths[:,1]),'initial':stats(depths[:,2])},depths


def native():
    from concurrent.futures import ProcessPoolExecutor
    start=time.monotonic();bank=json.loads((ROOT/'outputs/xdl/packet_bank.json').read_text())
    items=[(n,m) for n,m in bank['sequences'].items() if not n.startswith('zgz_')];assert len(items)==72
    rows={};arrays=[]
    with ProcessPoolExecutor(max_workers=16) as pool:
        for name,value,a in pool.map(native_worker,items):
            rows[name]=value;arrays.append(a)
            print(json.dumps({'native_sequence':name,'n':len(a),'target_max':value['target']['max']}),flush=True)
    a=np.concatenate(arrays);gtcoverage=json.loads((ROOT/'outputs/xde/e3_preflight/actual_wrist_depth_coverage.json').read_text())
    lo,hi=gtcoverage['sequences']['zgz_local']['actual_wrist_z_valid_range_mm']
    result={'protocol_sha256':sha(OUT/'protocol.json'),'definition':'actual wristz;exhaustive legal K4W50 main endpoints;repeated version sequences count as actual legal exposures',
            'n_sequences':72,'n_legal_endpoints':len(a),'summary':{k:stats(a[:,i]) for i,k in enumerate(['target','previous','initial'])},
            'actual_local_range_mm':[lo,hi],'target_inside_local':int(((a[:,0]>=lo)&(a[:,0]<=hi)).sum()),'target_above_local':int((a[:,0]>hi).sum()),
            'by_sequence':rows,'wall_seconds':time.monotonic()-start,'cpu_workers':16,'cpu_budget_hours':2.,'executed_script_sha256':sha(__file__)}
    save('native_population.json',result)
    np.savez_compressed(OUT/'native_histogram.npz',edges_mm=np.arange(250,901,5),target_counts=np.histogram(a[:,0],np.arange(250,901,5))[0],previous_counts=np.histogram(a[:,1],np.arange(250,901,5))[0])
    print(json.dumps({'native_done':result['summary'],'inside_local':result['target_inside_local'],'seconds':result['wall_seconds']}),flush=True)


def support():
    bank=json.loads((ROOT/'outputs/xdl/packet_bank.json').read_text());records=[];cacheids={}
    for name,meta in sorted(bank['sequences'].items()):
        path=E2/'cache'/f'{name}.npz';audit=json.loads(path.with_suffix('.json').read_text());assert sha(path)==audit['cache_sha256'];cacheids[name]=sha(path)
        ap=ROOT/next(s['path'] for s in meta['sources'] if s['path'].endswith('_aux.npz'));j0=root_joint(np.load(ap)['betas'])
        with np.load(path) as z:
            for i,r in enumerate(audit['rows']):
                if r['dz_mm']!=0:continue
                prev=z['incoming51'][i];target=z['target51'][i]
                records.append({**r,'packet_id':f'{name}:{r["end_ms"]}', 'subject':name.split('_')[0],'mode':'local' if '_local' in name else 'global',
                                'cache_clean_row':i,'target_wrist_z_mm':float((target[2]+j0[2])*1000),'previous_wrist_z_mm':float((prev[2]+j0[2])*1000),
                                'motion_mm':float(np.linalg.norm((target[:3]-prev[:3]).astype(np.float64))*1000),
                                'root_aa':prev[3:6].tolist(),'finger_aa':prev[6:].tolist()})
    fit=[r for r in records if r['split']=='fit'];T=float(np.quantile([r['target_wrist_z_mm'] for r in fit],.8));ec=float(np.median([r['raw_count'] for r in fit]));motion=float(np.median([r['motion_mm'] for r in fit]))
    for r in records:
        r['high']=r['target_wrist_z_mm']>T;r['cell']=f'{r["subject"]}|{r["mode"]}|{int(r["raw_count"]>ec)}|{int(r["motion_mm"]>motion)}'
        r['block_conservative']=re.sub(r'_v[234](?=:)', '',r['segment'])
    cells={};D0=[];D1=[]
    for cell in sorted({r['cell'] for r in fit}):
        rows=sorted([r for r in fit if r['cell']==cell],key=lambda r:(r['sequence'],r['end_ms']))
        low=[r for r in rows if not r['high']];high=[r for r in rows if r['high']];N=5*(min(len(low),5*len(high))//5)
        d0=low[:N];d1=low[:4*N//5]+high[:N//5];D0.extend(r['packet_id'] for r in d0);D1.extend(r['packet_id'] for r in d1)
        cells[cell]={'low':len(low),'high':len(high),'N':N,'D0':[r['packet_id'] for r in d0],'D1':[r['packet_id'] for r in d1],
                     'D0_distribution':{key:stats([r[key] for r in d0]) for key in ['target_wrist_z_mm','raw_count','motion_mm']},
                     'D1_distribution':{key:stats([r[key] for r in d1]) for key in ['target_wrist_z_mm','raw_count','motion_mm']}}
    matched={key for key,v in cells.items() if v['N']>0};highcheck=[r for r in records if r['split']=='check' and r['high'] and r['cell'] in matched]
    subjects=sorted({r['subject'] for r in highcheck});blocks=sorted({r['block_conservative'] for r in highcheck})
    passed=len(subjects)>=3 and len(blocks)>=20 and bool(D0)
    result={'protocol_sha256':sha(OUT/'protocol.json'),'T80_actual_wrist_mm':T,'fit_raw_count_median':ec,'fit_motion_median_mm':motion,'cells':cells,
            'D0':D0,'D1':D1,'n_unique_per_arm':len(D0),'n_low_D1':len(D0)*4//5,'n_high_D1':len(D0)//5,
            'highcheck_subjects':subjects,'highcheck_conservative_blocks':blocks,'n_highcheck_packets':len(highcheck),
            'fit_admissible':passed,'status':'READY_FOR_MATCHED_HEADS' if passed else 'SUPPORT_INSUFFICIENT_NO_FIT',
            'records':records,'source_cache_sha256':cacheids,'executed_script_sha256':sha(__file__)}
    assert len(D0)==len(set(D0))==len(D1)==len(set(D1))
    save('support.json',result);(OUT/'support.sha256').write_text(sha(OUT/'support.json')+'\n')
    print(json.dumps({k:result[k] for k in ['status','T80_actual_wrist_mm','n_unique_per_arm','n_highcheck_packets','highcheck_subjects']}|{'highcheck_blocks':len(blocks)}),flush=True)


def gpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='6':raise RuntimeError('Only physicalGPU6 allowed')
    import torch
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    return torch.device('cuda:0')


def flow():
    import ast
    import torch
    from config import load_config
    from tools.xdl.data import build_train_dataset,counter_rng
    from tools.xdl import diagnostics as D
    from semkine import domrand as DR
    from semkine.events import collate_packets
    device=gpu();start=time.monotonic();protocol=json.loads((OUT/'protocol.json').read_text())
    for name,digest in protocol['sources'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('frozen source changed: '+name)
    cfg=load_config(RUN/'config_resolved.yaml');ds=build_train_dataset(cfg);original_domrand=ds.domrand
    model,_,identity=D.load_model(str(RUN),device);model.xyz_keep_aux=True
    for parameter in model.parameters():parameter.requires_grad_(False)
    frozen=json.loads((ROOT/'outputs/xde/e3_preflight/samples.json').read_text());indices=[r['sample_index'] for r in frozen['identities']]
    assert len(indices)==1000
    tree=ast.parse((ROOT/'tools/xcl/sources/step1_source_labels.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rasterize')
    namespace={'np':np,'H':180,'W':240,'BARY_EPS':1e-5};exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen_XA_rasterizer','exec'),namespace)
    rasterize=namespace['rasterize']
    faces=model.mano.f.cpu().numpy();lbs=model.xyz.lbs_w.cpu().numpy();hard=lbs.argmax(-1)
    faceparts=lbs[faces].sum(1).argmax(-1)
    table=model.xyz.vert_faces.cpu().numpy()
    captured={}
    handle=model.xyz.fc1.register_forward_pre_hook(lambda _m,inputs:captured.update(full=inputs[0].detach().float().cpu().numpy()[0].copy()))
    calls=0;results=[];failures=[]

    def forward(packet,previous=None):
        nonlocal calls
        batch=collate_packets([packet]).to(device)
        if previous is not None:batch=dataclasses.replace(batch,prev_state=torch.as_tensor(previous[None],device=device))
        target=batch.target.float().cpu().numpy()[0].copy();incoming=batch.prev_state.float().cpu().numpy()[0].copy()
        batch=dataclasses.replace(batch,target=torch.zeros_like(batch.target))
        model.set_hand_context(batch.betas,batch.camera_K);captured.clear()
        with torch.no_grad():prediction=model.forward_packet(batch)
        calls+=1;aux=model.xyz_last
        if not captured or captured['full'].shape!=(707,):raise RuntimeError('full707 capture failed')
        # Every main has hot events in this fixed recipe; retain an explicit failure if not.
        if not len(packet.events):raise RuntimeError('unexpected empty packet')
        with torch.no_grad():
            uv=aux['verts'][...,:2]/aux['verts'][...,2:3].clamp_min(1e-6)
            uv=uv.masked_fill(aux['verts'][...,2:3]<=1e-6,1e6)
            top=((uv[:,None]-aux['d'][:,:,:2,None].transpose(-1,-2))**2).sum(-1).topk(16,dim=-1,largest=False).indices
        ret={k:aux[k][0].detach().float().cpu().numpy().copy() for k in ['xdl_bank','xdl_state','xdl_weights','xdl_uv']}
        ret.update({k:aux['assoc'][k][0].detach().cpu().numpy().copy() for k in ['vid','A','q','hit','valid','z']})
        ret.update(src=aux['xdl_src'][0].cpu().numpy().copy(),top16=top[0].cpu().numpy().copy(),full707=captured['full'],
                   prediction=prediction[0].float().cpu().numpy().copy(),incoming=incoming,target=target)
        if not all(np.isfinite(v).all() for v in ret.values()):raise FloatingPointError('nonfinite forward')
        return ret

    def masses(r):
        n=len(r['q']);m=np.zeros((n,779),np.float32)
        np.add.at(m,(np.arange(n)[:,None],r['vid']),r['A']*r['q'][:,None]);m[:,-1]=1-r['q']
        assert np.allclose(m.sum(1),1,atol=2e-6)
        return m

    def compare(a,b,parts):
        assert np.array_equal(a['src'],b['src']);mf,md=masses(a),masses(b)
        tv=.5*np.abs(mf-md).sum(-1);idsA,idsB=a['top16'],b['top16']
        replacement=1-(idsA[:,:,None]==idsB[:,None,:]).any(-1).mean(-1)
        score={'n_events':len(tv),'vertex_mass_tv':stats(tv),'top16_members_replaced':stats(replacement),
               'top16_any_set_changed':float((replacement>0).mean()),'hit_near_none_changed':float(((a['hit']!=b['hit'])|(a['valid']!=b['valid'])).mean()),
               'bank_query_weight_tv':stats(.5*np.abs(a['xdl_weights']-b['xdl_weights']).sum(-1)),
               'query_uv_l2_px':stats(np.linalg.norm(a['xdl_uv']-b['xdl_uv'],axis=-1)),
               'bank64_l2':float(np.linalg.norm(a['xdl_bank']-b['xdl_bank'])),'state99_l2':float(np.linalg.norm(a['xdl_state']-b['xdl_state'])),
               'full707_l2':float(np.linalg.norm(a['full707']-b['full707'])),'endpoint_pseudo_part_valid':int((parts>=0).sum()),'endpoint_pseudo_part_background':int((parts<0).sum())}
        for name,r,m in [('focal',a,mf),('depth',b,md)]:
            valid=parts>=0;candidate=table[r['top16']];cp=faceparts[np.maximum(candidate,0)]
            covers=((cp==parts[:,None,None])&(candidate>=0)).any(axis=(1,2))&valid
            retained=((hard[r['vid']]==parts[:,None])*r['q'][:,None]*r['A']).sum(-1)
            delta=(r['prediction'][:3]-r['target'][:3])*1000
            score[name]={'candidate_pseudo_part_coverage_n':int(covers.sum()),'candidate_pseudo_part_den':int(valid.sum()),
                         'retained_pseudo_part_mass':stats(retained[valid]),'q':stats(r['q']),
                         'signed_xyz_mm':delta,'wrist_endpoint_mm':float(np.linalg.norm(delta)),
                         'incoming_signed_xyz_mm':(r['incoming'][:3]-r['target'][:3])*1000,
                         'hold_wrist_mm':float(np.linalg.norm((r['incoming'][:3]-r['target'][:3])*1000))}
        return score,tv

    for ordinal,index in enumerate(indices):
        if time.monotonic()-start>3300:raise RuntimeError('flow GPU wall budget reached')
        step,row=divmod(index,1024);legal=int(counter_rng(3407,step,row,'clip_selection').integers(len(ds.legal_index)))
        si,end,ra,rb=map(int,ds.legal_index[legal]);h=ds._handle(si)
        assert h.seq==frozen['identities'][ordinal]['seq'] and end==frozen['identities'][ordinal]['end_ms']
        sample=DR.sample_params(original_domrand,counter_rng(3407,step,row,'shared_domrand'),180*240*2*50)
        packs={}
        for mode in ('focal','depth'):
            ds.domrand=dataclasses.replace(original_domrand,scale_mode=mode);packs[mode]=ds[index]
        ds.domrand=original_domrand
        fpack=[*packs['focal']['prefix'],packs['focal']['main']];dpack=[*packs['depth']['prefix'],packs['depth']['main']]
        event_equal=all(np.array_equal(f.events,d.events) for f,d in zip(fpack,dpack));assert event_equal
        raw={f'events_{k}':p.events for k,p in enumerate(fpack)};metadata={'ordinal':ordinal,'sample_index':index,'seq':h.seq,'end_ms':end,'scale':sample.scale,'event_equal':event_equal,'event_counts':[len(p.events) for p in fpack]}
        states={}
        for mode in ('focal','depth'):
            packets=[*packs[mode]['prefix'],packs[mode]['main']]
            dsm=dataclasses.replace(sample,scale_mode=mode);clean_gt=[];noise=[]
            for k,p in enumerate(packets):
                base=h.pos51[p.meta['incoming_ms']]
                trueprev,_=DR.transform_labels(base,dsm,h.camera_K,h.j0,.375)
                clean_gt.append(trueprev);noise.append(p.prev_state-trueprev if k in (0,3) else np.zeros(51,np.float32))
            raw.update({f'{mode}_packet_prev':np.stack([p.prev_state for p in packets]),f'{mode}_packet_target':np.stack([p.target for p in packets]),
                        f'{mode}_packet_K':np.stack([p.camera_K for p in packets]),f'{mode}_clean_prev':np.stack(clean_gt),f'{mode}_noise':np.stack(noise)})
            D.reset(model);teacher=forward(packets[-1]);states[f'{mode}_teacher']=teacher
            D.reset(model);previous=None;burn=[]
            for packet in packets[:3]:
                current=forward(packet,previous);previous=current['prediction'];burn.append(previous)
            states[f'{mode}_burn']=forward(packets[-1],previous);raw[f'{mode}_burn_predictions']=np.stack(burn)
        assert np.allclose(raw['focal_noise'],raw['depth_noise'],atol=1e-7)
        # Offline pseudo source: same-time clean target, all faces, no model predictions.
        with torch.no_grad():
            target=torch.as_tensor(fpack[-1].target[None],device=device);betas=torch.as_tensor(h.betas[None],device=device)
            verts,_=model._fk(target,betas)
        k=fpack[-1].camera_K;fxyc=.375*np.array([k[0,0],k[1,1],k[0,2],k[1,2]])
        _,partmap,_=rasterize(verts[0].cpu().numpy(),faces,faceparts,fxyc)
        src=states['focal_teacher']['src'];xy=fpack[-1].events[src,:2].astype(np.int64);parts=partmap[xy[:,1],xy[:,0]];raw['endpoint_pseudo_part']=parts
        for path in ('teacher','burn'):
            a,b=states[f'focal_{path}'],states[f'depth_{path}'];score,tv=compare(a,b,parts);metadata[path]=score;raw[f'{path}_vertex_mass_tv']=tv
        for name,values in states.items():
            for key,value in values.items():raw[f'{name}_{key}']=value
        metadata['clock']=[{key:p.meta[key] for key in ['incoming_ms','end_idx','prefix_index']} for p in fpack]
        out=OUT/'flow_cache'/f'{ordinal:04d}.npz';out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**raw)
        metadata['raw_sha256']=sha(out);save(f'flow_cache/{ordinal:04d}.json',metadata);results.append(metadata)
        if (ordinal+1)%100==0:
            print(json.dumps({'flow_done':ordinal+1,'calls':calls,'wall_seconds':time.monotonic()-start,'physical_gpu':6}),flush=True)
    handle.remove()
    summary={'protocol_sha256':sha(OUT/'protocol.json'),'identity':identity,'n_counters':len(results),'calls':calls,'failures':failures,
             'all_finite':True,'events_equal_all':all(r['event_equal'] for r in results),'physical_gpu':6,'wall_seconds':time.monotonic()-start,
             'source_batch':1,'executed_script_sha256':sha(__file__),'summary':{},
             'physical_approximation':json.loads((ROOT/'outputs/xde/e3_preflight/projection.json').read_text()),
             'scope':'frozenQ1 input-path sensitivity;no retraining/no causalallocation/no replacementofoldE3gate;partsource is noiselessfocalGTendpoint geometricpseudo-source'}
    for path in ('teacher','burn'):
        rr=[r[path] for r in results];n=sum(r['n_events'] for r in rr)
        s={'n_events':n,'vertex_mass_tv_event_weighted_mean':sum(r['vertex_mass_tv']['mean']*r['n_events'] for r in rr)/n,
           'top16_members_replaced_event_weighted_mean':sum(r['top16_members_replaced']['mean']*r['n_events'] for r in rr)/n,
           'top16_any_set_changed_event_weighted':sum(r['top16_any_set_changed']*r['n_events'] for r in rr)/n,
           'hit_near_none_changed_event_weighted':sum(r['hit_near_none_changed']*r['n_events'] for r in rr)/n}
        for metric in ['bank64_l2','state99_l2','full707_l2']:s[metric]=stats([r[metric] for r in rr])
        for metric in ['bank_query_weight_tv','query_uv_l2_px']:s[metric+'_mean_per_counter']=stats([r[metric]['mean'] for r in rr])
        for mode in ('focal','depth'):
            den=sum(r[mode]['candidate_pseudo_part_den'] for r in rr)
            s[mode]={'wrist_endpoint_mm':stats([r[mode]['wrist_endpoint_mm'] for r in rr]),
                     'signed_z_mm':stats([r[mode]['signed_xyz_mm'][2] for r in rr]),'hold_wrist_mm':stats([r[mode]['hold_wrist_mm'] for r in rr]),
                     'candidate_pseudo_part_coverage_n':sum(r[mode]['candidate_pseudo_part_coverage_n'] for r in rr),'candidate_pseudo_part_den':den,
                     'retained_pseudo_part_mass_mean':sum(r[mode]['retained_pseudo_part_mass']['mean']*r[mode]['candidate_pseudo_part_den'] for r in rr if r[mode]['candidate_pseudo_part_den'])/den if den else None}
        summary['summary'][path]=s
    save('flow_result.json',summary)
    print(json.dumps({'flow_complete':len(results),'calls':calls,'wall_seconds':summary['wall_seconds'],'summary':summary['summary']}),flush=True)


def native_details():
    """Post-census descriptive drilldown; never changes fit or sampling gates."""
    population=json.loads((OUT/'native_population.json').read_text())
    bank=json.loads((ROOT/'outputs/xdl/packet_bank.json').read_text())
    lo,hi=population['actual_local_range_mm'];maximum=population['summary']['target']['max'];details={}
    for name,meta in sorted(bank['sequences'].items()):
        source=lambda ext:ROOT/next(s['path'] for s in meta['sources'] if s['path'].endswith(ext))
        gt=read_labels(source('.meta'));j0=root_joint(np.load(source('_aux.npz'))['betas']);depth=gt[:,2].astype(np.float64)*1000+j0[2]*1000
        if name.startswith('zgz_'):
            valid=np.concatenate([np.arange(a,b) for a,b in meta['valid_runs_ms']]);previous=np.asarray([r['previous_end_ms'] for r in meta['rows']])
            details[name]={'n_valid_labels':len(valid),'above_exhaustive_native_max':int((depth[valid]>maximum).sum()),
                           'fraction_above_exhaustive_native_max':float((depth[valid]>maximum).mean()),'n_queries':len(previous),
                           'query_previous_above_native_max':int((depth[previous]>maximum).sum())}
            continue
        index=np.concatenate([np.arange(a+199,b) for a,b in meta['valid_runs_ms'] if b>a+199]);inside=index[(depth[index]>=lo)&(depth[index]<=hi)]
        if not len(inside):continue
        groups=np.split(inside,np.flatnonzero(np.diff(inside)>1)+1)
        motion=np.linalg.norm((gt[inside,:3]-gt[inside-50,:3]).astype(np.float64),axis=1)*1000
        details[name]={'n_overlap':len(inside),'spans_ms':[[g[0],g[-1],len(g)] for g in groups],
                       'overlap_motion50mm':stats(motion),'overlap_depth_mm':stats(depth[inside])}
    save('native_overlap_details.json',{'note':'Postpopulation descriptive drilldown;does not change sample boundary or fitgate',
        'native_population_sha256':sha(OUT/'native_population.json'),'actual_local_range_mm':[lo,hi],
        'native_max_actual_wrist_mm':maximum,'inside_fraction':population['target_inside_local']/population['n_legal_endpoints'],
        'by_sequence':details,'executed_script_sha256':sha(__file__)})


def audit_worker(ordinal):
    path=OUT/'flow_cache'/f'{ordinal:04d}.npz';meta=json.loads(path.with_suffix('.json').read_text())
    assert sha(path)==meta['raw_sha256'] and meta['ordinal']==ordinal and meta['event_equal']
    with np.load(path) as z:arrays={k:z[k] for k in z.files}
    assert all(np.isfinite(v).all() for v in arrays.values())
    expected_end=meta['end_ms'];clocks=meta['clock']
    assert [c['end_idx'] for c in clocks]==[expected_end-150,expected_end-100,expected_end-50,expected_end]
    assert [c['incoming_ms'] for c in clocks]==[expected_end-199,expected_end-150,expected_end-100,expected_end-50]
    assert [len(arrays[f'events_{k}']) for k in range(4)]==meta['event_counts']
    maxima={'mass_conservation':0.,'qA_tv':0.,'noise_equality':float(np.max(np.abs(arrays['focal_noise']-arrays['depth_noise']))),'endpoint_metric_mm':0.,'feature_concatenation':0.}
    assert maxima['noise_equality']<=1e-7
    for mode in ('focal','depth'):
        assert np.array_equal(arrays[f'{mode}_teacher_incoming'],arrays[f'{mode}_packet_prev'][3])
        assert np.array_equal(arrays[f'{mode}_burn_incoming'],arrays[f'{mode}_burn_predictions'][2])
        for path_name in ('teacher','burn'):
            prefix=f'{mode}_{path_name}_';q=arrays[prefix+'q'];A=arrays[prefix+'A']
            assert np.array_equal(arrays[prefix+'target'],arrays[f'{mode}_packet_target'][3])
            assert np.array_equal(arrays[prefix+'src'],arrays['focal_teacher_src'])
            assert np.array_equal(arrays[prefix+'full707'][544:608],arrays[prefix+'xdl_bank'])
            assert np.array_equal(arrays[prefix+'full707'][608:],arrays[prefix+'xdl_state'])
            error=np.linalg.norm((arrays[prefix+'prediction'][:3]-arrays[prefix+'target'][:3])*1000)
            maxima['endpoint_metric_mm']=max(maxima['endpoint_metric_mm'],abs(float(error)-meta[path_name][mode]['wrist_endpoint_mm']))
            maxima['mass_conservation']=max(maxima['mass_conservation'],float(np.max(np.abs((q[:,None]*A).sum(-1)+(1-q)-1))))
    for path_name in ('teacher','burn'):
        masses=[]
        for mode in ('focal','depth'):
            prefix=f'{mode}_{path_name}_';q=arrays[prefix+'q'];m=np.zeros((len(q),779),np.float32)
            np.add.at(m,(np.arange(len(q))[:,None],arrays[prefix+'vid']),arrays[prefix+'A']*q[:,None]);m[:,-1]=1-q;masses.append(m)
        tv=.5*np.abs(masses[0]-masses[1]).sum(-1)
        maxima['qA_tv']=max(maxima['qA_tv'],float(np.max(np.abs(tv-arrays[path_name+'_vertex_mass_tv']))))
    return {'ordinal':ordinal,'maxima':maxima,'npz_sha256':meta['raw_sha256'],'metadata_sha256':sha(path.with_suffix('.json'))}


def audit():
    from concurrent.futures import ProcessPoolExecutor
    started=time.monotonic();result=json.loads((OUT/'flow_result.json').read_text());rows=[]
    assert sha(OUT/'executed_flow_source.py')==result['executed_script_sha256']
    assert len(list((OUT/'flow_cache').glob('*.npz')))==len(list((OUT/'flow_cache').glob('*.json')))==1000
    with ProcessPoolExecutor(max_workers=16) as pool:
        rows=list(pool.map(audit_worker,range(1000)))
    maxima={key:max(r['maxima'][key] for r in rows) for key in rows[0]['maxima']}
    assert maxima['mass_conservation']<2e-6 and maxima['qA_tv']==0 and maxima['endpoint_metric_mm']==0
    frozen=json.loads((ROOT/'outputs/xde/e3_preflight/samples.json').read_text())['identities']
    for i,f in enumerate(frozen):
        m=json.loads((OUT/'flow_cache'/f'{i:04d}.json').read_text())
        assert (m['sample_index'],m['seq'],m['end_ms'])==(f['sample_index'],f['seq'],f['end_ms'])
    save('cpu_final_audit.json',{'status':'PASS','n_raw':1000,'failures':[],'all_finite':True,'all_cache_hashes_valid':True,
        'checks':['frozen1000counter identities','49/50/50/50ms clock','main event stride rows equal on both paths/modes',
                  'same additive noise','teacher actualdataset prior','burn own thirdpred is mainprior','zero modeltargets in executedsource',
                  'bank64/state99 equal full707 segments','qAvertex mass plus unmatched conserved','raw qATV and endpoint metrics exactly replayed'],
        'maxima':maxima,'raw_manifest':rows,'wall_seconds':time.monotonic()-started,'cpu_workers':16,
        'protocol_sha256':sha(OUT/'protocol.json'),'flow_source_sha256':result['executed_script_sha256'],'audit_source_sha256':sha(__file__),
        'additional_source_sha256':{'tools/xcl/sources/step1_source_labels.py':sha(ROOT/'tools/xcl/sources/step1_source_labels.py')},
        'source_note':'Rasterizer was read at execution; its additional hash is archived at final audit, not part of original protocol hashes.'})
    print(json.dumps({'audit':'PASS','maxima':maxima,'wall_seconds':time.monotonic()-started}),flush=True)


def angle_seam_audit():
    """Descriptive CPU review only; retain every row and every original gate."""
    from tools.xdf.angle_probe import load_arrays,angle_errors,write
    a,meta,mean,sources,p=load_arrays();inc=a['incoming51'];target=a['target51']
    direct=np.linalg.norm(inc[:,3:6].astype(np.float64)-target[:,3:6],axis=1)
    physical=angle_errors(inc,target,mean,'root')[:,0]
    seam=(direct>np.pi)&(physical<90)
    split=np.asarray([r['split'] for r in meta]);seq=np.asarray([r['sequence'] for r in meta])
    dz=np.asarray([r['dz_mm'] for r in meta]);block=np.asarray([r.get('block','clean') for r in meta])
    sign=np.asarray([r.get('sign',0) for r in meta]);angle=np.asarray([r['angle_cache'] for r in meta])
    groups={name:split==name for name in ('fit','check')};groups.update({name:seq==name for name in ('zgz_local','zgz_global')})
    masks={}
    for name,g in groups.items():
        masks[name+'/all']=g
        for z in (0,-50,50):
            masks[f'{name}/clean_z{z}']=g&~angle&(dz==z)
            for b in ('root','finger'):
                masks[f'{name}/{b}_both_z{z}']=g&angle&(block==b)&(dz==z)
                for s in (-1,1):masks[f'{name}/{b}_{s:+d}_z{z}']=g&angle&(block==b)&(sign==s)&(dz==z)
    def score(v,m):
        return {'n':int(m.sum()),'mean_deg':float(v[m].mean()) if m.any() else None,
                'sum_sample_mean_deg':float(v[m].sum()),'p95_sample_mean_deg':float(np.quantile(v[m],.95)) if m.any() else None}
    metrics={};row_errors={int(i):{} for i in np.flatnonzero(seam)}
    for b in ('root','finger'):
        predictions={'hold':inc,'Q1':a['prediction51']}
        for seed in (1701,1702):
            for head in ('F','B','state','wrongFull'):
                path=ROOT/'outputs/xdf/angles/heads'/f'{b}_{head}_s{seed}.predictions.npz'
                predictions[f'{head}_s{seed}']=np.load(path)['prediction51'];sources[str(path.relative_to(ROOT))]=sha(path)
        for name,pred in predictions.items():
            error=angle_errors(pred,target,mean,b).mean(1);key=f'{b}/{name}'
            metrics[key]={g:{'all_rows':score(error,m),'seam_rows':score(error,m&seam),'without_seam_descriptive_only':score(error,m&~seam)} for g,m in masks.items()}
            for i in row_errors:row_errors[i][key]=float(error[i])
    records=[dict(meta[i],cache_row=i,root_AA_difference_norm_rad=float(direct[i]),
                  root_SO3_target_error_deg=float(physical[i]),incoming_root_AA_norm_rad=float(np.linalg.norm(inc[i,3:6])),
                  target_root_AA_norm_rad=float(np.linalg.norm(target[i,3:6])),head_error_deg=row_errors[i]) for i in row_errors]
    target_contributions={}
    for name,g in groups.items():
        squares=direct**2/.35**2
        target_contributions[name]={'n':int(g.sum()),'n_seam':int((g&seam).sum()),'fraction_raw_root_AA_target_squared_norm_from_seam':float(squares[g&seam].sum()/squares[g].sum())}
    finger_direct=np.linalg.norm((inc[:,6:]+mean-target[:,6:]-mean).reshape(-1,3),axis=1)
    path=ROOT/'outputs/xdf/angles/aa_seam_audit.json'
    write(path,clean({'status':'descriptive_only_no_rows_removed_no_refit_no_gate_changed','n_rows':len(meta),'n_root_seam':int(seam.sum()),
        'definition':'root direct AA correction norm > pi while SO3 incoming-to-true-next target error <90deg',
        'angle_protocol_sha256':sha(ROOT/'outputs/xdf/angles/protocol.json'),'source_sha256':sha(__file__),'sources':sources,
        'rows':records,'metrics':metrics,'target_contributions':target_contributions,'finger_pairs_direct_gt_pi':int((finger_direct>np.pi).sum()),
        'interpretation':'Physical perturbation is valid. Canonical AA seam makes some production-compatible additive targets large. This is a probe limitation; it does not establish the cause of a negative F-vs-state gate. All primary results retain every row.'}))
    print(json.dumps({'angle_seam_rows':int(seam.sum()),'target_contributions':target_contributions,'path':str(path)}),flush=True)


def gradient_strata():
    """Authorized CPU descriptive supplement; frozen index sets before gradients."""
    import torch
    from tools.xdf import objective_probe as O
    started=time.monotonic();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    a,meta,sources,counts=O.fit_data();n=len(meta)
    nmain,nangle=counts['interface'],counts['angles'];angle=np.zeros(n,bool);angle[nmain:nmain+nangle]=True
    from scipy.spatial.transform import Rotation
    direct=np.linalg.norm(a['incoming51'][:,3:6].astype(np.float64)-a['target51'][:,3:6],axis=1)
    physical=(Rotation.from_rotvec(a['incoming51'][:,3:6]).inv()*Rotation.from_rotvec(a['target51'][:,3:6])).magnitude()
    seam=angle&(direct>np.pi)&(physical<np.pi/2)
    groups={'main_fit':np.arange(nmain),'angle_fit':np.arange(nmain,nmain+nangle),'own_fit':np.arange(nmain+nangle,n),
            'angle_fit_seam':np.flatnonzero(seam),'angle_fit_nonseam':np.flatnonzero(angle&~seam)}
    assert len(groups['angle_fit_seam'])==6
    points={'original_Q1':next(RUN.glob('*-step=6000.ckpt')),
            'L0_s1701':ROOT/'outputs/xdf/objective/heads/z1_s1701_r0.pt',
            'L1_s1701':ROOT/'outputs/xdf/objective/heads/z2_s1701_r0.pt'}
    dst=ROOT/'outputs/xdf/angles';protocol_path=dst/'gradient_strata_protocol.json'
    p={'study':'XDF descriptive gradient strata, no training or changed primary gate','scope':'all rows in each declared group; no selected minibatch',
       'groups':{name:{'n':len(i),'indices':i,'index_int64_sha256':hashlib.sha256(i.astype('<i8').tobytes()).hexdigest()} for name,i in groups.items()},
       'heads':{name:{'path':str(path.relative_to(ROOT)),'sha256':sha(path)} for name,path in points.items()},
       'fit_sources':sources,'fit_counts':counts,'script_sha256':sha(__file__),'objective_source_sha256':sha(ROOT/'tools/xdf/objective_probe.py'),
       'weights':'every parameter point evaluated at zweight1; zweight2 derived exactly by doubling z loss/gradient before common log scale, no second fit',
       'parameters':'fc1 weight+bias; component losses exactly objective_probe.loss_parts; groupwide mean, then log10',
       'prior_audit_coverage':'original first512 ordered fit rows are interface only, inspect identities below; no angles or own',
       'old512_subjects':sorted({r['sequence'].split('_')[0] for r in meta[:512]}),'old512_source_groups':sorted({r['source_group'] for r in meta[:512]}),
       'limits':{'wall_seconds':300,'physical_gpu':None,'torch_threads':4,'blas_threads':1},
       'causal_limit':'descriptive frozen-parameter gradients; neither target energy nor negative cosine establishes causal training interference'}
    if protocol_path.exists():raise RuntimeError('gradient strata protocol already frozen')
    O.save(protocol_path,p);protocol_sha=sha(protocol_path);results=[];failures=[]
    for name,path in points.items():
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        head=O.make_head(torch.device('cpu'),checkpoint['state_dict'])
        for group,indices in groups.items():
            if time.monotonic()-started>=300:
                failures.append({'status':'budget_stop','next_point':name,'next_group':group});break
            x=torch.as_tensor(a['full707'][indices]);initial=torch.as_tensor(a['incoming51'][indices]);target=torch.as_tensor(a['target51'][indices])
            audit=O.gradient_audit(head,x,initial,target,1)
            w2=json.loads(json.dumps(audit));w2['weighted_prelog_block_loss']['z']*=2;w2['fc1_gradient_norm']['z']*=2
            w2['unlogged_loss']=sum(w2['weighted_prelog_block_loss'].values());w2['log_common_derivative']=1/(w2['unlogged_loss']*np.log(10))
            for w,record in [(1,audit),(2,w2)]:
                record['log10_of_group_mean_prelog_loss']=float(np.log10(record['unlogged_loss']))
                record['postlog_fc1_component_gradient_norm']={k:v*record['log_common_derivative'] for k,v in record['fc1_gradient_norm'].items()}
            results.append({'parameter_point':name,'group':group,'native_objective_z_weight':2 if name=='L1_s1701' else 1,
                            'zweight1':audit,'zweight2':w2})
            O.save(dst/'gradient_strata_result.json',{'status':'partial','protocol_sha256':protocol_sha,'results':results,'failures':failures,'wall_seconds':time.monotonic()-started})
            print(json.dumps({'gradient_point':name,'group':group,'n':len(indices),'seconds':time.monotonic()-started}),flush=True)
        if failures:break
    O.save(dst/'gradient_strata_result.json',{'status':'budget_stopped' if failures else 'complete','protocol_sha256':protocol_sha,
        'results':results,'failures':failures,'wall_seconds':time.monotonic()-started,'CPU_only':True,'torch_threads':4,
        'finite':all(np.isfinite(r['zweight1']['unlogged_loss']) for r in results),'executed_script_sha256':sha(__file__)})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','native','support','flow','native-details','audit','angle-seam','gradient-strata']);args=ap.parse_args()
    {'prepare':prepare,'native':native,'support':support,'flow':flow,'native-details':native_details,'audit':audit,'angle-seam':angle_seam_audit,'gradient-strata':gradient_strata}[args.command]()
