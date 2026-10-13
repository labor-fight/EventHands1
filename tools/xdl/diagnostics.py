#!/usr/bin/env python3
"""Fixed-event counterfactuals and own-state recovery for XD-V2.

All inference targets are zero. GT only creates declared diagnostic initial states
and scores endpoints. Shared C0 natural states are frozen separately, never fed
back after the first query. Raw arrays and failed trajectories remain auditable.
"""
from __future__ import annotations
import argparse
import copy
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'model'))
from config import load_config
from mano_layer import ManoLayer
from model import MNISTModel
from semkine import eval_track as ET
from semkine.events import EventPacketBatch
from tools.xdl import geometry as G
from tools.xdl.prepare import sha,write_json
OUT=REPO/'outputs/xdl'


def load_bank(out=OUT):
    path=Path(out)/'packet_bank.json'
    return json.loads(path.read_text()),sha(path)


def load_data(cfg,name,meta):
    root=Path(cfg['DATA']['ROOT']);events,offsets,aux,gt=ET.load_sequence(root,meta['directory'],name)
    tp=root/meta['directory']/f'{name}_tsub.npy'
    return {'events':events,'offsets':offsets,'aux':aux,'gt':gt,'tsub':np.load(tp,mmap_mode='r') if tp.exists() else None}


def context(data,device):return ET_context(data,device)


def ET_context(data,device):
    return torch.as_tensor(np.array(data['aux']['betas']),dtype=torch.float32,device=device).reshape(1,-1), \
           torch.as_tensor(np.array(data['aux']['camera_K']),dtype=torch.float32,device=device).reshape(1,3,3)


def load_model(run,device,config=None):
    path=Path(run)
    if not path.is_absolute() and not path.exists():path=REPO/'outputs/semkine'/str(run)
    if path.is_dir():
        checkpoints=list(path.glob('*-step=6000.ckpt'))
        if len(checkpoints)!=1:raise RuntimeError(f'exact unique step6000 required: {path}: {checkpoints}')
        path=checkpoints[0]
    if not path.is_file() or '-step=6000.ckpt' not in path.name:raise ValueError(f'exact named step6000 endpoint required: {path}')
    cfgpath=Path(config) if config else path.parent/'config_resolved.yaml'
    cfg=load_config(str(cfgpath));saved=torch.load(path,map_location='cpu')
    if int(saved.get('global_step',-1))!=6000:raise ValueError('checkpoint global_step is not6000')
    if cfg['MODEL'].get('XDL_STUDY',False):
        from semkine.xyz_recovery_study import RecoveryXYZModel
        model=RecoveryXYZModel.load_from_checkpoint(str(path),cfg=cfg,map_location='cpu',strict=True)
    else:model=MNISTModel.load_from_checkpoint(str(path),cfg=cfg,map_location='cpu',strict=True)
    if getattr(model,'xyz_assoc_prev','self')!='self':raise ValueError('GT association is forbidden')
    model=model.to(device).float().eval()
    # Fast MANO is the same graph with CPU parent indices; default model untouched.
    for module in model.modules():
        if isinstance(module,ManoLayer):module.fast_chain=True
    return model,cfg,{'checkpoint':str(path),'checkpoint_sha256':sha(path),'config':str(cfgpath),'config_sha256':sha(cfgpath),'global_step':6000,
                     'parameters':sum(p.numel() for p in model.parameters()),'model_class':type(model).__name__}


def reset(model):
    if hasattr(model,'reset_time_history'):model.reset_time_history()
    if hasattr(model,'reset_recovery_context'):model.reset_recovery_context()
    if hasattr(model,'clear_eval_cache'):model.clear_eval_cache()


def make_packet(ev5,states,beta,camera,device,row):
    states=torch.as_tensor(states,dtype=torch.float32,device=device);batch=len(states);ev=np.asarray(ev5,np.float32)
    n=len(ev);values=np.tile(ev,(batch,1))
    if n:values[:,0]=np.repeat(np.arange(batch),n)
    return EventPacketBatch(events=torch.from_numpy(values).to(device),ptr=torch.arange(batch+1,device=device,dtype=torch.long)*n,
        sequence_id=torch.zeros(batch,device=device,dtype=torch.long),
        t_start_us=torch.full((batch,),int(row['event_start_ms'])*1000,device=device,dtype=torch.long),
        t_end_us=torch.full((batch,),(int(row['end_ms'])+1)*1000,device=device,dtype=torch.long),
        delta_t_s=torch.full((batch,),.05,device=device),is_sequence_start=torch.zeros(batch,device=device,dtype=torch.bool),
        is_sequence_end=torch.zeros(batch,device=device,dtype=torch.bool),target=torch.zeros_like(states),prev_state=states,
        betas=beta.expand(batch,-1),camera_K=camera.expand(batch,-1,-1),lnes=None)


def raw_events(data,row):return ET._window_events(data['events'],data['offsets'],data['tsub'],int(row['end_ms']),50)


@torch.no_grad()
def predict(model,ev,states,beta,camera,device,row,batch_size=1):
    result=np.full(np.asarray(states).shape,np.nan,np.float32);failures=[];attempts=0;elapsed=0.;observation_audits=[]
    old_aux=getattr(model,'xyz_keep_aux',False);model.xyz_keep_aux=True
    for begin in range(0,len(states),batch_size):
        incoming=np.asarray(states)[begin:begin+batch_size]
        packet=make_packet(ev,incoming,beta,camera,device,row)
        if bool(packet.target.any()):raise AssertionError('prediction target must be zero')
        if device.type=='cuda':torch.cuda.synchronize(device)
        start=time.perf_counter();attempts+=len(incoming)
        handle=None;bank_nodes=[]
        if hasattr(model,'observation_bank'):
            handle=model.observation_bank.encoder.register_forward_hook(lambda _m,_i,out:bank_nodes.append(out[1].detach()))
        try:
            output=model.forward_packet(packet)
            if device.type=='cuda':torch.cuda.synchronize(device)
            result[begin:begin+len(incoming)]=output.detach().float().cpu().numpy()
            for k,p in enumerate(result[begin:begin+len(incoming)]):
                if not np.isfinite(p).all():failures.append({'row':begin+k,'reason':'nonfinite_prediction'})
            auxiliary=getattr(model,'xyz_last',{}) if len(ev) else {}
            tok=auxiliary.get('tok');mask=auxiliary.get('mask')
            for k in range(len(incoming)):
                audit={'row':begin+k,'token_sha256':None,'state_free_nodes_sha256':None}
                if tok is not None:
                    valid=mask[k] if mask is not None else torch.ones(tok.shape[1],device=tok.device,dtype=torch.bool)
                    audit['token_sha256']=hashlib.sha256(tok[k][valid].float().cpu().numpy().tobytes()).hexdigest()
                    audit['sampled_count']=int(valid.sum())
                    if bank_nodes:audit['state_free_nodes_sha256']=hashlib.sha256(bank_nodes[0][k][valid].float().cpu().numpy().tobytes()).hexdigest()
                for field in ('xdl_uv','xdl_raw_uv','xdl_projection_valid','xdl_outside'):
                    if field in auxiliary:audit[field]=auxiliary[field][k].detach().cpu().numpy()
                observation_audits.append(audit)
        except Exception as error:
            failures.extend({'row':begin+k,'reason':type(error).__name__,'message':str(error)} for k in range(len(incoming)))
        finally:
            if handle is not None:handle.remove()
        elapsed+=time.perf_counter()-start
    model.xyz_keep_aux=old_aux
    return result,{'observation_audits':observation_audits,'failures':failures,'queries_attempted':attempts,'cold_batched_seconds':elapsed,'batch_size':batch_size}


@torch.no_grad()
def recursive_loop(model,data,rows,initials,beta,camera,device,batch_size=1):
    reset(model);states=np.asarray(initials,np.float32).copy();predictions=[states.copy()];failed=np.zeros(len(states),bool)
    failures=[];count=0;seconds=0.;intervals=[];offsets=[];observations=[]
    for step,row in enumerate(rows):
        interval=int(row['end_ms'])-int(row['previous_end_ms']);offset=int(row['event_start_ms'])-int(row['previous_end_ms'])
        if interval not in (49,50) or offset not in (0,1):raise ValueError('invalid actual raw state/event clock')
        intervals.append(interval);offsets.append(offset)
        valid=np.flatnonzero(~failed);nextstates=np.full_like(states,np.nan)
        if len(valid):
            ev=raw_events(data,row);p,audit=predict(model,ev,states[valid],beta,camera,device,row,batch_size)
            nextstates[valid]=p;count+=audit['queries_attempted'];seconds+=audit['cold_batched_seconds']
            for item in audit['observation_audits']:
                item=dict(item);item['row']=int(valid[item['row']]);item['step']=step+1;item['end_ms']=int(row['end_ms']);observations.append(item)
            for failure in audit['failures']:
                item=dict(failure);item['row']=int(valid[item['row']]);item['step']=step+1;failures.append(item)
        failed|=~np.isfinite(nextstates).all(-1);states=nextstates;predictions.append(states.copy())
    reset(model)
    return np.stack(predictions,axis=1),{'failures':failures,'queries_attempted':count,'cold_batched_seconds':seconds,
                                       'state_interval_ms':intervals,'query_start_offset_ms':offsets,'observations':observations,
                                       'unexecuted_after_failure':int(sum(len(initials)-np.sum(np.isfinite(p).all(-1)) for p in predictions[1:]))}


def reference_lookup(reference,name,row):
    source=reference[name]
    item=source[int(row['index'])]
    for key in ('run_id','end_ms','previous_end_ms'):
        if int(item[key])!=int(row[key]):raise ValueError(f'common trace timestamp mismatch: {name} {key}')
    return np.asarray(item['prev'],np.float32)


def load_reference(out=OUT):
    files=sorted(Path(out).glob('reference*.pt'))
    result={};identities=[]
    for path in files:
        cached=torch.load(path,map_location='cpu')
        if cached.get('bank_sha256')!=sha(Path(out)/'packet_bank.json'):raise ValueError(f'reference bank mismatch: {path}')
        if result.keys() & cached['traces'].keys():raise ValueError('duplicate sequences in common references')
        result.update(cached['traces']);identities.append({'path':str(path),'sha256':sha(path)})
    if not result:raise ValueError('missing frozen common C0 reference; run reference first')
    return result,identities


@torch.no_grad()
def create_reference(run,device,out=OUT,shard=0,shards=1,sequences=None):
    bank,banksha=load_bank(out);model,cfg,identity=load_model(run,device)
    if Path(identity['checkpoint']).name!='xa_c0_s3407-step=6000.ckpt':raise ValueError('common reference must be declared C0 seed3407')
    allnames=list(bank['sequences']);names=[n for i,n in enumerate(allnames) if i%shards==shard]
    if sequences:names=[n for n in names if n in sequences]
    target=Path(out)/f'reference_shard{shard}of{shards}.pt'
    if target.exists():
        saved=torch.load(target,map_location='cpu')
        if saved.get('bank_sha256')==banksha and saved.get('identity')==identity and set(saved['traces'])==set(names):
            print(f'identity-verified existing reference {target}',flush=True);return saved
        raise ValueError('reference exists with different contract; never overwrite')
    oldpath=REPO/'outputs/xb/c0_val_reference.pt';old=torch.load(oldpath,map_location='cpu');oldckpt=Path(old['checkpoint'])
    if sha(oldckpt)!=identity['checkpoint_sha256']:raise ValueError('legacy shared C0 reference weight identity mismatch')
    traces={};queries=0;start=time.perf_counter();rng=np.random.default_rng(0)
    seen_noise={}
    for seq,meta in bank['sequences'].items():
        if meta['cohort']!='seen':continue
        for rid,_ in enumerate(meta['valid_runs_ms']):
            if any(r['run_id']==rid for r in meta['rows']):seen_noise[(seq,rid)]=ET.sample_init_noise(cfg,rng,1.)
    # Exact original RNG stream follows zgz_global/local order in old reference;
    # seen streams independently seed0, stable per sequence, with identity recorded.
    for name in names:
        meta=bank['sequences'][name];data=load_data(cfg,name,meta);beta,camera=context(data,device);model.set_hand_context(beta,camera)
        if name in old['traces']:
            rows=old['traces'][name]
            if len(rows)!=len(meta['rows']):raise ValueError('old zgz trace length mismatch')
            for r,s in zip(meta['rows'],rows):
                if any(int(r[k])!=int(s[k]) for k in ('run_id','end_ms','previous_end_ms')) or int(r['n_raw_events'])!=int(s['n_raw_events']):raise ValueError('old C0 reference input clock/count mismatch')
            traces[name]=rows
        else:
            rows=[]
            for rid,(a,b) in enumerate(meta['valid_runs_ms']):
                segment=[r for r in meta['rows'] if r['run_id']==rid]
                if not segment:continue
                initial=data['gt'][a].copy()+seen_noise[(name,rid)]
                predicted,audit=recursive_loop(model,data,segment,initial[None],beta,camera,device,1);queries+=audit['queries_attempted']
                for i,row in enumerate(segment):
                    rows.append({**{k:row[k] for k in ('run_id','end_ms','previous_end_ms','n_raw_events')},'prev':predicted[0,i],'pred':predicted[0,i+1]})
            traces[name]=rows
        print(f'common C0 reference {name}: {len(traces[name])} packets ({len(traces)}/{len(names)})',flush=True)
    payload={'study':G.STUDY,'identity':identity,'bank_sha256':banksha,'traces':traces,'queries_attempted':queries,
             'elapsed_seconds':time.perf_counter()-start,'shard':shard,'shards':shards,'seen_init_rng':'global numpy rng0 over all 72 seen sequences in split order and nonempty valid runs, precomputed before sharding; GT+existing init noise',
             'zgz_source':{'path':str(oldpath),'sha256':sha(oldpath)}}
    tmp=target.with_suffix('.pt.tmp');torch.save(payload,tmp);tmp.replace(target);return payload


@torch.no_grad()
def fk(mano,states,beta,device):
    values=np.asarray(states,np.float32);shape=values.shape[:-1];flat=values.reshape(-1,51)
    joints=np.full((len(flat),21,3),np.nan,np.float32);verts=np.full((len(flat),778,3),np.nan,np.float32)
    finite=np.flatnonzero(np.isfinite(flat).all(-1))
    if len(finite):
        j,v=ET.mano_fk(mano,flat[finite],beta,device);joints[finite]=j;verts[finite]=v
    return joints.reshape(shape+(21,3)),verts.reshape(shape+(778,3))


def endpoint(mano,pred,truth,beta,device,mean):
    pred=np.asarray(pred,np.float32);truth=np.broadcast_to(np.asarray(truth,np.float32),pred.shape)
    pj,pv=fk(mano,pred,beta,device);gj,gv=fk(mano,truth,beta,device)
    pr,pl=state_rotations_safe(pred,mean);gr,gl=state_rotations_safe(truth,mean)
    return {'wrist_mm':np.linalg.norm(pj[...,0,:]-gj[...,0,:],axis=-1)*1000,
            'mpjpe_abs_mm':np.linalg.norm(pj-gj,axis=-1).mean(-1)*1000,
            'mpjpe_ra_mm':np.linalg.norm(ET.root_align(pj)-ET.root_align(gj),axis=-1).mean(-1)*1000,
            'mpvpe_abs_mm':np.linalg.norm(pv-gv,axis=-1).mean(-1)*1000,
            'mpvpe_ra_mm':np.linalg.norm(ET.root_align(pv)-ET.root_align(gv),axis=-1).mean(-1)*1000,
            'root_deg':safe_geodesic(pr,gr),'finger_deg':safe_geodesic(pl,gl).mean(-1)},pj,gj


def state_rotations_safe(states,mean):
    values=np.asarray(states,np.float64);shape=values.shape[:-1];flat=values.reshape(-1,51);valid=np.isfinite(flat).all(-1)
    root=np.full((len(flat),3,3),np.nan);local=np.full((len(flat),15,3,3),np.nan)
    if valid.any():root[valid],local[valid]=G.rotations(flat[valid],mean)
    return root.reshape(shape+(3,3)),local.reshape(shape+(15,3,3))


def safe_geodesic(a,b):
    a=np.asarray(a);b=np.asarray(b);shape=a.shape[:-2];flat=a.reshape(-1,3,3);other=b.reshape(-1,3,3);valid=np.isfinite(flat).all((1,2))&np.isfinite(other).all((1,2))
    out=np.full(len(flat),np.nan)
    if valid.any():out[valid]=G.geodesic_deg(flat[valid],other[valid])
    return out.reshape(shape)


def pair_metrics(mano,initials,outputs,beta,device,mean):
    ij,_=fk(mano,initials,beta,device);yj,_=fk(mano,outputs,beta,device)
    ir,il=state_rotations_safe(initials,mean);yr,yl=state_rotations_safe(outputs,mean)
    return [{'translation':G.vector_pair_metrics(ij[0,0]*1000,ij[i,0]*1000,yj[0,0]*1000,yj[i,0]*1000),
             'root':G.rotation_pair_metrics(ir[0],ir[i],yr[0],yr[i]),
             'finger':G.rotation_pair_metrics(il[0],il[i],yl[0],yl[i]),
             'joints_absolute':G.vector_pair_metrics(ij[0]*1000,ij[i]*1000,yj[0]*1000,yj[i]*1000),
             'joints_ra':G.vector_pair_metrics(ET.root_align(ij[0])*1000,ET.root_align(ij[i])*1000,ET.root_align(yj[0])*1000,ET.root_align(yj[i])*1000)} for i in range(len(initials))]


def aa_seams(state,mean):
    if not np.isfinite(state).all():return {'root':float('nan'),'local':np.full(15,np.nan)}
    root,local=G.rotations(state,mean)
    from scipy.spatial.transform import Rotation
    return {'root':float(np.pi-Rotation.from_matrix(root).magnitude()),'local':np.pi-Rotation.from_matrix(local).magnitude()}


def initial_states(mano,clean,natural,specs,mean,beta,device):
    clean=np.asarray(clean,np.float32)
    # MANO's actual wrist is J0(beta)+transl, invariant to root/local rotations.
    # Verify actual FK after all interventions instead of relying on transl alone.
    initials=np.stack([G.perturb_state(clean,s,mean,natural=natural) for s in specs])
    joints,_=fk(mano,initials,beta,device);baseline,_=fk(mano,clean[None],beta,device);delta=(joints[:,0]-baseline[0,0])*1000
    clean_root,clean_local=G.rotations(clean,mean);root,local=state_rotations_safe(initials,mean)
    root_angle=safe_geodesic(np.broadcast_to(clean_root,root.shape),root);local_angle=safe_geodesic(np.broadcast_to(clean_local,local.shape),local)
    records=[]
    for i,s in enumerate(specs):
        if s['block'] in ('clean','natural'):continue
        scale,angle=G.SCALES[s['scale']];expect=angle*180/np.pi
        wrist=np.linalg.norm(delta[i]);block=s['block'];checks={}
        if block in ('root','finger'):checks['wrist_preserved']=wrist<=.01
        if block in ('translation','mixed'):checks['translation_amplitude']=abs(wrist-scale*1000)<=.01
        if block in ('root','mixed'):checks['root_angle']=abs(root_angle[i]-expect)<=math.degrees(1e-4)
        else:checks['root_unchanged']=root_angle[i]<=math.degrees(1e-4)
        changed=[s['joint']] if 'joint' in s else list(range(15))
        if block in ('finger','mixed'):
            checks['finger_angles']=bool(np.max(abs(local_angle[i,changed]-expect))<=math.degrees(1e-4))
            untouched=[j for j in range(15) if j not in changed]
            if untouched:checks['other_fingers_unchanged']=bool(np.max(local_angle[i,untouched])<=math.degrees(1e-4))
        else:checks['fingers_unchanged']=bool(np.max(local_angle[i])<=math.degrees(1e-4))
        records.append({'spec_id':s['id'],'wrist_delta_mm':float(wrist),'root_angle_deg':float(root_angle[i]),'local_angles_deg':local_angle[i].tolist(),'checks':checks})
    passed=all(all(r['checks'].values()) for r in records)
    if not passed:raise ValueError('physical MANO intervention contract failed: '+str([r for r in records if not all(r['checks'].values())][:3]))
    return initials,{'passed':passed,'records':records}


@torch.no_grad()
def bank_diagnostic(model,mano,cfg,bank,reference,device,batch_size=1,limit=None):
    records=[];controls=[];empty=[];queries=0;seconds=0.;mean=mano.hands_mean.cpu().numpy();num=0
    for seq,meta in bank['sequences'].items():
        data=load_data(cfg,seq,meta);beta,camera=context(data,device);model.set_hand_context(beta,camera)
        for row in meta['bank']:
            if limit is not None and num>=limit:return {'rows':records,'controls':controls,'empty':empty,'queries_attempted':queries,'cold_batched_seconds':seconds,'partial_smoke':True}
            ev=raw_events(data,row)
            if hashlib.sha256(ev.tobytes()).hexdigest()!=row['events_sha256']:raise ValueError('fixed event packet identity drift')
            clean=data['gt'][row['previous_end_ms']].copy();natural=reference_lookup(reference,seq,row);specs=row['specs']
            initials,physical=initial_states(mano,clean,natural,specs,mean,beta,device);reset(model)
            output,audit=predict(model,ev,initials,beta,camera,device,row,batch_size);queries+=audit['queries_attempted'];seconds+=audit['cold_batched_seconds']
            errors,_,_=endpoint(mano,output,data['gt'][row['end_ms']],beta,device,mean);paired=pair_metrics(mano,initials,output,beta,device,mean)
            base={'seq':seq,'cohort':meta['cohort'],'subject':seq.split('_')[0],'packet_id':row['packet_id'],'clock':{k:row[k] for k in ('end_ms','previous_end_ms','state_interval_ms','query_start_offset_ms')},'event_sha256':row['events_sha256'],'n_raw_events':len(ev)}
            for i,s in enumerate(specs):records.append({**base,'spec':s,'initial':initials[i],'aa_seam_distance_rad':aa_seams(initials[i],mean),'prediction':output[i],'endpoint':{k:v[i] for k,v in errors.items()},'paired':paired[i],
                'observation':next((a for a in audit['observation_audits'] if a['row']==i),{}),'finite':bool(np.isfinite(output[i]).all()),'failures':[f for f in audit['failures'] if f['row']==i]})
            if meta['cohort']=='zgz':
                wanted=row['control_specs'];idx=[next(i for i,s in enumerate(specs) if s['id']==w['id']) for w in wanted]
                for mode in ('swap','mirror'):
                    transformed,transaudit=G.transform_events(ev,mode);reset(model)
                    wrong,a=predict(model,transformed,initials[idx],beta,camera,device,row,batch_size);queries+=a['queries_attempted'];seconds+=a['cold_batched_seconds']
                    err,_,_=endpoint(mano,wrong,data['gt'][row['end_ms']],beta,device,mean)
                    for j,i in enumerate(idx):controls.append({**base,'mode':mode,'transform':transaudit,'spec':specs[i],'initial':initials[i],'prediction':wrong[j],
                        'endpoint':{k:v[j] for k,v in err.items()},'real_event_endpoint':{k:v[i] for k,v in errors.items()},'output_change_l2':float(np.linalg.norm(wrong[j]-output[i])),
                        'observation':next((v for v in a['observation_audits'] if v['row']==j),{}),'real_event_observation':next((v for v in audit['observation_audits'] if v['row']==i),{}),'finite':bool(np.isfinite(wrong[j]).all()),'failures':[f for f in a['failures'] if f['row']==j]})
                reset(model);hold,a=predict(model,np.zeros((0,5),np.float32),initials[:1],beta,camera,device,row,1);queries+=1;seconds+=a['cold_batched_seconds']
                empty.append({**base,'exact_hold':bool(np.array_equal(hold[0],initials[0])),'initial':initials[0],'prediction':hold[0],'failures':a['failures']})
            num+=1;print(f'bank {row["packet_id"]}: {len(specs)} priors ({num}/{bank["counts"]["packets"]})',flush=True)
    return {'rows':records,'controls':controls,'empty':empty,'queries_attempted':queries,'cold_batched_seconds':seconds,'physical_contract_passed':True,'partial_smoke':False}


def score_trajectories(mano,predictions,truth,times,specs,initials,beta,device,mean):
    errors,pj,gj=endpoint(mano,predictions,np.broadcast_to(truth,predictions.shape),beta,device,mean)
    roots,locals_=state_rotations_safe(predictions,mean);gtroots,gtlocals=state_rotations_safe(truth,mean);records=[]
    for i,s in enumerate(specs):
        paired={
            'translation':np.linalg.norm((pj[i,:,0]-pj[0,:,0])*1000,axis=-1),
            'root':np.linalg.norm(roots[i]-roots[0],axis=(-1,-2)),
            'finger':np.linalg.norm((locals_[i]-locals_[0]).reshape(len(times),-1),axis=-1)}
        rho={block:(None if not np.isfinite(d[0]) or d[0]<=1e-9 else d/d[0]) for block,d in paired.items()}
        record={'spec':s,'initial':initials[i],'predictions':predictions[i],'times_ms':times,'errors':{k:v[i] for k,v in errors.items()},
                'auc':{k:G.time_auc(v[i],times) for k,v in errors.items()},'paired_distance':paired,'rho':rho,
                'rho_auc':{k:None if v is None else G.time_auc(v,times) for k,v in rho.items()},
                'recovery':G.sustained_recovery(errors['mpjpe_abs_mm'][i],errors['mpjpe_ra_mm'][i],times),
                'half_recovery':{k:G.sustained_half(v,times) for k,v in paired.items()},
                'finite':bool(np.isfinite(predictions[i]).all()),
                'motion':{'absolute':G.motion_response(pj[i],gj[i],times,False),'root_relative':G.motion_response(pj[i],gj[i],times,True),
                          'root':G.rotation_motion_response(roots[i],gtroots,times),'finger':G.rotation_motion_response(locals_[i],gtlocals,times),
                          'raw50_root':G.rotation_motion_response(roots[i],gtroots,times,50,False),'raw50_finger':G.rotation_motion_response(locals_[i],gtlocals,times,50,False),
                          'raw50_absolute':G.motion_response(pj[i],gj[i],times,False,50,False),'raw50_root_relative':G.motion_response(pj[i],gj[i],times,True,50,False)}}
        records.append(record)
    return records


@torch.no_grad()
def recovery_diagnostic(model,mano,cfg,bank,reference,device,legacy=False,limit=None,batch_size=1):
    result=[];queries=0;seconds=0.;mean=mano.hands_mean.cpu().numpy();count=0
    for seq,meta in bank['sequences'].items():
        if legacy and meta['cohort']!='zgz':continue
        data=load_data(cfg,seq,meta);beta,camera=context(data,device);model.set_hand_context(beta,camera)
        for clip in meta['clips']:
            if limit is not None and count>=limit:return {'rows':result,'queries_attempted':queries,'cold_batched_seconds':seconds,'partial_smoke':True}
            rows=[meta['rows'][i] for i in clip['end_indices']];first=rows[0];initial_time=first['previous_end_ms']
            specs=G.recovery_specs(clip['clip_id']);initials,physical=initial_states(mano,data['gt'][initial_time],reference_lookup(reference,seq,first),specs,mean,beta,device)
            pred,audit=recursive_loop(model,data,rows,initials,beta,camera,device,batch_size);queries+=audit['queries_attempted'];seconds+=audit['cold_batched_seconds']
            times=np.asarray([initial_time]+[r['end_ms'] for r in rows]);truth=data['gt'][times]
            scored=score_trajectories(mano,pred,truth,times,specs,initials,beta,device,mean)
            for i,r in enumerate(scored):result.append({'seq':seq,'cohort':meta['cohort'],'subject':seq.split('_')[0],'clip_id':clip['clip_id'],'run_id':clip['run_id'],
                'clock':{'state_interval_ms':audit['state_interval_ms'],'query_start_offset_ms':audit['query_start_offset_ms']},
                'observation_audits':[f for f in audit['observations'] if f['row']==i],'failures':[f for f in audit['failures'] if f['row']==i],**r})
            count+=1;print(f'recovery {clip["clip_id"]}: 18 own-state trajectories x20 ({count})',flush=True)
    return {'rows':result,'queries_attempted':queries,'cold_batched_seconds':seconds,'partial_smoke':False}


@torch.no_grad()
def full_motion(model,mano,cfg,bank,reference,device,limit=None,batch_size=1):
    result=[];short=[];queries=0;seconds=0.;mean=mano.hands_mean.cpu().numpy();count=0
    for seq,meta in bank['sequences'].items():
        if meta['cohort']!='zgz':continue
        data=load_data(cfg,seq,meta);beta,camera=context(data,device);model.set_hand_context(beta,camera)
        for rid,(a,b) in enumerate(meta['valid_runs_ms']):
            rows=[r for r in meta['rows'] if r['run_id']==rid]
            if not rows:continue
            if limit is not None and count>=limit:return {'rows':result,'short_natural':short,'queries_attempted':queries,'cold_batched_seconds':seconds,'partial_smoke':True}
            specs=G.recovery_specs(f'{seq}:run{rid}',small_only=True);initials,physical=initial_states(mano,data['gt'][a],None,specs,mean,beta,device)
            pred,audit=recursive_loop(model,data,rows,initials,beta,camera,device,batch_size);queries+=audit['queries_attempted'];seconds+=audit['cold_batched_seconds']
            times=np.asarray([a]+[r['end_ms'] for r in rows]);truth=data['gt'][times]
            scored=score_trajectories(mano,pred,truth,times,specs,initials,beta,device,mean)
            for i,r in enumerate(scored):result.append({'seq':seq,'cohort':'zgz','subject':'zgz','run_id':rid,'n_queries':len(rows),
                'observation_audits':[f for f in audit['observations'] if f['row']==i],'failures':[f for f in audit['failures'] if f['row']==i],**r})
            if len(rows)<20:
                natural=reference_lookup(reference,seq,rows[0]);npred,naudit=recursive_loop(model,data,rows,natural[None],beta,camera,device,batch_size)
                queries+=naudit['queries_attempted'];seconds+=naudit['cold_batched_seconds']
                # Pair short natural against clean's own loop, preserving short-run coverage.
                combined=np.concatenate([pred[:1],npred],axis=0);ispec=[specs[0],{'block':'natural','id':'natural'}]
                nscore=score_trajectories(mano,combined,truth,times,ispec,np.stack([initials[0],natural]),beta,device,mean)[1]
                short.append({'seq':seq,'cohort':'zgz','run_id':rid,'n_queries':len(rows),'observation_audits':naudit['observations'],'failures':naudit['failures'],**nscore})
            count+=1;print(f'motion {seq} run{rid}: {len(rows)} x9, short={len(rows)<20}',flush=True)
    return {'rows':result,'short_natural':short,'queries_attempted':queries,'cold_batched_seconds':seconds,'partial_smoke':False}


def describe(values):
    values=list(values);defined=[v for v in values if v is not None]
    result={'n':len(values),'n_defined':len(defined),'n_undefined':len(values)-len(defined),'finite':all(np.isfinite(v) for v in defined)}
    if not defined or not result['finite']:return {**result,'mean':None,'median':None,'min':None,'max':None}
    arr=np.asarray(defined,np.float64);return {**result,'mean':float(arr.mean()),'median':float(np.median(arr)),'min':float(arr.min()),'max':float(arr.max())}


def summarize_bank(value):
    result={};groups=defaultdict(list)
    for r in value['rows']:groups[(r['cohort'],r['seq'] if r['cohort']=='zgz' else r['subject'],r['spec']['block'],r['spec'].get('scale','none'))].append(r)
    for key,rows in groups.items():
        item={'n_queries':len(rows),'n_failures':sum(not r['finite'] for r in rows),'endpoint':{k:describe(r['endpoint'][k] for r in rows) for k in rows[0]['endpoint']}}
        block=key[2]
        if block in ('translation','root','finger'):
            values=[r['paired'][block]['rho'] for r in rows];item['rho']=describe(values)
            defined=[v for v in values if v is not None];item['rho_fraction_lt1']=None if not defined or not item['rho']['finite'] else float(np.mean(np.asarray(defined)<1))
        elif block in ('natural','mixed'):
            item['rho_by_block']={b:describe(r['paired'][b]['rho'] for r in rows) for b in ('translation','root','finger')}
        result['|'.join(key)]=item
    return {'groups':result,'n_rows':len(value['rows']),'n_controls':len(value['controls']),'n_empty':len(value['empty']),
            'empty_all_exact_hold':all(r['exact_hold'] for r in value['empty']),
            'mandatory_finite':all(r['finite'] for r in value['rows']),
            'control_finite':all(r['finite'] for r in value['controls']),
            'queries_attempted':value['queries_attempted'],'cold_batched_seconds':value['cold_batched_seconds']}


def summarize_trajectories(value):
    result={};groups=defaultdict(list)
    for r in value['rows']:groups[(r['cohort'],r['seq'] if r['cohort']=='zgz' else r['subject'],r['spec']['block'],r['spec'].get('scale','none'))].append(r)
    for key,rows in groups.items():
        result['|'.join(key)]={'n_trajectories':len(rows),'n_failures':sum(not r['finite'] for r in rows),
            'auc':{k:describe(r['auc'][k] if r['finite'] else float('nan') for r in rows) for k in rows[0]['auc']},
            'rho_auc':{k:describe(r['rho_auc'][k] if r['finite'] else float('nan') for r in rows) for k in rows[0]['rho_auc']},
            'initially_satisfied':sum(r['recovery']['initially_satisfied'] for r in rows),
            'sustained_recovered':sum(r['recovery']['hit_step'] is not None for r in rows),
            'right_censored':sum(r['recovery']['right_censored'] for r in rows),
            'motion':{space:{metric:describe(r['motion'][space][metric] if r['finite'] else float('nan') for r in rows) for metric in ('gain','gain_bias','cosine','lag_ms')} for space in ('absolute','root_relative','root','finger')}}
    return {'groups':result,'n_rows':len(value['rows']),'n_short_natural':len(value.get('short_natural',[])),
            'mandatory_finite':all(r['finite'] for r in value['rows']) and all(r['finite'] for r in value.get('short_natural',[])),
            'queries_attempted':value['queries_attempted'],'cold_batched_seconds':value['cold_batched_seconds']}


def save_raw(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with gzip.open(tmp,'wb',compresslevel=3) as handle:torch.save(value,handle)
    tmp.replace(path)


def load_raw(path):
    with gzip.open(path,'rb') as handle:return torch.load(handle,map_location='cpu')


def save_summary(path,payload):
    invalid=[];clean=G.sanitize(payload,invalid=invalid)
    clean.update(invalid_numeric_paths=invalid,numeric_valid=not invalid,numeric_status='FAIL' if invalid else 'PASS')
    write_json(path,clean)


def diagnose(run,device,out=OUT,config=None,legacy=False,sections=('bank','recovery','motion'),batch_size=1,limit=None):
    if batch_size<1:raise ValueError('batch_size must be positive')
    if batch_size!=1 and limit is None:raise ValueError('formal diagnostics require serial batch_size=1; GPU float32 FK/barycentric batch equivalence failed')
    bank,banksha=load_bank(out);reference,refids=load_reference(out);model,cfg,identity=load_model(run,device,config)
    mano=ManoLayer(cfg['MANO']['NPZ'],add_mean=False).to(device).eval();mano.fast_chain=True
    name=Path(identity['checkpoint']).parent.name;target=Path(out)/'diagnostics'/name;target.mkdir(parents=True,exist_ok=True)
    summaries={};start=time.perf_counter()
    for section in sections:
        rawpath=target/f'{section}.pt.gz';jsonpath=target/f'{section}.json'
        if rawpath.exists() and jsonpath.exists() and limit is None:
            old=json.loads(jsonpath.read_text())
            if old['identity']!=identity or old['bank_sha256']!=banksha or old['references']!=refids or old.get('prediction_batch_size')!=batch_size:raise ValueError('existing diagnostic identity drift')
            summaries[section]=old;continue
        if section=='bank':value=bank_diagnostic(model,mano,cfg,bank,reference,device,batch_size,limit);summary=summarize_bank(value)
        elif section=='recovery':value=recovery_diagnostic(model,mano,cfg,bank,reference,device,legacy,limit,batch_size);summary=summarize_trajectories(value)
        elif section=='motion':value=full_motion(model,mano,cfg,bank,reference,device,limit,batch_size);summary=summarize_trajectories(value)
        else:raise ValueError(section)
        payload={'study':G.STUDY,'identity':identity,'bank_sha256':banksha,'references':refids,'section':section,'legacy':legacy,
            'partial_smoke':limit is not None,'prediction_batch_size':batch_size,'summary':summary,'measurement_contract':{'target':'zero','rotation_direction':'left/right conditional; contraction primary',
            'clock':'actual incoming49ms first,50ms later; event offset0/1ms','motion':'exact200ms displacements; centered5-point GT smoothing only offline (SO3 chordal mean for root/local); edge/low-energy/flat lag undefined',
            'latency':'cold batched model queries includes FK/association/readout; sensor waiting/capture not measured'}}
        save_raw(rawpath,{'metadata':payload,'data':value});payload['raw']={'path':str(rawpath),'sha256':sha(rawpath)};save_summary(jsonpath,payload);summaries[section]=payload
    summary={'study':G.STUDY,'identity':identity,'bank_sha256':banksha,'references':refids,'sections':summaries,
             'elapsed_seconds':time.perf_counter()-start,'partial_smoke':limit is not None}
    save_summary(target/'diagnostic.json',summary);print(json.dumps({'diagnostic':str(target/'diagnostic.json'),'elapsed_seconds':summary['elapsed_seconds']},ensure_ascii=False),flush=True)
    return summary


def refresh_motion_counts(value):
    """Revise ancillary exact-endpoint counts while retaining every numerical score."""
    revised=copy.deepcopy(value);changes=[]
    for collection in ('rows','short_natural'):
        for index,row in enumerate(revised.get(collection,[])):
            times=set(int(t) for t in row.get('times_ms',[]))
            for space,motion in row.get('motion',{}).items():
                if 'eligible_time_windows' not in motion:continue
                horizon=int(motion['horizon_ms'])
                count=sum(t+horizon in times for t in times)
                original=motion['eligible_time_windows']
                motion['eligible_time_windows']=count
                if original!=count:changes.append({'path':f'{collection}[{index}].motion.{space}.eligible_time_windows','old':original,'new':count})
    return revised,changes


def refresh_summary(path):
    """Write a new JSON audit; original GPU raw/JSON files remain immutable."""
    input_path=Path(path);input_sha=sha(input_path)
    if input_path.suffix=='.json':
        source_json=json.loads(input_path.read_text());path=Path(source_json['raw']['path'])
        if sha(path)!=source_json['raw']['sha256']:raise ValueError('original JSON/raw identity mismatch')
    else:path=input_path
    original=load_raw(path);value,changes=refresh_motion_counts(original['data'])
    metadata=copy.deepcopy(original['metadata']);section=metadata['section']
    checks={'gain_cos_lag_unchanged':True,'prediction_arrays_unchanged':True,'error_arrays_unchanged':True}
    for collection in ('rows','short_natural'):
        for old,row in zip(original['data'].get(collection,[]),value.get(collection,[])):
            for space,motion in old.get('motion',{}).items():
                for field in ('gain','gain_bias','cosine','lag_ms','lag_correlations'):
                    if G.sanitize(motion.get(field))!=G.sanitize(row['motion'][space].get(field)):checks['gain_cos_lag_unchanged']=False
            for field in ('initial','prediction','predictions'):
                if field in old and not np.array_equal(old[field],row[field],equal_nan=True):checks['prediction_arrays_unchanged']=False
            for field in ('endpoint','errors','paired_distance','rho'):
                for key,observed in old.get(field,{}).items():
                    if observed is None:
                        if row[field][key] is not None:checks['error_arrays_unchanged']=False
                    elif not np.array_equal(observed,row[field][key],equal_nan=True):checks['error_arrays_unchanged']=False
    if not all(checks.values()):raise AssertionError('reanalysis changed scientific observations')
    metadata['summary']=summarize_bank(value) if section=='bank' else summarize_trajectories(value)
    metadata['original_file']={'path':str(input_path),'sha256':input_sha}
    metadata['raw']={'path':str(path),'sha256':sha(path)}
    metadata['reanalysis']={'source_raw':str(path),'source_raw_sha256':sha(path),
        'reason':'Exact200ms eligible-window count honors first49ms interval. Original gain/cos/lag and all numeric state/error arrays already used exact timestamps and are retained.',
        'patched_ancillary_counts':changes,'verification':checks,'prediction_rerun':False,
        'score_implementation_sha256':sha(REPO/'tools/xdl/diagnostics.py'),
        'source':[{'path':str(REPO/p),'sha256':sha(REPO/p)} for p in ('tools/xdl/prepare.py','tools/xdl/geometry.py','tools/xdl/diagnostics.py')]}
    name=path.name.removesuffix('.pt.gz');jsonpath=path.with_name(name+'.reanalysis.json');save_summary(jsonpath,metadata)
    print(json.dumps({'reanalysis':str(jsonpath),'source_raw_sha256':metadata['reanalysis']['source_raw_sha256'],'changed_ancillary_fields':len(changes),'verification':checks},ensure_ascii=False),flush=True)
    return metadata


def score(path):
    raw=load_raw(path);section=raw['metadata']['section'];summary=summarize_bank(raw['data']) if section=='bank' else summarize_trajectories(raw['data'])
    result={'raw_sha256':sha(path),'section':section,'summary':summary};print(json.dumps(G.sanitize(result),ensure_ascii=False,allow_nan=False));return result


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('reference');r.add_argument('--run',default='xa_c0_s3407');r.add_argument('--device',default='cuda:0');r.add_argument('--out',default=str(OUT));r.add_argument('--shard',type=int,default=0);r.add_argument('--shards',type=int,default=1);r.add_argument('--sequences')
    d=sub.add_parser('diagnose');d.add_argument('--run',required=True);d.add_argument('--config');d.add_argument('--device',default='cuda:0');d.add_argument('--out',default=str(OUT));d.add_argument('--legacy',action='store_true');d.add_argument('--sections',default='bank,recovery,motion');d.add_argument('--batch-size',type=int,default=1);d.add_argument('--limit',type=int)
    s=sub.add_parser('score');s.add_argument('input',nargs='?');s.add_argument('--raw');s.add_argument('--refresh-summary',action='store_true')
    args=p.parse_args();torch.set_num_threads(8)
    if args.command=='reference':create_reference(args.run,torch.device(args.device),args.out,args.shard,args.shards,args.sequences.split(',') if args.sequences else None)
    elif args.command=='diagnose':diagnose(args.run,torch.device(args.device),args.out,args.config,args.legacy,tuple(args.sections.split(',')),args.batch_size,args.limit)
    elif args.refresh_summary:
        if not (args.raw or args.input):p.error('score needs sectionJSON positional or --raw')
        refresh_summary(args.raw or args.input)
    else:
        if not (args.raw or args.input):p.error('score needs --raw or raw positional')
        score(args.raw or args.input)

if __name__=='__main__':main()
