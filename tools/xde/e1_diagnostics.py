#!/usr/bin/env python3
"""E1: immutable-XDL-output reanalysis and batch-one counterfactual inference.

Only writes outputs/xde/e1. CUDA workers require an explicit singleton physical
GPU 2 or 3 visibility mask. Prepare freezes selection before any new inference.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
from collections import defaultdict
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'model'))
OUT=ROOT/'outputs/xde/e1'
RUNS=['xdl_q1_s3407','xdl_q0_s3407','xdl_n1_s3407','xa_c0_s3407']
SEQS=['zgz_local','zgz_global']
SEED=20261014

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def rel(p):
    try:return Path(p).absolute().relative_to(ROOT).as_posix()
    except ValueError:return os.path.relpath(p,ROOT)

def ident(p):return {'path':rel(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}

def clean_json(x):
    if isinstance(x,dict):return {str(k):clean_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean_json(v) for v in x]
    if isinstance(x,np.ndarray):return clean_json(x.tolist())
    if isinstance(x,np.generic):return clean_json(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def write_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(clean_json(x),ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n');t.replace(p)

def load_raw(p):
    with gzip.open(p,'rb') as f:return torch.load(f,map_location='cpu',weights_only=False)

def save_raw(p,x):
    t=Path(str(p)+'.tmp')
    with gzip.open(t,'wb',compresslevel=2) as f:torch.save(x,f)
    t.replace(p)

def meta51(p):
    with Path(p).open('rb') as f:
        n=struct.unpack('<i',f.read(4))[0];assert n==51
        a=np.fromfile(f,dtype=np.dtype([('data',np.float64,51),('m0',np.uint8),('m1',np.uint8)]))
    assert np.all(a['m0']==4) and np.all(a['m1']==13)
    return a['data'].astype(np.float32)

def gtpath(bank,seq):
    return Path(next(s['path'] for s in bank['sequences'][seq]['sources'] if s['path'].endswith('.meta')))

def midpoint(items,n):
    if len(items)<=n:return list(items)
    return [items[int(np.floor((i+.5)*len(items)/n))] for i in range(n)]

def stats(a):
    a=np.asarray(a,np.float64)
    if not len(a):return {'n':0,'n_finite':0,'mean':None,'median':None}
    valid=np.isfinite(a).all(axis=tuple(range(1,a.ndim))) if a.ndim>1 else np.isfinite(a)
    good=a[valid]
    return {'n':len(a),'n_finite':int(valid.sum()),'mean':np.mean(good,axis=0) if len(good) else None,
            'median':np.median(good,axis=0) if len(good) else None,'p90':np.quantile(good,.9,axis=0) if len(good) else None}

def error_summary(pred,initial,truth):
    e=(np.asarray(pred,np.float64)[:,:3]-np.asarray(truth,np.float64)[:,:3])*1000
    h=(np.asarray(initial,np.float64)[:,:3]-np.asarray(truth,np.float64)[:,:3])*1000
    ne,nh=np.linalg.norm(e,axis=1),np.linalg.norm(h,axis=1)
    update=e-h;wanted=-h;den=np.linalg.norm(update,axis=1)*np.linalg.norm(wanted,axis=1)
    cos=np.divide(np.sum(update*wanted,axis=1),den,out=np.full(len(e),np.nan),where=den>1e-8)
    return {'signed_error_xyz_mm':stats(e),'wrist_mm':stats(ne),'hold_wrist_mm':stats(nh),
            'improvement_over_hold_mm':stats(nh-ne),'n_better_than_hold':int(np.sum(ne<nh)),
            'n_worse_than_hold':int(np.sum(ne>nh)),'negative_z_fraction':float(np.mean(e[:,2]<0)),
            'z_squared_error_fraction':float(np.sum(e[:,2]**2)/np.sum(e**2)) if np.sum(e**2)>0 else None,
            'direction_to_target_cosine':stats(cos)}

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Protocol exists; do not overwrite frozen selection')
    bankpath=ROOT/'outputs/xdl/packet_bank.json';bank=json.loads(bankpath.read_text())
    mp=ROOT/'outputs/xdl/diagnostics/xdl_q1_s3407/motion.pt.gz';motion=load_raw(mp)['data']['rows']
    chosen={};counts={};sources=[ident(bankpath),ident(mp)]
    for seq in SEQS:
        gt=meta51(gtpath(bank,seq));meta=bank['sequences'][seq]
        look={r['run_id']:r for r in motion if r['seq']==seq and r['spec']['block']=='clean'}
        candidates=[]
        for rid,m in sorted(look.items()):
            rows=[r for r in meta['rows'] if r['run_id']==rid]
            assert len(m['predictions'])==len(rows)+1
            # Skip the initial GT state; use nonoverlapping windows from query1 onward.
            for off in range(1,len(rows)-20+1,20):
                row=rows[off];initial=np.asarray(m['predictions'][off],np.float32)
                t=int(row['previous_end_ms']);assert int(m['times_ms'][off])==t
                if not np.isfinite(initial).all():continue
                z=float((float(initial[2])-float(gt[t,2]))*1000)
                bucket=int(np.digitize(z,[-50.,-10.,10.,50.]))
                candidates.append({'seq':seq,'run_id':rid,'offset':off,'end_indices':[int(r['index']) for r in rows[off:off+20]],
                                   'initial':initial.tolist(),'initial_time_ms':t,'initial_z_error_mm':z,'z_bucket':bucket,
                                   'clip_id':f'{seq}:own:{rid}:{t}'})
        pick=[]
        for b in range(5):pick.extend(midpoint([c for c in candidates if c['z_bucket']==b],4))
        used={x['clip_id'] for x in pick};pick+=midpoint([c for c in candidates if c['clip_id'] not in used],24-len(pick))
        chosen[seq]=sorted(pick,key=lambda x:(x['run_id'],x['offset']))
        counts[seq]={'eligible_own_clips':len(candidates),'selected':len(pick),'candidate_z_bins':{str(b):sum(c['z_bucket']==b for c in candidates) for b in range(5)}}
    for run in RUNS:
        for section in ['bank']:
            sources.append(ident(ROOT/f'outputs/xdl/diagnostics/{run}/{section}.pt.gz'))
        meta=json.loads((ROOT/f'outputs/xdl/diagnostics/{run}/bank.json').read_text())
        for name in ['checkpoint','config']:sources.append(ident(Path(meta['identity'][name])))
    for seq,meta in bank['sequences'].items():
        for s in meta['sources']:
            # Rehash metadata used for the new CPU calculation, not merely copy old hashes.
            p=Path(s['path']);found=ident(p);assert found['sha256']==s['sha256'];sources.append(found)
    sourcepaths=['tools/xde/e1_diagnostics.py','tools/xdl/diagnostics.py','tools/xdl/geometry.py','semkine/xyz_recovery_study.py',
                 'semkine/xyz_observation.py','semkine/xyz_mesh.py','semkine/encoder.py','semkine/eval_track.py','model/model.py','model/mano_layer.py']
    sourceids=[ident(ROOT/p) for p in sourcepaths if (ROOT/p).exists()]
    protocol={'study':'XDE-E1-20261014','seed':SEED,'batch_size':1,'runs':RUNS,'physical_gpus':[2,3],
              'budget_gpu_hours':2,'max_worker_wall_seconds':3500,'weights_fixed':True,'target':'zero for all model calls',
              'clean_sequence':'zgz_local','clean_all_packet_ids':[r['packet_id'] for r in bank['sequences']['zgz_local']['rows']],
              'clean_expected_per_run':len(bank['sequences']['zgz_local']['rows']),
              'own_selection':{'rule':'nonoverlapping20query windows, offset1 then21 etc within existing Q1 clean-start own trajectories; four chronological midpoint ranks per fixed signed-z bin then chronological midpoint fill to24; no future outcome used',
                               'z_bin_edges_mm':[-50,-10,10,50],'counts':counts,'selected':chosen},
              'interventions':{'time_shuffle':'Within each polarity randomly permute original timestamp values among unchanged xy events; stable sort by reassigned timestamp. Deterministic SHA256 seed of packet_id + fixed20261014. Preserves count/xy/p and timestamp multisets; changes temporal spatial correspondence and possibly stride-sampled support.',
                               'own_paired':'At each saved factual Q1 incoming state compare new real and shuffled one-step outputs with holding that same state.',
                               'own_rollout':'From same selected initial, separately recurse real and shuffled streams for20queries; hold branch keeps same initial51 for every step.',
                               'bank_controls':'Existing 24bank packets/sequence: sameclean and signed z +/-5,+/-50mm; add strong shuffle only, real already saved; rerun real clean for numeric identity check.'},
              'cpu_strata':{'depth_mm':[600,650,700],'event_count':[128,2048],'true_wrist_motion_mm':[1,5],'run_age_ms':[200,1000]},
              'bias_gate':{'fraction_worse_than_hold_min':.75,'fraction_negative_z_min':.75,'z_squared_fraction_min':.8},
              'score':'fixed beta wrist vector equivalent to transl; mm; normal actual next-time motion included in hold; rotations via existing XDL SO3; RA mesh still vertex0 historical definition',
              'numeric_contract':'EventHandsTrain Python; torch fp32 eval; batch1 model inference; cudnn.allow_tf32=True and cuda.matmul.allow_tf32=False match unmodified XDL diagnostics defaults; torch CPU threads8. FK scoring can batch because it cannot affect predictions; saved-output replay differences reported.',
              'inputs':sources,'source':sourceids}
    write_json(OUT/'protocol.json',protocol)
    print(json.dumps({'protocol':rel(OUT/'protocol.json'),'counts':counts,'clean_packets':protocol['clean_expected_per_run']}),flush=True)

def transform(ev,packet_id):
    original=np.asarray(ev);v=original.copy();n=len(v)
    rng=np.random.default_rng(int(hashlib.sha256(f'{SEED}|{packet_id}'.encode()).hexdigest()[:16],16))
    for p in np.unique(v[:,4]):
        ii=np.flatnonzero(v[:,4]==p);v[ii,3]=original[rng.permutation(ii),3]
    changed=float(np.mean(v[:,3]!=original[:,3])) if n else None
    mean_shift=float(np.mean(np.abs(v[:,3]-original[:,3]))*1000) if n else None
    corr=float(np.corrcoef(original[:,3],v[:,3])[0,1]) if n>1 and np.std(original[:,3])>0 and np.std(v[:,3])>0 else None
    order=np.argsort(v[:,3],kind='stable');v=v[order]
    ns=min(n,2048);idx=(np.arange(ns,dtype=np.float32)*(np.float32(n)/max(ns,1))).astype(np.int64)
    jac=len(set(idx)&set(order[idx]))/max(1,len(set(idx)|set(order[idx]))) if n else None
    assert len(v)==n and np.array_equal(np.sort(v[:,3]),np.sort(original[:,3]))
    return v,{'count':n,'timestamp_changed_fraction':changed,'mean_abs_timestamp_shift_ms':mean_shift,
              'original_reassigned_time_correlation':corr,'stride_sample_original_identity_jaccard':jac,
              'original_sha256':hashlib.sha256(original.tobytes()).hexdigest(),'transformed_sha256':hashlib.sha256(v.tobytes()).hexdigest()}

def worker(group):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in ('2','3'):raise RuntimeError('E1 requires explicit singleton CUDA_VISIBLE_DEVICES=2 or3')
    torch.set_num_threads(8);torch.set_num_interop_threads(2)
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    from tools.xdl import diagnostics as D
    started=time.monotonic();device=torch.device('cuda:0');protocol=json.loads((OUT/'protocol.json').read_text())
    protocolsha=sha(OUT/'protocol.json');bank=json.loads((ROOT/'outputs/xdl/packet_bank.json').read_text())
    runs=RUNS[:1] if group=='q1' else RUNS[1:]
    def budget():
        if time.monotonic()-started>protocol['max_worker_wall_seconds']:raise RuntimeError('worker wall budget exhausted')
    for run in runs:
        budget();model,cfg,identity=D.load_model(run,device)
        mano=D.ManoLayer(cfg['MANO']['NPZ'],add_mean=False).to(device).eval();mano.fast_chain=True
        seq='zgz_local';meta=bank['sequences'][seq];data=D.load_data(cfg,seq,meta);beta,camera=D.context(data,device);model.set_hand_context(beta,camera)
        rows=meta['rows'];preds=[];audits=[];s0=time.monotonic()
        for i,row in enumerate(rows):
            budget();D.reset(model);ev=D.raw_events(data,row);initial=data['gt'][row['previous_end_ms']]
            pred,audit=D.predict(model,ev,initial[None],beta,camera,device,row,1);preds.append(pred[0]);audits.append(audit)
            if i==99:
                elapsed=time.monotonic()-s0
                write_json(OUT/f'{run}_speed100.json',{'run':run,'packets':100,'elapsed_s':elapsed,'ms_per_packet':elapsed*10,'estimated_full_clean_s':elapsed*len(rows)/100,'physical_gpu':visible})
                print(json.dumps({'speed100':run,'seconds':elapsed,'full_clean_estimate_s':elapsed*len(rows)/100}),flush=True)
        p=np.stack(preds);initial=np.stack([data['gt'][r['previous_end_ms']] for r in rows]);truth=np.stack([data['gt'][r['end_ms']] for r in rows])
        mean=mano.hands_mean.cpu().numpy();errors,_,_=D.endpoint(mano,p,truth,beta,device,mean);held,_,_=D.endpoint(mano,initial,truth,beta,device,mean)
        summary=error_summary(p,initial,truth);summary.update({k:stats(v) for k,v in errors.items() if k!='wrist_mm'})
        summary['bias_gate_pass']=summary['n_worse_than_hold']/len(rows)>=.75 and summary['negative_z_fraction']>=.75 and summary['z_squared_error_fraction']>=.8
        raw={'identity':identity,'protocol_sha256':protocolsha,'seq':seq,'rows':rows,'initial':initial,'predictions':p,'truth':truth,'errors':errors,'hold_errors':held,'audits':audits}
        path=OUT/f'{run}_clean_local.pt.gz';save_raw(path,raw)
        write_json(OUT/f'{run}_clean_local.json',{'run':run,'coverage':{'completed':len(p),'expected':len(rows)},'summary':summary,'raw':ident(path),'elapsed_s':time.monotonic()-s0})
        print(json.dumps(clean_json({'clean_complete':run,'summary':summary})),flush=True)
        if group=='q1':
            own_diagnostic(D,model,mano,cfg,bank,protocol,device,budget)
        del model,mano;torch.cuda.empty_cache()
    write_json(OUT/f'worker_{group}_receipt.json',{'status':'complete','runs':runs,'physical_gpu':visible,'elapsed_seconds':time.monotonic()-started,'allocated_gpu_hours':(time.monotonic()-started)/3600,'protocol_sha256':protocolsha,'source':ident(Path(__file__))})

def own_diagnostic(D,model,mano,cfg,bank,protocol,device,budget):
    motion=load_raw(ROOT/'outputs/xdl/diagnostics/xdl_q1_s3407/motion.pt.gz')['data']['rows']
    oldbank=load_raw(ROOT/'outputs/xdl/diagnostics/xdl_q1_s3407/bank.pt.gz')['data']['rows']
    mean=mano.hands_mean.cpu().numpy()
    for seq in SEQS:
        data=D.load_data(cfg,seq,bank['sequences'][seq]);beta,camera=D.context(data,device);model.set_hand_context(beta,camera)
        look={r['run_id']:r for r in motion if r['seq']==seq and r['spec']['block']=='clean'}
        pairs=[];trajectories=[];control=[]
        def predict(ev,state,row):
            budget();D.reset(model);p,a=D.predict(model,ev,np.asarray(state,np.float32)[None],beta,camera,device,row,1)
            return p[0],a
        for clip in protocol['own_selection']['selected'][seq]:
            rows=[bank['sequences'][seq]['rows'][i] for i in clip['end_indices']]
            source=look[clip['run_id']];offset=clip['offset'];initial=np.asarray(clip['initial'],np.float32)
            trajectories_states={'real':[initial.copy()],'shuffle':[initial.copy()],'hold':[initial.copy()]}
            for j,row in enumerate(rows):
                ev=D.raw_events(data,row);wrong,audit=transform(ev,row['packet_id']);factual=np.asarray(source['predictions'][offset+j],np.float32)
                real,ra=predict(ev,factual,row);shuffled,wa=predict(wrong,factual,row)
                pairs.append({'clip_id':clip['clip_id'],'row':row,'initial':factual,'truth':data['gt'][row['end_ms']],
                              'real':real,'shuffle':shuffled,'hold':factual.copy(),'transform':audit,
                              'old_real_output':source['predictions'][offset+j+1],'replay_max_abs_state_difference':float(np.max(np.abs(real-source['predictions'][offset+j+1]))),
                              'real_audit':ra,'shuffle_audit':wa})
                # Separate recursively updated branches; no truth or factual resets.
                for mode,events in [('real',ev),('shuffle',wrong)]:
                    previous=trajectories_states[mode][-1]
                    if np.array_equal(previous,factual):p=real if mode=='real' else shuffled
                    else:p,_=predict(events,previous,row)
                    trajectories_states[mode].append(p)
                trajectories_states['hold'].append(initial.copy())
            times=np.asarray([clip['initial_time_ms']]+[r['end_ms'] for r in rows]);truth=data['gt'][times]
            specs=[{'block':x,'id':x} for x in ['real','shuffle','hold']]
            pp=np.stack([trajectories_states[x] for x in ['real','shuffle','hold']])
            scored=D.score_trajectories(mano,pp,truth,times,specs,np.stack([initial]*3),beta,device,mean)
            trajectories.extend({'clip_id':clip['clip_id'],'seq':seq,'initial_z_error_mm':clip['initial_z_error_mm'],**r} for r in scored)
            print(f'own {seq} {clip["clip_id"]}:20 completed',flush=True)
        # The frozen 24 bank packets add signed-z cases absent from own-local errors.
        for row in bank['sequences'][seq]['bank']:
            ev=D.raw_events(data,row);wrong,audit=transform(ev,row['packet_id'])
            wanted=[r for r in oldbank if r['seq']==seq and r['packet_id']==row['packet_id'] and
                    (r['spec']['block']=='clean' or r['spec']['block']=='translation' and r['spec']['axis']==2)]
            assert len(wanted)==5
            for r in wanted:
                p,a=predict(wrong,r['initial'],row)
                record={'row':row,'spec':r['spec'],'initial':r['initial'],'truth':data['gt'][row['end_ms']],
                        'real':r['prediction'],'shuffle':p,'hold':r['initial'],'transform':audit,'audit':a}
                if r['spec']['block']=='clean':
                    rp,_=predict(ev,r['initial'],row);record['real_replay_max_abs_state_difference']=float(np.max(np.abs(rp-r['prediction'])))
                control.append(record)
        path=OUT/f'q1_own_{seq}.pt.gz';save_raw(path,{'protocol_sha256':sha(OUT/'protocol.json'),'pairs':pairs,'trajectories':trajectories,'bank_time_controls':control})
        def summarize_pairs(rr):
            out={}
            ini=np.stack([x['initial'] for x in rr]);gt=np.stack([x['truth'] for x in rr])
            for mode in ['real','shuffle']:
                pp=np.stack([x[mode] for x in rr]);out[mode]=error_summary(pp,ini,gt)
                ep,_,_=D.endpoint(mano,pp,gt,beta,device,mean);out[mode].update({k:stats(v) for k,v in ep.items() if k!='wrist_mm'})
            er=np.linalg.norm((np.stack([x['real'][:3] for x in rr])-gt[:,:3]).astype(np.float64),axis=1)*1000
            ew=np.linalg.norm((np.stack([x['shuffle'][:3] for x in rr])-gt[:,:3]).astype(np.float64),axis=1)*1000
            out['wrong_minus_real_wrist_mm']=stats(ew-er);out['n_real_better_than_shuffle']=int(np.sum(er<ew))
            out['transform']={k:stats([x['transform'][k] if x['transform'][k] is not None else np.nan for x in rr]) for k in ['timestamp_changed_fraction','mean_abs_timestamp_shift_ms','original_reassigned_time_correlation','stride_sample_original_identity_jaccard']}
            return out
        ts={}
        for mode in ['real','shuffle','hold']:
            rr=[r for r in trajectories if r['spec']['block']==mode]
            ts[mode]={'n_clips':len(rr),'auc':{k:stats([r['auc'][k] for r in rr]) for k in rr[0]['auc']},
                      'initially_satisfied':sum(r['recovery']['initially_satisfied'] for r in rr),
                      'new_sustained_recovery':sum(not r['recovery']['initially_satisfied'] and r['recovery']['hit_step'] is not None for r in rr)}
        write_json(OUT/f'q1_own_{seq}.json',{'seq':seq,'paired':summarize_pairs(pairs),'trajectory':ts,
                   'bank_time_controls':{s:summarize_pairs([r for r in control if r['spec']['id']==s]) for s in sorted({r['spec']['id'] for r in control})},
                   'max_factual_replay_state_difference':max(r['replay_max_abs_state_difference'] for r in pairs),'raw':ident(path)})

def cpu_reuse():
    torch.set_num_threads(2);bank=json.loads((ROOT/'outputs/xdl/packet_bank.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    gt={s:meta51(gtpath(bank,s)) for s in bank['sequences']};all_records=[];summaries={}
    for run in RUNS:
        raw=load_raw(ROOT/f'outputs/xdl/diagnostics/{run}/bank.pt.gz');records=[]
        for r in raw['data']['rows']:
            if r['spec']['block']!='clean':continue
            seq=r['seq'];meta=bank['sequences'][seq];end=r['clock']['end_ms'];previous=r['clock']['previous_end_ms']
            source=next(x for x in meta['bank'] if x['packet_id']==r['packet_id']);age=previous-meta['valid_runs_ms'][source['run_id']][0]
            target=gt[seq][end];assert np.array_equal(r['initial'],gt[seq][previous])
            item={'run':run,'seq':seq,'cohort':r['cohort'],'packet_id':r['packet_id'],'initial':r['initial'],'prediction':r['prediction'],'truth':target,
                  'depth_mm':float(gt[seq][previous,2]*1000),'event_count':r['n_raw_events'],
                  'true_wrist_motion_mm':float(np.linalg.norm(target[:3].astype(np.float64)-r['initial'][:3])*1000),'run_age_ms':age}
            records.append(item)
        summary={}
        for cohort in ['seen','zgz_local','zgz_global']:
            rr=[r for r in records if r['cohort']=='seen'] if cohort=='seen' else [r for r in records if r['seq']==cohort]
            if rr:summary[cohort]=error_summary(np.stack([r['prediction'] for r in rr]),np.stack([r['initial'] for r in rr]),np.stack([r['truth'] for r in rr]))
            for dimension,edges in protocol['cpu_strata'].items():
                for b in range(len(edges)+1):
                    group=[r for r in rr if int(np.digitize(r[dimension],edges))==b]
                    summary[f'{cohort}|{dimension}|bin{b}']={'n':len(group),'edges':edges}
                    if group:summary[f'{cohort}|{dimension}|bin{b}'].update(error_summary(np.stack([r['prediction'] for r in group]),np.stack([r['initial'] for r in group]),np.stack([r['truth'] for r in group])))
        summaries[run]=summary;all_records.extend(records)
    write_json(OUT/'existing_bank_support.json',{'protocol_sha256':sha(OUT/'protocol.json'),'summary':summaries,'clean_packets':all_records,'scope':'Saved bank outputs only: 72 seen onepacket/sequence plus24 each zgz; all bins including empty retained. No new inference or support-extrapolation causal claim.'})
    print('CPU existing bank support complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','cpu']);p.add_argument('--group',choices=['q1','other']);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='cpu':cpu_reuse()
    else:worker(a.group)
