#!/usr/bin/env python3
"""Freeze raw W50 packet/20-step clip IDs from CPU metadata, before scoring."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'model'))
from tools.xdl import geometry as G


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def midpoint_indices(count,requested):
    if count<requested:raise ValueError(f'insufficient legal windows: {count} < {requested}')
    return [int(np.floor((j+.5)*count/requested)) for j in range(requested)]


def packet_rows(valid_runs):
    rows=[]
    for rid,(a,b) in enumerate(np.asarray(valid_runs,np.int64).reshape(-1,2)):
        prev=int(a)
        for end in np.arange(a+49,b,50,dtype=np.int64):
            rows.append({'run_id':rid,'end_ms':int(end),'previous_end_ms':prev,
                         'event_start_ms':int(end)-49,'state_interval_ms':int(end)-prev,
                         'query_start_offset_ms':int(end)-49-prev})
            prev=int(end)
    return rows


def clip_candidates(rows,steps=20):
    candidates=[]
    for rid in sorted({r['run_id'] for r in rows}):
        indices=[i for i,r in enumerate(rows) if r['run_id']==rid]
        for offset in range(0,len(indices)-steps+1,steps):
            chosen=indices[offset:offset+steps]
            if any(rows[j]['previous_end_ms']!=rows[i]['end_ms'] for i,j in zip(chosen,chosen[1:])):raise ValueError('discontinuous clip clock')
            candidates.append({'run_id':rid,'start_index':chosen[0],'end_indices':chosen})
    return candidates


def prepare(cfg_path,out=REPO/'outputs/xdl',workers=24):
    from config import load_config
    from semkine.dataset import sequences_for_split,splits_manifest
    from semkine import eval_track as ET
    cfg=load_config(str(cfg_path));root=Path(cfg['DATA']['ROOT'])
    split=splits_manifest(cfg) or root/'splits_semkine.json'
    pairs=[('seen',i,*s) for i,s in enumerate(sequences_for_split(root,'train',split))]+[('zgz',None,*s) for s in sequences_for_split(root,'val_core',split)]
    if len(pairs)!=74:raise ValueError('requires 72 seen and 2 zgz sequences')
    def one(item):
        cohort,ordinal,seq,directory=item;base=root/directory/seq
        aux=np.load(str(base)+'_aux.npz',allow_pickle=True);offsets=np.load(str(base)+'_offsets.npy',mmap_mode='r')
        runs=np.asarray(aux['valid_runs_ms'],np.int64).reshape(-1,2);rows=packet_rows(runs)
        for i,row in enumerate(rows):
            row.update(index=i,packet_id=f"{seq}:{row['run_id']}:{row['end_ms']}",n_raw_events=int(offsets[row['end_ms']+1]-offsets[row['event_start_ms']]))
        banks=[rows[i] for i in midpoint_indices(len(rows),24 if cohort=='zgz' else 1)]
        candidates=clip_candidates(rows,20);clips=[candidates[i] for i in midpoint_indices(len(candidates),24 if cohort=='zgz' else 1)]
        for clip in clips:
            clip.update(clip_id=f"{seq}:{clip['run_id']}:{rows[clip['start_index']]['end_ms']}")
        short=[{'run_id':rid,'end_indices':[i for i,r in enumerate(rows) if r['run_id']==rid]} for rid in range(len(runs))]
        short=[r for r in short if 0<len(r['end_indices'])<20]
        # Read only selected E to freeze its exact content/support identity.
        events=np.load(str(base)+'_events.npy',mmap_mode='r');tp=Path(str(base)+'_tsub.npy');tsub=np.load(tp,mmap_mode='r') if tp.exists() else None
        for row in banks:
            ev=ET._window_events(events,offsets,tsub,row['end_ms'],50)
            row['events_sha256']=hashlib.sha256(ev.tobytes()).hexdigest()
            specs=G.probe_specs(row['packet_id'],ordinal if cohort=='seen' else None)
            row.update(specs=specs,control_specs=G.control_specs(specs,row['packet_id']) if cohort=='zgz' else [])
        paths=[Path(str(base)+suffix) for suffix in ('_offsets.npy','_aux.npz','.meta')]
        return seq,{'cohort':cohort,'seen_ordinal':ordinal,'directory':directory,'rows':rows,'bank':banks,'clips':clips,'short_runs':short,
                    'valid_runs_ms':runs.tolist(),'n_legal_clips':len(candidates),'sources':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in paths],
                    'events_file':{'path':str(base)+'_events.npy','bytes':events.nbytes,'shape':list(events.shape)},'tsub_present':tsub is not None}
    with ThreadPoolExecutor(max_workers=workers) as pool:sequences=dict(pool.map(one,pairs))
    target=Path(out)/'packet_bank.json'
    payload={'study':G.STUDY,'window_ms':50,'steps':20,'selection':'chronological midpoint ranks; nonoverlapping 20-packet candidates per valid run',
             'split':{'path':str(split),'sha256':sha(split)},'config':{'path':str(cfg_path),'sha256':sha(cfg_path)},'sequences':sequences,
             'counts':{'packets':sum(len(s['bank']) for s in sequences.values()),'clips':sum(len(s['clips']) for s in sequences.values()),
                       'bank_queries_per_checkpoint':sum(sum(len(r['specs']) for r in s['bank']) for s in sequences.values()),
                       'zgz_packets':sum(len(s['rows']) for s in sequences.values() if s['cohort']=='zgz'),
                       'short_zgz_queries':sum(sum(len(r['end_indices']) for r in s['short_runs']) for s in sequences.values() if s['cohort']=='zgz')}}
    if target.exists():
        old=json.loads(target.read_text())
        if old!=payload:raise RuntimeError('frozen metadata differs; use a new protocol version/output directory')
    else:write_json(target,payload)
    print(json.dumps({'path':str(target),'sha256':sha(target),'counts':payload['counts']},ensure_ascii=False),flush=True)
    return payload

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(REPO/'configs/xa/xa_c0_s3407.yaml'));p.add_argument('--out',default=str(REPO/'outputs/xdl'));p.add_argument('--workers',type=int,default=24)
    a=p.parse_args();prepare(a.config,a.out,a.workers)
