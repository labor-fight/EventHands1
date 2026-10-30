#!/usr/bin/env python3
"""XDF fixed-candidate, three-pass full software feedback timing."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'model'))
from tools.xdf.angle_probe import sha, write
OUT = ROOT/'outputs/xdf/pipeline'
HEADS = {
    'F_s1701': ROOT/'outputs/xdf/interface/heads/F_s1701.pt',
    'L0_s1701': ROOT/'outputs/xdf/objective/heads/z1_s1701_r0.pt',
    'L1_s1701': ROOT/'outputs/xdf/objective/heads/z2_s1701_r0.pt',
}
MODES = ['Q1', 'F_s1701', 'L0_s1701', 'L1_s1701']


def stats(a):
    a=np.asarray(a,dtype=np.float64)
    assert len(a) and np.isfinite(a).all()
    return {'n':len(a),'mean':float(a.mean()), **{f'p{q}':float(np.percentile(a,q)) for q in (50,95,99)}}


def main():
    import torch
    from tools.xdl import diagnostics as D
    from mano_layer import ManoLayer
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','run'],default='run');args=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='4':raise RuntimeError('physical GPU4 allocated only')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    bank,banksha=D.load_bank();names=['zgz_global','zgz_local']
    protocol={'study':'XDF-three-pass-software-feedback-v1','modes':MODES,'physical_gpu':4,
        'names':names,'passes':3,'order':'cyclic mode rotation by pass index; global then local within mode',
        'batch':1,'threads':4,'window_ms':50,'warmup':20,'query_bank_sha256':banksha,
        'expected_queries_per_mode_pass':2590,'initial':'GT only at each valid run start, own thereafter',
        'scope':'events in host memory, slice, H2D, FK/association/forward and intervention, output MANO, host-ready state/mesh/joints, feedback clone; synchronization overhead included',
        'excludes':['sequence file open','sensor acquisition','real-time queue','display','valid-run GT initialization'],
        'F_implementation':'original Q1 forward including its head, plus normalized extra 707-to-256-to-3 head replacing translation; cost is actual diagnostic composition',
        'objective_implementation':'replace original fc1/fc2; no extra head',
        'head_paths':{k:str(v.relative_to(ROOT)) for k,v in HEADS.items()},
        'selection':'seed1701 non-repeat fixed in advance, no best-head selection',
        'acceptance':'p95 <=1.05 same-pass Q1 in both sequences, all three passes; candidate OR Q1 p95 repeat range / mean >.05 makes relative timing undetermined; all Q1 and candidate denominators must be complete',
        'preexecution_amendment':'v2 propagates Q1 repeat instability or incomplete denominators to every relative timing gate; v1 retained, no timing had run',
        'nonfinite':'preserve failure and entire failed denominator, no finite-only PASS',
        'budget_gpu_hours':.5,'script_sha256':sha(__file__)}
    p=OUT/'protocol_v2.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:write(p,protocol)
    if args.phase=='prepare':print(json.dumps({'protocol_sha256':sha(p)}));return
    started=time.perf_counter();dev=torch.device('cuda:0')
    for h in HEADS.values():assert h.exists(),str(h)
    write(OUT/'input_manifest.json',{'protocol_sha256':sha(p),'heads':{k:sha(v) for k,v in HEADS.items()},'script_sha256':sha(__file__)})
    model,cfg,identity=D.load_model(str(ROOT/'outputs/semkine/xdl_q1_s3407'),dev)
    mano=ManoLayer(cfg['MANO']['NPZ'],add_mean=False).to(dev).eval();mano.fast_chain=True
    original=[{k:v.clone() for k,v in layer.state_dict().items()} for layer in (model.xyz.fc1,model.xyz.fc2)]
    saved={k:torch.load(v,map_location='cpu',weights_only=False) for k,v in HEADS.items()}
    from tools.xdf.interface_probe import make_head
    fh=make_head().to(dev).eval();fh.load_state_dict(saved['F_s1701']['state_dict'])
    center=torch.as_tensor(saved['F_s1701']['mean'],device=dev);scale=torch.as_tensor(saved['F_s1701']['std'],device=dev)
    seqdata={name:D.load_data(cfg,name,bank['sequences'][name]) for name in names}
    records=[];failures=[];calls=0
    with torch.no_grad():
        for repeat in range(3):
            order=MODES[repeat:]+MODES[:repeat]
            for mode in order:
                for j,layer in enumerate((model.xyz.fc1,model.xyz.fc2)):
                    sd=original[j] if mode in ('Q1','F_s1701') else {k:saved[mode]['state_dict'][f'{2*j}.{k}'].to(dev) for k in ('weight','bias')}
                    layer.load_state_dict(sd)
                cap={};hook=None
                if mode=='F_s1701':hook=model.xyz.fc1.register_forward_pre_hook(lambda _m,inp:cap.update(full=inp[0]))
                def forward(packet):
                    pred=model.forward_packet(packet)
                    if mode=='F_s1701':
                        pred=pred.clone();pred[:,:3]=packet.prev_state[:,:3]+fh((cap['full']-center)/scale)*.05
                    return pred
                for name in names:
                    data=seqdata[name];beta,K=D.context(data,dev);model.set_hand_context(beta,K);D.reset(model)
                    rows=bank['sequences'][name]['rows'];trace=[];predictions=[];truths=[];state=None;previous_run=None
                    for row in rows[:20]:
                        packet=D.make_packet(D.raw_events(data,row),data['gt'][row['previous_end_ms']][None],beta,K,dev,row)
                        y=forward(packet);dec=D.ET.decode_to_mano_inputs(y,'mano_full_axis_angle',mano.hands_components,mano.hands_mean)
                        mano(beta,dec['global_orient'],dec['local_full_aa'],dec['transl']);calls+=1
                    torch.cuda.synchronize();D.reset(model);torch.cuda.reset_peak_memory_stats(dev)
                    failure=None
                    for index,row in enumerate(rows):
                        if row['run_id']!=previous_run:
                            D.reset(model);state=torch.tensor(data['gt'][row['previous_end_ms']],device=dev).view(1,51);previous_run=row['run_id']
                        torch.cuda.synchronize();t0=time.perf_counter()
                        ev=D.raw_events(data,row);t1=time.perf_counter()
                        packet=D.make_packet(ev,np.zeros((1,51),np.float32),beta,K,dev,row);packet.prev_state=state
                        torch.cuda.synchronize();t2=time.perf_counter()
                        pred=forward(packet);torch.cuda.synchronize();t3=time.perf_counter();calls+=1
                        dec=D.ET.decode_to_mano_inputs(pred,'mano_full_axis_angle',mano.hands_components,mano.hands_mean)
                        verts,joints=mano(beta,dec['global_orient'],dec['local_full_aa'],dec['transl']);torch.cuda.synchronize();t4=time.perf_counter()
                        host=pred.cpu().numpy().copy();hv=verts.cpu().numpy().copy();hj=joints.cpu().numpy().copy();state=pred.detach().clone()
                        torch.cuda.synchronize();t5=time.perf_counter()
                        if not all(np.isfinite(a).all() for a in (host,hv,hj)):
                            failure={'mode':mode,'repeat':repeat,'sequence':name,'index':index,'expected':len(rows),'reason':'nonfinite full own loop'};failures.append(failure);break
                        trace.append([(t5-t0)*1000,(t1-t0)*1000,(t2-t1)*1000,(t3-t2)*1000,(t4-t3)*1000,(t5-t4)*1000,len(ev),row['end_ms']])
                        predictions.append(host[0]);truths.append(data['gt'][row['end_ms']])
                        if time.perf_counter()-started>1800:raise TimeoutError('shared pipeline .5GPUh limit')
                    key=f'{mode}_r{repeat}_{name}';trace=np.asarray(trace)
                    rec={'mode':mode,'repeat':repeat,'sequence':name,'expected':len(rows),'completed':len(trace),'failure':failure,'full_denominator':failure is None,
                         'peak_allocated_bytes':torch.cuda.max_memory_allocated(dev)}
                    if len(trace):
                        rec['timings']={k:stats(trace[:,j]) for j,k in enumerate(['total_ms','slice_ms','packet_ms','forward_ms','output_MANO_ms','host_feedback_ms','n_events'])}
                        rec['queries_per_compute_second']=1000/rec['timings']['total_ms']['mean']
                        rec['event_age_contract_ms']={'window':50,'max_age_at_start_upper_bound':50,'max_age_after_compute_upper_bound_p95':50+rec['timings']['total_ms']['p95'],'measured_sensor_age':None}
                        if failure is None:
                            metrics,_,_=D.endpoint(mano,np.stack(predictions),np.stack(truths),beta,dev,mano.hands_mean.cpu().numpy())
                            rec['endpoint']={k:stats(v) for k,v in metrics.items()}
                        np.savez_compressed(OUT/f'{key}.npz',trace=trace,prediction51=np.asarray(predictions),target51=np.asarray(truths))
                    records.append(rec);write(OUT/f'{key}.json',rec)
                    print(json.dumps({'mode':mode,'repeat':repeat,'sequence':name,'completed':len(trace),'p95_ms':rec.get('timings',{}).get('total_ms',{}).get('p95'),'failure':failure}),flush=True)
                if hook is not None:hook.remove()
    gates=[]
    for name in names:
        baselines=[r for r in records if r['mode']=='Q1' and r['sequence']==name]
        base_complete=len(baselines)==3 and all(r['full_denominator'] for r in baselines)
        base_vals=[r['timings']['total_ms']['p95'] for r in baselines if r['full_denominator']]
        base_spread=None if not base_complete else (max(base_vals)-min(base_vals))/np.mean(base_vals)
        for mode in MODES:
            these=[r for r in records if r['mode']==mode and r['sequence']==name]
            vals=[r['timings']['total_ms']['p95'] for r in these if r['full_denominator']]
            spread=None if len(vals)!=3 else (max(vals)-min(vals))/np.mean(vals)
            ratios=[] if not base_complete else [r['timings']['total_ms']['p95']/next(q['timings']['total_ms']['p95'] for q in baselines if q['repeat']==r['repeat']) for r in these if r['full_denominator']]
            gates.append({'sequence':name,'mode':mode,'p95_repeat_relative_range':None if spread is None else float(spread),'Q1_p95_repeat_relative_range':None if base_spread is None else float(base_spread),'ratios_to_same_pass_Q1':ratios,
                          'decision':'incomplete' if len(vals)!=3 or not base_complete else 'undetermined_repeat_variation' if spread>.05 or base_spread>.05 else 'pass' if all(v<=1.05 for v in ratios) else 'fail'})
    write(OUT/'result.json',{'status':'complete','protocol_sha256':sha(p),'identity':identity,'calls_including_warmup':calls,'records':records,'gates':gates,'failures':failures,
        'wall_seconds':time.perf_counter()-started,'physical_gpu':4,'gpu_hours':(time.perf_counter()-started)/3600,'timing_exclusions':protocol['excludes']})


if __name__=='__main__':main()
