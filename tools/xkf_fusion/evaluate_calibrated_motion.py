#!/usr/bin/env python3
"""Actual uncached full-forward112 loops for a training-only fitted scalar I ramp.

Both fixed1000 and training-fitted policies own their complete nine-condition
trajectories. Canonical per-sample full forward is mandatory. No dev fitting.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'model')]
import numpy as np,torch
from semkine.xkf_calibrated_motion_gate import XKFCalibratedMotionModel,CalibratedMotionCountPolicy
from tools.xkf_fusion import evaluate_tuned as Q
from tools.xkf_fusion.bn_probe import parameter_digest
from tools.xkf_fusion.train_calibration_tuned import validate_clocks

BASE_POLICY=dict(gain_scale=[.5,.25,.5],count_gate='ramp1000',translation_reference='joint_centroid',motion_count_gate='ramp1000')

def load_runtime(path,seed,checkpoint_sha,canonical_split=None):
    value=json.loads(Path(path).read_text())
    allowed={'kind','threshold','seed','checkpoint_sha256','fit_path','fit_sha256','prereg_path','prereg_sha256','training_source_sha256','selection_family','accepted_for_dev'}
    if set(value)!=allowed or value['kind']!='motion_count_ramp' or value['seed']!=seed or value['checkpoint_sha256']!=checkpoint_sha:
        raise ValueError('runtime schema/seed/checkpoint identity differs')
    threshold=value['threshold']
    if isinstance(threshold,bool) or not isinstance(threshold,(float,int)) or not math.isfinite(threshold) or threshold<=0:
        raise ValueError('invalid fitted threshold')
    if value['accepted_for_dev'] is not True or value['selection_family'] not in ('shared','per_seed'):
        raise ValueError('runtime did not pass training-only held-out acceptance')
    inputs={str(Path(path).resolve()):Q.E.sha(path)}
    for key in ('fit','prereg'):
        p=Path(value[key+'_path'])
        if not p.is_absolute() or Q.E.sha(p)!=value[key+'_sha256']:raise ValueError('runtime audit file changed: '+key)
        inputs[str(p)]=value[key+'_sha256']
    if not isinstance(value['training_source_sha256'],dict) or len(value['training_source_sha256'])<3:
        raise ValueError('training source manifest missing')
    for name,digest in value['training_source_sha256'].items():
        p=Path(name)
        if not p.is_absolute() or Q.E.sha(p)!=digest:raise ValueError('training source changed: '+name)
        inputs[name]=digest
    fit=json.loads(Path(value['fit_path']).read_text())
    if (fit.get('kind')!='TRAIN_ONLY_I_RAMP_FIT' or fit.get('accepted_for_dev') is not True
            or fit.get('status')!='ACCEPTED_FOR_DEV_EVALUATION'
            or fit.get('selected_family')!=value['selection_family']
            or fit.get('thresholds',{}).get(str(seed))!=threshold
            or fit.get('training_source_sha256')!=value['training_source_sha256']
            or fit.get('prereg_sha256')!=value['prereg_sha256']
            or fit.get('metrics',{}).get(value['selection_family'],{}).get('eligible') is not True):
        raise ValueError('runtime contradicts actual accepted fit evidence')
    if set(fit['thresholds'])!={'3407','3408','3409'}:
        raise ValueError('fit must preserve every seed')
    if value['selection_family']=='shared' and len(set(fit['thresholds'].values()))!=1:
        raise ValueError('shared fit has inconsistent thresholds')
    prereg=json.loads(Path(value['prereg_path']).read_text())
    if prereg.get('read_development_data') is not False or prereg.get('rows_per_seed')!=2304 or prereg.get('conditions')!=10:
        raise ValueError('fit pre-registration differs from training-only full2304x10')
    for name,digest in {**fit.get('source_sha256',{}),**fit.get('oof_sha256',{})}.items():
        p=Path(name);p=p if p.is_absolute() else ROOT/p
        if Q.E.sha(p)!=digest:raise ValueError('fitted source or held-out evidence changed')
        inputs[str(p.resolve())]=digest
    protocols={}
    for name in value['training_source_sha256']:
        p=Path(name)
        if p.name!='protocol.json':continue
        doc=json.loads(p.read_text());identity=doc.get('identity',{});s=identity.get('seed')
        if (doc.get('kind')!='CANONICAL_TRAIN_I_MOTION_DIAGNOSIS' or doc.get('partial_smoke') is not False
                or doc.get('sample_count')!=2304 or len(doc.get('conditions',[]))!=10 or identity.get('global_step')!=2400):
            raise ValueError('fit used an incomplete or wrong training probe')
        seqs=doc.get('canonical_train_sequences',[])
        subjects={x.split('_')[0] for x in seqs}
        if len(seqs)!=72 or subjects!={'ch','lfz','lpc','lr','ly','lyh','lyq','ycy','ylf'}:
            raise ValueError('fit used noncanonical training sequences')
        splits=[Path(n) for n in doc.get('input_sha256',{}) if Path(n).name=='splits_semkine.json']
        if len(splits)!=1 or (canonical_split is not None and splits[0].resolve()!=Path(canonical_split).resolve()):
            raise ValueError('fit canonical split identity missing or changed')
        for n,d in doc['input_sha256'].items():
            if Q.E.sha(n)!=d:raise ValueError('training probe input changed')
            inputs[n]=d
        if s in protocols:raise ValueError('duplicate seed protocol')
        protocols[s]=doc
    if set(protocols)!={3407,3408,3409} or protocols[seed]['identity']['checkpoint_sha256']!=checkpoint_sha:
        raise ValueError('fit checkpoint or three-seed training identity differs')
    return value,inputs

def sources():
    return Q.sources()|{p:Q.E.sha(ROOT/p) for p in ('semkine/xkf_centroid.py','semkine/xkf_motion_gate.py',
        'semkine/xkf_stable_geometry.py','semkine/xkf_motion_gate_runtime.py','semkine/xkf_calibrated_motion_gate.py',
        'tools/xkf_fusion/evaluate_calibrated_motion.py','tools/xkf_fusion/bn_probe.py','tools/xkf_fusion/train_calibration_tuned.py')}

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','checkpoint','runtime','output'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--batch-size',type=int,choices=(1,9),default=9)
    parser.add_argument('--limit-clips',type=int,default=0);parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.limit_clips<0:parser.error('limit-clips must be nonnegative')
    if args.output.exists():raise FileExistsError(args.output)
    Q.RE.setup();Q.E.XKFModel=XKFCalibratedMotionModel;torch.set_num_threads(4)
    device=torch.device(args.device);model,cfg,identity=Q.E.load_model(args.config,args.checkpoint,device)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False);identity['training_clocks']=validate_clocks(model,saved,False);del saved
    if identity['global_step']!=2400:raise ValueError('exact raw2400 required')
    runtime,input_hashes=load_runtime(args.runtime,identity['seed'],identity['checkpoint_sha256'],Path(cfg['DATA']['ROOT'])/'splits_semkine.json')
    model.configure_stability('per_sample_model');model.kmode='fused'
    for module in model.modules():
        if isinstance(module,Q.ManoLayer):module.fast_chain=True
    policies=[dict(BASE_POLICY,id='fixed1000',motion_count_threshold=1000.),dict(BASE_POLICY,id='train_fitted',motion_count_threshold=runtime['threshold'])]
    for p in policies:CalibratedMotionCountPolicy(p)
    bank,banksha=Q.R.D.load_bank();clips,excluded=Q.R.schedule(bank,10)
    if len(clips)!=112 or sum(c['observed_steps'] for c in clips)!=902:raise ValueError('canonical full112/902frame schedule changed')
    selected=clips[:args.limit_clips] if args.limit_clips else clips
    before=sources();parameters=parameter_digest(model);buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
    modes=('fixed1000','train_fitted','hold');args.output.mkdir(parents=True);(args.output/'raw').mkdir()
    protocol=dict(kind='XKF_TRAIN_CALIBRATED_I_RAMP_ACTUAL_FORWARD',formal=False,automatic_full_training=False,
        identity=identity,source_sha256=before,runtime=runtime,input_sha256=input_hashes,bank_sha256=banksha,
        policies=policies,clips=selected,excluded=excluded,conditions=Q.R.CONDITIONS,metrics=Q.R.METRICS,modes=modes,
        partial_smoke=bool(args.limit_clips),canonical_clips=112,stability_mode='per_sample_model',outer_batch_size=args.batch_size,
        direct_cache=False,inference='Actual full forward_packet per sample including D; only raw events and LNES cached. All targets zero; GT only initialization/scoring.',
        numerical_reference='fixed1000 takes exact existing MotionCountFusionPolicy branch; no integer rounding or extra root clipping',
        raw_trace_contract='motion_predictions=(frames,9,2,51) before I gate; gated_motion_predictions same shape; corrected_direct=(frames,9,2,51); gains=(frames,9,2,19); motion_gate=(frames,9,2)')
    Q.write_json(args.output/'protocol.json',protocol)
    if args.prepare_only:return
    all_records=[];caches={};started=time.perf_counter();raw_hashes={};actual_calls=0
    for index,clip in enumerate(selected):
        sequence=clip['sequence']
        if sequence not in caches:
            data=Q.R.D.load_data(cfg,sequence,bank['sequences'][sequence]);caches[sequence]=Q.PacketCache(model,data,device)
        cache=caches[sequence];clean=torch.from_numpy(cache.data['gt'][clip['initial_gt_ms']]).to(device).float()
        initial=Q.R.initial_states(clean);previous=initial[:,None].expand(-1,2,-1).clone()
        collected={k:[] for k in ('predictions','truth','metrics','physical_wrist_error_xyz_mm','state_t_error_xyz_mm','motion_predictions','gated_motion_predictions','corrected_direct','effective_gains','motion_gate','lnes_window_ms','n_raw_events','packet_ids','end_ms')}
        Q.R.D.reset(model)
        for rowindex in clip['row_indices']:
            row=bank['sequences'][sequence]['rows'][rowindex];outputs=[];traces=[]
            for ci,policy in enumerate(policies):
                model.configure_fusion(policy);model.fusion_policy.capture=True;values=[]
                for start in range(0,9,args.batch_size):
                    packet,measurements=cache.packet(row,previous[start:start+args.batch_size,ci],[])
                    if measurements or bool(packet.target.any()):raise ValueError('D cache or nonzero target')
                    Q.R.D.reset(model);values.append(model.forward_packet(packet));actual_calls+=packet.batch_size
                outputs.append(torch.cat(values));trace=model.fusion_policy.trace
                if len(trace)!=9 or any(v['motion'].shape!=(1,51) for v in trace):raise ValueError('full forward was not one genuine sample per call')
                traces.append({key:torch.cat([v[key] for v in trace]) for key in trace[0]})
            previous=torch.stack(outputs,1);states=torch.cat((previous,initial[:,None]),1)
            truth=torch.from_numpy(cache.data['gt'][row['end_ms']]).to(device).float().unsqueeze(0)
            metric,wrist,state_t=Q.R.frame_metrics(model,states.reshape(-1,51),truth,cache.beta)
            if not bool(torch.isfinite(states).all() and torch.isfinite(metric).all()):raise FloatingPointError('nonfinite full trajectory')
            collected['predictions'].append(states.cpu().numpy());collected['truth'].append(truth[0].cpu().numpy())
            for key,value in [('metrics',metric.reshape(9,3,-1)),('physical_wrist_error_xyz_mm',wrist.reshape(9,3,3)),('state_t_error_xyz_mm',state_t.reshape(9,3,3))]:collected[key].append(value.cpu().numpy())
            for key,tracekey in [('motion_predictions','motion'),('gated_motion_predictions','gated_motion'),('corrected_direct','direct'),('effective_gains','effective_gains'),('motion_gate','motion_gate')]:collected[key].append(torch.stack([v[tracekey] for v in traces],1).cpu().numpy())
            collected['lnes_window_ms'].append(int(packet.lnes_window_ms[0]));collected['n_raw_events'].append(int(packet.counts[0]));collected['packet_ids'].append(row['packet_id']);collected['end_ms'].append(row['end_ms'])
        Q.R.D.reset(model)
        if cache.measurements:raise ValueError('D cache unexpectedly populated')
        raw={key:np.stack(value) for key,value in collected.items()};raw.update(initial=initial.cpu().numpy(),clean=clean.cpu().numpy())
        path=args.output/'raw'/f'{index:03d}_{clip["cohort"]}_run{clip["run_id"]}.npz';np.savez_compressed(path,**raw);raw_hashes[path.name]=Q.E.sha(path)
        all_records.extend(Q.per_clip_rows(clip,raw,modes))
        Q.write_json(args.output/'progress.json',dict(completed_clips=index+1,total_clips=len(selected),elapsed_seconds=time.perf_counter()-started))
        if index%8==0:print(json.dumps(dict(seed=identity['seed'],completed_clips=index+1)),flush=True)
    if sources()!=before or parameter_digest(model)!=parameters:raise ValueError('source/parameter mutation')
    for name,value in model.named_buffers():
        if not torch.equal(value.cpu(),buffers[name]):raise ValueError('buffer mutation '+name)
    for name,digest in input_hashes.items():
        if Q.E.sha(name)!=digest:raise ValueError('fit/source input changed '+name)
    if Q.E.sha(args.config)!=identity['config_sha256'] or Q.E.sha(args.checkpoint)!=identity['checkpoint_sha256']:raise ValueError('model input changed')
    with (args.output/'per_clip.jsonl').open('w') as stream:
        for row in all_records:stream.write(json.dumps(Q.E.safe_json(row),allow_nan=False)+'\n')
    if actual_calls!=sum(c['observed_steps'] for c in selected)*18 or len(raw_hashes)!=len(selected):
        raise ValueError('missing actual forward calls or raw clips')
    Q.write_json(args.output/'summary.json',dict(kind=protocol['kind'],formal=False,automatic_full_training=False,identity=identity,
        execution_complete=True,canonical_schedule_clip_count=112,canonical_observed_frames=902,
        partial_smoke=bool(args.limit_clips),clip_count=len(selected),modes=modes,metrics=Q.R.METRICS,aggregate=Q.R.aggregate(all_records),
        raw_sha256=raw_hashes,source_sha256=before,protocol_sha256=Q.E.sha(args.output/'protocol.json'),input_sha256=input_hashes,
        actual_uncached_single_sample_forward_calls=actual_calls,all_parameters_unchanged_sha256=parameters,all_buffers_unchanged=True,
        elapsed_seconds=time.perf_counter()-started))
    print(json.dumps(dict(output=str(args.output),clips=len(selected))),flush=True)
if __name__=='__main__':main()
