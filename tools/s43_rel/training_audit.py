"""Independent read-only audit of persisted S43 relation training artifacts.

No fitting, evaluation labels, production source edits or optimizer updates.
The optional sampling replay reads only source events and validates saved hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .data import ContinuousRunLanes, FIT7, SELECT
from .model import load_relation_model, TypedSupportAdapter
from .joint_reader import JointGramSupportAdapter
from tools.s43_br.data import DATA_ROOT, PUBLIC_INITIAL_STATE, read_manifest

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/s43_rel_20261027'
ARMS=('s43','enhanced','no_order','full')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n')


def tensor_changes(model,saved):
    initial=dict(model.named_parameters())
    state=saved['model_state_dict'];groups={}
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    param_ids=[p for g in saved['optimizer_state_dict']['param_groups'] for p in g['params']]
    if len(trainable)!=len(param_ids):raise AssertionError('optimizer parameter ordering/count mismatch')
    opt_names=dict(zip([n for n,_ in trainable],param_ids))
    for prefix in ('base.','sensor_adapter.','typed_adapter.direction.','typed_adapter.support.','typed_adapter.output.'):
        changed=total=param_tensors=changed_tensors=momentum_nonzero=0;l2=0.;steps=[]
        for name,p in initial.items():
            if not name.startswith(prefix):continue
            old=p.detach();new=state[name];delta=new-old
            n=int(torch.count_nonzero(delta));changed+=n;total+=delta.numel();param_tensors+=1;changed_tensors+=bool(n)
            l2+=float(delta.double().square().sum())
            if name in opt_names:
                slot=saved['optimizer_state_dict']['state'].get(opt_names[name],{})
                momentum_nonzero+=int(torch.count_nonzero(slot.get('exp_avg',torch.zeros(1))))
                if 'step' in slot:steps.append(int(slot['step']))
        groups[prefix]=dict(parameters=total,changed_parameters=changed,param_tensors=param_tensors,
            changed_tensors=changed_tensors,delta_L2=math.sqrt(l2),Adam_momentum_nonzero=momentum_nonzero,
            actual_Adam_steps=sorted(set(steps)))
    bn=[n for n in model.state_dict() if n.startswith('base.') and ('running_mean' in n or 'running_var' in n or 'num_batches_tracked' in n)]
    groups['BN_buffers']=dict(count=len(bn),all_unchanged=all(torch.equal(model.state_dict()[n],state[n]) for n in bn))
    return groups


def runtime_check(saved,readers):
    runtime,lanes=saved['runtime'],saved['lanes'];active=[]
    for lane in lanes['lanes']:
        sid=lane['seq_id'];run=lane['run'];reader=readers[run.recording]
        expected=int(reader.offsets[lane['cursor_ms']]-reader.offsets[run.start_ms])
        row=dict(seq_id=sid,run_start_ms=run.start_ms,cursor_ms=lane['cursor_ms'],ordinal=lane['ordinal'],
            age_ms=lane['cursor_ms']-run.start_ms,prefix_source_events=expected,
            context_commit_clock_matches=runtime['last_cutoffs'][sid]==lane['cursor_ms'])
        if sid in runtime['memories']:
            mem=runtime['memories'][sid]
            if 'metadata' in mem:
                m=mem['metadata'];observed=int(mem['arrays']['_count'].sum())
                row.update(sensor_total_events=observed,full_prefix_count_exact=observed==expected,
                    sensor_boundary_exact=m['_boundary']==run.start_ms*1000,
                    sensor_watermark_exact=m['_watermark']==lane['cursor_ms']*1000,
                    sensor_source_ordinal_exact=m['_ordinal']==lane['ordinal']-1,
                    memory_commit_count=m['_commit_count'],memory_version=m['_version'])
                if not all(row[k] for k in ('full_prefix_count_exact','sensor_boundary_exact','sensor_watermark_exact','sensor_source_ordinal_exact')):
                    raise AssertionError('persistent numeric sensor prefix mismatch')
            else:
                row['EMA_watermark_exact']=mem['cutoff_ms']==lane['cursor_ms']
        active.append(row)
    if runtime['queries']!=lanes['outputs'] or runtime['commits']!=lanes['outputs']:
        raise AssertionError('query/commit/output counts differ')
    return dict(queries=runtime['queries'],commits=runtime['commits'],resets=runtime['resets'],active=active,
        tails_consumed_before_released_run_reset=dict(events=lanes['tail_events'],ms=lanes['tail_ms']),
        finite_training_horizon='Active runs remain checkpointed at optimizer-budget end; unvisited future events are not consumed or claimed.',
        no_step_reset_evidence='For relation arms, cumulative int64 counts equal full native offset span from each run boundary, not just the last W50.')


def source_replay(cfg,curves,subjects):
    lanes=ContinuousRunLanes(cfg['REL']['DATA_ROOT'],subjects=subjects,
        batch_size=cfg['STAGES']['small']['BATCH_SIZE'],seed=cfg['REL']['SEED'])
    clip=cfg['STAGES']['small']['CLIP_STEPS'];cumulative=hashlib.sha256()
    previous={};continuations=boundaries=tail_packets=0
    def append(h,a):
        a=np.ascontiguousarray(a);h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    for row in curves:
        h=hashlib.sha256()
        for phase in range(clip):
            for item in lanes.next_batch():
                sid=item['seq_id'];start=item['start_ms'];cutoff=item['cutoff_ms']
                if sid in previous:
                    if previous[sid]!=start:raise AssertionError('stream prefix discontinuity')
                    continuations+=1
                else:
                    if start!=item['run_start_ms'] or not item['reset']:raise AssertionError('false reset boundary')
                    boundaries+=1
                previous[sid]=cutoff
                if item['events'].size and (item['events'][:,2].min()<start or item['events'][:,2].max()>=cutoff):
                    raise AssertionError('future/late events in sampling replay')
                h.update(sid.encode());append(h,np.asarray(item['source_range'],np.int64));append(h,item['events'])
                append(h,item['betas']);append(h,item['K_cam'])
                tail=item.get('released_tail')
                if tail is not None:
                    append(h,tail['events']);append(h,np.asarray(tail['source_range'],np.int64));tail_packets+=1
        digest=h.hexdigest();cumulative.update(digest.encode())
        if digest!=row['step_source_digest'] or cumulative.hexdigest()!=row['sample_digest']:
            raise AssertionError('independent source replay differs at optimizer step '+str(row['step']))
        if row['outputs']!=lanes.outputs or row['events']!=lanes.events or row['commits']!=lanes.outputs:
            raise AssertionError('saved output/event/commit counter differs from replay')
    return dict(pass_all_steps=True,optimizer_steps=len(curves),continuous_packets=continuations,
        genuine_run_boundaries=boundaries,tail_packets=tail_packets,events=lanes.events,outputs=lanes.outputs,
        actual_digest=cumulative.hexdigest()),lanes


def zero_init_gradient_check():
    torch.manual_seed(3407);result={}
    f=torch.randn(2,51,24);a=torch.randn(2,128,51);lam=torch.rand(2,128,24);conditions=torch.ones(2,3)
    for name,cls in (('points',TypedSupportAdapter),('joint_gram',JointGramSupportAdapter)):
        model=cls();out=model(f,a,lam,conditions)
        if torch.count_nonzero(out):raise AssertionError('typed increment is nonzero at initialization')
        out.sum().backward()
        upstream=sum(float(p.grad.abs().sum()) for m in (model.direction,model.support) for p in m.parameters())
        final=sum(float(p.grad.abs().sum()) for p in model.output.parameters())
        if upstream!=0 or final<=0:raise AssertionError('zero-init first-gradient role differs')
        result[name]=dict(exact_zero_initial_increment=True,first_backward_hidden_gradient_L1=upstream,
            first_backward_output_gradient_L1=final,interpretation='Output learns on first update; hidden adapter receives gradients after output weights move.')
    return result


def run(out=OUT/'audits/small_training.json'):
    torch.set_num_threads(1);started=time.perf_counter()
    report=dict(schema='s43_relation_independent_training_audit_v1',source_sha256=sha(__file__),
        scope='Read-only artifacts/source replay plus synthetic gradient algebra; zero optimizer updates and no task scores.',stages={},hard_failures=[])
    full_lanes=ContinuousRunLanes(DATA_ROOT,subjects=FIT7+SELECT,batch_size=8,seed=3407)
    manifest=read_manifest(DATA_ROOT)
    original_train=set(manifest['train']['trials'])
    if set(full_lanes.readers)!=original_train:raise AssertionError('full training recordings differ from original72 manifest')
    short=[]
    for name,reader in full_lanes.readers.items():
        for run_id,(a,b) in enumerate(reader.runs):
            if int(b)-int(a)<50:
                short.append(dict(recording=name,run_id=run_id,ms=int(b)-int(a),events=int(reader.offsets[int(b)]-reader.offsets[int(a)])))
    report['full_data_manifest']=dict(subjects=list(FIT7+SELECT),recordings=len(full_lanes.readers),
        exact_original_train72=set(full_lanes.readers)==original_train,eligible_runs=len(full_lanes.runs),
        short_runs_excluded=len(short),short_run_ms_excluded=sum(x['ms'] for x in short),short_run_events_excluded=sum(x['events'] for x in short),
        short_run_policy='Runs shorter than W50 provide no training target; excluded jointly, not called fully consumed.',
        planned_full=dict(updates=6000,batch=8,clip=4,supervised_outputs=192000,historical_reference_samples=6144000,fraction=.03125),
        warning='192000 new supervised outputs per arm are 3.125% of historical6144000; paired finetune, not a fresh equal-budget retraining.')
    stage_traces={};replay=None
    for stage in ('small','small_r1'):
        result={};traces=[]
        for arm in ARMS:
            folder=OUT/'train'/stage/arm
            cfg=read(folder/'config_resolved.json');complete=read(folder/'complete.json')
            curves=[json.loads(s) for s in (folder/'curve.jsonl').read_text().splitlines() if s.strip()]
            saved=torch.load(folder/'last.ckpt',map_location='cpu',weights_only=False)
            expected=cfg['STAGES']['small'];n=expected['STEPS'];outputs=n*expected['BATCH_SIZE']*expected['CLIP_STEPS']
            if [r['step'] for r in curves]!=list(range(1,n+1)) or saved['step']!=n or complete['completed_steps']!=n:
                raise AssertionError('actual optimizer update count mismatch')
            if saved['lanes']['outputs']!=outputs or complete['actual_outputs']!=outputs:
                raise AssertionError('actual supervised output count mismatch')
            if sha(folder/'last.ckpt')!=complete['checkpoint_sha256']:raise AssertionError('checkpoint checksum mismatch')
            cumulative=hashlib.sha256()
            for row in curves:
                cumulative.update(row['step_source_digest'].encode())
                if row['sample_digest']!=cumulative.hexdigest():raise AssertionError('source hash chain mismatch')
            if saved['sample_digest']!=cumulative.hexdigest() or complete['sample_digest']!=cumulative.hexdigest():
                raise AssertionError('final source digest mismatch')
            model,_=load_relation_model(cfg,device='cpu')
            changes=tensor_changes(model,saved)
            if changes['base.']['changed_parameters']==0:raise AssertionError('base was not trained')
            reader=cfg['REL'].get('SUPPORT_READER','points')
            if saved.get('reader','points')!=reader:raise AssertionError('checkpoint reader marker mismatch')
            scheduler=saved['scheduler_state_dict'];optimizer=saved['optimizer_state_dict']
            opt_steps=sorted(set(int(v['step']) for v in optimizer['state'].values() if 'step' in v))
            if opt_steps!=[n] or scheduler['last_epoch']!=n:raise AssertionError('optimizer/scheduler steps missing')
            raw=np.asarray([r['clip_raw_losses'] for r in curves],np.float64)
            optimized=np.log10(np.maximum(raw,1e-12)).mean(1)
            logged=np.asarray([r['loss'] for r in curves])
            if not np.allclose(logged,raw.mean(1),rtol=1e-12,atol=1e-12):raise AssertionError('raw curve averaging mismatch')
            gap=np.log10(raw.mean(1))-optimized
            result[arm]=dict(completed_updates=n,actual_outputs=outputs,events=saved['lanes']['events'],reader=reader,
                sample_digest=saved['sample_digest'],per_step_sources=len(curves),source_files_unchanged_at_save=complete['source_files_unchanged'],
                all_base_parameters_require_grad=all(p.requires_grad for p in model.base.parameters()),
                parameters=changes,optimizer=dict(type='Adam',state_entries=len(optimizer['state']),actual_step_values=opt_steps,
                    state_present=True,weight_decay=optimizer['param_groups'][0]['weight_decay']),
                scheduler=dict(saved=True,last_epoch=scheduler['last_epoch'],step_count=scheduler['_step_count'],last_lr=scheduler['_last_lr'],
                    warmup_steps=expected['WARMUP_STEPS'],first_update_used_lr=expected['LR']/expected['WARMUP_STEPS'],
                    first_curve_record_lr=curves[0]['lr'],curve_semantics='after scheduler.step: next-update LR, not the just-applied update'),
                loss=dict(original_raw_type=model.base.loss_type,original_log10=model.base.log10_loss,
                    raw_forward_loss_method='base._compute_loss, original so3_trans_fk float32',
                    actual_gradient_objective='mean_phase(log10(max(raw_loss_phase,1e-12)))',
                    curve_loss='mean_phase(raw_loss_phase)',not_large_batch_equivalent=True,
                    Jensen_gap_log10mean_minus_meanlog_mean=float(gap.mean()),Jensen_gap_max=float(gap.max()),
                    minimum_raw_loss=float(raw.min()),clamp_activated=bool(np.any(raw<1e-12)),
                    gradient_clipping=dict(rule='global L2 clip_grad_norm_ after all phase/clip backward calls, before Adam.step',
                        threshold=cfg['REL'].get('GRAD_CLIP',10.),
                        max_preclip_norm=max(r['grad_norm'] for r in curves),
                        updates_above_threshold=sum(r['grad_norm']>cfg['REL'].get('GRAD_CLIP',10.) for r in curves)),
                    loss_weights={k:float(getattr(model.base,k)) for k in ('rot_weight','trans_weight','fk_weight','abs_fk_weight')},
                    omitted_original_extra_terms={k:float(getattr(model.base,k,0.)) for k in ('distill_weight','accel_weight','gain_reg_w')}),
                runtime=runtime_check(saved,full_lanes.readers),rng_state_saved=all(k in saved for k in ('python_rng','numpy_rng','torch_rng','cuda_rng')))
            traces.append([r['step_source_digest'] for r in curves])
            if replay is None:
                replay,_=source_replay(cfg,curves,FIT7)
            del model,saved
        if not all(t==traces[0] for t in traces):raise AssertionError('four-arm source traces are not paired')
        result['four_arm_per_step_sources_equal']=True
        stage_traces[stage]=traces[0]
        report['stages'][stage]=result
    report['R0_R1_per_step_sources_equal']=stage_traces['small']==stage_traces['small_r1']
    if not report['R0_R1_per_step_sources_equal']:raise AssertionError('R0/R1 source schedules changed')
    report['independent_actual_sampling_replay']=replay
    report['zero_init_gradient_roles']=zero_init_gradient_check()
    marker={}
    for config_stage,checkpoint_stage in (('small_r1','small'),('small','small_r1')):
        cfg=read(OUT/'train'/config_stage/'full/config_resolved.json')
        checkpoint=OUT/'train'/checkpoint_stage/'full/step=120.ckpt'
        try:load_relation_model(cfg,checkpoint=checkpoint,device='cpu')
        except ValueError as e:
            if 'SUPPORT_READER mismatch' not in str(e):raise
            marker[f'{checkpoint_stage}_checkpoint_into_{config_stage}_reader']=dict(rejected=True,error=str(e))
        else:raise AssertionError('same-shaped mismatched reader checkpoint was silently accepted')
    report['reader_marker_cross_load']=marker
    report['training_initialization']=dict(base='same original S43 step6000 checkpoint, all base parameters trainable',
        public_state=PUBLIC_INITIAL_STATE.tolist(),label_initialization=False,
        first_window='At each valid-run boundary use public state immediately, supervise first W50, then own detached prediction each hop.',
        no_training_two_window_canonical_burnin=True,
        distinction='E12 mechanism histories and final evaluation warmup are separate contracts; LR warmup is not observation warmup.')
    report['verdict']='PASS_WITH_EXPLICIT_PROTOCOL_DIFFERENCES'
    report['seconds']=time.perf_counter()-started
    dump(out,report)
    print(json.dumps(dict(path=str(out),verdict=report['verdict'],seconds=report['seconds'])),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT/'audits/small_training.json')
    args=p.parse_args();run(args.out)
