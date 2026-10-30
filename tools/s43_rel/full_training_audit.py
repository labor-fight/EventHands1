"""Read-only final6000-step audit; no training, score, or frozen-source change.

Formal execution requires all four complete.json files. --prefix-only checks
the common journal prefix without interpreting unfinished training as done.
Input hashes are actual training-journal pairing evidence, NOT a fresh hash of
billions of raw events. Independent run clocks and event-offset counts can be
replayed for all6000 steps without materializing those raw event payloads.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .core import RelConfig
from .data import ContinuousRunLanes,FIT7,SELECT
from .geometry import GeometryConfig
from .model import load_relation_model
from .training import RelationRuntime
from .training_audit import tensor_changes,runtime_check
from tools.s43_br.data import DATA_ROOT,read_manifest

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/s43_rel_20261027'
ARMS=('s43','enhanced','no_order','full')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n')


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def curves(path,partial=False):
    lines=Path(path).read_text().splitlines();rows=[]
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if partial and i==len(lines)-1:break
            raise
    return rows


def paired_prefix(folder):
    rows={arm:curves(folder/arm/'curve.jsonl',partial=True) for arm in ARMS}
    common=min(map(len,rows.values()))
    for arm,trace in rows.items():
        if [r['step'] for r in trace]!=list(range(1,len(trace)+1)):
            raise AssertionError(arm+' optimizer journal has duplicates/gaps')
    for step in range(common):
        hashes={rows[a][step]['step_source_digest'] for a in ARMS}
        if len(hashes)!=1:raise AssertionError('four-arm input digest differs at step '+str(step+1))
        if len({rows[a][step]['outputs'] for a in ARMS})!=1 or len({rows[a][step]['events'] for a in ARMS})!=1:
            raise AssertionError('four-arm source/output counts differ at step '+str(step+1))
    return rows,dict(common_paired_optimizer_steps=common,all_input_hashes_equal=True,
        journal_lengths={a:len(v) for a,v in rows.items()},
        evidence='Actual training-recorded step_source_digest and counts; not an independent bulk raw-event rehash.',
        completed_arms=[a for a in ARMS if (folder/a/'complete.json').exists()])


def journal_chain(trace):
    h=hashlib.sha256();previous=None;restarts=[]
    for row in trace:
        candidate=h.copy();candidate.update(row['step_source_digest'].encode())
        if candidate.hexdigest()!=row['sample_digest']:
            if previous is None:raise AssertionError('invalid first sample hash chain')
            candidate=hashlib.sha256(previous.encode());candidate.update(row['step_source_digest'].encode())
            if candidate.hexdigest()!=row['sample_digest']:raise AssertionError('sample hash chain mismatch')
            restarts.append(row['step'])
        h=candidate;previous=row['sample_digest']
    return dict(final_digest=h.hexdigest(),resume_hash_chain_boundaries=restarts)


def metadata_source_replay(lanes,trace,batch_size=8,clip=4):
    """Independent run/RNG/clock and offset-count replay; no raw tuples read."""
    rng=np.random.default_rng(3407);runs=lanes.runs
    order=np.empty(0,np.int64);position=traversal=0;slots=[None]*batch_size
    outputs=events=resets=tail_ms=tail_events=0
    per_recording={name:dict(outputs=0,queried_source_events=0,tail_source_events=0) for name in lanes.readers}
    for row in trace:
        for _ in range(clip):
            for lane_id in range(batch_size):
                slot=slots[lane_id]
                if slot is None or slot['cursor_ms']+50>slot['run'].stop_ms:
                    if slot is not None and slot['cursor_ms']<slot['run'].stop_ms:
                        run=slot['run'];reader=lanes.readers[run.recording]
                        n=int(reader.offsets[run.stop_ms]-reader.offsets[slot['cursor_ms']])
                        tail_ms+=run.stop_ms-slot['cursor_ms'];tail_events+=n;events+=n
                        per_recording[run.recording]['tail_source_events']+=n
                    if position>=len(order):
                        order=rng.permutation(len(runs));position=0;traversal+=1
                    run=runs[int(order[position])];position+=1;resets+=1
                    slot=slots[lane_id]=dict(run=run,cursor_ms=run.start_ms,ordinal=0,
                        seq_id=f'lane{lane_id}:pass{traversal}:{position}:{run.recording}:{run.run_id}')
                run=slot['run'];start=slot['cursor_ms'];stop=start+50;reader=lanes.readers[run.recording]
                n=int(reader.offsets[stop]-reader.offsets[start]);events+=n;outputs+=1
                per_recording[run.recording]['outputs']+=1;per_recording[run.recording]['queried_source_events']+=n
                slot['cursor_ms']=stop;slot['ordinal']+=1
        if (row['outputs'],row['events'],row['resets'],row['commits'])!=(outputs,events,resets,outputs):
            raise AssertionError('independent clock/offset replay differs at step '+str(row['step']))
    state=dict(order=order,position=position,traversal=traversal,lanes=slots,rng=rng.bit_generator.state,
        outputs=outputs,events=events,resets=resets,tail_events=tail_events,tail_ms=tail_ms)
    report=dict(pass_all_recorded_steps=True,optimizer_steps=len(trace),outputs=outputs,events=events,
        resets=resets,tail_events=tail_events,tail_ms=tail_ms,traversal=traversal,
        actual_recordings_with_outputs=sum(v['outputs']>0 for v in per_recording.values()),
        per_recording=per_recording,
        scope='Only run metadata, deterministic RNG and mmap event offsets replayed independently; no raw-event hash recreated.')
    return state,report


def numeric_restore(model,cfg,saved):
    runtime=RelationRuntime(model,RelConfig(**cfg.get('MEMORY',{})),GeometryConfig(**cfg.get('GEOMETRY',{})),prep_workers=1)
    runtime.load_state_dict(saved['runtime'])
    snapshot=runtime.state_dict()
    if not equal(snapshot,saved['runtime']):raise AssertionError('full runtime checkpoint numerical restore differs')
    arrays=0;query_checks=[]
    for sid,mem in runtime.memories.items():
        original=saved['runtime']['memories'][sid]
        if 'arrays' in original:
            arrays+=sum(a.nbytes for a in original['arrays'].values())
            for name,a in original['arrays'].items():
                if np.shares_memory(a,getattr(mem,name)):raise AssertionError('restored observation array aliases checkpoint storage')
            E=mem.topology.n_edges;indices=np.unique(np.linspace(0,E-1,min(64,E)).astype(np.int32))
            q=mem.readout(mem.watermark_ms,indices)
            t=int(round(mem.watermark_ms*1000));a=original['arrays'];c=mem.config
            A=(a['_A']*np.exp(-(t-a['_ta'])[:,None,None]/(np.asarray(c.tau_ms)*1000))).astype(c.dtype)
            Z=(a['_Z'][indices]*np.exp(-(t-a['_tz'][indices])[:,None,None,None,None]/(np.asarray(c.gamma_ms)*1000))).astype(c.dtype)
            np.testing.assert_array_equal(q.A,A);np.testing.assert_array_equal(q.Z,Z)
            query_checks.append(dict(seq_id=sid,selected_edges=len(indices),A_Z_lazy_read_exact=True))
        else:
            arrays+=original['A'].nbytes
            if not np.array_equal(mem.A,original['A']):raise AssertionError('EMA restoration differs')
    return dict(full_runtime_roundtrip_bitwise=True,restored_sequences=len(runtime.poses),
        restored_sensor_sequences=len(runtime.memories),numeric_sensor_cache_bytes=arrays,
        numeric_lazy_query_checks=query_checks,
        scope='CPU restoration and lazy values only; no label read, decode, feedback or optimizer step executed.')


def run(folder,out,authorization,prefix_only=False):
    started=time.perf_counter();torch.set_num_threads(1)
    rows,pairing=paired_prefix(folder)
    if prefix_only:
        result=dict(schema='s43_full_training_prefix_audit_v1',complete=False,paired=pairing,
            scope='Read-only common-prefix journal audit; unfinished arms are not complete.')
        dump(out,result);return result
    if set(pairing['completed_arms'])!=set(ARMS):
        raise RuntimeError('NOT_READY: formal audit requires all four6000-step complete.json files')
    if any(len(rows[a])!=6000 for a in ARMS):raise AssertionError('completed journal does not contain exactly6000 updates')
    auth=read(authorization);protocol=BASE/'protocol.json'
    if auth.get('full_training_authorized') is not True or auth.get('protocol_sha256')!=sha(protocol):
        raise AssertionError('full-training authorization/protocol identity mismatch')
    source_lanes=ContinuousRunLanes(DATA_ROOT,subjects=FIT7+SELECT,batch_size=8,seed=3407)
    original_manifest=read_manifest(DATA_ROOT)
    if set(source_lanes.readers)!=set(original_manifest['train']['trials']):raise AssertionError('data is not original72 trials')
    dry_state,dry_report=metadata_source_replay(source_lanes,rows['s43'])
    report=dict(schema='s43_full_training_final_audit_v1',complete=True,source_sha256=sha(__file__),
        training_root=str(folder),authorization_sha256=sha(authorization),protocol_sha256=sha(protocol),
        scope='Read-only final checkpoint/source journals/numeric restoration; no task scores or optimizer updates.',
        input_pairing=pairing,independent_offset_clock_replay=dry_report,arms={},hard_failures=[],
        data=dict(subjects=list(FIT7+SELECT),recordings=72,original_train_manifest_exact=True,
            eligible_runs=len(source_lanes.runs),actual_recordings_with_outputs=dry_report['actual_recordings_with_outputs']),
        budget=dict(updates_per_arm=6000,batch=8,clip=4,supervised_outputs_per_arm=192000,
            historical_reference_samples=6144000,fraction=.03125,
            role='Paired finetune from same original step6000 checkpoint; not fresh equal-budget historical retraining.'),
        inherited_small_audit=str(BASE/'audits/small_training.json'))
    reference_sources=None;reference_weights=None
    for arm in ARMS:
        directory=folder/arm;cfg=read(directory/'config_resolved.json');complete=read(directory/'complete.json');trace=rows[arm]
        if complete['completed_steps']!=6000 or complete['actual_outputs']!=192000:raise AssertionError('completion counters wrong')
        if not complete['source_files_unchanged']:raise AssertionError('source hash guard failed during training')
        if complete['protocol_sha256']!=sha(protocol):raise AssertionError('completed protocol hash differs')
        configs=auth['config_sha256'];matching=[Path(p) for p,h in configs.items() if h==complete['config_sha256']]
        if not matching or any(sha(p)!=complete['config_sha256'] for p in matching):raise AssertionError('frozen config hash differs')
        if sha(cfg['REL']['BASE_CHECKPOINT'])!=complete['base_checkpoint_sha256']:
            raise AssertionError('canonical initialization checkpoint hash differs')
        sources=complete['source_files']
        if any(sha(p)!=h for p,h in sources.items()):raise AssertionError('a frozen training dependency changed after the run')
        if reference_sources is None:reference_sources=sources
        if sources!=reference_sources:raise AssertionError('four arms used different executable dependency hashes')
        chain=journal_chain(trace)
        saved=torch.load(directory/'last.ckpt',map_location='cpu',weights_only=False)
        grid=torch.load(directory/'step=6000.ckpt',map_location='cpu',weights_only=False)
        if saved['step']!=6000 or grid['step']!=6000:raise AssertionError('last/grid step differs')
        if saved['arm']!=arm or grid['arm']!=arm or saved['config_sha256']!=complete['config_sha256'] or grid['config_sha256']!=complete['config_sha256']:
            raise AssertionError('last/grid arm or frozen config identity differs')
        if sha(directory/'last.ckpt')!=complete['checkpoint_sha256']:raise AssertionError('last checkpoint checksum differs')
        if saved['sample_digest']!=chain['final_digest'] or complete['sample_digest']!=chain['final_digest']:
            raise AssertionError('final saved journal chain differs')
        if not equal(saved['model_state_dict'],grid['model_state_dict']):raise AssertionError('grid deployment weights differ from last training weights')
        if any(k in grid for k in ('optimizer_state_dict','scheduler_state_dict','runtime','lanes')):
            raise AssertionError('grid deployment artifact contains unrelated training cache')
        if saved.get('reader')!='joint_gram' or grid.get('reader')!='joint_gram' or cfg['REL'].get('SUPPORT_READER')!='joint_gram':
            raise AssertionError('R1 reader identity missing or mismatched')
        model,_=load_relation_model(cfg,device='cpu')
        changes=tensor_changes(model,saved)
        if changes['base.']['changed_parameters']==0 or not changes['BN_buffers']['all_unchanged']:
            raise AssertionError('base training or frozen BN check failed')
        step_values=sorted(set(int(v['step']) for v in saved['optimizer_state_dict']['state'].values() if 'step' in v))
        if step_values!=[6000] or saved['scheduler_state_dict']['last_epoch']!=6000:
            raise AssertionError('Adam/scheduler actual step count differs')
        if not all(k in saved for k in ('python_rng','numpy_rng','torch_rng','cuda_rng')) or not saved['cuda_rng']:
            raise AssertionError('resumable RNG state is missing')
        for key in dry_state:
            if not equal(saved['lanes'][key],dry_state[key]):raise AssertionError('saved sampler differs from independent metadata replay: '+key)
        runtime_report=runtime_check(saved,source_lanes.readers)
        model.load_state_dict(saved['model_state_dict'],strict=True)
        restored=numeric_restore(model,cfg,saved)
        loss=np.asarray([r['clip_raw_losses'] for r in trace],np.float64)
        gap=np.log10(loss.mean(1))-np.log10(np.maximum(loss,1e-12)).mean(1)
        curve_loss=np.asarray([r['loss'] for r in trace])
        if not np.allclose(loss.mean(1),curve_loss,rtol=1e-12,atol=1e-12):raise AssertionError('curve loss is not mean raw phase loss')
        optimizer_bytes=sum(t.numel()*t.element_size() for slot in saved['optimizer_state_dict']['state'].values() for t in slot.values() if isinstance(t,torch.Tensor))
        weight_bytes=sum(t.numel()*t.element_size() for t in grid['model_state_dict'].values())
        report['arms'][arm]=dict(actual_optimizer_updates=6000,actual_supervised_outputs=saved['lanes']['outputs'],
            actual_events=saved['lanes']['events'],actual_recordings_derived=dry_report['actual_recordings_with_outputs'],
            source_guard=True,config_guard=True,reader='joint_gram',chain=chain,parameters=changes,
            optimizer=dict(actual_state_steps=step_values,state_entries=len(saved['optimizer_state_dict']['state']),saved=True),
            scheduler=dict(saved=True,last_epoch=saved['scheduler_state_dict']['last_epoch'],last_lr=saved['scheduler_state_dict']['_last_lr'],warmup_steps=cfg['STAGES']['full']['WARMUP_STEPS']),
            RNG=dict(python_saved=True,numpy_saved=True,torch_cpu_saved=True,cuda_state_tensors=len(saved['cuda_rng']),
                sampler_generator_exact=True,CUDA_restored_during_CPU_audit=False),
            runtime=runtime_report,numeric_checkpoint_restore=restored,
            checkpoint_roles=dict(deployment_grid=str(directory/'step=6000.ckpt'),resumable_last=str(directory/'last.ckpt'),
                model_tensors_identical=True,grid_file_bytes=(directory/'step=6000.ckpt').stat().st_size,
                last_file_bytes=(directory/'last.ckpt').stat().st_size,model_tensor_bytes=weight_bytes,extra_Adam_tensor_bytes=optimizer_bytes,
                preference='Use step=6000.ckpt for online memory/latency; last includes8 training-lane states, Adam and RNG.',
                memory_caveat='torch.load(map_location=cuda) may move extra Adam tensors to GPU; NumPy sensor caches remain CPU arrays, not all training cache is CUDA memory.'),
            loss=dict(objective='mean_phase(log10(max(L_phase,1e-12)))',curve='mean_phase(rawL)',
                equals_log_of_large_batch_mean=False,Jensen_gap_mean=float(gap.mean()),Jensen_gap_max=float(gap.max()),
                clamp_activated=bool((loss<1e-12).any()),gradient_clip=cfg['REL'].get('GRAD_CLIP',10.),
                clip_triggered_updates=sum(r['grad_norm']>cfg['REL'].get('GRAD_CLIP',10.) for r in trace)))
        if arm=='full':
            wrong=copy.deepcopy(cfg);wrong['REL']['SUPPORT_READER']='points'
            try:load_relation_model(wrong,checkpoint=directory/'step=6000.ckpt',device='cpu')
            except ValueError as error:
                if 'SUPPORT_READER mismatch' not in str(error):raise
                report['R1_checkpoint_wrong_reader_rejected']=str(error)
            else:raise AssertionError('R1 same-shaped checkpoint silently accepted as point reader')
        del model,saved,grid
    report['zero_initialization_and_gradient_role']='Validated in inherited small audit; full checkpoints retain active branch updates and inactive branch zeros. No redundant new backward test.'
    report['unconsumed_scope']='Short<50ms runs are jointly excluded; active future run tails beyond the finite6000-step budget remain checkpointed, not claimed consumed.'
    report['verdict']='PASS_WITH_EXPLICIT_PROTOCOL_DIFFERENCES'
    report['seconds']=time.perf_counter()-started
    dump(out,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--train-root',type=Path,default=BASE/'train/full_r1')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--authorization',type=Path,default=BASE/'authorization_full_r1.json')
    parser.add_argument('--prefix-only',action='store_true')
    args=parser.parse_args()
    output=args.out or BASE/'audits'/('full_training_prefix.json' if args.prefix_only else 'full_training.json')
    result=run(args.train_root,output,args.authorization,args.prefix_only)
    print(json.dumps(dict(output=str(output),complete=result['complete'],verdict=result.get('verdict','PREFIX_ONLY'))),flush=True)
