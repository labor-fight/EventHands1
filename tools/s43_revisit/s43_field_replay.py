"""Stage5 fixed-last, target-free continuous replay and separate saved-trace score.

The trained shared runtime is reused unchanged. Primary beta0 input never
opens target metadata or auxiliary betas. Scoring has a separate cohort gate.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,os,resource,struct,time,traceback
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_info
from tools.s43_br.data import DATA_ROOT,PUBLIC_INITIAL_STATE
from tools.s43_rel.mean_shape_replay import MeanShapeInputReader,InferenceAccessGuard,_mano,fk
from .s43_field_data import make_input_batch
from .s43_field_runtime import SharedS43Runtime
from .s43_field_native import NativeS43Runtime
from .s43_field_training_R1 import MODEL_VARIANT,R1_SHA
from .s43_field_full_training import build_model,code_dependencies,EXECUTION_VARIANT,PILOT_FREEZE_SHA

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/s43_revisit_20261028'
TRAIN=BASE/'s43_full_field_R1'
ENGINEERING=BASE/'s43_field_replay_R1_engineering'
PROTOCOL=BASE/'stage5_evaluation_protocol.json'
PROTOCOL_SHA='70be35e8a24bfaf232c538b3249af614b23108759362ae3a0215ee6bebb672a0'
MODELS=('canonical','s43','field_box','field_box_collapse','field_static','field_static_collapse')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def encode(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,Path):return str(v)
        raise TypeError('unsupported JSON evidence type: '+type(v).__name__)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=encode)+'\n')


def read(path):return json.loads(Path(path).read_text())
def sync(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)


def configure(device='cpu',threads=2):
    if int(threads)<1:raise ValueError('positive CPU threads required')
    if torch.device(device).type=='cuda':
        visible=os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible is None:raise ValueError('explicit physical GPU4–7 CUDA_VISIBLE_DEVICES required')
        devices=[int(x) for x in visible.split(',')]
        if any(x not in (4,5,6,7) for x in devices) or torch.device(device).index not in range(len(devices)):raise ValueError('only physical GPU4–7')
    torch.set_num_threads(int(threads));torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False


def all_weights_gate(train_root=TRAIN):
    root=Path(train_root);marker=root/'all_final_weights_complete.json';d=read(marker)
    audit=root/'audits/full.json';a=read(audit)
    arms=set(MODELS[1:])
    if d.get('status')!='COMPLETE' or set(d.get('arms',[]))!=arms or d.get('supervised_outputs_per_arm')!=6144000:raise RuntimeError('allfivefullfixedlastweights must complete before formal prediction')
    if a.get('status')!='PASS' or not a.get('all_per_update_source_target_traces_paired') or not a.get('all_sources_unchanged'):raise RuntimeError('fivefull source/budget/pairing audit must pass')
    if a.get('sourcefreeze_sha256')!=d.get('sourcefreeze_sha256') or a.get('supervised_outputs_per_arm')!=6144000:raise RuntimeError('fullmarker/audit freeze or budget mismatch')
    if (d.get('execution_variant')!=EXECUTION_VARIANT or d.get('pilot_sourcefreeze_sha256')!=PILOT_FREEZE_SHA or
            a.get('execution_variant')!=EXECUTION_VARIANT or a.get('pilot_sourcefreeze_sha256')!=PILOT_FREEZE_SHA or
            sha(d['sourcefreeze_path'])!=d['sourcefreeze_sha256']):raise RuntimeError('full execution identity/pilot bridge mismatch')
    frozen=read(d['sourcefreeze_path'])
    for path,expected in frozen['code'].items():
        if sha(path)!=expected:raise RuntimeError('full training/serving executable identity changed: '+path)
    pilot=root/'audits/pilot.json'
    if sha(pilot)!=d['pilot_audit_sha256'] or read(pilot).get('status')!='PASS':raise RuntimeError('adopted pilot audit identity mismatch')
    checkpoints={}
    for arm in MODELS[1:]:
        p=Path(d['final_weights'][arm]);c=read(p.with_name('complete.json'))
        if (c.get('status')!='COMPLETE' or c.get('stage')!='full' or c.get('actual_supervised_outputs')!=6144000
                or not c.get('source_files_unchanged') or c.get('sourcefreeze_sha256')!=d['sourcefreeze_sha256']
                or sha(p)!=c.get('final_weights_sha256')):raise RuntimeError('fixedlastweight identity/budget mismatch: '+arm)
        checkpoints[arm]=str(p)
    return dict(marker=str(marker),marker_sha256=sha(marker),audit=str(audit),audit_sha256=sha(audit),checkpoints=checkpoints,
        sourcefreeze_path=d['sourcefreeze_path'],sourcefreeze_sha256=d['sourcefreeze_sha256'],execution_variant=EXECUTION_VARIANT,pilot_sourcefreeze_sha256=PILOT_FREEZE_SHA)


def load_weights(checkpoint,device='cpu'):
    """Explicit serving registry; the failed attempt0 schema is never guessed."""
    header=torch.load(checkpoint,map_location='cpu');schema=header.get('schema');del header
    if schema=='s43_field_full_training_v1':
        from .s43_field_full_training import load_weights as loader
    elif schema=='s43_field_training_R1_v1':
        from .s43_field_training_R1 import load_weights as loader
    else:raise ValueError('explicit R1 pilot or native-ring full checkpoint schema required')
    return loader(checkpoint,device)


def load_model(arm,checkpoint=None,device='cpu'):
    if arm not in MODELS:raise ValueError('unknown fixed model')
    if arm=='canonical':
        if checkpoint is not None:raise ValueError('canonical uses the frozen original checkpoint')
        model,contract=build_model('s43',device)
        from tools.s43_integration.model import DEFAULT_CKPT,DEFAULT_CFG
        payload=dict(arm='canonical',variant='canonical_frozen_reference',model_variant='canonical_s43',runtime_contract=contract,
            base_checkpoint=str(DEFAULT_CKPT),base_config=str(DEFAULT_CFG),supervised_outputs=0)
    else:
        if checkpoint is None:raise ValueError('trained arm requires explicit fixedlast checkpoint')
        model,payload=load_weights(checkpoint,device)
        if payload['arm']!=arm:raise ValueError('checkpoint arm mismatch')
        if (payload.get('variant')!='reacquisition_R1' or payload.get('R1_protocol_sha256')!=R1_SHA or
                payload.get('model_variant')!=('canonical_s43' if arm=='s43' else MODEL_VARIANT)):raise ValueError('explicit R1 serving variant required; never load old failing factory')
    payload=dict(payload)
    outputs=int(payload.get('supervised_outputs',0));engineering_only=bool(payload.get('engineering_only',False))
    payload['serving_status']=dict(loaded_checkpoint_schema=payload.get('schema','canonical_frozen_reference'),
        actual_supervised_outputs=outputs,trained_from_artifact=not engineering_only and (arm=='canonical' or outputs>0),
        engineering_only=engineering_only,full_Stage5_budget_complete=not engineering_only and outputs==6144000,
        count_scope='Stage5 supervised outputs; canonical reference has priortraining, not re-inferred fromthisstage0 counter',
        inherited_model_metadata='static trained=False/untrained schema describes initial factory only; loadedartifact budget/status above is authoritative',
        coordinate_metadata='10class/63proxy strings inherited from coordinateprototype do not describe actual originalS43 51D/MANO decoder')
    model.eval();model.requires_grad_(False)
    return model,payload


def buffer_digest(model):
    h=hashlib.sha256()
    for k,v in model.named_buffers():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def decode_surface(model,result):
    """Original MANO mean-once FK in metres, including full D2H output."""
    base=getattr(model,'base',model);device=result.prediction.device
    sync(device);a=time.perf_counter()
    v,j=base._fk(result.prediction.float(),result.prepared.beta_input.float())
    sync(device);b=time.perf_counter()
    v=v[0].float().cpu().numpy().copy();j=j[0].float().cpu().numpy().copy()
    p=result.prediction[0].float().cpu().numpy().copy();sync(device);c=time.perf_counter()
    if v.shape!=(778,3) or j.shape!=(21,3) or not np.isfinite(np.r_[v.ravel(),j.ravel(),p]).all():raise ArithmeticError('nonfinite/full MANO surface failure; no hold/clamp')
    return p,v,j,dict(surface_fk_ms=(b-a)*1000,surface_and_pose_D2H_ms=(c-b)*1000)


def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def prefix_identity(runtime,prepared,sid):
    """Accuracy-audit overhead only; excluded from resource service runs."""
    lane=runtime.lanes[sid];h=hashlib.sha256()
    if isinstance(runtime,NativeS43Runtime):h.update(lane['last_timestamp'].tobytes())
    else:
        state=lane['memory'].state_dict()
        for k,v in sorted(state['arrays'].items()):h.update(k.encode());h.update(v.tobytes())
    hh=hashlib.sha256()
    for t,p in lane['history']:hh.update(np.asarray([t],np.int64).tobytes());hh.update(p.tobytes())
    return dict(sensor_sha256=h.hexdigest(),history_sha256=hh.hexdigest(),
        observation_input_sha256=tensor_sha(prepared.legacy_images if runtime.arm=='legacy_s43' else prepared.field40))


class FullStreamRunner:
    """One runtime, independent true-run resets, all tails consumed exactly once."""
    def __init__(self,model,contract,*,native_s43=False,beta=None,hop=50,commit_ms=None,
                 perturbation='none',audit_prefix=True,decoder=decode_surface,phase_callback=None):
        if hop not in (25,50,100):raise ValueError('query hop25|50|100')
        commit_ms=hop if commit_ms is None else int(commit_ms)
        if (hop,commit_ms) not in ((25,25),(50,50),(100,100),(25,50)):raise ValueError('unsupported fixed query/commit schedule')
        self.model=model.eval();self.hop=hop;self.commit_ms=commit_ms;self.perturbation=perturbation
        self.beta=np.array(np.zeros(10) if beta is None else beta,np.float32);self.audit_prefix=audit_prefix
        if self.beta.shape!=(10,) or not np.isfinite(self.beta).all():raise ValueError('external beta10 required')
        if native_s43:C=NativeS43Runtime
        elif contract.get('readout_backend')=='native_ring_bins_v1':
            from .runtime_fastreadout import SharedS43RuntimeFastReadout
            C=SharedS43RuntimeFastReadout
        elif 'runtime_workers' in contract:
            from .s43_field_runtime_R1 import SharedS43RuntimeR1
            C=SharedS43RuntimeR1
        else:C=SharedS43Runtime # explicit reference/engineering contract only
        if native_s43 and contract['runtime_arm']!='legacy_s43':raise ValueError('native S43 only')
        runtime_kwargs={} if native_s43 or 'runtime_workers' not in contract else dict(runtime_workers=int(contract['runtime_workers']))
        self.runtime=C(model,arm=contract['runtime_arm'],transport_chunk_ms=1,
            temporal_collapse=contract['temporal_collapse'],tensor_readout=contract['tensor_readout'],
            state_fp32=contract['state_fp32'],allow_external_beta=bool(np.any(self.beta)),**runtime_kwargs)
        self.decoder=decoder;self.phase=phase_callback or (lambda name,ms:None)
        self.phase_category='startup';self.peak_resident_state_bytes=0

    def close(self):
        if hasattr(self.runtime,'close'):self.runtime.close()

    def diagnostics(self):
        return copy.deepcopy(self.model.diagnostic_state_dict()) if hasattr(self.model,'diagnostic_state_dict') else None

    def _item(self,reader,recording,run_id,a,b,start,T,ordinal,reset):
        packet=reader.packet(start,T)
        return dict(lane=0,valid=True,padding=False,seq_id=f'{recording}:run{run_id}',reset=reset,
            recording=recording,run_id=run_id,run_start_ms=a,run_stop_ms=b,start_ms=start,cutoff_ms=T,
            target_ms=T-1,packet_ordinal=ordinal,initial_state=PUBLIC_INITIAL_STATE.copy(),
            beta_input=self.beta.copy(),K_source=reader.K_cam.copy(),**packet)

    def _inject(self,sid):
        if self.perturbation=='none':return None
        from .s43_field_perturb import perturb_pose
        lane=self.runtime.lanes[sid]
        if self.runtime._prepared is not None:raise RuntimeError('perturb only after warmup commit')
        before=prefix_identity(self.runtime,type('P',(),dict(legacy_images=torch.empty(0),field40=torch.empty(0)))(),sid)['sensor_sha256']
        changed,meta=perturb_pose(lane['pose'],self.perturbation);changed=np.asarray(changed,np.float32)
        lane['pose']=changed.copy();lane['history'][-1]=(lane['history'][-1][0],changed.copy());lane['pose_version']+=1
        after=prefix_identity(self.runtime,type('P',(),dict(legacy_images=torch.empty(0),field40=torch.empty(0)))(),sid)['sensor_sha256']
        if before!=after:raise AssertionError('perturbation changed sensor observations')
        return dict(meta,sensor_before_sha256=before,sensor_after_sha256=after,cutoff_ms=lane['watermark'])

    @torch.inference_mode()
    def run(self,reader,recording,*,runs=None,on_output=None,on_tail=None):
        traces=[];warm=[];failures=[];injections=[];source_h=hashlib.sha256();source_count=0;source_fully_consumed=True
        camera={key:dict(query_rows=0,commit_rows=0,finger_nearplane_query_rows=0,finger_nearplane_commit_rows=0,
            root_nonpositive_query_rows=0,root_nonpositive_commit_rows=0,finger_z_min_m=None,finger_z_max_m=None,root_z_min_m=None,root_z_max_m=None) for key in ('warmup','steady')}
        device=self.runtime.device;buffers_before=buffer_digest(self.model);started=time.perf_counter();timings=[]
        selected=list(enumerate(reader.runs)) if runs is None else list(runs)
        expected=sum(int(reader.offsets[int(b)])-int(reader.offsets[int(a)]) for _,(a,b) in selected)
        diagnostics_before=self.diagnostics();run_diagnostics=[]
        for run_id,(a,b) in selected:
            run_diagnostic_before=self.diagnostics()
            a,b=int(a),int(b);sid=f'{recording}:run{run_id}';cursor=a;ordinal=0;broken=False;reset_required=False
            schedule=[a+50,a+100] if b-a>=100 else [a+50] if b-a>=50 else []
            schedule += list(range(a+100+self.hop,b+1,self.hop))
            schedule=[T for T in schedule if T<=b]
            for step,T in enumerate(schedule):
                self.phase_category='warmup' if step<2 else 'steady'
                sync(device);cycle=time.perf_counter();pack_start=cycle
                item=None;candidate=None;accepted=False
                try:
                    item=self._item(reader,recording,run_id,a,b,cursor,T,ordinal,cursor==a)
                    candidate=source_h.copy();candidate.update(np.asarray(item['events'],np.int64).tobytes())
                    self.phase('source_pack_digest_ms',(time.perf_counter()-pack_start)*1000)
                    batch_start=time.perf_counter();batch=make_input_batch([item],batch_id=ordinal)
                    self.phase('input_contract_hash_ms',(time.perf_counter()-batch_start)*1000)
                    t=time.perf_counter();prepared=self.runtime.inputs(batch);sync(device)
                    source_h=candidate;source_count+=len(item['events']);cursor=T;ordinal+=1;accepted=True
                    self.phase('ingest_readout_context_H2D_ms',(time.perf_counter()-t)*1000)
                    lane=self.runtime.lanes[sid]
                    resident=lane['memory'].nbytes()+sum(p.nbytes+8 for _,p in lane['history'])+lane['pose'].nbytes+lane['beta'].nbytes+lane['K'].nbytes
                    if lane['last_timestamp'] is not None:resident+=lane['last_timestamp'].nbytes
                    self.peak_resident_state_bytes=max(self.peak_resident_state_bytes,resident)
                    audit=prefix_identity(self.runtime,prepared,sid) if self.audit_prefix else {}
                    t=time.perf_counter();result=self.runtime.query(prepared,inference=True);sync(device)
                    self.phase('render_geometry_sampling_network_ms',(time.perf_counter()-t)*1000)
                    p,v,j,dt=self.decoder(self.model,result)
                    for k,value in dt.items():self.phase(k,value)
                    is_warm=step<2;do_commit=is_warm or (T-(a+100))%self.commit_ms==0
                    t=time.perf_counter()
                    if do_commit:self.runtime.commit(result)
                    else:self.runtime.discard_query(result)
                    sync(device);self.phase('commit_or_discard_ms',(time.perf_counter()-t)*1000)
                    elapsed=(time.perf_counter()-cycle)*1000;timings.append(elapsed)
                    self.phase('warmup_cycle_ms' if is_warm else 'steady_cycle_ms',elapsed)
                    # Output diagnostics use the surface already decoded above.
                    # No extra FK, target access, state mutation or service-time
                    # instrumentation; counts are outside the measured cycle.
                    z=j[1:,2];zroot=float(j[0,2]);bad_finger=bool(np.any(z<=.03));bad_root=zroot<=0
                    cs=camera['warmup' if is_warm else 'steady'];cs['query_rows']+=1;cs['commit_rows']+=int(do_commit)
                    cs['finger_nearplane_query_rows']+=int(bad_finger);cs['finger_nearplane_commit_rows']+=int(bad_finger and do_commit)
                    cs['root_nonpositive_query_rows']+=int(bad_root);cs['root_nonpositive_commit_rows']+=int(bad_root and do_commit)
                    for key,value,fn in (('finger_z_min_m',float(z.min()),min),('finger_z_max_m',float(z.max()),max),('root_z_min_m',zroot,min),('root_z_max_m',zroot,max)):
                        cs[key]=value if cs[key] is None else fn(cs[key],value)
                    row=dict(run_id=int(run_id),cutoff_ms=T,end_ms=T-1,committed=do_commit,
                        pred=p,vertices_m=v,joints_m=j,cycle_ms=elapsed,native_prefix_count=source_count,
                        camera_finger_nearplane=bad_finger,camera_root_nonpositive=bad_root,camera_finger_z_min_m=float(z.min()),camera_finger_z_max_m=float(z.max()),**audit)
                    if self.audit_prefix:
                        mesh_h=hashlib.sha256(v.tobytes());mesh_h.update(j.tobytes());row['surface_sha256']=mesh_h.hexdigest()
                        metadata=getattr(self.model,'last_query_metadata',())
                        meta=metadata[0] if len(metadata)==1 else None
                        row.update(local_query_applicable=meta is not None,
                            query_used_mode='legacy_no_local_query' if meta is None else str(meta['used_mode']),
                            used_uniform_reacquisition=False if meta is None else bool(meta['used_uniform_reacquisition']),
                            initial_camera_invalid_count=-1 if meta is None else int(meta['initial_finger_invalid_count']),
                            actual_2d_valid_samples=-1 if meta is None else int(meta['actual_2d_valid_samples']))
                    (warm if is_warm else traces).append({k:value for k,value in row.items() if k not in ('vertices_m','joints_m')})
                    if on_output:on_output(row,is_warm)
                    if step==1:
                        meta=self._inject(sid)
                        if meta is not None:injections.append(dict(run_id=int(run_id),**meta))
                except Exception as exc:
                    lane=self.runtime.lanes.get(sid)
                    # Input construction can fail after ingestion, e.g. H2D OOM.
                    # A prefix is accepted only if its complete logical source
                    # cursor is acknowledged by the runtime itself.
                    if (not accepted and item is not None and lane is not None and
                            lane['watermark']==T and lane['source_end']==item['source_range'][1]):
                        source_h=candidate;source_count+=len(item['events']);cursor=T;ordinal+=1;accepted=True
                    reset_required=not accepted
                    failures.append(dict(run_id=int(run_id),cutoff_ms=T,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
                        complete_logical_packet_accepted=accepted,accepted_prefix_ms=cursor,reset_required=reset_required,
                        core_state=None if lane is None or not hasattr(lane['memory'],'stats') else lane['memory'].stats()))
                    # No fabricated pose/hold prediction. Consume the rest as tail,
                    # release the failed run, preserve every preceding real output.
                    self.runtime._prepared=self.runtime._pending=None;broken=True;break
            if reset_required:
                # A rejected/partially applied invalid input is not silent time.
                # Keep all preceding real predictions and mark the unconsumed
                # interval explicitly; never retry/crop a malformed packet.
                source_fully_consumed=False
                if sid in self.runtime.lanes:
                    self.runtime.lanes[sid]['memory'].release();del self.runtime.lanes[sid];self.runtime.audit['releases']+=1
                if on_tail:on_tail(dict(run_id=int(run_id),start_ms=cursor,stop_ms=b,failed=True,unconsumed=True,reset_required=True))
                run_diagnostics.append(dict(run_id=int(run_id),before=run_diagnostic_before,after=self.diagnostics()))
                continue
            if sid not in self.runtime.lanes:
                context=dict(seq_id=sid,initial_state=PUBLIC_INITIAL_STATE,beta_input=self.beta,K_source=reader.K_cam,run_start_ms=a)
                self.runtime._reset(context)
            self.phase_category='tail';tail_start=time.perf_counter();release=dict(seq_id=sid,valid=True,recording=recording,start_ms=cursor,cutoff_ms=b,packet_ordinal=ordinal,events=None)
            try:
                if cursor<b:release.update(reader.packet(cursor,b))
                self.runtime.inputs(make_input_batch([],releases=[release],batch_id=ordinal));sync(device)
                if release['events'] is not None:
                    source_h.update(np.asarray(release['events'],np.int64).tobytes());source_count+=len(release['events'])
            except Exception as exc:
                source_fully_consumed=False;broken=True
                failures.append(dict(run_id=int(run_id),cutoff_ms=b,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
                    phase='tail_consume_release',accepted_prefix_ms=cursor,reset_required=True))
                if sid in self.runtime.lanes:
                    self.runtime.lanes[sid]['memory'].release();del self.runtime.lanes[sid];self.runtime.audit['releases']+=1
            self.phase('tail_consume_release_ms',(time.perf_counter()-tail_start)*1000)
            if on_tail:on_tail(dict(run_id=int(run_id),start_ms=cursor,stop_ms=b,failed=broken))
            run_diagnostics.append(dict(run_id=int(run_id),before=run_diagnostic_before,after=self.diagnostics()))
        if self.runtime.lanes:raise AssertionError('every sequence must be released')
        transaction_audit_passed=source_count==expected and self.runtime.audit['ingested_events']==expected
        if source_fully_consumed and not transaction_audit_passed:
            failures.append(dict(error_type='SourceAuditMismatch',error='accepted source count/runtime audit/expected count differ',
                accepted_count=source_count,runtime_count=self.runtime.audit['ingested_events'],expected_count=expected,reset_required=True))
        buffers_after=buffer_digest(self.model)
        if buffers_before!=buffers_after:failures.append(dict(error_type='ModelBufferMutation',error='eval query mutated model buffers',reset_required=True))
        for cs in camera.values():
            for reason in ('finger_nearplane','root_nonpositive'):
                for denominator in ('query','commit'):
                    n=cs[denominator+'_rows'];cs[reason+'_'+denominator+'_fraction']=cs[reason+'_'+denominator+'_rows']/n if n else None
        return traces,warm,dict(status='COMPLETE' if not failures else 'FAILED_PRESERVED',recording=recording,
            native_input_count=source_count,native_input_sha256=source_h.hexdigest(),expected_native_count=expected,
            source_fully_consumed=source_fully_consumed,unconsumed_native_count=expected-source_count,
            source_transaction_audit_passed=transaction_audit_passed,
            query_ms=self.hop,commit_ms=self.commit_ms,window_ms=50,lag_ms=1,inference_beta=self.beta.tolist(),
            warmup_outputs=len(warm),outputs=len(traces),valid_runs=len(selected),failures=failures,injections=injections,
            runtime_audit=self.runtime.audit,model_buffers_unchanged=buffers_before==buffers_after,process_wall_seconds=time.perf_counter()-started,
            runtime_class=type(self.runtime).__name__,runtime_workers=getattr(self.runtime,'runtime_workers',None),
            readout_backend='native_timestamp_no_unused_field' if isinstance(self.runtime,NativeS43Runtime) else ('native_ring_bins_v1' if type(self.runtime).__name__=='SharedS43RuntimeFastReadout' else 'reference_snapshot_bins'),
            model_diagnostics_before=diagnostics_before,model_diagnostics_after=self.diagnostics(),run_diagnostics=run_diagnostics,
            actual_decoder_contract='originalS43 51D state, originalmean-once MANOFK; any inherited coordinate-query metadata mentioning10class/63proxy describes the old coordinateprototype only',
            output_camera_validity=camera,output_camera_definition='alreadydecoded outputjoints: any20finger z<=.03m, decodedjoint0/wrist z<=0; outputquery and committedrow denominators separate; this is not fieldinput previous-query validity or physicalvisibility',
            audit_prefix_overhead_in_cycle=self.audit_prefix,peak_resident_numeric_state_bytes=self.peak_resident_state_bytes,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def source_files(reader,checkpoint,payload):
    paths=list(code_dependencies())+[Path(__file__),Path(__file__).with_name('s43_field_native.py'),PROTOCOL,
        BASE/'stage5_reacquisition_R1_protocol.json',ROOT/'assets/mano_right.npz',Path(payload['base_checkpoint']),Path(payload['base_config'])]+reader.input_paths()
    perturb=Path(__file__).with_name('s43_field_perturb.py')
    if perturb.exists():paths.append(perturb)
    if checkpoint is not None:paths.append(Path(checkpoint))
    return sorted(set(Path(p).resolve() for p in paths))


def save_trace(path,rows,warm,beta):
    def arrays(records):
        return dict(pred=np.asarray([r['pred'] for r in records],np.float32).reshape(-1,51),
            run_id=np.asarray([r['run_id'] for r in records],np.int64),end_ms=np.asarray([r['end_ms'] for r in records],np.int64),
            cutoff_ms=np.asarray([r['cutoff_ms'] for r in records],np.int64),committed=np.asarray([r['committed'] for r in records],bool),
            cycle_ms=np.asarray([r['cycle_ms'] for r in records],np.float64),native_prefix_count=np.asarray([r['native_prefix_count'] for r in records],np.int64),
            camera_finger_nearplane=np.asarray([r['camera_finger_nearplane'] for r in records],bool),camera_root_nonpositive=np.asarray([r['camera_root_nonpositive'] for r in records],bool),
            camera_finger_z_min_m=np.asarray([r['camera_finger_z_min_m'] for r in records],np.float32),camera_finger_z_max_m=np.asarray([r['camera_finger_z_max_m'] for r in records],np.float32),
            local_query_applicable=np.asarray([r.get('local_query_applicable',False) for r in records],bool),
            query_used_mode=np.asarray([r.get('query_used_mode','not_audited') for r in records],dtype='U48'),
            used_uniform_reacquisition=np.asarray([r.get('used_uniform_reacquisition',False) for r in records],bool),
            initial_camera_invalid_count=np.asarray([r.get('initial_camera_invalid_count',-1) for r in records],np.int16),
            actual_2d_valid_samples=np.asarray([r.get('actual_2d_valid_samples',-1) for r in records],np.int32),
            **{key:np.asarray([r.get(key,'') for r in records],dtype='U64') for key in ('sensor_sha256','history_sha256','observation_input_sha256','surface_sha256')})
    a=arrays(rows);a.update({'warmup_'+k:v for k,v in arrays(warm).items()});a['inference_beta']=np.asarray(beta,np.float32)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**a)


def predict(arm,recording,out,*,checkpoint=None,device='cpu',threads=2,hop=50,commit_ms=None,
            root=DATA_ROOT,train_root=TRAIN,engineering=False,stop_ms=None,provided_beta=None,perturbation='none',native_s43=False):
    out=Path(out)
    if out.with_suffix('.npz').exists() or out.with_suffix('.json').exists():raise FileExistsError('fresh output identity required')
    configure(device,threads)
    if sha(PROTOCOL)!=PROTOCOL_SHA:raise RuntimeError('evaluation protocol changed')
    gate=None if engineering else all_weights_gate(train_root)
    if not engineering and arm!='canonical':
        expected=Path(gate['checkpoints'][arm])
        if checkpoint is None:checkpoint=expected
        if Path(checkpoint).resolve()!=expected.resolve():raise ValueError('formal uses fixed final cohort weights only')
    beta=np.zeros(10,np.float32);beta_receipt=None
    if provided_beta is not None:
        beta_receipt=read(provided_beta)
        if beta_receipt.get('condition')!='explicit_known_beta_GT_assisted_diagnostic':raise ValueError('knownshape must be explicitly labelled conditional')
        beta=np.asarray(beta_receipt['external_beta10'],np.float32)
    with InferenceAccessGuard() as guard:
        reader=MeanShapeInputReader(recording,root);model,payload=load_model(arm,checkpoint,device)
        if not engineering and arm!='canonical' and payload.get('schema')!='s43_field_full_training_v1':raise ValueError('formal finalcohort requires new full-execution checkpoint schema')
        files=source_files(reader,checkpoint,payload)
        if provided_beta is not None:files.append(Path(provided_beta).resolve())
        before={str(p):sha(p) for p in files}
        # Actual dependencies, invocation and source identity exist before the
        # first neural output. Hashing is outside service wall and warms OS cache.
        dump(out.with_suffix('.sourcefreeze.json'),dict(schema='s43_field_eval_execution_source_v1',created_ns=time.time_ns(),
            formal=not engineering,allfive_weights_gate=gate,sources=before,arm=arm,recording=recording,
            command=dict(checkpoint=None if checkpoint is None else str(checkpoint),device=device,threads=threads,hop=hop,
                commit_ms=commit_ms or hop,provided_beta=provided_beta,perturbation=perturbation,native_s43=native_s43,stop_ms=stop_ms)))
        runs=None
        if stop_ms is not None:
            if not engineering:raise ValueError('formal evaluation consumes complete preregistered valid runs')
            runs=[(i,(int(a),min(int(b),int(stop_ms)))) for i,(a,b) in enumerate(reader.runs) if int(a)<stop_ms]
        runner=FullStreamRunner(model,payload['runtime_contract'],native_s43=native_s43,beta=beta,
            hop=hop,commit_ms=commit_ms,perturbation=perturbation,audit_prefix=True)
        try:rows,warm,summary=runner.run(reader,recording,runs=runs)
        finally:runner.close()
        save_trace(out.with_suffix('.npz'),rows,warm,beta)
        after={str(p):sha(p) for p in files}
        summary.update(schema='s43_full_field_target_free_trace_v1',arm=arm,engineering=engineering,checkpoint=str(checkpoint),
            prediction_archive=str(out.with_suffix('.npz')),prediction_sha256=sha(out.with_suffix('.npz')),
            sourcefreeze_sha256=sha(out.with_suffix('.sourcefreeze.json')),source_files_unchanged=before==after,
            input_guard=guard.receipt(),shape_condition='beta0' if provided_beta is None else 'explicit_GT_assisted_known_shape',
            known_shape_provenance=beta_receipt,native_s43_without_unused_field=native_s43,
            checkpoint_variant=payload.get('variant'),model_variant=payload.get('model_variant'),R1_protocol_sha256=R1_SHA,
            checkpoint_schema=payload.get('schema','canonical_reference'),execution_variant=payload.get('execution_variant','canonical_reference'),
            artifact_status=payload['serving_status'],
            precision='FP32eval/TF32off/renderFP32/stateFP32',threads=threads,
            actual_threadpools=threadpool_info(),
            output_storage='O(T) prediction log separate from bounded observation/history state; full surfaces decoded+transferred but not stored perframe')
        if before!=after:raise RuntimeError('execution source identity changed')
        dump(out.with_suffix('.json'),summary)
    return summary


def canonical_jobs():
    protocol=read(PROTOCOL);jobs=[]
    for arm in MODELS:
        for recording in protocol['primary']['recordings']:jobs.append(dict(arm=arm,recording=recording,profile='primary',hop=50,commit_ms=50,perturbation='none',known_beta=False))
        for recording in ('zgz_local','zgz_global'):
            jobs.append(dict(arm=arm,recording=recording,profile='known_beta',hop=50,commit_ms=50,perturbation='none',known_beta=True))
            for hop,commit in ((25,25),(100,100),(25,50)):
                jobs.append(dict(arm=arm,recording=recording,profile=f'q{hop}_c{commit}',hop=hop,commit_ms=commit,perturbation='none',known_beta=False))
            for condition in protocol['previous_state']['conditions']:
                jobs.append(dict(arm=arm,recording=recording,profile=condition,hop=50,commit_ms=50,perturbation=condition,known_beta=False))
    return jobs


def evaluation_plan(out,train_root=TRAIN):
    jobs=canonical_jobs();root=Path(out)
    for job in jobs:
        job['out']=str(root/'traces'/job['profile']/job['arm']/job['recording'])
        job['checkpoint']=None if job['arm']=='canonical' else str(Path(train_root)/'train/full'/job['arm']/'final_weights.ckpt')
    result=dict(schema='s43_full_field_evaluation_plan_v1',protocol_sha256=sha(PROTOCOL),models=list(MODELS),jobs=jobs,
        total_traces=len(jobs),primary_traces=36,all_traces_before_GT_pose_scoring=True)
    dump(root/'evaluation_plan.json',result);return result


def validate_plan(plan,gate=None):
    """Exact unique preregistered semantic grid, not a caller's job count."""
    if sha(PROTOCOL)!=PROTOCOL_SHA:raise RuntimeError('evaluation protocol source identity changed')
    fields=('arm','recording','profile','hop','commit_ms','perturbation','known_beta')
    identity=lambda j:tuple(j[k] for k in fields)
    expected={identity(j) for j in canonical_jobs()}
    actual=[identity(j) for j in plan['jobs']]
    if (plan.get('protocol_sha256')!=PROTOCOL_SHA or plan.get('models')!=list(MODELS) or
            len(actual)!=120 or len(set(actual))!=120 or set(actual)!=expected or
            len({str(Path(j['out']).resolve()) for j in plan['jobs']})!=120):raise RuntimeError('exact unique fixed120 evaluation job grid required')
    for job in plan['jobs']:
        if job['arm']=='canonical':
            if job.get('checkpoint') is not None:raise RuntimeError('canonical source checkpoint is fixed by serving factory')
        elif gate is not None and Path(job['checkpoint']).resolve()!=Path(gate['checkpoints'][job['arm']]).resolve():raise RuntimeError('plan checkpoint differs fromcompletefixedlastcohort')
    return True


def validate_cohort(plan,profiles=None,*,root=DATA_ROOT):
    """No target reads: verify every saved file/hash/address before any metric."""
    validate_plan(plan)
    jobs=[j for j in plan['jobs'] if profiles is None or j['profile'] in profiles];loaded={};reference={};sources={};readers={};frozen_sources={}
    if not jobs:raise ValueError('empty score cohort')
    for job in jobs:
        p=Path(job['out']);d=read(p.with_suffix('.json'));archive=p.with_suffix('.npz')
        if (d.get('engineering') or d.get('status')!='COMPLETE' or not d.get('source_files_unchanged') or
                not d.get('source_fully_consumed') or not d.get('model_buffers_unchanged') or d['prediction_sha256']!=sha(archive)):raise RuntimeError('incomplete/failed/mutated formal trace: '+str(p))
        if sha(p.with_suffix('.sourcefreeze.json'))!=d['sourcefreeze_sha256']:raise RuntimeError('execution source manifest identity differs')
        execution=read(p.with_suffix('.sourcefreeze.json'));command=execution['command']
        if (execution.get('formal') is not True or execution.get('arm')!=job['arm'] or execution.get('recording')!=job['recording'] or
                command.get('hop')!=job['hop'] or command.get('commit_ms')!=job['commit_ms'] or
                command.get('perturbation')!=job['perturbation'] or command.get('native_s43') is not False or command.get('stop_ms') is not None):raise RuntimeError('actualtrace execution condition differs fromfixedjob')
        for path,expected_sha in execution['sources'].items():
            if path in frozen_sources and frozen_sources[path]!=expected_sha:raise RuntimeError('paired execution source changed between jobs: '+path)
            frozen_sources[path]=expected_sha
        checkpoint=command.get('checkpoint')
        if (job['checkpoint'] is None)!=(checkpoint is None) or (checkpoint is not None and Path(checkpoint).resolve()!=Path(job['checkpoint']).resolve()):raise RuntimeError('actualtrace checkpoint differs fromfixedjob')
        if d['arm']!=job['arm'] or d['recording']!=job['recording'] or d['query_ms']!=job['hop'] or d['commit_ms']!=job['commit_ms']:raise RuntimeError('saved trace execution identity differs fromfixedplan')
        if job['arm']!='canonical' and (d.get('checkpoint_schema')!='s43_field_full_training_v1' or d.get('execution_variant')!=EXECUTION_VARIANT):raise RuntimeError('actual formal weight execution schema differs')
        guard=d['input_guard']
        if guard['blocked_aux_beta_attempts'] or guard['blocked_pose_meta_attempts'] or guard['aux_betas_accessed'] or guard['pose_metadata_opened']:raise RuntimeError('targetfree input firewall rejected access')
        with np.load(archive,allow_pickle=False) as z:
            a={k:z[k].copy() for k in z.files if not k.startswith('warmup_')}
            w={k.removeprefix('warmup_'):z[k].copy() for k in z.files if k.startswith('warmup_')}
        n=len(a['pred'])
        if a['pred'].shape!=(n,51) or not np.isfinite(a['pred']).all() or n!=d['outputs']:raise ValueError('saved prediction shape/count')
        for key in ('local_query_applicable','query_used_mode','used_uniform_reacquisition','initial_camera_invalid_count','actual_2d_valid_samples'):
            if a[key].shape!=(n,):raise ValueError('saved query-mode scalar shape/count')
        if job['arm'].startswith('field_'):
            if (not a['local_query_applicable'].all() or np.any(a['initial_camera_invalid_count']<0) or
                    np.any(a['initial_camera_invalid_count']>20) or np.any(a['actual_2d_valid_samples']<=0) or
                    np.any(a['actual_2d_valid_samples']>35280)):raise ValueError('invalid field query-mode diagnostics')
        elif a['local_query_applicable'].any():raise ValueError('legacy baseline must not claimlocal query metadata')
        if a['inference_beta'].shape!=(10,) or not np.isfinite(a['inference_beta']).all():raise ValueError('saved shape identity')
        if not job['known_beta'] and not np.array_equal(a['inference_beta'],np.zeros(10,np.float32)):raise ValueError('beta0 formal trace contains externalshape')
        if bool(command.get('provided_beta'))!=job['known_beta'] or d['shape_condition']!=('explicit_GT_assisted_known_shape' if job['known_beta'] else 'beta0'):raise RuntimeError('explicit shape-condition identity differs')
        if not np.array_equal(a['inference_beta'],np.asarray(d['inference_beta'],np.float32)):raise ValueError('saved beta archive/summary differ')
        key=(job['profile'],job['recording']);addresses=np.column_stack((a['run_id'],a['end_ms']))
        if job['recording'] not in readers:readers[job['recording']]=MeanShapeInputReader(job['recording'],root)
        reader=readers[job['recording']]
        if job['known_beta']:
            provided=Path(command['provided_beta']);beta=read(provided)
            if (sha(provided)!=execution['sources'].get(str(provided.resolve())) or beta!=d['known_shape_provenance'] or
                    beta.get('recording')!=job['recording'] or beta.get('source_member')!='betas' or
                    Path(beta['source_file']).resolve()!=Path(str(reader.base)+'_aux.npz').resolve() or
                    beta['source_sha256']!=sha(beta['source_file']) or not np.array_equal(np.asarray(beta['external_beta10'],np.float32),a['inference_beta'])):raise RuntimeError('knownshape provider/source/vector identity differs')
        if job['perturbation']=='none':
            if d['injections']:raise RuntimeError('nominal trace unexpectedly perturbed')
        else:
            expected_runs=[i for i,(aa,bb) in enumerate(reader.runs) if int(bb)-int(aa)>=100]
            if ([x['run_id'] for x in d['injections']]!=expected_runs or
                    any(x['condition']!=job['perturbation'] or x['sensor_before_sha256']!=x['sensor_after_sha256'] for x in d['injections'])):raise RuntimeError('onceperrun warmup perturbation contract differs')
        expected=np.asarray([(i,T-1) for i,(aa,bb) in enumerate(reader.runs) for T in range(int(aa)+100+job['hop'],int(bb)+1,job['hop'])],np.int64).reshape(-1,2)
        if not np.array_equal(addresses,expected):raise AssertionError('trace must contain every declaredvalidrun output address')
        expected_warm=np.asarray([(i,T-1) for i,(aa,bb) in enumerate(reader.runs) for T in (int(aa)+50,int(aa)+100) if T<=int(bb)],np.int64).reshape(-1,2)
        if (not np.array_equal(np.column_stack((w['run_id'],w['end_ms'])),expected_warm) or
                w['pred'].shape!=(len(expected_warm),51) or not np.isfinite(w['pred']).all() or
                len(expected_warm)!=d['warmup_outputs'] or not w['committed'].all()):raise AssertionError('own2x50 warmup address/finite/commit contract differs')
        if not np.array_equal(w['cutoff_ms'],w['end_ms']+1) or not np.array_equal(a['cutoff_ms'],a['end_ms']+1):raise AssertionError('native1ms output lag differs')
        committed=np.asarray([(int(T)-(int(reader.runs[int(r),0])+100))%job['commit_ms']==0 for r,T in zip(a['run_id'],a['cutoff_ms'])],bool)
        if not np.array_equal(a['committed'],committed):raise AssertionError('pure extra query unexpectedly committed or mandatorycommit skipped')
        expected_count=sum(int(reader.offsets[int(bb)])-int(reader.offsets[int(aa)]) for aa,bb in reader.runs)
        if d['native_input_count']!=expected_count:raise AssertionError('incomplete validrun native source consumption')
        source=(d['native_input_count'],d['native_input_sha256'])
        if job['recording'] in sources and sources[job['recording']]!=source:raise AssertionError('cadence/condition/model source count/digest differ')
        sources[job['recording']]=source
        if key in reference and not np.array_equal(reference[key],addresses):raise AssertionError('same profile cohort addresses differ')
        reference[key]=addresses;loaded[(job['profile'],job['arm'],job['recording'])]=(job,a,d)
    if len(loaded)!=len(jobs):raise RuntimeError('cohort keys duplicated or missing')
    for path,expected_sha in frozen_sources.items():
        if sha(path)!=expected_sha:raise RuntimeError('saved execution dependency/input changed before score: '+path)
    return loaded


def posthoc_metrics(pred,gt,pred_beta,gt_beta):
    mano=_mano();pj,pv=fk(mano,pred,pred_beta);gj,gv=fk(mano,gt,gt_beta)
    norm=lambda x:np.linalg.norm(x,axis=-1).mean(-1)*1000.
    values=dict(mpjpe_abs_mm=norm(pj-gj),mpvpe_abs_mm=norm(pv-gv),
        mpjpe_ra_mm=norm((pj-pj[:,:1])-(gj-gj[:,:1])),mpvpe_ra_mm=norm((pv-pv[:,:1])-(gv-gv[:,:1])),
        mpvpe_joint_wrist_aligned_mm=norm((pv-pj[:,:1])-(gv-gj[:,:1])),root_error_mm=np.linalg.norm(pj[:,0]-gj[:,0],axis=-1)*1000.)
    joint_error=np.linalg.norm((pj-pj[:,:1])-(gj-gj[:,:1]),axis=-1)*1000.
    for finger,indices in (('thumb',(1,2,3,4)),('index',(5,6,7,8)),('middle',(9,10,11,12)),('ring',(13,14,15,16)),('pinky',(17,18,19,20))):
        values[finger+'_j0_mm']=joint_error[:,indices].mean(-1)
    if not all(np.isfinite(x).all() for x in values.values()):raise ArithmeticError('invalid metric; retain failed evidence')
    return values


def read_score_targets(base,ends):
    base=Path(base)
    with np.load(str(base)+'_aux.npz',allow_pickle=False) as aux:beta=np.asarray(aux['betas'],np.float32).reshape(10)
    path=Path(str(base)+'.meta')
    with path.open('rb') as f:n,=struct.unpack('<i',f.read(4))
    if n!=51:raise ValueError('score target must be51D')
    dtype=np.dtype([('data','<f8',51),('m0','u1'),('m1','u1')])
    if (path.stat().st_size-4)%dtype.itemsize:raise ValueError('truncated score metadata')
    data=np.memmap(path,dtype=dtype,mode='r',offset=4);selected=data[ends]
    if np.any(selected['m0']!=4) or np.any(selected['m1']!=13):raise ValueError('invalid score markers')
    target=np.asarray(selected['data'],np.float32).copy()
    if not np.isfinite(target).all() or not np.isfinite(beta).all():raise ValueError('nonfinite score targets')
    return target,beta


def query_mode_groups(trace,values):
    def group(mask):
        return dict(n=int(mask.sum()),metrics={k:float(v[mask].astype(np.float64).mean()) for k,v in values.items()} if mask.any() else None)
    applicable=trace['local_query_applicable'];fallback=trace['used_uniform_reacquisition']
    return dict(scope='descriptive/endogenous query-mode strata fromdeployedstate metadata, noGTpose selection andno causal fallback-effect claim; knownbeta profile retainsitsGT-assistedshapecondition',
        by_used_mode={str(mode):group(trace['query_used_mode']==mode) for mode in np.unique(trace['query_used_mode'])},
        fallback=group(applicable&fallback),normal_local=group(applicable&~fallback),
        initial_camera_invalid=group(applicable&(trace['initial_camera_invalid_count']>0)),
        local_query_applicable_rows=int(applicable.sum()),all_outputs=len(applicable))


def score(plan_path,out,*,train_root=TRAIN,root=DATA_ROOT):
    gate=all_weights_gate(train_root);plan=read(plan_path)
    validate_plan(plan,gate)
    loaded=validate_cohort(plan,root=root) # all120 predictions before any target pose access
    verified_ns=time.time_ns();first_gt_ns=time.time_ns();reports={};perfiles={};targets={}
    for (profile,arm,recording),(job,a,d) in loaded.items():
        reader=MeanShapeInputReader(recording,root);cache=(recording,a['end_ms'].tobytes())
        if cache not in targets:targets[cache]=read_score_targets(reader.base,a['end_ms'])
        gt,gtbeta=targets[cache];values=posthoc_metrics(a['pred'],gt,a['inference_beta'],gtbeta)
        dest=Path(out)/'per_frame'/profile/arm/(recording+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,run_id=a['run_id'],end_ms=a['end_ms'],gt=gt,**values)
        key='/'.join((profile,arm,recording));reports[key]=dict(n=len(gt),**{k:float(v.astype(np.float64).mean()) for k,v in values.items()})
        perfiles[(profile,arm,recording)]=(a,values)
    # Reuse the frozen ALL masks by exact addresses; no new GT thresholds/tubes.
    strata={}
    for path,expected in read(PROTOCOL)['local_finger_strata']['paths'].items():
        p=ROOT/path
        if sha(p)!=expected:raise RuntimeError('old frozen strata identity differs')
        recording=p.name.removesuffix('_nominal_motion_strata.npz')
        with np.load(p,allow_pickle=False) as z:old={k:z[k] for k in z.files}
        lookup={(int(r),int(t)):i for i,(r,t) in enumerate(zip(old['run_id'],old['end_ms']))}
        masks=[k for k,v in old.items() if v.dtype==bool and v.shape==old['end_ms'].shape]
        for (profile,arm,rec),(a,values) in perfiles.items():
            if rec!=recording:continue
            address=[lookup.get((int(r),int(t)),-1) for r,t in zip(a['run_id'],a['end_ms'])];matched=np.asarray(address)>=0
            item=dict(matched=int(matched.sum()),unmatched=int((~matched).sum()),masks={})
            for mask in masks:
                selected=np.zeros(len(address),bool);selected[matched]=old[mask][np.asarray(address)[matched]]
                item['masks'][mask]=dict(n=int(selected.sum()),metrics={k:float(v[selected].astype(np.float64).mean()) for k,v in values.items()} if selected.any() else None)
            strata['/'.join((profile,arm,rec))]=item
    common={}
    for arm in MODELS:
        for rec in ('zgz_local','zgz_global'):
            profiles=('primary','q25_c25','q100_c100');maps={p:{(int(r),int(t)):i for i,(r,t) in enumerate(zip(perfiles[(p,arm,rec)][0]['run_id'],perfiles[(p,arm,rec)][0]['end_ms']))} for p in profiles}
            addresses=sorted(set.intersection(*(set(m) for m in maps.values())))
            common[arm+'/'+rec]=dict(n=len(addresses),profiles={p:{k:float(v[[maps[p][x] for x in addresses]].astype(np.float64).mean()) for k,v in perfiles[(p,arm,rec)][1].items()} for p in profiles})
    primary={}
    for arm in MODELS:
        def aggregate(recordings):
            vv=[perfiles[('primary',arm,r)][1] for r in recordings]
            return dict(n=sum(len(v['mpjpe_ra_mm']) for v in vv),**{k:float(np.concatenate([v[k] for v in vv]).astype(np.float64).mean()) for k in vv[0]})
        primary[arm]={'local':aggregate(['zgz_local']),'global':aggregate(['zgz_global']),'overall':aggregate(['zgz_local','zgz_global'])}
    fixed_commit={}
    for arm in MODELS:
        for rec in ('zgz_local','zgz_global'):
            normal=perfiles[('primary',arm,rec)][0];extra=perfiles[('q25_c50',arm,rec)][0]
            lookup={(int(r),int(t)):i for i,(r,t) in enumerate(zip(extra['run_id'],extra['end_ms']))}
            idx=np.asarray([lookup[(int(r),int(t))] for r,t in zip(normal['run_id'],normal['end_ms'])],np.int64)
            params=normal['pred']-extra['pred'][idx]
            j0,v0=fk(_mano(),normal['pred'],normal['inference_beta']);j1,v1=fk(_mano(),extra['pred'][idx],extra['inference_beta'])
            maxj=float(np.linalg.norm(j0-j1,axis=-1).max()*1000);maxv=float(np.linalg.norm(v0-v1,axis=-1).max()*1000)
            exact={k:bool(np.array_equal(normal[k],extra[k][idx])) for k in ('sensor_sha256','history_sha256','observation_input_sha256')}
            # Matching committed outputs are exactly the history knots used by
            # the next query. Report byte hashes, but apply the preregistered
            # FP32 pose tolerance to history values rather than calling a
            # one-bit float difference a categorical history mutation.
            normal_archive=np.load(Path(loaded[('primary',arm,rec)][0]['out']).with_suffix('.npz'),allow_pickle=False)
            extra_archive=np.load(Path(loaded[('q25_c50',arm,rec)][0]['out']).with_suffix('.npz'),allow_pickle=False)
            try:warm_diff=float(np.abs(normal_archive['warmup_pred']-extra_archive['warmup_pred']).max())
            finally:normal_archive.close();extra_archive.close()
            history_maxdiff=max(float(np.abs(params).max()),warm_diff)
            fixed_commit[arm+'/'+rec]=dict(common_outputs=len(idx),exact_prefix_checks=exact,
                full_surface_output_hash_exact=bool(np.array_equal(normal['surface_sha256'],extra['surface_sha256'][idx])),
                history_committed_pose_parameter_max_abs=history_maxdiff,
                parameter_max_abs=float(np.abs(params).max()),translation_max_abs_m=float(np.abs(params[:,:3]).max()),
                axis_angle_max_abs_rad=float(np.abs(params[:,3:]).max()),joint_max_distance_mm=maxj,vertex_max_distance_mm=maxv,
                declared_atol=dict(translation_m=1e-6,axis_angle_rad=1e-6,surface_mm=.001),
                within_declared_numeric_tolerance=exact['sensor_sha256'] and exact['observation_input_sha256'] and bool(history_maxdiff<=1e-6 and max(maxj,maxv)<=.001))
    result=dict(schema='s43_full_field_saved_trace_report_v1',status='COMPLETE',protocol_sha256=PROTOCOL_SHA,allfive_weights_gate=gate,
        all_saved_traces_verified_ns=verified_ns,first_target_pose_access_ns=first_gt_ns,all_predictions_saved_before_GT_pose=True,
        known_GT_shape_condition='explicitprovidedshape onlyafter36beta0traces; this isGT-assisteddiagnostic, not a claimall120inputsareGT-free',
        primary_heldperson=primary,all_recordings_profiles=reports,common_cadence_grid_summary=common,fixed_commit_extra_query_parity=fixed_commit,old_exact_address_strata=strata,
        output_camera_validity={'/'.join(key):entry[2]['output_camera_validity'] for key,entry in loaded.items()},
        query_mode_strata={'/'.join(key):query_mode_groups(a,v) for key,(a,v) in perfiles.items()},
        input_query_diagnostics={'/'.join(key):dict(before=entry[2].get('model_diagnostics_before'),after=entry[2].get('model_diagnostics_after'),per_run=entry[2].get('run_diagnostics')) for key,entry in loaded.items()},
        metrics='J0 jointwrist / V0 vertex0 separately, no Procrustes, recursiveRA frameweighted local+global; absoluteJ/V and wristV retained',
        limitation='single seed; ycy/ylf trained diagnostics; zgz historically developed heldperson; knownbeta GT-assistedconditional; W200/S39stack not supported by newW50frontend')
    dump(Path(out)/'results.json',result);return result


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('plan');a.add_argument('--out',required=True)
    a=sub.add_parser('predict');a.add_argument('--arm',choices=MODELS,required=True);a.add_argument('--recording',required=True);a.add_argument('--out',required=True)
    a.add_argument('--checkpoint');a.add_argument('--device',default='cpu');a.add_argument('--threads',type=int,default=2);a.add_argument('--hop',type=int,default=50);a.add_argument('--commit-ms',type=int)
    a.add_argument('--root',default=str(DATA_ROOT));a.add_argument('--train-root',default=str(TRAIN));a.add_argument('--engineering',action='store_true');a.add_argument('--stop-ms',type=int)
    a.add_argument('--provided-beta');a.add_argument('--perturbation',default='none');a.add_argument('--native-s43',action='store_true')
    a=sub.add_parser('score');a.add_argument('--plan',required=True);a.add_argument('--out',required=True);a.add_argument('--train-root',default=str(TRAIN));a.add_argument('--root',default=str(DATA_ROOT))
    args=vars(p.parse_args());command=args.pop('command')
    if command=='plan':evaluation_plan(**args)
    elif command=='predict':
        try:predict(**args)
        except BaseException as exc:
            dump(Path(args['out']).with_suffix('.failure.json'),dict(status='FAILED',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),no_hold_or_clamp=True))
            raise
    else:score(args.pop('plan'),**args)


if __name__=='__main__':main()
