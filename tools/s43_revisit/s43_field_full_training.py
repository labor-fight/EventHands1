"""Full-only pure execution revision of the frozen R1 trainer.
The copied training mathematics/budget is unchanged; native-ring integer
readout avoids redundant snapshot/coarse/stack copies. Old pilot identity is
explicitly bridged and remains immutable. Full runs still start canonical.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
BASE_OUT=ROOT/'outputs/s43_revisit_20261028'
OUT=BASE_OUT/'s43_full_field_R1'
R1_PROTOCOL=BASE_OUT/'stage5_reacquisition_R1_protocol.json'
R1_SHA='0f865d0afabc2d998296b44ceea86bcf0853e3219740c4a3dfd5ae292219abef'
MODEL_VARIANT='invalid_geometry_uniform_r1'
EXECUTION_VARIANT='native_ring_readout_full_v1'
PILOT_FREEZE=OUT/'engineering/actual_sourcefreeze.json'
PILOT_FREEZE_SHA='fa2fb3d7a1f6159b96279f21db2d106d014809192dcf5a719faed57087b91249'
PROTOCOL=BASE_OUT/'stage5_s43_full_protocol.json'
PROTOCOL_SHA='0254d1a0afaf7fc437f88c434c68e4a84b72a18b366cf64c1be3724501659852'
ADDENDUM=BASE_OUT/'stage5_box_time_addendum.json'
ADDENDUM_SHA='88fa2a9311744f99de218d15cfb2baa9f63aaab83623c079f6722aa1589398fe'
ARMS=('s43','field_box','field_static_collapse','field_static','field_box_collapse')
FULL_SUBJECTS=('ch','lfz','lpc','lr','ly','lyh','lyq','ycy','ylf')
FIT_SUBJECTS=FULL_SUBJECTS[:7]
STAGES=dict(small=dict(target=32768,batch=8,clip=4,lr=.000125,warmup=1024),
            full=dict(target=6144000,batch=64,clip=4,lr=.001,warmup=65536))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
      default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v))+'\n',encoding='utf-8')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def tensor_digest(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        a=value.detach().cpu().contiguous().numpy();h.update(key.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def arm_contract(arm):
    if arm not in ARMS:raise ValueError('unknown Stage5 arm')
    return dict(runtime_arm='legacy_s43' if arm=='s43' else 'hand_box' if arm.startswith('field_box') else 'static',
      temporal_collapse=arm.endswith('_collapse'),tensor_readout=True,state_fp32=True,allow_external_beta=False,
      model_variant='canonical_s43' if arm=='s43' else MODEL_VARIANT,geometry_workers=1,runtime_workers=1,readout_backend='native_ring_bins_v1')


def build_model(arm,device='cpu',*,base_checkpoint=None,base_config=None,seed=3407):
    """Serving-safe factory: no data/protocol/target files are opened here."""
    from tools.s43_integration.model import load_legacy,DEFAULT_CKPT,DEFAULT_CFG
    from .s43_field_model_reacquire import load_s43_field_reacquire_model
    from .s43_field_runtime import install_state_fp32
    torch.manual_seed(int(seed));contract=arm_contract(arm)
    kwargs=dict(base_checkpoint=base_checkpoint or DEFAULT_CKPT,base_config=base_config or DEFAULT_CFG)
    if arm=='s43':model,_=load_legacy(kwargs['base_checkpoint'],kwargs['base_config'],device)
    else:model,_=load_s43_field_reacquire_model(arm=contract['runtime_arm'],use_prev_mlp=True,geometry_workers=1,device=device,**kwargs)
    base=getattr(model,'base',model);base.render_fp32=True
    install_state_fp32(model)
    for p in model.parameters():p.requires_grad_(True)
    return model,contract


def load_weights(checkpoint,device='cpu'):
    """Weights-only/full checkpoint loader; never opens dataset or sourcefreeze."""
    saved=torch.load(checkpoint,map_location='cpu')
    if saved.get('schema')!='s43_field_full_training_v1' or not saved.get('STATE_FP32') or not saved.get('RENDER_FP32') or not saved.get('LOSS_FP32'):
        raise ValueError('explicit R1 FP32-state/render checkpoint required')
    if saved.get('execution_variant')!=EXECUTION_VARIANT or saved.get('pilot_sourcefreeze_sha256')!=PILOT_FREEZE_SHA:raise ValueError('explicit full-execution identity required')
    if saved.get('variant')!='reacquisition_R1' or saved.get('R1_protocol_sha256')!=R1_SHA:raise ValueError('R1 constructor identity required')
    if saved.get('model_variant')!=arm_contract(saved['arm'])['model_variant']:raise ValueError('wrong R1 model variant')
    model,contract=build_model(saved['arm'],device,base_checkpoint=saved['base_checkpoint'],
      base_config=saved['base_config'],seed=saved['seed'])
    if saved['runtime_contract']!=contract:raise ValueError('checkpoint runtime semantics differ')
    model.load_state_dict(saved['model_state_dict'],strict=True)
    return model,saved


class OutputScheduler:
    """One fixed cosine schedule in actual supervised-output units."""
    def __init__(self,optimizer,*,target,warmup,base_lr,floor=.02):
        self.optimizer=optimizer;self.target=int(target);self.warmup=int(warmup)
        self.base_lr=float(base_lr);self.floor=float(floor);self.outputs=0;self.updates=0
    def multiplier(self,outputs):
        if outputs<=self.warmup:return outputs/self.warmup
        p=min(max((outputs-self.warmup)/(self.target-self.warmup),0.),1.)
        return self.floor+(1-self.floor)*.5*(1+math.cos(math.pi*p))
    def prepare(self,outputs):
        if not self.outputs<outputs<=self.target:raise ValueError('scheduler output clock')
        lr=self.base_lr*self.multiplier(outputs)
        for group in self.optimizer.param_groups:group['lr']=lr
        return lr
    def advance(self,outputs):self.outputs=int(outputs);self.updates+=1
    def state_dict(self):return {k:getattr(self,k) for k in ('target','warmup','base_lr','floor','outputs','updates')}
    def load_state_dict(self,state):
        if any(state[k]!=getattr(self,k) for k in ('target','warmup','base_lr','floor')):raise ValueError('scheduler identity')
        self.outputs=int(state['outputs']);self.updates=int(state['updates'])


def original_objective(raw_loss,model):
    """Exactly original FP32 log10, with no new clamp or loss component."""
    base=getattr(model,'base',model);value=raw_loss.float()
    return value.log10() if base.log10_loss else value


def average_phase_gradients(parameters,phases):
    if phases<1:raise ValueError('positive supervised phase count')
    for p in parameters:
        if p.grad is not None:p.grad.div_(phases)


def code_dependencies():
    from tools.s43_rel.training import _source_files
    files=dict(_source_files())
    names=('tools/s43_revisit/s43_field_full_training.py','tools/s43_revisit/s43_field_full_queue.py',
      'tools/s43_revisit/runtime_fastreadout.py','tools/s43_revisit/field_tensor_fast.py',
      'tools/s43_revisit/s43_field_training_R1.py','tools/s43_revisit/s43_field_queue_R1.py',
      'tools/s43_revisit/s43_field_training.py','tools/s43_revisit/s43_field_queue.py',
      'tools/s43_revisit/s43_field_runtime_R1.py','tools/s43_revisit/s43_field_model_reacquire.py',
      'tools/s43_revisit/field_core.py','tools/s43_revisit/field_tensor.py',
      'tools/s43_revisit/field_query.py','tools/s43_revisit/witness_geometry.py',
      'tools/s43_revisit/s43_field_model.py','tools/s43_revisit/s43_field_data.py',
      'tools/s43_revisit/s43_field_runtime.py','tools/s43_rel/mean_shape_replay.py',
      'assets/mano_right.npz')
    return dict(files,**{str((ROOT/n).resolve()):sha(ROOT/n) for n in names})


def verify_frozen(freeze,config=None,*,full_input_check=False):
    from .s43_field_training_R1 import verify_frozen as verify_pilot_source
    if sha(PILOT_FREEZE)!=PILOT_FREEZE_SHA:raise RuntimeError('old pilot source identity changed')
    verify_pilot_source(PILOT_FREEZE)
    f=read(freeze)
    if f['schema']!='s43_field_stage5_full_execution_actual_sourcefreeze_v1' or sha(PROTOCOL)!=PROTOCOL_SHA or sha(ADDENDUM)!=ADDENDUM_SHA or sha(R1_PROTOCOL)!=R1_SHA:
        raise RuntimeError('Stage5 source/protocol identity')
    if f['code']!=code_dependencies():raise RuntimeError('frozen executable source changed')
    for p,h in {**f['controls'],**f['configs'],**f['engineering']}.items():
        if sha(p)!=h:raise RuntimeError('frozen control/config/engineering changed: '+p)
    if config is not None and f['configs'].get(str(Path(config).resolve()))!=sha(config):raise RuntimeError('unfrozen training config')
    if full_input_check:
        for p,h in read(f['input_manifest'])['files'].items():
            if sha(p)!=h:raise RuntimeError('frozen data changed: '+p)
    return f


def init_configs():
    from tools.s43_br.data import DATA_ROOT
    from tools.s43_integration.model import DEFAULT_CKPT,DEFAULT_CFG
    for stage,spec in STAGES.items():
        if stage!='full':continue
        for arm in ARMS:
            file=OUT/'configs/full_fastreadout'/(arm+'.json')
            if file.exists():raise FileExistsError(file)
            dump(file,dict(schema='s43_field_stage5_full_execution_config_v1',execution_variant=EXECUTION_VARIANT,stage=stage,arm=arm,seed=3407,
              data_root=str(DATA_ROOT),base_checkpoint=str(DEFAULT_CKPT),base_config=str(DEFAULT_CFG),
              supervised_outputs=spec['target'],batch_size=spec['batch'],clip_steps=spec['clip'],
              learning_rate=spec['lr'],warmup_supervised_outputs=spec['warmup'],cosine_floor=.02,grad_clip=10.,
              precision='bf16',BN_train=True,STATE_FP32=True,RENDER_FP32=True,LOSS_FP32=True,
              transport_chunk_ms=50,cpu_threads=8,save_every_outputs=8192 if stage=='small' else 262144,
              subjects=list(FIT_SUBJECTS if stage=='small' else FULL_SUBJECTS),
              runtime_contract=arm_contract(arm),fresh_canonical_init=True,artificial_pose_augmentation=False))


def input_manifest(dest):
    from .s43_field_data import TargetFreeRunLanes
    from tools.s43_rel.mean_shape_replay import InferenceAccessGuard
    with InferenceAccessGuard() as guard:schedule=TargetFreeRunLanes(subjects=FULL_SUBJECTS,batch_size=1)
    files={schedule.root/'splits_semkine.json'};trials=[]
    for name,reader in schedule.readers.items():
        for suffix in ('_events.npy','_offsets.npy','_aux.npz','_tsub.npy','.meta'):files.add(Path(str(reader.base)+suffix))
        counts=np.diff(np.asarray(reader.offsets,np.int64));window=np.asarray(reader.offsets[50:],np.int64)-np.asarray(reader.offsets[:-50],np.int64)
        trials.append(dict(recording=name,events=len(reader.events),duration_ms=len(counts),runs=reader.runs.tolist(),
          K_source=reader.K_cam.tolist(),max_native_ms_events=int(counts.max(initial=0)),max_any50ms_events=int(window.max(initial=0))))
    if len(trials)!=72:raise RuntimeError('Stage5 original nine-subject/72trial contract changed')
    dump(dest,dict(schema='s43_field_stage5_input_identity_v1',trials=trials,subjects=list(FULL_SUBJECTS),
      files={str(p.resolve()):sha(p) for p in sorted(files)},input_guard=guard.receipt(),
      GT_labels_semantically_read=False,meta_tsub_aux_beta_bytes_hashed_only=True,data=schedule.manifest()))


def sourcefreeze(dest,engineering,commands):
    if Path(dest).exists():raise FileExistsError(dest)
    if sha(PROTOCOL)!=PROTOCOL_SHA or sha(ADDENDUM)!=ADDENDUM_SHA or sha(R1_PROTOCOL)!=R1_SHA:raise RuntimeError('wrong preregistration')
    controls={str(PILOT_FREEZE.resolve()):PILOT_FREEZE_SHA,str(PROTOCOL.resolve()):PROTOCOL_SHA,str(ADDENDUM.resolve()):ADDENDUM_SHA,str(R1_PROTOCOL.resolve()):R1_SHA}
    for name,h in read(R1_PROTOCOL)['prior_identity'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('preserve failed attempt0 identity: '+name)
        controls[str((ROOT/name).resolve())]=h
    for name,h in read(PROTOCOL)['prior_identity'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('preserve prior negative artifact: '+name)
        controls[str((ROOT/name).resolve())]=h
    manifest=OUT/'engineering/input_manifest.json'
    if not manifest.exists():raise RuntimeError('freeze real data identity first')
    controls[str(manifest.resolve())]=sha(manifest)
    extension=OUT/'engineering/full_execution_extension.json'
    if not extension.exists():raise RuntimeError('implementation clarification required')
    controls[str(extension.resolve())]=sha(extension);checks={}
    for p in engineering:
        r=read(p)
        if not (r.get('status') in ('PASS','PASS_EXACT_READOUT_PRED_CPU_GRAD_GPU_GRAD_MEASURED') or r.get('unit_tests_passed') is True):raise RuntimeError('engineering not PASS')
        checks[str(Path(p).resolve())]=sha(p)
    if not checks:raise RuntimeError('engineering required')
    configs={str((OUT/'configs/full_fastreadout'/(arm+'.json')).resolve()):sha(OUT/'configs/full_fastreadout'/(arm+'.json')) for arm in ARMS}
    canonical=read(OUT/'configs/full_fastreadout/s43.json')
    expected=read(BASE_OUT/'s43_field_model_engineering/cpu_receipt.json')
    for key in ('base_checkpoint','base_config'):
        if sha(canonical[key])!=expected[key+'_sha256']:raise RuntimeError('original canonical identity changed')
        controls[str(Path(canonical[key]).resolve())]=sha(canonical[key])
    dump(dest,dict(schema='s43_field_stage5_full_execution_actual_sourcefreeze_v1',created_utc=datetime.now(timezone.utc).isoformat(),
      protocol_sha256=PROTOCOL_SHA,addendum_sha256=ADDENDUM_SHA,R1_protocol_sha256=R1_SHA,code=code_dependencies(),controls=controls,
      configs=configs,engineering=checks,input_manifest=str(manifest.resolve()),commands=commands,
      pilot_sourcefreeze_sha256=PILOT_FREEZE_SHA,execution_variant=EXECUTION_VARIANT,
      pilot_authorized=False,full_only_after_all5_pilot_engineering_pass=True,new_task_scores_seen=False))


def phase_digest(h,batch):
    h.update(batch.token.encode())  # token hashes actual immutable native packets incl. real releases.
    for item in batch:
        if not item.get('valid',True) or item.get('padding',False):continue
        for key in ('K_source','beta_input'):
            a=np.asarray(item[key]);h.update(key.encode());h.update(a.tobytes())
        if item.get('reset'):h.update(np.asarray(item['initial_state'],np.float32).tobytes())


def observation_stats(model):
    value=getattr(model,'observation_stats',{})
    return value() if callable(value) else value


def model_metadata(model,arm,*,trained_outputs=0):
    base=getattr(model,'base',model)
    values=model.metadata() if hasattr(model,'metadata') else dict(total=sum(p.numel() for p in model.parameters()),
      trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),old_LNES_control=True)
    values.update(execution_variant=EXECUTION_VARIANT,readout_backend='native_ring_bins_v1',pilot_sourcefreeze_sha256=PILOT_FREEZE_SHA,variant='reacquisition_R1',model_variant='canonical_s43' if arm=='s43' else MODEL_VARIANT,
      R1_protocol_sha256=R1_SHA,runtime_workers=1,logical_arm=arm,trained=bool(trained_outputs),actual_supervised_outputs=trained_outputs,
      STATE_FP32=True,RENDER_FP32=base.render_fp32,LOSS_FP32=True,BN_training=True,
      mixed_precision_note='common explicitFP32state/render/loss adapter; not bitwise oldwholeforwardBF16 trajectory')
    return values


def save_checkpoint(output,model,runtime,schedule,supervision,optimizer,scheduler,cfg,config,freeze,source_chain,step,elapsed):
    base=getattr(model,'base',model);done=scheduler.outputs
    state=dict(schema='s43_field_full_training_v1',arm=cfg['arm'],stage=cfg['stage'],seed=cfg['seed'],
      supervised_outputs=done,optimizer_updates=step,base_checkpoint=cfg['base_checkpoint'],base_config=cfg['base_config'],
      runtime_contract=cfg['runtime_contract'],STATE_FP32=True,RENDER_FP32=True,LOSS_FP32=True,
      model_state_dict=model.state_dict(),config=cfg,config_sha256=sha(config),sourcefreeze_sha256=sha(freeze),
      model_metadata=model_metadata(model,cfg['arm'],trained_outputs=done),source_chain=source_chain,
      execution_variant=EXECUTION_VARIANT,pilot_sourcefreeze_sha256=PILOT_FREEZE_SHA,pilot_audit_sha256=sha(OUT/'audits/pilot.json'),
      variant='reacquisition_R1',model_variant=cfg['runtime_contract']['model_variant'],R1_protocol_sha256=R1_SHA,
      pose_diagnostics=dict(runtime.pose_diagnostics),
      elapsed_seconds=elapsed,training_state_included=True,
      runtime=runtime.state_dict(schedule=schedule,include_model=False),optimizer_state_dict=optimizer.state_dict(),
      scheduler_state_dict=scheduler.state_dict(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
      supervision_reads=supervision.reads)
    temporary=output/'.last.tmp.ckpt';torch.save(state,temporary);os.replace(temporary,output/'last.ckpt')
    keys=('schema','arm','stage','seed','supervised_outputs','optimizer_updates','base_checkpoint','base_config','runtime_contract',
      'STATE_FP32','RENDER_FP32','LOSS_FP32','model_state_dict','config_sha256','sourcefreeze_sha256','model_metadata',
      'variant','model_variant','R1_protocol_sha256','pose_diagnostics','execution_variant','pilot_sourcefreeze_sha256','pilot_audit_sha256')
    weights={k:state[k] for k in keys};weights['training_state_included']=False
    torch.save(weights,output/f'outputs={done}.ckpt')
    if done==cfg['supervised_outputs']:
        final=output/'.final.tmp.ckpt';torch.save(weights,final);os.replace(final,output/'final_weights.ckpt')


def train(config,freeze,output,device='cuda:0',resume=None):
    from tools.s43_rel.mean_shape_replay import InferenceAccessGuard
    from .s43_field_data import TargetFreeRunLanes,LazySupervisionReader
    from .runtime_fastreadout import SharedS43RuntimeFastReadout as SharedS43Runtime
    f=verify_frozen(freeze,config);cfg=read(config);output=Path(output)
    if cfg['arm'] not in ARMS or cfg['stage']!='full' or cfg.get('execution_variant')!=EXECUTION_VARIANT:raise ValueError('fixed full-only execution config required')
    if cfg['stage']=='full':
        audit=read(OUT/'audits/pilot.json')
        if audit.get('status')!='PASS' or audit['sourcefreeze_sha256']!=PILOT_FREEZE_SHA:raise RuntimeError('allfivepilot engineering/source audit required')
    if output.exists() and any(output.iterdir()) and resume is None:raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not device.startswith('cuda') or visible not in ('4','5','6','7'):raise RuntimeError('one process per explicit physicalGPU4–7')
    os.environ.setdefault('PYTHONUTF8','1');torch.set_num_threads(cfg['cpu_threads'])
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,contract=build_model(cfg['arm'],device,base_checkpoint=cfg['base_checkpoint'],base_config=cfg['base_config'],seed=cfg['seed'])
    if contract!=cfg['runtime_contract']:raise ValueError('runtime config mismatch')
    base=getattr(model,'base',model);initial_common=tensor_digest({k:v for k,v in base.state_dict().items() if not k.startswith('conv1.')})
    initial_all=tensor_digest(model.state_dict())
    initial_parameters=tensor_digest({k:v for k,v in base.named_parameters() if not k.startswith('conv1.')})
    initial_bn={k:int(v.item()) for k,v in model.state_dict().items() if k.endswith('num_batches_tracked')}
    with InferenceAccessGuard() as setup_guard:
        schedule=TargetFreeRunLanes(cfg['data_root'],subjects=cfg['subjects'],batch_size=cfg['batch_size'],seed=cfg['seed'])
    runtime=SharedS43Runtime(model,arm=contract['runtime_arm'],temporal_collapse=contract['temporal_collapse'],
      tensor_readout=True,state_fp32=True,allow_external_beta=False,transport_chunk_ms=cfg['transport_chunk_ms'])
    supervision=LazySupervisionReader(cfg['data_root']);params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(params,lr=cfg['learning_rate'])
    scheduler=OutputScheduler(optimizer,target=cfg['supervised_outputs'],warmup=cfg['warmup_supervised_outputs'],base_lr=cfg['learning_rate'])
    source_chain='';step=0;elapsed_before=0.;guard_counts=dict(blocked_aux_beta_attempts=0,blocked_pose_meta_attempts=0)
    if resume:
        saved=torch.load(resume,map_location='cpu')
        if saved['schema']!='s43_field_full_training_v1' or saved['config_sha256']!=sha(config) or saved['sourcefreeze_sha256']!=sha(freeze):raise ValueError('resume identity mismatch')
        model.load_state_dict(saved['model_state_dict']);optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict']);runtime.load_state_dict(saved['runtime'],schedule=schedule)
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        supervision.reads=saved['supervision_reads'];source_chain=saved['source_chain'];step=saved['optimizer_updates'];elapsed_before=saved['elapsed_seconds']
        if scheduler.outputs!=runtime.audit['supervised_outputs']:raise ValueError('resume budget clocks differ')
        curve=output/'curve.jsonl'
        if curve.exists():
            records=[json.loads(s) for s in curve.read_text(encoding='utf-8').splitlines()]
            kept=[r for r in records if r['update']<=step];stale=[r for r in records if r['update']>step]
            if stale:
                dump(output/('uncheckpointed_trace_after_'+str(step)+'.json'),stale)
                curve.write_text(''.join(json.dumps(r)+'\n' for r in kept),encoding='utf-8')
    manifest=dict(schema='s43_field_full_execution_manifest_v1',execution_variant=EXECUTION_VARIANT,
      pilot_sourcefreeze_sha256=PILOT_FREEZE_SHA,pilot_audit_sha256=sha(OUT/'audits/pilot.json'),config=cfg,config_sha256=sha(config),sourcefreeze_sha256=sha(freeze),
      initial_common_digest=initial_common,initial_all_digest=initial_all,initial_common_parameters_digest=initial_parameters,
      initial_BN_counters=initial_bn,
      canonical_sha256=sha(cfg['base_checkpoint']),
      model=model_metadata(model,cfg['arm']),data=schedule.manifest(),setup_input_guard=setup_guard.receipt(),
      mixed_precision='networkBF16;state/render/original_compute_loss/log10FP32;BNtrain',
      phase_objective='mean original log10(rawphase loss) over actualsupervised phases; finalpartial update dividesbyactualphasecount',
      budget='actualselected supervision only; ownwarmup/padding/tail/finalunselected nevercounted',
      scheduler='fixedcosine inactualsupervisedoutputs; lr beforeupdate uses outputsconsumedbythatupdate',resume=resume)
    if not resume:dump(output/'manifest.json',manifest)
    else:dump(output/('resume_at_'+str(scheduler.outputs)+'.json'),manifest)
    prepared=batch=None;failure_phase='start'
    start=time.perf_counter();next_save=((scheduler.outputs//cfg['save_every_outputs'])+1)*cfg['save_every_outputs']
    try:
        while scheduler.outputs<cfg['supervised_outputs']:
            model.train();optimizer.zero_grad(set_to_none=True);phases=0;raws=[];phase_sizes=[];objectives=[]
            input_digest=hashlib.sha256();target_digest=hashlib.sha256();before=runtime.audit.copy();network_seconds=0.;input_seconds=0.
            done=scheduler.outputs
            while phases<cfg['clip_steps'] and done<cfg['supervised_outputs']:
                t=time.perf_counter()
                with InferenceAccessGuard() as guard:
                    failure_phase='source_peek';batch=schedule.peek_batch();phase_digest(input_digest,batch)
                    failure_phase='input_ingest_readout';prepared=runtime.inputs(batch)
                    eligible=int(prepared.supervised_mask.sum());limit=cfg['supervised_outputs']-done
                    failure_phase='model_query'
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):
                        result=runtime.query(prepared,inference=not bool(eligible),supervised_limit=limit)
                input_seconds+=time.perf_counter()-t
                receipt=guard.receipt()
                for key in guard_counts:guard_counts[key]+=receipt[key]
                if any(guard_counts.values()):raise AssertionError('input firewall violation')
                if result.prediction.dtype!=torch.float32:raise AssertionError('own-feedback state must be FP32')
                failure_phase='lazy_supervision';truth=supervision.read(result);n=int(truth.mask.sum());input_digest.update(truth.mask.tobytes())
                if n:
                    failure_phase='original_loss_backward';t=time.perf_counter();loss,parts=runtime.loss(result,truth) # deliberately outside autocast
                    objective=original_objective(loss,model)
                    if not torch.isfinite(loss) or not torch.isfinite(objective):raise ArithmeticError('nonfinite original loss/objective')
                    objective.backward();network_seconds+=time.perf_counter()-t
                    raws.append(float(loss.detach()));objectives.append(float(objective.detach()));phase_sizes.append(n);phases+=1;done+=n
                    target_digest.update(truth.target51[truth.mask].tobytes());target_digest.update(truth.beta_target[truth.mask].tobytes())
                failure_phase='own_commit';runtime.commit(result);schedule.commit_batch(batch)
            average_phase_gradients(params,phases)
            grad=float(torch.nn.utils.clip_grad_norm_(params,cfg['grad_clip']))
            if not math.isfinite(grad):raise ArithmeticError('nonfinite gradient')
            failure_phase='optimizer';lr=scheduler.prepare(done);optimizer.step();scheduler.advance(done);step+=1
            digest=input_digest.hexdigest();source_chain=hashlib.sha256((source_chain+digest).encode()).hexdigest()
            record=dict(update=step,supervised_outputs=done,step_source_digest=digest,source_chain=source_chain,
              supervision_digest=target_digest.hexdigest(),lr=lr,grad_norm=grad,phase_sizes=phase_sizes,raw_losses=raws,
              phase_objectives=objectives,mean_phase_objective=float(np.mean(objectives)),raw_loss=float(np.mean(raws)),
              audit=dict(runtime.audit),step_warmup_outputs=runtime.audit['warmup_outputs']-before['warmup_outputs'],
              input_forward_seconds=input_seconds,loss_backward_seconds=network_seconds,
              model_observation_stats=observation_stats(model),
              pose_diagnostics=dict(runtime.pose_diagnostics),elapsed_seconds=elapsed_before+time.perf_counter()-start)
            with (output/'curve.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
            if step<=3 or step%10==0:print(json.dumps(dict(arm=cfg['arm'],stage=cfg['stage'],**record)),flush=True)
            if done>=next_save or done==cfg['supervised_outputs']:
                verify_frozen(freeze,config)
                save_checkpoint(output,model,runtime,schedule,supervision,optimizer,scheduler,cfg,config,freeze,source_chain,step,record['elapsed_seconds'])
                next_save=((done//cfg['save_every_outputs'])+1)*cfg['save_every_outputs']
        if runtime.audit['supervised_outputs']!=cfg['supervised_outputs'] or supervision.reads!=cfg['supervised_outputs']:raise AssertionError('actual supervised budget mismatch')
        verify_frozen(freeze,config,full_input_check=True)
        dump(output/'complete.json',dict(status='COMPLETE',arm=cfg['arm'],stage=cfg['stage'],actual_supervised_outputs=scheduler.outputs,
          optimizer_updates=step,scheduler=scheduler.state_dict(),runtime=dict(runtime.audit),supervision_reads=supervision.reads,
          source_chain=source_chain,sourcefreeze_sha256=sha(freeze),source_files_unchanged=True,input_guard=guard_counts,
          final_weights_sha256=sha(output/'final_weights.ckpt'),last_checkpoint_sha256=sha(output/'last.ckpt'),
          final_common_digest=tensor_digest({k:v for k,v in base.state_dict().items() if not k.startswith('conv1.')}),
          final_common_parameters_digest=tensor_digest({k:v for k,v in base.named_parameters() if not k.startswith('conv1.')}),
          initial_common_parameters_digest=initial_parameters,
          final_all_digest=tensor_digest(model.state_dict()),initial_common_digest=initial_common,initial_all_digest=initial_all,
          model_observation_stats=observation_stats(model),
              pose_diagnostics=dict(runtime.pose_diagnostics),
          elapsed_seconds=elapsed_before+time.perf_counter()-start))
    except BaseException as exc:
        context={}
        if prepared is not None:
            context=dict(previous_pose=prepared.previous.detach().float().cpu().numpy(),K_source=prepared.K_source.detach().float().cpu().numpy(),
              beta_input=prepared.beta_input.detach().float().cpu().numpy(),ref_ms=prepared.ref_ms,cutoff_ms=prepared.cutoff_ms,
              source_items=[{k:v for k,v in item.items() if k not in ('events',)} for item in prepared.items],
              query_metadata=getattr(model,'last_query_metadata',None))
        dump(output/'failure.json',dict(status='FAILED',error=repr(exc),failure_phase=failure_phase,context=context,update=step,supervised_outputs=scheduler.outputs,
          runtime=dict(runtime.audit),sourcefreeze_sha256=sha(freeze),old_scientific_results_unchanged=True))
        raise


def audit(stage,freeze):
    verify_frozen(freeze,full_input_check=True);budget=STAGES[stage]['target'];complete={};traces={};initial=set()
    for arm in ARMS:
        folder=OUT/'train'/stage/arm;c=read(folder/'complete.json');m=read(folder/'manifest.json')
        if c['status']!='COMPLETE' or c['actual_supervised_outputs']!=budget or c['supervision_reads']!=budget:raise AssertionError('fixed budget')
        if c['sourcefreeze_sha256']!=sha(freeze) or not c['source_files_unchanged'] or any(c['input_guard'].values()):raise AssertionError('source/input guard')
        if c['final_common_parameters_digest']==c['initial_common_parameters_digest']:raise AssertionError('original common parameters did not update')
        rows=[json.loads(s) for s in (folder/'curve.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(rows)!=c['optimizer_updates'] or rows[-1]['supervised_outputs']!=budget:raise AssertionError('actual curve counts')
        if sum(sum(r['phase_sizes']) for r in rows)!=budget:raise AssertionError('actualphase supervised count')
        traces[arm]=[(r['supervised_outputs'],r['step_source_digest'],r['supervision_digest']) for r in rows]
        saved=torch.load(folder/'last.ckpt',map_location='cpu')
        if saved.get('execution_variant')!=EXECUTION_VARIANT or saved.get('pilot_sourcefreeze_sha256')!=PILOT_FREEZE_SHA:raise AssertionError('full execution identity')
        if saved.get('pilot_audit_sha256')!=sha(OUT/'audits/pilot.json'):raise AssertionError('original pilot gate identity')
        if saved['runtime'].get('readout_backend')!='native_ring_bins_v1':raise AssertionError('native ring getter identity')
        if (saved.get('schema')!='s43_field_full_training_v1' or saved.get('variant')!='reacquisition_R1' or saved.get('R1_protocol_sha256')!=R1_SHA
            or saved.get('model_variant')!=arm_contract(arm)['model_variant'] or saved['runtime_contract']!=arm_contract(arm)):
            raise AssertionError('R1 checkpoint constructor identity')
        if saved['runtime'].get('runtime_variant')!='ordered_lane_parallel_R1' or saved['runtime'].get('runtime_workers')!=1:raise AssertionError('fixed R1 execution identity')
        if saved['runtime']['pose_diagnostics']['committed_rows']!=saved['runtime']['audit']['committed_outputs']:raise AssertionError('R1 pose diagnostic count')
        steps=[float(s['step']) for s in saved['optimizer_state_dict']['state'].values()]
        if not steps or min(steps)!=max(steps) or max(steps)!=c['optimizer_updates']:raise AssertionError('actualAdam update count')
        if saved['scheduler_state_dict']['outputs']!=budget or saved['runtime']['audit']['supervised_outputs']!=budget:raise AssertionError('saved scheduler/runtime clocks')
        if saved['runtime']['schedule']['state']['outputs']!=saved['runtime']['audit']['committed_outputs']:raise AssertionError('all validwarm/context outputs acknowledged')
        if saved['runtime']['schedule']['state']['events']!=saved['runtime']['audit']['ingested_events']:raise AssertionError('actualsource event counters differ')
        initial_bn=m['initial_BN_counters']
        if not initial_bn or not all(int(saved['model_state_dict'][k].item())>n for k,n in initial_bn.items()):raise AssertionError('BN counters didnot advance from actualpretrained initial values')
        weights=torch.load(folder/'final_weights.ckpt',map_location='cpu')
        if any(weights[k]!=saved[k] for k in ('schema','variant','model_variant','R1_protocol_sha256','runtime_contract','execution_variant','pilot_sourcefreeze_sha256')):raise AssertionError('R1 deployment constructor identity differs')
        if weights['training_state_included'] or tensor_digest(weights['model_state_dict'])!=tensor_digest(saved['model_state_dict']):raise AssertionError('deploymentweights mismatch')
        initial.add(m['initial_common_digest']);complete[arm]=dict(c,optimizer_step_min=min(steps),optimizer_step_max=max(steps))
        if stage=='full' and m['initial_all_digest']!=read(OUT/'train/small'/arm/'manifest.json')['initial_all_digest']:
            raise AssertionError('full didnot restart freshcanonical/newlayerseed')
        del saved,weights
    if len(initial)!=1:raise AssertionError('canonical common initialization not paired')
    reference=traces[ARMS[0]]
    if any(traces[a]!=reference for a in ARMS[1:]):raise AssertionError('all per-update source/target/budget traces must pair')
    path=OUT/'audits'/('pilot.json' if stage=='small' else 'full.json')
    dump(path,dict(status='PASS',stage=stage,execution_variant=EXECUTION_VARIANT,pilot_sourcefreeze_sha256=PILOT_FREEZE_SHA,sourcefreeze_sha256=sha(freeze),arms=list(ARMS),
      supervised_outputs_per_arm=budget,total_supervised_outputs=budget*len(ARMS),all_per_update_source_target_traces_paired=True,
      all_model_original_common_updated=True,all_sources_unchanged=True,actual_counts=complete,
      no_accuracy_selection=True,full_fresh_canonical_init_required=True))


def engineering(dest,device='cuda:0',batch_size=8):
    """Actual five-arm batch workload, fake public supervision, zero updates."""
    import gc
    from .s43_field_data import make_input_batch,SupervisionBatch
    from .runtime_fastreadout import SharedS43RuntimeFastReadout as SharedS43Runtime
    from tools.s43_rel.mean_shape_replay import MeanShapeInputReader,InferenceAccessGuard
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not device.startswith('cuda') or os.environ.get('CUDA_VISIBLE_DEVICES') not in ('4','5','6','7'):raise RuntimeError('GPU4–7engineering only')
    with InferenceAccessGuard() as guard:reader=MeanShapeInputReader('ch_local')
    result={}
    for arm in ARMS:
        def run_one():
            torch.cuda.reset_peak_memory_stats();wall_start=time.perf_counter();model,contract=build_model(arm,device)
            model.train();rt=SharedS43Runtime(model,arm=contract['runtime_arm'],temporal_collapse=contract['temporal_collapse'])
            for ordinal in range(3):
                a=ordinal*50;b=a+50;packet=reader.packet(a,b);items=[]
                for lane in range(batch_size):
                    items.append(dict(lane=lane,valid=True,padding=False,seq_id=f'engineering{lane}',reset=ordinal==0,
                      recording='ch_local',run_id=0,run_start_ms=0,run_stop_ms=150,start_ms=a,cutoff_ms=b,target_ms=b-1,
                      packet_ordinal=ordinal,initial_state=np.array([0,0,.6]+[0.]*48,np.float32),
                      beta_input=np.zeros(10,np.float32),K_source=reader.K_cam.copy(),**packet))
                with InferenceAccessGuard() as scope:
                    torch.cuda.synchronize();phase_start=time.perf_counter();formed=rt.inputs(make_input_batch(items,batch_id=ordinal));torch.cuda.synchronize();input_time=time.perf_counter()-phase_start
                    phase_start=time.perf_counter()
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=rt.query(formed,inference=ordinal<2)
                    torch.cuda.synchronize();query_time=time.perf_counter()-phase_start
                assert not scope.receipt()['aux_betas_accessed'] and not scope.receipt()['pose_metadata_opened']
                if ordinal==2:
                    target=np.tile(np.asarray([.001,0,.6]+[0.]*48,np.float32),(batch_size,1))
                    fake=SupervisionBatch(target,np.zeros((batch_size,10),np.float32),out.supervised_mask.copy(),out.input_token)
                    raw,parts=rt.loss(out,fake);obj=original_objective(raw,model);obj.backward()
                    assert out.prediction.dtype==torch.float32 and torch.isfinite(obj)
                    finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());assert finite
                    info=dict(prediction_dtype=str(out.prediction.dtype),raw_loss_dtype=str(raw.dtype),lossfinite=True,gradfinite=True,
                      supervised_rows=int(out.supervised_mask.sum()),warmup_outputs=rt.audit['warmup_outputs'],
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      third_input_seconds=input_time,third_query_seconds=query_time,wall_seconds=time.perf_counter()-wall_start,
                      runtime_workers=rt.runtime_workers,model_observation_stats=observation_stats(model))
                rt.commit(out)
            rt.close()
            if hasattr(model,'close_geometry_workers'):model.close_geometry_workers()
            return info
        result[arm]=run_one();gc.collect();torch.cuda.empty_cache()
    dump(dest,dict(schema='s43_field_trainer_workload_engineering_v1',status='PASS',trainer_sha256=sha(__file__),
      queue_sha256=sha(ROOT/'tools/s43_revisit/s43_field_full_queue.py'),batch_size=batch_size,device=device,
      actual_native_input='ch_local0:150ms repeatedindependentlanes',public_fake_supervision=True,real_GT_files_opened=False,
      optimizer_updates=0,arms=result,input_guard=guard.receipt(),numerics='networkBF16;state/render/loss/log10FP32'))


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    s.add_parser('init-configs')
    a=s.add_parser('input-manifest');a.add_argument('--out',type=Path,default=OUT/'engineering/input_manifest.json')
    a=s.add_parser('freeze');a.add_argument('--out',type=Path,default=OUT/'engineering/full_execution_sourcefreeze.json');a.add_argument('--engineering',nargs='+',required=True);a.add_argument('--commands',nargs='+',required=True)
    a=s.add_parser('train');a.add_argument('--config',type=Path,required=True);a.add_argument('--source-freeze',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--device',default='cuda:0');a.add_argument('--resume',type=Path)
    a=s.add_parser('audit');a.add_argument('--stage',choices=('full',),required=True);a.add_argument('--source-freeze',type=Path,required=True)
    a=s.add_parser('engineering');a.add_argument('--out',type=Path,required=True);a.add_argument('--device',default='cuda:0');a.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args()
    if a.command=='init-configs':init_configs()
    elif a.command=='input-manifest':input_manifest(a.out)
    elif a.command=='freeze':sourcefreeze(a.out,a.engineering,a.commands)
    elif a.command=='audit':audit(a.stage,a.source_freeze)
    elif a.command=='engineering':engineering(a.out,a.device,a.batch_size)
    else:train(a.config,a.source_freeze,a.out,a.device,a.resume)


if __name__=='__main__':main()
