"""Supplementary real-full checkpoint restore, no targets/optimizer updates.

This read-only auditor is not a training dependency and never changes weights,
source cursors or files used by active jobs.
"""
from pathlib import Path
import argparse,copy,random
import numpy as np
import torch
from . import s43_field_full_training as T
from . import s43_field_training_R1 as Prior
from .s43_field_data import TargetFreeRunLanes
from .runtime_fastreadout import SharedS43RuntimeFastReadout
from tools.s43_rel.mean_shape_replay import InferenceAccessGuard


def run(checkpoint,out):
    torch.set_num_threads(1);freeze=T.OUT/'engineering/full_execution_sourcefreeze.json';T.verify_frozen(freeze)
    saved=torch.load(checkpoint,map_location='cpu');cfg=saved['config'];folder=Path(checkpoint).parent
    if saved['schema']!='s43_field_full_training_v1' or saved['execution_variant']!=T.EXECUTION_VARIANT or saved['sourcefreeze_sha256']!=T.sha(freeze):raise AssertionError('full checkpoint identity')
    model,payload=T.load_weights(checkpoint,'cpu');base=getattr(model,'base',model)
    manifest=T.read(folder/'manifest.json')
    digest=T.tensor_digest({k:v for k,v in base.named_parameters() if not k.startswith('conv1.')})
    if digest==manifest['initial_common_parameters_digest']:raise AssertionError('original common parameters did not update')
    BN={k:int(saved['model_state_dict'][k])-int(initial) for k,initial in manifest['initial_BN_counters'].items()}
    if not BN or min(BN.values())<=0:raise AssertionError('actual BN advance')
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'])
    optimizer.load_state_dict(saved['optimizer_state_dict']);steps=[int(v['step']) for v in optimizer.state.values()]
    if not steps or min(steps)!=max(steps) or max(steps)!=saved['optimizer_updates']:raise AssertionError('Adam restore clock')
    scheduler=T.OutputScheduler(optimizer,target=cfg['supervised_outputs'],warmup=cfg['warmup_supervised_outputs'],base_lr=cfg['learning_rate'])
    scheduler.load_state_dict(saved['scheduler_state_dict'])
    if scheduler.outputs!=saved['supervised_outputs'] or scheduler.outputs!=saved['supervision_reads']:raise AssertionError('supervision/scheduler restore')
    c=saved['runtime_contract']
    with InferenceAccessGuard() as guard:
        schedule=TargetFreeRunLanes(cfg['data_root'],subjects=cfg['subjects'],batch_size=cfg['batch_size'],seed=cfg['seed'])
        runtime=SharedS43RuntimeFastReadout(model,arm=c['runtime_arm'],temporal_collapse=c['temporal_collapse'],transport_chunk_ms=cfg['transport_chunk_ms'],runtime_workers=1)
        runtime.load_state_dict(saved['runtime'],schedule=schedule)
        if not torch.equal(torch.get_rng_state(),saved['runtime']['torch_rng']):raise AssertionError('CPU RNG restore')
        if runtime.audit!=saved['runtime']['audit'] or runtime.pose_diagnostics!=saved['runtime']['pose_diagnostics']:raise AssertionError('runtime clock/pose diagnostics')
        state=runtime.state_dict(include_model=False)
        for sid,lane in saved['runtime']['lanes'].items():
            got=state['lanes'][sid]
            for key,value in lane['memory']['arrays'].items():np.testing.assert_array_equal(value,got['memory']['arrays'][key])
            if lane['memory']['scalars']!=got['memory']['scalars']:raise AssertionError('core scalar source clocks')
            np.testing.assert_array_equal(lane['pose'],got['pose']);np.testing.assert_array_equal(lane['beta'],got['beta'])
            if len(lane['history'])!=len(got['history']):raise AssertionError('causal history length')
            for a,b in zip(lane['history'],got['history']):
                if a[0]!=b[0]:raise AssertionError('causal history clock')
                np.testing.assert_array_equal(a[1],b[1])
            if lane['last_timestamp'] is not None:np.testing.assert_array_equal(lane['last_timestamp'],got['last_timestamp'])
        a=schedule.peek_batch();token=a.token
        # A second independent restored sampler must produce the same actual
        # next native packet token; no GT/pose supervision reader is created.
        second=TargetFreeRunLanes(cfg['data_root'],subjects=cfg['subjects'],batch_size=cfg['batch_size'],seed=cfg['seed'])
        second.load_state_dict(saved['runtime']['schedule']);b=second.peek_batch()
        if token!=b.token:raise AssertionError('actual source cursor/RNG restoration')
        if any(np.any(lane['beta']) for lane in runtime.lanes.values()):raise AssertionError('public input beta0')
        runtime.close()
    deploy=folder/f"outputs={saved['supervised_outputs']}.ckpt";weights=torch.load(deploy,map_location='cpu')
    if weights['training_state_included'] or T.tensor_digest(weights['model_state_dict'])!=T.tensor_digest(saved['model_state_dict']):raise AssertionError('weightsonly equality')
    try:Prior.load_weights(deploy,'cpu')
    except ValueError:old_reject=True
    else:raise AssertionError('old loader silently accepted full execution schema')
    T.dump(out,dict(status='PASS_AVAILABLE_FULL_CHECKPOINT_RESTORE',full_training_complete=False,checkpoint=str(checkpoint),checkpoint_sha256=T.sha(checkpoint),
        sourcefreeze_sha256=T.sha(freeze),audit_source_sha256=T.sha(__file__),arm=cfg['arm'],actual_supervised_outputs=saved['supervised_outputs'],optimizer_updates=saved['optimizer_updates'],
        original_common_parameters_changed=True,BN_counter_actual_increase_min=min(BN.values()),BN_counter_actual_increase_max=max(BN.values()),
        optimizer_step_min=min(steps),optimizer_step_max=max(steps),scheduler_and_supervision_clock_exact=True,live_lanes=len(saved['runtime']['lanes']),
        all_numeric_core_arrays_and_source_scalars_pose_history_exact=True,next_actual_native_source_token_exact=True,next_source_token=token,
        CPU_rng_restored_exact=True,saved_CUDA_rng_present=isinstance(saved['runtime']['cuda_rng'],list),
        GPU_rng_note='CPUaudit retains/inspects savedCUDAstate but doesnot executeGPUresume. EarlierrealGPUresume tolerance evidence remains separate.',
        all_input_beta_zero=True,old_loader_rejects_new_schema=old_reject,weightsonly_equals_stateful_checkpoint=True,
        input_guard=guard.receipt(),GT_reader_constructed=False,optimizer_updates_executed_by_auditor=0,active_training_unmodified=True))
    if hasattr(model,'close_geometry_workers'):model.close_geometry_workers()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=T.ARMS,default='s43');p.add_argument('--checkpoint',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args();checkpoint=a.checkpoint or T.OUT/'train/full'/a.arm/'last.ckpt';out=a.out or T.OUT/'audits'/f'first_full_checkpoint_{a.arm}.json';run(checkpoint,out)
if __name__=='__main__':main()
