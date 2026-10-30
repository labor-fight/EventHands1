"""Read-only supplementary Stage5 R1 checkpoint and canonical identity audit.

No labels/events are decoded and no training source is modified. This does not
replace the five-arm full gate; incomplete live checkpoints remain PARTIAL.
"""
import argparse
from pathlib import Path
import torch
from . import s43_field_training_R1 as T

def run(dest):
    f=T.OUT/'engineering/actual_sourcefreeze.json';T.verify_frozen(f)
    prior_path=T.BASE_OUT/'s43_field_model_engineering/cpu_receipt.json'
    composite=T.read(T.BASE_OUT/'s43_field_model_engineering/receipt.json')
    if composite['source_and_engineering_sha256'][str(prior_path.resolve())]!=T.sha(prior_path):raise AssertionError('frozen model-engineering nested receipt changed')
    prior=T.read(prior_path)
    cfg=T.read(T.OUT/'configs/full/s43.json')
    expected={cfg['base_checkpoint']:prior['base_checkpoint_sha256'],cfg['base_config']:prior['base_config_sha256']}
    for p,h in expected.items():
        if T.sha(p)!=h:raise AssertionError('canonical model/config identity drift: '+p)
    manifests={};checkpoints={}
    for stage in T.STAGES:
        for arm in T.ARMS:
            folder=T.OUT/'train'/stage/arm
            if not (folder/'manifest.json').exists():continue
            m=T.read(folder/'manifest.json');c=m['config']
            if c['base_checkpoint']!=cfg['base_checkpoint'] or c['base_config']!=cfg['base_config'] or m['canonical_sha256']!=prior['base_checkpoint_sha256']:
                raise AssertionError('actual canonical initialization identity: '+str(folder))
            manifests[f'{stage}/{arm}']=dict(canonical_sha256=m['canonical_sha256'],initial_all_digest=m['initial_all_digest'])
            if not (folder/'last.ckpt').exists():continue
            saved=torch.load(folder/'last.ckpt',map_location='cpu')
            if saved['schema']!='s43_field_training_R1_v1' or saved['variant']!='reacquisition_R1' or saved['R1_protocol_sha256']!=T.R1_SHA:raise AssertionError('R1 header')
            if saved['runtime_contract']!=T.arm_contract(arm) or saved['sourcefreeze_sha256']!=T.sha(f):raise AssertionError('checkpoint semantics/source')
            steps=[int(s['step']) for s in saved['optimizer_state_dict']['state'].values()]
            if not steps or min(steps)!=max(steps) or max(steps)!=saved['optimizer_updates']:raise AssertionError('Adam clock')
            runtime=saved['runtime'];done=saved['supervised_outputs']
            if any(k not in saved for k in ('python_rng','numpy_rng')) or not isinstance(runtime.get('torch_rng'),torch.Tensor) or not isinstance(runtime.get('cuda_rng'),list):raise AssertionError('complete RNG state missing')
            if not runtime.get('model_modes') or not all(isinstance(v,bool) for v in runtime['model_modes'].values()):raise AssertionError('saved module modes missing')
            if done!=saved['scheduler_state_dict']['outputs'] or done!=runtime['audit']['supervised_outputs'] or done!=saved['supervision_reads']:raise AssertionError('actual supervision clocks')
            if runtime['schedule']['state']['outputs']!=runtime['audit']['committed_outputs'] or runtime['schedule']['state']['events']!=runtime['audit']['ingested_events']:raise AssertionError('source/output acknowledgement')
            if runtime['pose_diagnostics']['committed_rows']!=runtime['audit']['committed_outputs']:raise AssertionError('physical state diagnostics')
            advanced={k:int(saved['model_state_dict'][k])-initial for k,initial in m['initial_BN_counters'].items()}
            if not advanced or min(advanced.values())<=0:raise AssertionError('pretrained BN counters did not advance')
            checkpoints[f'{stage}/{arm}']=dict(supervised_outputs=done,updates=saved['optimizer_updates'],optimizer_step_min=min(steps),optimizer_step_max=max(steps),
              BN_counter_increase_min=min(advanced.values()),BN_counter_increase_max=max(advanced.values()),
              saved_python_numpy_torch_and_cuda_rng=True,live_lanes=len(runtime['lanes']),pose_diagnostics=runtime['pose_diagnostics'])
            del saved
    T.dump(dest,dict(status='PASS_IDENTITY_AND_AVAILABLE_CHECKPOINTS',full_training_complete=False,
        sourcefreeze_sha256=T.sha(f),audit_source_sha256=T.sha(__file__),canonical_files=expected,manifests=manifests,checkpoints=checkpoints,
        original_frozen_sources_unchanged=True,GT_decoded=False,new_task_scores_read=False,
        note='Supplementary external audit, not a replacement pilot/full gate. Canonical identities are inherited from the earlier frozen model-engineering receipt.'))
    return len(checkpoints)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=T.OUT/'audits/external_identity_and_checkpoints.json');a=p.parse_args()
    torch.set_num_threads(1);print(run(a.out))
if __name__=='__main__':main()
