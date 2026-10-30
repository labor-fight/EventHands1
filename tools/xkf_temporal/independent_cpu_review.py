#!/usr/bin/env python3
"""Independent two-arm BF16 CPU step, discarded in memory after verification."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'model')]
import torch
from config import load_config
from mano_layer import ManoLayer
from tools.xdl.data import counter_rng,packet_hash
from tools.xkf_fusion.motion_data import MotionDataset,arm_config,collate_incremental
from tools.xkf_temporal.model import TemporalDModel
from tools.xkf_temporal.train import parent_audit,all_buffer_hashes
from tools.xkf_statistics.collect import sha


def require(value,message):
    if not value:raise ValueError(message)


def main():
    torch.set_num_threads(4)
    output=ROOT/'outputs/xkf/fusion_20261015_r3/temporal_preparation/independent_cpu_review'
    require(not output.exists(),'no overwrite');output.mkdir(parents=True)
    sources={str(ROOT/name):sha(ROOT/name) for name in ('tools/xkf_temporal/model.py','tools/xkf_temporal/train.py',
        'tools/xkf_temporal/independent_cpu_review.py','tools/xkf_joint/model.py','tools/xkf_fusion/motion_data.py','tools/xkf_repair/data.py')}
    config=ROOT/'configs/xkf_repair/repair_s3407.yaml'
    parent=ROOT/'outputs/xkf/fusion_20261015_r3/direct_finetune/incremental_finetune/raw_s3407/xkf_ionly_raw_s3407-step=2600.ckpt'
    cfg=arm_config(load_config(str(config)),'raw');saved=torch.load(parent,map_location='cpu',weights_only=False)
    dataset=MotionDataset(cfg,start_step=2600,max_steps=2601)
    row=next(i for i in range(20) if counter_rng(3407,2600,i,'self').random()<.5)
    batch=collate_incremental([dataset[row]]);before_prefix=packet_hash(batch['prefix'][-1]);main_hash=packet_hash(batch['main'])
    reports={};gradient_states={};captured_states={}
    for weight in (0.,1.):
        local=copy.deepcopy(cfg);local['MODEL']['XKF_TEMPORAL_D_WEIGHT']=weight
        torch.manual_seed(3407);model=TemporalDModel(local);restored=parent_audit(model,saved,parent)
        for module in model.modules():
            if isinstance(module,ManoLayer):module.fast_chain=True
        model._trainer=SimpleNamespace(global_step=2600);logs={};model.log=lambda name,value,**unused:logs.update({name:float(value.detach().float())})
        optimizer=model.configure_optimizers()['optimizer'];optimizer.load_state_dict(copy.deepcopy(saved['optimizer_states'][0]))
        before=all_buffer_hashes(model);calls=[];observed={}
        hook=model.direct.register_forward_pre_hook(lambda *unused:calls.append(torch.is_grad_enabled()))
        original=model.paired_direct_loss
        def capture(zm,am,main,zp,ap,previous):
            result=original(zm,am,main,zp,ap,previous)
            observed.update(zm=zm,am=am,main=main,zp=zp,ap=ap,previous=previous,ld=result[0],parts=result[1])
            return result
        model.paired_direct_loss=capture;model.train()
        with torch.autocast('cpu',dtype=torch.bfloat16):loss=model.training_step(copy.deepcopy(batch),0)
        hook.remove();require(calls==[False,False,False,True,True],'actual full burn and two differentiable D forwards differ')
        require(packet_hash(batch['prefix'][-1])==before_prefix and not bool(batch['prefix'][-1].target.any()),'prefix labels mutated')
        require(torch.equal(observed['previous'].target,batch['previous_gt']),'previous supervision label differs')
        # Full physical FK independently reproduces the auxiliary-joint shortcut.
        with torch.autocast('cpu',enabled=False):
            jm=model._fk(observed['zm'].float(),batch['main'].betas.float())[1]
            jp=model._fk(observed['zp'].float(),batch['main'].betas.float())[1]
            with torch.no_grad():
                gm=model._fk(batch['main'].target.float(),batch['main'].betas.float())[1]
                gp=model._fk(batch['previous_gt'].float(),batch['main'].betas.float())[1]
            auxm=observed['am']['joints0'].float()+observed['zm'][:,None,:3].float()
            auxp=observed['ap']['joints0'].float()+observed['zp'][:,None,:3].float()
            geometric_difference=max(float((jm-auxm).abs().max()),float((jp-auxp).abs().max()))
            require(geometric_difference<2e-7,'shortcut is not physical FK')
            manual_motion=(jm-jp-(gm-gp)).square().mean()
            require(torch.allclose(manual_motion,observed['parts']['d_temporal_joint_mse'],rtol=2e-5,atol=1e-12),'temporal true displacement formula differs')
            dmain,_=model._direct_loss(observed['zm'],observed['am'],observed['main'])
            dprev,_=model._direct_loss(observed['zp'],observed['ap'],observed['previous'])
            manual_loss=.5*(dmain+dprev)+weight*.5*model.lambda_t/model.normalizer*manual_motion
            require(torch.allclose(manual_loss,observed['ld'],rtol=2e-5,atol=1e-12),'two-frame or temporal normalization differs')
            gm_grad,gp_grad=torch.autograd.grad(manual_motion,(jm,jp),retain_graph=True)
            require(float(gm_grad.norm())>0 and torch.equal(gm_grad,-gp_grad),'temporal prediction endpoints do not both carry equal/opposite gradient')
        loss.backward();gradients={}
        for name,parameter in model.named_parameters():
            require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),'missing/nonfinite gradient '+name)
            gradients[name]=parameter.grad.detach().float().clone()
        require(len(gradients)==96 and len(restored['parameter_order'])==96,'96 original Adam slots required')
        require(all(int(optimizer.state[p]['step'])==2400 for p in model.parameters()),'wrong pre-step Adam clock')
        optimizer.step();require(all(int(optimizer.state[p]['step'])==2401 for p in model.parameters()),'wrong post-step Adam clock')
        require(all_buffer_hashes(model)==before and len(before)==65,'named buffers changed')
        require(model.xdl_self_rows==1 and model.xdl_burn_rows==3,'original Gaussian/self exposure differs')
        require(not hasattr(model,'fusion_policy'),'deployment adapter leaked into training')
        marker={};model.on_save_checkpoint(marker)
        require(marker['xkf_joint_continuation']['target_global_step']==2800
            and marker['xkf_temporal_continuation']['target_parameter_steps']==2600,'new debug endpoint mislabeled')
        arm='control' if weight==0 else 'temporal';gradient_states[arm]=gradients
        captured_states[arm]={key:observed[key].detach().clone() for key in ('zm','zp')}
        reports[arm]=dict(weight=weight,loss=float(loss),logs=logs,manual_temporal_joint_mse=float(manual_motion),
            independent_FK_vs_aux_max_difference_m=geometric_difference,
            both_temporal_endpoint_gradients_nonzero_and_opposite=True,
            temporal_endpoint_gradient_norm=float(gm_grad.norm()),
            all96_Adam_before=2400,all96_Adam_after=2401,all65_buffers_unchanged=True,
            direct_call_gradient_enabled=calls,original_self_rows=int(model.xdl_self_rows),original_burn_rows=int(model.xdl_burn_rows),
            gradient_norms={name:float(g.norm()) for name,g in gradients.items()},buffer_sha256=before,
            saved_marker=marker)
        print(json.dumps(dict(arm=arm,loss=float(loss),manual_motion=float(manual_motion))),flush=True)
    require(all(torch.equal(gradient_states['control'][n],gradient_states['temporal'][n]) for n in gradient_states['control'] if not n.startswith('direct.')),'I gradients changed across temporal arm')
    changed_d=[n for n in gradient_states['control'] if n.startswith('direct.') and not torch.equal(gradient_states['control'][n],gradient_states['temporal'][n])]
    require(changed_d,'temporal objective produced no actual D gradient difference')
    require(all(torch.equal(captured_states['control'][k],captured_states['temporal'][k]) for k in ('zm','zp')),'arms do not begin from same predictions')
    require(all(sha(path)==digest for path,digest in sources.items()),'reviewed source changed while executing')
    result=dict(kind='INDEPENDENT_TEMPORAL_TWO_ARM_CPU_STEP',status='PASS',formal=False,trained_run=False,
        source_sha256=sources,input_sha256={str(config):sha(config),str(parent):sha(parent)},
        sample_identity=batch['sampling_identity'],main_packet_sha256=main_hash,previous_packet_sha256=before_prefix,
        I_gradients_bitwise_identical_between_arms=True,identical_preupdate_D_predictions=True,
        changed_D_gradient_tensors=changed_d,arms=reports,scope='Two real BF16 CPU Adam updates discarded in memory; no checkpoint written and no quality claim.')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('INDEPENDENT_REVIEW_PASS',flush=True)


if __name__=='__main__':main()
