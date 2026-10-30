"""Real CPU S43 smoke and paired, two-step TBPTT=1 adapter fine-tuning."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

from .model import load_integrated,load_legacy,S43IntegratedModel
from .memory import batch_from_windows

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/s43_integration_20261027'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a))+'\n')


def sources(name):
    proto=json.loads((OUT/'protocol.json').read_text())
    assert sha(ROOT/proto['plan'])==proto['plan_sha256']
    for p,value in proto['production_sources'].items():
        if sha(ROOT/p)!=value:raise RuntimeError(f'frozen source changed: {p}')
    snapshot={'argv':__import__('sys').argv,'plan_sha256':proto['plan_sha256'],
       'files':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tools/s43_integration').glob('*')) if p.is_file()},
       'at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'device':'cpu'}
    dump(OUT/'sources'/f'{name}.json',snapshot)


def windows(clips,phase,previous):
    result=[]
    for c,p in zip(clips,previous):
        w=dict(c['windows'][phase]);w['history']=p.detach() if hasattr(p,'detach') else np.array(p,copy=True)
        w['history_time_us']=int(w['start_us'])-1000
        result.append(w)
    return result


def tensors(clips):
    return (torch.from_numpy(np.stack([c['history'] for c in clips]).astype(np.float32)),
            torch.from_numpy(np.stack([c['betas'] for c in clips]).astype(np.float32)),
            torch.from_numpy(np.stack([c['K_cam'] for c in clips]).astype(np.float32)))


def predict_clip_batch(model,clips,lead_cache=None,indices=None):
    previous,betas,camera=tensors(clips)
    losses=[];predictions=[];metadata=[]
    for phase in range(2):
        if phase==0 and lead_cache is not None:
            reps=[lead_cache[int(i)] for i in indices]
        else:
            reps=batch_from_windows(windows(clips,phase,previous),mode=model.mode)
        predicted=model(reps,previous,betas,camera)
        target=torch.from_numpy(np.stack([c['targets'][phase] for c in clips]).astype(np.float32))
        loss,parts=model.loss(predicted,target,betas)
        losses.append(loss);predictions.append(predicted)
        metadata.append([r['meta'] for r in reps])
        previous=predicted.detach()  # TBPTT=1 and no graph in the representation
        if previous.grad_fn is not None:raise AssertionError('state graph leaked')
    return torch.stack(losses).mean(),predictions,metadata


def evaluate_loss(model,clips):
    model.eval();loss=[]
    with torch.no_grad():
        for i in range(0,len(clips),2):
            value,_,_=predict_clip_batch(model,clips[i:i+2]);loss.append(float(value))
    return float(np.mean(loss))


def smoke(configs,out):
    from .data import load_clips
    sources('smoke');torch.set_num_threads(2);torch.manual_seed(3407)
    clips=load_clips(); selected=[c for c in clips if c['partition']=='train'][:2]
    results=[]
    for config in configs:
        model,cfg=load_integrated(config);model.train()
        prev,beta,camera=tensors(selected)
        reps=batch_from_windows(windows(selected,0,prev),mode=model.mode)
        old_ln=torch.tensor(np.stack([r['lnes'] for r in reps]))
        with torch.no_grad():old=model.base(old_ln,prev,beta,camera)
        prediction=model(reps,prev,beta,camera)
        loss,_=model.loss(prediction,torch.tensor(np.stack([c['targets'][0] for c in selected])),beta)
        objective=loss.clamp_min(1e-12).log10() if model.base.log10_loss else loss
        if objective.requires_grad:objective.backward()
        adapter_grad=sum(float(p.grad.norm()) for p in model.adapter.parameters() if p.grad is not None)
        conv_grad=sum(float(p.grad.norm()) for p in model.base.conv1.parameters() if p.grad is not None)
        if not bool(torch.isfinite(prediction).all() and torch.isfinite(loss)):raise AssertionError('nonfinite smoke')
        parity=float((prediction-old).abs().max()) if model.mode in ('legacy','enhanced') else None
        if parity is not None and parity!=0:raise AssertionError('default/identity compatibility failed')
        if model.mode!='legacy' and adapter_grad<=0:raise AssertionError('adapter receives no useful gradient')
        results.append({'config':str(config),'mode':model.mode,'loss':float(loss),'legacy_max_abs_diff':parity,
                        'adapter_gradient_norm_sum':adapter_grad,'input_conv_gradient_norm_sum':conv_grad,
                        'output_shape':list(prediction.shape),'params':model.counts()})
    dump(out,{'real_frozen_S43_checkpoint':True,'real_native_event_windows':True,'GT_input':False,
              'samples':[c['clip_id'] for c in selected],'results':results,
              'scope':'actual forward/backward, not a stub test'})
    print(out,flush=True)


def train(config,out_dir):
    from .data import load_clips
    out_dir=Path(out_dir)
    if (out_dir/'last.ckpt').exists() or (out_dir/'train.json').exists():raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    import yaml
    cfg=yaml.safe_load(Path(config).read_text());settings=cfg['INTEGRATION'];arm=settings['ARM']
    sources(f'train_{arm}')
    torch.set_num_threads(int(settings['CPU_THREADS']));torch.manual_seed(int(settings['SEED']))
    np.random.seed(int(settings['SEED']));rng=np.random.default_rng(int(settings['SEED']))
    model,cfg=load_integrated(cfg)
    clips=load_clips(settings['CLIPS_DIR'])
    fit=[c for c in clips if c['partition']=='train'];validation=[c for c in clips if c['partition']=='validation']
    trainable=[p for p in model.parameters() if p.requires_grad]
    if not trainable:raise ValueError('A0 is frozen; do not send it to the training command')
    def frozen_name(name):
        return not ((model.train_conv1 and name.startswith('conv1.')) or
                    (model.train_readout and name.startswith('rn.fc.')))
    frozen_before={n:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
                   for n,v in model.base.state_dict().items() if frozen_name(n)}
    lead_cache={}
    for i,clip in enumerate(fit):
        lead_cache[i]=batch_from_windows(windows([clip],0,[clip['history']]),mode=model.mode)[0]
    initial_fit=evaluate_loss(model,fit[:8]);initial_val=evaluate_loss(model,validation)
    model.train();optimizer=torch.optim.Adam(trainable,lr=float(settings['LR']))
    before=time.perf_counter();records=[];sample_indices=[];maximum_adapter_grad=0.;skipped_no_grad=0
    for step in range(int(settings['STEPS'])):
        ix=rng.integers(0,len(fit),size=int(settings['BATCH_SIZE']));sample_indices.append(ix.tolist())
        batch=[fit[int(i)] for i in ix];optimizer.zero_grad(set_to_none=True)
        loss,_,metas=predict_clip_batch(model,batch,lead_cache,ix)
        objective=loss.clamp_min(1e-12).log10() if model.base.log10_loss else loss
        if not bool(torch.isfinite(objective)):raise ArithmeticError(f'nonfinite loss at {step}')
        adapter_grad=conv_grad=total_grad=0.
        if objective.requires_grad:
            objective.backward()
            adapter_grad=sum(float(p.grad.norm()) for p in model.adapter.parameters() if p.grad is not None)
            conv_grad=sum(float(p.grad.norm()) for p in model.base.conv1.parameters() if p.grad is not None)
            total_grad=float(torch.nn.utils.clip_grad_norm_(trainable,float(settings['GRAD_CLIP'])))
            if not np.isfinite(total_grad):raise ArithmeticError('nonfinite gradient')
            optimizer.step()
        else:skipped_no_grad+=1
        maximum_adapter_grad=max(maximum_adapter_grad,adapter_grad)
        records.append({'step':step+1,'loss':float(loss.detach()),'objective':float(objective.detach()),
                        'adapter_grad':adapter_grad,'conv1_grad':conv_grad,'grad_before_clip':total_grad,
                        'event_counts':[[m['native_source_count'] for m in row] for row in metas]})
        if step%10==0:print(json.dumps({'arm':arm,**records[-1],'seconds':time.perf_counter()-before}),flush=True)
    training_seconds=time.perf_counter()-before
    final_fit=evaluate_loss(model,fit[:8]);final_val=evaluate_loss(model,validation)
    frozen_after={n:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
                   for n,v in model.base.state_dict().items() if frozen_name(n)}
    if frozen_before!=frozen_after:raise AssertionError('frozen trunk/decoder/BN changed')
    if model.mode!='legacy' and maximum_adapter_grad<=0:raise AssertionError('adapter unused')
    model.save(out_dir/'last.ckpt',cfg,extra={'steps':settings['STEPS'],'TBPTT':1,'seed':settings['SEED']})
    dump(out_dir/'train.json',{'arm':arm,'config':str(config),'config_sha':sha(config),'checkpoint_sha':sha(out_dir/'last.ckpt'),
         'steps':settings['STEPS'],'batch_size':settings['BATCH_SIZE'],'clip_steps':2,'TBPTT':1,'seed':settings['SEED'],
         'lr':settings['LR'],'params':model.counts(),'initial_fit_loss':initial_fit,'final_fit_loss':final_fit,
         'initial_SELECT_loss':initial_val,'final_SELECT_loss':final_val,'training_seconds':training_seconds,
         'frozen_trunk_decoder_BN_unchanged':True,'raw_loss_unchanged':True,
         'maximum_adapter_gradient':maximum_adapter_grad,'skipped_no_gradient_batches':skipped_no_grad,
         'sample_indices':sample_indices,'fit_clip_ids':[c['clip_id'] for c in fit],
         'validation_clip_ids':[c['clip_id'] for c in validation],'curve':records,
         'lead_cache':'deterministic frozen-canonical-warmup condition, reused; second step rebuilt from own detached prediction',
         'GT_role':'targets only; never representation history/ROI/texture'})
    print(json.dumps({'done':arm,'fit_loss':(initial_fit,final_fit),'SELECT_loss':(initial_val,final_val)}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('smoke');q.add_argument('--configs',nargs='+',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    q=s.add_parser('train');q.add_argument('--config',type=Path,required=True);q.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=''
    if a.command=='smoke':smoke(a.configs,a.out)
    else:train(a.config,a.out_dir)


if __name__=='__main__':main()
