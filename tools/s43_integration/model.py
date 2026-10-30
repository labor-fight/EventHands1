"""A representation bridge in front of the unchanged S43 forward/loss.

This is an opt-in experimental branch. A finite B-R decision contract passing
does not imply that its new continuous decoder is statistically effective.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CKPT=ROOT/'outputs/semkine/s43n_sp_s3407/s43n_sp_s3407-step=6000.ckpt'
DEFAULT_CFG=ROOT/'outputs/semkine/s43n_sp_s3407/config_resolved.yaml'


def load_legacy(checkpoint=DEFAULT_CKPT,config=DEFAULT_CFG,device='cpu'):
    """Load original Lightning identity without hijacking the model namespace."""
    folder=str(ROOT/'model')
    if folder not in sys.path: sys.path.insert(0,folder)
    name='_s43_integration_legacy_model'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,ROOT/'model/model.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        spec.loader.exec_module(module)
    import yaml,copy
    cfg=yaml.safe_load(Path(config).read_text()) if not isinstance(config,dict) else config
    cfg=copy.deepcopy(cfg)
    for option in ('S43_EVENT_REP','S43_TRAIN_CONV1','S43_BR_TEMPERATURE','S43_TRAIN_READOUT'):
        cfg.get('MODEL',{}).pop(option,None)
    base=sys.modules[name].MNISTModel.load_from_checkpoint(str(checkpoint),cfg=cfg,map_location=device)
    return base.to(device).eval(),cfg


class EnhancedAdapter(nn.Module):
    def __init__(self):
        super().__init__();self.projection=nn.Conv2d(6,2,1,bias=False)
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.weight[0,0,0,0]=1.;self.projection.weight[1,1,0,0]=1.

    def forward(self,reps,device,dtype):
        image=torch.as_tensor(np.stack([r['enhanced'] for r in reps]),device=device,dtype=dtype)
        return self.projection(image.permute(0,3,1,2)).permute(0,2,3,1)


class BRImageAdapter(nn.Module):
    """Finite evidence -> mixture of public predicted event images -> S43.

    No raw observed LNES is used. The forward-model templates and coordinates
    are conditions, not measured image/depth. Soft candidate weights are a
    learned adapter, not calibrated full-state posterior probabilities.
    """
    def __init__(self,k=3,temperature=1000.):
        super().__init__();self.k=int(k);self.temperature=float(temperature)
        self.scorer=nn.Sequential(nn.Linear(self.k*53,32),nn.Tanh(),nn.Linear(32,self.k))
        self.projection=nn.Conv2d(2,2,1,bias=False)
        with torch.no_grad():
            self.scorer[-1].weight.zero_();self.scorer[-1].bias.zero_()
            self.projection.weight.zero_()
            self.projection.weight[0,0,0,0]=1.;self.projection.weight[1,1,0,0]=1.

    def forward(self,reps,device,dtype):
        r=torch.as_tensor(np.stack([v['R'] for v in reps]),device=device,dtype=dtype)
        templates=torch.as_tensor(np.stack([v['templates'] for v in reps]),device=device,dtype=dtype)
        if r.shape[1:]!=(self.k,53) or templates.shape[1:3]!=(self.k,2):
            raise ValueError('BR adapter expects R[B,K,53], templates[B,K,2,H,W]')
        valid=r[:,:,52]>0
        if not torch.all(valid.any(1)): raise ValueError('all BR candidates unsupported')
        scaled=r.clone()
        scaled[:,:,51]=(scaled[:,:,51]/self.temperature).clamp(-50,0)
        scaled[:,:,:51]=scaled[:,:,:51].clamp(-10,10)
        logits=scaled[:,:,51]+self.scorer(scaled.flatten(1))
        logits=logits.masked_fill(~valid,-torch.inf)
        weights=logits.softmax(1)
        image=(weights[:,:,None,None,None]*templates).sum(1)
        image=self.projection(image)
        h,w=reps[0]['lnes'].shape[:2]
        image=F.interpolate(image,size=(h,w),mode='nearest')
        return image.permute(0,2,3,1)


class S43IntegratedModel(nn.Module):
    def __init__(self,base,mode='legacy',train_conv1=False,temperature=1000.,k=3,train_readout=False):
        super().__init__()
        if mode not in ('legacy','enhanced','br'): raise ValueError(mode)
        self.base=base;self.mode=mode;self.train_conv1=bool(train_conv1)
        self.temperature=float(temperature);self.k=int(k)
        self.train_readout=bool(train_readout)
        if self.k!=3 or not np.isfinite(self.temperature) or self.temperature<=0:
            raise ValueError('integration fixes K=3 and a finite positive temperature')
        self.adapter=nn.Identity() if mode=='legacy' else EnhancedAdapter() if mode=='enhanced' else BRImageAdapter(k,temperature)
        for p in self.base.parameters(): p.requires_grad_(False)
        if self.train_conv1:
            for p in self.base.conv1.parameters(): p.requires_grad_(True)
        if self.train_readout:
            for p in self.base.rn.fc.parameters(): p.requires_grad_(True)
        self.base.eval()

    def train(self,mode=True):
        super().train(mode)
        self.base.eval()  # pretrained normalization stays fixed in every arm
        self.adapter.train(mode)
        return self

    def event_image(self,reps,device=None,dtype=torch.float32):
        device=device or next(self.base.parameters()).device
        if self.mode=='legacy':
            return torch.as_tensor(np.stack([r['lnes'] for r in reps]),device=device,dtype=dtype)
        active=[i for i,r in enumerate(reps) if r['has_events'] and
                r.get('meta',{}).get('status','ok')=='ok' and not r.get('meta',{}).get('padding',False)]
        h,w=reps[0]['lnes'].shape[:2]
        image=torch.zeros((len(reps),h,w,2),device=device,dtype=dtype)
        if active:
            ix=torch.tensor(active,device=device,dtype=torch.long)
            image=image.index_copy(0,ix,self.adapter([reps[i] for i in active],device,dtype))
        return image

    def forward(self,reps,prev,betas=None,camera_K=None):
        if len(reps)!=len(prev): raise ValueError('batch row mismatch')
        image=self.event_image(reps,prev.device,prev.dtype)
        result=self.base(image,prev,betas=betas,camera_K=camera_K)
        if self.mode!='legacy':
            live=torch.tensor([r['has_events'] and r.get('meta',{}).get('status','ok')=='ok'
                               for r in reps],device=prev.device,dtype=torch.bool)
            result=torch.where(live[:,None],result,prev)
        return result

    def loss(self,pred,target,betas=None):
        return self.base._compute_loss(pred,target,betas)

    def counts(self):
        return {'total':sum(p.numel() for p in self.parameters()),
                'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'adapter':sum(p.numel() for p in self.adapter.parameters()),
                'original_trunk_and_decoder_changed':False,'new_loss':False,
                'input_conv1_trainable':self.train_conv1,'existing_readout_trainable':self.train_readout}

    def save(self,path,cfg,extra=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists(): raise FileExistsError(path)
        # Explicit schema: old Lightning cannot silently treat this as LNES.
        torch.save({'schema':'s43_repr_integration_v1','mode':self.mode,
                    'temperature':self.temperature,'k':self.k,'cfg':cfg,
                    'base_state_dict':self.base.state_dict(),
                    'adapter_state_dict':self.adapter.state_dict(),
                    'train_conv1':self.train_conv1,'train_readout':self.train_readout,'extra':extra or {}},path)


def load_integrated(config,checkpoint=None):
    import yaml
    cfg=yaml.safe_load(Path(config).read_text()) if not isinstance(config,dict) else config
    model_cfg=cfg.get('MODEL',{})
    mode=model_cfg.get('S43_EVENT_REP','legacy')
    old=cfg.get('INTEGRATION',{}).get('BASE_CHECKPOINT',str(DEFAULT_CKPT))
    base,_=load_legacy(old,cfg)
    wrapper=S43IntegratedModel(base,mode,model_cfg.get('S43_TRAIN_CONV1',False),
                  model_cfg.get('S43_BR_TEMPERATURE',1000.),3,model_cfg.get('S43_TRAIN_READOUT',False))
    if checkpoint:
        content=torch.load(checkpoint,map_location='cpu')
        if (content.get('schema')!='s43_repr_integration_v1' or content['mode']!=mode or
                float(content.get('temperature',float('nan')))!=wrapper.temperature or
                int(content.get('k',-1))!=wrapper.k):
            raise ValueError('checkpoint representation/config mismatch')
        wrapper.base.load_state_dict(content['base_state_dict'],strict=True)
        wrapper.adapter.load_state_dict(content['adapter_state_dict'],strict=True)
    return wrapper,cfg
