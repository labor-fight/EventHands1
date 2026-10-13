"""State independent event bank and matched F/Q queries for XDL only."""
from __future__ import annotations
import math
import torch
from torch import nn
from semkine.event_gnn import EventGNN
from semkine.xyz_mesh import stride_sample


@torch.no_grad()
def project_state_joints(joints, intrinsics, height=180, width=240):
    """Projection in event pixels; invalid depth uses center, outside clamps.

    Returns processed UV, valid-depth bits, raw UV and out-of-frame bits.
    The caller rejects nonfinite FK; this helper also handles projection overflow.
    """
    fx,fy,cx,cy=intrinsics
    x,y,z=joints.float().unbind(-1)
    safe=z.clamp_min(1e-6)
    raw=torch.stack([fx[:,None]*x/safe+cx[:,None],fy[:,None]*y/safe+cy[:,None]],-1)
    valid=(z>1e-6)&torch.isfinite(joints).all(-1)&torch.isfinite(raw).all(-1)
    outside=valid&((raw[...,0]<0)|(raw[...,0]>width-1)|(raw[...,1]<0)|(raw[...,1]>height-1))
    clipped=torch.stack([raw[...,0].clamp(0,width-1),raw[...,1].clamp(0,height-1)],-1)
    fallback=torch.tensor([width/2.,height/2.],device=joints.device,dtype=torch.float32)
    return torch.where(valid[...,None],clipped,fallback),valid,raw,outside


class _CommonSupportGNN(EventGNN):
    def _sample(self, events, ptr):
        return stride_sample(ptr,self.max_nodes)


class ObservationBank(nn.Module):
    """Raw token7 GNN, full-support queries and a shared 34 -> 32 -> 4 MLP.

    Slot embeddings are learned slot IDs. Both query modes compute the same
    signed event-minus-current-MANO offsets; only the spatial centers differ.
    No persistent event cache or previous-state-dependent encoder is used.
    """
    def __init__(self,height=180,width=240,max_nodes=2048):
        super().__init__()
        self.height,self.width=int(height),int(width)
        self.encoder=_CommonSupportGNN(height=height,width=width,hidden=32,n_layers=2,
            k=8,window=32,max_nodes=max_nodes,readout=False,node_attrs='token7')
        self.slot_embedding=nn.Parameter(torch.randn(16,32)/math.sqrt(32))
        self.compress=nn.Sequential(nn.Linear(34,32),nn.ReLU(),nn.Linear(32,4))
        yy,xx=torch.meshgrid(torch.arange(4),torch.arange(4),indexing='ij')
        atlas=torch.stack([(xx.flatten()+.5)*width/4.,(yy.flatten()+.5)*height/4.],-1)
        self.register_buffer('atlas',atlas,persistent=False)

    def encode(self,events,ptr,delta_t_s):
        _,h,_,px,py,mask=self.encoder(events,ptr,delta_t_s,return_nodes=True)
        src,expected=stride_sample(ptr,self.encoder.max_nodes)
        if not events.shape[0]:
            src=src[:,:0]; expected=expected[:,:0]
        if not torch.equal(mask,expected):
            raise RuntimeError('bank and XYZ sampling mask mismatch')
        return {'h':h,'xy':torch.stack([px,py],-1),'mask':mask,'src':src}

    def query(self,observation,mano_uv,mode):
        if mode not in ('fixed','mano'):
            raise ValueError('query mode must be fixed or mano')
        h,xy,mask=observation['h'],observation['xy'],observation['mask']
        B,N,_=h.shape
        if mano_uv.shape!=(B,16,2):
            raise ValueError('MANO UV must be [B,16,2]')
        if N==0:
            # The consumer connects all parameters for a fully empty batch.
            return h.new_zeros(B,64),h.new_zeros(B,16,0)
        # Float32 softmax/offset arithmetic protects small support under bf16.
        with torch.autocast(device_type=h.device.type,enabled=False):
            centers=mano_uv.float() if mode=='mano' else self.atlas.float()[None].expand(B,-1,-1)
            content=torch.einsum('jc,bnc->bjn',self.slot_embedding.float(),h.float())/math.sqrt(32)
            distance=(xy.float()[:,None]-centers[:,:,None]).square().sum(-1)/(2.*16.**2)
            logits=(content-distance).masked_fill(~mask[:,None],-torch.inf)
            live=mask.any(-1)[:,None,None]
            logits=torch.where(live,logits,torch.zeros_like(logits))
            soft=logits.softmax(-1)*mask[:,None].float()
            uniform=mask[:,None].float()/mask.sum(-1)[:,None,None].clamp_min(1)
            weights=.9*soft+.1*uniform
            pooled_h=torch.einsum('bjn,bnc->bjc',weights,h.float())
            offset=(xy.float()[:,None]-mano_uv.float()[:,:,None])/16.
            pooled_offset=(weights[...,None]*offset).sum(2)
            pooled=torch.cat([pooled_h,pooled_offset],-1)
        compressed=self.compress(pooled.to(self.compress[0].weight.dtype))
        compressed=compressed*live.to(compressed.dtype)
        return compressed.flatten(1),weights
