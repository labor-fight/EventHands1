"""Opt-in XDL five-arm model; baseline MNISTModel remains unchanged."""
from __future__ import annotations
import copy
import dataclasses
import numpy as np
import torch
from torch import nn
from model import MNISTModel
from semkine.encoder import event_tokens
from semkine.events import EventPacketBatch,EV_X,EV_Y
from semkine.xyz_mesh import associate_rays,stride_sample
from semkine.xyz_observation import ObservationBank,project_state_joints


def select_packet_rows(batch,rows):
    """Select ragged rows without changing events, clocks or hand context."""
    ids=rows.detach().cpu().tolist()
    pieces=[]; counts=[]
    for new,old in enumerate(ids):
        ev=batch.packet(old).clone()
        counts.append(ev.shape[0])
        bi=ev.new_full((ev.shape[0],1),new)
        pieces.append(torch.cat([bi,ev],-1))
    events=torch.cat(pieces,0) if pieces else batch.events[:0]
    ptr=torch.tensor([0]+list(np.cumsum(counts)),device=batch.ptr.device,dtype=torch.long)
    kwargs={}
    for field in dataclasses.fields(batch):
        value=getattr(batch,field.name)
        if field.name not in ('events','ptr'):
            kwargs[field.name]=None if value is None else value.index_select(0,rows.to(value.device))
    return EventPacketBatch(events=events,ptr=ptr,**kwargs)


class RecoveryXYZModel(MNISTModel):
    MODEL_KEYS=MNISTModel.MODEL_KEYS|frozenset({'XDL_STUDY','XDL_QUERY','XDL_SELF','XDL_OBSERVATION_NULL','XDL_BURN_MODE'})
    TRACK_KEYS=MNISTModel.TRACK_KEYS|frozenset({'XDL_SELF_P','XDL_WARMUP_START','XDL_RAMP_END'})

    def __init__(self,cfg=None,num_classes=None):
        cfg=copy.deepcopy(cfg or {})
        mc=cfg.setdefault('MODEL',{})
        if mc.get('XDL_STUDY') is not True:
            raise ValueError('RecoveryXYZModel requires MODEL.XDL_STUDY=true')
        self.xdl_query=mc.get('XDL_QUERY','mano')
        self.xdl_self=bool(mc.get('XDL_SELF',False))
        self.xdl_null=bool(mc.get('XDL_OBSERVATION_NULL',False))
        self.xdl_burn_mode=mc.get('XDL_BURN_MODE','selected')
        if self.xdl_burn_mode not in ('selected','full'):
            raise ValueError('XDL_BURN_MODE must be selected or full')
        if self.xdl_query not in ('fixed','mano'):
            raise ValueError('XDL_QUERY must be fixed or mano')
        super().__init__(cfg=cfg,num_classes=num_classes)
        if not self.xyz_mesh or self.pose_repr!='mano_full_axis_angle':
            raise ValueError('XDL requires xyz_mesh and the 51D residual AA layout')
        if self.xyz.geom or self.xyz.residual_mode!='legacy' or self.xyz_assoc_mode!='hit_first' or self.xyz_assoc_prev!='self':
            raise ValueError('XDL requires common C0 geometry, legacy residual and self hit_first association')
        if self.unroll_p or self.rollout_p or self.self_p or self.prevpos_embed or self.prev_render or not self.predict_delta or not self.zero_event_gate:
            raise ValueError('XDL requires bare delta tracking, zero-event hold and no other conditioning paths')
        if self.xyz.N_SLOTS*self.xyz.slot_proj.out_channels//self.xyz.N_SLOTS!=544:
            raise ValueError('XDL requires the 544D compressed legacy observation interface')
        self.observation_bank=ObservationBank(self.render_h,self.render_w,self.xyz_max_nodes)
        self.xyz.fc1=nn.Linear(544+64+99,self.xyz.fc1.out_features)
        self.xyz.fc1.register_forward_pre_hook(self._recovery_readout)
        self._recovery_inputs=None
        self.xdl_burn_rows=0; self.xdl_self_rows=0
        track=cfg.get('TRACK',{})
        self.xdl_probability=float(track.get('XDL_SELF_P',.5))
        self.xdl_warmup=int(track.get('XDL_WARMUP_START',500))
        self.xdl_ramp_end=int(track.get('XDL_RAMP_END',2000))
        if (self.xdl_probability,self.xdl_warmup,self.xdl_ramp_end)!=(.5,500,2000):
            raise ValueError('XDL-v2 fixes self probability=.5 and curriculum500..2000')

    def reset_recovery_context(self):
        self._recovery_inputs=None

    def _recovery_readout(self,module,inputs):
        feature=inputs[0]
        if self._recovery_inputs is None or feature.shape[-1]!=592:
            raise RuntimeError('XDL readout needs a prepared 544+48 legacy input')
        bank,state=self._recovery_inputs
        legacy=feature[:,:544]
        if not bool(torch.isfinite(legacy).all()) or not bool(torch.isfinite(bank).all()):
            raise FloatingPointError('nonfinite final observation before N1 zeroing')
        if self.xdl_null:
            legacy=legacy*0.; bank=bank*0.
        return (torch.cat([legacy,bank.to(feature),state.to(feature)],-1),)+inputs[1:]

    def _xyz_mesh_forward(self,batch,prev):
        events,ptr=batch.events,batch.ptr
        B=batch.batch_size
        if events.shape[0]==0:
            self.route_stats={}
            # DDP reducer participation: an actual empty batch holds every row,
            # yet all trainable parameters must remain in its autograd graph.
            link=sum((p.reshape(-1)[0]*0. for p in self.parameters() if p.requires_grad),prev.new_zeros(()))
            return prev.new_zeros(B,51)+link
        if self.route_prev_override is not None:
            raise ValueError('XDL forbids undeclared prior/association overrides')
        betas,camera=self._resolve_betas_K(prev,batch.betas,batch.camera_K)
        src,mask=stride_sample(ptr,self.xyz_max_nodes)
        tok=event_tokens(events,ptr,batch.delta_t_s,self.render_h,self.render_w)
        tok=tok[src.reshape(-1)].view(B,-1,tok.shape[-1])*mask[...,None]
        # Preserve original float32 FK/association under bf16 training.
        with torch.no_grad(),torch.autocast(device_type=prev.device.type,enabled=False):
            verts,joints=self._fk(prev.float(),betas.float())
            if not bool(torch.isfinite(verts).all()) or not bool(torch.isfinite(joints).all()):
                raise FloatingPointError('nonfinite MANO FK')
            inverse=torch.argsort(self.mano.mano16_tips_to_openpose21)
            joints16=joints[:,inverse[:16]]
            intrinsics=self._intrinsics(camera.float())
            uv,valid,raw,outside=project_state_joints(joints16,intrinsics,self.render_h,self.render_w)
            fx,fy,cx,cy=intrinsics
            ev=events[src.reshape(-1)].view(B,-1,events.shape[1])
            px,py=ev[...,EV_X].float(),ev[...,EV_Y].float()
            d=torch.stack([(px-cx[:,None])/fx[:,None],(py-cy[:,None])/fy[:,None],torch.ones_like(px)],-1)
            fpx=(fx*fy).sqrt()
            assoc=associate_rays(d,mask,verts,self.mano.f,self.xyz.vert_faces,
                band=self.xyz_band_px/fpx,front_tol=self.xyz_front_px/fpx,z_tol=self.xyz_z_tol,k=self.xyz_k)
            # Common state99 = normalized translation, original48AA, UV32, valid16.
            state=torch.cat([prev[:,:3].float()/.1,prev[:,3:].float(),
                             (uv/uv.new_tensor([self.render_w,self.render_h])).flatten(1),valid.float()],-1)
        observation=self.observation_bank.encode(events,ptr,batch.delta_t_s)
        if not torch.equal(src,observation['src']) or not torch.equal(mask,observation['mask']):
            raise RuntimeError('bank support differs from legacy XYZ support')
        bank,weights=self.observation_bank.query(observation,uv,self.xdl_query)
        self._recovery_inputs=(bank,state)
        try:
            delta,stats,aux=self.xyz(tok,d,mask,verts,assoc,prev[:,3:51],ablate=self.ablate_evidence)
        finally:
            self.reset_recovery_context()
        self.route_stats=stats
        if getattr(self,'xyz_keep_aux',False):
            self.xyz_last={'assoc':assoc,'d':d,'mask':mask,'verts':verts,'tok':tok,'delta':delta.detach(),
                **{k:v.detach() for k,v in aux.items()},'xdl_uv':uv,'xdl_raw_uv':raw,
                'xdl_projection_valid':valid,'xdl_outside':outside,'xdl_weights':weights.detach(),
                'xdl_bank':bank.detach(),'xdl_state':state,'xdl_src':src}
        return delta

    def forward_packet(self,batch):
        prediction=super().forward_packet(batch)
        # Base addition casts prev to autocast dtype. Restore exact float32
        # incoming coordinates on truly empty rows, including mixed batches.
        return torch.where((batch.counts<=0)[:,None],batch.prev_state.float(),prediction.float())

    def self_probability(self,step):
        return self.xdl_probability*max(0.,min(1.,(int(step)-self.xdl_warmup)/float(self.xdl_ramp_end-self.xdl_warmup)))

    def _predict_batch(self,batch):
        if not isinstance(batch,dict) or 'main' not in batch:
            return super()._predict_batch(batch)
        main=batch['main']
        step=int(self.global_step)
        rows=batch['global_row'].detach().cpu().tolist()
        if len(rows)!=main.batch_size:
            raise ValueError('global_row count differs from main batch')
        probability=self.self_probability(step) if self.xdl_self and self.training else 0.
        from tools.xdl.data import counter_rng
        take=[index for index,row in enumerate(rows) if counter_rng(self.cfg.get('SEED',3407),step,row,'self').random()<probability]
        if take:
            prefixes=batch['prefix']
            if len(prefixes)!=3:
                raise ValueError('XDL requires exactly three prefix packets')
            chosen=torch.tensor(take,device=main.prev_state.device,dtype=torch.long)
            burn_rows=(main.batch_size if self.xdl_burn_mode=='full' else len(take))
            was_training=self.training
            dtype=main.prev_state.device.type
            enabled=torch.is_autocast_cpu_enabled() if dtype=='cpu' else torch.is_autocast_enabled()
            amp_dtype=torch.get_autocast_cpu_dtype() if dtype=='cpu' else torch.get_autocast_gpu_dtype()
            try:
                self.eval()
                with torch.no_grad(),torch.autocast(device_type=dtype,enabled=enabled,dtype=amp_dtype,cache_enabled=False):
                    previous=None
                    for prefix in prefixes:
                        packed=prefix if self.xdl_burn_mode=='full' else select_packet_rows(prefix,chosen)
                        if previous is not None:
                            packed=dataclasses.replace(packed,prev_state=previous)
                        previous=self.forward_packet(packed).detach()
            finally:
                self.train(was_training)
                self.reset_recovery_context()
            changed=main.prev_state.clone()
            chosen_previous=previous.index_select(0,chosen) if self.xdl_burn_mode=='full' else previous
            changed.index_copy_(0,chosen,chosen_previous.to(changed))
            main=dataclasses.replace(main,prev_state=changed)
            self.xdl_self_rows+=len(take); self.xdl_burn_rows+=3*burn_rows
        return super()._predict_batch(main)
