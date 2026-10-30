"""Shared continuous field/core runtime; optimizer-free and target-free.

Training and online use the same TimeFieldMemory and S43FieldModel forward.
Only the explicitly selected legacy control constructs a last-event image.
Checkpoints are complete window/commit boundaries, never half a transaction.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import hashlib
import numpy as np
import torch

from .field_core import TimeFieldConfig,TimeFieldMemory,integer,MAX_COUNT
from .s43_field_data import FieldInputBatch,make_input_batch


def _state_readout_float(module,inputs,output):return output.float()


def install_state_fp32(model):
    """Instance-only four-arm adapter; do not round the own-feedback state.

    Network convolutions/readout may use BF16 autocast. Casting the common
    rn.fc result makes original prev_mlp/add/skip operations promote to FP32.
    This preserves mathematics, not legacy BF16 state bit patterns.
    """
    base=getattr(model,'base',model)
    if not hasattr(base,'rn') or not hasattr(base.rn,'fc'):raise ValueError('stateFP32 adapter requires original rn.fc')
    if not getattr(base.rn.fc,'_s43_field_state_fp32_installed',False):
        base.rn.fc.register_forward_hook(_state_readout_float)
        base.rn.fc._s43_field_state_fp32_installed=True


def _bounded_history(history,T):
    keep=0
    for i,(stamp,_) in enumerate(history):
        if stamp<=T-50:keep=i
    result=history[keep:]
    if len(result)>52:raise AssertionError('native-ms commit history bound exceeded')
    return result


def _legacy_last_image(events,T):
    """Only called for legacy_s43; original native-ms W50 convention."""
    out=np.zeros((180,240,2),np.float32)
    if len(events):
        np.maximum.at(out,(events[:,1],events[:,0],events[:,3]),
                      ((events[:,2]-(T-50))/50.).astype(np.float32))
    return out


def _legacy_image_from_times(last_timestamp,T):
    valid=(last_timestamp>=0)&(last_timestamp>=T-50)&(last_timestamp<T)
    return np.where(valid,(last_timestamp-(T-50))/50.,0).astype(np.float32)


def temporal_collapse_field(field40,*,native_r2=False):
    """Same conservative native-bin collapse for both global and local paths.

    Preserve every spatial/polarity total; distribute by actual bin widths.
    Coverage stays observed exposure. This happens before local sampling.

    native_r2=True requires the actual r2 nearest-block field emitted by the
    shared integer core/readout. Strict equality checks reject nonblock
    tensors instead of silently sampling them. Generic callers retain the
    original full-image formula by default.
    """
    if native_r2:
        if field40.dtype!=torch.float32 or field40.ndim!=4 or field40.shape[1:]!=(40,180,240):
            raise ValueError('native_r2 collapse requires float32[B,40,180,240]')
        for first,last,spacing in ((0,16,2),(24,40,4)):
            x=field40[:,first:last]
            block=x[:,:,::spacing,::spacing][:,:,:,None,:,None].expand(-1,-1,-1,spacing,-1,spacing)
            if not torch.equal(x.reshape(-1,16,180//spacing,spacing,240//spacing,spacing),block):
                raise ValueError('native_r2 F2px/Q4px nearest-block contract violated; use generic reference')
    output=field40.clone()
    widths=torch.tensor([7,6,6,6,7,6,6,6],device=field40.device,dtype=torch.float64)
    for first,last in ((0,16),(24,40)):
        spacing=(2 if first==0 else 4) if native_r2 else 1
        x=field40[:,first:last,::spacing,::spacing].double().reshape(-1,8,2,180//spacing,240//spacing)
        total=torch.expm1(x).sum(1,keepdim=True)
        value=torch.log1p(total*widths[None,:,None,None,None]/50.).flatten(1,2).to(field40.dtype)
        output[:,first:last]=value.repeat_interleave(spacing,-2).repeat_interleave(spacing,-1) if native_r2 else value
    return output


@dataclass(frozen=True)
class PreparedFieldInputs:
    items:tuple
    input_token:str
    valid_indices:tuple
    field40:torch.Tensor
    previous:torch.Tensor
    beta_input:torch.Tensor
    K_source:torch.Tensor
    ref_ms:np.ndarray
    cutoff_ms:np.ndarray
    histories:tuple
    legacy_images:object
    supervised_mask:np.ndarray
    state_versions:tuple


@dataclass(frozen=True)
class FieldPrediction:
    items:tuple
    input_token:str
    prediction:torch.Tensor
    supervised_mask:np.ndarray
    warmup_mask:np.ndarray
    budget_unselected_mask:np.ndarray
    valid_mask:np.ndarray
    inputs_complete:bool
    prepared:PreparedFieldInputs
    owner:object


class SharedS43Runtime:
    def __init__(self,model,*,arm=None,transport_chunk_ms=50,warmup_queries=2,field_config=None,
                 temporal_collapse=False,tensor_readout=True,state_fp32=True,allow_external_beta=False):
        self.model=model;self.arm=arm or getattr(model,'arm',None)
        if self.arm not in ('global_only','hand_box','static','legacy_s43'):raise ValueError('explicit field arm or legacy_s43 required')
        if int(transport_chunk_ms)<1 or int(warmup_queries)!=2:raise ValueError('positive transport chunk and fixed2own warmups required')
        self.chunk_ms=int(transport_chunk_ms);self.warmup_queries=2
        self.temporal_collapse=bool(temporal_collapse);self.tensor_readout=bool(tensor_readout)
        self.state_fp32=bool(state_fp32)
        self.allow_external_beta=bool(allow_external_beta)
        if self.state_fp32:install_state_fp32(model)
        # Existing loaded-instance option, common to every Stage5 arm. Keep
        # state-conditioned FK/projection outside network BF16 autocast.
        getattr(model,'base',model).render_fp32=True
        if self.temporal_collapse and self.arm not in ('static','hand_box'):raise ValueError('collapse control requires a static or hand_box field reader')
        self.config=field_config or TimeFieldConfig();self.lanes={};self._owner=object()
        if (self.config.resolution,self.config.width,self.config.height,self.config.horizon_ms,self.config.time_bins)!=(2,240,180,50,8):
            raise ValueError('this shared frontend requires the declared2px W50/8bin field')
        self._prepared=self._pending=None
        self.audit=dict(ingested_events=0,tail_events=0,tail_ms=0,resets=0,releases=0,
            model_forward_rows=0,committed_outputs=0,warmup_outputs=0,supervised_outputs=0,
            budget_unselected_outputs=0,commits=0,legacy_LNES_images=0,source_chain='')

    @property
    def device(self):return next(self.model.parameters()).device

    def _reset(self,item):
        sid=item['seq_id']
        if sid in self.lanes:raise ValueError('reset cannot overwrite a live sequence')
        p=np.asarray(item['initial_state'],np.float32).copy();b=np.asarray(item['beta_input'],np.float32).copy()
        K=np.asarray(item['K_source'],np.float32).copy();start=int(item['run_start_ms'])
        if p.shape!=(51,) or b.shape!=(10,) or K.shape!=(3,3) or not np.isfinite(np.r_[p,b,K.ravel()]).all():raise ValueError('invalid public context')
        if np.any(b) and not self.allow_external_beta:raise ValueError('training beta_input0; external shape requires explicit conditional diagnostic flag')
        self.lanes[sid]=dict(memory=TimeFieldMemory(self.config,seq_id=sid,boundary_ms=start),pose=p,
            beta=b,K=K,history=[(start,p.copy())],watermark=start,last_commit=start,last_logical_ordinal=-1,
            core_ordinal=0,source_end=None,pose_version=0,queries_committed=0,
            last_timestamp=np.full((180,240,2),-1,np.int64) if self.arm=='legacy_s43' else None)
        self.audit['resets']+=1

    def _ingest(self,item,*,tail=False):
        sid=item['seq_id'];lane=self.lanes[sid];events=np.asarray(item['events'])
        T=integer(item['cutoff_ms'],'cutoff');start=integer(item['start_ms'],'start')
        lo,hi=(integer(x,'source_range') for x in item['source_range']);logical_ordinal=integer(item['packet_ordinal'],'ordinal')
        if start!=lane['watermark'] or T<=start or logical_ordinal!=lane['last_logical_ordinal']+1:
            raise ValueError('logical source clock/ordinal is not the next continuous prefix')
        if lane['source_end'] is not None and lo!=lane['source_end']:raise ValueError('source rows must be exactly continuous')
        # A transport cap is never a W50/window cap. Validate the complete
        # logical packet without applying the core's per-transport row cap.
        if events.ndim!=2 or events.shape[1]!=4 or events.dtype.kind not in 'iu':raise ValueError('native integer Nx4 packet required')
        if events.dtype.kind=='u' and events.size and events.max()>np.iinfo(np.int64).max:raise ValueError('native integer overflow')
        events=events.astype(np.int64,copy=False)
        if lo<0 or hi-lo!=len(events) or hi>np.iinfo(np.int64).max:raise ValueError('source range/payload mismatch')
        if lane['memory']._pending_time is not None:raise AssertionError('logical query boundary must be a complete prefix')
        if len(events):
            x,y,t,p=events.T
            if (np.any(x<0) or np.any(x>=240) or np.any(y<0) or np.any(y>=180) or np.any((p!=0)&(p!=1))
                    or np.any(t[1:]<t[:-1]) or np.any(t<start)):raise ValueError('native pixel/polarity/order outside logical prefix')
            if np.any(t>=T):raise ValueError('logical packet must be half-open, not a future pending group')
            # Each new native-ms row starts empty; a row cannot overflow when
            # the entire logical payload has <= uint32_max events.
            if len(events)>MAX_COUNT:
                key=(((t-start)*(180//self.config.resolution)+y//self.config.resolution)*(240//self.config.resolution)+x//self.config.resolution)*2+p
                if np.unique(key,return_counts=True)[1].max()>MAX_COUNT:raise OverflowError('logical native-cell count overflow')
        for a in range(start,T,self.chunk_ms):
            b=min(a+self.chunk_ms,T)
            left=int(np.searchsorted(events[:,2],a));right=int(np.searchsorted(events[:,2],b))
            cursor=left
            while cursor<right or cursor==left==right:
                end=min(cursor+self.config.max_packet_events,right)
                wm=b if end==right else int(events[end,2])
                lane['memory'].ingest(events[cursor:end],wm,lane['core_ordinal'],source_range=(lo+cursor,lo+end),seq_id=sid)
                lane['core_ordinal']+=1;cursor=end
                if end==right:break
        if self.arm=='legacy_s43':
            if len(events):np.maximum.at(lane['last_timestamp'],(events[:,1],events[:,0],events[:,3]),events[:,2])
        lane['watermark']=T;lane['source_end']=hi;lane['last_logical_ordinal']=logical_ordinal
        self.audit['ingested_events']+=len(events)
        if tail:self.audit['tail_events']+=len(events);self.audit['tail_ms']+=T-start
        h=hashlib.sha256(self.audit['source_chain'].encode());h.update(str((sid,lo,hi,start,T)).encode());h.update(events.tobytes())
        self.audit['source_chain']=h.hexdigest()

    def inputs(self,batch):
        if not isinstance(batch,FieldInputBatch):raise TypeError('FieldInputBatch required')
        if make_input_batch(batch.items,releases=batch.releases,batch_id=batch.batch_id).token!=batch.token:raise ValueError('input batch mutated')
        if self._prepared is not None:
            if batch.token==self._prepared.input_token:return self._prepared
            raise RuntimeError('commit the previous output before advancing another prefix')
        live=[i for i,item in enumerate(batch) if item.get('valid',True) and not item.get('padding',False)]
        if len({batch[i]['seq_id'] for i in live})!=len(live):raise ValueError('parallel rows require independent sequence lanes')
        for release in batch.releases:
            sid=release['seq_id']
            if sid not in self.lanes:raise ValueError('release of unknown sequence')
            if release['events'] is not None:self._ingest(release,tail=True)
            self.lanes[sid]['memory'].release();del self.lanes[sid];self.audit['releases']+=1
        fields=[];previous=[];beta=[];K=[];refs=[];cutoffs=[];histories=[];images=[];versions=[]
        supervised=np.zeros(len(batch),bool)
        for i in live:
            item=batch[i];sid=item['seq_id']
            if set(item)&{'target','targets','beta_target','betas','lnes','previous_GT'}:raise ValueError('supervision or old image in input record')
            if item.get('reset',False):self._reset(item)
            if sid not in self.lanes:raise ValueError('sequence requires a boundary reset')
            lane=self.lanes[sid]
            if (not np.array_equal(lane['beta'],item['beta_input']) or not np.array_equal(lane['K'],item['K_source'])):
                raise ValueError('external beta/K changed inside a run; reset/version it explicitly')
            self._ingest(item)
            if self.arm!='legacy_s43':
                snapshot=lane['memory'].readout(item['cutoff_ms'])
                fields.append(snapshot if self.tensor_readout else snapshot.to_grid40ch())
            previous.append(lane['pose']);beta.append(lane['beta']);K.append(lane['K'])
            refs.append(lane['last_commit']);cutoffs.append(item['cutoff_ms'])
            histories.append(tuple((t,p.copy()) for t,p in lane['history']))
            versions.append((sid,lane['memory'].version,lane['pose_version'],lane['watermark']))
            supervised[i]=lane['queries_committed']>=self.warmup_queries
            if self.arm=='legacy_s43':images.append(_legacy_image_from_times(lane['last_timestamp'],item['cutoff_ms']))
        tensor=lambda values,shape:torch.as_tensor(np.stack(values) if values else np.empty((0,*shape),np.float32),device=self.device,dtype=torch.float32)
        if self.tensor_readout and fields:
            from .field_tensor import tensorize
            field_tensor=tensorize(fields,self.device).global40
        else:field_tensor=tensor(fields,(40,180,240))
        if self.temporal_collapse:field_tensor=temporal_collapse_field(field_tensor,native_r2=True)
        prepared=PreparedFieldInputs(batch.items,batch.token,tuple(live),field_tensor,
            tensor(previous,(51,)),tensor(beta,(10,)),tensor(K,(3,3)),np.asarray(refs,np.int64),np.asarray(cutoffs,np.int64),
            tuple(histories),tensor(images,(180,240,2)) if self.arm=='legacy_s43' else None,supervised,tuple(versions))
        if live:self._prepared=prepared
        return prepared

    def query(self,prepared,*,inference=False,supervised_limit=None):
        if len(prepared.valid_indices) and prepared is not self._prepared:raise ValueError('foreign or committed prepared prefix')
        with torch.set_grad_enabled(not inference and torch.is_grad_enabled()):
            if prepared.valid_indices:
                if self.arm=='legacy_s43':
                    pred=self.model(prepared.legacy_images,prepared.previous,betas=prepared.beta_input,camera_K=prepared.K_source)
                    self.audit['legacy_LNES_images']+=len(pred)
                else:
                    pred=self.model(prepared.field40,prepared.previous,beta_input=prepared.beta_input,K_source=prepared.K_source,
                        ref_ms=prepared.ref_ms,cutoff_ms=prepared.cutoff_ms,histories=prepared.histories)
                if pred.shape!=(len(prepared.valid_indices),51) or not torch.isfinite(pred).all():raise ArithmeticError('invalid prediction; no hold fallback')
                result=torch.zeros((len(prepared.items),51),device=pred.device,dtype=pred.dtype)
                result=result.index_copy(0,torch.tensor(prepared.valid_indices,device=pred.device),pred)
                self.audit['model_forward_rows']+=len(pred)
            else:result=torch.zeros((len(prepared.items),51),device=self.device)
        valid=np.zeros(len(prepared.items),bool);valid[list(prepared.valid_indices)]=True
        selected=prepared.supervised_mask.copy()
        if supervised_limit is not None:
            if isinstance(supervised_limit,bool) or int(supervised_limit)!=supervised_limit or supervised_limit<0:
                raise ValueError('nonnegative integer remaining supervision budget required')
            selected[np.flatnonzero(selected)[int(supervised_limit):]]=False
        outcome=FieldPrediction(prepared.items,prepared.input_token,result,selected,
            valid&~prepared.supervised_mask,prepared.supervised_mask&~selected,valid,True,prepared,self._owner)
        if prepared.valid_indices:self._pending=outcome
        return outcome

    def forward(self,batch,*,inference=False,supervised_limit=None):
        return self.query(self.inputs(batch),inference=inference,supervised_limit=supervised_limit)

    def loss(self,result,supervision):
        """Original loss on actual supervised rows only, after all inputs exist."""
        if (result.owner is not self._owner or supervision.input_token!=result.input_token
                or not np.array_equal(supervision.mask,result.supervised_mask)):
            raise ValueError('supervision belongs to a different formed input prefix')
        index=np.flatnonzero(result.supervised_mask)
        if not len(index):return None,{}
        take=torch.as_tensor(index,device=result.prediction.device)
        target=torch.as_tensor(supervision.target51[index],device=result.prediction.device)
        beta=torch.as_tensor(supervision.beta_target[index],device=result.prediction.device)
        with torch.autocast(device_type=result.prediction.device.type,enabled=False):
            if self.arm=='legacy_s43':return self.model._compute_loss(result.prediction[take].float(),target.float(),beta.float())
            return self.model.loss(result.prediction[take].float(),target.float(),beta_target=beta.float())

    def commit(self,result):
        if result.owner is not self._owner:raise ValueError('foreign runtime prediction')
        if not result.prepared.valid_indices:return
        if result is not self._pending:raise ValueError('duplicate/stale prediction commit')
        poses=result.prediction.detach().float().cpu().numpy()
        if poses.shape!=(len(result.items),51) or not np.isfinite(poses).all():raise ArithmeticError('invalid detached pose')
        for row,version in zip(result.prepared.valid_indices,result.prepared.state_versions):
            item=result.items[row];sid=item['seq_id'];lane=self.lanes[sid]
            if version!=(sid,lane['memory'].version,lane['pose_version'],lane['watermark']) or item['cutoff_ms']<=lane['last_commit']:
                raise ValueError('observation/context changed before commit')
        for row in result.prepared.valid_indices:
            item=result.items[row];lane=self.lanes[item['seq_id']];T=int(item['cutoff_ms'])
            lane['pose']=poses[row].copy();lane['last_commit']=T;lane['pose_version']+=1;lane['queries_committed']+=1
            lane['history']=_bounded_history(lane['history']+[(T,lane['pose'].copy())],T)
            self.audit['commits']+=1;self.audit['committed_outputs']+=1
            category=('supervised_outputs' if result.supervised_mask[row] else
                      'budget_unselected_outputs' if result.budget_unselected_mask[row] else 'warmup_outputs')
            self.audit[category]+=1
        self._prepared=self._pending=None

    def discard_query(self,result):
        """Keep the ingested prefix, discard feedback only (e.g. extra25ms query)."""
        if result.owner is not self._owner:raise ValueError('foreign runtime prediction')
        if not result.prepared.valid_indices:return
        if result is not self._pending:raise ValueError('stale/duplicate query discard')
        self._prepared=self._pending=None

    def state_dict(self,*,schedule=None,include_model=True):
        if self._prepared is not None or self._pending is not None:raise RuntimeError('checkpoint after complete prediction commit only')
        lanes={}
        for sid,lane in self.lanes.items():
            lanes[sid]=copy.deepcopy({k:v for k,v in lane.items() if k!='memory'})
            lanes[sid]['memory']=lane['memory'].state_dict()
        return dict(schema='s43_shared_field_runtime_v1',arm=self.arm,config=copy.deepcopy(self.config.__dict__),
            chunk_ms=self.chunk_ms,warmup_queries=2,lanes=lanes,audit=copy.deepcopy(self.audit),
            temporal_collapse=self.temporal_collapse,tensor_readout=self.tensor_readout,
            state_fp32=self.state_fp32,
            render_fp32=True,loss_autocast=False,
            allow_external_beta=self.allow_external_beta,
            schedule=None if schedule is None else schedule.state_dict(),
            torch_rng=torch.get_rng_state().clone(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            model_modes={name:module.training for name,module in self.model.named_modules()},
            model=None if not include_model else {k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()})

    def load_state_dict(self,state,*,schedule=None):
        if (state['schema']!='s43_shared_field_runtime_v1' or state['arm']!=self.arm or state['config']!=self.config.__dict__
                or state['warmup_queries']!=2 or state['chunk_ms']!=self.chunk_ms):
            raise ValueError('runtime checkpoint identity mismatch')
        if (state['temporal_collapse']!=self.temporal_collapse or state['tensor_readout']!=self.tensor_readout
                or state['state_fp32']!=self.state_fp32):
            raise ValueError('runtime readout/collapse checkpoint identity mismatch')
        if state['allow_external_beta']!=self.allow_external_beta:raise ValueError('conditional shape contract differs')
        if state.get('render_fp32') is not True or state.get('loss_autocast') is not False:raise ValueError('common precision contract differs')
        lanes={}
        for sid,value in state['lanes'].items():
            lane=copy.deepcopy({k:v for k,v in value.items() if k!='memory'})
            lane['memory']=TimeFieldMemory(self.config,seq_id=sid).load_state_dict(value['memory']);lanes[sid]=lane
        if schedule is not None:
            if state['schedule'] is None:raise ValueError('runtime checkpoint omitted schedule cursor')
            schedule.load_state_dict(state['schedule'])
        if state['model'] is not None:self.model.load_state_dict(state['model'],strict=True)
        for name,module in self.model.named_modules():module.training=state['model_modes'][name]
        self.lanes=lanes;self.audit=copy.deepcopy(state['audit']);self._prepared=self._pending=None;self._owner=object()
        torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda_rng'])

    def evidence(self):
        """Scientific state identity; excludes pure-query timing/audit counters."""
        result={}
        for sid,lane in self.lanes.items():
            state=lane['memory'].state_dict();h=hashlib.sha256()
            for key,value in sorted(state['arrays'].items()):h.update(key.encode());h.update(value.tobytes())
            h.update(repr(sorted(state['scalars'].items())).encode())
            result[sid]=dict(sensor_sha256=h.hexdigest(),pose=lane['pose'].copy(),last_commit=lane['last_commit'],
                core_ordinal=lane['core_ordinal'],source_end=lane['source_end'],history=[(t,p.copy()) for t,p in lane['history']])
            if lane['last_timestamp'] is not None:
                result[sid]['legacy_last_timestamp_sha256']=hashlib.sha256(lane['last_timestamp'].tobytes()).hexdigest()
        return result
