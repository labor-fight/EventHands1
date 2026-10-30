"""R1 execution wrapper: independent lanes run concurrently, collection is ordered.

The frozen integer core and ingest kernel are called without modification. No
network/RNG work occurs in the pool. Audit source hashes are reduced in the
original row order, rather than in worker-completion order.
"""
from concurrent.futures import ThreadPoolExecutor
import copy, hashlib
import numpy as np
import torch
from .s43_field_runtime import (SharedS43Runtime as ReferenceRuntime,
    PreparedFieldInputs, temporal_collapse_field, _legacy_image_from_times)
from .s43_field_data import FieldInputBatch, make_input_batch


class SharedS43RuntimeR1(ReferenceRuntime):
    def __init__(self,model,*,runtime_workers=1,**kwargs):
        super().__init__(model,**kwargs)
        if runtime_workers not in (1,16):raise ValueError('fixed16 production workers; 1 is engineering reference')
        self.runtime_workers=int(runtime_workers)
        self.pose_diagnostics=dict(committed_rows=0,root_z_min_m=None,root_z_max_m=None,root_nonpositive_rows=0)
        self._pool=ThreadPoolExecutor(max_workers=self.runtime_workers,thread_name_prefix='field-lane') if runtime_workers>1 else None

    def _lane_input(self,item):
        # Each job owns a distinct lane and its numeric buffers. The proxy only
        # separates the otherwise global audit accumulator; it does not clone
        # the core/history or change any per-sequence kernel.
        proxy=copy.copy(self);proxy.audit=self.audit.copy();proxy.lanes={item['seq_id']:self.lanes[item['seq_id']]}
        ReferenceRuntime._ingest(proxy,item)
        lane=proxy.lanes[item['seq_id']]
        if self.arm=='legacy_s43':field=None;image=_legacy_image_from_times(lane['last_timestamp'],item['cutoff_ms'])
        else:
            snap=lane['memory'].readout(item['cutoff_ms']);field=snap if self.tensor_readout else snap.to_grid40ch();image=None
        return field,image

    def inputs(self,batch):
        if not isinstance(batch,FieldInputBatch):raise TypeError('FieldInputBatch required')
        if make_input_batch(batch.items,releases=batch.releases,batch_id=batch.batch_id).token!=batch.token:raise ValueError('input batch mutated')
        if self._prepared is not None:
            if batch.token==self._prepared.input_token:return self._prepared
            raise RuntimeError('commit the previous output before advancing another prefix')
        live=[i for i,item in enumerate(batch) if item.get('valid',True) and not item.get('padding',False)]
        if len({batch[i]['seq_id'] for i in live})!=len(live):raise ValueError('parallel rows require independent sequence lanes')
        # Release/tail chronology is unchanged and precedes the next batch.
        for release in batch.releases:
            sid=release['seq_id']
            if sid not in self.lanes:raise ValueError('release of unknown sequence')
            if release['events'] is not None:self._ingest(release,tail=True)
            self.lanes[sid]['memory'].release();del self.lanes[sid];self.audit['releases']+=1
        for i in live:
            item=batch[i];sid=item['seq_id']
            if set(item)&{'target','targets','beta_target','betas','lnes','previous_GT'}:raise ValueError('supervision or old image in input record')
            if item.get('reset',False):self._reset(item)
            if sid not in self.lanes:raise ValueError('sequence requires a boundary reset')
            lane=self.lanes[sid]
            if not np.array_equal(lane['beta'],item['beta_input']) or not np.array_equal(lane['K'],item['K_source']):raise ValueError('external beta/K changed inside a run; reset/version it explicitly')
        items=[batch[i] for i in live]
        jobs=list(self._pool.map(self._lane_input,items)) if self._pool is not None else [self._lane_input(x) for x in items]
        fields=[];previous=[];beta=[];K=[];refs=[];cutoffs=[];histories=[];images=[];versions=[]
        supervised=np.zeros(len(batch),bool)
        for i,(field,image) in zip(live,jobs):
            item=batch[i];sid=item['seq_id'];lane=self.lanes[sid];events=np.asarray(item['events']).astype(np.int64,copy=False)
            lo,hi=(int(x) for x in item['source_range']);start=int(item['start_ms']);T=int(item['cutoff_ms'])
            self.audit['ingested_events']+=len(events)
            h=hashlib.sha256(self.audit['source_chain'].encode());h.update(str((sid,lo,hi,start,T)).encode());h.update(events.tobytes());self.audit['source_chain']=h.hexdigest()
            if field is not None:fields.append(field)
            if image is not None:images.append(image)
            previous.append(lane['pose']);beta.append(lane['beta']);K.append(lane['K']);refs.append(lane['last_commit']);cutoffs.append(T)
            histories.append(tuple((t,p.copy()) for t,p in lane['history']))
            versions.append((sid,lane['memory'].version,lane['pose_version'],lane['watermark']))
            supervised[i]=lane['queries_committed']>=self.warmup_queries
        tensor=lambda values,shape:torch.as_tensor(np.stack(values) if values else np.empty((0,*shape),np.float32),device=self.device,dtype=torch.float32)
        if self.tensor_readout and fields:
            from .field_tensor import tensorize
            field_tensor=tensorize(fields,self.device).global40
        else:field_tensor=tensor(fields,(40,180,240))
        if self.temporal_collapse:field_tensor=temporal_collapse_field(field_tensor,native_r2=True)
        prepared=PreparedFieldInputs(batch.items,batch.token,tuple(live),field_tensor,tensor(previous,(51,)),tensor(beta,(10,)),tensor(K,(3,3)),np.asarray(refs,np.int64),np.asarray(cutoffs,np.int64),tuple(histories),tensor(images,(180,240,2)) if self.arm=='legacy_s43' else None,supervised,tuple(versions))
        if live:self._prepared=prepared
        return prepared

    def commit(self,result):
        super().commit(result)
        indices=result.prepared.valid_indices
        if indices:
            z=result.prediction.detach().float().cpu().numpy()[list(indices),2]
            d=self.pose_diagnostics;d['committed_rows']+=len(z);d['root_nonpositive_rows']+=int((z<=0).sum())
            d['root_z_min_m']=float(z.min()) if d['root_z_min_m'] is None else min(d['root_z_min_m'],float(z.min()))
            d['root_z_max_m']=float(z.max()) if d['root_z_max_m'] is None else max(d['root_z_max_m'],float(z.max()))

    def state_dict(self,*,schedule=None,include_model=True):
        state=super().state_dict(schedule=schedule,include_model=include_model)
        state['runtime_variant']='ordered_lane_parallel_R1';state['runtime_workers']=self.runtime_workers
        state['pose_diagnostics']=copy.deepcopy(self.pose_diagnostics)
        state['model_diagnostics']=self.model.diagnostic_state_dict() if hasattr(self.model,'diagnostic_state_dict') else None
        return state

    def load_state_dict(self,state,*,schedule=None):
        if state.get('runtime_variant')!='ordered_lane_parallel_R1' or state.get('runtime_workers')!=self.runtime_workers:raise ValueError('R1 parallel runtime checkpoint identity')
        super().load_state_dict(state,schedule=schedule)
        self.pose_diagnostics=copy.deepcopy(state['pose_diagnostics'])
        if state.get('model_diagnostics') is not None:self.model.load_diagnostic_state_dict(state['model_diagnostics'])

    def close(self):
        if self._pool is not None:self._pool.shutdown(wait=True);self._pool=None
