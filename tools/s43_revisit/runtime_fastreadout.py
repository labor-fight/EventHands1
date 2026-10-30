"""Independent full-execution adapter; frozen ingest/model/state math unchanged.

Copied R1 input assembly with two changes: lane outputs retain a memory handle,
and one native-ring integer readout batches those handles. No observation is
mutated by the getter; the returned field owns independent buffers.
"""
import copy,hashlib
import numpy as np
import torch
from .s43_field_runtime_R1 import SharedS43RuntimeR1
from .s43_field_runtime import SharedS43Runtime as ReferenceRuntime,PreparedFieldInputs,temporal_collapse_field,_legacy_image_from_times
from .s43_field_data import FieldInputBatch,make_input_batch

READOUT_BACKEND='native_ring_bins_v1'

class SharedS43RuntimeFastReadout(SharedS43RuntimeR1):
    def __init__(self,model,**kwargs):
        super().__init__(model,**kwargs)
        if not self.tensor_readout:raise ValueError('fast native getter requires shared tensor readout')

    def _lane_input(self,item):
        proxy=copy.copy(self);proxy.audit=self.audit.copy();proxy.lanes={item['seq_id']:self.lanes[item['seq_id']]}
        ReferenceRuntime._ingest(proxy,item)
        lane=proxy.lanes[item['seq_id']]
        if self.arm=='legacy_s43':return None,_legacy_image_from_times(lane['last_timestamp'],item['cutoff_ms'])
        return lane['memory'],None

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
            from .field_tensor_fast import tensorize_memories
            field_tensor=tensorize_memories(fields,device=self.device).global40
        else:field_tensor=tensor(fields,(40,180,240))
        if self.temporal_collapse:field_tensor=temporal_collapse_field(field_tensor,native_r2=True)
        prepared=PreparedFieldInputs(batch.items,batch.token,tuple(live),field_tensor,tensor(previous,(51,)),tensor(beta,(10,)),tensor(K,(3,3)),np.asarray(refs,np.int64),np.asarray(cutoffs,np.int64),tuple(histories),tensor(images,(180,240,2)) if self.arm=='legacy_s43' else None,supervised,tuple(versions))
        if live:self._prepared=prepared
        return prepared

    def state_dict(self,*,schedule=None,include_model=True):
        state=super().state_dict(schedule=schedule,include_model=include_model)
        state['readout_backend']=READOUT_BACKEND
        return state

    def load_state_dict(self,state,*,schedule=None):
        if state.get('readout_backend')!=READOUT_BACKEND:raise ValueError('native-ring readout execution identity')
        super().load_state_dict(state,schedule=schedule)
