"""Native S43 serving control, without an unused dense field audit ring.

Only the bounded original per-pixel last-event timestamps are maintained.
The same immutable batch, forward, own-state commit and precision contracts
as SharedS43Runtime are used. No target or auxiliary shape is read here.
"""
from __future__ import annotations
import hashlib
import numpy as np
from .field_core import integer
from .s43_field_runtime import SharedS43Runtime


class SourceClock:
    """Fixed-size source/version metadata; no event history or field array."""
    def __init__(self):self.version=0;self._pending_time=None
    def release(self):self.version+=1
    def nbytes(self):return 0


class NativeS43Runtime(SharedS43Runtime):
    def __init__(self,model,**kwargs):
        if kwargs.pop('arm','legacy_s43')!='legacy_s43':raise ValueError('native serving is only original S43')
        super().__init__(model,arm='legacy_s43',**kwargs)

    def _reset(self,item):
        sid=item['seq_id']
        if sid in self.lanes:raise ValueError('reset cannot overwrite a live sequence')
        p=np.array(item['initial_state'],np.float32,copy=True);b=np.array(item['beta_input'],np.float32,copy=True)
        K=np.array(item['K_source'],np.float32,copy=True);start=integer(item['run_start_ms'],'run_start')
        if p.shape!=(51,) or b.shape!=(10,) or K.shape!=(3,3) or not np.isfinite(np.r_[p,b,K.ravel()]).all():raise ValueError('invalid public context')
        if np.any(b) and not self.allow_external_beta:raise ValueError('nonzero shape requires explicit external condition')
        self.lanes[sid]=dict(memory=SourceClock(),pose=p,beta=b,K=K,history=[(start,p.copy())],
            watermark=start,last_commit=start,last_logical_ordinal=-1,core_ordinal=0,source_end=None,
            pose_version=0,queries_committed=0,last_timestamp=np.full((180,240,2),-1,np.int64))
        self.audit['resets']+=1

    def _ingest(self,item,*,tail=False):
        lane=self.lanes[item['seq_id']];e=np.asarray(item['events'])
        T=integer(item['cutoff_ms'],'cutoff');start=integer(item['start_ms'],'start')
        lo,hi=(integer(x,'source_range') for x in item['source_range']);ordinal=integer(item['packet_ordinal'],'ordinal')
        if start!=lane['watermark'] or T<=start or ordinal!=lane['last_logical_ordinal']+1:raise ValueError('continuous source clock/ordinal required')
        if lane['source_end'] is not None and lo!=lane['source_end']:raise ValueError('source rows must be exactly continuous')
        if e.ndim!=2 or e.shape[1]!=4 or e.dtype.kind not in 'iu':raise ValueError('integer native Nx4 required')
        if e.dtype.kind=='u' and e.size and e.max()>np.iinfo(np.int64).max:raise ValueError('integer overflow')
        e=e.astype(np.int64,copy=False)
        if lo<0 or hi-lo!=len(e) or hi>np.iinfo(np.int64).max:raise ValueError('source range/payload mismatch')
        if len(e):
            x,y,t,p=e.T
            if (np.any(x<0) or np.any(x>=240) or np.any(y<0) or np.any(y>=180) or np.any((p!=0)&(p!=1))
                or np.any(t[1:]<t[:-1]) or np.any(t<start) or np.any(t>=T)):raise ValueError('native events outside half-open source prefix')
        # Validate the entire logical packet before modifying observation state.
        # Preserve exactly-once native-ms transport and never crop event payloads.
        for a in range(start,T,self.chunk_ms):
            b=min(a+self.chunk_ms,T);left=int(np.searchsorted(e[:,2],a));right=int(np.searchsorted(e[:,2],b))
            packet=e[left:right]
            if len(packet):np.maximum.at(lane['last_timestamp'],(packet[:,1],packet[:,0],packet[:,3]),packet[:,2])
            lane['core_ordinal']+=1
        lane['memory'].version+=1;lane['watermark']=T;lane['source_end']=hi;lane['last_logical_ordinal']=ordinal
        self.audit['ingested_events']+=len(e)
        if tail:self.audit['tail_events']+=len(e);self.audit['tail_ms']+=T-start
        h=hashlib.sha256(self.audit['source_chain'].encode());h.update(str((item['seq_id'],lo,hi,start,T)).encode());h.update(e.tobytes())
        self.audit['source_chain']=h.hexdigest()

    def evidence(self):
        return {sid:dict(legacy_last_timestamp_sha256=hashlib.sha256(l['last_timestamp'].tobytes()).hexdigest(),
            pose=l['pose'].copy(),last_commit=l['last_commit'],core_ordinal=l['core_ordinal'],source_end=l['source_end'],
            history=[(t,p.copy()) for t,p in l['history']]) for sid,l in self.lanes.items()}

    def state_dict(self,**kwargs):raise NotImplementedError('native serving is a reset/ingest/query/commit client, not a training-resume format')
