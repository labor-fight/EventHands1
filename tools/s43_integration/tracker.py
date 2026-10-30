"""Separate event memory, pure output query, and explicit hand-state commit."""
from __future__ import annotations
import time
import numpy as np
import torch
from .memory import SequenceBank


class S43StreamTracker:
    def __init__(self,model,*,max_sequences=16,unsupported_policy='hold'):
        if unsupported_policy not in ('hold','raise'): raise ValueError(unsupported_policy)
        self.model=model.eval();self.bank=SequenceBank(model.mode,max_sequences=max_sequences)
        self.unsupported_bank=SequenceBank('legacy',max_sequences=max_sequences)
        self.unsupported_policy=unsupported_policy
        self.states={};self.timing={'network_seconds':0.,'commit_seconds':0.,'queries':0,'commits':0}

    def _start_epoch(self,sid,start,hand):
        try:
            self.bank.reset(sid,start,hand['pose'],hand['betas'],hand['K_cam'],
                            segment_id=hand['segment_id'],history_time_us=hand['pose_time_us'])
            if sid in self.unsupported_bank.states:self.unsupported_bank.release(sid)
            hand['supported']=True;hand['unsupported_reason']=None
        except (ValueError,ArithmeticError) as error:
            if self.model.mode!='br' or self.unsupported_policy=='raise': raise
            if sid in self.bank.states:self.bank.release(sid)
            self.unsupported_bank.reset(sid,start,hand['pose'],hand['betas'],hand['K_cam'],
                    segment_id=hand['segment_id'],history_time_us=hand['pose_time_us'])
            hand['supported']=False;hand['unsupported_reason']=str(error)
        hand['start_us']=int(start)
        hand['epoch_closed']=False

    def reset(self,seq_id,start_us,state,betas,K_cam,*,segment_id=0,state_time_us=None):
        def arr(a):return a.detach().cpu().numpy().copy() if hasattr(a,'detach') else np.array(a,copy=True)
        hand={'pose':arr(state).reshape(51).astype(np.float32),'betas':arr(betas).reshape(10),
              'K_cam':arr(K_cam).reshape(3,3),'pose_time_us':int(start_us-1000 if state_time_us is None else state_time_us),
              'segment_id':segment_id,'version':0,'pending':None}
        self._start_epoch(seq_id,int(start_us),hand);self.states[seq_id]=hand
        return self

    def _memory(self,sid):
        return self.bank if self.states[sid]['supported'] else self.unsupported_bank

    def ingest(self,seq_id,events,source_ids,watermark_us,padding_mask=None):
        if self.states[seq_id]['epoch_closed']: raise ValueError('committed epoch is closed')
        return self._memory(seq_id).ingest(seq_id,events,source_ids,watermark_us,padding_mask)

    def representation(self,seq_id,cutoff_us):
        if self.states[seq_id]['epoch_closed']: raise ValueError('committed epoch is closed')
        rep=self._memory(seq_id).readout(seq_id,cutoff_us)
        if not self.states[seq_id]['supported']:
            rep['meta']['status']='unsupported'
            rep['meta']['unsupported_reason']=self.states[seq_id]['unsupported_reason']
            rep['meta']['policy']='hold hand state; observations still counted, no raw-LNES fallback'
        return rep

    @torch.no_grad()
    def query(self,seq_id,cutoff_us):
        """Prediction may be read repeatedly; hand state/version never changes."""
        hand=self.states[seq_id];rep=self.representation(seq_id,cutoff_us)
        device=next(self.model.parameters()).device
        prev=torch.as_tensor(hand['pose'][None],device=device,dtype=torch.float32)
        beta=torch.as_tensor(hand['betas'][None],device=device,dtype=torch.float32)
        camera=torch.as_tensor(hand['K_cam'][None],device=device,dtype=torch.float32)
        t0=time.perf_counter();prediction=self.model([rep],prev,beta,camera)
        elapsed=time.perf_counter()-t0
        if not torch.isfinite(prediction).all():raise ArithmeticError('nonfinite S43 prediction')
        self.timing['network_seconds']+=elapsed;self.timing['queries']+=1
        result={'state':prediction[0].detach().cpu().numpy().copy(),'seq_id':seq_id,
                'cutoff_us':int(cutoff_us),'target_time_us':rep['meta']['target_time_us'],
                'state_version':hand['version'],'representation_meta':rep['meta'],
                'network_seconds':elapsed}
        hand['pending']=result
        return result

    def commit(self,seq_id,result,*,begin_next=True):
        """A fixed epoch-end commit, never an implicit side effect of query."""
        t0=time.perf_counter();hand=self.states[seq_id]
        if (hand['epoch_closed'] or result['seq_id']!=seq_id or result['state_version']!=hand['version'] or
                result['cutoff_us']!=hand['start_us']+50000 or hand['pending'] is None or
                result is not hand['pending']):
            raise ValueError('stale, unrelated or non-epoch-boundary state commit')
        pose=np.asarray(result['state'],np.float32)
        if pose.shape!=(51,) or not np.isfinite(pose).all():raise ValueError('finite committed state[51] required')
        hand['pose']=pose.copy()
        hand['pose_time_us']=int(result['target_time_us']);hand['version']+=1;hand['pending']=None
        hand['epoch_closed']=True
        if begin_next:self._start_epoch(seq_id,int(result['cutoff_us']),hand)
        self.timing['commit_seconds']+=time.perf_counter()-t0;self.timing['commits']+=1
        return hand['pose'].copy()

    def release(self,seq_id):
        if seq_id in self.bank.states:self.bank.release(seq_id)
        if seq_id in self.unsupported_bank.states:self.unsupported_bank.release(seq_id)
        del self.states[seq_id]
