"""R1 cache adapter; geometric utility unchanged, tie clock amended pre-task.

Only regenerable epoch-LUT caches are omitted by deepcopy into memory.
Optional batches preserve reference reduction and normalized-clock ties;
true old absolute-time ties may differ under the separate frozen amendment.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
from pathlib import Path
import numpy as np

from . import witness_geometry as reference

REFERENCE_SHA='9d56ee645aebd789beee48d7b185765f1c189ee1f721a78477ad49b64fe11dd5'
CLOCK_AMENDMENT_SHA='e25dda82c31b48bea93ea8574dc86f410e563029087483b8c2117e27df14b4f9'


@lru_cache(maxsize=1)
def _identities():
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(reference.__file__)!=REFERENCE_SHA:raise RuntimeError('R1 requires exact frozen reference geometry')
    amendment=reference.ROOT/'outputs/s43_revisit_20261028/stage2_pre_task_clock_amendment.json'
    if sha(amendment)!=CLOCK_AMENDMENT_SHA:raise RuntimeError('runtime requires frozen pre-task clock amendment')
    return dict(reference_sha256=REFERENCE_SHA,runtime_sha256=sha(__file__),clock_amendment_sha256=CLOCK_AMENDMENT_SHA)


class CacheBoundMotionPolicy(reference.MotionPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.metadata['runtime']=dict(backend='cache_bound_normalized_clock_R1',**_identities(),
            cache_rule='deepcopy omits only derived_lut/epoch; caller unchanged',
            batch_rule='same point-major float reduction; normalized-clock reference ties',
            tie_rule='reference hash(tile,epoch-ref_ms); pre-task clock amendment, not old absolute-time tie parity')
        self._runtime_copy_cache_drops=0;self._runtime_copy_dropped_bytes=0
        self._runtime_batch_calls=0;self._runtime_batch_episodes=0;self._runtime_batch_workspace_bytes=0

    def __deepcopy__(self,memo):
        out=type(self).__new__(type(self));memo[id(self)]=out
        for key,value in self.__dict__.items():
            setattr(out,key,None if key in ('_lut','_epoch') else copy.deepcopy(value,memo))
        dropped=0 if self._lut is None else self._lut.nbytes
        out._runtime_copy_cache_drops+=int(dropped>0)
        out._runtime_copy_dropped_bytes+=dropped
        # Copying must not accidentally make immutable anchor arrays writable.
        for name in ('prev','beta','K','_endpoints','_static_lut'):
            value=getattr(out,name,None)
            if isinstance(value,np.ndarray):value.flags.writeable=False
        return out

    def _hash(self,tile,epoch):
        return reference.MotionPolicy._hash(tile,int(epoch)-self.ref_ms)

    def episode_score_details_many(self,counts,tile_xy,epoch_start_ms):
        """Exact reference scores; inputs orderedN, outputs list[small dict].

        Derived kernels may be shared within an epoch; they remain one bounded
        cache.  Generic covariance uses the scalar reference to avoid changing
        its norm reduction.  No core hook is installed by importing this file.
        """
        b=np.asarray(counts);tile=np.asarray(tile_xy)
        if b.ndim!=5 or b.shape[1:]!=(10,8,8,2) or not np.issubdtype(b.dtype,np.integer) or np.any(b<0):
            raise ValueError('nonnegative integer counts[N,10,8,8,2] required')
        if np.any(b>np.iinfo(np.uint32).max):raise OverflowError('episode count exceeds uint32')
        N=len(b)
        if tile.shape!=(N,2) or not np.isfinite(tile).all() or np.any(tile!=np.floor(tile)) or np.any(tile%8):
            raise ValueError('tile_xy[N,2]: integer8px aligned origins required')
        if np.any(tile<0) or np.any(tile[:,0]>=240) or np.any(tile[:,1]>=180):raise ValueError('tile origin outside sensor')
        epoch=np.asarray(epoch_start_ms)
        if epoch.ndim==0:epoch=np.repeat(epoch,N)
        if epoch.shape!=(N,) or not np.isfinite(epoch).all() or np.any(epoch!=np.floor(epoch)) or np.any(epoch<self.ref_ms):
            raise ValueError('native epoch start[N] before committed anchor or noninteger')
        if np.any(epoch>np.iinfo(np.int64).max):raise ValueError('epoch exceeds native clock range')
        epoch=epoch.astype(np.int64);tile=tile.astype(np.int64)
        self._runtime_batch_calls+=1;self._runtime_batch_episodes+=N
        if not N:return []
        if self.mode=='generic_temporal':
            return [super(CacheBoundMotionPolicy,self).episode_score_details(v,xy,int(t)) for v,xy,t in zip(b,tile,epoch)]
        result=[None]*N
        for t in dict.fromkeys(epoch.tolist()):
            ids=np.flatnonzero(epoch==t);xy=tile[ids]
            n=b[ids].astype(np.float64).sum(-1)
            valid=((xy[:,1,None,None]+np.arange(8)[None,:,None]<180)&
                   (xy[:,0,None,None]+np.arange(8)[None,None,:]<240))
            n*=valid[:,None]
            total=n.sum((1,2,3));rows=(xy[:,1,None]+np.arange(8))//4;cols=(xy[:,0,None]+np.arange(8))//4
            if self.mode=='witness':
                lookup=self._epoch_kernel(int(t))[:,:,np.minimum(rows,44)[:,:,None],cols[:,None,:]].transpose(2,0,1,3,4)
                centered=n-n.mean(axis=1,keepdims=True)
                product=lookup*centered[:,None]
                v=product.sum((2,3,4))/np.sqrt(1+total)[:,None]
            else:
                lookup=self._static_lut[:,np.minimum(rows,44)[:,:,None],cols[:,None,:]].transpose(1,0,2,3)
                centered=n.sum(1);product=lookup*centered[:,None]
                v=product.sum((2,3))/np.sqrt(1+total)[:,None]
            live=sum(a.nbytes for a in (ids,xy,n,valid,total,rows,cols,lookup,centered,product,v))
            self._runtime_batch_workspace_bytes=max(self._runtime_batch_workspace_bytes,live)
            for j,i in enumerate(ids):
                key=self._hash(xy[j],int(t));best=float(np.abs(v[j]).max());ties=np.flatnonzero(np.abs(v[j])==best)
                owner=int(ties[key%len(ties)])
                sign=int(np.sign(v[j,owner])) if self.mode=='witness' else ((1 if key&8 else -1) if best else 0)
                result[int(i)]=dict(owner=owner,priority=best,sign=sign,
                    signed_score=float(v[j,owner]) if self.mode=='witness' else sign*best)
        if self.mode=='witness':self._epoch_kernel(int(epoch[-1]))
        return result

    def receipt(self):
        result=super().receipt()
        result['runtime']=dict(**_identities(),copy_cache_drops=self._runtime_copy_cache_drops,
            lineage_copy_dropped_bytes=self._runtime_copy_dropped_bytes,batch_calls=self._runtime_batch_calls,
            batch_episodes=self._runtime_batch_episodes,batch_visible_array_workspace_max_bytes=self._runtime_batch_workspace_bytes,
            workspace_scope='live named batch temporaries only; excludes input/core stack,Python objects and hidden NumPy scratch; not measured RSS peak',
            counter_scope='copy-drop counts follow this copy lineage; not a process-wide total')
        return result
