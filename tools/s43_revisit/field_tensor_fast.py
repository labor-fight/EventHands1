"""Independent exact integer readout adapter; frozen core/tensorizer unchanged.

Build one owned int64 batch directly, then use the original Torch formulas,
operation order and dtypes. The memory-direct path reads only completed ring
rows at the current watermark, never the pending native-ms group. No aliases
of mutable count storage escape and no memory/source/pose/GT state is changed.
"""
from __future__ import annotations
import numpy as np
import torch
from .field_core import TimeFieldConfig,TimeFieldMemory,TimeFieldSnapshot
from .field_tensor import TensorField


def _config(values):
    if not values:raise ValueError('nonempty same-config batch required')
    c=values[0].config
    if any(x.config!=c for x in values):raise ValueError('same count configuration required')
    if (c.width,c.height,c.time_bins,c.horizon_ms)!=(240,180,8,50) or c.resolution not in (1,2):
        raise ValueError('fast adapter fixes240x180,eight bins in50native-ms,resolution1/2')
    return c


def _counter_domain(counts,shape):
    if counts.shape!=shape:raise ValueError('native count ring shape differs')
    if counts.dtype!=np.uint32:
        if counts.dtype.kind in 'iu' and (np.any(counts<0) or counts.max(initial=0)>np.iinfo(np.uint32).max):
            raise OverflowError('outside frozen uint32 native-count domain')
        raise ValueError('fast adapter requires frozen uint32 native counts')
    # The declared input domain proves signed64 aggregation cannot overflow:
    # <=50 native rows *16 native pixels *uint32_max < int64_max. No data scan
    # or uint64 full-batch intermediary is needed for this immutable dtype.


def _arrays(B,c):
    integer=np.zeros((B,c.time_bins,*c.shape,2),np.int64)
    exposure=np.zeros((B,c.time_bins),np.int64)
    widths=np.tile(np.bincount(np.arange(c.horizon_ms)*c.time_bins//c.horizon_ms,
                              minlength=c.time_bins).astype(np.int64),(B,1))
    return integer,exposure,widths


def _to_torch(integer,exp,widths,c,device):
    """Copied original arithmetic/order, fed equal contiguous int64 arrays."""
    if np.any(widths<=0) or np.any(exp>widths):raise ValueError('invalid native exposure')
    payload=int(integer.nbytes+exp.nbytes+widths.nbytes)
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        counts=torch.as_tensor(integer,device=device)
        exposure=torch.as_tensor(exp,device=device).float()/torch.as_tensor(widths,device=device).float()
        B,T,H,W,P=counts.shape
        step=4//c.resolution
        coarse=counts.reshape(B,T,45,step,60,step,P).sum((3,5))
        density=counts.float().permute(0,1,4,2,3)/(c.resolution**2)
        logfine=torch.log1p(density)
        cells=torch.cat((logfine,exposure[:,:,None,None,None].expand(B,T,1,H,W)),2)
        f=logfine.repeat_interleave(c.resolution,-2).repeat_interleave(c.resolution,-1)
        q=torch.log1p(coarse.float()).permute(0,1,4,2,3).repeat_interleave(4,-2).repeat_interleave(4,-1)
        coverage=exposure[:,:,None,None].expand(B,T,180,240)
        grid=torch.cat((f.flatten(1,2),coverage,q.flatten(1,2)),1)
        fine_support=(counts.sum((1,4))>0)[:,None].repeat_interleave(c.resolution,-2).repeat_interleave(c.resolution,-1)
        coarse_support=(coarse.sum((1,4))>0)[:,None].repeat_interleave(4,-2).repeat_interleave(4,-1)
        support=fine_support|coarse_support
    return TensorField(grid,cells,support,payload,int(grid.numel()*grid.element_size()))


def aggregate_snapshots(snapshots):
    """Public engineering integer boundary; returns owned counts/exposure/widths."""
    values=list(snapshots)
    if not all(isinstance(s,TimeFieldSnapshot) for s in values):raise ValueError('TimeFieldSnapshot required')
    c=_config(values);integer,exp,widths=_arrays(len(values),c)
    for lane,s in enumerate(values):
        _counter_domain(s.counts,(c.horizon_ms,*c.shape,2))
        if s.times_ms.shape!=(c.horizon_ms,) or s.observed.shape!=(c.horizon_ms,):raise ValueError('snapshot native-clock shape')
        left=int(s.cutoff_ms)-c.horizon_ms
        for i in np.flatnonzero(s.observed):
            t=int(s.times_ms[i])
            if t<left or t>=s.cutoff_ms:raise ValueError('observed row outside current half-open window')
            b=(t-left)*c.time_bins//c.horizon_ms
            integer[lane,b]+=s.counts[i];exp[lane,b]+=1
    return integer,exp,widths,c


def _sum_run(ring,start,stop,out,*,add):
    """A logical native-ms run crosses the physical50-row ring at most once."""
    span=stop-start;first=start%50;end=min(50,first+span)
    if add:out+=ring[first:end].sum(axis=0,dtype=np.int64)
    else:np.sum(ring[first:end],axis=0,dtype=np.int64,out=out)
    remaining=span-(end-first)
    if remaining:out+=ring[:remaining].sum(axis=0,dtype=np.int64)


def aggregate_memories(memories,cutoffs=None):
    """Read current complete ring prefix, pending group and future excluded.

    Caller must not ingest/reset the same memory concurrently with this pure
    read. All sums are written to independent arrays before returning.
    """
    values=list(memories)
    if not all(isinstance(m,TimeFieldMemory) for m in values):raise ValueError('frozen TimeFieldMemory instances required')
    c=_config(values);B=len(values)
    if cutoffs is None:cutoffs=np.asarray([m.watermark_ms for m in values],np.int64)
    else:
        a=np.asarray(cutoffs)
        if a.ndim==0:a=np.repeat(a,B)
        if a.shape!=(B,) or a.dtype.kind not in 'iu':raise ValueError('native integer current cutoff scalar or[B] required')
        cutoffs=a.astype(np.int64,copy=False)
    integer,exp,widths=_arrays(B,c)
    for lane,(memory,T) in enumerate(zip(values,cutoffs)):
        memory._check();T=int(T)
        if T!=memory.watermark_ms:raise ValueError('only current complete watermark may be read')
        _counter_domain(memory._counts,(c.horizon_ms,*c.shape,2))
        left=T-c.horizon_ms
        edges=left+(np.arange(c.time_bins+1)*c.horizon_ms+c.time_bins-1)//c.time_bins
        for b,(a,z) in enumerate(zip(edges[:-1],edges[1:])):
            times=np.arange(a,z,dtype=np.int64)
            observed=(times>=memory._boundary)&(memory._times[times%c.horizon_ms]==times)
            exp[lane,b]=int(observed.sum())
            if not observed.any():continue
            # Real complete-prefix exposure is a contiguous suffix after a
            # boundary. Retain exact handling of holes in restored/test rings.
            starts=np.flatnonzero(observed&~np.r_[False,observed[:-1]])
            ends=np.flatnonzero(observed&~np.r_[observed[1:],False])+1
            for run,(lo,hi) in enumerate(zip(starts,ends)):
                _sum_run(memory._counts,int(times[lo]),int(times[hi-1])+1,integer[lane,b],add=run>0)
    return integer,exp,widths,c


def tensorize(snapshots,device='cpu'):
    integer,exp,widths,c=aggregate_snapshots(snapshots)
    return _to_torch(integer,exp,widths,c,device)


def tensorize_memories(memories,cutoffs=None,device='cpu'):
    integer,exp,widths,c=aggregate_memories(memories,cutoffs)
    return _to_torch(integer,exp,widths,c,device)
