"""Native-ms streaming count field; no LNES, MANO, learned memory or sampler.

The stored resolution is explicit. At resolution2 each value is a true2x2-cell
count; grid40 uses count/4 nearest-upsampled density, not reconstructed1px data.
One dense pending row protects the oldest complete ring row while t==watermark
is incomplete. Ingest preflights sparse increments and never clones the ring.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import time

import numpy as np


MAX_COUNT=np.iinfo(np.uint32).max


def integer(value,name):
    try:ok=not isinstance(value,(bool,np.bool_)) and np.isscalar(value) and bool(np.isfinite(value)) and int(value)==value
    except (TypeError,ValueError,OverflowError):ok=False
    if not ok:raise ValueError(name+' must be a finite integer')
    return int(value)


def ro(array):
    array=np.asarray(array);array.setflags(write=False);return array


@dataclass(frozen=True)
class TimeFieldConfig:
    resolution: int=2
    width: int=240
    height: int=180
    horizon_ms: int=50
    time_bins: int=8
    max_packet_events: int=262144

    def __post_init__(self):
        for name in ('resolution','width','height','horizon_ms','time_bins','max_packet_events'):
            value=integer(getattr(self,name),name)
            if value<=0:raise ValueError(name+' must be positive')
            object.__setattr__(self,name,value)
        if self.resolution not in (1,2):raise ValueError('resolution1 or2 only')
        if self.width%4 or self.height%4:raise ValueError('sensor dimensions must divide into4px coarse cells')
        if self.time_bins>self.horizon_ms:raise ValueError('more bins than native-ms support')

    @property
    def shape(self):return self.height//self.resolution,self.width//self.resolution


@dataclass(frozen=True)
class TimeFieldSnapshot:
    config: TimeFieldConfig
    cutoff_ms: int
    boundary_ms: int
    seq_id: object
    version: int
    counts: np.ndarray
    times_ms: np.ndarray
    observed: np.ndarray
    source_range_span: np.ndarray
    source_events_per_ms: np.ndarray
    metadata: dict

    def rasterize(self):
        c=self.config;left=self.cutoff_ms-c.horizon_ms
        fine=np.zeros((c.time_bins,*c.shape,2),np.uint64)
        exposure=np.zeros(c.time_bins,np.uint16)
        for i,t in enumerate(self.times_ms):
            if self.observed[i]:
                b=(int(t)-left)*c.time_bins//c.horizon_ms
                fine[b]+=self.counts[i];exposure[b]+=1
        step=4//c.resolution
        coarse=fine.reshape(c.time_bins,c.height//4,step,c.width//4,step,2).sum((2,4),dtype=np.uint64)
        widths=np.bincount(np.arange(c.horizon_ms)*c.time_bins//c.horizon_ms,minlength=c.time_bins).astype(np.uint16)
        edges=left+(np.arange(c.time_bins+1)*c.horizon_ms+c.time_bins-1)//c.time_bins
        return dict(fine_counts=fine,coarse_counts=coarse,bin_exposure_ms=exposure,
                    bin_width_ms=widths,bin_edges_ms=edges,resolution_px=c.resolution,
                    fine_semantics='counts at declaredresolution; not nativepixel counts when resolution2')

    def to_grid40ch(self,*,log_counts=True,channels_first=True):
        c=self.config;r=self.rasterize()
        fine=r['fine_counts'].astype(np.float32)/(c.resolution*c.resolution)
        fine=fine.repeat(c.resolution,1).repeat(c.resolution,2)
        coverage=np.broadcast_to((r['bin_exposure_ms']/r['bin_width_ms']).astype(np.float32)[:,None,None],
                                 (c.time_bins,c.height,c.width)).copy()
        coarse=r['coarse_counts'].astype(np.float32).repeat(4,1).repeat(4,2)
        if log_counts:fine,coarse=np.log1p(fine),np.log1p(coarse)
        result=np.concatenate((fine.transpose(0,3,1,2).reshape(2*c.time_bins,c.height,c.width),coverage,
                               coarse.transpose(0,3,1,2).reshape(2*c.time_bins,c.height,c.width)),0).astype(np.float32,copy=False)
        return result if channels_first else result.transpose(1,2,0)

    def content_sha256(self):
        h=hashlib.sha256()
        for a in (self.counts,self.times_ms,self.observed,self.source_range_span,self.source_events_per_ms):h.update(a.tobytes())
        return h.hexdigest()


class TimeFieldMemory:
    """One independent sequence; audit timers are separate from observation state."""
    _ARRAYS=('_counts','_times','_ranges','_row_n','_pending')
    _SCALARS=('seq_id','_boundary','_watermark','_version','_last_ordinal','_last_source_end',
              '_source_count','_complete_count','_pending_time','_pending_n','_pending_lo','_pending_hi','_released')

    def __init__(self,config=TimeFieldConfig(),*,seq_id='default',boundary_ms=0):
        self.config=config;c=config
        self._counts=np.zeros((c.horizon_ms,*c.shape,2),np.uint32)
        self._times=np.full(c.horizon_ms,-1,np.int64)
        self._ranges=np.full((c.horizon_ms,2),-1,np.int64)
        self._row_n=np.zeros(c.horizon_ms,np.uint64)
        self._pending=np.zeros((*c.shape,2),np.uint32)
        self.reset(seq_id,boundary_ms)

    @property
    def watermark_ms(self):return self._watermark
    @property
    def version(self):return self._version

    def reset(self,seq_id,boundary_ms=0,*,start_ms=None):
        if not isinstance(seq_id,(str,int,np.integer)):raise ValueError('stable string/integer seq_id required')
        if start_ms is not None:
            if boundary_ms!=0:raise ValueError('choose boundary_ms or start_ms')
            boundary_ms=start_ms
        boundary_ms=integer(boundary_ms,'boundary_ms')
        if not 0<=boundary_ms<=np.iinfo(np.int64).max-self.config.horizon_ms:raise ValueError('negative/overflow boundary')
        self._counts.fill(0);self._pending.fill(0);self._times.fill(-1);self._ranges.fill(-1);self._row_n.fill(0)
        self.seq_id=int(seq_id) if isinstance(seq_id,np.integer) else seq_id
        self._boundary=self._watermark=boundary_ms
        self._version=0;self._last_ordinal=-1;self._last_source_end=0
        self._source_count=self._complete_count=0
        self._pending_time=None;self._pending_n=self._pending_lo=self._pending_hi=0;self._released=False
        self._audit=dict(ingest_calls=0,queries=0,completed_nonempty_ms=0,empty_rows_initialized=0,
                         ingest_seconds=0.,preflight_seconds=0.,flush_seconds=0.,readout_seconds=0.)
        return self

    def _check(self,seq_id=None):
        if self._released:raise RuntimeError('released field; reset before reuse')
        if seq_id is not None and seq_id!=self.seq_id:raise ValueError('mixed sequence; use another lane or reset')

    def _preflight(self,events,watermark,ordinal,source_range,seq_id):
        self._check(seq_id);c=self.config;a=np.asarray(events)
        if a.ndim==1 and a.size==0:a=np.empty((0,4),np.int64)
        if a.ndim!=2 or a.shape[1]!=4 or a.dtype.kind not in 'iu':raise ValueError('native integer N×4[x,y,t,p] required')
        if len(a)>c.max_packet_events:raise ValueError('transportpacketcap exceeded; chunk, never drop')
        if a.dtype.kind=='u' and a.size and a.max()>np.iinfo(np.int64).max:raise ValueError('native integer overflow')
        a=a.astype(np.int64,copy=False)
        wm=integer(watermark,'watermark_ms');ordinal=integer(ordinal,'packet_ordinal')
        if wm<self._watermark or wm>np.iinfo(np.int64).max-c.horizon_ms:raise ValueError('backward/overflow watermark')
        if ordinal<=self._last_ordinal:raise ValueError('duplicate/backward packet ordinal')
        if source_range is None:lo,hi=self._last_source_end,self._last_source_end+len(a)
        else:
            if len(source_range)!=2:raise ValueError('halfopen source_range required')
            lo,hi=(integer(x,'source_range') for x in source_range)
        if lo<0 or hi<lo or hi-lo!=len(a) or hi>np.iinfo(np.int64).max or (self._last_ordinal>=0 and lo!=self._last_source_end):
            raise ValueError('noncontiguous/overlapping source range; missing is not silence')
        if len(a):
            x,y,t,p=a.T
            if np.any(x<0) or np.any(x>=c.width) or np.any(y<0) or np.any(y>=c.height) or np.any((p!=0)&(p!=1)):
                raise ValueError('sensor coordinate/polarity outside dictionary')
            if np.any(t<self._watermark) or np.any(t<self._boundary) or np.any(t>wm) or np.any(t[1:]<t[:-1]):
                raise ValueError('late/future/nonmonotonic native time')
        groups=[];splits=np.r_[0,np.flatnonzero(a[1:,2]!=a[:-1,2])+1,len(a)] if len(a) else [0]
        for first,last in zip(splits[:-1],splits[1:]):
            first,last=int(first),int(last)
            g=a[first:last];t=int(g[0,2]);x,y,_,p=g.T
            cp=((y//c.resolution)*c.shape[1]+x//c.resolution)*2+p
            index,number=np.unique(cp,return_counts=True);number=number.astype(np.uint64)
            old=self._pending.ravel()[index].astype(np.uint64) if t==self._pending_time else 0
            if np.any(number+old>MAX_COUNT):raise OverflowError('uint32 declaredresolution count overflow')
            groups.append((t,index,number,last-first,lo+first,lo+last))
        return a,wm,ordinal,lo,hi,groups

    def _flush(self):
        if self._pending_time is None:return
        started=time.perf_counter();i=self._pending_time%self.config.horizon_ms
        self._counts[i]=self._pending
        self._times[i]=self._pending_time;self._ranges[i]=(self._pending_lo,self._pending_hi)
        self._row_n[i]=self._pending_n;self._complete_count+=self._pending_n
        self._pending.fill(0);self._pending_time=None;self._pending_n=0
        self._audit['completed_nonempty_ms']+=1;self._audit['flush_seconds']+=time.perf_counter()-started

    def ingest(self,events,watermark_ms=None,packet_ordinal=None,*,source_range=None,seq_id=None,valid=True,time_unit='ms'):
        if not valid:return self
        if time_unit!='ms':raise ValueError('explicit native integer-ms only')
        started=time.perf_counter()
        a,wm,ordinal,lo,hi,groups=self._preflight(events,watermark_ms,packet_ordinal,source_range,seq_id)
        preflight=time.perf_counter()-started
        # All contract/overflow failures above precede mutation. There is no
        # scorer callback or per-packet clone of the50-row field.
        for t,index,number,n,start,end in groups:
            if self._pending_time is not None and self._pending_time!=t:self._flush()
            if self._pending_time is None:
                self._pending_time=t;self._pending_lo=start;self._pending_n=0
            self._pending.ravel()[index]+=number.astype(np.uint32)
            self._pending_n+=n;self._pending_hi=end
            if t<wm:self._flush()
        if self._pending_time is not None and self._pending_time<wm:self._flush()
        # Complete zero-event exposure is explicit. At mosthorizon rows are
        # initialized after an arbitrarily long empty gap, never all past time.
        for t in range(max(self._boundary,wm-self.config.horizon_ms),wm):
            i=t%self.config.horizon_ms
            if self._times[i]!=t:
                self._counts[i].fill(0);self._times[i]=t;self._row_n[i]=0
                cursor=lo+int(np.searchsorted(a[:,2],t)) if len(a) else lo
                self._ranges[i]=(cursor,cursor);self._audit['empty_rows_initialized']+=1
        self._watermark=wm;self._last_ordinal=ordinal;self._last_source_end=hi
        self._source_count+=len(a);self._version+=1
        self._audit['ingest_calls']+=1;self._audit['preflight_seconds']+=preflight
        self._audit['ingest_seconds']+=time.perf_counter()-started
        return self

    def readout(self,cutoff_ms=None,*,seq_id=None):
        self._check(seq_id);started=time.perf_counter();c=self.config
        t=self._watermark if cutoff_ms is None else integer(cutoff_ms,'cutoff_ms')
        if t!=self._watermark:raise ValueError('only currentwatermark query; no consumedpast/future queries')
        times=np.arange(t-c.horizon_ms,t,dtype=np.int64);index=times%c.horizon_ms
        observed=(times>=self._boundary)&(self._times[index]==times)
        counts=self._counts[index];counts[~observed]=0
        span=self._ranges[index];span[~observed]=-1
        number=self._row_n[index];number[~observed]=0
        metadata=dict(resolution_px=c.resolution,native_sensor_grid=[c.height,c.width],stored_grid=list(c.shape),
            stored_cell_centers_native_px=dict(origin_xy=[(c.resolution-1)/2]*2,spacing_px=c.resolution),
            fine_value='declaredcell count/area, nearest density at originalgrid; not1pxsource reconstruction',
            coverage='complete observed native-ms at declaredspatialresolution; reset-beforetime is unobserved',
            bin_edges_ms=(t-c.horizon_ms+(np.arange(c.time_bins+1)*c.horizon_ms+c.time_bins-1)//c.time_bins).tolist(),
            ingested_events=self._source_count,complete_prefix_events=self._complete_count,
            retained_window_events=int(number.sum(dtype=np.uint64)),pending_native_time=self._pending_time,
            pending_events=self._pending_n,source_range_semantics='exactchronological native-ms span, not per-pixel identities',
            numeric_state_bytes=self.nbytes(),lnes_used=False,pose_or_label_used=False)
        result=TimeFieldSnapshot(c,t,self._boundary,self.seq_id,self._version,ro(counts),ro(times),ro(observed),ro(span),ro(number),metadata)
        self._audit['queries']+=1;self._audit['readout_seconds']+=time.perf_counter()-started
        return result

    def nbytes(self):return int(sum(getattr(self,name).nbytes for name in self._ARRAYS))

    def stats(self):
        return dict(self._audit,numeric_state_bytes=self.nbytes(),array_nbytes={k:getattr(self,k).nbytes for k in self._ARRAYS},
            ingested_events=self._source_count,complete_prefix_events=self._complete_count,watermark_ms=self._watermark,
            whole_ring_copies_per_ingest=0,pending_row_bytes=self._pending.nbytes,
            cost='sparseper-ms histpreflight; percompletedms rowcopy+pendingclear; emptyexposure atmost50rowchecks/packet; readout/grid global',
            exclusions='callerpacket, sparsepreflight/unique scratch, snapshot/grid, Pythonobjects and O(T)logs are separate')

    def state_dict(self):
        self._check()
        return dict(schema='s43_time_field_state_v1',config=asdict(self.config),
                    arrays={k:getattr(self,k).copy() for k in self._ARRAYS},
                    scalars={k:getattr(self,k) for k in self._SCALARS})

    def load_state_dict(self,state):
        if state.get('schema')!='s43_time_field_state_v1' or state.get('config')!=asdict(self.config):raise ValueError('stateconfig/schema mismatch')
        if set(state.get('arrays',{}))!=set(self._ARRAYS) or set(state.get('scalars',{}))!=set(self._SCALARS):raise ValueError('statefields mismatch')
        arrays={}
        for k in self._ARRAYS:
            v=np.asarray(state['arrays'][k]);old=getattr(self,k)
            if v.shape!=old.shape or v.dtype!=old.dtype:raise ValueError('statearray shape/dtype mismatch')
            arrays[k]=v.copy()
        s=dict(state['scalars'])
        if s['_watermark']<s['_boundary'] or (s['_pending_time'] is not None and s['_pending_time']!=s['_watermark']):raise ValueError('invalidstate clock')
        self.__dict__.update(arrays);self.__dict__.update(s)
        return self

    def release(self):self._check();self._released=True


FieldConfig=TimeFieldConfig
