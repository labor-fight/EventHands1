"""Bounded native-ms episode memory, independent of LNES and MANO internals.

Selection considers retained episodes plus the newly sealed epoch. Rejected
history is never resurrected: this is NOT exact sliding-window top-K or an exact
window reservoir. All samplers share this streaming rule, expiry and capacity.
Source ranges are uniqueness contracts; per-episode ranges are envelopes, not
lists of every source row. True equal-payload multiplicity is always retained.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time

import numpy as np


UINT32_MAX = np.iinfo(np.uint32).max
POLICIES = ("uniform", "recency", "density", "quota", "generic_temporal", "witness")


def _int(x, name):
    try:
        valid = not isinstance(x, (bool, np.bool_)) and np.isscalar(x) and bool(np.isfinite(x)) and int(x)==x
    except (TypeError,ValueError,OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be an integer")
    return int(x)


def _readonly(a):
    a = np.asarray(a)
    a.setflags(write=False)
    return a


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))


@dataclass(frozen=True)
class EpisodeConfig:
    width: int = 240
    height: int = 180
    epoch_ms: int = 10
    horizon_ms: int = 50
    tile: int = 8
    M: int = 128
    reservoir: int = 32
    coarse_cell: int = 4
    time_bins: int = 8
    seed: int = 3407
    policy: str = "witness"
    max_packet_events: int = 262144

    def __post_init__(self):
        for name in ("width", "height", "epoch_ms", "horizon_ms", "tile", "M",
                     "coarse_cell", "time_bins", "max_packet_events"):
            if _int(getattr(self, name), name) <= 0:
                raise ValueError(name + " must be positive")
        if _int(self.reservoir, "reservoir") < 0 or self.reservoir > self.M:
            raise ValueError("reservoir must be in [0,M]")
        _int(self.seed, "seed")
        if self.width % self.coarse_cell or self.height % self.coarse_cell:
            raise ValueError("coarse_cell must divide the sensor dimensions")
        if self.horizon_ms < self.epoch_ms or self.time_bins > self.horizon_ms:
            raise ValueError("invalid horizon or time_bins")
        if self.policy not in POLICIES:
            raise ValueError("unknown selection policy")

    @property
    def tile_shape(self):
        return (math.ceil(self.height/self.tile), math.ceil(self.width/self.tile))

    @property
    def quotas(self):
        base, rest = divmod(self.M-self.reservoir, 5)
        return tuple(base + (j < rest) for j in range(5))


@dataclass(frozen=True)
class EpisodeSnapshot:
    config: EpisodeConfig
    cutoff_ms: int
    seq_id: object
    version: int
    pose_version: int
    blocks: np.ndarray
    epoch_start_ms: np.ndarray
    tile_xy: np.ndarray
    observed_start_ms: np.ndarray
    observed_end_ms: np.ndarray
    owner: np.ndarray
    priority: np.ndarray
    sign: np.ndarray
    sign_known: np.ndarray
    selection_kind: np.ndarray
    selection_pose_version: np.ndarray
    hash_key: np.ndarray
    source_range_envelope: np.ndarray
    source_events: np.ndarray
    partial_counts: np.ndarray
    partial_epoch_start_ms: int
    partial_observed_start_ms: int
    partial_observed_end_ms: int
    coarse_counts: np.ndarray
    coarse_times_ms: np.ndarray
    coarse_observed: np.ndarray
    metadata: dict

    @property
    def episodes(self):
        """Primitive arrays; padding is outside the physical image, not silence."""
        return dict(counts=self.blocks, tile_xy=self.tile_xy, epoch_start_ms=self.epoch_start_ms,
                    observed_start_ms=self.observed_start_ms, observed_end_ms=self.observed_end_ms,
                    source_range_envelope=self.source_range_envelope, source_events=self.source_events,
                    spatial_observed_mask=self.episode_spatial_valid)

    @property
    def episode_spatial_valid(self):
        y = self.tile_xy[:,1,None,None]+np.arange(self.config.tile)[None,:,None]
        x = self.tile_xy[:,0,None,None]+np.arange(self.config.tile)[None,None,:]
        return _readonly((y<self.config.height)&(x<self.config.width))

    @property
    def coarse(self):
        return self.coarse_counts

    @property
    def partial(self):
        return self.partial_counts

    def dictionary_sha256(self, *, content_only=False):
        digest = hashlib.sha256()
        for array in (self.epoch_start_ms, self.tile_xy, self.observed_start_ms,
                      self.observed_end_ms, self.blocks):
            digest.update(array.tobytes())
        if not content_only:
            for array in (self.owner, self.priority, self.sign, self.sign_known,
                          self.selection_kind, self.selection_pose_version,
                          self.source_range_envelope, self.source_events):
                digest.update(array.tobytes())
        return digest.hexdigest()

    def content_sha256(self):
        digest = hashlib.sha256(self.dictionary_sha256(content_only=True).encode())
        for array in (self.partial_counts, self.coarse_counts, self.coarse_times_ms,
                      self.coarse_observed):
            digest.update(array.tobytes())
        digest.update(str((self.cutoff_ms,self.partial_epoch_start_ms,
                           self.partial_observed_start_ms,self.partial_observed_end_ms)).encode())
        return digest.hexdigest()

    def rasterize(self):
        """Return exact uint64 bin sums and fine exposure in native-ms units."""
        c = self.config
        left = self.cutoff_ms-c.horizon_ms
        fine = np.zeros((c.time_bins,c.height,c.width,2),np.uint64)
        exposure = np.zeros((c.time_bins,c.height,c.width),np.uint16)
        coarse = np.zeros((c.time_bins,c.height//c.coarse_cell,c.width//c.coarse_cell,2),np.uint64)
        coarse_exposure = np.zeros(c.time_bins,np.uint16)

        def add(block, xy, epoch, start, end):
            x,y = map(int,xy)
            hh,ww = min(block.shape[1],c.height-y),min(block.shape[2],c.width-x)
            for t in range(max(int(start),left),min(int(end),self.cutoff_ms)):
                b = (t-left)*c.time_bins//c.horizon_ms
                fine[b,y:y+hh,x:x+ww] += block[t-epoch,:hh,:ww]
                exposure[b,y:y+hh,x:x+ww] += 1

        for i in range(len(self.blocks)):
            add(self.blocks[i],self.tile_xy[i],int(self.epoch_start_ms[i]),
                int(self.observed_start_ms[i]),int(self.observed_end_ms[i]))
        add(self.partial_counts,(0,0),self.partial_epoch_start_ms,
            self.partial_observed_start_ms,self.partial_observed_end_ms)
        for i,t in enumerate(self.coarse_times_ms):
            if self.coarse_observed[i]:
                b = (int(t)-left)*c.time_bins//c.horizon_ms
                coarse[b] += self.coarse_counts[i]
                coarse_exposure[b] += 1
        widths = np.bincount(np.arange(c.horizon_ms)*c.time_bins//c.horizon_ms,
                             minlength=c.time_bins).astype(np.uint16)
        return dict(fine=fine, fine_exposure_ms=exposure, coarse=coarse,
                    coarse_exposure_ms=coarse_exposure, bin_width_ms=widths,
                    bin_edges_ms=left+(np.arange(c.time_bins+1)*c.horizon_ms+c.time_bins-1)//c.time_bins)

    def to_grid40ch(self, *, log_counts=True, channels_first=True):
        """fine16, exposure8, coarse16; nearest coarse counts are not fine data.

        Default log1p is an explicit feature transform. Raw exact count getters
        remain available; float32 grids cannot preserve arbitrary uint32 sums.
        """
        r = self.rasterize(); c = self.config
        fine = r['fine'].astype(np.float32)
        coarse = r['coarse'].astype(np.float32)
        if log_counts:
            fine,coarse = np.log1p(fine),np.log1p(coarse)
        coverage = r['fine_exposure_ms'].astype(np.float32)/r['bin_width_ms'][:,None,None]
        coarse = coarse.repeat(c.coarse_cell,1).repeat(c.coarse_cell,2)
        result = np.concatenate((fine.transpose(0,3,1,2).reshape(-1,c.height,c.width),
                                 coverage,
                                 coarse.transpose(0,3,1,2).reshape(-1,c.height,c.width)),0)
        return result if channels_first else result.transpose(1,2,0)


class EpisodeMemory:
    """Opaque scorer callback; the core imports no geometry, model or target."""

    _ARRAYS = ("_U","_Q","_Q_times","_pending_coarse","_blocks","_valid","_epoch","_tile",
               "_obs_start","_obs_end","_owner","_priority","_sign","_sign_known",
               "_pose","_kind","_hash","_source_min","_source_max","_source_n",
               "_u_source_min","_u_source_max","_u_source_n")

    def __init__(self, config=EpisodeConfig(), *, policy=None):
        self.config = config
        c = config; nt = int(np.prod(c.tile_shape))
        self._U = np.zeros((c.epoch_ms,c.height,c.width,2),np.uint32)
        self._Q = np.zeros((c.horizon_ms,c.height//c.coarse_cell,c.width//c.coarse_cell,2),np.uint32)
        self._Q_times = np.full(c.horizon_ms,-1,np.int64)
        self._pending_coarse = np.zeros(self._Q.shape[1:],np.uint32)
        self._blocks = np.zeros((c.M,c.epoch_ms,c.tile,c.tile,2),np.uint32)
        self._valid = np.zeros(c.M,bool)
        self._epoch = np.zeros(c.M,np.int64); self._tile = np.zeros((c.M,2),np.int32)
        self._obs_start = np.zeros(c.M,np.int64); self._obs_end = np.zeros(c.M,np.int64)
        self._owner = np.zeros(c.M,np.int8); self._priority = np.zeros(c.M,np.float64)
        self._sign = np.zeros(c.M,np.int8); self._sign_known = np.zeros(c.M,bool)
        self._pose = np.zeros(c.M,np.int64); self._kind = np.zeros(c.M,np.uint8)
        self._hash = np.zeros(c.M,np.uint64)
        self._source_min = np.full(c.M,-1,np.int64); self._source_max = np.full(c.M,-1,np.int64)
        self._source_n = np.zeros(c.M,np.uint64)
        self._u_source_min = np.full(nt,-1,np.int64); self._u_source_max = np.full(nt,-1,np.int64)
        self._u_source_n = np.zeros(nt,np.uint64)
        self.reset("default",policy=policy)

    @property
    def watermark_ms(self): return self._watermark
    @property
    def version(self): return self._version
    @property
    def pose_version(self): return self._pose_version
    @property
    def policy(self): return self._policy

    def reset(self, seq_id, boundary_ms=0, public_state=None, beta=None, K=None, *, policy=None, start_ms=None):
        if not isinstance(seq_id,(str,int,np.integer)):
            raise ValueError("seq_id must be a stable string or integer")
        if start_ms is not None:
            if boundary_ms != 0: raise ValueError("use boundary_ms or start_ms")
            boundary_ms = start_ms
        boundary_ms = _int(boundary_ms,"boundary_ms")
        if boundary_ms < 0: raise ValueError("negative sequence boundary")
        for name in self._ARRAYS: getattr(self,name).fill(0)
        for name in ("_Q_times","_source_min","_source_max","_u_source_min","_u_source_max"):
            getattr(self,name).fill(-1)
        self.seq_id = int(seq_id) if isinstance(seq_id,np.integer) else seq_id
        self._boundary = self._watermark = boundary_ms
        # The fixed epoch grid is anchored at the sequence's declared boundary.
        self._epoch_start = boundary_ms
        self._pending_time = None; self._pending_n = 0; self._u_events = 0
        self._last_ordinal = -1; self._last_source_end = 0
        self._source_count = 0; self._sealed_source_count = 0
        self._version = self._pose_version = self._commit_count = 0
        self._released = False; self._policy = copy.deepcopy(policy)
        self._epoch_policy = self._policy; self._epoch_pose_version = 0
        self._last_commit_ms = boundary_ms
        self._stats = dict(ingest_calls=0,queries=0,sealed_epochs=0,nonempty_candidates=0,
                           sign_balanced_selections=0,sign_unavailable_selections=0,
                           ingest_seconds=0.,selection_seconds=0.,query_seconds=0.,commit_seconds=0.)
        return self

    def _check(self,seq_id=None):
        if self._released: raise RuntimeError("released memory; reset before reuse")
        if seq_id is not None and seq_id != self.seq_id: raise ValueError("mixed sequence")

    def _key(self,epoch,tile):
        # Lane/sequence IDs isolate mutable state; they are NOT observation
        # features or sampling identities. Train/online naming must not alter
        # selection for identical native times/content/seed/context.
        return int.from_bytes(hashlib.blake2b(_json((self.config.seed,int(epoch)-self._boundary,
            tuple(map(int,tile)))).encode(),digest_size=8).digest(),"little")

    def _parse(self,events,watermark_ms,packet_ordinal,source_range,seq_id):
        self._check(seq_id); c = self.config
        raw = np.asarray(events)
        if raw.size == 0: raw = np.empty((0,4),np.int64)
        if raw.ndim != 2 or raw.shape[1] != 4: raise ValueError("events must be N×4 [x,y,t,p]")
        if len(raw) > c.max_packet_events: raise ValueError("transport packet cap exceeded; chunk without dropping events")
        if raw.dtype.kind not in "iu" or raw.dtype.kind == "b": raise ValueError("native integer events required")
        if raw.dtype.kind == 'u' and raw.size and raw.max() > np.iinfo(np.int64).max:
            raise ValueError("event integer overflow")
        raw = raw.astype(np.int64,copy=False)
        wm = _int(watermark_ms,"watermark_ms"); ordinal = _int(packet_ordinal,"packet_ordinal")
        if wm>np.iinfo(np.int64).max-c.horizon_ms-c.epoch_ms:
            raise ValueError("watermark integer overflow")
        if wm < self._watermark: raise ValueError("watermark cannot move backward")
        if ordinal <= self._last_ordinal: raise ValueError("duplicate/backward packet ordinal")
        if source_range is None: lo,hi = self._last_source_end,self._last_source_end+len(raw)
        else:
            if len(source_range)!=2: raise ValueError("half-open source_range required")
            lo,hi = (_int(v,"source_range") for v in source_range)
        if lo<0 or (self._last_ordinal>=0 and lo!=self._last_source_end) or hi<lo or hi-lo!=len(raw) or hi>np.iinfo(np.int64).max:
            raise ValueError("source range must be contiguous after first packet; gaps need an explicit missing contract")
        if len(raw):
            x,y,t,p = raw.T
            if np.any(x<0) or np.any(x>=c.width) or np.any(y<0) or np.any(y>=c.height):
                raise ValueError("event outside sensor")
            if np.any((p!=0)&(p!=1)): raise ValueError("explicit polarity slots0/1 required")
            if np.any(t<self._watermark) or np.any(t<self._boundary) or np.any(t>wm):
                raise ValueError("late event or future event beyond one incomplete native group")
            if np.any(t[1:]<t[:-1]): raise ValueError("native timestamps must be nondecreasing")
        # Sparse per-group histograms validate BOTH fine and coarse counters
        # before any mutation. Equal payloads are counted, never deduplicated.
        groups = []
        splits = np.r_[0,np.flatnonzero(raw[1:,2]!=raw[:-1,2])+1,len(raw)] if len(raw) else [0]
        for a,b in zip(splits[:-1],splits[1:]):
            group = raw[a:b]; t = int(group[0,2]); x,y,_,p = group.T
            cp = (y*c.width+x)*2+p
            index,count = np.unique(cp,return_counts=True); count = count.astype(np.uint64)
            existing = self._U[t-self._epoch_start].ravel()[index].astype(np.uint64) if self._epoch_start<=t<self._epoch_start+c.epoch_ms else 0
            if np.any(count+existing>UINT32_MAX): raise OverflowError("fine uint32 count overflow")
            coarse_cp = ((y//c.coarse_cell)*(c.width//c.coarse_cell)+x//c.coarse_cell)*2+p
            ci,cn = np.unique(coarse_cp,return_counts=True); cn = cn.astype(np.uint64)
            old = 0
            if self._pending_time == t:
                old = self._pending_coarse.ravel()[ci].astype(np.uint64)
            if np.any(cn+old>UINT32_MAX): raise OverflowError("coarse uint32 count overflow")
            tile_index = (y//c.tile)*c.tile_shape[1]+x//c.tile
            groups.append((t,index,count,ci,cn,tile_index,np.arange(lo+a,lo+b,dtype=np.int64)))
        return raw,wm,ordinal,lo,hi,groups

    def _fork(self):
        other = copy.copy(self)
        for name in self._ARRAYS: setattr(other,name,getattr(self,name).copy())
        other._policy,other._epoch_policy = copy.deepcopy((self._policy,self._epoch_policy))
        other._stats = self._stats.copy()
        return other

    def ingest(self,events,watermark_ms=None,packet_ordinal=None,*,source_range=None,seq_id=None,valid=True,time_unit="ms"):
        if not valid: return self  # padding reads no payload, sequence or source
        if time_unit != "ms": raise ValueError("this prototype accepts native integer ms only")
        started = time.perf_counter()
        raw,wm,ordinal,lo,hi,groups = self._parse(events,watermark_ms,packet_ordinal,source_range,seq_id)
        # Scorer/selection can fail. Only sealing packets need a bounded full
        # transaction; ordinary native-group additions are overflow-prevalidated.
        target = self._fork() if wm>=self._epoch_start+self.config.epoch_ms else self
        for t,index,count,ci,cn,tile_index,source_ids in groups:
            if target._pending_time is not None and target._pending_time<t: target._flush_pending()
            target._advance(t)
            row = target._U[t-target._epoch_start].ravel()
            row[index] += count.astype(np.uint32)
            target._pending_coarse.ravel()[ci] += cn.astype(np.uint32)
            target._pending_time = t; target._u_events += len(source_ids); target._pending_n += len(source_ids)
            tiles,inverse,number = np.unique(tile_index,return_inverse=True,return_counts=True)
            minimum = np.full(len(tiles),np.iinfo(np.int64).max,np.int64)
            maximum = np.full(len(tiles),-1,np.int64)
            np.minimum.at(minimum,inverse,source_ids); np.maximum.at(maximum,inverse,source_ids)
            old = target._u_source_min[tiles]
            target._u_source_min[tiles] = np.where(old<0,minimum,np.minimum(old,minimum))
            target._u_source_max[tiles] = np.maximum(target._u_source_max[tiles],maximum+1)
            target._u_source_n[tiles] += number.astype(np.uint64)
            if t<wm: target._flush_pending()
        if target._pending_time is not None and target._pending_time<wm: target._flush_pending()
        target._advance(wm)
        target._expire(wm)
        for t in range(max(target._boundary,wm-target.config.horizon_ms),wm):
            i = t%target.config.horizon_ms
            if target._Q_times[i]!=t:
                target._Q[i].fill(0); target._Q_times[i]=t
        target._watermark = wm; target._last_ordinal = ordinal; target._last_source_end = hi
        target._source_count += len(raw); target._version += 1
        target._stats['ingest_calls'] += 1
        target._stats['ingest_seconds'] += time.perf_counter()-started
        if target is not self: self.__dict__.update(target.__dict__)
        return self

    def _flush_pending(self):
        t = self._pending_time
        if t is None: return
        c = self.config
        # Fine/coarse increments were independently overflow-prevalidated.
        # No full fine-row reduction: only copy/clear the coarse pending row.
        self._Q[t%c.horizon_ms] = self._pending_coarse
        self._Q_times[t%c.horizon_ms] = t
        self._sealed_source_count += self._pending_n
        self._pending_coarse.fill(0); self._pending_n = 0
        self._pending_time = None

    def _expire(self,t):
        self._valid &= self._obs_end>t-self.config.horizon_ms

    def _advance(self,t):
        c = self.config
        while self._epoch_start+c.epoch_ms<=t:
            if self._pending_time is not None: self._flush_pending()
            end = self._epoch_start+c.epoch_ms
            self._expire(end)
            if self._u_events:
                self._select(end)
                self._U.fill(0); self._u_source_min.fill(-1); self._u_source_max.fill(-1); self._u_source_n.fill(0)
                self._u_events = 0
                self._epoch_start = end
            else:
                # No unsealed content: jumping an empty gap is O(horizon),
                # not proportional to the entire elapsed duration.
                self._epoch_start = self._boundary+(t-self._boundary)//c.epoch_ms*c.epoch_ms
            self._epoch_policy = self._policy; self._epoch_pose_version = self._pose_version
            self._stats['sealed_epochs'] += 1

    def _entries(self):
        return [dict(block=self._blocks[i],epoch=int(self._epoch[i]),tile=tuple(map(int,self._tile[i])),
            obs_start=int(self._obs_start[i]),obs_end=int(self._obs_end[i]),owner=int(self._owner[i]),
            priority=float(self._priority[i]),sign=int(self._sign[i]),sign_known=bool(self._sign_known[i]),
            pose=int(self._pose[i]),key=int(self._hash[i]),source_min=int(self._source_min[i]),
            source_max=int(self._source_max[i]),source_n=int(self._source_n[i])) for i in np.flatnonzero(self._valid)]

    def _score(self,block,tile,epoch,key):
        mode = self.config.policy
        if mode in ("uniform","recency","density"):
            return key%5, float(block.sum(dtype=np.uint64)), 0, False
        policy = self._epoch_policy
        if policy is None: raise RuntimeError(mode+" requires a committed episode scorer")
        if hasattr(policy,"episode_score_details"):
            value = policy.episode_score_details(block,tile,epoch)
            owner,priority,sign = value['owner'],value['priority'],value.get('sign',0)
            known = 'sign' in value
        else:
            owner,priority = policy.episode_scores(block,tile,epoch)
            sign,known = 0,False
        owner,sign = _int(owner,"owner"),_int(sign,"sign")
        priority = float(priority)
        if owner<0 or owner>=5 or sign not in (-1,0,1) or not math.isfinite(priority) or priority<0:
            raise ValueError("invalid selector output")
        if priority == 0: owner = key%5; sign=0
        return owner,priority,sign,known

    def _select(self,end):
        started = time.perf_counter(); c = self.config
        candidates = self._entries()
        for tile_index in np.flatnonzero(self._u_source_n):
            ty,tx = divmod(int(tile_index),c.tile_shape[1]); x,y = tx*c.tile,ty*c.tile
            block = np.zeros((c.epoch_ms,c.tile,c.tile,2),np.uint32)
            hh,ww = min(c.tile,c.height-y),min(c.tile,c.width-x)
            block[:,:hh,:ww] = self._U[:,y:y+hh,x:x+ww]
            key = self._key(self._epoch_start,(x,y))
            owner,priority,sign,known = self._score(block,(x,y),self._epoch_start,key)
            candidates.append(dict(block=block,epoch=self._epoch_start,tile=(x,y),obs_start=max(self._epoch_start,self._boundary),
                obs_end=end,owner=owner,priority=priority,sign=sign,sign_known=known,pose=self._epoch_pose_version,key=key,
                source_min=int(self._u_source_min[tile_index]),source_max=int(self._u_source_max[tile_index]),
                source_n=int(self._u_source_n[tile_index])))
        self._stats['nonempty_candidates'] += int(np.count_nonzero(self._u_source_n))
        hashed = sorted(range(len(candidates)),key=lambda i:(candidates[i]['key'],candidates[i]['epoch'],candidates[i]['tile']))
        chosen = hashed[:c.reservoir]; kinds = {i:1 for i in chosen}
        remaining = [i for i in hashed if i not in kinds]
        if c.policy == "uniform": priority_order = remaining
        elif c.policy == "recency": priority_order = sorted(remaining,key=lambda i:(-candidates[i]['epoch'],candidates[i]['key']))
        elif c.policy == "density": priority_order = sorted(remaining,key=lambda i:(-candidates[i]['source_n'],candidates[i]['key']))
        elif c.policy == "generic_temporal": priority_order = sorted(remaining,key=lambda i:(-candidates[i]['priority'],candidates[i]['key']))
        else:
            priority_order = []
            balanced = c.policy=="witness" and all(v['sign_known'] for v in candidates)
            self._stats['sign_balanced_selections' if balanced else 'sign_unavailable_selections'] += 1
            for owner,quota in enumerate(c.quotas):
                owned = [i for i in remaining if candidates[i]['owner']==owner]
                buckets = [(owned,quota)] if not balanced else [([i for i in owned if candidates[i]['sign']>0],(quota+1)//2),
                                                               ([i for i in owned if candidates[i]['sign']<0],quota//2)]
                for indices,cap in buckets:
                    priority_order.extend(sorted(indices,key=lambda i:(-candidates[i]['priority'],candidates[i]['key']))[:cap])
        for i in priority_order[:c.M-c.reservoir]:
            if i not in kinds: chosen.append(i);kinds[i]=2
        for i in hashed:
            if len(chosen)>=c.M: break
            if i not in kinds: chosen.append(i);kinds[i]=4
        # Canonical storage order makes hashes independent of heap/loop order.
        chosen.sort(key=lambda i:(candidates[i]['epoch'],candidates[i]['tile']))
        blocks = np.zeros_like(self._blocks)
        records = [(candidates[i],kinds[i]) for i in chosen]
        for k,(v,_) in enumerate(records): blocks[k]=v['block']
        self._blocks = blocks; self._valid.fill(False)
        for k,(v,kind) in enumerate(records):
            self._valid[k]=True; self._epoch[k]=v['epoch']; self._tile[k]=v['tile']
            self._obs_start[k]=v['obs_start']; self._obs_end[k]=v['obs_end']
            self._owner[k]=v['owner']; self._priority[k]=v['priority']; self._sign[k]=v['sign']
            self._sign_known[k]=v['sign_known']; self._pose[k]=v['pose']; self._kind[k]=kind; self._hash[k]=v['key']
            self._source_min[k]=v['source_min']; self._source_max[k]=v['source_max']; self._source_n[k]=v['source_n']
        self._stats['selection_seconds'] += time.perf_counter()-started

    def commit(self,policy=None,cutoff_ms=None,*,expected_version=None):
        self._check(); started = time.perf_counter()
        cutoff_ms = self._watermark if cutoff_ms is None else _int(cutoff_ms,"cutoff_ms")
        if cutoff_ms!=self._watermark: raise ValueError("commit requires current prefix")
        if expected_version is not None and expected_version!=self._version: raise ValueError("stale query commit")
        self._policy = copy.deepcopy(policy); self._pose_version += 1; self._commit_count += 1
        if self._epoch_start==cutoff_ms:
            self._epoch_policy = self._policy; self._epoch_pose_version = self._pose_version
        self._last_commit_ms = cutoff_ms; self._version += 1
        self._stats['commit_seconds'] += time.perf_counter()-started
        return self

    set_policy = commit

    def readout(self,cutoff_ms=None,*,seq_id=None):
        self._check(seq_id); started = time.perf_counter(); c=self.config
        t = self._watermark if cutoff_ms is None else _int(cutoff_ms,"cutoff_ms")
        if t!=self._watermark: raise ValueError("readout supports only current watermark, not consumed past or future")
        ids = np.flatnonzero(self._valid)
        times = np.arange(t-c.horizon_ms,t,dtype=np.int64)
        qi = times%c.horizon_ms
        observed = (times>=self._boundary)&(self._Q_times[qi]==times)
        coarse = self._Q[qi].copy(); coarse[~observed]=0
        partial = self._U.copy()
        # Never expose an incomplete t==watermark group through raw partial.
        visible_rows = (np.arange(c.epoch_ms)+self._epoch_start>=self._boundary)&(np.arange(c.epoch_ms)+self._epoch_start<t)
        partial[~visible_rows]=0
        values = [getattr(self,name)[ids].copy() for name in ("_blocks","_epoch","_tile","_obs_start","_obs_end","_owner",
            "_priority","_sign","_sign_known","_kind","_pose","_hash")]
        envelope = np.stack((self._source_min[ids],self._source_max[ids]),-1)
        metadata = dict(policy=c.policy,retained_episodes=len(ids),capacity=c.M,reservoir=c.reservoir,quotas=c.quotas,
            native_time_unit='integer ms',incomplete_native_group=self._pending_time,source_ingested=self._source_count,
            source_complete_prefix=self._sealed_source_count,source_range_semantics='envelope only; uniqueness enforced by packet/source-range contract',
            selection='streaming retained+new, not exact full-window topK/reservoir',selection_kind_bits={1:'hash_reserve',2:'priority',4:'hash_fill'},
            policy_metadata=copy.deepcopy(getattr(self._policy,'metadata',{})),resident_bytes=self.nbytes(),
            partial_selection_pose_version=self._epoch_pose_version,
            partial_epoch_policy_metadata=copy.deepcopy(getattr(self._epoch_policy,'metadata',{})),
            bin_edges_ms=(t-c.horizon_ms+(np.arange(c.time_bins+1)*c.horizon_ms+c.time_bins-1)//c.time_bins).tolist(),
            count_transform='to_grid40ch defaults log1p; raw integer arrays retained',coverage='retained fine native-ms exposure / bin width; missing != silence',
            coarse='full observed counts, nearest upsample replicated cell totals; never add to fine as new events')
        snapshot = EpisodeSnapshot(c,t,self.seq_id,self._version,self._pose_version,*map(_readonly,values),
            _readonly(envelope),_readonly(self._source_n[ids].copy()),_readonly(partial),self._epoch_start,
            max(self._epoch_start,self._boundary),min(t,self._epoch_start+c.epoch_ms),
            _readonly(coarse),_readonly(times),_readonly(observed),metadata)
        self._stats['queries'] += 1; self._stats['query_seconds'] += time.perf_counter()-started
        return snapshot

    def nbytes(self,*,include_policy=True):
        result = sum(getattr(self,name).nbytes for name in self._ARRAYS)
        if include_policy:
            policies = {id(p):p for p in (self._policy,self._epoch_policy) if p is not None}
            for policy in policies.values():
                value = getattr(policy,'state_nbytes',0)
                result += int(value() if callable(value) else value)
        return int(result)

    def stats(self):
        return dict(self._stats,resident_bytes=self.nbytes(),count_array_bytes=self._U.nbytes+self._Q.nbytes+self._blocks.nbytes,
            array_nbytes={name:getattr(self,name).nbytes for name in self._ARRAYS},
            numeric_state_bytes=self.nbytes(include_policy=False),policy_array_bytes=self.nbytes()-self.nbytes(include_policy=False),
            retained_episodes=int(self._valid.sum()),ingested_events=self._source_count,complete_prefix_events=self._sealed_source_count,
            commit_count=self._commit_count,watermark_ms=self._watermark,
            exclusions='Python object/metadata overhead, caller packets, bounded transaction/selection/query scratch and O(T) output logs separate')

    def state_dict(self):
        """Full bounded checkpoint, including opaque deepcopy-able scorer state."""
        self._check()
        scalars = copy.deepcopy({k:v for k,v in self.__dict__.items() if k not in self._ARRAYS and k!='config'})
        return dict(schema='s43_witness_memory_v1',config=asdict(self.config),scalars=scalars,
                    arrays={name:getattr(self,name).copy() for name in self._ARRAYS})

    def load_state_dict(self,state):
        if state.get('schema')!='s43_witness_memory_v1' or state.get('config')!=asdict(self.config):
            raise ValueError("checkpoint schema/config mismatch")
        if set(state['arrays'])!=set(self._ARRAYS): raise ValueError("checkpoint array schema mismatch")
        restored = {}
        for name in self._ARRAYS:
            value = np.asarray(state['arrays'][name]); old = getattr(self,name)
            if value.shape!=old.shape or value.dtype!=old.dtype: raise ValueError("checkpoint shape/dtype mismatch")
            restored[name]=value.copy()
        scalars=copy.deepcopy(state['scalars'])
        if scalars.get('_watermark',-1)<scalars.get('_boundary',0): raise ValueError("invalid checkpoint clock")
        self.__dict__.update(scalars); self.__dict__.update(restored)
        return self

    def release(self):
        self._check(); self._released=True
