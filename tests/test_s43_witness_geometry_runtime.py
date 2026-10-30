"""R1 cache/normalized-clock engineering; no target or task risk is read."""
import copy
import numpy as np
import pytest

from tools.s43_revisit.witness_geometry import MotionPolicy,PUBLIC_INITIAL
from tools.s43_revisit.witness_geometry_runtime import CacheBoundMotionPolicy
from tools.s43_revisit.witness_core import EpisodeMemory,EpisodeConfig

K=np.array([[225.,0.,120.],[0.,225.,90.],[0.,0.,1.]])


class ClockReference(MotionPolicy):
    """Frozen utility/cache with only the authorized tie-clock normalization."""
    def _hash(self,tile,epoch):return MotionPolicy._hash(tile,int(epoch)-self.ref_ms)


def policy(cls=CacheBoundMotionPolicy,ref=0,mode='witness'):
    return cls(PUBLIC_INITIAL,np.zeros(10),K,ref,mode=mode)


def block():
    a=np.zeros((10,8,8,2),np.uint32);a[:,4,4,0]=np.arange(1,11);return a


def raw_events(n=1400):
    r=np.random.default_rng(3407)
    return np.column_stack((r.integers(0,240,n),r.integers(0,180,n),np.sort(r.integers(0,60,n)),r.integers(0,2,n))).astype(np.int64)


def feed(m,raw,stop=60,chunk=7,boundary=0):
    pos=0
    for ordinal,end in enumerate(range(boundary+chunk,boundary+stop+chunk,chunk)):
        end=min(end,boundary+stop);hi=int(np.searchsorted(raw[:,2],end))
        m.ingest(raw[pos:hi],end,ordinal,source_range=(pos,hi));pos=hi
        if end==boundary+stop:break
    return m.readout()


def test_copy_discards_only_derived_cache_and_preserves_alias_caller_and_anchor():
    p=policy();expected=p.episode_score_details(block(),(120,80),0)
    before=p.state_nbytes;lut=p._lut
    a,b=copy.deepcopy((p,p))
    assert a is b and a is not p and a._lut is None and a._epoch is None
    assert p._lut is lut and p.state_nbytes==before
    assert a.state_nbytes==before-540000
    for name in ('prev','beta','K','_endpoints'):
        np.testing.assert_array_equal(getattr(a,name),getattr(p,name))
        assert not getattr(a,name).flags.writeable
    assert a.episode_score_details(block(),(120,80),0)==expected
    assert a.receipt()['runtime']['lineage_copy_dropped_bytes']==540000


def test_real_mid_epoch_commit_counts_one_warm_lut_and_respects_budget():
    p=policy();m=EpisodeMemory(EpisodeConfig(policy='witness'),policy=p).reset('budget',policy=p)
    m.ingest(np.array([[120,80,0,0]],np.int64),10,0,source_range=(0,1))
    m.ingest(np.array([[120,80,11,0]],np.int64),15,1,source_range=(1,2))
    new=policy(ref=15);new.episode_scores(block(),(120,80),15)
    m.commit(new,15)
    assert new._lut is not None  # caller remains untouched
    assert m.policy._lut is None and m._epoch_policy._lut is not None
    assert m.nbytes()<=6*2**20


@pytest.mark.parametrize('mode',['witness','quota','generic_temporal'])
def test_scalar_dictionary_exact_against_normalized_clock_reference(mode):
    c=EpisodeConfig(policy=mode,M=32,reservoir=8);raw=raw_events()
    a=feed(EpisodeMemory(c).reset('same',policy=policy(ClockReference,mode=mode)),raw)
    b=feed(EpisodeMemory(c).reset('same',policy=policy(mode=mode)),raw)
    assert a.dictionary_sha256()==b.dictionary_sha256()
    np.testing.assert_array_equal(a.to_grid40ch(),b.to_grid40ch())


@pytest.mark.parametrize('mode',['witness','quota','generic_temporal'])
def test_batch_exact_allfields_including_bottom_zeros_intmax_and_mixed_epochs(mode):
    rng=np.random.default_rng(4)
    xy=np.asarray([(8*x,8*y) for y in range(23) for x in range(30)])
    counts=rng.integers(0,31,(len(xy),10,8,8,2),dtype=np.uint32)
    counts[:5]=0;counts[5:10]=np.iinfo(np.uint32).max
    epochs=1000+(np.arange(len(xy))%5)*10
    ref=policy(ClockReference,1000,mode);batch=policy(ref=1000,mode=mode)
    expected=[ref.episode_score_details(b,t,int(e)) for b,t,e in zip(counts,xy,epochs)]
    actual=batch.episode_score_details_many(counts,xy,epochs)
    assert actual==expected
    assert batch._epoch==ref._epoch


class BatchHarness(EpisodeMemory):
    """Test-only many-callback adapter; production core is never patched."""
    def _select(self,end):
        ids=np.flatnonzero(self._u_source_n);xy=[];blocks=[];c=self.config
        for idx in ids:
            ty,tx=divmod(int(idx),c.tile_shape[1]);x,y=tx*c.tile,ty*c.tile
            b=np.zeros((c.epoch_ms,c.tile,c.tile,2),np.uint32)
            hh,ww=min(c.tile,c.height-y),min(c.tile,c.width-x);b[:,:hh,:ww]=self._U[:,y:y+hh,x:x+ww]
            xy.append((x,y));blocks.append(b)
        if not blocks:return super()._select(end)
        original=self._epoch_policy
        details=original.episode_score_details_many(np.stack(blocks),np.asarray(xy),self._epoch_start)
        lookup=dict(zip(xy,details))
        class Lookup:
            def episode_score_details(self,b,t,e):return lookup[t]
        self._epoch_policy=Lookup()
        try:return super()._select(end)
        finally:self._epoch_policy=original


def test_batch_callback_harness_preserves_selected_ids_counts_and_priority():
    c=EpisodeConfig(policy='witness',M=32,reservoir=8);raw=raw_events()
    a=feed(EpisodeMemory(c).reset('x',policy=policy()),raw)
    b=feed(BatchHarness(c).reset('x',policy=policy()),raw)
    assert a.dictionary_sha256()==b.dictionary_sha256()
    np.testing.assert_array_equal(a.blocks,b.blocks)


def test_non_aligned_boundary3_uses_public_ref3_without_rounding():
    p=policy(ref=3);m=EpisodeMemory(EpisodeConfig(policy='witness')).reset('b3',boundary_ms=3,policy=p)
    m.ingest(np.array([[120,80,3,0]],np.int64),13,0,source_range=(100,101))
    np.testing.assert_array_equal(m.readout().epoch_start_ms,[3])


def test_policy_clock_shift_plus1000_is_exact_with_amended_reference():
    xy=np.array([(120,80),(232,176),(0,0)]);counts=np.stack((block(),block(),block()))
    a=policy(ref=0);b=policy(ref=1000)
    assert a.episode_score_details_many(counts,xy,0)==b.episode_score_details_many(counts,xy,1000)


def test_real_memory_clock_shift_respects_relative_core_source_keys():
    raw=raw_events();c=EpisodeConfig(policy='witness',M=32,reservoir=8)
    a=feed(EpisodeMemory(c).reset('A',policy=policy()),raw)
    shifted=raw.copy();shifted[:,2]+=1000
    b=feed(EpisodeMemory(c).reset('different_lane',boundary_ms=1000,policy=policy(ref=1000)),shifted,boundary=1000)
    np.testing.assert_array_equal(a.blocks,b.blocks)
    np.testing.assert_array_equal(a.owner,b.owner)
    np.testing.assert_array_equal(a.priority,b.priority)
    np.testing.assert_array_equal(a.epoch_start_ms,b.epoch_start_ms-1000)
    np.testing.assert_array_equal(a.to_grid40ch(),b.to_grid40ch())


def test_cache_dropping_checkpoint_retains_future_selection():
    raw=raw_events();c=EpisodeConfig(policy='witness',M=32,reservoir=8)
    a=EpisodeMemory(c).reset('restore',policy=policy());pos=int(np.searchsorted(raw[:,2],25))
    a.ingest(raw[:pos],25,0,source_range=(0,pos));state=a.state_dict()
    b=EpisodeMemory(c).load_state_dict(state)
    for m in (a,b):m.ingest(raw[pos:],60,1,source_range=(pos,len(raw)))
    assert a.readout().dictionary_sha256()==b.readout().dictionary_sha256()
