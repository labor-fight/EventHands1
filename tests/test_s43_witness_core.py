"""Independent raw-count and temporal/source contracts; no GT/task scoring."""
import copy
import pickle

import numpy as np
import pytest

from tools.s43_revisit.witness_core import EpisodeConfig, EpisodeMemory, UINT32_MAX


class ToyPolicy:
    def __init__(self,scale=1.,ref=0):
        self.scale=scale;self.ref=ref
        self.metadata={'scale':scale,'ref':ref,'engineering_only':True}
        self.owned=np.zeros(4,np.float32)

    @property
    def state_nbytes(self):return self.owned.nbytes

    def episode_score_details(self,counts,tile_xy,epoch_start_ms):
        if epoch_start_ms<self.ref:raise ValueError('future anchor used on old epoch')
        signed=float(np.einsum('thwp,t->',counts.astype(float),np.arange(10)-4.5))
        return dict(owner=(tile_xy[0]//8+3*(tile_xy[1]//8))%5,
                    priority=abs(signed)*self.scale,sign=int(np.sign(signed)))


class BadPolicy:
    def episode_score_details(self,*args):raise ValueError('injected selector failure')


def events(duration=80,n=1500,seed=3407,width=16,height=16):
    rng=np.random.default_rng(seed)
    t=np.sort(rng.integers(0,duration,n))
    return np.column_stack((rng.integers(0,width,n),rng.integers(0,height,n),t,rng.integers(0,2,n))).astype(np.int64)


def feed(memory,raw,stop,chunk):
    offset=ordinal=0
    for end in range(chunk,stop+chunk,chunk):
        end=min(end,stop)
        hi=int(np.searchsorted(raw[:,2],end))
        memory.ingest(raw[offset:hi],end,ordinal,source_range=(offset,hi))
        ordinal+=1;offset=hi
        if end==stop:break
    return memory.readout(stop)


def arrays_equal(a,b):
    for name in EpisodeMemory._ARRAYS:np.testing.assert_array_equal(getattr(a,name),getattr(b,name))


def test_full_capacity_matches_independent_raw_raster_and_coarse():
    raw=events();c=EpisodeConfig(width=16,height=16,policy='uniform')
    snap=feed(EpisodeMemory(c).reset('raw'),raw,80,7)
    r=snap.rasterize();reference=np.zeros((8,16,16,2),np.uint64)
    coarse=np.zeros((8,4,4,2),np.uint64)
    take=raw[(raw[:,2]>=30)&(raw[:,2]<80)]
    bins=(take[:,2]-30)*8//50
    np.add.at(reference,(bins,take[:,1],take[:,0],take[:,3]),1)
    np.add.at(coarse,(bins,take[:,1]//4,take[:,0]//4,take[:,3]),1)
    np.testing.assert_array_equal(r['fine'],reference)
    np.testing.assert_array_equal(r['coarse'],coarse)
    assert np.all(r['fine_exposure_ms']==r['bin_width_ms'][:,None,None])
    assert int(snap.source_events.sum())<=len(raw)


@pytest.mark.parametrize('mode',['uniform','recency','density','quota','generic_temporal','witness'])
def test_chunks_and_same_ms_multiset_preserve_content(mode):
    raw=events(n=1200);cfg=EpisodeConfig(width=16,height=16,M=10,reservoir=2,policy=mode)
    snapshots=[feed(EpisodeMemory(cfg,policy=ToyPolicy()).reset('shared',policy=ToyPolicy()),raw,80,ch) for ch in (1,7,50)]
    rng=np.random.default_rng(11);permuted=raw.copy()
    for t in np.unique(raw[:,2]):
        ix=np.flatnonzero(raw[:,2]==t);permuted[ix]=permuted[rng.permutation(ix)]
    snapshots.append(feed(EpisodeMemory(cfg,policy=ToyPolicy()).reset('shared',policy=ToyPolicy()),permuted,80,7))
    for s in snapshots[1:]:
        np.testing.assert_array_equal(s.blocks,snapshots[0].blocks)
        np.testing.assert_array_equal(s.to_grid40ch(),snapshots[0].to_grid40ch())
        assert s.dictionary_sha256(content_only=True)==snapshots[0].dictionary_sha256(content_only=True)
        assert len(s.blocks)<=cfg.M


def test_pending_group_halfopen_and_oldest_coarse_slot_not_overwritten():
    m=EpisodeMemory(EpisodeConfig(width=8,height=8,policy='uniform')).reset('pending')
    m.ingest(np.array([[0,0,0,0]]),1,0,source_range=(0,1))
    m.ingest(np.array([[1,1,50,1]]),50,1,source_range=(1,2))
    s=m.readout(50)
    assert s.metadata['source_ingested']==2 and s.metadata['source_complete_prefix']==1
    assert int(s.coarse_counts.sum())==1  # time0 still in[0,50)
    assert not s.partial_counts.any()  # time50 incomplete and excluded
    m.ingest(np.array([[1,1,50,1]]),51,2,source_range=(2,3))
    s=m.readout(51)
    assert int(s.coarse_counts.sum())==2  # genuine multiplicity at time50
    assert s.metadata['source_complete_prefix']==3


def test_snapshot_pure_immutable_and_future_past_guard():
    raw=events(duration=15,n=100,width=8,height=8)
    m=EpisodeMemory(EpisodeConfig(width=8,height=8,policy='uniform')).reset('query')
    m.ingest(raw,15,0,source_range=(0,len(raw)))
    before=m.state_dict();s=m.readout(15);s.to_grid40ch();s.to_grid40ch(log_counts=False,channels_first=False)
    after=m.state_dict()
    for k in before['arrays']:np.testing.assert_array_equal(before['arrays'][k],after['arrays'][k])
    assert m.version==s.version
    with pytest.raises(ValueError):s.partial_counts[0,0,0,0]=1
    with pytest.raises(ValueError):m.readout(14)
    with pytest.raises(ValueError):m.readout(16)
    assert s.to_grid40ch().shape==(40,8,8)


def test_unretained_fine_is_missing_and_partial_is_full_fine():
    c=EpisodeConfig(width=16,height=8,M=1,reservoir=0,policy='density')
    m=EpisodeMemory(c).reset('coverage')
    e=np.array([[0,0,0,0]]*10+[[8,0,0,1]],np.int64)
    m.ingest(e,10,0,source_range=(0,len(e)))
    r=m.readout(10).rasterize()
    assert int(r['fine'][:,:,8:,:].sum())==0
    assert int(r['fine_exposure_ms'][:,:,8:].sum())==0
    assert int(r['coarse'][:,:,2:,:].sum())==1
    m.ingest(np.empty((0,4),np.int64),15,1,source_range=(len(e),len(e)))
    r=m.readout(15).rasterize()
    assert np.all(r['fine_exposure_ms'][:,:,8:].sum(0)==5)


def test_sequence_boundary_padding_release_and_isolation():
    c=EpisodeConfig(width=8,height=8,policy='uniform')
    a=EpisodeMemory(c).reset('A',boundary_ms=23);b=EpisodeMemory(c).reset('B',boundary_ms=23)
    e=np.array([[0,0,23,0]],np.int64)
    a.ingest(e,24,0,source_range=(0,1),seq_id='A')
    assert not b.readout(23).coarse.any()
    with pytest.raises(ValueError):a.ingest(e,24,1,source_range=(1,2),seq_id='B')
    v=a.version
    a.ingest(object(),-999,-5,seq_id='bad',valid=False)
    assert a.version==v
    a.release()
    with pytest.raises(RuntimeError):a.readout()
    a.reset('B',23);assert not a.readout(23).to_grid40ch().any()


def test_duplicate_late_future_invalid_source_rejections_are_atomic():
    c=EpisodeConfig(width=8,height=8,policy='uniform');m=EpisodeMemory(c).reset('atomic')
    e=np.array([[0,0,0,0]],np.int64);m.ingest(e,1,0,source_range=(0,1))
    before=pickle.dumps(m.state_dict())
    cases=[(e,1,0,(1,2)),(np.array([[0,0,1,0]]),2,1,(0,1)),
           (e,2,1,(1,2)),(np.array([[0,0,4,0]]),2,1,(1,2)),
           (np.array([[0,0,1,0]]),2,1,(1,3))]
    for payload,wm,ordinal,span in cases:
        with pytest.raises(ValueError):m.ingest(payload,wm,ordinal,source_range=span)
        assert pickle.dumps(m.state_dict())==before


def test_packet_cap_is_transport_only():
    c=EpisodeConfig(width=8,height=8,policy='uniform',max_packet_events=2)
    m=EpisodeMemory(c).reset('cap');before=pickle.dumps(m.state_dict())
    with pytest.raises(ValueError):m.ingest(np.array([[0,0,0,0]]*3),1,0,source_range=(0,3))
    assert pickle.dumps(m.state_dict())==before
    m.ingest(np.array([[0,0,0,0]]*2),0,0,source_range=(0,2))
    m.ingest(np.array([[0,0,0,0]]*2),1,1,source_range=(2,4))
    assert int(m.readout(1).coarse.sum())==4


def test_absolute_first_source_range_then_gap_rejected_atomically():
    m=EpisodeMemory(EpisodeConfig(width=8,height=8,policy='uniform')).reset('sourcegap')
    m.ingest(np.array([[0,0,0,0]]),1,0,source_range=(100,101))
    before=pickle.dumps(m.state_dict())
    with pytest.raises(ValueError,match='contiguous'):
        m.ingest(np.array([[0,0,1,0]]),2,1,source_range=(102,103))
    assert pickle.dumps(m.state_dict())==before
    m.ingest(np.array([[0,0,1,0]]),2,1,source_range=(101,102))
    assert int(m.readout(2).coarse.sum())==2


@pytest.mark.parametrize('coarse_only',[False,True])
def test_uint32_overflow_rejection_preserves_genuine_multiplicity(coarse_only):
    c=EpisodeConfig(width=8,height=8,policy='uniform');m=EpisodeMemory(c).reset('overflow')
    m.ingest(np.array([[0,0,0,0]]),0,0,source_range=(0,1))
    state=m.state_dict();u=state['arrays']['_U'];u.fill(0)
    u[0,0,0,0]=UINT32_MAX-(1 if coarse_only else 0)
    if coarse_only:u[0,0,1,0]=1
    state['scalars']['_u_events']=int(UINT32_MAX)
    state['scalars']['_pending_n']=int(UINT32_MAX)
    state['arrays']['_pending_coarse'][0,0,0]=UINT32_MAX
    state['arrays']['_u_source_n'][0]=UINT32_MAX
    state['scalars']['_last_source_end']=int(UINT32_MAX)
    state['scalars']['_source_count']=int(UINT32_MAX)
    m.load_state_dict(state);before=pickle.dumps(m.state_dict())
    e=np.array([[2 if coarse_only else 0,0,0,0]])
    with pytest.raises(OverflowError):m.ingest(e,1,1,source_range=(int(UINT32_MAX),int(UINT32_MAX)+1))
    assert pickle.dumps(m.state_dict())==before


def test_scorer_failure_transaction_restores_all_state():
    c=EpisodeConfig(width=8,height=8,policy='witness');m=EpisodeMemory(c,policy=BadPolicy()).reset('callback',policy=BadPolicy())
    m.ingest(np.array([[0,0,1,0]]),5,0,source_range=(0,1))
    before=pickle.dumps(m.state_dict())
    with pytest.raises(ValueError,match='injected'):m.ingest(np.array([[1,0,7,1]]),10,1,source_range=(1,2))
    assert pickle.dumps(m.state_dict())==before


def test_state_restore_includes_partial_group_policy_and_future_selection():
    c=EpisodeConfig(width=16,height=16,M=8,reservoir=2,policy='witness');raw=events(duration=60,n=700)
    a=EpisodeMemory(c,policy=ToyPolicy()).reset('restore',policy=ToyPolicy())
    hi=np.searchsorted(raw[:,2],25,side='right')
    a.ingest(raw[:hi],25,0,source_range=(0,int(hi)))
    b=EpisodeMemory(c).load_state_dict(pickle.loads(pickle.dumps(a.state_dict())))
    for memory in (a,b):memory.ingest(raw[hi:],60,1,source_range=(int(hi),len(raw)))
    arrays_equal(a,b)
    assert a.readout(60).dictionary_sha256()==b.readout(60).dictionary_sha256()
    assert a.nbytes()==b.nbytes()
    state=b.state_dict();state['arrays']['_U']=state['arrays']['_U'].astype(np.uint64)
    before=b.readout(60).content_sha256()
    with pytest.raises(ValueError):b.load_state_dict(state)
    assert b.readout(60).content_sha256()==before


def test_commit_preserves_dictionary_and_defers_mid_epoch_policy():
    c=EpisodeConfig(width=16,height=16,M=128,policy='witness');raw=events(duration=40,n=700)
    m=EpisodeMemory(c,policy=ToyPolicy()).reset('commit',policy=ToyPolicy())
    hi=np.searchsorted(raw[:,2],25);m.ingest(raw[:hi],25,0,source_range=(0,int(hi)))
    s=m.readout(25);old=s.dictionary_sha256();v=s.version
    before=m.nbytes();m.commit(ToyPolicy(scale=3,ref=25),25,expected_version=v)
    assert m.nbytes()==before+16  # old+future policy caches counted
    assert m.readout(25).dictionary_sha256()==old
    with pytest.raises(ValueError):m.commit(ToyPolicy(),25,expected_version=v)
    m.ingest(raw[hi:],40,1,source_range=(int(hi),len(raw)))
    s=m.readout(40)
    assert np.all(s.selection_pose_version[s.epoch_start_ms==20]==0)
    assert np.all(s.selection_pose_version[s.epoch_start_ms==30]==1)


def test_memory_bounded_through_many_epochs_empty_gap_and_bottom_padding():
    c=EpisodeConfig(width=8,height=8,M=4,reservoir=1,policy='uniform');m=EpisodeMemory(c).reset('bounded')
    n=m.nbytes()
    for k in range(200):
        e=np.array([[k%8,0,k*10,0]],np.int64)
        m.ingest(e,k*10+10,k,source_range=(k,k+1))
        assert m.nbytes()==n and len(m.readout().blocks)<=4
    m.ingest(np.empty((0,4),np.int64),1000000,200,source_range=(200,200))
    s=m.readout();assert not s.blocks.size and not s.coarse.any()
    assert np.all(s.rasterize()['fine_exposure_ms'].sum(0)==0)  # at epoch boundary, expired fine unavailable
    p=EpisodeMemory(EpisodeConfig(policy='uniform')).reset('padding')
    p.ingest(np.array([[239,179,1,1]],np.int64),10,0,source_range=(0,1))
    s=p.readout();assert s.blocks.shape==(1,10,8,8,2)
    assert not s.episode_spatial_valid[0,4:].any()
    assert s.blocks[0,:,4:].sum()==0


def test_signed_quotas_reserve_dedup_and_global_generic_comparator():
    # Enough genuine nonempty blocks in every owner/sign bucket. This checks
    # exact priority quota counts, not just that selection remains bounded.
    rows=[]
    for y in range(0,180,8):
        for x in range(0,240,8):
            rows.append([x,y,0 if (x//8+y//8)%2 else 9,0])
    raw=np.array(sorted(rows,key=lambda v:v[2]),np.int64)
    m=EpisodeMemory(EpisodeConfig(policy='witness'),policy=ToyPolicy()).reset('signed',policy=ToyPolicy())
    m.ingest(raw,10,0,source_range=(0,len(raw)));s=m.readout(10)
    assert len(s.blocks)==128
    assert int(np.sum(s.selection_kind==1))==32
    assert int(np.sum(s.selection_kind==2))==96
    assert len(set(zip(s.epoch_start_ms.tolist(),map(tuple,s.tile_xy.tolist()))))==128
    for owner,q in enumerate(m.config.quotas):
        assert int(np.sum((s.selection_kind==2)&(s.owner==owner)&(s.sign>0)))==(q+1)//2
        assert int(np.sum((s.selection_kind==2)&(s.owner==owner)&(s.sign<0)))==q//2
    assert int(s.source_events.sum())==128
    g=EpisodeMemory(EpisodeConfig(policy='generic_temporal'),policy=ToyPolicy()).reset('generic',policy=ToyPolicy())
    g.ingest(raw,10,0,source_range=(0,len(raw)))
    assert int(np.sum(g.readout(10).selection_kind==2))==96


def test_lane_names_do_not_change_sampling_but_states_are_isolated():
    raw=events(duration=80,n=1300)
    cfg=EpisodeConfig(width=16,height=16,M=6,reservoir=2,policy='witness')
    a=EpisodeMemory(cfg).reset('lane0:pass3',policy=ToyPolicy())
    b=EpisodeMemory(cfg).reset('recording/run17',policy=ToyPolicy())
    sa=feed(a,raw,80,1);sb=feed(b,raw,80,7)
    assert sa.dictionary_sha256()==sb.dictionary_sha256()
    assert sa.content_sha256()==sb.content_sha256()
    a.ingest(np.array([[0,0,80,0]],np.int64),81,1000,source_range=(len(raw),len(raw)+1))
    assert b.watermark_ms==80 and b.readout(80).content_sha256()==sb.content_sha256()
    with pytest.raises(ValueError):a.readout(81,seq_id='recording/run17')


def test_core_time_origin_covariance_with_time_covariant_scorer():
    # This is a core property under a time-covariant scorer. The separately
    # frozen real MotionPolicy has absolute-epoch tie hashes; this test does
    # NOT certify its full tie domain as time-origin invariant.
    raw=events(duration=80,n=1300)
    shifted=raw.copy();shifted[:,2]+=1000
    cfg=EpisodeConfig(width=16,height=16,M=6,reservoir=2,policy='witness')
    a=EpisodeMemory(cfg).reset('native',0,policy=ToyPolicy(ref=0))
    b=EpisodeMemory(cfg).reset('translated',1000,policy=ToyPolicy(ref=1000))
    sa=feed(a,raw,80,7)
    offset=ordinal=0
    for end in list(range(1007,1080,7))+[1080]:
        hi=np.searchsorted(shifted[:,2],end)
        b.ingest(shifted[offset:hi],int(end),ordinal,source_range=(int(offset),int(hi)))
        offset=hi;ordinal+=1
    sb=b.readout(1080)
    np.testing.assert_array_equal(sa.tile_xy,sb.tile_xy)
    np.testing.assert_array_equal(sa.epoch_start_ms,sb.epoch_start_ms-1000)
    np.testing.assert_array_equal(sa.blocks,sb.blocks)
    np.testing.assert_array_equal(sa.to_grid40ch(),sb.to_grid40ch())
