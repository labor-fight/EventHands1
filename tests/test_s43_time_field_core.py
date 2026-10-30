"""Count/reference and streaming contracts; no task scores or model fitting."""
import pickle

import numpy as np
import pytest

from tools.s43_revisit.field_core import TimeFieldConfig,TimeFieldMemory,MAX_COUNT


def raw_events(n=1000,duration=150,width=16,height=12,seed=3407):
    rng=np.random.default_rng(seed)
    return np.column_stack((rng.integers(0,width,n),rng.integers(0,height,n),
        np.sort(rng.integers(0,duration,n)),rng.integers(0,2,n))).astype(np.int64)


def feed(m,raw,start,stop,chunk,source_start=0):
    cursor=ordinal=0
    for end in list(range(start+chunk,stop,chunk))+[stop]:
        hi=int(np.searchsorted(raw[:,2],end))
        m.ingest(raw[cursor:hi],end,ordinal,source_range=(source_start+cursor,source_start+hi))
        cursor=hi;ordinal+=1
    return m.readout(stop)


@pytest.mark.parametrize('resolution',[1,2])
def test_independent_native_reference_and_coarse_density_mass(resolution):
    raw=raw_events();cfg=TimeFieldConfig(width=16,height=12,resolution=resolution)
    snap=feed(TimeFieldMemory(cfg).reset('reference'),raw,0,150,7)
    expected=np.zeros((50,12//resolution,16//resolution,2),np.uint64)
    selected=raw[(raw[:,2]>=100)&(raw[:,2]<150)]
    np.add.at(expected,(selected[:,2]-100,selected[:,1]//resolution,selected[:,0]//resolution,selected[:,3]),1)
    np.testing.assert_array_equal(snap.counts,expected)
    assert np.all(snap.observed)
    r=snap.rasterize();assert int(r['fine_counts'].sum())==len(selected)==int(r['coarse_counts'].sum())
    grid=snap.to_grid40ch()
    assert np.isclose(np.expm1(grid[:16].astype(float)).sum(),len(selected),rtol=1e-6)
    assert np.isclose(np.expm1(grid[24:].astype(float)).sum()/16,len(selected),rtol=1e-6)


def test_sealed_field2_feature_grid_bitwise_matches_frozen_baseline():
    from tools.s43_revisit.witness_probe import DenseFieldMemory
    raw=raw_events(n=700,duration=100,width=240,height=180)
    native=TimeFieldMemory().reset('newA',boundary_ms=13)
    old=DenseFieldMemory(2,13)
    raw[:,2]+=13;lo=0
    for ordinal,ms in enumerate(range(13,113)):
        hi=np.searchsorted(raw[:,2],ms+1);packet=raw[lo:hi]
        native.ingest(packet,ms+1,ordinal,source_range=(int(lo),int(hi)))
        old.ingest(packet,ms);lo=hi
        if ms+1 in (14,23,50,63,88,113):
            np.testing.assert_array_equal(native.readout(ms+1).to_grid40ch(),old.grid(ms+1)[0])


def test_chunks_and_same_ms_multiset_exact_content():
    raw=raw_events();cfg=TimeFieldConfig(width=16,height=12)
    snapshots=[feed(TimeFieldMemory(cfg).reset('same'),raw,0,150,ch) for ch in (1,7,50)]
    perm=raw.copy();rng=np.random.default_rng(99)
    for t in np.unique(raw[:,2]):
        ids=np.flatnonzero(raw[:,2]==t);perm[ids]=raw[rng.permutation(ids)]
    snapshots.append(feed(TimeFieldMemory(cfg).reset('same'),perm,0,150,7))
    for s in snapshots[1:]:
        assert s.content_sha256()==snapshots[0].content_sha256()
        np.testing.assert_array_equal(s.to_grid40ch(),snapshots[0].to_grid40ch())


def test_pending_same_ms_multichunk_and_oldest_row_halfopen():
    m=TimeFieldMemory(TimeFieldConfig(width=8,height=8)).reset('pending')
    m.ingest(np.array([[0,0,0,0]]*3),1,0,source_range=(0,3))
    m.ingest(np.array([[1,1,50,1]]*4),50,1,source_range=(3,7))
    s=m.readout(50);assert s.counts.sum()==3 and s.metadata['pending_events']==4
    assert s.metadata['retained_window_events']==3
    m.ingest(np.array([[1,1,50,1]]*5),51,2,source_range=(7,12))
    s=m.readout(51);assert s.counts.sum()==9 and s.metadata['pending_events']==0
    assert s.metadata['complete_prefix_events']==12
    assert s.source_events_per_ms[-1]==9
    np.testing.assert_array_equal(s.source_range_span[-1],[3,12])


def test_empty_complete_coverage_vs_unobserved_and_huge_gap():
    m=TimeFieldMemory(TimeFieldConfig(width=8,height=8)).reset('empty',73)
    assert not m.readout(73).observed.any()
    m.ingest(np.empty((0,4),np.int64),80,0,source_range=(100,100))
    s=m.readout(80);assert int(s.observed.sum())==7 and not s.counts.any()
    assert s.rasterize()['bin_exposure_ms'].sum()==7
    m.ingest(np.empty((0,4),np.int64),1000000,1,source_range=(100,100))
    s=m.readout();assert s.observed.all() and not s.counts.any()
    assert np.all(s.to_grid40ch()[16:24]==1)


def test_query_is_observation_pure_and_snapshot_immutable():
    raw=raw_events(duration=20,n=100,width=8,height=8)
    m=TimeFieldMemory(TimeFieldConfig(width=8,height=8)).reset('pure')
    m.ingest(raw,20,0,source_range=(0,len(raw)))
    before=pickle.dumps(m.state_dict());s=m.readout(20);s.to_grid40ch();s.rasterize()
    assert before==pickle.dumps(m.state_dict())
    with pytest.raises(ValueError):s.counts[0,0,0,0]=1
    with pytest.raises(ValueError):m.readout(19)
    with pytest.raises(ValueError):m.readout(21)


def test_source_gap_duplicate_late_future_and_invalid_packets_atomic():
    m=TimeFieldMemory(TimeFieldConfig(width=8,height=8)).reset('guards')
    m.ingest(np.array([[1,1,0,0]]),1,0,source_range=(30,31))
    before=pickle.dumps(m.state_dict())
    trials=[(np.array([[1,1,1,1]]),2,0,(31,32)),
      (np.array([[1,1,1,1]]),2,1,(32,33)),(np.array([[1,1,1,1]]),2,1,(30,31)),
      (np.array([[1,1,0,1]]),2,1,(31,32)),(np.array([[1,1,3,1]]),2,1,(31,32)),
      (np.array([[1,1,1,-1]]),2,1,(31,32)),(np.array([[1.,1.,1.,1.]]),2,1,(31,32)),
      (np.array([[8,1,1,1]]),2,1,(31,32))]
    for a,wm,ordinal,span in trials:
        with pytest.raises(ValueError):m.ingest(a,wm,ordinal,source_range=span)
        assert pickle.dumps(m.state_dict())==before


def test_native_uint32_aggregate_overflow_atomic():
    m=TimeFieldMemory(TimeFieldConfig(width=8,height=8)).reset('overflow')
    m.ingest(np.array([[0,0,0,0]]),0,0,source_range=(0,1))
    state=m.state_dict();state['arrays']['_pending'][0,0,0]=MAX_COUNT-1
    state['scalars'].update(_pending_n=int(MAX_COUNT)-1,_source_count=int(MAX_COUNT)-1,
        _last_source_end=int(MAX_COUNT)-1,_pending_hi=int(MAX_COUNT)-1)
    m.load_state_dict(state);before=pickle.dumps(m.state_dict())
    # Two different native pixels aggregate into the SAME2px cell.
    ev=np.array([[0,0,0,0],[1,1,0,0]],np.int64)
    with pytest.raises(OverflowError):m.ingest(ev,1,1,source_range=(int(MAX_COUNT)-1,int(MAX_COUNT)+1))
    assert pickle.dumps(m.state_dict())==before


def test_packet_domain_is_not_total_stream_rate_cap():
    m=TimeFieldMemory(TimeFieldConfig(width=8,height=8,max_packet_events=2)).reset('cap')
    before=pickle.dumps(m.state_dict())
    with pytest.raises(ValueError):m.ingest(np.array([[0,0,0,0]]*3),1,0,source_range=(0,3))
    assert before==pickle.dumps(m.state_dict())
    m.ingest(np.array([[0,0,0,0]]*2),0,0,source_range=(0,2))
    m.ingest(np.array([[0,0,0,0]]*2),1,1,source_range=(2,4))
    assert m.readout(1).counts.sum()==4


def test_lane_names_and_time_origin_do_not_change_features_or_isolation():
    raw=raw_events();shift=raw.copy();shift[:,2]+=1003
    c=TimeFieldConfig(width=16,height=12)
    a=TimeFieldMemory(c).reset('lane0:pass99')
    b=TimeFieldMemory(c).reset('recording/run17',1003)
    sa=feed(a,raw,0,150,1);sb=feed(b,shift,1003,1153,7)
    np.testing.assert_array_equal(sa.counts,sb.counts)
    np.testing.assert_array_equal(sa.observed,sb.observed)
    np.testing.assert_array_equal(sa.to_grid40ch(),sb.to_grid40ch())
    np.testing.assert_array_equal(sa.times_ms,sb.times_ms-1003)
    a.ingest(np.array([[0,0,150,0]]),151,1000,source_range=(len(raw),len(raw)+1))
    assert b.watermark_ms==1153
    with pytest.raises(ValueError):a.readout(seq_id='recording/run17')


def test_padding_reset_release_and_checkpoint_restore_pending_group():
    c=TimeFieldConfig(width=8,height=8);a=TimeFieldMemory(c).reset('state')
    a.ingest(np.array([[1,1,5,1]]),5,0,source_range=(100,101))
    before=pickle.dumps(a.state_dict());a.ingest(object(),-5,-6,seq_id='alien',valid=False)
    assert before==pickle.dumps(a.state_dict())
    b=TimeFieldMemory(c).load_state_dict(pickle.loads(before))
    for m in (a,b):m.ingest(np.array([[1,1,5,1]]),6,1,source_range=(101,102))
    assert a.readout(6).content_sha256()==b.readout(6).content_sha256()
    bad=b.state_dict();bad['arrays']['_pending']=bad['arrays']['_pending'].astype(np.uint64)
    before=pickle.dumps(b.state_dict())
    with pytest.raises(ValueError):b.load_state_dict(bad)
    assert before==pickle.dumps(b.state_dict())
    a.release()
    with pytest.raises(RuntimeError):a.readout()
    a.reset('new',9);assert not a.readout(9).observed.any()


def test_numeric_state_fixed_and_ring_not_replaced_by_ingest():
    m=TimeFieldMemory().reset('bounded');size=m.nbytes();address=m._counts.ctypes.data
    for i in range(120):
        m.ingest(np.array([[0,0,i,0]],np.int64),i+1,i,source_range=(i,i+1))
        assert m.nbytes()==size and m._counts.ctypes.data==address
    assert size==4408000
    assert m.stats()['whole_ring_copies_per_ingest']==0
    assert m.readout(120).counts.sum()==50


def test_invalid_empty_shape_and_clock_overflow_are_atomic():
    cfg=TimeFieldConfig(width=8.,height=np.int64(8),resolution=2.)
    m=TimeFieldMemory(cfg).reset('configuration')
    before=pickle.dumps(m.state_dict())
    with pytest.raises(ValueError):m.ingest(np.empty((0,3),np.int64),1,0)
    with pytest.raises(ValueError):m.reset('overflow',2**63)
    assert before==pickle.dumps(m.state_dict())
