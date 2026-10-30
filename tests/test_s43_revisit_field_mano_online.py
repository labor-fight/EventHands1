"""Target-free streaming engineering, zero-initialized weights only."""
import inspect,json
import numpy as np
import pytest
import torch
from tools.s43_revisit import field_mano_online as p

@pytest.fixture
def engine():
    torch.set_num_threads(2);torch.manual_seed(3407)
    model=p.IdentityFeatureReader('static','mano51')
    stats=dict(global_scale=np.ones(40,np.float32),local_scale=np.ones(5,np.float32),context_mean=np.zeros(70,np.float32),context_std=np.ones(70,np.float32))
    return p.FieldManoOnline(model,stats,np.asarray([[600.,0,320],[0,600,240],[0,0,1]]))

def feed(e,lo,hi,offset=0):
    for t in range(lo,hi):
        e.ingest(np.asarray([[100,90,t,t%2]],np.int64),t+1,t,source_range=(offset+t-lo,offset+t-lo+1))

def test_pure_query_fullsurface_and_explicit_stale_commit(engine):
    e=engine;feed(e,0,50);state=e.memory.readout(50).content_sha256();pose=e.pose.copy();n=len(e.history)
    a=e.query();b=e.query()
    assert e.pose_version==0 and len(e.history)==n and e.memory.readout(50).content_sha256()==state
    np.testing.assert_array_equal(e.pose,pose);np.testing.assert_array_equal(a.state51,p.PUBLIC_INITIAL)
    np.testing.assert_array_equal(a.vertices_m,b.vertices_m)
    assert a.vertices_m.shape==(778,3) and a.joints_m.shape==(21,3)
    e.commit(a)
    with pytest.raises(ValueError,match='stale'):e.commit(b)
    assert e.pose_version==1

def test_padding_crosslane_reset_and_release(engine):
    e=engine;initial=e.memory.state_dict();e.ingest(object(),object(),object(),source_range=object(),seq_id='wrong',valid=False)
    assert e.memory.version==0
    with pytest.raises(ValueError,match='mixed'):e.ingest(np.empty((0,4),np.int64),1,0,source_range=(0,0),seq_id='wrong')
    for k,v in initial['arrays'].items():np.testing.assert_array_equal(e.memory.state_dict()['arrays'][k],v)
    feed(e,0,50);old=e.query();e.reset('new')
    with pytest.raises(ValueError):e.commit(old)
    assert e.memory.version==0 and e.pose_version==0 and len(e.history)==1
    e.release()
    with pytest.raises(RuntimeError):e.query()

def test_history_bounded_under_native_ms_commits(engine):
    e=engine
    for t in range(1,101):
        e.ingest(np.empty((0,4),np.int64),t,t-1,source_range=(0,0))
        value=p.SurfaceQuery(e.seq_id,t,e.pose_version,e.memory.version,p.readonly(e.pose),np.zeros((778,3)),np.zeros((21,3)),np.zeros(51),{},e._owner)
        e.commit(value)
    assert len(e.history)<=52 and e.history[0][0]<=50 and e.history[-1][0]==100
    assert e.memory.nbytes()==4408000

def test_source_us_quantization_exact_once_cursor(tmp_path):
    raw=np.asarray([[1,2,1,0],[3,4,999,1],[5,6,1000,0]],np.int64);file=tmp_path/'events.npy';np.save(file,raw)
    stream=p.MMapEventStream(file,'us',start_ms=0,stop_ms=3);assert np.shares_memory(stream.times,stream.raw);rows=[]
    for t in range(3):
        packet,span=stream.peek_ms(t);assert stream.cursor==span[0];rows.append(packet);stream.commit_ms(t,span[1])
    got=np.concatenate(rows);np.testing.assert_array_equal(got[:,2],[0,0,1]);assert stream.cursor==3;stream.close()

def test_bad_source_cap_reject_without_cursor_mutation(tmp_path):
    file=tmp_path/'events.npy';np.save(file,np.asarray([[1,1,1,0],[1,1,0,0]],np.int64))
    with pytest.raises(ValueError,match='order'):p.MMapEventStream(file)
    np.save(file,np.asarray([[1,1,0,0],[1,1,0,1]],np.int64));stream=p.MMapEventStream(file,max_packet_events=1)
    with pytest.raises(ValueError,match='cap'):stream.peek_ms(0)
    assert stream.cursor==0 and stream.next_ms==0;stream.close()

def test_K_scaled_once_beta_only_explicit(tmp_path):
    file=tmp_path/'calibration.json';source=np.asarray([[600.,0,320],[0,600,240],[0,0,1]])
    file.write_text(json.dumps(dict(K_source=source.tolist(),betas=[1.]*10)))
    a,b,beta,_=p.calibration(file);assert b[0,0]==225 and np.all(beta==0)
    _,_,given,_=p.calibration(file,'provided');assert np.all(given==1)
    file.write_text(json.dumps(dict(K_event=b.tolist())));aa,bb,_,_=p.calibration(file)
    np.testing.assert_array_equal(aa,source);np.testing.assert_array_equal(bb,b)

def test_no_target_function_entry_and_sparse_arm_rejected():
    for fn in (p.FieldManoOnline,p.FieldManoOnline.query,p.run):
        assert not any('target' in name or 'ground_truth' in name for name in inspect.signature(fn).parameters)
    assert 'quota_anatomical' not in p.COMPLETE_ARMS
