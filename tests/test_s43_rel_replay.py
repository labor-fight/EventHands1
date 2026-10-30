"""Independent online prefixes, query purity and W50/hop separation."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn

from tools.s43_rel.model import S43RelationModel
from tools.s43_rel.core import RelConfig
from tools.s43_rel.data import last_lnes
from tools.s43_rel.training import RelationRuntime
from tools.s43_rel.replay import CausalRelationTracker, RecentWindow, duplicate_query_check


class Base(nn.Module):
    sparse_trunk = True
    active_head = False
    transl_head = None
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 3, 3, padding=1)
        self.rn = nn.Module()
        self.rn.fc = nn.Linear(3, 51)
        self.mano = nn.Identity()
    def forward(self, image, prev, betas=None, camera_K=None):
        return self.rn.fc(self.conv1(image.permute(0, 3, 1, 2)).mean((2, 3))) + prev


class Provider:
    def bind(self, pose, betas, camera, topology):
        ids = np.arange(min(50, topology.n_edges))
        def read(snapshot):
            mass = snapshot.Z.reshape(len(ids), 24).sum(0)
            return {"a": np.ones((4, 51)), "lam": np.tile(mass[None], (4, 1)),
                    "f": np.tile(mass[None], (51, 1))}
        return SimpleNamespace(edge_ids=ids, read=read)


def events():
    # Include repeated payloads and same-ms events; a real source ID still
    # distinguishes true multiplicity from duplicate ingestion.
    return np.array([[x, y, t, p] for t in range(50) for x, y, p in ((0, 0, 0), (2, 0, 1), (2, 0, 1))])


@pytest.mark.parametrize("arm", ("s43", "enhanced", "no_order", "full"))
def test_training_batch_vs_native_ms_prefix_same_inputs(arm):
    torch.manual_seed(3407)
    model = S43RelationModel(Base(), arm).eval()
    cfg = RelConfig(backend="numpy")
    runtime = RelationRuntime(model, cfg)
    runtime._thread_geometry.provider = Provider()
    tracker = CausalRelationTracker(model, cfg)
    tracker.provider = Provider()
    pose = np.array([0., 0., .6] + [0.] * 48, np.float32)
    beta, K, ev = np.zeros(10, np.float32), np.eye(3, dtype=np.float32), events()
    tracker.reset("test", 0, pose, beta, K)
    offset = 0
    for ms in range(50):
        packet = ev[ev[:, 2] == ms]
        tracker.ingest(packet, watermark_ms=ms + 1, packet_ordinal=ms, source_range=(offset, offset + len(packet)))
        offset += len(packet)
    online, _, _ = tracker.representation(50)
    item = {"reset": True, "released_seq_id": None, "seq_id": "test", "run_start_ms": 0,
            "start_ms": 0, "cutoff_ms": 50, "initial_state": pose, "betas": beta, "K_cam": K,
            "events": ev, "packet_ordinal": 0, "source_range": (0, len(ev)), "lnes": last_lnes(ev, 0, 50)}
    training = runtime.inputs([item])[0][0]
    for key in ("lnes", "sensor") + (("f", "a", "lam") if arm in ("full", "no_order") else ()):
        np.testing.assert_allclose(online[key], training[key], rtol=3e-6, atol=2e-6)
    assert online["delta_commit_ms"] == training["delta_commit_ms"] == 50


def test_rolling_last50_is_independent_of_25or100_ingest():
    stream = np.array([[1, 2, t, 0] for t in range(150)])
    window = RecentWindow(0)
    for a, b in ((0, 25), (25, 50), (50, 75), (75, 100), (100, 150)):
        window.ingest(stream[a:b], b)
        np.testing.assert_array_equal(window.read(b)[:, 2], np.arange(max(0, b - 50), b))
    assert window.nbytes <= 100 * 4 * 8
    with pytest.raises(ValueError, match="duplicate/future"):
        window.ingest(stream[100:150], 175)


def test_query_pure_commit_explicit_and_stale_rejected():
    torch.set_num_threads(1)
    model = S43RelationModel(Base(), "s43").eval()
    tracker = CausalRelationTracker(model)
    pose = np.zeros(51, np.float32)
    tracker.reset("test", 0, pose, np.zeros(10, np.float32), np.eye(3, dtype=np.float32))
    ev = events()
    tracker.ingest(ev, watermark_ms=50, packet_ordinal=0, source_range=(0, len(ev)))
    result, audit = duplicate_query_check(tracker, 50)
    assert audit["prediction_max_abs"] == 0.
    assert tracker.commits == 0 and np.array_equal(tracker.pose, pose)
    tracker.commit(result)
    assert tracker.commits == 1 and np.array_equal(tracker.pose, result["state"])
    with pytest.raises(ValueError, match="stale"):
        tracker.commit(result)
    pending = tracker.query(50)
    tracker.ingest(np.array([[2, 1, 50, 1]]), watermark_ms=51, packet_ordinal=1,
                   source_range=(len(ev), len(ev) + 1))
    with pytest.raises(ValueError, match="stale"):
        tracker.commit(pending)


@pytest.mark.parametrize("arm", ("s43", "enhanced", "no_order", "full"))
def test_cap_reject_is_atomic_before_observation_update(arm):
    model = S43RelationModel(Base(), arm).eval()
    tracker = CausalRelationTracker(model, RelConfig(backend="numpy"), max_window_events=4)
    tracker.provider = Provider()
    tracker.reset("cap", 0, np.array([0.,0.,.6]+[0.]*48,np.float32),
                  np.zeros(10,np.float32), np.eye(3,dtype=np.float32))
    first = np.array([[0,0,0,0],[2,0,0,1],[2,0,1,1]],np.int64)
    tracker.ingest(first,watermark_ms=2,packet_ordinal=0,source_range=(0,3))
    pending = tracker.query(2)
    before = (tracker.evidence_digest(),tracker.window.read(2).copy(),tracker.observation_version,
              tracker.watermark_ms,tracker.last_ordinal,tracker.last_source_end,tracker.window.peak_event_count)
    with pytest.raises(OverflowError,match="before observation update"):
        tracker.ingest(np.array([[0,0,2,0],[2,0,2,1]]),watermark_ms=3,
                       packet_ordinal=1,source_range=(3,5))
    assert tracker.evidence_digest() == before[0]
    np.testing.assert_array_equal(tracker.window.read(2),before[1])
    assert (tracker.observation_version,tracker.watermark_ms,tracker.last_ordinal,
            tracker.last_source_end,tracker.window.peak_event_count) == before[2:]
    assert tracker.pending is pending
    # Same ordinal/source range can be retried with an admissible packet:
    # the failed request consumed neither observations nor source identity.
    tracker.ingest(np.array([[0,0,2,0]]),watermark_ms=3,packet_ordinal=1,source_range=(3,4))
    assert tracker.window.event_count == 4


def test_large_packet_vs_ms_cap_and_exact_support():
    times = np.repeat(np.arange(50),5000)
    ev = np.column_stack((np.zeros(len(times),np.int64),np.ones(len(times),np.int64),times,
                          np.zeros(len(times),np.int64)))
    whole, milliseconds = RecentWindow(0),RecentWindow(0)
    whole.ingest(ev,50)
    for ms in range(50):
        milliseconds.ingest(ev[ms*5000:(ms+1)*5000],ms+1)
    np.testing.assert_array_equal(whole.read(50),milliseconds.read(50))
    assert whole.event_count == milliseconds.event_count == 250000
    assert whole.peak_event_count == milliseconds.peak_event_count == 250000
    assert whole.nbytes == milliseconds.nbytes == 250000*32 <= whole.max_window_events*32
    whole.ingest(np.empty((0,4),np.int64),100)
    assert whole.event_count == whole.nbytes == 0


def test_cap_checks_intermediate_prefix_even_if_packet_final_window_small():
    stream = np.array([[0,0,0,0]]*5+[[0,0,100,0]],np.int64)
    window = RecentWindow(0,max_window_events=4)
    with pytest.raises(OverflowError):
        window.ingest(stream,101)
    assert window.watermark_ms == window.event_count == window.nbytes == 0
    # Splitting at a native group reaches the same reject, with no partial
    # commit of the accepted-looking final W50 slice.
    with pytest.raises(OverflowError):
        window.ingest(stream[:5],1)
    assert window.watermark_ms == window.event_count == 0


@pytest.mark.parametrize("arm", ("s43", "enhanced", "no_order", "full"))
def test_within_cap_packet_strategy_preserves_nonzero_reader_predictions(arm):
    torch.manual_seed(3407)
    model = S43RelationModel(Base(),arm).eval()
    with torch.no_grad():
        model.sensor_adapter.weight.fill_(.0001)
        model.typed_adapter.output.weight.fill_(.0001)
    ev = events()
    results = []
    for chunk in (1,7,50):
        tracker = CausalRelationTracker(model,RelConfig(backend="numpy"),max_window_events=200)
        tracker.provider=Provider()
        tracker.reset("same",0,np.array([0.,0.,.6]+[0.]*48,np.float32),
                      np.zeros(10,np.float32),np.eye(3,dtype=np.float32))
        source,ordinal=0,0
        for begin in range(0,50,chunk):
            end=min(50,begin+chunk);packet=ev[(ev[:,2]>=begin)&(ev[:,2]<end)]
            tracker.ingest(packet,watermark_ms=end,packet_ordinal=ordinal,
                           source_range=(source,source+len(packet)))
            source+=len(packet);ordinal+=1
        rep,_,_=tracker.representation(50)
        results.append((tracker.query(50)["state"],rep,tracker.window.read(50)))
    for actual,rep,raw in results[1:]:
        np.testing.assert_array_equal(raw,results[0][2])
        for key in ("lnes","sensor")+(("f","a","lam") if arm in ("full","no_order") else ()):
            np.testing.assert_allclose(rep[key],results[0][1][key],rtol=3e-6,atol=2e-6)
        np.testing.assert_allclose(actual,results[0][0],rtol=3e-6,atol=2e-6)


@pytest.mark.parametrize("hop,expected_ends", ((25, [124,149,174,199,224,249]),
                                               (50, [149,199,249]), (100, [199])))
def test_complete_replay_scores_only_after_all_valid_sources(monkeypatch, tmp_path, hop, expected_ends):
    from tools.s43_rel import replay as module
    ledger = {"ingested": 0, "target_reads": 0}
    class Reader:
        def __init__(self, *args):
            self.root, self.base = tmp_path, tmp_path / "fake"
            self.runs = np.array([[0,260],[300,340]])
            self.offsets = np.arange(341)
            self.events = np.tile(np.array([[1,2,0]], np.uint8), (340,1))
            self.betas, self.K_cam = np.zeros(10,np.float32), np.eye(3,dtype=np.float32)
        def target(self, end):
            assert ledger["ingested"] == 300
            ledger["target_reads"] += 1
            return np.full(51,end,np.float32)
    class Tracker(module.CausalRelationTracker):
        def ingest(self, events, **kwargs):
            super().ingest(events, **kwargs)
            ledger["ingested"] += len(events)
    marker = tmp_path / "marker"
    marker.write_text("immutable fixture")
    cfg = {"REL":{"BASE_CHECKPOINT":str(marker),"BASE_CONFIG":str(marker)}}
    base = Base().eval()
    model = S43RelationModel(base,"s43").eval()
    monkeypatch.setattr(module,"SequenceReader",Reader)
    monkeypatch.setattr(module,"CausalRelationTracker",Tracker)
    monkeypatch.setattr(module,"load_relation_model",lambda *args,**kwargs:(model,cfg))
    monkeypatch.setattr(module,"load_legacy",lambda *args,**kwargs:(base,cfg))
    monkeypatch.setattr(module,"source_snapshot",lambda *args:{})
    monkeypatch.setattr(module,"_evaluation_functions",lambda:{})
    monkeypatch.setattr(module,"_metrics",lambda *args:({"dummy_metric":1.},{}))
    out = tmp_path / "replay"
    report = module.replay(marker,None,"fake",out,hop=hop,device="cpu",threads=1,root=tmp_path)
    assert report["all_valid_events_consumed_once"]
    assert report["source_count"] == report["native_ms_packets"] == 300
    assert report["outputs"] == len(expected_ends) == ledger["target_reads"]
    assert report["runs"][0]["warmup_windows"] == 2
    assert report["runs"][1]["outputs"] == 0
    arrays = np.load(out.with_suffix(".npz"))
    np.testing.assert_array_equal(arrays["end_ms"],expected_ends)
    if hop == 25:
        ledger["ingested"] = ledger["target_reads"] = 0
        fixed_out = tmp_path / "fixed25"
        fixed = module.replay(marker,None,"fake",fixed_out,hop=25,fixed_commit_ms=50,
                              device="cpu",threads=1,root=tmp_path)
        ledger["ingested"] = ledger["target_reads"] = 0
        normal_out = tmp_path / "normal50"
        normal = module.replay(marker,None,"fake",normal_out,hop=50,
                               device="cpu",threads=1,root=tmp_path)
        fixed_arrays, normal_arrays = np.load(fixed_out.with_suffix(".npz")), np.load(normal_out.with_suffix(".npz"))
        np.testing.assert_array_equal(fixed_arrays["pred"][1::2],normal_arrays["pred"])
