"""DT2 package D contracts: the deployment form of the tracker (model/deploy_fast.py) is bit-for-bit the existing one.

What is pinned (every GPU test needs CUDA; the data gates also need the zgz evaluation data and the dt_dz_l3_s3407 run):

  GpuLNES      the GPU event image equals the host splat (`semkine.events.splat_event_image`) bit for bit: random packets of
               every shape that matters (empty, one event, all in one millisecond, many events on one pixel, ties, the
               polarity clip, events outside the frame dropped), the channel sets (last), (last, first), (last, count),
               (last, count, first), windows 50 / 30, the capacity edge; no host synchronisation, graph capturable
  HostGains    the host evaluation of a gain schedule is `GainSchedule`'s
  DeployTracker  against the loop `evalx.run_sequence` runs (host LNES, the unpatched eager model, `AdaptiveFilter` itself),
               closed loop, bit for bit, for graph scope model / filter / all, eager, host LNES, no filter, the capacity
               fallback, velocity filters (state kept in place across graph replays), empty packets, new sequence / segment
  gates        (b) evalx closed loop of the bare tracker with `enable_fast_render`: RA == 10.04394757480695 and both
               sequences' predictions equal the recorded npz bit for bit; (c) the GPU LNES equals `build_lnes` on all
               2590 evaluation packets; (d) the DeployTracker (graph + GPU LNES + GPU filter) on both sequences against the
               recorded `filter_eval` json of the same spec; (e) graph == eager on >= 40 real packets

    CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 python -m pytest tests/test_dt_deploy.py -q -p no:cacheprovider
"""
from __future__ import annotations

import copy
import faulthandler
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
MAIN = Path("/data1/lyq/code/mesh/EventHands1")
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "tracking")]
import deploy_fast as DF                                              # noqa: E402
import render_fast as RF                                              # noqa: E402
from config import load_config                                        # noqa: E402
from model import MNISTModel                                          # noqa: E402
from semkine import eval_track as ET                                  # noqa: E402
from semkine import events as EV                                      # noqa: E402
from semkine.anchored import AdaptiveFilter, GainSchedule             # noqa: E402

CONFIG = REPO / "configs" / "rt" / "rt_cnntrack.yaml"
RUN = MAIN / "outputs" / "semkine" / "dt_dz_l3_s3407"
SWEEP = MAIN / "outputs" / "dt2" / "filter_sweep" / "dt_dz_l3_s3407" / "afad1f21_rn0.3-0.5-0.8_f0.8_t0.5.json"
H, W = 180, 240
HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    # never hang on the shared GPU (a C-thread timer: it fires even if the main thread is stuck in a library call)
    faulthandler.dump_traceback_later(int(os.environ.get("DT_TEST_WATCHDOG_S", "1700")), exit=True)
needs_gpu = pytest.mark.skipif(not HAS_CUDA, reason="needs a GPU")
needs_data = pytest.mark.skipif(not (RUN / "config_resolved.yaml").exists() or not SWEEP.exists(),
                                reason="dt_dz_l3_s3407 run / zgz data / filter sweep not found")
BASE_K = torch.tensor([[603.4507, 0.0, 325.09183], [0.0, 602.95654, 242.09796], [0.0, 0.0, 1.0]])
CH_SETS = [("last",), ("last", "first"), ("last", "count"), ("last", "count", "first")]


def bits_equal(a, b):
    a, b = torch.as_tensor(a).detach().cpu().contiguous(), torch.as_tensor(b).detach().cpu().contiguous()
    return a.shape == b.shape and a.dtype == b.dtype == torch.float32 and torch.equal(a.view(torch.int32), b.view(torch.int32))


# ------------------------------------------------------------------------------------------------ raw packets
def random_packet(n, seed, window=50, kind="uniform"):
    """`(ev (n, 3) uint8, bounds (window,) int32)`: time ordered events of one packet, of the given shape."""
    g = np.random.default_rng(seed)
    if kind == "ms0":
        ms = np.zeros(n, np.int64)
    elif kind == "last_ms":
        ms = np.full(n, window - 1, np.int64)
    elif kind == "ties":
        ms = np.sort(g.choice([3, 3, 3, 17, 17, window - 1], n))
    else:
        ms = np.sort(g.integers(0, window, n))
    if kind == "hot":                                    # a handful of pixels, many events each
        cells = g.integers(0, 6, (n, 2))
        x, y = 100 + cells[:, 0], 80 + cells[:, 1]
    else:
        x, y = g.integers(0, W, n), g.integers(0, H, n)
    p = g.integers(0, 2, n)
    ev = np.stack([x, y, p], 1).astype(np.uint8)
    counts = np.bincount(ms, minlength=window)
    return ev, np.cumsum(counts).astype(np.int32)


PACKET_CASES = [(0, "uniform"), (1, "uniform"), (7, "ms0"), (500, "last_ms"), (3000, "uniform"), (3000, "hot"),
                (2000, "ties"), (30000, "uniform"), (30000, "hot"), (262144, "uniform")]


def test_host_lnes_from_is_build_lnes_cpu():
    """`host_lnes_from` (the packet in the GPU form) is the evaluator's `build_lnes`: same call, same bits."""
    for window in (50, 30):
        for n, kind in PACKET_CASES[:9]:
            ev, bounds = random_packet(n, n + 3, window, kind)
            offsets = np.concatenate([[0], bounds]).astype(np.int64)                 # offsets[start .. end + 1], start = 0
            for chs in CH_SETS:
                a = ET.build_lnes(ev, offsets, window - 1, window, chs)
                b = DF.host_lnes_from(ev, bounds, window, H, W, chs)
                assert a.shape == b.shape and np.array_equal(a.view(np.int32), b.view(np.int32)), (window, n, kind, chs)
    ev, bounds = random_packet(2000, 1)
    off = np.concatenate([np.zeros(10, np.int64), 5 + np.concatenate([[0], bounds])])     # a window in the middle of a stream
    assert np.array_equal(DF.packet_bounds(off, 10 + 49, 50), bounds)
    ms = np.sort(np.random.default_rng(1).integers(0, 50, 1000))                        # a live packet's per-event ms
    assert np.array_equal(DF.bounds_from_ms(ms, 50), np.cumsum(np.bincount(ms, minlength=50)).astype(np.int32))
    assert np.array_equal(DF.bounds_from_ms(ms[:0], 50), np.zeros(50, np.int32))


# ------------------------------------------------------------------------------------------------ GpuLNES
@needs_gpu
@pytest.mark.parametrize("chs", CH_SETS, ids=lambda c: "+".join(c))
@pytest.mark.parametrize("window", [50, 30])
def test_gpu_lnes_bitwise_random_packets_gpu(chs, window):
    g = DF.GpuLNES("cuda", window, H, W, chs, capacity=262144)
    for n, kind in PACKET_CASES:
        ev, bounds = random_packet(n, n + 5, window, kind)
        host = DF.host_lnes_from(ev, bounds, window, H, W, chs)
        got = g.lnes(ev, bounds)
        assert got.shape == (1, H, W, 2 * len(chs)) and got.dtype == torch.float32
        assert bits_equal(got[0], torch.from_numpy(host)), (n, kind)
    # the buffers are reused: a small packet after a big one must not see the big one's events
    ev, bounds = random_packet(40, 99, window)
    assert bits_equal(g.lnes(ev, bounds)[0], torch.from_numpy(DF.host_lnes_from(ev, bounds, window, H, W, chs)))


@needs_gpu
def test_gpu_lnes_polarity_clip_and_out_of_frame_gpu():
    g = DF.GpuLNES("cuda", 50, H, W, ("last", "count", "first"), capacity=4096)
    ev, bounds = random_packet(1500, 4)
    ev[::7, 2] = 3                                                     # the host clips the stored polarity to 0 / 1
    host = DF.host_lnes_from(ev, bounds, 50, H, W, ("last", "count", "first"))
    assert bits_equal(g.lnes(ev, bounds)[0], torch.from_numpy(host))
    # events outside the frame (the host splat would raise): dropped, i.e. the image of the in-frame events alone
    ev, bounds = random_packet(1500, 8)
    bad = np.zeros(len(ev), bool)
    bad[::11] = True
    ev2 = ev.copy()
    ev2[bad, 0] = 240 + (np.arange(int(bad.sum())) % 10)
    ev2[bad, 1] = np.where(np.arange(int(bad.sum())) % 2 == 0, 5, 180 + np.arange(int(bad.sum())) % 9)
    ms = np.repeat(np.arange(50), np.diff(bounds, prepend=0))
    keep = ~(ev2[:, 0] >= W) & ~(ev2[:, 1] >= H)
    ev3, ms3 = ev2[keep], ms[keep]
    bounds3 = np.cumsum(np.bincount(ms3, minlength=50)).astype(np.int32)
    host = DF.host_lnes_from(ev3, bounds3, 50, H, W, ("last", "count", "first"))
    assert bits_equal(g.lnes(ev2, bounds)[0], torch.from_numpy(host))


@needs_gpu
def test_gpu_lnes_capacity_and_validation_gpu():
    g = DF.GpuLNES("cuda", 50, H, W, ("last",), capacity=1000)
    ev, bounds = random_packet(1000, 1)
    assert g.fits(1000) and not g.fits(1001)
    assert bits_equal(g.lnes(ev, bounds)[0], torch.from_numpy(DF.host_lnes_from(ev, bounds, 50, H, W)))      # exactly full
    ev, bounds = random_packet(1001, 2)
    with pytest.raises(ValueError):
        g.stage(ev, bounds)
    with pytest.raises(ValueError):
        g.stage(ev[:10], bounds)                                       # bounds[-1] != len(ev)
    with pytest.raises(ValueError):
        g.stage(ev, bounds[:10])
    with pytest.raises(ValueError):
        DF.GpuLNES("cuda", 50, H, W, ("first",))                       # `last` first, as DATA.EVENT_CHANNELS


@needs_gpu
def test_gpu_lnes_is_sync_free_and_graph_capturable_gpu():
    g = DF.GpuLNES("cuda", 50, H, W, ("last", "count", "first"), capacity=65536)
    ev, bounds = random_packet(20000, 3)
    g.stage(ev, bounds)
    ref = g.build().clone()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        g.build()
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    graph, out = RF.capture_checked(g.build)
    ev2, bounds2 = random_packet(9000, 4)                              # new content, same graph
    g.stage(ev2, bounds2)
    graph.replay()
    assert bits_equal(out[0], torch.from_numpy(DF.host_lnes_from(ev2, bounds2, 50, H, W, ("last", "count", "first"))))
    assert not bits_equal(out, ref)


# ------------------------------------------------------------------------------------------------ gains
def test_host_gains_equal_gain_schedule_cpu():
    spec = {"a_root": [[300, 0.3], [3000, 0.5], [30000, 0.8]], "a_rest": 0.8, "a_trans": [[100, 0.2], [20000, 0.9]]}
    hg = DF.HostGains(spec)
    sch = [GainSchedule(hg.canon[k]) for k in ("a_root", "a_rest", "a_trans")]
    for n in (1, 5, 99, 100, 300, 301, 1000, 3000, 12345, 20000, 30000, 31000, 10 ** 6):
        want = []
        for s in sch:
            v = s(torch.tensor([float(n)], dtype=torch.float64))
            want.append(v if isinstance(v, float) else float(v.reshape(-1)[0]))
        assert hg(n) == pytest.approx(tuple(want), rel=0, abs=1e-12), n
    with pytest.raises(ValueError):
        DF.HostGains({"a_root": 0.5, "a_rest": 0.5, "beta_root": 0.2})


# ------------------------------------------------------------------------------------------------ the tracker, synthetic
SPEC = DF.RECOMMENDED_SPEC
SPEC_VEL = {"a_root": [[300, 0.3], [3000, 0.5], [30000, 0.8]], "a_rest": 0.8, "a_trans": 0.5, "beta_root": 0.3,
            "beta_trans": 0.2, "decay": 0.5}


def _model(dev, channels=("last",), seed=0):
    cfg = load_config(CONFIG)
    if channels != ("last",):
        cfg["DATA"]["EVENT_CHANNELS"] = list(channels)
    torch.manual_seed(seed)
    m = MNISTModel(cfg).eval()
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for p in m.parameters():                                       # no exactly-zero tensors: the render matters
            z = p == 0
            if z.any():
                p[z] = 0.02 * torch.randn(int(z.sum()), generator=g)
        m.rn.fc.weight.mul_(0.02)                                      # a tame loop
        m.rn.fc.bias.mul_(0.02)
    return m.to(dev)


def _seq_state(seed):
    g = np.random.default_rng(seed)
    prev0 = np.zeros(51, np.float32)
    prev0[:3] = [0.02, 0.01, 0.5]
    prev0[3:6] = g.normal(size=3) * 0.3
    prev0[6:] = np.clip(g.normal(size=45) * 0.3, -1, 1)
    betas = (0.5 * g.normal(size=10)).astype(np.float32)
    K = BASE_K.clone().numpy()
    K[0, 0] *= 1.02
    K[1, 2] += 4
    return prev0, betas, K


def _packets(T, seed, window=50, big=None):
    """T packets of mixed size; packet 0 empty, packet 3 all in ms 0 (zero image though not empty), the rest random."""
    out = []
    for t in range(T):
        if t == 0:
            out.append(random_packet(0, seed, window))
        elif t == 3:
            out.append(random_packet(400, seed + t, window, "ms0"))
        else:
            n = int([300, 2500, 9000, 600, 30000][t % 5]) if big is None or t != 7 else big
            out.append(random_packet(n, seed + t, window, "hot" if t % 4 == 2 else "uniform"))
    return out


def _reference_loop(model, spec, prev0, betas, K, packets, channels=("last",), window=50):
    """`evalx.run_sequence`'s step, as is: host LNES, the UNPATCHED eager model, `AdaptiveFilter` (or the bare model)."""
    dev = next(model.parameters()).device
    b = torch.as_tensor(betas).to(dev).view(1, -1)
    k = torch.as_tensor(K).to(dev).view(1, 3, 3)
    net = AdaptiveFilter.from_spec(model, spec).to(dev).eval() if spec is not None else model
    net.set_hand_context(b, k)
    prev_t = torch.from_numpy(prev0).view(1, -1).to(dev)
    if spec is not None:
        net.reset_state(prev_t)
    out = []
    with torch.no_grad():
        for ev, bounds in packets:
            x = torch.from_numpy(DF.host_lnes_from(ev, bounds, window, H, W, channels)).unsqueeze(0).to(dev)
            n = int(bounds[-1])
            pred = net(x, prev_t, n_events=n) if spec is not None else net(x, prev_t)
            prev_t = pred
            out.append(pred.cpu().numpy()[0])
    return np.stack(out)


def _deploy_loop(tr, prev0, betas, K, packets, reset_at=()):
    tr.begin_sequence(betas, K)
    tr.begin_segment(prev0)
    out = []
    for i, (ev, bounds) in enumerate(packets):
        if i in reset_at:
            tr.begin_segment(prev0)
        out.append(tr.step_events(ev, bounds))
    return np.stack(out)


def _same(a, b, what):
    assert a.shape == b.shape and np.array_equal(a.view(np.int32), b.view(np.int32)), (
        f"{what}: max abs diff {np.abs(a - b).max()}, differing steps {np.where((a.view(np.int32) != b.view(np.int32)).any(1))[0][:10]}")


MODES = {
    "eager": dict(graph=False),
    "graph-model": dict(graph=True, scope="model"),
    "graph-filter": dict(graph=True, scope="filter"),
    "graph-all": dict(graph=True, scope="all"),
    "eager-host-lnes": dict(graph=False, lnes="host"),
    "graph-filter-host-lnes": dict(graph=True, scope="filter", lnes="host"),
}


@needs_gpu
@pytest.mark.parametrize("mode", list(MODES))
def test_deploy_tracker_equals_the_eval_loop_gpu(mode):
    dev = "cuda"
    model = _model(dev)
    ref_model = copy.deepcopy(model)                                   # never patched
    prev0, betas, K = _seq_state(11)
    packets = _packets(12, 100)
    ref = _reference_loop(ref_model, SPEC, prev0, betas, K, packets)
    tr = DF.DeployTracker(model, SPEC, **MODES[mode])
    got = _deploy_loop(tr, prev0, betas, K, packets)
    _same(got, ref, f"DeployTracker {mode}")
    assert tr.n_fallback == 0
    assert not np.array_equal(ref[1], prev0)                           # the loop moves; packet 0 (empty) holds the state
    assert np.array_equal(ref[0].view(np.int32), prev0.view(np.int32))


@needs_gpu
@pytest.mark.parametrize("chs", CH_SETS[1:], ids=lambda c: "+".join(c))
def test_deploy_tracker_channel_sets_gpu(chs):
    model = _model("cuda", chs)
    ref_model = copy.deepcopy(model)
    prev0, betas, K = _seq_state(12)
    packets = _packets(8, 200)
    ref = _reference_loop(ref_model, SPEC, prev0, betas, K, packets, chs)
    for kw in (dict(graph=False), dict(graph=True, scope="all")):
        m = copy.deepcopy(ref_model)
        got = _deploy_loop(DF.DeployTracker(m, SPEC, channels=chs, **kw), prev0, betas, K, packets)
        _same(got, ref, f"{chs} {kw}")


@needs_gpu
def test_deploy_tracker_bare_and_capacity_fallback_gpu():
    model = _model("cuda")
    ref_model = copy.deepcopy(model)
    prev0, betas, K = _seq_state(13)
    packets = _packets(12, 300, big=5000)
    ref_bare = _reference_loop(ref_model, None, prev0, betas, K, packets)
    ref = _reference_loop(ref_model, SPEC, prev0, betas, K, packets)
    for scope in ("model", "filter", "all"):
        # capacity 3000: packets of 9000 / 30000 / 5000 events go through the host splat, the rest through the GPU
        tr = DF.DeployTracker(copy.deepcopy(ref_model), SPEC, capacity=3000, scope=scope)
        _same(_deploy_loop(tr, prev0, betas, K, packets), ref, f"capacity fallback, scope {scope}")
        assert tr.n_fallback == sum(int(b[-1]) > 3000 for _, b in packets) and tr.n_fallback >= 4
        tr = DF.DeployTracker(copy.deepcopy(ref_model), None, capacity=3000, scope=scope)
        _same(_deploy_loop(tr, prev0, betas, K, packets), ref_bare, f"bare tracker, scope {scope}")
    tr = DF.DeployTracker(copy.deepcopy(ref_model), None, graph=False)
    _same(_deploy_loop(tr, prev0, betas, K, packets), ref_bare, "bare tracker, eager")


@needs_gpu
def test_deploy_tracker_velocity_filter_state_is_kept_in_the_graph_gpu():
    model = _model("cuda")
    ref_model = copy.deepcopy(model)
    prev0, betas, K = _seq_state(14)
    packets = _packets(14, 400)
    # a segment restart in the middle (the velocity must restart from zero, in place)
    ref = np.concatenate([_reference_loop(ref_model, SPEC_VEL, prev0, betas, K, packets[:7]),
                          _reference_loop(ref_model, SPEC_VEL, prev0, betas, K, packets[7:])])
    plain = np.concatenate([_reference_loop(ref_model, SPEC, prev0, betas, K, packets[:7]),
                            _reference_loop(ref_model, SPEC, prev0, betas, K, packets[7:])])
    assert not np.array_equal(ref, plain)                              # the velocity term matters
    for kw in (dict(graph=False), dict(graph=True, scope="filter"), dict(graph=True, scope="all")):
        tr = DF.DeployTracker(copy.deepcopy(ref_model), SPEC_VEL, **kw)
        _same(_deploy_loop(tr, prev0, betas, K, packets, reset_at=(7,)), ref, f"velocity filter {kw}")


@needs_gpu
def test_deploy_tracker_new_sequence_needs_no_recapture_gpu():
    model = _model("cuda")
    ref_model = copy.deepcopy(model)
    tr = DF.DeployTracker(model, SPEC, scope="all")
    packets = _packets(7, 500)
    for seed in (21, 22):
        prev0, betas, K = _seq_state(seed)
        ref = _reference_loop(ref_model, SPEC, prev0, betas, K, packets)
        _same(_deploy_loop(tr, prev0, betas, K, packets), ref, f"sequence {seed}")
    assert set(tr._graphs) == {"all"}                                  # one graph served both sequences
    assert tr.model.mano.fast_chain and DF.is_fast_render(tr.model)


@needs_gpu
def test_deploy_tracker_host_filter_is_close_gpu():
    """The pre-DT2 host-side filter (numpy `anchor_blend`) differs from the device float64 filter only by float32 rounding of
    the transcendental functions; it exists for the timing ladder (tools/dt/bench_loop.py)."""
    model = _model("cuda")
    ref_model = copy.deepcopy(model)
    prev0, betas, K = _seq_state(15)
    packets = _packets(8, 600)
    ref = _reference_loop(ref_model, SPEC, prev0, betas, K, packets)
    tr = DF.DeployTracker(model, SPEC, filt="host", scope="model")
    got = _deploy_loop(tr, prev0, betas, K, packets)
    assert np.abs(got - ref).max() < 1e-3
    with pytest.raises(ValueError):
        DF.DeployTracker(copy.deepcopy(ref_model), SPEC, filt="host", scope="all")      # a host filter cannot be in the graph


@needs_gpu
def test_deploy_tracker_guards_gpu():
    model = _model("cuda")
    with pytest.raises(ValueError):
        DF.DeployTracker(model.train(), SPEC)
    model.eval()
    with pytest.raises(ValueError):
        DF.DeployTracker(model, SPEC, filt="none")
    with pytest.raises(ValueError):
        DF.DeployTracker(model, None, filt="gpu")
    with pytest.raises(ValueError):
        DF.DeployTracker(model, SPEC, lnes="host", scope="all")
    with pytest.raises(ValueError):
        DF.DeployTracker(model, dict(SPEC, host=True), filt="gpu")
    tr = DF.DeployTracker(model, SPEC)
    ev, bounds = random_packet(10, 1)
    with pytest.raises(RuntimeError, match="begin_sequence"):
        tr.step_events(ev, bounds)


@needs_gpu
def test_deploy_stages_have_no_host_sync_gpu():
    model = _model("cuda")
    tr = DF.DeployTracker(model, SPEC, graph=False)
    prev0, betas, K = _seq_state(16)
    tr.begin_sequence(betas, K)
    tr.begin_segment(prev0)
    ev, bounds = random_packet(5000, 5)
    tr.step_events(ev, bounds)
    tr.glnes.stage(ev, bounds)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        tr._stage_lnes()
        raw = tr._stage_model(True)
        tr._stage_filter(raw)
        tr._stage_model(False)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()


@needs_gpu
def test_unpatched_model_cannot_be_graphed_cleanly_gpu():
    """A model that still synchronises with the host is refused at capture, with a pointer to the fix, and the tracker
    (and CUDA) keep working once the model is patched."""
    model = _model("cuda")
    ref_model = copy.deepcopy(model)
    tr = DF.DeployTracker(model, SPEC, fast_render=False, scope="model")        # the original render syncs with the host
    prev0, betas, K = _seq_state(17)
    tr.begin_sequence(betas, K)
    tr.begin_segment(prev0)
    packets = _packets(3, 700)
    ev, bounds = packets[1]
    with pytest.raises(RuntimeError, match="enable_fast_render"):
        tr.step_events(ev, bounds)
    assert not tr._graphs
    DF.enable_fast_render(model)
    ref = _reference_loop(ref_model, SPEC, prev0, betas, K, packets)
    _same(_deploy_loop(tr, prev0, betas, K, packets), ref, "after patching the model")


# ------------------------------------------------------------------------------------------------ data gates
@pytest.fixture(scope="module")
def zgz():
    """The dt_dz_l3_s3407 model (last checkpoint), its config, the two val_core sequences and the evalx module."""
    import evalx as EX
    from mano_layer import ManoLayer
    from semkine.dataset import sequences_for_split
    cfg = load_config(RUN / "config_resolved.yaml")
    dev = torch.device("cuda")
    ckpt, step, _ = EX.find_ckpt(RUN, "last")
    model = MNISTModel.load_from_checkpoint(str(ckpt), cfg=cfg, map_location=dev).to(dev).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(dev).eval()
    root = Path(cfg["DATA"]["ROOT"])
    seqs = sequences_for_split(root, "val_core", Path(cfg["DATA"]["SPLITS_MANIFEST"]))
    return SimpleNamespace(EX=EX, cfg=cfg, model=model, mano=mano, root=root, seqs=seqs, dev=dev)


def _overall_ra(z, results):
    """`evalx.evaluate`'s overall RA (frame-weighted over the sequences) from per-sequence run_sequence results."""
    ms = {s: z.EX.per_step_metrics(z.mano, r, z.dev) for s, r in results.items()}
    n = sum(len(m["mpjpe_ra_mm"]) for m in ms.values())
    return float(sum(m["mpjpe_ra_mm"].sum() for m in ms.values()) / n), ms


def _bare_loop(z, model):
    rng = np.random.default_rng(0)
    return {s: z.EX.run_sequence(model, z.cfg, z.root, d, s, z.dev, rng, "model") for s, d in z.seqs}


@needs_gpu
@needs_data
def test_gate_b_evalx_closed_loop_with_fast_render_is_bitwise_gpu(zgz):
    """(b) the recorded evaluation of dt_dz_l3_s3407 (RA 10.04394757480695, evalx_val_core_last_tf_pert.npz) reproduced bit
    for bit by the same evalx loop on the model after `enable_fast_render`; disabled it is the same again."""
    z = zgz
    rec = json.loads((RUN / "evalx_val_core_last_tf_pert.json").read_text())["model"]["overall"]["mpjpe_ra_mm"]
    assert rec == 10.04394757480695
    npz = np.load(RUN / "evalx_val_core_last_tf_pert.npz")
    DF.enable_fast_render(z.model)
    try:
        res = _bare_loop(z, z.model)
    finally:
        DF.disable_fast_render(z.model)
    ra, _ = _overall_ra(z, res)
    for s, r in res.items():
        ref = npz[f"model|{s}|pred"]
        assert r["pred"].shape == ref.shape
        assert np.array_equal(r["pred"].view(np.int32), ref.view(np.int32)), f"{s}: predictions differ from the recorded npz"
    assert ra == 10.04394757480695, ra
    print(f"\n[gate b] fast render, evalx closed loop: RA {ra!r} (recorded {rec!r}); both predictions bit-identical to the npz")


def _all_packets(z):
    for s, d in z.seqs:
        events, offsets, aux, pos51 = ET.load_sequence(z.root, d, s)
        runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
        for a, b in runs:
            for end in np.arange(a + 49, b, 50, dtype=np.int64):
                yield s, events, offsets, int(end)


@needs_gpu
@needs_data
@pytest.mark.parametrize("chs", [("last",), ("last", "first"), ("last", "count", "first")], ids=lambda c: "+".join(c))
def test_gate_c_gpu_lnes_equals_build_lnes_on_all_eval_packets_gpu(zgz, chs):
    """(c) every evaluation packet of both zgz sequences: GPU image == host `build_lnes`, `torch.equal`."""
    g = DF.GpuLNES("cuda", 50, H, W, chs, capacity=262144)
    n_ok = n = n_over = n_max = 0
    for s, events, offsets, end in _all_packets(zgz):
        host = torch.from_numpy(ET.build_lnes(events, offsets, end, 50, chs))
        a0, a1 = int(offsets[end - 49]), int(offsets[end + 1])
        n_max = max(n_max, a1 - a0)
        if a1 - a0 > g.capacity:
            n_over += 1
            continue
        got = g.lnes(np.asarray(events[a0:a1]), DF.packet_bounds(offsets, end))[0].cpu()
        n += 1
        n_ok += int(torch.equal(got, host))
    assert n == 2590 - n_over and n_ok == n, (n_ok, n, n_over)
    print(f"\n[gate c] {'+'.join(chs)}: {n_ok}/{n} packets torch.equal, {n_over} over the capacity, max {n_max} events per packet")


def _deploy_run(z, **kw):
    """The protocol on both sequences with a DeployTracker; returns ({seq: run dict}, step seconds)."""
    model = copy.deepcopy(z.model)
    tr = DF.DeployTracker(model, DF.RECOMMENDED_SPEC, **kw)
    rng = np.random.default_rng(0)
    times = []
    res = {s: DF.run_closed_loop(tr, z.cfg, z.root, d, s, rng, times) for s, d in z.seqs}
    return res, times, tr


@needs_gpu
@needs_data
def test_gate_d_deploy_tracker_against_filter_eval_and_gate_e_graph_vs_eager_gpu(zgz):
    """(d) graph + GPU LNES + GPU filter on both sequences, against the recorded filter_eval json (same spec); (e) the same
    run eagerly gives the same bits at every step (2590 of 2590 packets, closed loop)."""
    z = zgz
    rec = json.loads(SWEEP.read_text())
    rec_ra = rec["model"]["overall"]["mpjpe_ra_mm"]
    assert rec["filter"]["spec"]["a_trans"] == 0.5 and rec_ra == 9.65916875301641
    npz = np.load(SWEEP.with_suffix(".npz"))
    res_g, t_g, tr_g = _deploy_run(z, graph=True, scope="all")
    res_e, t_e, tr_e = _deploy_run(z, graph=False)
    ra_g, _ = _overall_ra(z, res_g)
    ra_e, _ = _overall_ra(z, res_e)
    n_same = n_tot = n_rec = 0
    for s in res_g:
        a, b = res_g[s]["pred"], res_e[s]["pred"]
        same = (a.view(np.int32) == b.view(np.int32)).all(1)
        n_same += int(same.sum())
        n_tot += len(same)
        ref = npz[f"model|{s}|pred"]
        n_rec += int((a.view(np.int32) == ref.view(np.int32)).all(1).sum())
        assert np.array_equal(res_g[s]["count"], npz[f"model|{s}|count"])
    print(f"\n[gate d] DeployTracker graph/all RA {ra_g!r}, eager {ra_e!r}, filter_eval json {rec_ra!r}: "
          f"diff {abs(ra_g - rec_ra):.3e} mm; steps identical to the recorded npz {n_rec}/{n_tot}")
    print(f"[gate e] graph vs eager (closed loop, bitwise): {n_same}/{n_tot} packets; fallbacks {tr_g.n_fallback}")
    print(f"[timing, shared GPU, back-to-back] graph/all median {1e3 * np.median(t_g):.2f} ms, "
          f"eager {1e3 * np.median(t_e):.2f} ms per step")
    assert abs(ra_g - rec_ra) <= 1e-3, (ra_g, rec_ra)
    assert n_same == n_tot == 2590 and ra_g == ra_e


@needs_gpu
@needs_data
def test_gate_e_graph_replay_equals_eager_forward_on_40_real_packets_gpu(zgz):
    """(e) open loop on 40 real packets (the state fed in is the ground truth, so nothing is chaotic): the CUDA-graph replay
    of the model's own forward == the original eager forward (original render, original syncs), 40 of 40, bit for bit."""
    z = zgz
    s, d = z.seqs[1]
    events, offsets, aux, pos51 = ET.load_sequence(z.root, d, s)
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=z.dev).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=z.dev).view(1, 3, 3)
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
    ends = [int(e) for a, b in runs for e in np.arange(a + 49, b, 50)][::7][:40]
    assert len(ends) == 40
    model = copy.deepcopy(z.model)
    model.set_hand_context(betas, K)
    xs = [torch.from_numpy(ET.build_lnes(events, offsets, e, 50, ("last",))).unsqueeze(0).to(z.dev) for e in ends]
    ps = [torch.from_numpy(pos51[e - 50].astype(np.float32)).view(1, -1).to(z.dev) for e in ends]
    with torch.no_grad():
        eager = [model(x, p) for x, p in zip(xs, ps)]
        DF.enable_fast_render(model)
        gf = RF.GraphedForward(model)
        gf.capture(xs[0], ps[0], betas, K)
        graph = [gf(x, p) for x, p in zip(xs, ps)]
        fast_eager = [model(x, p) for x, p in zip(xs, ps)]
    n_graph = sum(bits_equal(a, b) for a, b in zip(graph, eager))
    n_fast = sum(bits_equal(a, b) for a, b in zip(fast_eager, eager))
    print(f"\n[gate e] open loop, 40 real packets: graph replay == original eager {n_graph}/40, fast-render eager {n_fast}/40")
    assert n_graph == 40 and n_fast == 40


@needs_gpu
@needs_data
def test_bench_loop_smoke_gpu(tmp_path):
    """tools/dt/bench_loop.py runs end to end (all seven variants, one short round, back to back) and its sanity line says
    every variant's outputs equal v0's."""
    import subprocess
    out = tmp_path / "bench.json"
    cmd = [sys.executable, str(REPO / "tools" / "dt" / "bench_loop.py"), "--run-dir", str(RUN), "--packets", "24", "--rounds", "1",
           "--modes", "b2b", "--no-latency-row", "--tag", "smoke", "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy(), timeout=900)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-3000:])
    j = json.loads(out.read_text())
    assert set(j["variants"]) == set(DF_BENCH_VARIANTS)
    for v, per in j["sanity_max_abs_vs_v0"].items():
        # v0..v3 share the host filter and run bit-identical stages; v4.. run the device float64 filter (host float64 vs
        # device float64 numpy / torch: at most float32 rounding of the state)
        assert all(d <= (0.0 if v in ("v0", "v1", "v2", "v3") else 1e-5) for d in per.values()), (v, per)
    for v in DF_BENCH_VARIANTS:
        for kind in ("global", "local"):
            st = j["table"][v][kind]["b2b"]
            assert st["n"] == 24 and 0 < st["min"] <= st["p50"] <= st["p90"] <= st["p99"]


DF_BENCH_VARIANTS = ("v0", "v1", "v2", "v3", "v4", "v4g", "v5")
