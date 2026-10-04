"""Fast data pipe (`SEMKINE_FAST_PIPE` / `DATA.FAST_PIPE`): bit-for-bit against the legacy sample path.

`SemKineDataset.__getitem__` has an opt-in re-implementation of the dense training sample that costs
about a third of the legacy path and must return *the same bits* (class comment `fast pipe` in
`semkine/dataset.py` gives the argument; this file is the check). Pinned here, on small samples that run
on the CPU in about a minute:

  (a) the switch: off by default, on by the argument / `SEMKINE_FAST_PIPE` / `DATA.FAST_PIPE`, and
      declined (with a reason) for what the argument does not cover;
  (b) the sample itself, for every branch: domrand on / off, `depth` and `focal` scale, geometric
      identity, no hot pixels, hot pixels dense enough to collide, no dropout, dropout of everything,
      fixed and speed-augmented windows, flip / swap / neither, event channels `last` and
      `last,first`, and the triplet sample (the draws of the two earlier windows follow the plain
      sample's, so a stream that drifted anywhere shows up there);
  (c) edge windows: a window without events, the first samples of a run;
  (d) the guards: a sub-ms stamp >= 1000 or an event outside the frame makes the fast path decline and
      the legacy path answer, with the same bits;
  (e) `domrand.map_pixel_grid` against `domrand.transform_events`, the function it re-evaluates;
  (f) a real `DataLoader` with workers, batch by batch.

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_dt_pipe_fast.py -q -s
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "model")]
from config import load_config                                  # noqa: E402
from semkine import domrand as DR                               # noqa: E402
from semkine import dataset as DS                               # noqa: E402
from semkine.dataset import build_dataset                       # noqa: E402

CFG_DIR = REPO / "configs" / "dt"
DATA_ROOT = REPO / "data" / "hand_data51"
H, W = 180, 240

pytestmark = pytest.mark.skipif(not DATA_ROOT.exists(), reason="hand_data51 not available")


# ------------------------------------------------------------------------------------- helpers
def _components(cfg):
    return np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)


def _cfg(base="dt_dz_l3.yaml", **over):
    """A config of `configs/dt`, with dotted-path overrides (`AUG.DOMRAND.KEEP_MIN=0.5`)."""
    cfg = copy.deepcopy(load_config(CFG_DIR / base))
    for dotted, v in over.items():
        d = cfg
        keys = dotted.split("__")
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = v
    return cfg


#: a diverse pool of training sequences: long runs (lyq_local, ch_global), many short runs (lyh_local 56
#: runs, lpc_local 28, ycy_local 11) and two `_v4` recordings, which carry their own camera K. The
#: full 72-sequence set is covered by the G1 sweep of the work package (tools/dt2/pipe_fast_g1.py), not
#: here: indexing 4.5M samples takes seconds per dataset.
POOL = ("lyq_local", "lyh_local", "lpc_local", "ycy_local", "ch_global", "ch_global_v4", "ycy_local_v4")


@pytest.fixture(scope="module", autouse=True)
def _train_pool():
    orig = DS.sequences_for_split

    def pooled(root, split, manifest_path=None):
        seqs = orig(root, split, manifest_path)
        return [s for s in seqs if s[0] in POOL] if split == "train" else seqs

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(DS, "sequences_for_split", pooled)
        yield
    _CACHE.clear()
    _HANDLES.clear()


_CACHE = {}
_HANDLES = {}


def _shared_handles(ds):
    """Open every sequence once and share the handles (memmaps, MANO pivot) between all variants:
    they depend on the data only, and each dataset would otherwise hold its own file descriptors."""
    if not _HANDLES:
        for si in range(len(ds.sequences)):
            ds._handle(si)
        _HANDLES.update(ds._handles)
    ds._handles = _HANDLES


def _ds(base="dt_dz_l3.yaml", **over):
    """Dataset of a config variant (memoised), fast pipe off; the caller flips it with `set_fast_pipe`."""
    key = (base, repr(sorted(over.items())))
    if key not in _CACHE:
        cfg = _cfg(base, **over)
        ds = build_dataset(cfg, "train", _components(cfg), train=True)
        ds.set_fast_pipe(False)
        _shared_handles(ds)
        _CACHE[key] = ds
    return _CACHE[key]


def _flat(x):
    if isinstance(x, dict):
        out = []
        for k in ("w0", "w1", "w2"):
            out += list(x[k])
        return out + [x["valid"]]
    return list(x)


def _assert_same(a, b, what=""):
    fa, fb = _flat(a), _flat(b)
    assert len(fa) == len(fb), what
    for i, (x, y) in enumerate(zip(fa, fb)):
        assert x.dtype == y.dtype and x.shape == y.shape, f"{what} tensor {i}: {x.dtype}{tuple(x.shape)} vs {y.dtype}{tuple(y.shape)}"
        assert torch.equal(x, y), f"{what} tensor {i} differs"
        # torch.equal calls -0.0 and 0.0 equal; the bytes must match too
        assert x.contiguous().numpy().tobytes() == y.contiguous().numpy().tobytes(), f"{what} tensor {i}: bytes differ"


def _legacy(ds, i):
    ds.set_fast_pipe(False)
    return ds[i]


def _pick(ds, n, seed):
    rs = np.random.default_rng(seed)
    idx = [int(i) for i in rs.integers(0, len(ds), n)]
    rel = ds.index[:, 1] - ds.index[:, 2]
    starts = np.flatnonzero(rel < ds.max_w + 4)         # first samples of a run: clipped windows
    return idx + [int(i) for i in rs.choice(starts, max(n // 6, 4), replace=False)]


#: (id, base config, overrides). `__` is the path separator of `_cfg`.
VARIANTS = [
    ("dz_l3", "dt_dz_l3.yaml", {}),
    ("focal", "dt_base.yaml", {}),
    ("hot_2e-5", "dt_dz_l3_nfh.yaml", {}),
    ("keep_0.10", "dt_dz_l3_nfk.yaml", {}),
    ("w50_fixed", "dt_dz_l3_w50.yaml", {}),
    ("w100", "dt_dz_l3_w100.yaml", {}),
    ("last_first", "dt_dz_l3_lf.yaml", {}),
    ("no_swap", "dt_dz_l3_nosw.yaml", {}),
    ("domrand_off", "dt_dz_l3.yaml", {"AUG__DOMRAND__ENABLED": False}),
    ("hot_0", "dt_dz_l3.yaml", {"AUG__DOMRAND__HOT_PIXEL_RATE": 0.0}),
    ("hot_dense_collide", "dt_dz_l3.yaml", {"AUG__DOMRAND__HOT_PIXEL_RATE": 3.0e-3}),
    ("keep_1", "dt_dz_l3.yaml", {"AUG__DOMRAND__KEEP_MIN": 1.0, "AUG__DOMRAND__KEEP_MAX": 1.0}),
    ("keep_0_drop_all", "dt_dz_l3.yaml", {"AUG__DOMRAND__KEEP_MIN": 0.0, "AUG__DOMRAND__KEEP_MAX": 0.0}),
    ("geom_identity", "dt_dz_l3.yaml", {"AUG__DOMRAND__ROLL_DEG": 0.0, "AUG__DOMRAND__SCALE_MIN": 1.0,
                                        "AUG__DOMRAND__SCALE_MAX": 1.0, "AUG__DOMRAND__SHIFT_PX": 0.0}),
    ("big_geometry_oob", "dt_dz_l3.yaml", {"AUG__DOMRAND__ROLL_DEG": 90.0, "AUG__DOMRAND__SCALE_MIN": 0.4,
                                           "AUG__DOMRAND__SCALE_MAX": 2.5, "AUG__DOMRAND__SHIFT_PX": 60.0}),
    ("no_flip_no_speed", "dt_dz_l3.yaml", {"AUG__POLARITY_FLIP": False, "AUG__SPEED_AUG": False}),
    ("lf_domrand_off", "dt_dz_l3_lf.yaml", {"AUG__DOMRAND__ENABLED": False}),
    ("lf_no_swap_hot_dense", "dt_dz_l3_lf.yaml", {"AUG__PIXEL_POLARITY_SWAP": False,
                                                  "AUG__DOMRAND__HOT_PIXEL_RATE": 3.0e-3}),
    ("triplet", "dt_acc.yaml", {}),
    ("triplet_lf", "dt_acc.yaml", {"DATA__EVENT_CHANNELS": ["last", "first"]}),
    ("triplet_off_no_swap", "dt_acc.yaml", {"AUG__PIXEL_POLARITY_SWAP": False, "AUG__DOMRAND__ENABLED": False}),
    ("triplet_frac_1", "dt_acc.yaml", {"DATA__TRIPLET_FRAC": 1.0}),
]


# ------------------------------------------------------------------------------------- (a) switch
def test_switch_off_by_default_and_on_by_argument_env_and_config(monkeypatch):
    monkeypatch.delenv(DS.FAST_PIPE_ENV, raising=False)
    cfg = _cfg()
    comps = _components(cfg)
    ds = build_dataset(cfg, "train", comps, train=True)
    assert not ds.fast_pipe and not ds.fast_pipe_active, "the fast pipe must be opt-in"

    monkeypatch.setenv(DS.FAST_PIPE_ENV, "1")
    ds = build_dataset(cfg, "train", comps, train=True)
    assert ds.fast_pipe and ds.fast_pipe_active and ds.fast_pipe_reason is None
    monkeypatch.setenv(DS.FAST_PIPE_ENV, "0")
    assert not build_dataset(cfg, "train", comps, train=True).fast_pipe
    monkeypatch.delenv(DS.FAST_PIPE_ENV)

    cfg2 = copy.deepcopy(cfg)
    cfg2["DATA"]["FAST_PIPE"] = True
    assert build_dataset(cfg2, "train", comps, train=True).fast_pipe_active
    cfg2["DATA"]["FAST_PIPE"] = "false"
    with pytest.raises(ValueError):
        build_dataset(cfg2, "train", comps, train=True)       # a quoted "false" is truthy; refuse it

    # an explicit argument beats the environment
    monkeypatch.setenv(DS.FAST_PIPE_ENV, "1")
    d = build_dataset(cfg, "train", comps, train=True)
    d.set_fast_pipe(False)
    assert not d.fast_pipe_active


def test_declined_configurations_say_why(monkeypatch):
    monkeypatch.setenv(DS.FAST_PIPE_ENV, "1")
    cfg = _cfg()
    comps = _components(cfg)
    val = build_dataset(cfg, "val_core", comps, train=False)
    assert val.fast_pipe and not val.fast_pipe_active and "evaluation" in val.fast_pipe_reason
    raw = build_dataset(cfg, "train", comps, train=True, input_mode="raw_packed")
    assert not raw.fast_pipe_active and "legacy_lnes" in raw.fast_pipe_reason
    cfg_c = _cfg(DATA__EVENT_CHANNELS=["last", "count"])
    d = build_dataset(cfg_c, "train", comps, train=True)
    assert not d.fast_pipe_active and "EVENT_CHANNELS" in d.fast_pipe_reason
    cfg_c = _cfg(DATA__EVENT_CHANNELS=["last", "count", "first"])
    assert not build_dataset(cfg_c, "train", comps, train=True).fast_pipe_active
    # a declined dataset still serves samples, from the legacy path
    assert len(d[0]) == 5


# ------------------------------------------------------------------------------------- (b) samples
@pytest.mark.parametrize("name,base,over", VARIANTS, ids=[v[0] for v in VARIANTS])
def test_fast_sample_is_bitwise_the_legacy_sample(name, base, over):
    ds = _ds(base, **over)
    ds.set_fast_pipe(True)
    assert ds.fast_pipe_active, ds.fast_pipe_reason
    n = 80 if name in ("dz_l3", "last_first", "triplet", "triplet_lf") else 36
    for i in _pick(ds, n, seed=zlib.crc32(name.encode()) % 1000):
        a = _legacy(ds, i)
        ds.set_fast_pipe(True)
        b = ds._fast_item(i)                  # raises _FastFallback instead of silently falling back
        _assert_same(a, b, f"{name} idx {i}")
        _assert_same(a, ds[i], f"{name} idx {i} through __getitem__")
    ds.set_fast_pipe(False)


def test_triplet_valid_and_aliased_samples_are_both_seen():
    ds = _ds("dt_acc.yaml")
    ds.set_fast_pipe(True)
    valid = []
    for i in _pick(ds, 90, seed=5):
        a = _legacy(ds, i)
        ds.set_fast_pipe(True)
        b = ds._fast_item(i)
        _assert_same(a, b, f"idx {i}")
        valid.append(bool(b["valid"]))
        if not valid[-1]:
            assert b["w0"] is b["w1"] is b["w2"], "an invalid triplet aliases one window"
    ds.set_fast_pipe(False)
    assert any(valid) and not all(valid), "the sample must contain valid and invalid triplets"


# ------------------------------------------------------------------------------------- (c) edge windows
def _patched(ds, si, **repl):
    h = ds._handle(si)
    ds._handles[si] = dataclasses.replace(h, **repl)
    return h


def test_window_without_events_and_window_with_few():
    ds = _ds("dt_dz_l3.yaml")
    rs = np.random.default_rng(3)
    for trial in range(10):
        i = int(rs.integers(0, len(ds)))
        rng = np.random.default_rng((ds.seed * 1_000_003 + i) & 0x7FFFFFFF)
        si, end, window, _ = ds._window(i, rng)
        h = ds._handle(si)
        start = end - window + 1
        off = h.offsets.copy()
        keep_n = [0, 0, 1, 3][trial % 4]                    # events left in the window: none, one, a few
        off[start + 1:end + 2] = off[start]
        if keep_n:
            off[end + 1] = off[start] + keep_n
            off[end + 2:] = np.maximum(off[end + 2:], off[end + 1])
        try:
            _patched(ds, si, offsets=off)
            a = _legacy(ds, i)
            ds.set_fast_pipe(True)
            b = ds._fast_item(i)
            _assert_same(a, b, f"idx {i} with {keep_n} events")
        finally:
            ds._handles[si] = h
            ds.set_fast_pipe(False)


# ------------------------------------------------------------------------------------- (d) guards
def _small_sequence(ds):
    sizes = [(len(ds._handle(si).events), si) for si in range(len(ds.sequences))]
    return min(sizes)[1]


def test_submillisecond_stamp_at_or_above_1000_declines_to_the_legacy_path():
    ds = _ds("dt_dz_l3.yaml")
    si = _small_sequence(ds)
    h = ds._handle(si)
    idx = np.flatnonzero(ds.index[:, 0] == si)[:12]
    bad = np.full(len(h.events), 1500, np.uint16)           # `us // 1000` would then be ms + 1
    try:
        _patched(ds, si, tsub=bad)
        for i in idx:
            i = int(i)
            ds.set_fast_pipe(True)
            with pytest.raises(DS._FastFallback):
                ds._fast_item(i)
            a = _legacy(ds, i)
            ds.set_fast_pipe(True)
            _assert_same(a, ds[i], f"idx {i}: declined sample answered by the legacy path")
    finally:
        ds._handles[si] = h
        ds.set_fast_pipe(False)


def test_event_outside_the_frame_declines_to_the_legacy_path():
    ds = _ds("dt_dz_l3_nosw.yaml")
    si = _small_sequence(ds)
    h = ds._handle(si)
    ev = np.array(h.events)
    ev[:, 0] = np.where(np.arange(len(ev)) % 7 == 0, 250, ev[:, 0])    # x >= W, but < 256
    idx = np.flatnonzero(ds.index[:, 0] == si)[:6]
    try:
        _patched(ds, si, events=ev)
        ds.set_fast_pipe(True)
        declined = 0
        for i in idx:
            i = int(i)
            try:
                ds._fast_item(i)
            except DS._FastFallback:
                declined += 1
        assert declined == len(idx)
    finally:
        ds._handles[si] = h
        ds.set_fast_pipe(False)


# ------------------------------------------------------------------------------------- (e) pixel map
def test_map_pixel_grid_is_transform_events():
    K = np.array([[247.5, 0, 119.7], [0, 247.3, 90.2], [0, 0, 1]], np.float32)
    rs = np.random.default_rng(0)
    xs0 = rs.integers(0, W, 4000)
    ys0 = rs.integers(0, H, 4000)
    n_oob = 0
    for trial in range(300):
        dr = DR.DomRandSample(
            roll_rad=float(np.deg2rad(rs.uniform(-60, 60))), scale=float(np.exp(rs.uniform(-1.0, 1.0))),
            shift_px=(float(rs.uniform(-70, 70)), float(rs.uniform(-70, 70))), keep=1.0, n_hot=0)
        if trial % 25 == 0:
            dr = DR.DomRandSample(0.0, 1.0, (0.0, 0.0), 1.0, 0)
        xs, ys = xs0.copy(), ys0.copy()
        keep = DR.transform_events(xs, ys, dr, K, 0.375, W, H)
        ref = np.where(keep, ys * W + xs, H * W)
        got = DR.map_pixel_grid(xs0, ys0, dr, K, 0.375, W, H)
        assert np.array_equal(ref, got), trial
        n_oob += int((~keep).sum())
    assert n_oob > 1000, "the test must exercise events that leave the frame"


# ------------------------------------------------------------------------------------- (f) DataLoader
@pytest.mark.parametrize("base", ["dt_dz_l3.yaml", "dt_dz_l3_lf.yaml", "dt_acc.yaml"])
def test_dataloader_batches_are_the_same_bytes(base):
    ds = _ds(base)

    def digest(fast):
        ds.set_fast_pipe(fast)
        g = torch.Generator().manual_seed(11)
        dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2, generator=g, drop_last=True)
        out = []
        for b, batch in enumerate(dl):
            h = hashlib.sha256()
            for t in _flat(batch):
                h.update(t.contiguous().numpy().tobytes())
            out.append(h.hexdigest())
            if b == 3:
                break
        return out

    try:
        assert digest(False) == digest(True)
    finally:
        ds.set_fast_pipe(False)


# ------------------------------------------------------------------------------------- cost (informative)
def test_fast_path_is_not_slower(capsys):
    ds = _ds("dt_dz_l3.yaml")
    idx = _pick(ds, 40, seed=9)
    for fast in (False, True):
        ds.set_fast_pipe(fast)
        for i in idx[:10]:
            ds[i]
    t = {}
    for fast in (False, True, False, True):
        ds.set_fast_pipe(fast)
        a = time.perf_counter()
        for i in idx:
            ds[i]
        t.setdefault(fast, []).append(time.perf_counter() - a)
    ds.set_fast_pipe(False)
    legacy, fast = min(t[False]) / len(idx) * 1e3, min(t[True]) / len(idx) * 1e3
    with capsys.disabled():
        print(f"\n[pipe_fast] legacy {legacy:.2f} ms/sample, fast {fast:.2f} ms/sample, x{legacy / fast:.2f}")
    assert fast < legacy
