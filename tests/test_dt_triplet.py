"""Triplet training sample of the dense LNES path (task dataset-triplet).

With `DATA.TRIPLET` on, `SemKineDataset.__getitem__` returns `{"w0", "w1", "w2", "valid"}`: three
consecutive, equally long windows, the last of which (`w2`) is the plain sample bit for bit, so a
model can put an acceleration (second-difference) loss on three consecutive predictions. Pinned here:

  (a) with the flag off the dataset still returns what the *unmodified* dataset returned: a dump of
      48 train + 16 val (+ 16 train-without-domrand) samples taken before the edit
      (`outputs/dt/ref/dataset_ref_rt_cnntrack.pt`, made by `outputs/dt/ref/make_dataset_ref.py`
      from the pristine file, which that script refuses to read once it has been edited);
  (b) `w2` is the plain sample, valid or not; the geometry of the three windows (targets exactly w ms
      apart, events of the right ms ranges, one camera / one polarity flip / one swap mask) and the
      draws that are per window (event dropout, hot pixels, prev noise);
  (c) the valid coin and the run-history rule; invalid samples alias `w2` and compute nothing extra;
  (d) `default_collate` under a real `DataLoader` with workers, deterministic;
  (e) the CPU cost per sample against the plain dataset.

    python -m pytest tests/test_dt_triplet.py -q -s
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "model")]
from config import load_config                                  # noqa: E402
from semkine import domrand as DR                               # noqa: E402
from semkine.dataset import build_dataset                       # noqa: E402

CFG = REPO / "configs" / "rt" / "rt_cnntrack.yaml"
REF = Path(os.environ.get("DT_TRIPLET_REF",
                          REPO / "outputs" / "dt" / "ref" / "dataset_ref_rt_cnntrack.pt"))
DATA_ROOT = REPO / "data" / "hand_data51"
H, W = 180, 240
NAMES = ("lnes", "prev", "target", "betas", "K")
#: a cheap pool of training sequences: long runs (lyq_local, ch_global) and many short ones
#: (lyh_local 56 runs, lpc_local 28, ycy_local 11), which is where the history rule bites
POOL = ("lyq_local", "lyh_local", "lpc_local", "ycy_local", "ch_global")
#: first index of the pool that lies deep inside a long run (lyq_local: one run of 64.7 s)
DEEP = 20000

pytestmark = pytest.mark.skipif(not DATA_ROOT.exists(), reason="hand_data51 not available")


# ------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def cfg():
    return load_config(CFG)


@pytest.fixture(scope="module")
def components(cfg):
    return np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)


def _variant(cfg, triplet=None, frac=None, domrand="full", noise=True, **data):
    """A copy of the rt_cnntrack config. `domrand`: full (as shipped) / geom (the pixel map is the
    only randomness: no dropout, no hot pixels) / off. `noise=False` zeroes the prev noise."""
    c = copy.deepcopy(cfg)
    if triplet is not None:
        c["DATA"]["TRIPLET"] = triplet
    if frac is not None:
        c["DATA"]["TRIPLET_FRAC"] = frac
    c["DATA"].update(data)
    d = c["AUG"]["DOMRAND"]
    if domrand == "off":
        d["ENABLED"] = False
    elif domrand == "geom":
        d.update(KEEP_MIN=1.0, KEEP_MAX=1.0, HOT_PIXEL_RATE=0.0)
    if not noise:
        c["TRACK"].update(PREV_NOISE_T=0.0, PREV_NOISE_R=0.0, PREV_NOISE_POSE=0.0)
    return c


@pytest.fixture(scope="module")
def full_off(cfg, components):
    return build_dataset(cfg, "train", components, train=True)


@pytest.fixture(scope="module")
def full_on(cfg, components):
    return build_dataset(_variant(cfg, triplet=True), "train", components)


@pytest.fixture(scope="module")
def pool_manifest(cfg, tmp_path_factory):
    m = json.loads(Path(cfg["DATA"]["SPLITS_MANIFEST"]).read_text())["train"]
    sub = {"train": {"trials": list(POOL), "legacy_dir": {s: m["legacy_dir"][s] for s in POOL}}}
    p = tmp_path_factory.mktemp("dt_triplet") / "splits_pool.json"
    p.write_text(json.dumps(sub))
    return p


@pytest.fixture(scope="module")
def pool(cfg, components, pool_manifest):
    """`pool(**variant)` -> a train dataset over POOL, built through build_dataset like training."""
    def make(train=None, input_mode=None, **variant):
        return build_dataset(_variant(cfg, **variant), "train", components, train=train,
                             input_mode=input_mode, manifest_path=pool_manifest)
    return make


# -------------------------------------------------------------------------------------- helpers
def _same(a, b) -> bool:
    """Two legacy 5-tuples, bit for bit."""
    return len(a) == len(b) == 5 and all(torch.equal(x, y) for x, y in zip(a, b))


def _aliased(a, b) -> bool:
    """The very same tensor objects, i.e. nothing was computed for `a`."""
    return len(a) == len(b) == 5 and all(x is y for x, y in zip(a, b))


def _first_of_runs(ds):
    ix = ds.index                                    # (seq, end, run start, run end)
    new = np.ones(len(ix), bool)
    new[1:] = (ix[1:, 0] != ix[:-1, 0]) | (ix[1:, 2] != ix[:-1, 2])
    return np.nonzero(new)[0]


def _probe(ds, n_spread=24, offsets=(0, 130), every=2):
    """Indices spread over the dataset plus samples at fixed offsets after run starts."""
    firsts = _first_of_runs(ds)[::every]
    near = (firsts[:, None] + np.asarray(offsets)[None]).ravel()
    spread = np.linspace(0, len(ds) - 1, n_spread).astype(np.int64)
    return sorted(set(np.minimum(np.concatenate([near, spread]), len(ds) - 1).tolist()))


def _replay(ds, idx):
    """The plain sample's own first draws, replayed from the (seed, index) stream: the window and
    the domrand realisation. Returns the generator, left right after them."""
    rng = np.random.default_rng((ds.seed * 1_000_003 + int(idx)) & 0x7FFFFFFF)
    si, end, w, run_a = ds._window(int(idx), rng)
    dr = DR.sample_params(ds.domrand, rng, ds.height * ds.width * 2 * w)
    return rng, si, end, w, run_a, dr


def _draw_polarity(rng, ds):
    """The plain sample's polarity draws, which follow the window / domrand / noise draws."""
    flip = int(rng.integers(0, 2)) == 1
    swap = rng.integers(0, 2, size=(ds.height, ds.width)).astype(bool)
    return flip, swap


def _window_events(h, e, w):
    """x, y, polarity, ms-in-window of the events of ms [e - w + 1, e], straight from the files."""
    s = e - w + 1
    a0, a1 = int(h.offsets[s]), int(h.offsets[e + 1])
    ev = np.asarray(h.events[a0:a1]).astype(np.int64)
    ms = np.repeat(np.arange(w, dtype=np.int64), np.diff(h.offsets[s:e + 2]))
    return ev[:, 0].copy(), ev[:, 1].copy(), np.clip(ev[:, 2], 0, 1), ms


def _lnes(xs, ys, ps, ms, w, flip, swap):
    """The LNES an independent reading of the spec gives: one normalised timestamp per (pixel,
    polarity), the later event overwriting the earlier one; flip and swap mask on the polarity."""
    if flip:
        ps = 1 - ps
    if len(xs):
        ps = np.where(swap[ys, xs], 1 - ps, ps)
    img = np.zeros((H, W, 2), np.float32)
    if len(xs):
        img[ys, xs, ps] = ms.astype(np.float32) / float(w)
    return torch.from_numpy(img)


def _assert_batches_equal(a, b):
    assert set(a) == set(b)
    assert torch.equal(a["valid"], b["valid"])
    for k in ("w0", "w1", "w2"):
        assert len(a[k]) == len(b[k]) == 5
        for x, y in zip(a[k], b[k]):
            assert torch.equal(x, y)


# ---------------------------------------------------------------------- (a) the flag off: no change
@pytest.fixture(scope="module")
def ref():
    if not REF.exists():
        pytest.fail(f"reference dump {REF} is missing; it has to come from the pristine dataset "
                    f"(outputs/dt/ref/make_dataset_ref.py, run before the edit)")
    return torch.load(REF)


def test_flag_off_reproduces_the_dump_of_the_unmodified_dataset(ref, cfg, components, full_off):
    meta = ref["meta"]
    val = build_dataset(cfg, "val", components)
    c_nodr = _variant(cfg, domrand="off")
    nodr = build_dataset(c_nodr, "train", components, train=True)
    assert not full_off.triplet and not val.triplet and not nodr.triplet
    assert len(full_off) == meta["len_train"] and len(val) == meta["len_val"], \
        "the sample index changed (data or index construction), the dump no longer applies"
    n = 0
    for key, ds in (("train", full_off), ("val", val), ("train_nodr", nodr)):
        for row in ref[key]:
            got = ds[row["idx"]]
            assert isinstance(got, tuple) and len(got) == 5
            for name, g in zip(NAMES, got):
                assert torch.equal(g, row[name]), f"{key}[{row['idx']}].{name} differs from the dump"
            n += 1
    assert n == 48 + 16 + 16


def test_flag_on_keeps_w2_equal_to_the_dump(ref, full_on):
    assert full_on.triplet and full_on.triplet_frac == 1.0 / 3.0
    n_valid = 0
    for row in ref["train"]:
        s = full_on[row["idx"]]
        assert _same(s["w2"], tuple(row[n] for n in NAMES)), f"w2 at {row['idx']} moved"
        n_valid += int(s["valid"])
    assert 0 < n_valid < len(ref["train"]), "the dump should exercise both branches"


def test_the_flag_is_ignored_off_the_training_split(cfg, components, ref):
    on = _variant(cfg, triplet=True)
    for split in ("val", "test"):
        ds = build_dataset(on, split, components)
        assert not ds.triplet and ds.fixed_window == 50
        assert isinstance(ds[0], tuple) and len(ds[0]) == 5
    val = build_dataset(on, "val", components)
    for row in ref["val"]:
        assert _same(val[row["idx"]], tuple(row[n] for n in NAMES))
    # raw input faces return packets, never triplets, whatever the config says
    assert not build_dataset(on, "val", components, input_mode="raw_packed").triplet


# ------------------------------------------------------------------------- config / constructor
def test_triplet_needs_the_dense_lnes_face(cfg, components, pool_manifest):
    on = _variant(cfg, triplet=True)
    for mode in ("raw_packed", "both"):
        with pytest.raises(ValueError, match="legacy_lnes"):
            build_dataset(on, "train", components, input_mode=mode, manifest_path=pool_manifest)
    with pytest.raises(ValueError, match="legacy_lnes"):
        build_dataset(_variant(cfg, triplet=True, INPUT_MODE="raw_packed"), "train", components,
                      manifest_path=pool_manifest)
    # off (or absent) it stays what it was
    for mode in ("raw_packed", "both"):
        ds = build_dataset(_variant(cfg, triplet=False), "train", components, input_mode=mode,
                           manifest_path=pool_manifest)
        assert not ds.triplet


def test_triplet_frac_and_flag_validation(cfg, components, pool_manifest):
    def build(**kw):
        return build_dataset(_variant(cfg, **kw), "train", components, manifest_path=pool_manifest)

    for bad in (0, 0.0, -0.5, 1.0000001, 2, float("nan"), "x", None):
        with pytest.raises(ValueError, match="TRIPLET_FRAC"):
            build(triplet=True, TRIPLET_FRAC=bad)
    for good in (1.0, 0.5, 1e-6):
        ds = build(triplet=True, TRIPLET_FRAC=good)
        assert ds.triplet and ds.triplet_frac == good
    # an unused fraction is not validated: the flag off means no new behaviour of any kind
    assert not build(triplet=False, TRIPLET_FRAC=7).triplet
    assert not build().triplet
    with pytest.raises(ValueError, match="must be a bool"):
        build(triplet="false")          # a quoted YAML scalar is truthy; refuse rather than switch on


# ------------------------------------------------------------------ (b) w2 is the plain sample
def test_w2_is_the_plain_sample_valid_or_not(full_off, full_on):
    firsts = _first_of_runs(full_off)
    idx = sorted(set(np.linspace(0, len(full_off) - 1, 40).astype(int).tolist()
                     + np.minimum(firsts[::6] + 5, len(full_off) - 1).tolist()))
    n_valid = 0
    for i in idx:
        plain, s = full_off[i], full_on[i]
        assert _same(s["w2"], plain), f"w2 differs from the plain sample at {i}"
        n_valid += int(s["valid"])
    assert 0 < n_valid < len(idx), (n_valid, len(idx))


def test_the_sample_is_a_pure_function_of_seed_and_index(cfg, components, pool, pool_manifest):
    a, b = pool(triplet=True, frac=1.0), pool(triplet=True, frac=1.0)
    c = build_dataset(_variant(cfg, triplet=True, frac=1.0) | {"SEED": 8}, "train", components,
                      manifest_path=pool_manifest)
    for i in (DEEP, DEEP + 1, DEEP + 777):
        x, y, z = a[i], b[i], c[i]
        assert bool(x["valid"]) and torch.equal(x["valid"], y["valid"])
        for k in ("w0", "w1", "w2"):
            assert _same(x[k], y[k])
        assert not torch.equal(x["w1"][0], z["w1"][0]), "the seed does not reach the extra windows"


# ---------------------------------------------------------------- (b) geometry of the 3 windows
def test_windows_are_consecutive_and_labelled_by_the_right_milliseconds(pool):
    """DR off and no prev noise: every number can be checked against the sequence files."""
    ds = pool(triplet=True, frac=1.0, domrand="off", noise=False)
    assert ds.triplet and not ds.domrand.enabled
    n_valid = n_unique = n_checked = n_shift_differs = 0
    for i in _probe(ds):
        rng, si, end, w, run_a, _ = _replay(ds, i)
        flip, swap = _draw_polarity(rng, ds)          # no domrand / noise draws sit in between
        s = ds[i]
        assert set(s) == {"w0", "w1", "w2", "valid"}
        valid = bool(s["valid"])
        assert valid == (end - run_a + 1 >= 3 * w)    # frac = 1: the coin always passes
        if not valid:
            continue
        n_valid += 1
        h = ds._handle(si)
        ends = (end - 2 * w, end - w, end)
        for k, e in enumerate(ends):
            lnes, prev, target, betas, K = s[f"w{k}"]
            assert torch.equal(target, torch.from_numpy(h.pos51[e])), (i, k)
            assert torch.equal(prev, torch.from_numpy(h.pos51[e - w + 1])), (i, k)
            assert torch.equal(betas, torch.from_numpy(h.betas))
            assert torch.equal(K, torch.from_numpy(h.camera_K))
            want = _lnes(*_window_events(h, e, w), w, flip, swap)
            assert torch.equal(lnes, want), f"window {k} of sample {i}: LNES is not that of ms " \
                                            f"[{e - w + 1}, {e}]"
            # the check has teeth: the same window moved by one millisecond looks different
            shifted = _lnes(*_window_events(h, e - 1, w), w, flip, swap)
            n_checked += 1
            n_shift_differs += int(not torch.equal(lnes, shifted))
        for k in range(2):                            # the event ranges abut: no gap, no overlap
            assert h.offsets[ends[k] + 1] == h.offsets[ends[k + 1] - w + 1]
        # the same, read back from the returned labels alone: the target of window k is the pose
        # of ms e_k, found by value in the ground truth, and the targets are exactly w ms apart
        rows = [np.nonzero((h.pos51 == s[f"w{k}"][2].numpy()).all(axis=1))[0] for k in range(3)]
        prws = [np.nonzero((h.pos51 == s[f"w{k}"][1].numpy()).all(axis=1))[0] for k in range(3)]
        if all(len(r) == 1 for r in rows + prws):
            n_unique += 1
            t, p = [int(r[0]) for r in rows], [int(r[0]) for r in prws]
            assert t == list(ends)
            assert t[1] - t[0] == w and t[2] - t[1] == w
            assert [a - b + 1 for a, b in zip(t, p)] == [w, w, w]
    assert n_valid >= 25, n_valid
    assert n_unique >= 0.8 * n_valid, (n_unique, n_valid)
    assert n_shift_differs >= 0.8 * n_checked, (n_shift_differs, n_checked)


def test_the_three_windows_share_one_virtual_camera_and_one_label_transform(pool):
    """Domrand on its geometry only (the pixel map is the one random thing), prev noise off: the
    labels and the events of every window must follow the single realisation `dr`."""
    ds = pool(triplet=True, frac=1.0, domrand="geom", noise=False)
    n_valid = n_moved = 0
    for i in _probe(ds):
        rng, si, end, w, _, dr = _replay(ds, i)
        flip, swap = _draw_polarity(rng, ds)
        s = ds[i]
        if not s["valid"]:
            continue
        n_valid += 1
        n_moved += int(not dr.geometric_identity)
        h = ds._handle(si)
        for k, e in enumerate((end - 2 * w, end - w, end)):
            lnes, prev, target, _, K = s[f"w{k}"]
            gt = np.stack([h.pos51[e - w + 1], h.pos51[e]])
            lab, k_dr = DR.transform_labels(gt, dr, h.camera_K, h.j0, ds.render_scale)
            assert torch.equal(prev, torch.from_numpy(lab[0])), (i, k)
            assert torch.equal(target, torch.from_numpy(lab[1])), (i, k)
            assert torch.equal(K, torch.from_numpy(k_dr)), (i, k)
            xs, ys, ps, ms = _window_events(h, e, w)
            keep = DR.transform_events(xs, ys, dr, h.camera_K, ds.render_scale, W, H)
            want = _lnes(xs[keep], ys[keep], ps[keep], ms[keep], w, flip, swap)
            assert torch.equal(lnes, want), f"window {k} of sample {i}: events do not follow dr"
        assert torch.equal(s["w0"][4], s["w2"][4]) and torch.equal(s["w1"][4], s["w2"][4])
        if not dr.geometric_identity:
            assert not torch.equal(s["w2"][4], torch.from_numpy(h.camera_K)), "K was not moved"
    assert n_valid >= 25 and n_moved >= 0.9 * n_valid, (n_valid, n_moved)


def test_dropout_hot_pixels_and_noise_are_drawn_per_window(pool):
    """With the shipped domrand and noise: each window draws its own keep mask, hot pixels and prev
    noise, all with the realisation's rates (one `keep`, one hot-pixel count) from the one stream;
    an invalid sample draws nothing extra."""
    ds = pool(triplet=True, frac=1.0, domrand="full", noise=True)
    keep_calls, hot_calls, noise_calls = [], [], []
    real_keep, real_hot, real_noise = DR.keep_mask, DR.sample_hot_pixels, ds._sample_prev_noise

    def keep(n, k, rng):
        keep_calls.append((n, k, id(rng)))
        return real_keep(n, k, rng)

    def hot(n, win, width, height, rng):
        out = real_hot(n, win, width, height, rng)
        hot_calls.append((n, win, id(rng), out))
        return out

    def noise(rng):
        out = real_noise(rng)
        noise_calls.append(out)
        return out

    n_valid = 0
    with mock.patch.object(DR, "keep_mask", keep), mock.patch.object(DR, "sample_hot_pixels", hot), \
            mock.patch.object(ds, "_sample_prev_noise", noise):
        for i in range(DEEP, DEEP + 12):
            keep_calls.clear(), hot_calls.clear(), noise_calls.clear()
            s = ds[i]
            assert bool(s["valid"])
            n_valid += 1
            assert len(keep_calls) == len(hot_calls) == len(noise_calls) == 3
            assert len({c[1] for c in keep_calls}) == 1, "one keep rate (dr.keep) for all windows"
            assert len({c[0] for c in hot_calls}) == 1, "one hot-pixel count (dr.n_hot)"
            assert len({c[2] for c in keep_calls + hot_calls}) == 1, "one stream"
            if hot_calls[0][0] > 0:
                hs = [c[3] for c in hot_calls]
                assert not np.array_equal(hs[0], hs[1]) and not np.array_equal(hs[1], hs[2])
            ns = noise_calls
            assert not np.array_equal(ns[0], ns[1]) and not np.array_equal(ns[1], ns[2])
        # a sample near a run start cannot be a triplet: a single window's worth of draws
        short = next(j for j in _first_of_runs(ds) if ds.index[j, 1] - ds.index[j, 2] + 1 < 90)
        keep_calls.clear(), hot_calls.clear(), noise_calls.clear()
        s = ds[int(short)]
        assert not s["valid"]
        assert len(hot_calls) == 1 and len(noise_calls) == 1 and len(keep_calls) <= 1
    assert n_valid == 12


def test_prev_noise_differs_between_windows_but_has_one_distribution(pool):
    stats = pytest.importorskip("scipy.stats")
    ds = pool(triplet=True, frac=1.0, domrand="full", noise=True)
    parts = {"t": slice(0, 3), "r": slice(3, 6), "pose": slice(6, 51)}
    n = 240
    cols = {k: [[], [], []] for k in parts}
    for i in range(DEEP, DEEP + n):
        _, si, end, w, _, dr = _replay(ds, i)
        s = ds[i]
        assert bool(s["valid"])
        h = ds._handle(si)
        got = []
        for k, e in enumerate((end - 2 * w, end - w, end)):
            gt = np.stack([h.pos51[e - w + 1], h.pos51[e]])
            clean = DR.transform_labels(gt, dr, h.camera_K, h.j0, ds.render_scale)[0][0]
            got.append(s[f"w{k}"][1].numpy() - clean)         # the noise, up to float32 rounding
        assert not np.allclose(got[0], got[1], atol=1e-6) and not np.allclose(got[1], got[2], atol=1e-6)
        assert not np.allclose(got[0], got[2], atol=1e-6)
        for k, sl in parts.items():
            for j in range(3):
                cols[k][j].append(float(np.linalg.norm(got[j][sl])))
    for k in parts:
        a = np.asarray(cols[k])                                # (3 windows, n samples)
        assert a.min() >= 0 and a[2].mean() > 0
        # one distribution: neither earlier window differs from the main window's
        for j in (0, 1):
            p = stats.ks_2samp(a[j], a[2]).pvalue
            assert p > 1e-3, f"prev noise ({k}) of w{j} and w2 are not one distribution: p={p:.2e}"
        # and the draws are independent between windows (shared draws would correlate ~ 1)
        for j in range(3):
            for l in range(j + 1, 3):
                rho = stats.spearmanr(a[j], a[l]).correlation
                assert abs(rho) < 0.25, f"noise norms ({k}) of w{j}, w{l} correlate: {rho:.2f}"


# ---------------------------------------------------------------------- (c) validity of samples
@pytest.mark.parametrize("frac", [1.0 / 3.0, 0.75])
def test_valid_fraction_follows_triplet_frac(pool, frac):
    """400 consecutive samples deep inside a long run (history never limits): a binomial count."""
    ds = pool(triplet=True, frac=frac)
    n = 400
    k = sum(bool(ds[i]["valid"]) for i in range(DEEP, DEEP + n))
    sd = (n * frac * (1.0 - frac)) ** 0.5
    assert abs(k - n * frac) <= 4.0 * sd, f"{k}/{n} valid for frac {frac:.3f} (sd {sd:.1f})"


def test_frac_one_makes_every_sample_with_enough_history_valid(pool):
    ds = pool(triplet=True, frac=1.0)
    assert all(bool(ds[i]["valid"]) for i in range(DEEP, DEEP + 60))


def test_samples_near_a_run_start_are_invalid_when_history_is_short(pool):
    """valid <=> end - run_start + 1 >= 3 w. Short runs (< 3 * WINDOW_MIN = 90 ms of history) can
    never give a triplet; a long run's first samples only do for short windows."""
    ds = pool(triplet=True, frac=1.0, domrand="off", noise=False)
    n_short = n_hist_invalid = n_hist_valid = 0
    for j in _first_of_runs(ds)[::2]:
        for off in (0, 1, 45, 100, 250, 600):
            i = int(j) + off
            if i >= len(ds) or ds.index[i, 2] != ds.index[j, 2] or ds.index[i, 0] != ds.index[j, 0]:
                continue
            _, _, end, w, run_a, _ = _replay(ds, i)
            hist = end - run_a + 1
            s = ds[i]
            assert bool(s["valid"]) == (hist >= 3 * w), (i, hist, w)
            if hist < 3 * ds.min_w:
                n_short += 1
                assert not s["valid"] and _aliased(s["w0"], s["w2"]) and _aliased(s["w1"], s["w2"])
            n_hist_invalid += int(hist < 3 * w)
            n_hist_valid += int(hist >= 3 * w)
    assert n_short >= 10 and n_hist_invalid >= 20 and n_hist_valid >= 20, \
        (n_short, n_hist_invalid, n_hist_valid)


def test_invalid_samples_alias_w2_and_compute_nothing_extra(pool):
    ds = pool(triplet=True, frac=0.5)
    n_valid = n_invalid = 0
    with mock.patch.object(ds, "_triplet_window", wraps=ds._triplet_window) as tw:
        for i in range(DEEP, DEEP + 40):
            tw.reset_mock()
            s = ds[i]
            if s["valid"]:
                n_valid += 1
                assert tw.call_count == 2
                assert not any(a is b for a, b in zip(s["w0"], s["w2"]))
                assert not any(a is b for a, b in zip(s["w1"], s["w2"]))
                assert not torch.equal(s["w0"][2], s["w2"][2]) and not torch.equal(s["w1"][2], s["w2"][2])
            else:
                n_invalid += 1
                assert tw.call_count == 0
                assert _aliased(s["w0"], s["w2"]) and _aliased(s["w1"], s["w2"])
                assert s["valid"].dtype == torch.bool and s["valid"].ndim == 0
    assert n_valid >= 8 and n_invalid >= 8, (n_valid, n_invalid)


# ---------------------------------------------------------------------------- (d) the DataLoader
def test_default_collate_through_a_real_dataloader(pool):
    ds = pool(triplet=True, frac=0.5)
    idx = list(range(DEEP + 1000, DEEP + 1024))

    def loader(shuffle=False, seed=None):
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        return DataLoader(Subset(ds, idx), batch_size=8, shuffle=shuffle, num_workers=2, generator=g)

    batches = list(loader())
    assert len(batches) == 3
    n_valid = 0
    for bi, b in enumerate(batches):
        assert set(b) == {"w0", "w1", "w2", "valid"}
        assert b["valid"].dtype == torch.bool and tuple(b["valid"].shape) == (8,)
        for key in ("w0", "w1", "w2"):
            assert isinstance(b[key], list) and len(b[key]) == 5
            assert [tuple(t.shape) for t in b[key]] == [(8, H, W, 2), (8, 51), (8, 51), (8, 10), (8, 3, 3)]
            assert all(t.dtype == torch.float32 for t in b[key])
        # what the workers collated is what default_collate makes of the samples read in-process
        _assert_batches_equal(b, default_collate([ds[i] for i in idx[bi * 8:(bi + 1) * 8]]))
        for j in range(8):                      # invalid rows repeat w2 in w0 and w1, valid ones do not
            same = all(torch.equal(b["w0"][t][j], b["w2"][t][j]) and torch.equal(b["w1"][t][j], b["w2"][t][j])
                       for t in range(5))
            assert same == (not bool(b["valid"][j]))
        n_valid += int(b["valid"].sum())
    assert 0 < n_valid < 24, n_valid
    # deterministic for a fixed seed (shuffled order included), and the seed matters
    a, b, c = list(loader(True, 11)), list(loader(True, 11)), list(loader(True, 12))
    for x, y in zip(a, b):
        _assert_batches_equal(x, y)
    assert any(not torch.equal(x["w2"][2], z["w2"][2]) for x, z in zip(a, c))


# --------------------------------------------------------------------------------- (e) CPU cost
def test_cpu_cost_per_sample_against_the_plain_dataset(full_off, full_on):
    """Mean CPU seconds per sample over 64 samples spread over the training index, triplet on at
    frac 1/3 against off, same indices. A valid sample costs ~3 plain ones (it builds 3 windows),
    an invalid one ~1; so the mean lies near 1 + 2 * (valid fraction)."""
    idx = [int(i) for i in np.linspace(2000, len(full_off) - 2000, 64)]

    def cpu(ds, i):
        t0 = time.process_time()
        ds[i]
        return time.process_time() - t0

    for ds in (full_off, full_on):                   # warm-up: opens the memmaps, fills the cache
        for i in idx:
            ds[i]
    rounds = [([cpu(full_off, i) for i in idx], [cpu(full_on, i) for i in idx]) for _ in range(3)]
    off = np.median([r[0] for r in rounds], axis=0)  # per sample: median over the 3 rounds
    on = np.median([r[1] for r in rounds], axis=0)
    valid = np.array([bool(full_on[i]["valid"]) for i in idx])
    res = {
        "n_samples": len(idx), "n_valid": int(valid.sum()), "frac": full_on.triplet_frac,
        "mean_cpu_s_off": float(off.mean()), "mean_cpu_s_on": float(on.mean()),
        "ratio_on_over_off": float(on.mean() / off.mean()),
        "ratio_valid_samples": float(on[valid].sum() / off[valid].sum()),
        "ratio_invalid_samples": float(on[~valid].sum() / off[~valid].sum()),
        "cpu_s_per_sample_off_rounds": [float(np.mean(r[0])) for r in rounds],
        "cpu_s_per_sample_on_rounds": [float(np.mean(r[1])) for r in rounds],
    }
    print("\nDT-TRIPLET-TIMING " + json.dumps(res))
    if os.environ.get("DT_TRIPLET_JSON"):
        Path(os.environ["DT_TRIPLET_JSON"]).write_text(json.dumps(res, indent=1))
    assert 3 <= res["n_valid"] <= 61
    # generous bounds (a shared, busy host): valid ~3x, invalid ~1x, the mean in between
    assert 1.8 < res["ratio_valid_samples"] < 4.5, res
    assert 0.6 < res["ratio_invalid_samples"] < 1.5, res
    assert 1.0 < res["ratio_on_over_off"] < 2.4, res
