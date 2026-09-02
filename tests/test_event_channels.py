"""The wider event image must add information without moving the baseline.

LNES writes one timestamp per (pixel, polarity) and lets later events overwrite earlier ones. At the
50 ms operating point that discards 73.8% of the events into a surface which is then 97.3% empty
(`outputs/semkine/lnes_capacity.json`). `count` and `first` are the cheapest way to stop discarding.

Two properties have to hold, and neither is safe to assume:

* with `EVENT_CHANNELS` unset or `("last",)`, every byte the old path produced is reproduced, so no
  existing arm moves and the comparison against them stays valid;
* the added planes really do carry what overwriting deleted -- multiplicity and the interval the
  slot was active in -- rather than being a rearrangement of the same one number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from semkine import events as EV                                  # noqa: E402

H, W, WINDOW = 12, 16, 50


def _legacy_splat(xs, ys, ps, ms_rel, window, height, width):
    """The implementation as it stood before the shared builder, copied verbatim."""
    img = np.zeros((height, width, 2), np.float32)
    if len(xs):
        img[ys.astype(np.intp), xs.astype(np.intp), ps.astype(np.intp)] = (
            ms_rel.astype(np.float32) / float(window))
    return img


def _stream(n=4000, seed=0):
    r = np.random.default_rng(seed)
    ms = np.sort(r.integers(0, WINDOW, n))
    return (r.integers(0, W, n), r.integers(0, H, n), r.integers(0, 2, n), ms)


def test_last_only_is_bitwise_identical_to_the_old_splat():
    xs, ys, ps, ms = _stream()
    new = EV.splat_event_image(xs, ys, ps, ms, WINDOW, H, W, ("last",))
    old = _legacy_splat(xs, ys, ps, ms, WINDOW, H, W)
    assert new.shape == old.shape
    assert np.array_equal(new, old), "the default path is not the legacy LNES"
    # And it stays first when other planes are added, so a wider model's first two channels see
    # exactly what a narrow one saw.
    wide = EV.splat_event_image(xs, ys, ps, ms, WINDOW, H, W, ("last", "count", "first"))
    assert np.array_equal(wide[:, :, :2], old)
    assert wide.shape == (H, W, 6)


def test_empty_and_single_event_windows():
    e = np.zeros(0, np.int64)
    assert EV.splat_event_image(e, e, e, e, WINDOW, H, W, ("last", "count", "first")).sum() == 0
    one = np.array([3]), np.array([4]), np.array([1]), np.array([10])
    img = EV.splat_event_image(*one, WINDOW, H, W, ("last", "count", "first"))
    # One event: first == last, and the count plane holds log1p(1)/log1p(COUNT_REF).
    assert img[4, 3, 1] == pytest.approx(10 / WINDOW)
    assert img[4, 3, 5] == pytest.approx(10 / WINDOW)
    assert img[4, 3, 3] == pytest.approx(np.log1p(1) / np.log1p(EV.COUNT_REF))


def test_the_new_planes_carry_what_overwriting_deleted():
    """Three events in one slot: LNES keeps the last, the wide image recovers all three."""
    xs, ys, ps = np.array([5, 5, 5]), np.array([6, 6, 6]), np.array([0, 0, 0])
    ms = np.array([4, 20, 44])
    img = EV.splat_event_image(xs, ys, ps, ms, WINDOW, H, W, ("last", "count", "first"))
    assert img[6, 5, 0] == pytest.approx(44 / WINDOW)                      # last
    assert img[6, 5, 4] == pytest.approx(4 / WINDOW)                       # first
    assert img[6, 5, 2] == pytest.approx(np.log1p(3) / np.log1p(EV.COUNT_REF))
    # The dwell time -- the cue no single timestamp can express.
    assert img[6, 5, 0] - img[6, 5, 4] == pytest.approx(40 / WINDOW)

    # On a realistic stream the added planes must not be a function of `last`: if they were, the
    # extra channels would be capacity without information.
    xs, ys, ps, ms = _stream(seed=3)
    w = EV.splat_event_image(xs, ys, ps, ms, WINDOW, H, W, ("last", "count", "first"))
    occ = w[:, :, 2] > 0
    for k, name in ((2, "count"), (4, "first")):
        c = np.corrcoef(w[:, :, 0][occ], w[:, :, k][occ])[0, 1]
        assert abs(c) < 0.97, f"the {name} plane is nearly a copy of last (r={c:.3f})"


def test_first_matches_the_unique_reference():
    """The fast reversed splat against the slow definition, on realistic event densities.

    `first` is taken as last-write-wins over a reversed stream, which is 32x cheaper than
    `np.unique(..., return_index=True)` (0.11 ms against 3.4 ms per 23k-event window, i.e. the
    whole data-loading budget at batch 1024) but relies on duplicate-write order. That order is
    reliable only when the reversal is materialised, not a negative-stride view, so the two forms
    are compared here rather than assumed equal.
    """
    for seed in range(6):
        r = np.random.default_rng(seed)
        n = 4000
        ms = np.sort(r.integers(0, WINDOW, n))
        xs, ys, ps = r.integers(0, W, n), r.integers(0, H, n), r.integers(0, 2, n)
        got = EV.splat_event_image(xs, ys, ps, ms, WINDOW, H, W, ("last", "first"))[:, :, 2:]

        idx = (ys.astype(np.intp) * W + xs.astype(np.intp)) * 2 + ps.astype(np.intp)
        want = np.zeros((H, W, 2), np.float32)
        uniq, pos = np.unique(idx, return_index=True)
        want.reshape(-1)[uniq] = (ms.astype(np.float32) / WINDOW)[pos]
        assert np.array_equal(got, want), f"seed {seed}: fast `first` differs from the reference"


def test_config_validation_rejects_silent_mistakes():
    assert EV.event_channels({}) == ("last",)
    assert EV.event_channels({"DATA": {}}) == ("last",)
    for bad in (["count", "last"], ["last", "typo"], [], ["last", "last"]):
        with pytest.raises(ValueError, match="EVENT_CHANNELS"):
            EV.event_channels({"DATA": {"EVENT_CHANNELS": bad}})


def test_model_input_width_follows_the_config():
    from model import MNISTModel                                   # noqa: E402
    mano = REPO / "assets/mano_right.npz"
    if not mano.exists():
        pytest.skip("mano_right.npz not available")

    def cfg(ch=None):
        c = {"MODEL": {"POSE_REPR": "mano_full_axis_angle", "OUTPUT_DIM": 51,
                       "PREDICT_DELTA": True, "PREVPOS_EMBED": True},
             "DATA": {"HEIGHT": H, "WIDTH": W}, "MANO": {"NPZ": str(mano)},
             "LOSS": {"LAMBDA_POSE": 1.0, "LAMBDA_T": 1.0, "LAMBDA_R": 1.0,
                      "NORMALIZER": 51, "LOG10": False},
             "TRAIN": {"LR": 1e-3, "WARMUP_STEPS": 0}}
        if ch:
            c["DATA"]["EVENT_CHANNELS"] = ch
        return c

    assert MNISTModel(cfg()).conv1.in_channels == 2
    assert MNISTModel(cfg(["last", "count", "first"])).conv1.in_channels == 6
