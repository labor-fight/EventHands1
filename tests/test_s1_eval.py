#!/usr/bin/env python3
"""S1 gates for the extended evaluator: legacy parity, bucket bookkeeping, statistics.

The parity test is what licenses every later comparison: with buckets and delta-trust off, the
extended evaluator must reproduce `model/eval_track.py` to within 1e-6, so the protocol reform
cannot be mistaken for a result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from config import load_config              # noqa: E402
from mano_layer import ManoLayer            # noqa: E402
from model import MNISTModel               # noqa: E402
import eval_track as LEGACY                 # noqa: E402

from semkine import buckets as BK           # noqa: E402
from semkine import eval_track as NEW       # noqa: E402
from semkine import metrics as MT           # noqa: E402

CFG = ROOT / "configs" / "eventhands_track_render51.yaml"
CKPT = (ROOT / "outputs/hand_data51/track_render51"
        / "track_render51-step=1000-val_loss=val_loss=0.8015.ckpt")
PROBE, PROBE_DIR = "zgz_local", "val"

pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="frozen checkpoint unavailable")


@pytest.fixture(scope="module")
def setup():
    cfg = load_config(CFG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel.load_from_checkpoint(str(CKPT), cfg=cfg, map_location=device)
    model = model.to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    return cfg, model, mano, device, Path(cfg["DATA"]["ROOT"])


def test_lnes_builder_matches_legacy(setup):
    cfg, _, _, _, root = setup
    ev, off, _, _ = NEW.load_sequence(root, PROBE_DIR, PROBE)
    ev2, off2, _, _ = LEGACY.load_sequence(root, PROBE_DIR, PROBE)
    for end in (200, 1000, 5000, 20000):
        a = NEW.build_lnes(ev, off, end, 50)
        b = LEGACY.build_lnes(ev2, off2, end, 50)
        assert np.array_equal(a, b), f"LNES differs at end={end}"


def test_init_noise_matches_legacy(setup):
    cfg, _, _, _, _ = setup
    for scale in (0.0, 1.0, 2.0):
        a = NEW.sample_init_noise(cfg, np.random.default_rng(3), scale)
        b = LEGACY.sample_init_noise(cfg, np.random.default_rng(3), scale)
        assert np.array_equal(a, b)


def test_extended_evaluator_reproduces_legacy(setup):
    """The S1 hard gate: identical metrics with buckets and delta-trust off."""
    cfg, model, mano, device, root = setup
    a = NEW.track_sequence(model, mano, cfg, root, PROBE_DIR, PROBE, 50, device,
                           np.random.default_rng(0), 1.0)
    b = LEGACY.track_sequence(model, mano, cfg, root, PROBE_DIR, PROBE, 50, device,
                              np.random.default_rng(0), 1.0)
    assert a["n_frames"] == b["n_frames"] and a["n_runs"] == b["n_runs"]
    for k in ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm",
              "jitter_all_mm_per_step", "gt_move_mm_per_step"):
        assert abs(a[k] - b[k]) <= 1e-6, f"{k}: {a[k]} vs {b[k]}"
    assert set(a["drift"]) == set(b["drift"])
    for kk in a["drift"]:
        assert abs(a["drift"][kk]["mpjpe_ra_mm"] - b["drift"][kk]["mpjpe_ra_mm"]) <= 1e-6
        assert a["drift"][kk]["n"] == b["drift"][kk]["n"]


def test_buckets_do_not_change_the_overall_metric(setup):
    cfg, model, mano, device, root = setup
    bm_path = root / "buckets" / "test_step50.json"
    if not bm_path.exists():
        pytest.skip("bucket manifest not built")
    bm = BK.load_bucket_manifest(bm_path)
    off = NEW.track_sequence(model, mano, cfg, root, PROBE_DIR, PROBE, 50, device,
                             np.random.default_rng(0), 1.0)
    on = NEW.track_sequence(model, mano, cfg, root, PROBE_DIR, PROBE, 50, device,
                            np.random.default_rng(0), 1.0, bucket_manifest=bm)
    assert abs(on["mpjpe_ra_mm"] - off["mpjpe_ra_mm"]) <= 1e-6
    assert "buckets" in on and on["buckets"], "bucket manifest did not attach"
    # Every bucket average must lie inside the observed per-step range.
    for name, e in on["buckets"].items():
        if e["n"]:
            assert 0.0 < e["mpjpe_ra_mm"] < 500.0, (name, e)


def test_bucket_manifest_step_count_is_checked(setup):
    cfg, _, _, _, root = setup
    bm_path = root / "buckets" / "test_step50.json"
    if not bm_path.exists():
        pytest.skip("bucket manifest not built")
    bm = BK.load_bucket_manifest(bm_path)
    with pytest.raises(ValueError, match="bucket manifest has"):
        BK.masks_for_sequence(bm, PROBE, 12345)


def test_delta_trust_identity_and_effect(setup):
    cfg, model, mano, device, root = setup
    base = NEW.track_sequence(model, mano, cfg, root, PROBE_DIR, PROBE, 50, device,
                              np.random.default_rng(0), 1.0, delta_trust=1.0)
    same = NEW.track_sequence(model, mano, cfg, root, PROBE_DIR, PROBE, 50, device,
                              np.random.default_rng(0), 1.0)
    assert abs(base["mpjpe_ra_mm"] - same["mpjpe_ra_mm"]) <= 1e-9, "rho=1 is not the identity"
    half = NEW.track_sequence(model, mano, cfg, root, PROBE_DIR, PROBE, 50, device,
                              np.random.default_rng(0), 1.0, delta_trust=0.5)
    assert abs(half["mpjpe_ra_mm"] - base["mpjpe_ra_mm"]) > 1e-3, "rho=0.5 had no effect"


def test_zero_trust_freezes_the_state(setup):
    """rho = 0 must hold the initial state exactly; a leak would show up as motion."""
    cfg, model, mano, device, root = setup
    r = NEW.track_sequence(model, mano, cfg, root, PROBE_DIR, PROBE, 50, device,
                           np.random.default_rng(0), 0.0, delta_trust=0.0)
    assert r["jitter_all_mm_per_step"] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------ statistics
def test_paired_bootstrap_detects_a_shift():
    keys = [f"s{i}" for i in range(16)]
    rng = np.random.default_rng(0)
    base = {k: 20.0 + rng.normal(0, 4.0) for k in keys}
    w = {k: 1000.0 for k in keys}
    shifted = {k: v - 1.0 for k, v in base.items()}
    iv = MT.paired_sequence_bootstrap(shifted, base, w, n_boot=4000)
    assert iv.excludes_zero and iv.mean == pytest.approx(-1.0, abs=1e-6)
    # An unpaired comparison of the same data cannot see a 1 mm shift under 4 mm spread.
    a = MT.sequence_bootstrap(list(shifted.values()), list(w.values()), n_boot=4000)
    b = MT.sequence_bootstrap(list(base.values()), list(w.values()), n_boot=4000)
    assert a.lo < b.mean < a.hi, "unpaired CIs should still overlap; pairing is what buys power"


def test_paired_bootstrap_on_noise_does_not_claim():
    keys = [f"s{i}" for i in range(16)]
    rng = np.random.default_rng(1)
    a = {k: 20.0 + rng.normal(0, 3.0) for k in keys}
    b = {k: a[k] + rng.normal(0, 0.4) for k in keys}   # replicate-level noise only
    w = {k: 1000.0 for k in keys}
    iv = MT.paired_sequence_bootstrap(a, b, w, n_boot=4000, seed=2)
    assert not iv.excludes_zero, f"claimed an effect on replicate noise: {iv}"


def test_single_sequence_is_reported_as_underpowered():
    iv = MT.sequence_bootstrap([19.2], [2590])
    assert not np.isfinite(iv.lo) and not np.isfinite(iv.hi)
    g = MT.decide_gate("x", iv, -0.5, "lower_is_better")
    assert g.verdict == "UNDERPOWERED"


def test_gate_requires_the_whole_interval():
    good = MT.Interval(mean=-1.0, lo=-1.6, hi=-0.6, n=20000)
    assert MT.decide_gate("g", good, 0.0, "lower_is_better").verdict == "PASS"
    straddle = MT.Interval(mean=-1.0, lo=-2.2, hi=0.3, n=20000)
    assert MT.decide_gate("g", straddle, 0.0, "lower_is_better").verdict == "FAIL"


def test_pooled_stats_flag_the_minimum():
    s = MT.pooled_replicate_stats([19.26, 20.4, 21.1, 21.69])
    assert s["mean"] == pytest.approx(20.6125)
    assert s["min_diagnostic_only"] == 19.26
    assert s["range"] == pytest.approx(2.43)
    assert s["mean"] > s["min_diagnostic_only"], "best-of-N must not be the reported centre"


def test_underpowered_bucket_cannot_pass():
    iv = MT.Interval(mean=-3.0, lo=-4.0, hi=-2.0, n=99)
    g = MT.decide_gate("b", iv, 0.0, "lower_is_better", regime="bucket", min_support=200)
    assert g.verdict == "UNDERPOWERED" and "support" in g.note
