"""DATA.PREV_LAG_MS (DT2 deviation 7): the conditioning state is GT `lag` ms before the window end instead of the window's
first ms. Absent: the sample is the recorded one (covered by tests/test_dt_pipe_fast.py and test_s1_dataset.py); present:
prev == pos51[max(end - lag + 1, run_start)] on both data paths, the events / target are unchanged, misuse raises.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python -m pytest tests/test_dt_prevlag.py -q
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "model")]
from config import load_config                                        # noqa: E402
from semkine.dataset import build_dataset                             # noqa: E402

CFG = REPO / "configs/dt/dt_dz_l3.yaml"
needs_data = pytest.mark.skipif(not CFG.exists() or not Path("/data1/lyq/code/mesh/EventHands/data/hand_data51").exists(),
                                reason="dt_dz_l3 config / data not found")


def cfg_plain(lag, window=200, triplet=False):
    cfg = load_config(CFG)
    cfg["AUG"]["DOMRAND"]["ENABLED"] = False
    cfg["AUG"]["SPEED_AUG"] = False
    cfg["DATA"]["WINDOW_MIN"] = cfg["DATA"]["WINDOW_MAX"] = window
    for k in ("PREV_NOISE_T", "PREV_NOISE_R", "PREV_NOISE_POSE"):
        cfg["TRACK"][k] = 0.0
    if lag is not None:
        cfg["DATA"]["PREV_LAG_MS"] = lag
    if triplet:
        cfg["DATA"]["TRIPLET"] = True
    return cfg


@needs_data
@pytest.mark.parametrize("fast", [False, True])
def test_prev_is_lagged_ground_truth_and_events_unchanged(fast):
    comps = np.load(load_config(CFG)["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    ds0 = build_dataset(cfg_plain(None), "train", comps, train=True)
    ds1 = build_dataset(cfg_plain(50), "train", comps, train=True)
    ds0.set_fast_pipe(fast); ds1.set_fast_pipe(fast)
    assert ds0.prev_lag_ms is None and ds1.prev_lag_ms == 50
    rng = np.random.default_rng(0)
    idxs = list(rng.integers(0, len(ds1), 12))
    # the index starts every run at a + WINDOW_MAX - 1, so the clip to the run start is exercised on the helper itself
    assert ds1._prev_index(1000 - 199, 1000, 980) == 980 and ds1._prev_index(1000 - 199, 1000, 900) == 951
    assert ds0._prev_index(1000 - 199, 1000, 900) == 1000 - 199
    for idx in idxs:
        si, end, run_a, _ = (int(v) for v in ds1.index[idx])
        h = ds1._handle(si)
        x0, p0, t0, b0, k0 = ds0[idx]
        x1, p1, t1, b1, k1 = ds1[idx]
        assert torch.equal(x0, x1) and torch.equal(t0, t1) and torch.equal(b0, b1) and torch.equal(k0, k1)
        want = torch.from_numpy(np.asarray(ds1._to_target(h.pos51[max(end - 49, run_a)]), np.float32))
        assert torch.equal(p1, want), idx
        window = min(200, end - run_a + 1)
        want0 = torch.from_numpy(np.asarray(ds0._to_target(h.pos51[end - window + 1]), np.float32))
        assert torch.equal(p0, want0), idx
        if end - run_a + 1 >= 200:
            assert not torch.equal(p0, p1)            # 200 ms window: the two conditioning states differ


@needs_data
def test_fast_and_legacy_paths_agree_with_lag():
    comps = np.load(load_config(CFG)["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    cfg = load_config(CFG)
    cfg["DATA"]["PREV_LAG_MS"] = 50
    ds = build_dataset(cfg, "train", comps, train=True)
    for idx in np.random.default_rng(1).integers(0, len(ds), 20):
        ds.set_fast_pipe(False); a = ds[int(idx)]
        ds.set_fast_pipe(True); b = ds[int(idx)]
        assert all(torch.equal(u, v) for u, v in zip(a, b)), idx


@needs_data
@pytest.mark.parametrize("bad", [0, -5, True, "50", 2.5])
def test_bad_values_raise(bad):
    comps = np.load(load_config(CFG)["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    with pytest.raises(ValueError):
        build_dataset(cfg_plain(bad), "train", comps, train=True)


@needs_data
def test_not_with_triplet():
    comps = np.load(load_config(CFG)["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    with pytest.raises(ValueError):
        build_dataset(cfg_plain(50, triplet=True), "train", comps, train=True)
