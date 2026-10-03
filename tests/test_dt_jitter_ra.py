"""Root-aligned jitter keys of evalx.jitter_decomp (DT round, main-table convention): joint 0 subtracted from all joints."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "tracking")]
import evalx as EX                                                    # noqa: E402
from mano_layer import ManoLayer                                      # noqa: E402

DEV = torch.device("cpu")
N = 240


@pytest.fixture(scope="module")
def setup():
    mano = ManoLayer(str(REPO / "assets/mano_right.npz"), add_mean=False).to(DEV).eval()
    t = np.arange(N, dtype=np.float32)
    gt = np.zeros((N, 51), np.float32)
    gt[:, 2] = 0.45
    gt[:, 0] = 0.002 * t                                                   # constant-velocity translation
    gt[:, 3:6] = np.array([1.5, 1.0, 1.3]) + 0.01 * t[:, None] * np.array([1.0, 0.0, 0.0])
    gt[:, 6:] = 0.05 * np.sin(0.05 * t)[:, None]
    run = np.concatenate([np.zeros(120, int), np.ones(120, int)])
    return mano, gt, run, np.zeros(10, np.float32), np.random.default_rng(0)


def jd(mano, pred, gt, run, betas):
    return EX.jitter_decomp(mano, {"pred": pred.astype(np.float32), "gt": gt, "run": run, "betas": betas}, DEV)


def test_perfect_tracking_is_zero_and_ratio_one(setup):
    mano, gt, run, betas, _ = setup
    j = jd(mano, gt.copy(), gt, run, betas)
    assert abs(j["acc_err_ra_mm"]) < 1e-4
    assert abs(j["jit_ratio_ra"] - 1.0) < 1e-6 and abs(j["acc_ratio_ra"] - 1.0) < 1e-6
    assert j["acc_err_ra_only_root_mm"] < 1e-4 and j["acc_err_ra_only_fingers_mm"] < 1e-4


def test_translation_noise_does_not_touch_root_aligned_joints(setup):
    mano, gt, run, betas, rng = setup
    p = gt.copy()
    p[:, :3] += rng.normal(0, 0.003, (N, 3))
    j = jd(mano, p, gt, run, betas)
    assert j["acc_err_mm"] > 5.0                                             # absolute joints feel it
    assert j["acc_err_ra_mm"] < 1e-3                                         # root-aligned joints do not
    assert j["acc_err_ra_only_transl_mm"] < 1e-3
    assert abs(j["jit_ra_pred_mm"] - j["jit_ra_gt_mm"]) < 1e-3


def test_finger_noise_is_identical_in_both_conventions(setup):
    mano, gt, run, betas, rng = setup
    p = gt.copy()
    p[:, 6:] += rng.normal(0, 0.05, (N, 45))
    j = jd(mano, p, gt, run, betas)
    assert j["acc_err_ra_mm"] > 1.0
    assert abs(j["acc_err_ra_mm"] - j["acc_err_mm"]) < 1e-5                  # the wrist is the same in pred and GT
    assert j["acc_err_ra_only_fingers_mm"] > 0.95 * j["acc_err_ra_mm"]
    assert j["acc_err_ra_only_root_mm"] < 1e-4


def test_root_rotation_noise_shows_in_both(setup):
    mano, gt, run, betas, rng = setup
    p = gt.copy()
    p[:, 3:6] += rng.normal(0, 0.05, (N, 3))
    j = jd(mano, p, gt, run, betas)
    assert j["acc_err_ra_mm"] > 1.0 and j["acc_err_mm"] > 1.0
    assert j["acc_err_ra_only_root_mm"] > 0.9 * j["acc_err_ra_mm"]


def test_gt_with_constant_translation_has_equal_conventions(setup):
    mano, gt, run, betas, _ = setup
    g2 = gt.copy()
    g2[:, :3] = np.array([0.0, 0.0, 0.45], np.float32)                      # no translation motion
    j = jd(mano, g2.copy(), g2, run, betas)
    assert abs(j["jit_ra_gt_mm"] - j["jit_gt_mm"]) < 0.05 * max(j["jit_gt_mm"], 1e-9) + 1e-6
    assert abs(j["acc_ra_gt_mm"] - j["acc_gt_mm"]) < 0.05 * max(j["acc_gt_mm"], 1e-9) + 1e-6


def test_segment_boundary_is_excluded(setup):
    mano, gt, run, betas, _ = setup
    p = gt.copy()
    p[120:, 6:] += 0.3                                                      # a jump exactly at the boundary
    j = jd(mano, p, gt, run, betas)
    # inside each segment the error is a smooth function of the pose (a constant angle offset through the nonlinear FK),
    # so its second difference is tiny; the same arrays without the boundary see the jump and are large
    assert j["acc_err_ra_mm"] < 0.05
    j_no_boundary = jd(mano, p, gt, np.zeros(N, int), betas)
    assert j_no_boundary["acc_err_ra_mm"] > 20 * max(j["acc_err_ra_mm"], 1e-3)
