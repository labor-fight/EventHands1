"""S45-LOC (docs/EXPERIMENT_RECORDS.md#s45loc-prereg-20261029): the virtual-zoom depth transplant keeps every joint on
its pixel up to the dolly-zoom term, perturbations have their registered sizes, drop masks / generators are
deterministic, and the part vote and the token quantile map behave."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from tools.s45loc import common as C  # noqa: E402

NPZ = ROOT / "assets" / "mano_right.npz"
DATA = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51")


def test_transplant_keeps_pixels():
    from mano_layer import ManoLayer
    from semkine import eval_track as ET
    from semkine.dataset import mano_root_joint
    if not (DATA / "val/zgz_local_aux.npz").exists():
        pytest.skip("dataset not mounted")
    _, _, aux, gt = ET.load_sequence(DATA, "val", "zgz_local")
    mano = ManoLayer(str(NPZ), add_mean=False).eval()
    betas = np.asarray(aux["betas"], np.float32)
    beta = torch.tensor(betas).view(1, -1)
    j0_z = float(mano_root_joint(betas, str(NPZ))[2])
    K = np.asarray(aux["camera_K"], np.float32)
    ends = [e for a, b in np.asarray(aux["valid_runs_ms"]).reshape(-1, 2) for e in range(int(a) + 49, int(b), 50)]
    states = np.stack([np.asarray(gt[m], np.float32) for m in ends[::40]])
    #: the zgz_local transplant uses k ~ 0.80; the residual is the dolly-zoom perspective term, ~1 px at most
    for k in (0.8, 1.25):
        moved = C.transplant_state(states, k, j0_z)
        assert np.array_equal(moved[:, [0, 1]], states[:, [0, 1]]) and np.array_equal(moved[:, 3:], states[:, 3:])
        Kp = C.transplant_K(K, k)
        assert Kp[0, 2] == K[0, 2] and Kp[1, 2] == K[1, 2]
        j, _ = ET.mano_fk(mano, states, beta, torch.device("cpu"))
        jm, _ = ET.mano_fk(mano, moved, beta, torch.device("cpu"))
        p0 = j0_z + states[:, 2].astype(np.float64)
        assert np.allclose(jm[:, 0, 2] - j[:, 0, 2], (k - 1) * p0, atol=2e-5)

        def pix(J, KK):
            return 0.375 * np.stack([KK[0, 0] * J[..., 0] / J[..., 2] + KK[0, 2], KK[1, 1] * J[..., 1] / J[..., 2] + KK[1, 2]], -1)
        err = np.linalg.norm(pix(j, K) - pix(jm, Kp), axis=-1)
        # measured 2026-10-29 on zgz_local: median 0.42 px, max 2.3 px (fingertips far off the root depth); the
        # transplant is therefore only the secondary, geometry-consistent depth variant, the zn shift the primary one
        assert err.max() < 3.0 and np.median(err) < 0.5, (err.max(), np.median(err))


def test_perturbations_have_registered_sizes():
    from scipy.spatial.transform import Rotation as R
    base = np.zeros(51, np.float32)
    base[2] = 0.6
    base[3:6] = [0.3, -1.2, 0.4]
    for seed in range(20):
        rng = np.random.default_rng(seed)
        t, dt = C.perturb_state(base, "T10", rng)
        assert np.isclose(np.linalg.norm(t[0:3] - base[0:3]), 0.010, atol=1e-6) and np.array_equal(t[3:], base[3:])
        r, dr = C.perturb_state(base, "R5", rng)
        rel = R.from_rotvec(r[3:6]) * R.from_rotvec(base[3:6]).inv()
        assert np.isclose(rel.magnitude(), np.deg2rad(5), atol=1e-5) and np.allclose(rel.as_rotvec(), dr, atol=1e-5)
        f, df = C.perturb_state(base, "Fs", rng)
        assert np.array_equal(f[:6], base[:6]) and np.allclose(f[6:] - base[6:], df, atol=1e-6)


def test_generators_and_masks_are_deterministic():
    a = C.rng_for("zgz_local", 1234, 1).random(5)
    b = C.rng_for("zgz_local", 1234, 1).random(5)
    c = C.rng_for("zgz_local", 1234, 2).random(5)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    keep = C.random_drop_mask(100, 37, C.rng_for("x", 1, 2))
    assert keep.sum() == 63 and np.array_equal(keep, C.random_drop_mask(100, 37, C.rng_for("x", 1, 2)))
    m = C.keep_fraction_mask(100000, 0.16, C.rng_for("x", 1, 1))
    assert abs(m.mean() - 0.16) < 0.01


def test_weighted_part_vote():
    part = np.array([0, 0, 1, 2, 2, 2])
    vid = np.array([[0, 1, 2, 0], [2, 3, 4, 2], [5, 5, 5, 5]])
    w = np.array([[0.2, 0.2, 0.6, 0.0], [0.9, 0.05, 0.05, 0.0], [0.0, 0.0, 0.0, 0.0]])
    assert C.weighted_part(vid, w, part, n_parts=3).tolist() == [1, 1, -1]


def test_interp_is_monotone_quantile_map():
    from tools.s45loc.run import interp
    xp = torch.tensor([0.0, 1.0, 1.0, 2.0], dtype=torch.float64)
    fp = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float64)
    x = torch.tensor([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0], dtype=torch.float64)
    y = interp(x, xp, fp)
    assert y[0] == 10 and y[-1] == 40 and np.isclose(float(y[2]), 15.0) and np.isclose(float(y[4]), 35.0)
    assert bool((y[1:] >= y[:-1]).all())


def test_task_registry_covers_the_prereg():
    assert set(C.TASKS) == set(C.STAGE1) | set(C.STAGE2) | set(C.AMEND1)
    for task, (domain, mode, intervention) in C.TASKS.items():
        assert domain in ("zgz", "zgz_local", "train_local", "train_base") and mode in ("e3",) + C.E1_MODES
        assert intervention in C.INTERVENTIONS, task
        if domain == "zgz_local":
            assert mode == "e3", "a closed loop on zgz_local alone would change the cached init-noise stream"


def test_rescale_events_about_center():
    ev = np.array([[0, 100, 50, 0.01, 1], [0, 120, 50, 0.02, 0], [0, 239, 179, 0.03, 1]], np.float32)
    same, keep = C.rescale_events(ev, (100.0, 50.0), 1.0)
    assert keep.all() and np.array_equal(same, ev)
    out, keep = C.rescale_events(ev, (100.0, 50.0), 1.05)
    assert keep.tolist() == [True, True, False]
    assert out[0, 1] == 100 and out[1, 1] == 121 and np.array_equal(out[:, [0, 3, 4]], ev[:2][:, [0, 3, 4]])
