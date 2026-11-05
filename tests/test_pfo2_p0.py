"""PFO2 zero-training probes (docs/EXPERIMENT_RECORDS.md#pfo2-p0-prereg-20261103): CPU tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
NPZ = ROOT / "assets/mano_right.npz"


def test_noise_calibration_hits_the_median():
    from semkine import pfo2_routes as PR2
    geo = PR2.build_geo(NPZ)
    for sigma in (2, 4, 6, 8, 12):
        scale, keep, _ = PR2.calibrate(geo["v"], geo["f"], geo["D"], geo["area"], float(sigma))
        got = PR2.displacement_median(geo["v"], geo["f"], geo["D"], geo["area"], scale, keep, n=40000, seed=7)
        assert abs(got - sigma) <= 0.12 * sigma, (sigma, got)


def test_misassignment_and_identity():
    from semkine import pfo2_routes as PR2
    geo = PR2.build_geo(NPZ)
    part = torch.from_numpy(geo["part"]).long()
    F = len(geo["f"])
    face = torch.randint(0, F, (2, 300), generator=torch.Generator().manual_seed(0))
    bary = torch.full((2, 300, 3), 1.0 / 3)
    valid = torch.ones(2, 300, dtype=torch.bool)
    eye = torch.eye(F)
    f2, b2, st = PR2.perturb(face, bary, valid, part, geo, eye, 1.0, 0.0, torch.Generator().manual_seed(1))
    assert bool((part[f2] != part[face]).all()) and st["mis"] == 600
    f3, b3, _ = PR2.perturb(face, bary, valid, part, geo, eye, 0.0, 1.0, torch.Generator().manual_seed(1))
    assert torch.equal(f3, face) and torch.allclose(b3, bary)


def test_packet_generator_is_a_function_of_the_events():
    from semkine import pfo2_routes as PR2

    class B:
        pass
    a, b = B(), B()
    a.ptr = b.ptr = torch.tensor([0, 5])
    a.events = torch.arange(30, dtype=torch.float32).view(5, 6)
    b.events = a.events.clone()
    x = torch.rand(3, generator=PR2.packet_generator(a, 0))
    y = torch.rand(3, generator=PR2.packet_generator(b, 0))
    b.events[0, 0] += 1
    w = torch.rand(3, generator=PR2.packet_generator(b, 0))
    assert torch.equal(x, y) and not torch.equal(x, w)


def test_sigma_star():
    from tools.pfo2.p0b import sigma_star
    lin = {s: 1.6 * (1 - s / 16.0) for s in (0, 2, 4, 6, 8, 12)}
    assert abs(sigma_star(lin, 0.8) - 8.0) < 1e-9
    assert sigma_star({s: 1.6 for s in (0, 2, 4, 6, 8, 12)}, 0.8) is None
    assert 0 < sigma_star({0: 1.6, 2: 0.1, 4: 0.0, 6: 0.0, 8: 0.0, 12: 0.0}, 0.8) < 2


def test_six_d_round_trip_and_perfect_anchor():
    from tools.pfo2.p0a import from_six_d, root_only, six_d
    from mano_layer import ManoLayer
    rng = np.random.default_rng(0)
    aa = rng.standard_normal((20, 3)) * 0.8
    assert np.allclose(from_six_d(six_d(aa)), aa, atol=1e-6)
    mano = ManoLayer(str(NPZ), add_mean=False).eval()
    gt = np.zeros((5, 51))
    gt[:, 2] = 0.45
    gt[:, 3:51] = rng.standard_normal((5, 48)) * 0.2
    assert float(root_only(mano, gt[:, 3:6], gt, np.zeros(10, np.float32)).max()) < 1e-3


def test_moments_match_direct_loop():
    from tools.pfo2.p0d import L_M, moments
    g = torch.Generator().manual_seed(0)
    B, V, J, C = 2, 7, 3, 4
    H, S = torch.randn(B, V, C, generator=g), (torch.rand(B, V, generator=g) > 0.3).float()
    W = torch.rand(V, J, generator=g)
    verts, joints = torch.randn(B, V, 3, generator=g), torch.randn(B, J, 3, generator=g)
    q = torch.linalg.qr(torch.randn(B, 3, 3, generator=g))[0]
    z0, z1 = moments(H, S, W, verts, joints, q)
    for b in range(B):
        for j in range(J):
            a = W[:, j] * S[b]
            den = a.sum().clamp_min(1e-6)
            assert torch.allclose(z0[b, j], (a[:, None] * H[b]).sum(0) / den, atol=1e-5)
            rel = (q[b].T @ (verts[b] - joints[b, j]).T).T / L_M
            ref = (a[:, None, None] * H[b][:, :, None] * rel[:, None, :]).sum(0) / den
            assert torch.allclose(z1[b, j], ref, atol=1e-4)


def test_state_error_correction_round_trip():
    from tools.pfo2.p0d import corrected, state_error
    rng = np.random.default_rng(1)
    gt = rng.standard_normal((6, 51)) * 0.3
    inc = gt + rng.standard_normal((6, 51)) * 0.1
    r, f = state_error(inc, gt)
    out = corrected(inc, r, f)
    from tools.s45loc.common import rodrigues
    for a, b in zip(out[:, 3:6], gt[:, 3:6]):
        assert np.allclose(rodrigues(a), rodrigues(b), atol=1e-6)
    assert np.allclose(out[:, 6:51], gt[:, 6:51])


def test_ridge_decoder_signal_and_null():
    from tools.pfo2.p0c import fit_predict, rho
    rng = np.random.default_rng(0)
    D, K = 300, 6
    A = rng.standard_normal((K, D))

    def sample(n, null=False):
        y = rng.standard_normal((n, K))
        X = (rng.standard_normal((n, K)) if null else y) @ A + rng.standard_normal((n, D)) * 2.0
        return X, y, rng.integers(0, 9, n)
    Xf, yf, gf = sample(3000)
    Xt, yt, _ = sample(800)
    (p,), _, _ = fit_predict(Xf, yf, gf, [Xt])
    assert rho(yt, p) > 0.9
    Xf, yf, gf = sample(3000, null=True)
    Xt, yt, _ = sample(800, null=True)
    (p,), _, _ = fit_predict(Xf, yf, gf, [Xt])
    assert rho(yt, p) < 0.05
