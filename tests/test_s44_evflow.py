"""S44 EvTree phase-0 library checks (docs/EXPERIMENT_RECORDS.md#s44-prereg-20261019).

1. plane fit recovers the normal velocity of a synthetic moving straight edge;
2. the SAE writer is prefix-equivalent: any chunking of the stream gives the same surface bit for bit;
3. the central-difference projected-vertex Jacobian matches autograd through the MANO layer;
4. the rate solve recovers a known rate from exact normal-flow constraints.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from s44 import evflow as EF  # noqa: E402

DEV = torch.device("cpu")


def _edge_surface(vx, vy, t0_us=1_000_000, h=EF.H, w=EF.W):
    """SAE of a straight edge moving with image velocity (vx, vy) px/s: T(x) = t0 + (n.x)/|v|, both polarities."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    sp = np.hypot(vx, vy)
    n = np.array([vx, vy]) / sp
    T = t0_us + (n[0] * (xx - w / 2) + n[1] * (yy - h / 2)) / sp * 1e6
    T = T - T.min() + 1000
    t = torch.from_numpy(np.stack([T, T]).astype(np.int64))
    return t, n, sp


@pytest.mark.parametrize("vx,vy", [(40.0, 0.0), (0.0, -25.0), (30.0, 40.0), (-200.0, 90.0)])
def test_plane_fit_recovers_normal_flow(vx, vy):
    T, n, sp = _edge_surface(vx, vy)
    y = torch.tensor([60, 90, 120])
    x = torch.tensor([80, 120, 160])
    pol = torch.zeros(3, dtype=torch.long)
    f = EF.plane_fit(T, pol, y, x, r=2, dfit_us=1e9)
    assert bool(f["valid"].all())
    assert torch.allclose(f["s"], torch.full((3,), sp, dtype=torch.float64), rtol=1e-4)
    assert torch.allclose(f["n"], torch.tensor(n)[None].expand(3, 2), atol=1e-3)   # integer-us surface rounding


def test_plane_fit_rejects_isolated_pixel():
    T = torch.full((2, EF.H, EF.W), -1, dtype=torch.int64)
    T[0, 50, 50] = 10_000
    f = EF.plane_fit(T, torch.tensor([0]), torch.tensor([50]), torch.tensor([50]), r=2, dfit_us=1e6)
    assert not bool(f["valid"].any())


class _FakeSeq:
    """A Seq stand-in with a random stream (several events per pixel and per ms, duplicates included)."""

    def __init__(self, n_ms=120, rate=400, seed=0):
        rng = np.random.default_rng(seed)
        counts = rng.poisson(rate, n_ms)
        self.offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        n = int(self.offsets[-1])
        self.events = np.stack([rng.integers(100, 140, n), rng.integers(70, 100, n), rng.integers(0, 2, n)],
                               1).astype(np.uint8)
        self.tsub = rng.integers(0, 1000, n).astype(np.uint16)


def test_sae_prefix_equivalence():
    sq = _FakeSeq()
    T0, R0 = EF.build_sae(sq, 10, 110, DEV)
    a0, a1 = EF.window_rows(sq, 10, 110)
    rng = np.random.default_rng(1)
    for _ in range(5):
        cuts = sorted(rng.choice(np.arange(1, a1 - a0), size=7, replace=False).tolist())
        T1, R1 = EF.build_sae(sq, 10, 110, DEV, chunks=cuts)
        assert torch.equal(T0, T1) and torch.equal(R0, R1)
    # one event at a time (the streaming writer) on a short window
    T2, R2 = EF.build_sae(sq, 40, 45, DEV)
    b0, b1 = EF.window_rows(sq, 40, 45)
    T3, R3 = EF.build_sae(sq, 40, 45, DEV, chunks=list(range(1, b1 - b0)))
    assert torch.equal(T2, T3) and torch.equal(R2, R3)


def test_sae_latest_event_semantics():
    sq = _FakeSeq(n_ms=30, rate=200, seed=3)
    T, R = EF.build_sae(sq, 0, 29, DEV)
    t = EF.event_times_us(sq, 0, 29)
    ev = sq.events
    pix = (np.clip(ev[:, 2], 0, 1).astype(np.int64) * EF.H + ev[:, 1]) * EF.W + ev[:, 0]
    ref = np.full(2 * EF.H * EF.W, -1, np.int64)
    np.maximum.at(ref, pix, t)
    assert np.array_equal(T.view(-1).numpy(), ref)
    hit = R.view(-1) >= 0
    rows = R.view(-1)[hit].numpy()
    assert np.array_equal(t[rows], T.view(-1)[hit].numpy())



def test_fd_jacobian_matches_autograd():
    m = EF.Mano64(DEV)
    torch.manual_seed(0)
    params = torch.zeros(2, 51, dtype=torch.float64)
    params[:, 2] = 0.6
    params[:, 3:6] = torch.randn(2, 3, dtype=torch.float64) * 0.4
    params[:, 6:] = torch.randn(2, 45, dtype=torch.float64) * 0.2
    betas = torch.zeros(1, 10, dtype=torch.float64)
    intr = (226.0, 226.0, 120.0, 90.0)
    _, uv, _, J = m.jacobian(params, betas, intr)
    p = params.clone().requires_grad_(True)
    v, _ = m.fk(p, betas)
    uv2, _ = m.project(v, intr)
    vid = [0, 333, 745]
    for b in range(2):
        for vi in vid:
            for c in range(2):
                g, = torch.autograd.grad(uv2[b, vi, c], p, retain_graph=True)
                assert torch.allclose(J[b, vi, c], g[b], rtol=1e-5, atol=1e-6)


def test_solve_rate_recovers_exact_rate():
    torch.manual_seed(1)
    A = torch.randn(400, 51, dtype=torch.float64)
    th = torch.randn(51, dtype=torch.float64)
    n = A / A.norm(dim=1, keepdim=True)
    s = n @ th
    est, Lam = EF.solve_rate(n, s.abs() * 0 + s, torch.ones(400, dtype=torch.float64), kappa=0.0, lam_p=1e-9,
                             prior_prec=torch.eye(51, dtype=torch.float64))
    assert torch.allclose(est, th, atol=1e-5)
