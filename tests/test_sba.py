#!/usr/bin/env python3
"""SBA (docs/EXPERIMENT_RECORDS.md#sba-prereg-20261022): the swept-area mean field and its solver on synthetic data.

1. The output grid reproduces the dataset's floor(3i/8) binning (3/3/2 gain, all source pixels counted once).
2. The MANO edge table is a manifold with a single open boundary (the wrist cut).
3. A sphere translated in the image plane sweeps 4 delta r of contour area (2 delta r advancing, 2 retreating),
   deposited on the correct side; zero motion sweeps nothing.
4. Poisson counts drawn from the mean field are best explained by the true translation.
5. The Fisher-scoring solver moves a MANO end state towards the truth (root-aligned joints) from events drawn
   from its own mean field.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "model")]
from semkine import sba as SB  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_SRC = torch.tensor([[603.45, 0.0, 325.09], [0.0, 602.96, 242.10], [0.0, 0.0, 1.0]], device=DEV)


def icosphere(level: int = 3):
    t = (1.0 + 5 ** 0.5) / 2
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
         [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
         [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
         [8, 6, 7], [9, 8, 1]]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(level):
        cache, nf = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[k] = len(v) - 1
            return cache[k]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = nf
    return np.array(v), np.array(f)


@pytest.fixture(scope="module")
def sphere():
    v, f = icosphere(4)
    topo = SB.Topology(torch.tensor(f, device=DEV))
    return torch.tensor(v, dtype=torch.float32, device=DEV), topo


def test_grid_binning():
    g = SB.make_grid(DEV)
    assert int(g.gain.sum()) == SB.SRC_W * SB.SRC_H
    assert set(g.gain.unique().tolist()) == {4.0, 6.0, 9.0}
    assert g.lut_x.tolist()[:9] == [0, 0, 0, 1, 1, 1, 2, 2, 3]
    i = torch.arange(SB.SRC_W, device=DEV)
    assert torch.equal(g.lut_x, (i * 3) // 8)


def test_mano_topology():
    from mano_layer import ManoLayer
    m = ManoLayer(str(REPO / "assets/mano_right.npz"), add_mean=False)
    topo = SB.Topology(m.f.to(DEV))
    F = m.f.shape[0]
    nb = int(topo.boundary.sum())
    assert 2 * (topo.n_edges - nb) + nb == 3 * F
    assert 0 < nb < 60                                   # the open wrist ring only


def _sphere_traj(v, shift_m, z0=0.6, r=0.05, S=1):
    """Sphere of radius r at depth z0, moved along x by shift_m over S segments: verts (S+1, V, 3)."""
    base = v * r + torch.tensor([0.0, 0.0, z0], device=DEV)
    a = torch.linspace(0, 1, S + 1, device=DEV)
    return torch.stack([base + torch.tensor([shift_m, 0.0, 0.0], device=DEV) * ai for ai in a])


@pytest.mark.parametrize("shift", [2e-4, 1e-3, 4e-3])
def test_sphere_sweep_area(sphere, shift):
    v, topo = sphere
    grid = SB.make_grid(DEV)
    V = _sphere_traj(v, shift)
    uv = SB.project_src(V, K_SRC)
    plan = SB.build_plan(V, uv, topo, grid)
    crop = SB.Crop(0, 0, SB.OUT_H, SB.OUT_W)
    A = SB.sweep_area(uv[None], plan, grid, crop, 1)[0]
    delta = float(K_SRC[0, 0]) * shift / 0.6                   # px (source)
    r_px = float(K_SRC[0, 0]) * 0.05 / (0.6 ** 2 - 0.05 ** 2) ** 0.5
    expect = 4 * delta * r_px
    tot = float(A.sum())
    assert abs(tot - expect) / expect < 0.06, (tot, expect)
    pos, neg = float(A[SB.CH_SIL_POS].sum()), float(A[SB.CH_SIL_NEG].sum())
    assert abs(pos - neg) / tot < 0.05
    assert float(A[SB.CH_INT].sum()) == 0.0
    # the advancing area lies right of the sphere centre (motion +x), the retreating area left of it
    cx = float(uv[0, :, 0].mean()) * SB.OUT_W / SB.SRC_W
    cols = torch.arange(SB.OUT_W, device=DEV, dtype=torch.float32)
    assert float((A[SB.CH_SIL_POS].sum((0, 1)) * cols).sum() / pos) > cx
    assert float((A[SB.CH_SIL_NEG].sum((0, 1)) * cols).sum() / neg) < cx


def test_zero_motion_sweeps_nothing(sphere):
    v, topo = sphere
    grid = SB.make_grid(DEV)
    V = _sphere_traj(v, 0.0)
    uv = SB.project_src(V, K_SRC)
    plan = SB.build_plan(V, uv, topo, grid)
    A = SB.sweep_area(uv[None], plan, grid, SB.Crop(0, 0, SB.OUT_H, SB.OUT_W), 1)
    assert float(A.abs().sum()) == 0.0


def test_poisson_consistency(sphere):
    """Events drawn from the mean field at the true shift: the likelihood scan peaks at the truth."""
    v, topo = sphere
    grid = SB.make_grid(DEV)
    S = 4
    crop = SB.Crop(60, 80, 70, 90)
    true = 3e-3
    V = _sphere_traj(v, true, S=S)
    uv = SB.project_src(V, K_SRC)
    plan = SB.build_plan(V, uv, topo, grid)
    A = SB.sweep_area(uv[None], plan, grid, crop, S)
    model = SB.Model(kappa_sil=2.0, kappa_int=1.0)
    bkg = torch.full((S, crop.h, crop.w), 0.01, device=DEV)
    win = SB.Window(torch.linspace(0, 1, S + 1, device=DEV), torch.full((S,), 5.0, device=DEV),
                    torch.zeros(2, S, crop.h, crop.w, device=DEV), bkg, 0.5, crop, 0, 0, 0.0, 0.0, None)
    lam = SB.rates(A, win, model)[0, 0]
    gen = torch.Generator(device=DEV).manual_seed(0)
    win.N = torch.stack([torch.zeros_like(lam), torch.poisson(lam, generator=gen)])
    shifts = [true * k for k in (0.0, 0.5, 0.8, 1.0, 1.25, 1.6, 2.0)]
    lls = []
    for s in shifts:
        Vs = _sphere_traj(v, s, S=S)
        uvs = SB.project_src(Vs, K_SRC)
        ps = SB.build_plan(Vs, uvs, topo, grid)
        As = SB.sweep_area(uvs[None], ps, grid, crop, S)
        lls.append(float(SB.loglik(SB.rates(As, win, model), SB.counts(win, model))[0]))
    assert int(np.argmax(lls)) == 3, lls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MANO solver test needs the GPU")
def test_solver_recovers_mano_state():
    from mano_layer import ManoLayer
    mano = ManoLayer(str(REPO / "assets/mano_right.npz"), add_mean=False).to(DEV).eval()
    geo = SB.Geometry(mano, DEV)
    betas = torch.zeros(1, 10, device=DEV)
    x0 = torch.zeros(51, device=DEV)
    x0[2] = 0.55
    x0[3:6] = torch.tensor([2.6, 0.2, -0.3], device=DEV)
    x1 = x0.clone()
    x1[0] += 0.004
    x1[6:51] += 0.05 * torch.randn(45, generator=torch.Generator().manual_seed(1)).to(DEV)
    S = 10
    alphas = torch.linspace(0, 1, S + 1, device=DEV)
    v, uv, _ = geo.trajectories(x0[None], x1[None], alphas, betas, K_SRC)
    plan = SB.build_plan(v[0], uv[0], geo.topo, geo.grid, geo.groups, geo.n_groups)
    crop = SB.crop_from(uv)
    model = SB.Model(kappa_sil=3.0, kappa_int=1.5)
    bkg = torch.full((S, crop.h, crop.w), 0.005, device=DEV)
    win = SB.Window(alphas, torch.full((S,), 5.0, device=DEV), torch.zeros(2, S, crop.h, crop.w, device=DEV),
                    bkg, 0.5, crop, 0, 0, 0.0, 0.0, None)
    A = SB.sweep_area(uv, plan, geo.grid, crop, S)
    lam = SB.rates(A, win, model)[0, 0]
    win.N = torch.stack([torch.zeros_like(lam), torch.poisson(lam, generator=torch.Generator(device=DEV)
                                                              .manual_seed(2))])
    ev = SB.Evaluator(geo, win, model, betas, K_SRC)
    prec = torch.ones(51, device=DEV)
    prec[:3], prec[3:6], prec[6:] = 1 / 0.005 ** 2, 1 / 0.05 ** 2, 1 / 0.05 ** 2
    pri = SB.Priors(inc_prec=torch.diag(prec))
    start = x1.clone()
    start[6:51] = x0[6:51]                                  # fingers start at "no motion"
    res = SB.solve(ev, x0, start, pri, pools=(4, 2, 1, 1, 1, 1, 1, 1))

    def joint_err(a):
        _, ja = geo.fk(a[None], betas)
        _, jb = geo.fk(x1[None], betas)
        return float(((ja - ja[:, :1]) - (jb - jb[:, :1])).norm(dim=-1).mean())
    assert joint_err(res.x1) < 0.8 * joint_err(start), (joint_err(start), joint_err(res.x1))
