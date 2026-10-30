#!/usr/bin/env python3
"""SBA round 3 (docs/EXPERIMENT_RECORDS.md#sba-r3-prereg-20261022): probe contract and likelihood utilities.

1. Anatomical axes on the template: + flexion curls the fingertip towards the palm, + abduction moves it laterally,
   + opposition brings the thumb tip towards its target.
2. The parent-frame primitive rotates the child bone about R_parent_world a by exactly theta at arbitrary poses.
3. FK bisection reaches the target joint error within 2 %, and the root rotation keeps the wrist fixed.
4. The root-compensated synergy has (almost) no root-explainable displacement left.
5. The batched Newton equals tools/sba/q0_landscape.py::profile; refined maxima / interpolation are exact on parabolas.
6. Event bookkeeping: chained sub-windows partition the events; thinning and time shuffling are deterministic and keep
   what they must keep; a single-window chain equals the single-window problem.
7. (data) The informative filter of the packet bank equals the window's event count; donor events land in the window.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "tools"), str(REPO), str(REPO / "model")]
from sba import common as CM  # noqa: E402
from sba import r3_lik as L  # noqa: E402
from sba import r3_probe as PR  # noqa: E402

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def ctx():
    return CM.Ctx(CPU)


@pytest.fixture(scope="module")
def betas():
    return torch.zeros(1, 10)


def template(ctx):
    x = torch.zeros(1, 51)
    x[0, 6:] = -ctx.mean.reshape(-1)
    x[0, 2] = 0.5
    return x


def random_pose(ctx, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(1, 51)
    x[0, :3] = torch.tensor([0.02, -0.03, 0.55])
    x[0, 3:6] = 0.6 * (torch.rand(3, generator=g) - 0.5)
    x[0, 6:] = 0.5 * (torch.rand(45, generator=g) - 0.5)
    return x


def test_template_axes(ctx, betas):
    tc = PR.template_checks(ctx, betas)
    assert min(tc["flex_tip_cos"].values()) >= PR.VALID["flex_tip_cos"]
    assert min(tc["abd_lateral_cos"].values()) >= 0.98
    assert tc["thumb_target_dist_mm"][1] < tc["thumb_target_dist_mm"][0]
    assert max(v for k, v in tc["twist_mm"].items() if k in PR.TWIST_HELD) < PR.VALID["twist_template_mm"]


def world_rot(ctx, x, jidx):
    """World rotation of MANO joint jidx's frame (root composed with the full local rotations down the chain)."""
    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
    chain = []
    j = jidx
    while j > 0:
        chain.append(j)
        j = parents[j]
    R = CM.so3_exp(x[0, 3:6])
    for j in reversed(chain):
        R = R @ CM.so3_exp(x[0, 6 + 3 * (j - 1):9 + 3 * (j - 1)] + ctx.mean[j - 1])
    return R


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("finger,level", [("index", 0), ("middle", 1), ("pinky", 0), ("ring", 1)])
def test_parent_frame_rotation(ctx, betas, seed, finger, level):
    """Rotating MANO joint j about axis a (parent frame) turns the bone j -> child about R_parent_world a by theta."""
    fr = PR.make_frame(ctx, betas)
    x = random_pose(ctx, seed)
    jidx = PR.MANO_J[finger][level]
    parent = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14][jidx]
    a = fr.flex[finger]
    th = math.radians(12.0)
    x2 = PR.rot_parent(ctx, x, jidx, a, torch.tensor([th]))
    op = PR.OP[finger]
    j1, j2 = PR.joints(ctx, x, betas)[0], PR.joints(ctx, x2, betas)[0]
    b1 = PR.unit(j1[op[level + 1]] - j1[op[level]])
    b2 = PR.unit(j2[op[level + 1]] - j2[op[level]])
    Rp = world_rot(ctx, x, parent) if parent > 0 else CM.so3_exp(x[0, 3:6])
    expect = CM.so3_exp(th * (Rp @ a)) @ b1
    assert float(torch.acos(torch.clamp(expect @ b2, -1, 1))) < 1e-3


@pytest.mark.parametrize("fam,target", [("pip", 5.0), ("mcpf", 10.0), ("mcpa", 5.0), ("syn", 3.0), ("root", 5.0),
                                        ("thm", 10.0)])
def test_bisection_reaches_target(ctx, betas, fam, target):
    fr = PR.make_frame(ctx, betas)
    x = random_pose(ctx, 7)[0]
    jc = PR.joints(ctx, x[None], betas)
    Bj = PR.root_jacobian(ctx, x, betas)
    axis = PR.thumb_axis(ctx, fr, x, betas).tolist() if fam == "thm" else ([0.0, 1.0, 0.0] if fam == "root" else None)
    for sign in (1, -1):
        deg, ok = PR.bisect(ctx, fr, fam, sign, target, x, betas, "index", axis, Bj, jc)
        assert ok
        sp = PR.Spec(fam, sign, deg, "index", axis)
        e21, _ = PR.errors(PR.joints(ctx, PR.apply(ctx, fr, sp, x[None], betas), betas), jc)
        e = float(PR.metric(e21, PR.BISECT[fam], "index")[0])
        assert abs(e / target - 1.0) <= PR.TOL
        e2, _ = PR.errors(PR.joints(ctx, PR.apply(ctx, fr, sp, x[None], betas, deg=torch.tensor([2 * sign * deg])),
                                    betas), jc)
        lo, hi = PR.VALID["linear_band"]
        assert lo <= float(PR.metric(e2, PR.BISECT[fam], "index")[0]) / e <= hi


def test_root_keeps_wrist(ctx, betas):
    x = random_pose(ctx, 3)
    w0 = ctx.wrist(x, betas)
    x2 = PR.root_rotvec(ctx, x, torch.tensor([[0.05, -0.1, 0.2]]), betas)
    assert float((ctx.wrist(x2, betas) - w0).norm()) < 1e-6


def test_root_compensation(ctx, betas):
    fr = PR.make_frame(ctx, betas)
    x = random_pose(ctx, 5)[0]
    jc = PR.joints(ctx, x[None], betas)
    Bj = PR.root_jacobian(ctx, x, betas)
    Xh, _ = PR._syn_rc(ctx, fr, x[None], torch.tensor([10.0]), Bj, jc, betas)
    jh = PR.joints(ctx, Xh, betas)
    d = ((jh - jh[:, :1]) - (jc - jc[:, :1])).reshape(1, 63)
    om = d @ torch.linalg.pinv(Bj).T
    res = d - om @ Bj.T
    assert float(1.0 - (res ** 2).sum() / (d ** 2).sum()) < 0.02


def test_zero_perturbation_renders_identical(ctx, betas):
    from sba.p0_visibility import render
    x = random_pose(ctx, 1)
    v, _ = ctx.geo.fk(x, betas)
    K = torch.tensor([[603.45, 0.0, 325.09], [0.0, 602.96, 242.10], [0.0, 0.0, 1.0]])
    uv = CM.SB.project_src(v, K)[0].numpy()
    z = v[0, :, 2].numpy()
    faces = ctx.mano.f.long().numpy()
    fg = np.zeros(len(faces), int)
    box = (int(uv[:, 0].min()) - 2, int(uv[:, 1].min()) - 2, int(uv[:, 0].max()) + 2, int(uv[:, 1].max()) + 2)
    assert (render(uv, z, faces, fg, box) != render(uv.copy(), z.copy(), faces, fg, box)).sum() == 0


def poisson_problems(seed, P=3, k=7, n=4000):
    """Profile problems with an interior optimum: counts drawn from w* . X + B with every w* well above zero; event
    cells and all-cell totals."""
    g = torch.Generator().manual_seed(seed)
    Xa = torch.rand(P, k, n, generator=g, dtype=torch.float64)
    Ba = torch.rand(n, generator=g, dtype=torch.float64) * 0.05 + 0.01
    ws = 1.0 + torch.rand(k, generator=g, dtype=torch.float64)
    Na = torch.poisson(torch.einsum("k,kn->n", ws, Xa[0]) + Ba, generator=g)  # events of hypothesis 0
    Xa[1:] = Xa[0] * (1.0 + 0.05 * torch.rand(P - 1, k, n, generator=g, dtype=torch.float64))
    cell = Na > 0
    return Xa[:, :, cell], Xa.sum(-1), Ba[cell], float(Ba.sum()), Na[cell]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_newton_equals_q0_profile(seed):
    """Interior optimum: identical to q0_landscape.profile."""
    from sba.q0_landscape import profile
    X, Xt, B, Bt, N = poisson_problems(seed)
    ll, w = L.newton(X, Xt, B, Bt, N, nan_unconverged=True)
    assert bool(torch.isfinite(ll).all()) and bool((w > 0).all())
    for p in range(X.shape[0]):
        ref = profile(X[p], Xt[p], B, Bt, N)
        assert abs(float(ll[p]) - ref) <= 1e-8 * max(1.0, abs(ref))


@pytest.mark.parametrize("seed", [2, 5, 9])
def test_newton_converges_where_plain_newton_oscillates(seed):
    """Weights pinned at the bound make q0's plain projected Newton oscillate; the active-set / backtracking version
    converges to a KKT point whose value is at least q0's 40-iteration value."""
    from sba.q0_landscape import profile
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(3, 7, 400, generator=g, dtype=torch.float64) * 0.3
    Xt = X.sum(-1) * (1.0 + torch.rand(3, 7, generator=g, dtype=torch.float64))
    B = torch.rand(400, generator=g, dtype=torch.float64) * 0.05 + 0.01
    N = torch.poisson(torch.full((400,), 2.0, dtype=torch.float64), generator=g) + 1
    ll, w = L.newton(X, Xt, B, float(B.sum()), N, iters=200, nan_unconverged=True)
    assert bool(torch.isfinite(ll).all())
    lam = torch.einsum("pk,pkn->pn", w, X) + B
    grad = torch.einsum("pkn,pn->pk", X, N / lam) - Xt
    scale = Xt.abs().amax(1, keepdim=True)
    assert bool((grad[w > 0].abs() <= 1e-6 * scale.expand_as(grad)[w > 0]).all())        # free: stationary
    assert bool((grad[w == 0] <= 1e-6 * scale.expand_as(grad)[w == 0]).all())           # at the bound: no ascent
    for p in range(3):
        assert float(ll[p]) >= profile(X[p], Xt[p], B, float(B.sum()), N) - 1e-8


def test_refine_and_interp_exact_on_parabola():
    s = np.asarray(L.SIG)
    y = -3.0 * (s - 0.953) ** 2 + 7.0
    m, sx, edge = L.refine_max(y)
    assert abs(m - 7.0) < 1e-9 and abs(sx - 0.953) < 1e-9 and not edge
    assert abs(L.interp(y, 0.971) - (-3.0 * (0.971 - 0.953) ** 2 + 7.0)) < 1e-9
    assert L.refine_max(s)[2]                                                  # increasing -> edge


def fake_events(n=5000, lo=1000, hi=1200, seed=0):
    r = np.random.default_rng(seed)
    ev = np.stack([r.integers(0, 240, n), r.integers(0, 180, n), r.integers(0, 2, n)], 1).astype(np.uint8)
    return ev, np.sort(r.integers(lo + 1, hi + 1, n))


def test_partition_thin_shuffle():
    ev_ms = fake_events()
    kms = [1000, 1050, 1100, 1150, 1200]
    parts = [L.take(ev_ms, kms[j], kms[j + 1]) for j in range(4)]
    assert sum(len(p[1]) for p in parts) == len(ev_ms[1])
    for j, p in enumerate(parts):
        assert np.all((p[1] > kms[j]) & (p[1] <= kms[j + 1]))
    a, b = L.thin(ev_ms, 0.25, 42), L.thin(ev_ms, 0.25, 42)
    assert np.array_equal(a[1], b[1]) and abs(len(a[1]) / len(ev_ms[1]) - 0.25) < 0.03
    sh = L.shuffle_times(ev_ms, 3)
    assert np.array_equal(np.sort(sh[1]), np.sort(ev_ms[1])) and np.array_equal(sh[0], ev_ms[0])
    assert not np.array_equal(sh[1], ev_ms[1])


class _Win:
    def __init__(self, g, S=4, h=6, w=5):
        self.N = torch.poisson(torch.full((2, S, h, w), 0.7), generator=g)
        self.uni = torch.rand(S, h, w, generator=g)
        self.bkg = torch.rand(S, h, w, generator=g) * 0.01 + 1e-3


def test_single_window_chain_equals_problem():
    g = torch.Generator().manual_seed(0)
    w = _Win(g)
    G = torch.rand(3, 6, 4 * 6 * 5, generator=g)
    a = L.newton(*L.problem(G, w))[0]
    b = L.newton(*L.chain_problem([G], [w]))[0]
    assert torch.allclose(a, b)


DATA = (CM.REPO / "outputs" / "xdl" / "packet_bank.json").exists()


@pytest.mark.skipif(not DATA, reason="packet bank not present")
def test_informative_filter_and_donor_retime(ctx):
    from sba.p0_landscape import packets
    for seq, d, index, row, rows in packets("zgz_local")[200:212]:
        sd = CM.SeqData(seq, d, CPU)
        prev, end = int(row["previous_end_ms"]), int(row["end_ms"])
        win = CM.window(ctx, sd, prev, end, sd.gt(prev), sd.gt(end))
        assert win.n_events == int(row["n_raw_events"])
        dr = rows[(index + len(rows) // 2) % len(rows)]
        ev, ms = CM.retime(sd.events(int(dr["previous_end_ms"]), int(dr["end_ms"])), int(dr["previous_end_ms"]),
                           prev, end)
        assert len(ms) == 0 or (ms.min() > prev and ms.max() <= end)
