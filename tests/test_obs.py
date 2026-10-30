"""OBS (docs/EXPERIMENT_RECORDS.md#obs-prereg-20261026): clock helpers, the OA1 discriminator, the OC gauge wrapper and
the root-offset fit. CPU, synthetic inputs only (no OBS reading)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from obs import common as C  # noqa: E402
from obs import oa_clock as OA  # noqa: E402
from obs import oc_calib as OC  # noqa: E402


def _clock_pair(n=500, seed=0):
    rng = np.random.default_rng(seed)
    dt = 16734.0 + rng.normal(0, 40, n - 1)
    dt[[120, 333]] *= 2                                   # two dropped frames
    c_h = np.concatenate([[0.0], np.cumsum(dt)])
    c_u = np.arange(n) * (c_h[-1] / (n - 1))
    return c_h, c_u


def test_clock_maps_are_inverse_and_e_vanishes_at_the_ends():
    aux = {"gt_timestamp_ns": (np.cumsum(np.r_[0, 16734000 + np.arange(99) % 7 * 1000]) + 10 ** 18).astype(np.int64)}
    c_h, c_u = C.clocks_us(aux)
    t = np.linspace(0, c_h[-1], 777)
    assert np.allclose(C.h_to_u(C.u_to_h(t, c_h, c_u), c_h, c_u), t, atol=1e-6)
    e = C.e_ms_at(np.array([0.0, c_h[-1] / 1e3]), c_h, c_u)
    assert np.allclose(e, 0.0, atol=1e-9)
    assert np.allclose(C.u_to_h(c_u, c_h, c_u), c_h)


def test_tolerance_grows_with_float32_ulp():
    tol = C.tol_us(np.array([1e3, 1e6, 7e7]))
    assert tol[0] < tol[1] < tol[2] and tol[2] > 3.0 + 2 * 7.0


def test_discriminator_on_synthetic_clocks_and_mapping_flips():
    c_h, c_u = _clock_pair()
    fam = OA.u_family(c_h)
    rng = np.random.default_rng(1)
    for name, c in (("H", c_h), ("U", c_u)):
        L = OA.synthetic_layers(c, 5000, rng)
        j = OA.judge(L, c_h, fam)
        assert j["verdict"] == name
        assert max(j["null_H"], j["null_U"]) <= OA.NULL_MAX
    L = OA.synthetic_layers(c_h, 5000, rng)
    mapped = np.unique(np.rint(C.h_to_u(L, c_h, c_u)).astype(np.int64))
    assert OA.judge(mapped, c_h, fam)["verdict"] == "U"


@pytest.mark.parametrize("m", [2, 7, 18, 30])
def test_v2e_slomo_u_clock_is_recognised(m):
    """v2e without the mask-time mapping: frame n at n * D / (N - 2 + (m - 1) / m); the judge does not know m."""
    c_h, _ = _clock_pair()
    n = len(c_h)
    c_true = C.clock_u(c_h, n - 2.0 + (m - 1.0) / m)
    L = OA.synthetic_layers(c_true, 5000, np.random.default_rng(m))
    j = OA.judge(L, c_h, OA.u_family(c_h))
    assert j["verdict"] == "U" and abs(j["U_pmax"] - (n - 2.0 + (m - 1.0) / m)) < 1e-9


def test_random_layers_alone_are_undecided():
    c_h, _ = _clock_pair()
    rng = np.random.default_rng(2)
    L = np.unique(rng.integers(0, int(c_h[-1]), 20000))
    assert OA.judge(L, c_h, OA.u_family(c_h))["verdict"] == "undecided"


def test_alpha_readings():
    from obs import oa_shift as OS
    assert OS.read_alpha(-0.25, 0.25) == "supports_H"
    assert OS.read_alpha(0.75, 1.25) == "supports_U"
    assert OS.read_alpha(0.25, 0.75) == "supports_U_attenuated"
    assert OS.read_alpha(-0.25, 1.25) == "cannot_discriminate"


class _IdNet(nn.Module):
    """Returns its previous state; records the state it was handed. FK / betas from a real S39 model."""

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.seen = None

    def _resolve_betas_K(self, P, betas, K):
        return self.m._resolve_betas_K(P, betas, K)

    def _fk(self, P, b):
        return self.m._fk(P, b)

    def forward(self, x, prevpos, betas=None, camera_K=None):
        self.seen = prevpos.clone()
        return prevpos.clone()


@pytest.fixture(scope="module")
def s39():
    model, _ = C.MC.load_s39(C.SEEDS[0], torch.device("cpu"))
    return model


def _state(B=3):
    p = torch.zeros(B, 51)
    p[:, 0] = torch.tensor([0.02, -0.05, 0.08])[:B]
    p[:, 2] = torch.tensor([0.55, 0.62, 0.70])[:B]
    p[:, 3:6] = torch.tensor([2.2, 0.1, -0.2])
    p[:, 6:] = 0.05 * torch.randn(B, 45, generator=torch.Generator().manual_seed(0))
    return p


def test_gauge_sigma_one_is_bitwise(s39):
    G = OC.gauge_net_class()
    x = torch.zeros(2, 180, 240, 2)
    x[:, 80:100, 100:140, 0] = 0.5
    p = _state(2)
    with torch.no_grad():
        assert torch.equal(G(s39, 1.0)(x, p), s39(x, p))


@pytest.mark.parametrize("sigma", [0.90, 0.94, 1.06])
def test_gauge_moves_wrist_to_w_over_sigma_and_round_trips(s39, sigma):
    G = OC.gauge_net_class()
    inner = _IdNet(s39)
    g = G(inner, sigma)
    p = _state()
    x = torch.zeros(3, 180, 240, 2)
    with torch.no_grad():
        z = g(x, p)
        w = g.wrist(p)
        we = g.wrist(inner.seen)
    assert torch.allclose(we, w / sigma, atol=1e-5)                       # event gauge: the wrist at w / sigma
    assert torch.allclose(z, p, atol=1e-5)                                 # an identity network round-trips exactly
    assert torch.equal(inner.seen[:, 3:], p[:, 3:])                        # only the translation moves


def test_root_offset_fit_recovers_left_and_right():
    from scipy.spatial.transform import Rotation as R
    rng = np.random.default_rng(3)
    gt = np.zeros((200, 51), np.float32)
    gt[:, 3:6] = R.random(200, random_state=4).as_rotvec()
    Rb = R.from_rotvec([0.05, -0.08, 0.03]).as_matrix()
    Rg = R.from_rotvec(gt[:, 3:6]).as_matrix()
    for side in ("left", "right"):
        Rp = np.einsum("ji,njk->nik", Rb, Rg) if side == "left" else np.einsum("nij,kj->nik", Rg, Rb)
        pred = gt.copy()
        pred[:, 3:6] = R.from_matrix(Rp).as_rotvec()
        left, right = OC.fit_root(pred, gt)
        fit = left if side == "left" else right
        assert np.allclose(fit, Rb, atol=1e-6)
        corr = OC.apply_root(pred, side, fit)
        assert np.allclose(R.from_rotvec(corr[:, 3:6]).as_matrix(), Rg, atol=1e-5)
    _ = rng


def test_partner_and_groups():
    assert OC.partner("zgz_local") == "zgz_global" and OC.partner("ch_global") == "ch_local"
    names = [g for g, _ in OC.groups()]
    assert names[0] == "zgz" and len(names) == 19


# ------------------------------------------------------------------------------------------------ OA1-R2
from obs import oa_phase as OP  # noqa: E402


def _clock_pair_long(n=1200, seed=5):
    rng = np.random.default_rng(seed)
    dt = 16760.0 + rng.normal(0, 60, n - 1) + 120 * np.sin(np.arange(n - 1) / 300.0)
    c_h = np.concatenate([[0.0], np.cumsum(dt)])
    return c_h, rng


@pytest.mark.parametrize("m_last", [2, 9, 30])
def test_phase_judge_on_harsh_v2e_like_layers(m_last):
    c_h, rng = _clock_pair_long()
    n = len(c_h)
    jh = OP.judge(OP.v2e_like_layers(c_h, 0.5, rng), c_h, rng)
    assert jh["verdict"] == "H"
    c_u = C.clock_u(c_h, n - 2.0 + (m_last - 1.0) / m_last)
    ju = OP.judge(OP.v2e_like_layers(c_u, 0.5, rng), c_h, rng)
    assert ju["verdict"] == "U" and abs(ju["U_pmax"] - (n - 2.0 + (m_last - 1.0) / m_last)) < 1e-9


def test_phase_judge_uniform_is_undecided_and_mapping_flips():
    c_h, rng = _clock_pair_long()
    j = OP.judge(OP.uniform_layers(c_h, 40000, rng), c_h, rng)
    assert j["verdict"] == "undecided" and abs(j["z_extra_uniform"]) <= OP.NULL_Z_MAX
    L = OP.v2e_like_layers(c_h, 0.5, rng)
    mapped = np.unique(np.rint(C.h_to_u(L, c_h, C.clock_u(c_h, len(c_h) - 1.0))).astype(np.int64))
    assert OP.judge(mapped, c_h, rng)["verdict"] == "U"


def test_llr_is_near_zero_for_uniform_phases():
    rng = np.random.default_rng(9)
    ph = rng.random(300000)
    n = rng.integers(0, 3000, len(ph))
    assert abs(OP.llr_folds(ph, n % 3 == 1, n % 3 == 2)) < 0.02
