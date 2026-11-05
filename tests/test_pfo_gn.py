"""PFO phase 2k explicit geometric step (tools/pfo/gn.py): CPU tests, no data."""
from __future__ import annotations

import numpy as np
import torch

from tests.test_pfo import ROOT


class _FK:
    """`MNISTModel._fk` for the S45 configs (pose_repr mano_full_axis_angle, ManoLayer(add_mean=False)), same calls."""

    def __init__(self):
        from mano_layer import ManoLayer
        self.mano = ManoLayer(ROOT / "assets/mano_right.npz", add_mean=False).eval()

    def __call__(self, params, betas):
        from pose_repr import decode_to_mano_inputs
        dec = decode_to_mano_inputs(params, "mano_full_axis_angle", self.mano.hands_components, self.mano.hands_mean)
        return self.mano(betas, dec["global_orient"], dec["local_full_aa"], dec["transl"])


def _state(gen):
    x = torch.zeros(51)
    x[0:3] = torch.tensor([0.01, -0.02, 0.45])
    x[3:6] = 0.3 * torch.randn(3, generator=gen)
    x[6:51] = 0.15 * torch.randn(45, generator=gen)
    return x


def _setup(seed=0, n=400):
    gen = torch.Generator().manual_seed(seed)
    fk = _FK()
    betas = torch.zeros(1, 10)
    f = lambda t: fk(t.view(1, -1), betas)[0][0]  # noqa: E731
    pick = torch.randint(0, fk.mano.f.shape[0], (n,), generator=gen)
    bary = torch.rand(n, 3, generator=gen, dtype=torch.float64) + 0.05
    bary = bary / bary.sum(-1, keepdim=True)
    return gen, f, fk.mano.f[pick], bary


def test_grid_and_rule_constants():
    from tools.pfo import gn
    names = [gn.combo_name(*c) for c in gn.combos()]
    assert len(names) == len(set(names)) == len(gn.ROBUST) * len(gn.GRID_MU) * len(gn.GRID_EPS) == 48
    assert gn.starts() == ("gt", "T10", "T30", "R5", "R15", "F05", "F15", "N")
    assert [gn.block(k) for k in ("T30", "R15", "F15", "N")] == [slice(0, 3), slice(3, 6), slice(6, 51), slice(0, 51)]
    assert gn.RHO_MAX == 0.45 and gn.B_SEEDS == (3407, 3408, 3409)


def test_forward_jacobian_matches_central_differences():
    gen, f, _, _ = _setup()
    X = torch.stack([_state(gen), _state(gen)])
    J = torch.func.vmap(torch.func.jacfwd(f))(X)
    h = 1e-3
    for s in range(2):
        fd = torch.stack([(f(X[s] + h * torch.eye(51)[k]) - f(X[s] - h * torch.eye(51)[k])) / (2 * h)
                          for k in range(51)], -1)
        assert float((J[s] - fd).abs().max()) < 2e-4


def test_zero_residual_gives_zero_step_and_one_step_recovers_exact_correspondences():
    from tools.pfo import gn
    gen, f, fv, bary = _setup(1)
    truth = _state(gen)
    with torch.no_grad():
        Vt = f(truth).double()
    P = (bary[:, :, None] * Vt[fv]).sum(1)
    ray = P / P[:, 2:3]                                                # exact rays (d_z = 1) of the template points
    start = truth.clone()
    start[0:3] += torch.tensor([0.003, -0.002, 0.004])
    start[3:6] += 0.03 * torch.randn(3, generator=gen)
    start[6:51] += 0.03 * torch.randn(45, generator=gen)
    X = torch.stack([truth, start])
    with torch.no_grad():
        V = torch.stack([f(x) for x in X]).double()
    JV = torch.func.vmap(torch.func.jacfwd(f))(X).double()
    _, e, de = gn.residuals(V, JV, fv, bary, ray)
    w = gn.weights("ls", e, gn.residuals(V, JV, fv, bary, ray)[0])
    H, g, sw = gn.normal_equations(e, de, w)
    assert float(sw[0]) == len(fv)
    step = gn.solve_grid(H, g, mus=(0.0,), epss=(1e-6,))[0, 0]
    assert float(step[0].abs().max()) < 1e-3                           # at the truth: no move
    x1 = X[1].double() + step[1]
    with torch.no_grad():
        V1 = f(x1.float()).double()
    P1 = (bary[:, :, None] * V1[fv]).sum(1)
    e1 = P1[:, :2] / P1[:, 2:3] - ray[:, :2]
    assert float(e1.norm(dim=-1).mean()) < 0.1 * float(e[1].norm(dim=-1).mean())


def test_huber_weights_and_depth_floor():
    from tools.pfo import gn
    P = torch.tensor([[[0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.01]]], dtype=torch.float64)
    e = torch.tensor([[[0.01, 0.0], [0.04, 0.0], [0.0, 0.0]]], dtype=torch.float64)   # 5 mm, 20 mm at 0.5 m
    assert torch.allclose(gn.weights("h10", e, P), torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float64))
    assert torch.equal(gn.weights("ls", e, P), torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float64))


def test_rho_matches_the_engine_formula():
    from tools.pfo import gn
    rng = np.random.default_rng(0)
    pred, base, delta = (rng.standard_normal((5, 51)).astype(np.float32) for _ in range(3))
    sl = gn.block("R15")
    ref = [float(np.dot(pred[i][sl].astype(np.float64) - base[i][sl], delta[i][sl].astype(np.float64))
                 / max(np.dot(delta[i][sl].astype(np.float64), delta[i][sl].astype(np.float64)), 1e-12)) for i in range(5)]
    assert np.allclose(gn.rho(pred, base, delta, sl), ref, rtol=1e-12, atol=1e-12)


def _fake_reports(gn, root, sel_scores, zgz):
    """Minimal readings for `select` / `decide`: B's e3 / pert, three ylf_local selections, three zgz reports."""
    from tools.pfo.common import save
    for seed, (e3, r15, f15) in zip(gn.B_SEEDS, ((5.8, 0.45, 0.40), (5.9, 0.45, 0.40), (5.8, 0.46, 0.42))):
        save(root / f"readings/pfo_b_s{seed}/e3/report.json", {"overall": {"mpjpe_ra_mm": e3}})
        save(root / f"readings/pfo_b_s{seed}/pert/report.json",
             {"per_sequence": {"zgz_local": {"R15": {"rho_median": r15}, "F15": {"rho_median": f15}}}})
    for route in gn.ROUTES:
        save(root / f"gn/{route}/sel/report.json",
             {"per_sequence": {"ylf_local": {"combos": {k: {"mean8": v} for k, v in sel_scores.items()}}}})
        kinds = lambda r15, f15: {"R15": {"rho_median": r15}, "F15": {"rho_median": f15}, "T30": {"rho_median": 0.5}}  # noqa: E731
        combos = {name: {"gt": {"mpjpe_ra_mm": e3}, "kinds": kinds(r15, f15)} for name, (e3, r15, f15) in zgz.items()}
        save(root / f"gn/{route}/zgz/report.json",
             {"per_sequence": {"zgz_local": {"combos": combos}, "zgz_global": {"combos": combos}},
              "overall": {**{k: {"gt_mpjpe_ra_mm": v["gt"]["mpjpe_ra_mm"]} for k, v in combos.items()},
                          "identity_mpjpe_ra_mm": 6.0}})


def test_select_and_decide_apply_the_written_rule(tmp_path, monkeypatch):
    from tools.pfo import gn
    from tools.pfo.common import read
    # B's E3 mean is 5.833; "a" is selected on ylf_local in every case
    for case, zgz, verdict, meeting in (("pass", {"a": (5.5, 0.30, 0.20), "b": (9.0, 0.10, 0.10)}, "PASS", ["a"]),
                                        ("bias", {"a": (6.5, 0.30, 0.20), "b": (5.0, 0.10, 0.10)}, "KILL", ["b"]),
                                        ("keep", {"a": (5.5, 0.45, 0.20), "b": (5.0, 0.10, 0.10)}, "KILL", ["b"])):
        root = tmp_path / case
        monkeypatch.setattr(gn, "OUT", root)
        monkeypatch.setattr(gn, "GN", root / "gn")
        monkeypatch.setattr(gn, "verify", lambda: None)
        _fake_reports(gn, root, {"a": 7.0, "b": 8.0}, zgz)
        gn.select()
        assert all(read(root / "gn/selection.json")[r]["combo"] == "a" for r in gn.ROUTES)
        gn.decide()
        dec = read(root / "gn/decision.json")
        assert dec["verdict"] == verdict, case
        assert dec["routes"]["sr1"]["descriptive_grid_points_meeting_both"] == meeting, case


def test_solve_grid_shapes():
    from tools.pfo import gn
    H = torch.eye(51, dtype=torch.float64).expand(3, 51, 51).clone()
    g = torch.ones(3, 51, dtype=torch.float64)
    out = gn.solve_grid(H, g)
    assert out.shape == (len(gn.GRID_MU), len(gn.GRID_EPS), 3, 51)
    assert torch.allclose(out[0, 0], -g / (1.0 + 1e-7 * torch.tensor(gn.SCALES, dtype=torch.float64) ** -2))
