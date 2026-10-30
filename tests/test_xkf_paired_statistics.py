from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'model')]
import numpy as np
import pytest
import torch

from semkine.xkf_paired_statistics import GROUPS, PairedStatisticsPolicy, fit_paired_gains


def test_correlated_biased_gated_fit_matches_independent_objective():
    rng = np.random.default_rng(381)
    common = rng.normal(size=(173, 51))
    ei = common + rng.normal(size=common.shape) + .6
    ed = .7*common + rng.normal(size=common.shape)*1.4 - .3
    gates = rng.uniform(size=len(ei))
    fitted = fit_paired_gains(ei, ed, gates, [.2]*19)
    for index, (start, stop) in enumerate(GROUPS):
        a, b = ei[:, start:stop], ed[:, start:stop]
        k = fitted['coefficients'][index]
        objective = lambda v: np.mean(np.sum(((1-gates[:, None]*v)*a + gates[:, None]*v*b)**2, axis=1))
        # Check against a dense independently evaluated scalar objective,
        # including both constrained endpoints and the previous coefficient.
        assert objective(k) <= min(objective(v) for v in np.linspace(0, 1, 501)) + 1e-12
        assert objective(k) <= objective(.2) + 1e-12
        assert fitted['groups'][index]['objective_at_fit'] == pytest.approx(objective(k))
        assert fitted['groups'][index]['vi_uncentered'] > fitted['groups'][index]['vi_centered']


def test_perfect_direct_measurement_and_degenerate_fallback():
    ei = torch.ones(8, 51)
    fit = fit_paired_gains(ei, torch.zeros_like(ei), torch.ones(8), [.13]*19)
    assert fit['coefficients'] == [1.]*19
    assert all(x['objective_at_fit'] == 0 for x in fit['groups'])
    for ed, gates in ((ei, torch.ones(8)), (torch.zeros_like(ei), torch.zeros(8))):
        fit = fit_paired_gains(ei, ed, gates, [.13]*19)
        assert fit['coefficients'] == [.13]*19
        assert all(x['degenerate_fallback'] for x in fit['groups'])


def spec():
    return dict(fixed_effective_gains=[.7]*19, gain_scale=[1., 1., 1.],
                count_gate='ramp1000', motion_count_gate='ramp1000',
                translation_reference='joint_centroid')


def test_runtime_uses_fitted_effective_coefficient_once():
    policy = PairedStatisticsPolicy(spec())
    counts = torch.tensor([0, 500, 2000])
    gains = policy.gains(torch.full((19,), .05), counts)
    assert torch.equal(gains, policy.gains(torch.full((19,), .9), counts))
    assert torch.equal(gains[0], torch.zeros(19))
    assert torch.equal(gains[1], torch.full((19,), .35))
    assert torch.equal(gains[2], torch.full((19,), .7))


@pytest.mark.parametrize('patch', [dict(gain_scale=[.5, .25, .5]), dict(count_gate=None),
                                  dict(motion_count_gate=None), dict(temperature_multiplier=1.),
                                  dict(fixed_effective_gains=[float('nan')]*19),
                                  dict(fixed_effective_gains=[True]*19)])
def test_runtime_rejects_other_interventions(patch):
    with pytest.raises(ValueError):
        PairedStatisticsPolicy(dict(spec(), **patch))


def test_nonfinite_rows_are_not_silently_removed():
    ei = torch.ones(8, 51)
    ei[3, 7] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        fit_paired_gains(ei, torch.zeros_like(ei), torch.ones(8), [.1]*19)
