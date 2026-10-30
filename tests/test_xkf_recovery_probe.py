"""Independent checks for exploratory recovery sampling and physical-error scoring."""
import numpy as np
import torch

from tools.xkf.recovery_probe import METRICS, aggregate, frame_metrics, schedule


class Geometry:
    def _fk(self, state, beta):
        # A root-dependent wrist offset makes physical wrist different from t.
        wrist = state[:, :3] + state[:, 3:6]
        shape = state.new_tensor([[0., 0., 0.], [0., .03, 0.], [0., .06, 0.]])
        joints = wrist[:, None] + shape[None]
        joints[:, 1:, 0] += state[:, 6:7]
        return joints.clone(), joints


def test_physical_wrist_and_parameter_t_distinct_and_cross_term_exact():
    truth = torch.zeros(1, 51)
    pred = truth.clone()
    pred[:, 0] = .01
    pred[:, 3] = -.02
    pred[:, 6] = .03
    result, dw, dt = frame_metrics(Geometry(), pred, truth, torch.zeros(1, 10))
    score = dict(zip(METRICS, result[0].tolist()))
    assert torch.allclose(dt[0], torch.tensor([10., 0., 0.]))
    assert torch.allclose(dw[0], torch.tensor([-10., 0., 0.]))
    assert score['wrist_relative_cross_mm2'] < 0  # opposing errors cancel in absolute score
    reconstructed = (score['physical_wrist_squared_mm2'] + score['relative_joint_mse_mm2']
                     + score['wrist_relative_cross_mm2'])
    assert abs(score['absolute_joint_mse_mm2'] - reconstructed) < 1e-3


def test_schedule_keeps_short_runs_and_excludes_only_undefined_queries():
    local_rows = [{'run_id': 0, 'index': i, 'previous_end_ms': 50 * i} for i in range(5)]
    bank = {'sequences': {
        'zgz_local': {'valid_runs_ms': [[0, 250], [300, 320]], 'rows': local_rows, 'clips': []},
        'zgz_global': {'valid_runs_ms': [[0, 50]], 'rows': [], 'clips': []},
    }}
    selected, excluded = schedule(bank, 10)
    assert [c['observed_steps'] for c in selected] == [5, 2]
    assert all(c['censored'] for c in selected)
    assert selected[1]['initial_gt_ms'] == 150
    assert len(excluded) == 1 and excluded[0]['run_id'] == 1


def test_aggregation_pools_frames_and_keeps_censoring_separate():
    rows = []
    for value, length in [(2., 10), (10., 2)]:
        score = dict.fromkeys(METRICS, value)
        rows.append({'cohort': 'local_cold', 'condition': 'clean', 'mode': 'hold',
                     'observed_steps': length, 'censored': length < 10,
                     'mean': score, 'first': score, 'endpoint': score})
    result = {r['scope']: r for r in aggregate(rows)}
    assert result['all_observed']['n_frames'] == 12
    assert np.isclose(result['all_observed']['pooled_frames']['abs_mpjpe_mm'], 40 / 12)
    assert result['all_observed']['equal_trajectory']['abs_mpjpe_mm'] == 6
    assert result['full_horizon']['n_trajectories'] == 1
