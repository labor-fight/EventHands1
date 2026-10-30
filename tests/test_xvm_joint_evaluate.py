"""Synthetic evaluation mathematics/control tests; never open SELECT data."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.xvm_joint.data import PUBLIC_INITIAL_STATE, XJInputBatch
from tools.xvm_joint.evaluate import (
    ARMS, METRICS, FourArmRuntime, aggregate_recordings, association_diagnostics,
    derived_normal_means, endpoint_metrics, gate_decision, expected_scored_frames,
    validate_recording_inventory,
)
from tools.xvm_joint.model import JointLocalOperator


def record(name, *, primary=8., spatial=9., global_error=10., derived=10.,
           frames=1, a=.8, aA=1., position_weight=1.):
    values = {'primary': primary, 'spatial': spatial, 'global': global_error, 'derived': derived}
    return dict(recording=name, person=name.split('_')[0], scored_frames=frames,
                arms={arm: {metric: value if frames else None for metric in METRICS}
                      for arm, value in values.items()},
                position=dict(primary_a_mae_px=a if position_weight else None,
                              absolute_a_mae_px=aA if position_weight else None,
                              weight_total=position_weight))


def test_absolute_derived_means_use_saved_proposal_not_corrected_previous():
    reference = torch.tensor([[[1., 0.], [4., 0.]]])
    saved_absolute = torch.tensor([[[3., 0.], [7., 0.]]])
    current_absolute = torch.tensor([[[8., 0.], [9., 0.]]])
    normal = torch.tensor([[[1., 0.], [1., 0.]]])
    means = derived_normal_means(saved_absolute, reference, current_absolute, normal)
    torch.testing.assert_close(means, torch.tensor([[[2., 5.], [3., 2.]]]))
    torch.testing.assert_close(means.sum(-1), torch.tensor([[7., 5.]]))
    # Substituting emitted previous would erase a and is an invalid control.
    assert not torch.equal(means, derived_normal_means(reference, reference, current_absolute, normal))


class ShapeSensitiveBase:
    def _fk(self, poses, beta):
        vertices = poses[:, None, :3].expand(-1, 778, -1).clone()
        joints = poses[:, None, :3].expand(-1, 21, -1).clone()
        vertices[:, 0, 1] += beta[:, 0] * .02
        joints[:, 0, 1] += beta[:, 0] * .01
        return vertices, joints


def test_metric_root_is_actual_fk_joint0_and_vertex_alignments_are_distinct():
    poses = torch.zeros(4, 51)
    gt = torch.zeros(51)
    beta_target = torch.zeros(10)
    beta_target[0] = 1
    values = endpoint_metrics(ShapeSensitiveBase(), poses, gt, beta_target)
    np.testing.assert_allclose(values['root_abs_mm'], 10., atol=1e-6)
    np.testing.assert_allclose(values['joint_abs_mm'], 10./21, atol=1e-6)
    np.testing.assert_allclose(values['vertex_abs_mm'], 20./778, atol=1e-6)
    # Target vertex0 moves20mm; actual wrist moves10mm. A mislabeled alignment
    # would make these equal and could hide a root/shape error.
    np.testing.assert_allclose(values['vertex_v0_mm'], 20.*777/778, atol=1e-5)
    np.testing.assert_allclose(values['vertex_j0_mm'], 10., atol=1e-5)


def test_person_macro_equally_weights_recordings_not_frame_counts():
    rows = [record('ycy_local', primary=1., frames=1, a=.2),
            record('ycy_global', primary=9., frames=99, a=.8)]
    summary = aggregate_recordings(rows, person='ycy')
    assert summary['arms']['primary']['joint_abs_mm'] == 5.
    assert summary['a_mae_primary_px'] == .5
    assert summary['scored_frames'] == 100
    assert summary['scored_frame_coverage_complete']


def test_frozen_gate_boundaries_and_strict_association_inequality():
    rows = [record('ycy_local', primary=10.5, spatial=10.6, a=.9),
            record('ylf_local', primary=10.5, spatial=10.6, a=.9)]
    persons = {p: aggregate_recordings([r for r in rows if r['person'] == p], person=p)
               for p in ('ycy', 'ylf')}
    passed = gate_decision(persons, rows)
    assert passed['G1_position'] and passed['G2_noninferiority'] and passed['G3_association']
    assert passed['decision'] == 'CONTINUE_PERMITTED'
    for row in rows:
        row['arms']['spatial']['joint_abs_mm'] = 10.5
    persons = {p: aggregate_recordings([r for r in rows if r['person'] == p], person=p)
               for p in ('ycy', 'ylf')}
    equal = gate_decision(persons, rows)
    assert equal['G3_association'] is False and equal['decision'] == 'STOP'


def test_position_na_blocks_only_relevant_gate_and_missing_frames_block_endpoints():
    rows = [record('ycy_local', position_weight=0.), record('ylf_local')]
    persons = {p: aggregate_recordings([r for r in rows if r['person'] == p], person=p)
               for p in ('ycy', 'ylf')}
    verdict = gate_decision(persons, rows)
    assert verdict['G1_position'] is None
    assert verdict['G2_noninferiority'] is True
    assert verdict['G3_association'] is True
    assert verdict['insufficient'] and verdict['decision'] == 'STOP'
    rows.append(record('ylf_global', frames=0, position_weight=0.))
    persons = {p: aggregate_recordings([r for r in rows if r['person'] == p], person=p)
               for p in ('ycy', 'ylf')}
    incomplete = gate_decision(persons, rows)
    assert incomplete['G2_noninferiority'] is None and incomplete['G3_association'] is None
    assert persons['ylf']['recordings'] == 2  # Missing source was not silently dropped.


def test_per_source_entropy_reports_unknown_separately_and_keeps_inactive_sources():
    aux = dict(active=torch.tensor([[True, True, True, False]]),
               association_mass=torch.tensor([[.125, .125, .25, 0.]]), source_count=torch.tensor([2.]),
               unknown_score=torch.tensor([[.5, .5, .5, 1.]]),
               validity_score=torch.ones(1, 4))
    diag = association_diagnostics(aux, torch.tensor([[0, 0, 1, 2]]),
                                   torch.tensor([[True, True, True, False]]))
    assert diag['source_supports'] == 3 and diag['active_sources'] == 2
    assert diag['source_coverage'] == 2/3
    assert diag['unknown_score_mean'] == .5
    np.testing.assert_allclose(diag['candidate_conditional_entropy_mean'], np.log(2)/2, rtol=1e-6)
    assert diag['per_source'][-1]['active'] is False


class RolloutBase(torch.nn.Module):
    def forward(self, images, previous, *, betas, camera_K):
        assert not betas.any()
        result = previous.clone()
        result[:, 0] += .01
        return result

    def _fk(self, poses, beta):
        vertices = poses[:, None, :3].expand(-1, 778, -1)
        joints = poses[:, None, :3].expand(-1, 21, -1)
        return vertices, joints


class RolloutGeometry:
    def prepare(self, f0, f1, q0, q1, K, both_exposed=None):
        B, V = len(q0), 2
        assert B == 3  # Corrected states of the three local arms are separate.
        uv0 = (q0[:, :2]/q0[:, 2:3])[:, None].expand(-1, V, -1).clone()
        uv1 = (q1[:, :2]/q1[:, 2:3])[:, None].expand(-1, V, -1).clone()
        normal = torch.zeros(B, V, 2);normal[..., 0] = 1
        jacobian = torch.zeros(B, V, 2, 51);jacobian[..., 0, 0] = 1/q0[:, 2, None]
        observed = both_exposed[:, None].expand(-1, V).clone()
        kwargs = dict(patches=torch.ones(B, V, 37, 16, 16), rho=torch.ones(B, V), observed=observed,
                      B0=jacobian, B1=jacobian, normal=normal, nominal_delta=(uv1-uv0)*observed[..., None],
                      source_ids=torch.zeros(B, V, dtype=torch.long), candidate_mask=observed,
                      spatial_logits=torch.tensor([[0., -2.]]).expand(B, -1))
        return SimpleNamespace(**kwargs, model_kwargs=kwargs, K_event=torch.eye(3)[None].expand(B, -1, -1),
                               vertex_ids=torch.zeros(B, V, dtype=torch.long), uv0=uv0, uv1=uv1)


def rollout_batch(start, *, reset=False, empty=False):
    events = np.empty((0, 4), np.int64) if empty else np.asarray([[20, 30, start*1000+999, 0]], np.int64)
    item = dict(lane=0, valid=True, padding=False, supervised=True, reset=reset,
                recording='synthetic_local', run_id=0, start_ms=start, cutoff_ms=start+50,
                field0=np.zeros((16, 180, 240), np.float32), field1=np.zeros((16, 180, 240), np.float32),
                beta_input=np.zeros(10, np.float32), initial_state=PUBLIC_INITIAL_STATE.copy(),
                K_source=np.eye(3, dtype=np.float32), both_snapshots_valid=True, events=events)
    return XJInputBatch((item,), 'synthetic-only-token', start//50)


def test_four_arms_recur_independently_and_empty_event_keeps_each_own_state():
    local = JointLocalOperator()
    with torch.no_grad():local.mean_head.bias.copy_(torch.tensor([1., .5]))
    runtime = FourArmRuntime(RolloutBase(), local, RolloutGeometry(), 'cpu')
    first = runtime.forward(rollout_batch(100, reset=True))
    assert first['all_arms_forward_complete'] and first['inputs_complete']
    assert runtime.previous is None  # No arm commits before the caller scores it.
    assert first['predictions'][0, 0] != first['predictions'][2, 0]
    torch.testing.assert_close(first['predictions'][3], first['absolute'][3])
    runtime.commit(first)
    second = runtime.forward(rollout_batch(150))
    torch.testing.assert_close(second['previous'], first['predictions'])
    torch.testing.assert_close(second['absolute_previous'], first['absolute'])
    assert second['absolute'][0, 0] != second['absolute'][2, 0]
    runtime.commit(second)
    empty = runtime.forward(rollout_batch(200, empty=True))
    torch.testing.assert_close(empty['predictions'], second['predictions'])


def test_expected_frames_remove_warmup_per_run_and_ignore_partial_packets():
    runs = [SimpleNamespace(start_ms=0, stop_ms=275),
            SimpleNamespace(start_ms=300, stop_ms=450),
            SimpleNamespace(start_ms=500, stop_ms=599)]
    assert expected_scored_frames(runs, warmup=2) == 4
    assert expected_scored_frames([(0, 100), (200, 349)], warmup=2) == 0


def test_shard_inventory_allows_repeated_person_but_no_missing_overlap_or_wrong_identity():
    expected = {'ycy': ['ycy_local', 'ycy_global'], 'ylf': ['ylf_local']}
    rows = [record('ycy_local'), record('ylf_local'), record('ycy_global')]
    validate_recording_inventory(rows, expected)
    for invalid in (rows[:-1], rows+[rows[0]], rows+[record('ylf_other')]):
        with pytest.raises(RuntimeError, match='inventory'):
            validate_recording_inventory(invalid, expected)
    changed = [dict(row) for row in rows]
    changed[0]['person'] = 'ylf'
    with pytest.raises(RuntimeError, match='identity'):
        validate_recording_inventory(changed, expected)
