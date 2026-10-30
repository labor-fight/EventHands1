"""Synthetic CPU contracts for XJ event fields and label-free input lanes.

These fixtures do not open the hand dataset or any experiment output.
"""
import json

import numpy as np
import pytest

from tools.xvm_joint.data import (
    HEIGHT, WIDTH, FIELD_CHANNELS, XJRunLanes, XJSupervisionReader,
    advance_field16, stack_inputs,
)


def test_field16_preserves_polarity_bin_boundaries_and_latest_timestamp():
    state = np.full((2, 3, 4), -1, np.int64)
    events = np.asarray([[1, 1, 0, 0], [1, 1, 9999, 0], [1, 1, 10000, 0],
                         [1, 1, 49999, 1]], np.int64)
    field, updated, meta = advance_field16(state, events, start_ms=0, cutoff_ms=50, run_start_ms=0)
    assert field.shape == (16, 3, 4)
    assert len(FIELD_CHANNELS) == 16
    assert state.max() == -1  # Transactional input state is unchanged.
    assert updated[0, 1, 1] == 10000 and updated[1, 1, 1] == 49999
    np.testing.assert_allclose(field[0, 1, 1], np.exp(-40 / 10), rtol=1e-6)
    np.testing.assert_allclose(field[1, 1, 1], np.exp(-.001 / 10), rtol=1e-6)
    assert field[6, 1, 1] == 2  # first10ms, stored polarity0
    assert field[8, 1, 1] == 1  # second10ms
    assert field[15, 1, 1] == 1  # last10ms, polarity1
    assert field[6:].sum() == 4
    assert not field.flags.writeable and not updated.flags.writeable
    assert meta['event_count'] == 4 and not meta['current_window_empty']
    permuted, same_state, _ = advance_field16(state, events[[3, 1, 0, 2]], start_ms=0, cutoff_ms=50, run_start_ms=0)
    np.testing.assert_array_equal(permuted, field)
    np.testing.assert_array_equal(same_state, updated)


def test_history_expires_without_turning_an_empty_window_into_no_exposure():
    state = np.full((2, 1, 1), -1, np.int64)
    _, state, _ = advance_field16(state, np.asarray([[0, 0, 0, 0]], np.int64),
                                 start_ms=0, cutoff_ms=50, run_start_ms=0)
    for start in range(50, 650, 50):
        field, state, meta = advance_field16(state, np.empty((0, 4), np.int64),
                                             start_ms=start, cutoff_ms=start + 50, run_start_ms=0)
    assert not field.any()  # age650ms is outside600ms, latest is retained only as an index.
    assert meta['history_exposure_ms'] == 600
    assert meta['current_window_empty'] and meta['current_window_observed']


def test_future_events_and_duplicate_cursor_consumption_are_rejected():
    state = np.full((2, 1, 1), -1, np.int64)
    with pytest.raises(ValueError, match='half-open'):
        advance_field16(state, np.asarray([[0, 0, 50000, 0]], np.int64),
                        start_ms=0, cutoff_ms=50, run_start_ms=0)
    _, updated, _ = advance_field16(state, np.asarray([[0, 0, 49000, 0]], np.int64),
                                     start_ms=0, cutoff_ms=50, run_start_ms=0)
    with pytest.raises(ValueError, match='already consumed'):
        advance_field16(updated, np.empty((0, 4), np.int64),
                        start_ms=0, cutoff_ms=50, run_start_ms=0)


def test_raw_and_log_counts_are_explicit_different_contracts():
    state = np.full((2, 1, 1), -1, np.int64)
    events = np.asarray([[0, 0, 1000, 0], [0, 0, 1000, 0]], np.int64)
    raw, _, _ = advance_field16(state, events, start_ms=0, cutoff_ms=50, run_start_ms=0)
    logged, _, meta = advance_field16(state, events, start_ms=0, cutoff_ms=50, run_start_ms=0, log_counts=True)
    assert raw[6, 0, 0] == 2
    np.testing.assert_allclose(logged[6, 0, 0], np.log(3), rtol=1e-6)
    assert meta['log_counts']


class FakeInputReader:
    def __init__(self, recording, root):
        self.recording = recording
        self.runs = np.asarray([[0, 250]], np.int64)
        self.K_source = np.asarray([[400, 0, 320], [0, 400, 240], [0, 0, 1]], np.float32)

    def packet(self, start, cutoff):
        # A real exposed but empty packet would also be legal. This one is
        # deliberately distinct at each step so the saved t0 snapshot is tested.
        events = np.asarray([[10 + start // 50, 20, start * 1000 + 999, 0]], np.int64)
        events.flags.writeable = False
        return dict(events=events, source_range=(start // 50, start // 50 + 1))


@pytest.fixture
def synthetic_root(tmp_path):
    # Artificial fixture only; this does not create or replace a project split.
    (tmp_path / 'splits_semkine.json').write_text(json.dumps({
        'train': {'trials': ['ch_local'], 'legacy_dir': {'ch_local': 'train'}},
        'val_core': {'trials': ['zgz_local'], 'legacy_dir': {'zgz_local': 'val'}},
    }))
    return tmp_path


def test_lanes_are_transactional_chronological_and_never_contain_targets(synthetic_root):
    lanes = XJRunLanes(synthetic_root, subjects=('ch',), batch_size=1,
                       reader_factory=FakeInputReader, finite=True)
    first = lanes.peek_batch()
    assert lanes.peek_batch() is first
    assert first[0]['reset'] and not first[0]['supervised']
    assert not first[0]['field0'].any()
    assert not first[0]['both_snapshots_valid']
    saved_first = first[0]['field1'].copy()
    assert not set(first[0]) & {'target0', 'target1', 'target', 'beta_target', 'betas'}
    lanes.commit_batch(first)
    with pytest.raises(ValueError, match='duplicate'):
        lanes.commit_batch(first)
    second = lanes.peek_batch()
    assert second[0]['start_ms'] == 50 and second[0]['cutoff_ms'] == 100
    np.testing.assert_array_equal(second[0]['field0'], saved_first)
    assert second[0]['both_snapshots_valid'] and not second[0]['supervised']
    packed = stack_inputs(second)
    assert packed['field0'].shape == (1, 16, HEIGHT, WIDTH)
    assert not packed['beta_input'].any()
    lanes.commit_batch(second)
    third = lanes.peek_batch()
    assert third[0]['supervised']
    assert (third[0]['reference_ms'], third[0]['target_ms']) == (99, 149)


def test_label_owner_is_lazy_and_access_requires_complete_inputs(synthetic_root):
    openings = []

    class FakeSupervisionReader:
        def __init__(self, recording, root):
            openings.append(recording)
            self.beta_target = np.full(10, .125, np.float32)

        def target(self, timestamp):
            return np.full(51, timestamp, np.float32)

    lanes = XJRunLanes(synthetic_root, subjects=('ch',), batch_size=1,
                       reader_factory=FakeInputReader)
    labels = XJSupervisionReader(synthetic_root, reader_factory=FakeSupervisionReader)
    first = lanes.peek_batch()
    with pytest.raises(RuntimeError, match='before supervision'):
        labels.read(first, inputs_complete=False)
    assert not openings
    warmup = labels.read(first, inputs_complete=True)
    assert not warmup.mask.any() and not openings
    lanes.commit_batch(first)
    second = lanes.peek_batch()
    lanes.commit_batch(second)
    third = lanes.peek_batch()
    supervision = labels.read(third, inputs_complete=True)
    assert openings == ['ch_local'] and labels.reads == 1
    assert supervision.mask.tolist() == [True]
    np.testing.assert_array_equal(supervision.target0[0], np.full(51, 99))
    np.testing.assert_array_equal(supervision.target1[0], np.full(51, 149))
    np.testing.assert_allclose(supervision.beta_target, .125)
    assert supervision.input_token == third.token
    assert not supervision.target1.flags.writeable


def test_schedule_checkpoint_reproduces_next_event_fields(synthetic_root):
    kwargs = dict(subjects=('ch',), batch_size=1, reader_factory=FakeInputReader)
    a = XJRunLanes(synthetic_root, **kwargs)
    first = a.peek_batch()
    with pytest.raises(RuntimeError, match='commit'):
        a.state_dict()
    a.commit_batch(first)
    saved = a.state_dict()
    b = XJRunLanes(synthetic_root, **kwargs)
    b.load_state_dict(saved)
    na, nb = a.peek_batch(), b.peek_batch()
    assert na.token == nb.token
    np.testing.assert_array_equal(na[0]['field0'], nb[0]['field0'])
    np.testing.assert_array_equal(na[0]['field1'], nb[0]['field1'])


def test_report_subject_and_out_of_manifest_recordings_are_refused(synthetic_root):
    with pytest.raises(ValueError, match='never zgz'):
        XJRunLanes(synthetic_root, subjects=('zgz',), reader_factory=FakeInputReader)
    with pytest.raises(ValueError, match='outside'):
        XJRunLanes(synthetic_root, subjects=('ch',), recordings=('invented_local',), reader_factory=FakeInputReader)
