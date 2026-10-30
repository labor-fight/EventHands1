"""Equation/causality audits for fixed A/Z, with no dataset or model loading."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.s43_rel.core import RelConfig, RelationMemory, all_pair_reference


def cfg(**kw):
    return RelConfig(**dict(dict(width=8, height=6, cell_size=2, dtype="float64", backend="numpy"), **kw))


def unchanged_observation(a, b):
    for name in a["arrays"]:
        assert np.array_equal(a["arrays"][name], b["arrays"][name]), name
    assert a["metadata"] == b["metadata"]


@pytest.mark.parametrize("backend", ["numpy", "cpp"])
@pytest.mark.parametrize("unit", ["ms", "us"])
def test_recurrence_equals_independent_event_pair_sum_at_every_prefix(backend, unit):
    c = cfg(backend=backend, time_unit=unit)
    e = np.array([[0, 0, 0, 1], [2, 0, 0, -1], [2, 0, 1, 1],
                  [0, 0, 1, -1], [2, 0, 1, 1], [4, 2, 2, -1],
                  [0, 0, 3, 1], [3, 0, 3, -1], [7, 5, 4, 1]], dtype=np.int64)
    scale = 1000 if unit == "us" else 1
    e[:, 2] *= scale
    m = RelationMemory(c, "subject/segment")
    begin = 0
    for ordinal, end_ms in enumerate([1, 2, 3, 4, 5]):
        end = np.searchsorted(e[:, 2], end_ms * scale)
        m.ingest(e[begin:end], end_ms, ordinal, source_range=(begin, end))
        expected_a, expected_z = all_pair_reference(e, c, end_ms, unit)
        actual = m.readout(end_ms)
        np.testing.assert_allclose(actual.A, expected_a, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(actual.Z, expected_z, rtol=2e-14, atol=2e-14)
        begin = end


@pytest.mark.parametrize("backend", ["numpy", "cpp"])
def test_same_timestamp_is_atomic_and_real_multiplicity_is_preserved(backend):
    c = cfg(backend=backend)
    e = np.array([[0, 0, 0, 1]] * 2 + [[2, 0, 0, -1]] * 3)
    m = RelationMemory(c)
    m.ingest(e, 1, 0)
    assert np.count_nonzero(m.readout(1).Z) == 0
    # True repetitions at source/target contribute 2 * 3 cross-time pairs.
    n = RelationMemory(c)
    e[2:, 2] = 1
    n.ingest(e, 2, 0)
    s = n.readout(2)
    edge = np.flatnonzero((n.topology.src == 0) & (n.topology.dst == 1))[0]
    expected = 6 * np.exp(-1 / np.asarray(c.tau_ms))[:, None] * np.exp(-1 / np.asarray(c.gamma_ms))[None, :]
    np.testing.assert_allclose(s.Z[edge, 1, 0], expected, atol=1e-14)
    # Within-time event storage order cannot manufacture direction.
    perm = e[[1, 0, 4, 2, 3]]
    q = RelationMemory(c)
    q.ingest(perm, 2, 0)
    np.testing.assert_array_equal(q.readout(2).Z, s.Z)


@pytest.mark.parametrize("backend", ["numpy", "cpp"])
def test_arbitrary_chunks_including_unfinished_group_equal_single_stream(backend):
    c = cfg(backend=backend)
    e = np.array([[0, 0, 0, 1], [2, 0, 0, -1], [2, 0, 0, -1],
                  [0, 0, 1, 1], [2, 0, 1, 1], [4, 0, 3, -1]])
    whole, split = RelationMemory(c), RelationMemory(c)
    whole.ingest(e, 4, 0)
    split.ingest(e[:1], 0, 0, source_range=(0, 1))
    split.ingest(e[1:2], 0, 1, source_range=(1, 2))
    assert not split.readout(0).A.any()  # staged timestamp zero is excluded
    split.ingest(e[2:4], 1, 2, source_range=(2, 4))
    split.ingest(e[4:5], 2, 3, source_range=(4, 5))
    split.ingest([], 3, 4, source_range=(5, 5))
    split.ingest(e[5:], 4, 5, source_range=(5, 6))
    a, b = whole.readout(4), split.readout(4)
    np.testing.assert_array_equal(a.A, b.A)
    np.testing.assert_array_equal(a.Z, b.Z)
    assert split.stats()["events"] == len(e)


def test_empty_prefix_pure_lazy_queries_and_empty_gap_decay():
    m = RelationMemory(cfg(), "s")
    assert not m.readout(0).A.any()
    assert not m.readout(0).Z.any()
    e = np.array([[0, 0, 0, 1], [2, 0, 1, -1]])
    m.ingest(e, 2, 0)
    snap = m.readout(2)
    state = m.export_state()
    for _ in range(3):
        np.testing.assert_array_equal(m.readout(2).Z, snap.Z)
    unchanged_observation(state, m.export_state())
    assert not snap.A.flags.writeable and not snap.Z.flags.writeable
    m.ingest([], 12, 1)
    end = m.readout(12)
    np.testing.assert_allclose(end.A, snap.A * np.exp(-10 / np.array(m.config.tau_ms)))
    np.testing.assert_allclose(end.Z, snap.Z * np.exp(-10 / np.array(m.config.gamma_ms)))
    assert m.stats()["sealed_groups"] == 2


def test_source_contract_guards_are_atomic_and_distinct_sources_repeat_payload():
    m = RelationMemory(cfg(), "s")
    e = np.array([[0, 0, 0, 1]])
    m.ingest(e, 0, 2, source_range=(0, 1))
    old = m.export_state()
    for kwargs in [dict(packet_ordinal=2, source_range=(1, 2)),
                   dict(packet_ordinal=3, source_range=(0, 1)),
                   dict(packet_ordinal=3, source_range=(2, 5))]:
        with pytest.raises(ValueError):
            m.ingest(e, 0, **kwargs)
        unchanged_observation(old, m.export_state())
    m.ingest(e, 1, 3, source_range=(1, 2))
    assert m.controls(1, coarse_factor=1).count[0, 1] == 2
    with pytest.raises(ValueError, match="source-range"):
        m.ingest([], 1, 4)  # cannot drop the declared source contract


def test_future_late_and_consumed_past_guards_and_missing_stream():
    m = RelationMemory(cfg())
    with pytest.raises(ValueError, match="future"):
        m.ingest([[0, 0, 2, 1]], 1, 0)
    m.ingest([[0, 0, 0, 1]], 1, 0)
    for events, wm in [([[0, 0, 0, 1]], 2), ([], 0)]:
        with pytest.raises(ValueError):
            m.ingest(events, wm, 1)
    with pytest.raises(ValueError, match="past half-open"):
        m.readout(0)
    with pytest.raises(ValueError, match="complete known prefix"):
        m.readout(2)
    m.ingest([], None, 1)
    assert m.readout(1).missing
    m.ingest([[2, 0, 3, 1]], None, 2)
    assert m.stats()["pending_events"] == 1
    with pytest.raises(ValueError, match="staging"):
        m.ingest([[2, 0, 4, 1]], None, 3)
    m.ingest([], 4, 3)
    assert not m.readout(4).missing


def test_watermark_inside_native_ms_bin_cannot_pretend_group_is_complete():
    m = RelationMemory(cfg())
    m.ingest([[0, 0, 0, 1]], .5, 0)
    with pytest.raises(ValueError, match="native timestamp bin"):
        m.readout(.5)
    # Quantized stamp zero can still receive true multiplicity before bin seal.
    m.ingest([[2, 0, 0, -1]], 1, 1)
    assert not m.readout(1).Z.any()
    assert m.controls(1, 1).count.sum() == 2


def test_native_structured_us_and_malformed_payloads():
    c = cfg(time_unit="us")
    e = np.array([(0, 0, 1, True), (2, 0, 2, False), (2, 0, 2, False)],
                 dtype=[("x", "i2"), ("y", "i2"), ("t", "i8"), ("p", "?")])
    m = RelationMemory(c)
    m.ingest(e, .003, 0)
    a, z = all_pair_reference(e, c, .003)
    np.testing.assert_allclose(m.readout(.003).A, a)
    np.testing.assert_allclose(m.readout(.003).Z, z)
    for bad in ([[0, 0, np.nan, 1]], [[8, 0, 4, 1]], [[0, 0, .5, 1]],
                [[0, 0, 4, 2]], [[0, 0, 5, 1], [2, 0, 4, 1]]):
        with pytest.raises(ValueError):
            m.ingest(bad, .006, 1)
    with pytest.raises(ValueError, match="units"):
        m.ingest([], 1, 1, time_unit="ms")


def test_sequence_isolation_padding_release_and_checkpoint_with_staging():
    a, b = RelationMemory(cfg(), "a"), RelationMemory(cfg(), "b")
    # Padding must not inspect garbage payload or mixed sequence ID.
    a.ingest(object(), packet_ordinal=None, valid=False, seq_id="b")
    assert a.version == 0
    with pytest.raises(ValueError, match="mixed sequence"):
        a.ingest([], 1, 0, seq_id="b")
    a.ingest([[0, 0, 0, 1], [2, 0, 1, -1]], 1, 0)
    state = a.export_state()
    restored = RelationMemory.from_state(state)
    for m in [a, restored]:
        m.ingest([[2, 0, 1, -1]], 2, 1)
    np.testing.assert_array_equal(a.readout(2).Z, restored.readout(2).Z)
    assert not b.readout(0).A.any()
    with pytest.raises(ValueError, match="identity"):
        RelationMemory(replace(cfg(), tau_ms=(4., 8., 32.))).load_state_dict(state)
    a.release("a")
    with pytest.raises(RuntimeError, match="released"):
        a.readout(2)
    assert not b.readout(0).Z.any()


def test_commit_checks_snapshot_freshness_and_does_not_write_observation():
    m = RelationMemory(cfg(), "s")
    m.ingest([[0, 0, 0, 1]], 1, 0)
    q = m.readout(1)
    before = m.export_state()["arrays"]
    m.commit(q)
    for k, v in before.items():
        np.testing.assert_array_equal(v, m.export_state()["arrays"][k])
    assert m.commit_count == 1
    with pytest.raises(ValueError, match="one commit"):
        m.commit(q)
    q = m.readout(1)
    m.ingest([], 2, 1)
    with pytest.raises(ValueError, match="stale"):
        m.commit(q)


def test_default_topology_shapes_reverse_and_partial_query():
    m = RelationMemory(RelConfig(backend="numpy"))
    t = m.topology
    assert m._A.shape == (10800, 2, 3)
    assert m._Z.shape == (169040, 2, 2, 3, 2)
    np.testing.assert_array_equal(t.reverse[t.reverse], np.arange(t.n_edges))
    np.testing.assert_array_equal(t.src[t.reverse], t.dst)
    np.testing.assert_allclose(t.delta_px[t.reverse], -t.delta_px)
    np.testing.assert_allclose(t.centers_px[0], [.5, .5])
    m.ingest([[0, 0, 0, 1], [2, 0, 1, 0]], 2, 0)
    ids = np.array([10, 1, 2])
    np.testing.assert_array_equal(m.readout(2, ids, False).Z, m.readout(2).Z[ids])
    q = m.readout(2, [], False)
    assert q.A is None and q.Z.shape[0] == 0
    with pytest.raises(ValueError):
        m.readout(2, [1, 1])


def test_long_stream_memory_is_constant_and_no_hidden_event_ledger():
    m = RelationMemory(cfg())
    first_size = m.stats()["resident_bytes"]
    for t in range(200):
        m.ingest([[2 * (t % 4), 0, t, t % 2]], t + 1, t,
                 source_range=(t, t + 1))
    assert m.stats()["resident_bytes"] == first_size
    assert m.stats()["source_ledger_entries"] == 1
    assert m.stats()["events"] == 200
    assert int(m.controls(200, 1).count.sum()) == 200
    # Checkpoint length does not depend on stream duration.
    assert set(m.export_state()["arrays"]) == set(m._array_names())


def test_float32_cpp_and_numpy_agree_for_random_actual_multiplicity():
    rng = np.random.default_rng(12)
    c = cfg(dtype="float32")
    n = 4000
    e = np.column_stack((rng.integers(0, 8, n), rng.integers(0, 6, n),
                         np.sort(rng.integers(0, 80, n)), rng.integers(0, 2, n)))
    a, b = RelationMemory(c), RelationMemory(replace(c, backend="cpp"))
    for m in (a, b):
        m.ingest(e, 80, 0)
    np.testing.assert_allclose(a.readout(80).A, b.readout(80).A, rtol=2e-6, atol=1e-5)
    np.testing.assert_allclose(a.readout(80).Z, b.readout(80).Z, rtol=2e-6, atol=1e-4)
