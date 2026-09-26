"""Sequence ownership of vertex memory in the recursive evaluator."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from semkine import eval_track as ET  # noqa: E402


class _FakeMano:
    hands_components = None
    hands_mean = None

    def __call__(self, betas, global_orient, local_full_aa, transl):
        # The evaluator's metrics need only correctly shaped finite coordinates.
        return transl[:, None].expand(-1, 778, -1), transl[:, None].expand(-1, 21, -1)


@pytest.fixture
def run_tracker(monkeypatch, tmp_path):
    events = np.zeros((10, 3), dtype=np.float32)
    events[:, :2] = (100, 80)
    offsets = np.arange(11, dtype=np.int64)
    poses = np.zeros((10, 51), dtype=np.float32)
    poses[:, 0] = np.arange(10)
    aux = {
        "betas": np.zeros(10, dtype=np.float32),
        "camera_K": np.eye(3, dtype=np.float32),
        "valid_runs_ms": np.array([[0, 4], [6, 10]]),
        "category": "local",
    }
    monkeypatch.setattr(ET, "load_sequence", lambda *args: (events, offsets, aux, poses))
    monkeypatch.setattr(ET, "_dbg_perstep", lambda *args: None)
    monkeypatch.setattr(ET, "decode_to_mano_inputs", lambda p, *args: {
        "global_orient": p[:, 3:6], "local_full_aa": p[:, 6:], "transl": p[:, :3],
    })

    def run(model, seq="fixture", **kwargs):
        return ET.track_sequence(
            model, _FakeMano(), {"MODEL": {"POSE_REPR": "mano_full_axis_angle"}},
            tmp_path, "val", seq, 2, torch.device("cpu"),
            np.random.default_rng(0), 0.0, **kwargs,
        )

    return run


class _MemoryModel:
    encoder_name = "event_guided_mesh"
    egm_memory = True

    def __init__(self):
        self.received_states = []
        self.emitted_states = []
        self.prev_states = []

    def track_packet(self, batch, node_state=None):
        assert not torch.is_grad_enabled()
        self.received_states.append(node_state)
        self.prev_states.append(batch.prev_state.clone())
        next_state = torch.ones(1) if node_state is None else node_state + 1
        self.emitted_states.append(next_state)
        prediction = batch.prev_state.clone()
        prediction[:, 0] += 1
        return prediction, next_state

    def forward_packet(self, batch):
        raise AssertionError("memory-enabled tracking must use track_packet")


def test_memory_is_carried_between_packets_and_reset_at_valid_run(run_tracker):
    model = _MemoryModel()
    result = run_tracker(model)
    assert result["n_frames"] == 4 and result["n_runs"] == 2
    assert model.received_states[0] is None
    assert model.received_states[1] is model.emitted_states[0]
    assert model.received_states[2] is None
    assert model.received_states[3] is model.emitted_states[2]
    # Pose recursion continues independently of the latent memory recursion.
    assert [float(p[0, 0]) for p in model.prev_states] == [0.0, 1.0, 6.0, 7.0]


def test_reusing_model_for_another_sequence_starts_with_empty_memory(run_tracker):
    model = _MemoryModel()
    run_tracker(model, seq="first")
    run_tracker(model, seq="second")
    assert model.received_states[4] is None
    assert model.received_states[5] is model.emitted_states[4]
    assert model.received_states[6] is None


@pytest.mark.parametrize("explicitly_disabled", [False, True])
def test_legacy_raw_models_keep_forward_packet_path(run_tracker, explicitly_disabled):
    class LegacyModel:
        encoder_name = "mesh_graph"

        def __init__(self):
            self.prev_states = []
            if explicitly_disabled:
                self.egm_memory = False

        def forward_packet(self, batch):
            self.prev_states.append(batch.prev_state.clone())
            prediction = batch.prev_state.clone()
            prediction[:, 0] += 1
            return prediction

        def track_packet(self, *args, **kwargs):
            raise AssertionError("legacy model must keep forward_packet")

    model = LegacyModel()
    result = run_tracker(model)
    assert result["n_frames"] == 4
    assert [float(p[0, 0]) for p in model.prev_states] == [0.0, 1.0, 6.0, 7.0]


def test_delta_trust_changes_pose_feedback_without_resetting_memory(run_tracker):
    model = _MemoryModel()
    run_tracker(model, delta_trust=0.5)
    assert [float(p[0, 0]) for p in model.prev_states] == [0.0, 0.5, 6.0, 6.5]
    assert model.received_states[1] is model.emitted_states[0]
