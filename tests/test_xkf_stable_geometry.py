"""Real frozen-model checks for opt-in single-sample inference semantics."""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]

from semkine.xkf_stable_geometry import XKFStableGeometryModel
from semkine.xyz_recovery_study import select_packet_rows
from tools.xkf_fusion import batch_probe as B


@pytest.fixture(scope='module')
def actual():
    torch.set_num_threads(4)
    B.Q.XKFFusionModel = XKFStableGeometryModel
    model, cfg, _ = B.load(3408, {}, torch.device('cpu'))
    bank, _ = B.Q.R.D.load_bank()
    meta = bank['sequences']['zgz_local']
    data = B.Q.R.D.load_data(cfg, 'zgz_local', meta)
    cache = B.Q.PacketCache(model, data, torch.device('cpu'))
    clip = B.Q.R.schedule(bank, 10)[0][0]
    initial = B.Q.R.initial_states(torch.from_numpy(data['gt'][clip['initial_gt_ms']]).float())
    packet, _ = cache.packet(meta['rows'][clip['row_indices'][0]], initial[:2], [1.])
    return model, packet


def test_default_off_and_state_keys_unchanged(actual):
    model, packet = actual
    keys = tuple(model.state_dict())
    model.configure_stability('off')
    with torch.no_grad():
        a = model.incremental_prediction(packet)
        b = super(XKFStableGeometryModel, model).incremental_prediction(packet)
    assert torch.equal(a, b)
    for mode in ('per_sample_geometry', 'per_sample_i', 'per_sample_model', 'off'):
        model.configure_stability(mode)
        assert tuple(model.state_dict()) == keys


def test_geometry_exact_same_single_sample_rule(actual):
    model, packet = actual
    model.xyz_keep_aux = True
    model.configure_stability('per_sample_geometry')
    with torch.no_grad():
        model.incremental_prediction(packet)
        combined = model.xyz_last
        for row in range(packet.batch_size):
            one = select_packet_rows(packet, torch.tensor([row]))
            model.configure_stability('off')
            model.incremental_prediction(one)
            original = model.xyz_last
            for key in ('vid', 'hit', 'valid', 'A', 'z', 'q'):
                assert torch.equal(combined['assoc'][key][row:row + 1], original['assoc'][key]), key
            assert torch.equal(combined['verts'][row:row + 1], original['verts'])


def test_complete_i_is_batch_permutation_invariant(actual):
    model, packet = actual
    model.configure_stability('per_sample_i')
    with torch.no_grad():
        combined = model.incremental_prediction(packet)
        separate = torch.cat([model.incremental_prediction(select_packet_rows(packet, torch.tensor([i])))
                              for i in range(packet.batch_size)])
        reverse = model.incremental_prediction(select_packet_rows(packet, torch.tensor([1, 0])))
    assert torch.equal(combined, separate)
    assert torch.equal(combined, reverse.flip(0))


def test_complete_model_calls_actual_uncached_d_and_is_batch_invariant(actual, monkeypatch):
    model, packet = actual
    model.configure_stability('per_sample_model')
    model.configure_fusion({'id': 'test', 'gain_scale': [.5, .25, .5],
                            'translation_reference': 'joint_centroid', 'count_gate': 'ramp1000'})
    calls = []
    original = model.measure_direct
    def direct(batch, *args, **kwargs):
        calls.append(batch.batch_size)
        assert not bool(batch.target.any())
        return original(batch, *args, **kwargs)
    monkeypatch.setattr(model, 'measure_direct', direct)
    with torch.no_grad():
        combined = model.forward_packet(packet)
        separate = torch.cat([model.forward_packet(select_packet_rows(packet, torch.tensor([i])))
                              for i in range(packet.batch_size)])
    assert calls == [1, 1, 1, 1]
    assert torch.equal(combined, separate)


def test_stability_is_explicit_inference_only(actual):
    model, packet = actual
    with pytest.raises(ValueError, match='mode'):
        model.configure_stability('rounded')
    model.configure_stability('per_sample_geometry').train()
    with pytest.raises(ValueError, match='inference-only'):
        model.incremental_prediction(packet)
    model.eval().configure_stability('off').configure_fusion(None)
