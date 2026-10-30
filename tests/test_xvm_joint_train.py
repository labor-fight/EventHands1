"""CPU-only synthetic trainer contracts; no real recordings or weights opened."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from tools.xvm_joint.data import PUBLIC_INITIAL_STATE, XJInputBatch
from tools.xvm_joint.model import JointLocalOperator
from tools.xvm_joint.train import (
    Stage1Runtime, camera_valid, native_lnes_ms, reference_perturbation, source_record_weights,
    require_real_gpu_engineering,
)


def test_native_lnes_uses_original_ms_zero_code_not_us_fraction():
    events = np.asarray([[5, 6, 100000, 0], [5, 6, 149999, 0],
                         [7, 8, 100999, 1]], np.int64)
    image = native_lnes_ms(events, 100, 150)
    assert image.shape == (180, 240, 2)
    assert image[6, 5, 0] == np.float32(49 / 50)
    assert image[8, 7, 1] == 0
    np.testing.assert_array_equal(image, native_lnes_ms(events[::-1], 100, 150))
    with pytest.raises(ValueError, match='causal'):
        native_lnes_ms(np.asarray([[5, 6, 150000, 0]], np.int64), 100, 150)


def test_fixed_source_weights_do_not_manufacture_supervision_from_candidates():
    ids = torch.tensor([[7, 7, 7, 8, -1]])
    weights = source_record_weights(ids, torch.ones_like(ids, dtype=torch.bool))
    torch.testing.assert_close(weights, torch.tensor([[1/3, 1/3, 1/3, 1., 0.]]))
    assert not weights.requires_grad
    torch.testing.assert_close(weights.sum(), torch.tensor(2.))


def test_direct_train_engineering_gate_requires_exact_freeze_and_actual_backward(tmp_path):
    config = tmp_path / 'config.json'
    with pytest.raises(RuntimeError, match='required before training'):
        require_real_gpu_engineering(config, 'frozen-identity')
    folder = tmp_path / 'engineering'
    folder.mkdir()
    path = folder / 'real_gpu.json'
    receipt = dict(status='PASS', sourcefreeze_sha256='other-freeze', actual_backward_calls=1, optimizer_steps=0)
    path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match='exact source freeze'):
        require_real_gpu_engineering(config, 'frozen-identity')
    receipt['sourcefreeze_sha256'] = 'frozen-identity'
    receipt['actual_backward_calls'] = 0
    path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match='actual backward'):
        require_real_gpu_engineering(config, 'frozen-identity')
    receipt['actual_backward_calls'] = 1
    path.write_text(json.dumps(receipt))
    assert require_real_gpu_engineering(config, 'frozen-identity') == path


def test_reference_perturbation_reads_registered_keys_and_respects_clip():
    torch.manual_seed(3407)
    previous = torch.zeros(128, 51)
    previous[:, 2] = .6
    config = dict(reference_perturb_probability=1., reference_translation_std_m=.01,
                  reference_root_std_rad=.08, reference_fingers_std_rad=.03,
                  reference_clip_sigma=2.)
    reference, selected, noise = reference_perturbation(previous, config)
    assert selected.all()
    assert noise[:, :3].abs().sum() > 0
    assert noise[:, 3:6].abs().sum() > 0
    assert noise[:, 6:].abs().sum() > 0
    assert noise[:, :3].abs().max() <= .02000001
    assert noise[:, 3:6].abs().max() <= .16000001
    assert noise[:, 6:].abs().max() <= .06000001
    torch.testing.assert_close(reference, previous + noise)
    assert previous[:, 2].eq(.6).all()  # intervention never changes feedback in-place
    assert not previous[:, :2].any()


class FakeBase(torch.nn.Module):
    def __init__(self, order=None):
        super().__init__()
        self.register_parameter('frozen', torch.nn.Parameter(torch.tensor(1.), requires_grad=False))
        self.order = order

    def forward(self, lnes, previous, *, betas, camera_K):
        assert not betas.any()
        if self.order is not None:
            self.order.append('absolute_forward')
        delta = torch.zeros_like(previous)
        delta[:, 0] = .01
        return previous + delta

    def _fk(self, poses, betas):
        joints = poses[:, :3, None].transpose(1, 2).expand(-1, 21, -1)
        return joints, joints


def test_camera_guard_distinguishes_invalid_depth_from_nonfinite_state():
    base = FakeBase()
    states = torch.zeros(2, 51)
    states[:, 2] = torch.tensor([.6, -.1])
    assert camera_valid(base, states, torch.zeros(2, 10)).tolist() == [True, False]
    states[0, 2] = float('nan')
    with pytest.raises(ArithmeticError, match='Nonfinite'):
        camera_valid(base, states, torch.zeros(2, 10))


def test_runtime_forward_precedes_all_supervision_and_backward_reaches_local_head():
    order = []
    B, V = 1, 2
    item = dict(lane=0, valid=True, padding=False, supervised=True, reset=True,
                recording='synthetic_local', run_id=0, seq_id='synthetic:0', packet_ordinal=2,
                run_start_ms=0, run_stop_ms=200, start_ms=100, cutoff_ms=150,
                reference_ms=99, target_ms=149, source_range=(0, 1),
                field0=np.zeros((16, 180, 240), np.float32),
                field1=np.zeros((16, 180, 240), np.float32),
                events=np.asarray([[20, 30, 100999, 0]], np.int64),
                initial_state=PUBLIC_INITIAL_STATE.copy(), beta_input=np.zeros(10, np.float32),
                K_source=np.asarray([[400, 0, 320], [0, 400, 240], [0, 0, 1]], np.float32),
                both_snapshots_valid=True)
    batch = XJInputBatch((item,), 'synthetic-input-only-token', 0)

    class FakeLanes:
        commits = 0

        def peek_batch(self):
            order.append('input_fields')
            return batch

        def commit_batch(self, value):
            assert value is batch
            self.commits += 1

    class Guard:
        def __enter__(self):
            order.append('guard_enter')
            return self

        def __exit__(self, *args):
            order.append('guard_exit')

        def receipt(self):
            return dict(pose_metadata_opened=False, aux_betas_accessed=False)

    class FakeGeometry:
        def prepare(self, f0, f1, q0, q1, K, both_exposed=None):
            order.append('geometry_input')
            jacobian = torch.zeros(B, V, 2, 51)
            jacobian[..., 0, 0] = 1
            normal = torch.zeros(B, V, 2)
            normal[..., 0] = 1
            rho = torch.ones(B, V)
            kwargs = dict(patches=torch.ones(B, V, 37, 16, 16), rho=rho,
                          observed=torch.ones(B, V, dtype=torch.bool),
                          B0=jacobian, B1=jacobian, normal=normal,
                          nominal_delta=torch.tensor([[[.01, 0.], [.01, 0.]]]),
                          source_ids=torch.zeros(B, V, dtype=torch.long),
                          candidate_mask=torch.ones(B, V, dtype=torch.bool),
                          spatial_logits=torch.zeros(B, V))
            return SimpleNamespace(model_kwargs=kwargs, rho=rho, q0_reference=q0)

        def targets(self, prepared, gt0, gt1):
            assert order.index('local_forward') < order.index('supervision_read')
            order.append('geometry_targets')
            return dict(target_a_px=torch.full((B, V), .02), target_d_px=torch.full((B, V), .01),
                        supervised=torch.ones(B, V, dtype=torch.bool),
                        validity_target=torch.ones(B, V))

    class FakeLabels:
        reads = 0

        def read(self, value, *, inputs_complete):
            assert value is batch and inputs_complete
            assert 'local_forward' in order and order[-1] == 'guard_exit'
            order.append('supervision_read')
            self.reads += 1
            target0 = PUBLIC_INITIAL_STATE.copy()[None]
            target1 = target0.copy()
            target0[:, 0], target1[:, 0] = .02, .03
            return SimpleNamespace(mask=np.asarray([True]), target0=target0, target1=target1,
                                   beta_target=np.full((1, 10), .1, np.float32))

    class TracedLocal(JointLocalOperator):
        def forward(self, *args, **kwargs):
            order.append('local_forward')
            return super().forward(*args, **kwargs)

    runtime = Stage1Runtime.__new__(Stage1Runtime)
    runtime.config = dict(reference_perturb_probability=0.)
    runtime.device = torch.device('cpu')
    runtime.base = FakeBase(order)
    runtime.local = TracedLocal()
    runtime.guard_class = Guard
    runtime.lanes = FakeLanes()
    runtime.geometry = FakeGeometry()
    runtime.supervision = FakeLabels()
    runtime.previous = runtime.absolute_previous = None
    runtime.batches = 0
    result = runtime.step()
    assert result['inputs_complete']
    assert runtime.lanes.commits == 0
    assert order.index('local_forward') < order.index('supervision_read') < order.index('geometry_targets')
    assert result['metrics']['supervised_outputs'] == 1
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    assert runtime.local.mean_head.weight.grad is not None
    assert runtime.local.mean_head.weight.grad.abs().sum() > 0
    assert runtime.base.frozen.grad is None
    runtime.commit(result)
    assert runtime.lanes.commits == 1
    assert not runtime.previous.requires_grad
    torch.testing.assert_close(runtime.previous[:, 0], torch.tensor([.01]))
