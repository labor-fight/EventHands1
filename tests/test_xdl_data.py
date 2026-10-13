"""XDL raw clip clock, paired counter stream and distributed row contracts."""
from pathlib import Path
import sys
import struct

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.xdl.data import (RawRecoveryDataset, collate_recovery, counter_rng,
                            sample_coordinates, packet_hash, build_train_dataset)


@pytest.fixture
def tiny_root(tmp_path):
    n = 400
    base = tmp_path / 'train' / 'fixture_global'
    base.parent.mkdir()
    np.save(str(base) + '_events.npy', np.column_stack([
        np.arange(n) % 200, np.arange(n) % 150, np.arange(n) % 2]).astype(np.uint8))
    np.save(str(base) + '_offsets.npy', np.arange(n + 1, dtype=np.int64))
    np.save(str(base) + '_tsub.npy', np.full(n, 437, dtype=np.uint16))
    np.savez(str(base) + '_aux.npz', valid_runs_ms=np.array([[10, 270], [290, 400]]),
             category='global', betas=np.zeros(10, np.float32),
             camera_K=np.array([[400., 0, 120.], [0, 400., 90.], [0, 0, 1]], np.float32))
    dtype = np.dtype([('data', np.float64, 51), ('m0', np.uint8), ('m1', np.uint8)])
    labels = np.zeros(n, dtype=dtype)
    labels['data'][:, 0] = np.arange(n) * .001
    labels['data'][:, 2] = .5
    labels['m0'] = 4
    labels['m1'] = 13
    with open(str(base) + '.meta', 'wb') as f:
        f.write(struct.pack('<i', 51))
        labels.tofile(f)
    return tmp_path


def make(tiny_root, **kwargs):
    return RawRecoveryDataset(
        root=tiny_root, sequences=[('fixture_global', 'train')],
        components=np.eye(45, dtype=np.float32), mano_npz=str(REPO / 'assets/mano_right.npz'),
        seed=3407, max_steps=6000, **kwargs)


def test_counter_stream_independent_of_model_rng():
    a = counter_rng(3407, 12, 34, 'main_events').random(8)
    torch.manual_seed(5)
    torch.nn.Linear(999, 777)
    np.random.seed(818)
    b = counter_rng(3407, 12, 34, 'main_events').random(8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, counter_rng(3407, 12, 34, 'initial_noise').random(8))


def test_global_rows_cover_exact_optimizer_batch():
    for world in (1, 2, 4):
        all_rows = []
        for rank in range(world):
            idx = np.arange(rank, 1024 * 3, world)
            steps_rows = [sample_coordinates(int(x)) for x in idx]
            for step in range(3):
                rows = [r for s, r in steps_rows if s == step]
                all_rows.extend((step, r) for r in rows)
        for step in range(3):
            assert sorted(r for s, r in all_rows if s == step) == list(range(1024))
    assert sample_coordinates(6143999) == (5999, 1023)


def test_legal_k4_selection_and_incoming_clock(tiny_root):
    ds = make(tiny_root)
    assert len(ds) == 6000 * 1024
    assert len(ds.legal_index) == 61
    for idx in [0, 1, 512, 1023, 1024, 2047, 2048000]:
        item = ds[idx]
        packets = item['prefix'] + [item['main']]
        assert all(p.meta['run_start_ms'] == 10 and p.meta['run_end_ms'] == 270 for p in packets)
        assert all(np.isclose(p.delta_t_s, .05, atol=1e-12) for p in packets)
        assert all(p.meta['end_idx'] < 270 for p in packets)
        assert all(a.t_end_us == b.t_start_us for a, b in zip(packets, packets[1:]))
        # GT prediction endpoint is the final labelled millisecond, 1ms before evidence end.
        assert item['main'].meta['incoming_ms'] == item['main'].t_start_us // 1000 - 1
        assert packets[0].meta['incoming_ms'] == packets[0].t_start_us // 1000
        assert packets[0].meta['target_timestamp_us'] - packets[0].meta['incoming_ms'] * 1000 == 49000
        for p in packets[1:]:
            assert p.meta['target_timestamp_us'] - p.meta['incoming_ms'] * 1000 == 50000
        assert all(np.count_nonzero(p.target) == 0 for p in item['prefix'])
        assert np.isclose(item['main'].prev_state[0], item['main'].meta['incoming_ms'] * .001)
        assert np.isclose(item['main'].target[0], item['main'].meta['end_idx'] * .001)


def test_shared_geometry_and_main_hash_survives_prefix_reads(tiny_root):
    from semkine.domrand import DomRandConfig
    ds = make(tiny_root, domrand=DomRandConfig(enabled=True),
              polarity_flip=True, pixel_polarity_swap=True,
              prev_noise=(.005, .05, .05), prev_noise_mode='mixed')
    a = ds[4096]
    for idx in range(5):
        ds[idx]
    b = ds[4096]
    assert packet_hash(collate_recovery([a])['main']) == packet_hash(collate_recovery([b])['main'])
    for p in a['prefix']:
        assert np.array_equal(p.camera_K, a['main'].camera_K)
        assert p.meta['domrand'] == a['main'].meta['domrand']
    # Future prefix labels/state cannot be exposed to a burn model.
    assert all(np.count_nonzero(p.prev_state) == 0 for p in a['prefix'][1:])


def test_ragged_collate_has_full_row_and_step_identity(tiny_root):
    ds = make(tiny_root)
    batch = collate_recovery([ds[1024 + k] for k in (0, 2, 4)])
    batch['main'].validate()
    for p in batch['prefix']:
        p.validate()
    assert batch['global_row'].tolist() == [0, 2, 4]
    assert batch['sample_index'].tolist() == [1024, 1026, 1028]
    assert batch['optimizer_step'].tolist() == [1, 1, 1]


def test_resume_keeps_counter_sample_identity(tiny_root):
    full = make(tiny_root)
    resumed = make(tiny_root, start_step=3)
    assert len(resumed) == (6000 - 3) * 1024
    a, b = full[3072], resumed[0]
    assert a['sample_index'] == b['sample_index'] == 3072
    assert packet_hash(collate_recovery([a])['main']) == packet_hash(collate_recovery([b])['main'])


def test_no_legal_run_is_explicit_failure(tiny_root):
    base = tiny_root / 'train/fixture_global_aux.npz'
    old = dict(np.load(base))
    old['valid_runs_ms'] = np.array([[10, 100], [290, 400]])
    np.savez(base, **old)
    with pytest.raises(ValueError, match='legal'):
        make(tiny_root)


def test_worker_count_keeps_same_packet_stream(tiny_root):
    from torch.utils.data import DataLoader, Subset
    ds = make(tiny_root)
    hashes = []
    for workers in (0, 2):
        loader = DataLoader(Subset(ds, [0, 1, 1024, 1025]), batch_size=2,
                            num_workers=workers, shuffle=False, collate_fn=collate_recovery)
        hashes.append([packet_hash(b['main']) for b in loader])
    assert hashes[0] == hashes[1]


def test_five_arm_configs_differ_only_by_registered_condition(tmp_path):
    import importlib
    try:
        generate = importlib.import_module('tools.xdl.make_configs').make_configs
    except ModuleNotFoundError:
        pytest.fail('five-arm matched XDL config generator is not implemented')
    paths = generate(tmp_path)
    import yaml
    cfgs = [yaml.safe_load(p.read_text()) for p in paths]
    assert len(cfgs) == 5
    stripped = []
    for cfg in cfgs:
        assert cfg['SEED'] == 3407
        assert cfg['TRAIN']['DEVICES'] == 2
        assert cfg['TRAIN']['BATCH_SIZE_PER_GPU'] == 512
        assert cfg['TRAIN']['ACCUMULATE_GRAD_BATCHES'] == 1
        assert cfg['TRAIN']['MAX_STEPS'] == 6000
        assert cfg['TRAIN']['LR_SCHEDULE'] == 'constant'
        assert cfg['AUG']['SPEED_AUG'] is False
        assert cfg['MODEL']['XYZ_GEOM'] is False
        assert cfg['MODEL']['XDL_BURN_MODE'] == 'full'
        for key in ('XDL_QUERY', 'XDL_SELF', 'XDL_OBSERVATION_NULL'):
            cfg['MODEL'].pop(key)
        cfg['TRAIN'].pop('OUTPUT_DIR')
        cfg['TRAIN'].pop('RUN_NAME')
        cfg['EVAL'].pop('OUTPUT_DIR')
        stripped.append(cfg)
    assert all(c == stripped[0] for c in stripped)


def test_explicit_ddp_sampler_preserves_step_not_lightning_training_shuffle(tiny_root):
    import tools.xdl.data as data
    assert hasattr(data, 'rank_sampler'), 'explicit nonshuffled DDP sampler is missing'
    ds = make(tiny_root)
    for rank in (0, 1):
        sampler = data.rank_sampler(ds, replicas=2, rank=rank)
        assert sampler.shuffle is False
        sampler.set_epoch(17)
        first = list(iter(sampler))[:512]
        assert first == list(range(rank, 1024, 2))


def test_nonfinite_gradient_cannot_be_saved_as_completed_training(tmp_path):
    from types import SimpleNamespace
    from tools.xdl.train import RecoveryProvenance
    module = torch.nn.Linear(2, 1)
    module.weight.grad = torch.full_like(module.weight, float('nan'))
    callback = RecoveryProvenance(tmp_path)
    trainer = SimpleNamespace(global_rank=0, global_step=5999)
    with pytest.raises(FloatingPointError, match='gradient'):
        callback.on_before_optimizer_step(trainer, module, torch.optim.Adam(module.parameters()), 0)
    artifact = __import__('json').loads((tmp_path / 'nonfinite_gradient_rank0.json').read_text())
    assert artifact['status'] == 'FAIL'
