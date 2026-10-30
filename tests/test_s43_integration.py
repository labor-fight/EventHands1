"""S43 opt-in integration contracts.

The fast SequenceBank cases below inject a deliberately tiny public synthetic
provider. They test lifecycle/causality, not MANO physical correctness. The
separate engineering smoke run uses the real renderer and frozen checkpoint.
"""
from __future__ import annotations

import copy
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn


class FakeProvider:
    """Deterministic conditional paths; no labels or event argument exists."""
    def __init__(self):
        self.calls = 0

    def __call__(self, history, betas, K_cam, *, bins=50, image_size=(24, 32)):
        self.calls += 1
        states = np.broadcast_to(np.asarray(history), (3, bins + 1, 51)).copy()
        states[:, :, 0] += np.array([-.015, 0., .015])[:, None] * np.linspace(0., 1., bins + 1)
        levels = np.zeros((3, 3, np.prod(image_size), bins + 1), np.int32)
        for k in range(3):
            for m in range(3):
                levels[k, m, 0] = (k - 1) * np.arange(bins + 1) * (m + 1) // 4
                levels[k, m, 1] = (k - 1) * np.arange(bins + 1) * (m + 1) // 6
        return states, levels


def public_conditions():
    state = np.zeros(51, np.float32)
    state[2] = .6
    beta = np.zeros(10, np.float32)
    camera = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]], np.float32)
    return state, beta, camera


def events(start=0):
    # True repeat at one stored (pixel,polarity,timestamp), then another polarity.
    return np.array([[start + 2000, 0, 0, 0], [start + 2000, 0, 0, 0],
                     [start + 8000, 0, 0, 1], [start + 20000, 10, 10, 0]], np.int64)


class SequenceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module = importlib.import_module('tools.s43_integration.memory')
        cls.Bank = module.SequenceBank
        cls.batch = staticmethod(module.batch_from_windows)

    def bank(self, mode='br', **kwargs):
        return self.Bank(mode=mode, provider=FakeProvider(), **kwargs)

    def reset(self, bank, seq='a', start=0, history=None):
        h, b, k = public_conditions()
        bank.reset(seq, start, h if history is None else history, b, k,
                   segment_id='segment-0', history_time_us=start-1000)

    def consume(self, bank, seq='a', start=0):
        ev = events(start)
        bank.ingest(seq, ev, np.arange(len(ev)), watermark_us=start+50000)
        return bank.readout(seq, start+50000)

    def assertRepresentationEqual(self, a, b):
        for name in ('lnes', 'enhanced', 'R', 'templates'):
            np.testing.assert_allclose(a[name], b[name], rtol=0, atol=1e-11)
        self.assertEqual(a['has_events'], b['has_events'])

    def test_legacy_and_enhanced_do_not_call_geometric_provider(self):
        for mode in ('legacy', 'enhanced'):
            provider = FakeProvider()
            bank = self.Bank(mode=mode, provider=provider)
            self.reset(bank)
            result = self.consume(bank)
            self.assertEqual(provider.calls, 0)
            self.assertEqual(result['lnes'].shape, (180, 240, 2))
            self.assertEqual(result['enhanced'].shape, (180, 240, 6))

    def test_native_legacy_lnes_and_count_first_channels_match_known_events(self):
        bank = self.bank('enhanced'); self.reset(bank)
        result = self.consume(bank)
        self.assertAlmostEqual(float(result['lnes'][0, 0, 0]), 2/50, places=7)
        self.assertAlmostEqual(float(result['lnes'][0, 0, 1]), 8/50, places=7)
        np.testing.assert_array_equal(result['lnes'], result['enhanced'][..., :2])
        self.assertGreater(float(result['enhanced'][0, 0, 2]), 0.)

    def test_legacy_and_enhanced_match_the_existing_shared_event_builder(self):
        from semkine.events import splat_event_image
        ev = np.vstack([np.tile([2000, 0, 0, 0], (20, 1)), events()[2:]])
        bank = self.bank('enhanced'); self.reset(bank)
        bank.ingest('a', ev, np.arange(len(ev)), watermark_us=50000)
        actual = bank.readout('a', 50000)
        reference = splat_event_image(ev[:, 1], ev[:, 2], ev[:, 3], ev[:, 0]//1000,
                                     50, 180, 240, channels=('last', 'count', 'first'))
        np.testing.assert_array_equal(actual['enhanced'], reference)
        np.testing.assert_array_equal(actual['lnes'], reference[..., :2])

    def test_live_empty_and_missing_acquisition_have_distinct_evidence(self):
        a, b = self.bank(), self.bank(); self.reset(a); self.reset(b)
        ev = np.empty((0, 4), np.int64); ids = np.empty(0, np.int64)
        a.ingest('a', ev, ids, watermark_us=50000)
        b.ingest('a', ev, ids, watermark_us=50000, live=np.zeros(50, dtype=bool))
        ar, br = a.readout('a', 50000), b.readout('a', 50000)
        self.assertFalse(ar['has_events']); self.assertFalse(br['has_events'])
        self.assertGreater(float(np.abs(ar['R'][:, 51]-br['R'][:, 51]).max()), 1e-6)
        np.testing.assert_allclose(br['R'][:, 51], 0, rtol=0, atol=1e-10)

    def test_same_timestamp_permutations_and_packet_chunks_are_equal(self):
        a, b = self.bank(), self.bank()
        self.reset(a); self.reset(b)
        ev = events(); uid = np.arange(len(ev))
        a.ingest('a', ev, uid, watermark_us=50000)
        for ix in ([1, 0], [2], [3]):
            b.ingest('a', ev[ix], uid[ix], watermark_us=0)
        b.ingest('a', np.empty((0, 4), np.int64), np.empty(0, np.int64), watermark_us=50000)
        self.assertRepresentationEqual(a.readout('a', 50000), b.readout('a', 50000))

    def test_duplicate_source_is_idempotent_real_repeat_is_not_discarded(self):
        a, b = self.bank(), self.bank(); self.reset(a); self.reset(b)
        ev = events()
        a.ingest('a', ev, np.arange(len(ev)), watermark_us=0)
        before = a.digest('a')
        a.ingest('a', ev[:1], np.array([0]), watermark_us=0)
        self.assertEqual(before, a.digest('a'))
        b.ingest('a', ev[[0, 2, 3]], np.array([0, 2, 3]), watermark_us=0)
        for bank in (a, b):
            bank.ingest('a', np.empty((0, 4), np.int64), np.empty(0, np.int64), watermark_us=50000)
        ar, br = a.readout('a', 50000), b.readout('a', 50000)
        np.testing.assert_array_equal(ar['lnes'], br['lnes'])
        self.assertGreater(ar['enhanced'][0, 0, 2], br['enhanced'][0, 0, 2])

    def test_low_sensor_cap_does_not_truncate_native_multiplicity(self):
        bank = self.bank(); self.reset(bank)
        ev = np.tile(np.array([2000, 0, 0, 0]), (20, 1))
        bank.ingest('a', ev[:10], np.arange(10), watermark_us=0)
        bank.ingest('a', ev[10:], np.arange(10, 20), watermark_us=50000)
        self.assertEqual(bank.states['a'].native_counts[2, 0, 0], 20)
        self.assertEqual(bank.states['a'].carrier.counts[2, 0, 0], 8)
        self.assertEqual(bank.readout('a', 50000)['meta']['native_source_count'], 20)

    def test_padded_events_are_ignored_including_invalid_payloads(self):
        a, b = self.bank(), self.bank(); self.reset(a); self.reset(b)
        ev = events(); padded = np.vstack([ev, [-5, -9, -9, 9]])
        a.ingest('a', ev, np.arange(len(ev)), watermark_us=50000)
        b.ingest('a', padded, np.arange(len(padded)), watermark_us=50000,
                 padding_mask=np.array([False]*len(ev)+[True]))
        self.assertRepresentationEqual(a.readout('a', 50000), b.readout('a', 50000))

    def test_query_is_read_only_and_does_not_change_history(self):
        bank = self.bank(); self.reset(bank); self.consume(bank)
        before = bank.digest('a'); a = bank.readout('a', 50000); b = bank.readout('a', 50000)
        self.assertEqual(before, bank.digest('a'))
        self.assertRepresentationEqual(a, b)
        a['lnes'].fill(99); a['R'].fill(99)
        self.assertEqual(before, bank.digest('a'))

    def test_full_query_requires_matching_consumed_prefix(self):
        bank = self.bank(); self.reset(bank)
        bank.ingest('a', events()[:2], np.array([0, 1]), watermark_us=10000)
        partial = bank.readout('a', 10000)
        self.assertTrue(partial['has_events'])
        for cutoff in (9999, 20000, 0):
            with self.assertRaises((ValueError, RuntimeError)):
                bank.readout('a', cutoff)

    def test_half_open_epoch_and_watermark_validation(self):
        bank = self.bank(); self.reset(bank, start=100000)
        for ev, watermark in [
            (np.array([[99999, 0, 0, 0]]), 100000),
            (np.array([[150000, 0, 0, 0]]), 150000),
            (np.empty((0, 4), np.int64), 99000),
        ]:
            before = bank.digest('a')
            with self.assertRaises((ValueError, RuntimeError)):
                bank.ingest('a', ev, np.arange(len(ev)), watermark_us=watermark)
            self.assertEqual(before, bank.digest('a'))

    def test_submillisecond_watermark_does_not_consume_an_incomplete_bin(self):
        bank = self.bank(); self.reset(bank, start=100000)
        bank.ingest('a', np.array([[100001, 0, 0, 0]]), np.array([0]), watermark_us=100001)
        r = bank.readout('a', 100000)
        self.assertEqual(r['meta']['consumed_bins'], 0)
        self.assertFalse(r['has_events'])
        with self.assertRaises(ValueError): bank.readout('a', 101000)
        bank.ingest('a', np.empty((0, 4), np.int64), np.empty(0, np.int64), watermark_us=101000)
        self.assertTrue(bank.readout('a', 101000)['has_events'])

    def test_staged_future_observations_do_not_change_current_causal_prefix(self):
        a, b = self.bank(), self.bank(); self.reset(a); self.reset(b)
        prefix = np.array([[2000, 0, 0, 0]])
        a.ingest('a', prefix, np.array([0]), watermark_us=10000)
        b.ingest('a', np.vstack([prefix, [30000, 10, 10, 1]]), np.array([0, 1]), watermark_us=10000)
        self.assertRepresentationEqual(a.readout('a', 10000), b.readout('a', 10000))

    def test_two_sequences_keep_clocks_events_and_context_separate(self):
        bank = self.bank(); self.reset(bank, 'a', 0); self.reset(bank, 'b', 100000)
        self.consume(bank, 'a', 0)
        before = bank.digest('a')
        empty = np.empty((0, 4), np.int64)
        bank.ingest('b', empty, np.empty(0, np.int64), watermark_us=150000)
        self.assertFalse(bank.readout('b', 150000)['has_events'])
        self.assertTrue(bank.readout('a', 50000)['has_events'])
        self.assertEqual(before, bank.digest('a'))

    def test_reset_segment_discards_prior_sources_and_phase(self):
        bank = self.bank(); self.reset(bank); self.consume(bank)
        self.reset(bank, start=100000)
        empty = np.empty((0, 4), np.int64)
        bank.ingest('a', empty, np.empty(0, np.int64), watermark_us=150000)
        self.assertFalse(bank.readout('a', 150000)['has_events'])
        fresh = self.bank(); self.reset(fresh, start=100000)
        fresh.ingest('a', empty, np.empty(0, np.int64), watermark_us=150000)
        self.assertRepresentationEqual(bank.readout('a', 150000), fresh.readout('a', 150000))

    def test_sequence_and_source_caps_fail_closed(self):
        bank = self.bank(max_sequences=1, max_sources=3); self.reset(bank)
        before = bank.digest()
        with self.assertRaises((ValueError, OverflowError, RuntimeError)):
            self.reset(bank, 'b')
        self.assertEqual(before, bank.digest())
        with self.assertRaises((ValueError, OverflowError, RuntimeError)):
            bank.ingest('a', events(), np.arange(4), watermark_us=50000)
        self.assertEqual(before, bank.digest())
        bank.release('a')
        self.reset(bank, 'b')

    def test_history_is_detached_and_copied(self):
        h, _, _ = public_conditions()
        tensor = torch.tensor(h, requires_grad=True)
        bank = self.bank(); self.reset(bank, history=tensor)
        before = bank.digest('a')
        with torch.no_grad(): tensor[0] += .1
        self.assertEqual(before, bank.digest('a'))
        self.consume(bank)
        self.assertIsNone(tensor.grad)

    def test_empty_br_input_is_finite_and_no_event_flag_is_explicit(self):
        bank = self.bank(); self.reset(bank)
        bank.ingest('a', np.empty((0, 4), np.int64), np.empty(0, np.int64), watermark_us=50000)
        r = bank.readout('a', 50000)
        self.assertFalse(r['has_events'])
        self.assertTrue(np.isfinite(r['R']).all())
        self.assertEqual(r['R'].shape, (3, 53))
        self.assertEqual(r['templates'].shape, (3, 2, 24, 32))

    def test_offline_training_batch_and_online_prefix_are_equal(self):
        h, beta, camera = public_conditions()
        windows = [dict(seq_id='a', start_us=0, history=h, betas=beta, K_cam=camera,
                        events=events(), source_ids=np.arange(4), cutoff_us=50000),
                   dict(seq_id='b', start_us=100000, history=h, betas=beta, K_cam=camera,
                        events=events(100000), source_ids=np.arange(4), cutoff_us=150000)]
        batched = self.batch(windows, mode='br', provider=FakeProvider())
        bank = self.bank()
        for index, (seq, start) in enumerate([('a', 0), ('b', 100000)]):
            self.reset(bank, seq, start)
            online = self.consume(bank, seq, start)
            self.assertRepresentationEqual(online, batched[index])

    def test_invalid_padded_window_does_not_require_a_sequence(self):
        h, beta, camera = public_conditions()
        item = dict(seq_id=None, valid=False, start_us=0, history=h, betas=beta,
                    K_cam=camera, events=np.array([[-99, -99, -99, 8]]),
                    source_ids=np.array([-1]), cutoff_us=50000)
        batch = self.batch([item], mode='br', provider=FakeProvider())
        self.assertFalse(batch[0]['has_events'])
        self.assertTrue(np.all(batch[0]['lnes'] == 0))


class TinyLegacyS43(nn.Module):
    """A differentiable interface stub, explicitly not a checkpoint smoke."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 3, 3, padding=1)
        self.rn = nn.Sequential(nn.BatchNorm2d(3), nn.AdaptiveAvgPool2d(1),
                                nn.Flatten(), nn.Linear(3, 51))

    def forward(self, image, previous, betas=None, camera_K=None):
        image = image.to(previous)
        delta = self.rn(self.conv1(image.permute(0, 3, 1, 2)))
        empty = image.abs().sum((1, 2, 3)) == 0
        return previous + torch.where(empty[:, None], torch.zeros_like(delta), delta)

    def _compute_loss(self, prediction, target, betas=None):
        value = (prediction-target).square().mean()
        return value, {'stub_mse': value}


def model_representation(seed=7):
    rng = np.random.default_rng(seed)
    image = rng.uniform(0, 1, (180, 240, 2)).astype(np.float32)
    enhanced = np.concatenate([image, rng.uniform(0, 1, (180, 240, 4)).astype(np.float32)], -1)
    R = np.zeros((3, 53), np.float32)
    R[:, 51] = [-1000., 0., -1000.]
    R[:, 52] = 1
    templates = np.stack([np.full((2, 24, 32), value, np.float32) for value in [.15, .45, .9]])
    return {'lnes': image, 'enhanced': enhanced, 'R': R, 'templates': templates,
            'has_events': True, 'valid': True, 'meta': {'valid': True}}


class ModelAdapterContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Model = importlib.import_module('tools.s43_integration.model').S43IntegratedModel
        torch.set_num_threads(1)

    def model(self, mode='br', train_conv1=True):
        torch.manual_seed(13)
        return self.Model(TinyLegacyS43(), mode=mode, train_conv1=train_conv1,
                          temperature=1000., k=3)

    @staticmethod
    def conditions(n=1):
        previous = torch.zeros(n, 51)
        previous[:, 2] = .6
        return previous, torch.zeros(n, 10), torch.eye(3).repeat(n, 1, 1)

    def test_legacy_branch_matches_original_forward_exactly(self):
        model = self.model('legacy'); model.eval()
        representations = [model_representation(7), model_representation(8)]
        previous, beta, camera = self.conditions(2)
        image = torch.from_numpy(np.stack([r['lnes'] for r in representations]))
        with torch.no_grad():
            expected = model.base(image, previous, betas=beta, camera_K=camera)
            actual = model(representations, previous, beta, camera)
        self.assertTrue(torch.equal(expected, actual))

    def test_enhanced_adapter_starts_as_exact_last_channel_copy(self):
        model = self.model('enhanced'); model.eval()
        representations = [model_representation()]
        previous, beta, camera = self.conditions()
        with torch.no_grad():
            expected = model.base(torch.from_numpy(representations[0]['lnes'])[None],
                                  previous, betas=beta, camera_K=camera)
            actual = model(representations, previous, beta, camera)
        self.assertTrue(torch.equal(expected, actual))

    def test_br_ignores_raw_lnes_when_evidence_and_templates_are_fixed(self):
        model = self.model('br'); model.eval()
        a = model_representation(); b = copy.deepcopy(a); b['lnes'].fill(99)
        previous, beta, camera = self.conditions()
        with torch.no_grad():
            ya = model([a], previous, beta, camera)
            yb = model([b], previous, beta, camera)
        self.assertTrue(torch.equal(ya, yb))

    def test_br_uses_candidate_evidence(self):
        model = self.model('br'); model.eval()
        a = model_representation(); b = copy.deepcopy(a)
        b['R'][:, 51] = [0., -1000., -1000.]
        previous, beta, camera = self.conditions()
        with torch.no_grad():
            ya = model([a], previous, beta, camera)
            yb = model([b], previous, beta, camera)
        self.assertGreater(float((ya-yb).abs().max()), 1e-7)

    def test_adapter_backpropagates_without_unfreezing_trunk(self):
        model = self.model('br', train_conv1=False); model.train()
        previous, beta, camera = self.conditions()
        prediction = model([model_representation()], previous, beta, camera)
        loss, _ = model.base._compute_loss(prediction, torch.ones_like(prediction), beta)
        loss.backward()
        gradients = [p.grad for p in model.adapter.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and torch.count_nonzero(g) for g in gradients))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.base.parameters()))

    def test_only_input_conv_can_be_unfrozen_and_bn_statistics_stay_fixed(self):
        model = self.model('br', train_conv1=True); model.train()
        before = [m.running_mean.clone() for m in model.base.modules() if isinstance(m, nn.BatchNorm2d)]
        previous, beta, camera = self.conditions()
        model([model_representation()], previous, beta, camera).sum().backward()
        self.assertTrue(all(p.requires_grad for p in model.base.conv1.parameters()))
        for name, p in model.base.named_parameters():
            if not name.startswith('conv1.'):
                self.assertFalse(p.requires_grad)
        after = [m.running_mean for m in model.base.modules() if isinstance(m, nn.BatchNorm2d)]
        for a, b in zip(before, after): self.assertTrue(torch.equal(a, b))
        self.assertFalse(model.base.training)

    def test_empty_new_input_holds_state_exactly(self):
        for mode in ('legacy', 'enhanced', 'br'):
            model = self.model(mode); model.eval()
            r = model_representation()
            r['has_events'] = False; r['lnes'].fill(0); r['enhanced'].fill(0)
            previous, beta, camera = self.conditions()
            with torch.no_grad(): actual = model([r], previous, beta, camera)
            self.assertTrue(torch.equal(actual, previous))

    def test_padded_or_unsupported_br_row_never_enters_adapter(self):
        model = self.model('br'); model.eval()
        reps = [model_representation(7), model_representation(8), model_representation(9)]
        reps[1]['R'].fill(0); reps[1]['meta']['status'] = 'padding'; reps[1]['meta']['padding'] = True
        reps[2]['R'].fill(0); reps[2]['meta']['status'] = 'unsupported'
        previous, beta, camera = self.conditions(3)
        with torch.no_grad(): actual = model(reps, previous, beta, camera)
        self.assertTrue(torch.equal(actual[1:], previous[1:]))
        self.assertGreater(float((actual[0]-previous[0]).abs().max()), 1e-6)

    def test_checkpoint_roundtrip_preserves_output_and_representation_identity(self):
        module = importlib.import_module('tools.s43_integration.model')
        model = self.model('br', train_conv1=True); model.eval()
        cfg = {'MODEL': {'S43_EVENT_REP': 'br', 'S43_TRAIN_CONV1': True,
                         'S43_BR_TEMPERATURE': 1000.}}
        previous, beta, camera = self.conditions()
        reps = [model_representation()]
        with torch.no_grad(): expected = model(reps, previous, beta, camera)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'integration.pt'
            model.save(path, cfg, {'role': 'unit stub checkpoint'})
            with mock.patch.object(module, 'load_legacy', return_value=(TinyLegacyS43(), cfg)):
                restored, restored_cfg = module.load_integrated(cfg, checkpoint=path)
            restored.eval()
            with torch.no_grad(): actual = restored(reps, previous, beta, camera)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(restored_cfg, cfg)
        self.assertEqual(restored.temperature, model.temperature)
        self.assertEqual(restored.k, model.k)

    def test_checkpoint_rejects_mode_or_temperature_semantic_mismatch(self):
        module = importlib.import_module('tools.s43_integration.model')
        model = self.model('br')
        cfg = {'MODEL': {'S43_EVENT_REP': 'br', 'S43_BR_TEMPERATURE': 1000.}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'integration.pt'; model.save(path, cfg)
            for key, value in [('S43_EVENT_REP', 'enhanced'), ('S43_BR_TEMPERATURE', 500.)]:
                changed = copy.deepcopy(cfg); changed['MODEL'][key] = value
                with mock.patch.object(module, 'load_legacy', return_value=(TinyLegacyS43(), changed)):
                    with self.assertRaises(ValueError): module.load_integrated(changed, checkpoint=path)


class TrackerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tracker_module = importlib.import_module('tools.s43_integration.tracker')
        cls.memory_module = importlib.import_module('tools.s43_integration.memory')
        cls.Model = importlib.import_module('tools.s43_integration.model').S43IntegratedModel
        torch.set_num_threads(1)

    def tracker(self):
        parent = self.memory_module.SequenceBank
        class InjectedBank(parent):
            def __init__(self, mode='br', **kwargs):
                super().__init__(mode=mode, provider=FakeProvider(), **kwargs)
        torch.manual_seed(14)
        model = self.Model(TinyLegacyS43(), mode='br')
        with mock.patch.object(self.tracker_module, 'SequenceBank', InjectedBank):
            return self.tracker_module.S43StreamTracker(model)

    def reset(self, tracker, seq='a', start=0, state=None):
        h, b, k = public_conditions()
        tracker.reset(seq, start, h if state is None else state, b, k,
                      segment_id='s0', state_time_us=start-1000)

    def complete(self, tracker, seq='a', start=0):
        ev = events(start)
        tracker.ingest(seq, ev, np.arange(len(ev)), watermark_us=start+50000)
        return tracker.query(seq, start+50000)

    def test_query_does_not_update_hand_state_or_version(self):
        tracker = self.tracker(); self.reset(tracker)
        initial = tracker.states['a']['pose'].copy()
        first = self.complete(tracker)
        second = tracker.query('a', 50000)
        np.testing.assert_array_equal(first['state'], second['state'])
        np.testing.assert_array_equal(tracker.states['a']['pose'], initial)
        self.assertEqual(tracker.states['a']['version'], 0)

    def test_commit_only_latest_query_and_only_at_epoch_end(self):
        tracker = self.tracker(); self.reset(tracker)
        ev = events()[:3]
        tracker.ingest('a', ev, np.array([0, 1, 2]), watermark_us=10000)
        partial = tracker.query('a', 10000)
        with self.assertRaises(ValueError): tracker.commit('a', partial)
        tracker.ingest('a', events()[3:], np.array([3]), watermark_us=50000)
        first = tracker.query('a', 50000); second = tracker.query('a', 50000)
        with self.assertRaises(ValueError): tracker.commit('a', first)
        committed = tracker.commit('a', second)
        np.testing.assert_array_equal(committed, second['state'])
        self.assertEqual(tracker.states['a']['version'], 1)
        self.assertEqual(tracker.states['a']['start_us'], 50000)
        self.assertEqual(tracker.bank.states['a'].end, 0)
        self.assertEqual(tracker.bank.states['a'].seen, {})

    def test_final_epoch_cannot_be_committed_twice(self):
        tracker = self.tracker(); self.reset(tracker)
        result = self.complete(tracker)
        tracker.commit('a', result, begin_next=False)
        with self.assertRaises((ValueError, RuntimeError)):
            repeated = tracker.query('a', 50000)
            tracker.commit('a', repeated, begin_next=False)

    def test_commit_rejects_nonfinite_modified_result_before_state_mutation(self):
        tracker = self.tracker(); self.reset(tracker)
        result = self.complete(tracker)
        initial = tracker.states['a']['pose'].copy()
        result['state'][0] = np.nan
        with self.assertRaises((ValueError, ArithmeticError)):
            tracker.commit('a', result)
        np.testing.assert_array_equal(tracker.states['a']['pose'], initial)
        self.assertEqual(tracker.states['a']['version'], 0)

    def test_tensor_history_and_commits_do_not_retain_autograd_graphs(self):
        tracker = self.tracker()
        h, _, _ = public_conditions(); source = torch.tensor(h, requires_grad=True)
        self.reset(tracker, state=source)
        with torch.no_grad(): source[0] += .2
        self.assertEqual(tracker.states['a']['pose'][0], 0.)
        result = self.complete(tracker); tracker.commit('a', result)
        self.assertIsInstance(tracker.states['a']['pose'], np.ndarray)
        self.assertIsNone(source.grad)

    def test_interleaved_sequences_do_not_commit_each_other(self):
        tracker = self.tracker(); self.reset(tracker, 'a', 0); self.reset(tracker, 'b', 100000)
        original_b = tracker.states['b']['pose'].copy()
        result_a = self.complete(tracker, 'a', 0)
        tracker.commit('a', result_a)
        np.testing.assert_array_equal(tracker.states['b']['pose'], original_b)
        self.assertEqual(tracker.states['b']['version'], 0)
        self.assertEqual(tracker.bank.states['b'].end, 0)
        tracker.release('a')
        self.assertNotIn('a', tracker.states)
        self.assertNotIn('a', tracker.bank.states)

    def test_unsupported_geometry_explicitly_holds_but_still_accounts_events(self):
        tracker = self.tracker()
        def unsupported(*args, **kwargs):
            raise ValueError('synthetic provider intentionally unsupported')
        tracker.bank.provider = unsupported
        self.reset(tracker)
        self.assertFalse(tracker.states['a']['supported'])
        initial = tracker.states['a']['pose'].copy()
        result = self.complete(tracker)
        np.testing.assert_array_equal(result['state'], initial)
        meta = result['representation_meta']
        self.assertEqual(meta['status'], 'unsupported')
        self.assertEqual(meta['native_source_count'], len(events()))
        self.assertIn('no raw-LNES fallback', meta['policy'])

    def test_unsupported_raise_policy_propagates_without_creating_hand_state(self):
        tracker = self.tracker(); tracker.unsupported_policy = 'raise'
        def unsupported(*args, **kwargs):
            raise ValueError('synthetic provider intentionally unsupported')
        tracker.bank.provider = unsupported
        with self.assertRaises(ValueError): self.reset(tracker)
        self.assertNotIn('a', tracker.states)


if __name__ == '__main__':
    unittest.main()
