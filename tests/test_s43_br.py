"""Contracts for the CPU-only B-R prototype, independent of dataset performance."""
from __future__ import annotations

import importlib.util
import itertools
import math
from pathlib import Path
import random
import sys
import unittest

import numpy as np

from tools.s43_br.carrier import (
    Event, GridKernel, EvidenceMemory, count_pmf, counts_to_events, encode_batch,
)


ROOT = Path(__file__).resolve().parents[1]


def paths():
    return np.array([
        [[[0, 1, 3, 2, 1, 2, 4]], [[0, 2, 4, 3, 2, 3, 5]]],
        [[[0, -1, -3, -2, -1, -2, -4]], [[0, -2, -4, -3, -2, -3, -5]]],
    ], dtype=np.int64)


def kernel(**kwargs):
    return GridKernel(paths(), bins=3, substeps=2, **kwargs)


def counts():
    return np.array([[[2, 0]], [[0, 1]], [[2, 0]]], dtype=np.int64)


def memory(**kwargs):
    return EvidenceMemory(kernel(), **kwargs)


def consume(z, a=None):
    z.ingest(counts_to_events(counts() if a is None else a))
    z.advance(3000, watermark_us=3000)
    return z


class CarrierContracts(unittest.TestCase):
    def assertEvidenceEqual(self, a, b):
        np.testing.assert_allclose(a.ell, b.ell, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(a.phi, b.phi, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(a.likelihoods(), b.likelihoods(), rtol=1e-12, atol=1e-12)

    def test_normalized_count_kernel_including_empty_and_overflow(self):
        for n, alpha, nu in itertools.product([0, 1, 2, 7, 30], [0, .3, 1], [0, .02, 1]):
            p = count_pmf(n, alpha, nu, 3)
            self.assertTrue(np.all(p >= 0))
            self.assertAlmostEqual(float(p.sum()), 1, places=11)

    def test_normalized_full_hidden_transition(self):
        k = kernel()
        for b in range(k.B):
            _, plus, minus = k.predict(b)
            total = np.zeros((k.K, k.M, k.P, k.Q))
            for a, c in itertools.product(range(k.cap + 1), repeat=2):
                lp = k.observation_log_prob(plus, np.array([a])) + k.observation_log_prob(minus, np.array([c]))
                total += np.exp(lp)
            np.testing.assert_allclose(total, 1, rtol=1e-12, atol=1e-12)

    def test_forward_matches_frozen_standard_library_reference(self):
        location = ROOT / "outputs/s43_repr_definition_20261027/reference_carrier.py"
        spec = importlib.util.spec_from_file_location("s43_br_old_reference", location)
        ref = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = ref
        spec.loader.exec_module(ref)
        old = ref.EvidenceMemory(ref.GridKernel(paths().tolist(), bins=3, substeps=2))
        events = counts_to_events(counts())
        old.ingest([ref.Event(e.uid, e.t_us // 1000, e.pixel, e.polarity) for e in events])
        old.advance(3)
        new = consume(memory())
        np.testing.assert_allclose(new.ell, old.ell, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(new.phi, old.phi, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(new.likelihoods(), old.likelihoods(), rtol=1e-12, atol=1e-12)

    def test_offline_training_and_online_constructors_are_identical(self):
        off = encode_batch(paths(), counts(), substeps=2)
        online = consume(memory()).query()
        for name in ("log_likelihood", "log_weights", "phi", "R", "posterior"):
            np.testing.assert_allclose(off[name], online[name], rtol=1e-12, atol=1e-12)

    def test_chunking_and_query_schedule_do_not_change_evidence(self):
        a = consume(memory())
        b = memory()
        events = counts_to_events(counts())
        for bi in range(3):
            for e in events:
                if e.t_us // 1000 == bi:
                    b.ingest([e])
            b.advance((bi + 1) * 1000, watermark_us=(bi + 1) * 1000)
            b.query()
            b.query()
        self.assertEvidenceEqual(a, b)
        self.assertEqual(a.digest(), b.digest())

    def test_same_timestamp_event_permutation_preserves_true_multiplicity(self):
        events = counts_to_events(counts())
        shuffled = events.copy()
        random.Random(39).shuffle(shuffled)
        a, b = memory(), memory()
        a.ingest(events)
        b.ingest(shuffled)
        a.advance(3000, watermark_us=3000)
        b.advance(3000, watermark_us=3000)
        self.assertEqual(a.digest(), b.digest())
        self.assertEqual(int(a.counts[0, 0, 0]), 2)

    def test_future_staging_cannot_change_current_prefix(self):
        a, b = memory(), memory()
        a.ingest([Event(0, 0, 0, 0)])
        b.ingest([Event(0, 0, 0, 0), Event(1, 2000, 0, 1)])
        for z in (a, b):
            z.advance(1000, watermark_us=1000)
        self.assertEvidenceEqual(a, b)
        np.testing.assert_equal(a.query()["R"], b.query()["R"])
        with self.assertRaises(ValueError):
            b.query(cutoff_us=0)
        with self.assertRaises(ValueError):
            b.query(cutoff_us=2000)

    def test_query_is_read_only_and_returns_owned_arrays(self):
        z = consume(memory())
        before = z.digest()
        r = z.query()
        r["phi"].fill(0)
        r["ell"].fill(123)
        r["R"].fill(123)
        self.assertEqual(z.digest(), before)
        self.assertEqual(z.query()["R"].shape, (2, 53))

    def test_half_open_time_bins_and_prefix_watermark(self):
        z = memory(start_us=2000)
        z.ingest([Event(0, 2999, 0, 0), Event(1, 3000, 0, 1)])
        self.assertEqual(int(z.counts[0, 0, 0]), 1)
        self.assertEqual(int(z.counts[1, 0, 1]), 1)
        before = z.digest()
        for cut, water in [(3000, 2999), (3001, 4000), (1000, 4000), (6000, 6000)]:
            with self.assertRaises(ValueError):
                z.advance(cut, watermark_us=water)
            self.assertEqual(before, z.digest())

    def test_source_ids_are_idempotent_without_deleting_real_repeats(self):
        z = memory()
        event = Event(7, 0, 0, 0)
        self.assertEqual(z.ingest([event, event]), 1)
        self.assertEqual(z.ingest([event]), 0)
        self.assertEqual(z.ingest([Event(8, 0, 0, 0)]), 1)
        self.assertEqual(z.counts[0, 0, 0], 2)
        with self.assertRaises(ValueError):
            z.ingest([Event(7, 0, 0, 1)])

    def test_invalid_ingest_is_transactional_and_rejects_new_late_events(self):
        z = memory()
        before = z.digest()
        for bad in (Event(1, 0, 0, 2), Event(1, 0, 1, 0), Event(1, 0., 0, 0), Event(-1, 0, 0, 0)):
            with self.assertRaises(ValueError):
                z.ingest([Event(0, 0, 0, 0), bad])
            self.assertEqual(z.digest(), before)
        z.ingest([Event(0, 0, 0, 0)])
        z.advance(1000, watermark_us=1000)
        self.assertEqual(z.ingest([Event(0, 0, 0, 0)]), 0)
        with self.assertRaises(ValueError):
            z.ingest([Event(1, 0, 0, 0)])

    def test_empty_live_observation_and_missing_are_different(self):
        a, b = memory(), memory()
        a.advance(1000, watermark_us=1000)
        live = np.ones((3, 1), dtype=bool)
        live[0] = False
        b.advance(1000, watermark_us=1000, live=live)
        self.assertTrue(np.any(a.ell < 0))
        np.testing.assert_allclose(b.ell, 0, atol=1e-12)
        self.assertFalse(np.array_equal(b.phi, b.initial_phase))

    def test_dropout_conflicts_are_transactional(self):
        z = memory()
        z.ingest([Event(0, 1000, 0, 1)])
        before = z.digest()
        mask = np.ones((3, 1), dtype=bool)
        mask[1] = False
        with self.assertRaises(ValueError):
            z.advance(3000, watermark_us=3000, live=mask)
        self.assertEqual(z.digest(), before)

    def test_probability_is_not_normalized_away_per_observation(self):
        z = consume(memory())
        self.assertTrue(np.any(np.abs(z.ell) > 1e-5))
        self.assertAlmostEqual(float(z.query()["posterior"].sum()), 1, places=12)
        np.testing.assert_allclose(z.phi.sum(axis=-1), 1, rtol=1e-12, atol=1e-12)

    def test_shared_nuisance_integrated_once_after_all_bins(self):
        z = consume(memory(omega=[[.25, .75], [.6, .4]]))
        direct = np.log(np.sum(np.exp(z.ell) * np.exp(z.logomega), axis=1))
        np.testing.assert_allclose(z.likelihoods(), direct, rtol=1e-12, atol=1e-12)
        self.assertNotAlmostEqual(.5 * .9 * .1 + .5 * .1 * .9,
                                  (.5 * .9 + .5 * .1) * (.5 * .1 + .5 * .9))

    def test_zero_prior_remains_zero_and_unsupported_is_not_confidence(self):
        z = consume(memory(pi=[0, 1]))
        self.assertEqual(z.query()["posterior"][0], 0)
        static_paths = np.zeros((2, 1, 1, 3), dtype=np.int64)
        impossible = encode_batch(static_paths, np.array([[[1, 0]], [[0, 0]]]), alpha=1, nu=0)
        self.assertEqual(impossible["status"], "unsupported")
        self.assertFalse(impossible["posterior_defined"])
        np.testing.assert_equal(impossible["posterior"], 0)

    def test_numerical_reference_change_does_not_change_measurements(self):
        z = consume(memory())
        before = z.digest()
        old, new = z.query(), z.query(np.ones((2, 51)))
        np.testing.assert_equal(old["log_likelihood"], new["log_likelihood"])
        np.testing.assert_equal(old["R"][:, 51:], new["R"][:, 51:])
        self.assertEqual(z.digest(), before)

    def test_physical_path_refresh_replays_and_replaces_evidence(self):
        z = consume(memory())
        changed = paths()
        changed[0, :, :, 2:5] += 2
        new_kernel = GridKernel(changed, substeps=2)
        refreshed = z.replay(new_kernel)
        fresh = encode_batch(changed, counts(), substeps=2)
        np.testing.assert_allclose(refreshed.likelihoods(), fresh["log_likelihood"], rtol=1e-12, atol=1e-12)
        self.assertFalse(np.allclose(z.likelihoods(), refreshed.likelihoods()))
        with self.assertRaises(OverflowError):
            refreshed.replay(new_kernel)

    def test_replay_cannot_silently_change_boundary_or_condition(self):
        z = consume(memory())
        changed = paths()
        changed[..., 0] += 1
        with self.assertRaises(ValueError):
            z.replay(GridKernel(changed, substeps=2))
        with self.assertRaises(ValueError):
            z.replay(kernel(context_id="different physical boundary"))

    def test_bounded_source_account_and_epoch_extent(self):
        z = memory(max_sources=2)
        z.ingest([Event(0, 0, 0, 0), Event(1, 0, 0, 0)])
        before = z.digest()
        with self.assertRaises(OverflowError):
            z.ingest([Event(2, 0, 0, 0)])
        self.assertEqual(z.digest(), before)
        self.assertEqual(z.memory_bytes()["source_count"], 2)
        self.assertEqual(z.memory_bytes()["source_cap"], 2)
        with self.assertRaises(ValueError):
            z.ingest([Event(2, 3000, 0, 0)])

    def test_phase_and_kernel_ownership_are_independent(self):
        initial = np.ones((2, 2, 1, 3)) / 3
        z = memory(phase=initial)
        initial.fill(0)
        np.testing.assert_allclose(z.phi, 1 / 3)
        original = paths()
        k = GridKernel(original, substeps=2)
        original.fill(999)
        self.assertNotEqual(k.paths[0, 0, 0, 0], 999)
        with self.assertRaises(ValueError):
            k.paths[0, 0, 0, 0] = 999
        with self.assertRaises(AttributeError):
            k.cap = 7

    def test_polarity_swap_requires_matching_model_swap(self):
        x = encode_batch(paths(), counts(), substeps=2)
        y = encode_batch(-paths(), counts()[..., ::-1].copy(), substeps=2)
        np.testing.assert_allclose(x["log_likelihood"], y["log_likelihood"], rtol=1e-12, atol=1e-12)

    def test_subbin_nonmonotone_path_keeps_both_trigger_polarities(self):
        k = GridKernel(np.array([[[[0, 4, 0]]], [[[0, 0, 0]]]]), substeps=2, alpha=1, nu=0)
        _, plus, minus = k.predict(0)
        self.assertTrue(np.all(plus[0] > 0))
        self.assertTrue(np.all(minus[0] > 0))
        self.assertTrue(np.all(plus[1] == 0))
        self.assertTrue(np.all(minus[1] == 0))

    def test_per_nuisance_noise_and_overflow_before_thinning(self):
        k = kernel(alpha=[.1, .9], nu=[0, .05])
        self.assertEqual(k.alpha.shape, (2,))
        p_large = count_pmf(30, .1, 0, 3)
        p_wrong = count_pmf(3, .1, 0, 3)
        self.assertGreater(abs(float(p_large[0] - p_wrong[0])), .1)

    def test_observed_overflow_keeps_source_multiplicity_and_matches_batch(self):
        z = memory()
        events = [Event(i, 0, 0, 0) for i in range(7)]
        self.assertEqual(z.ingest(events), 7)
        self.assertEqual(len(z.seen), 7)
        self.assertEqual(z.counts[0, 0, 0], z.kernel.cap)
        z.advance(3000, watermark_us=3000)
        observation = np.zeros((3, 1, 2), dtype=np.int64)
        observation[0, 0, 0] = 7
        offline = encode_batch(paths(), observation, substeps=2)
        np.testing.assert_allclose(z.likelihoods(), offline["log_likelihood"], rtol=1e-12, atol=1e-12)

    def test_cap8_tiny_poisson_tail_keeps_strict_support(self):
        nu, cap = .02, 8
        first = math.exp(-nu) * nu ** cap / math.factorial(cap)
        # Direct positive recurrence is independent of the implementation's
        # convolution and pins relative accuracy of the very small tail.
        terms = [math.exp(-nu) * nu ** j / math.factorial(j) for j in range(cap, 30)]
        expected = math.fsum(terms)
        got = float(count_pmf(0, .9, nu, cap)[cap])
        self.assertGreater(got, first)
        self.assertAlmostEqual(got / expected, 1., places=13)
        static = np.zeros((2, 1, 1, 3), dtype=np.int64)
        observation = np.array([[[8, 8]], [[8, 8]]], dtype=np.int64)
        result = encode_batch(static, observation, cap=cap, nu=nu)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(np.isfinite(result["log_likelihood"]).all())
        np.testing.assert_allclose(result["log_likelihood"], 4 * math.log(expected), atol=1e-11, rtol=1e-12)

    def test_overflow_convolution_matches_positive_enumeration(self):
        for n, alpha, nu in itertools.product([0, 1, 7, 30], [.01, .5, .99], [.02, 2.]):
            cap = 8
            expected = 0.0
            for d in range(n + 1):
                pb = math.comb(n, d) * alpha ** d * (1-alpha) ** (n-d)
                if d >= cap:
                    expected += pb
                else:
                    expected += pb * math.fsum(math.exp(-nu) * nu ** j / math.factorial(j)
                                               for j in range(cap-d, 100))
            got = float(count_pmf(n, alpha, nu, cap)[cap])
            self.assertAlmostEqual(got / expected, 1., places=11)

    def test_large_latent_count_small_detection_retains_normalized_mass(self):
        # A finite positive enumeration of only the low-count terms is enough
        # to verify this extreme-n case; no n-sized loop or factorial is used.
        p = count_pmf(10**9, 1e-9, .02, 8)
        self.assertTrue(np.isfinite(p).all())
        self.assertGreater(p[-1], 0)
        self.assertAlmostEqual(float(p.sum()), 1., places=11)

    def test_handoff_carries_phase_and_nuisance_without_old_evidence(self):
        z = consume(memory())
        selected = int(np.argmax(z.query()["posterior"]))
        transfer = z.branch_handoff(selected)
        next_paths = np.broadcast_to(transfer["last_levels"][None, :, :, None], (2, 2, 1, 7)).copy()
        n = EvidenceMemory.from_handoff(GridKernel(next_paths, substeps=2, context_id="next-c0"), transfer)
        self.assertEqual(n.start_us, 3000)
        self.assertEqual(n.end, 0)
        self.assertEqual(n.seen, {})
        np.testing.assert_allclose(n.ell, 0)
        np.testing.assert_allclose(n.phi[0], transfer["phase"])
        np.testing.assert_allclose(np.exp(n.logomega[0]), transfer["omega"])
        self.assertLessEqual(n.memory_bytes()["array_bytes"], z.memory_bytes()["array_bytes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
