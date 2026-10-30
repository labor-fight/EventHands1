"""S45-RF P1 (docs/EXPERIMENT_RECORDS.md#s45rf-prereg-20261030): the pull decomposition recovers known gains, is blind
to per-recording constants, reads a state-only correction as K = 0 and a pull toward the packet's pose as K = a_e,
maps shifted packets / lagged priors as registered, and classifies at the registered boundaries."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from tools.s45rf import p1 as P  # noqa: E402

F = P.BLOCKS["finger"]


def synthetic(rng, n=400, a_m=0.1, pull=0.12, state=0.05, beta=0.04, c_m=None, noise=0.0, offset=0.0, k=5):
    """Own packet: pull toward the true end pose + a state-only correction toward the true previous pose;
    shifted packet: the same pull toward the shifted packet's end pose, the same state-only correction."""
    walk = np.cumsum(rng.normal(0, 0.05, (n + k + 1, 51)), 0)
    gt_prev, target = walk[:n], walk[1:n + 1]
    incoming = gt_prev + rng.normal(0, 0.3, (n, 51))
    mu = rng.normal(0, 0.5, 45)
    j = P.shifted_index(np.arange(n), n, k)
    s_prev, s_end = gt_prev[j], target[j]
    motion_only = a_m - pull                      # own-packet motion gain beyond the pull

    def respond(end, start):
        d = np.zeros((n, 51))
        d[:, F] = (motion_only * (end - start) + pull * (end - incoming) + state * (gt_prev - incoming))[:, F] \
            + beta * (mu[None] - incoming[:, F]) + offset + noise * rng.normal(size=(n, 45))
        return incoming + d
    true = {"s/incoming_used": incoming, "s/gt_previous": gt_prev, "s/target": target,
            "s/prediction": respond(target, gt_prev), "s/end_ms": np.arange(n)}
    shift = {"s/events_gt_previous": s_prev, "s/events_target": s_end, "s/prediction": respond(s_end, s_prev)}
    return true, shift, mu


def fit(true, shift, mu):
    d = P.design(true, shift, "s", F, mu)
    ft = P.solve([P.block_sums(d, "D", ("Mc", "E", "Bm"))])
    fs = P.solve([P.block_sums(d, "W", ("Mw", "Sw", "E", "Bm"))])
    return ft, fs


def test_decomposition_recovers_pull_and_state_gains_exactly():
    true, shift, mu = synthetic(np.random.default_rng(0))
    (a_m, a_e, beta), (c_m, k, c_e, c_b) = fit(true, shift, mu)
    assert np.allclose([a_m, a_e, beta], [0.1, 0.12 + 0.05, 0.04])
    assert np.allclose([c_m, k, c_e, c_b], [0.1, 0.12, 0.17, 0.04])
    assert np.isclose(k / a_e, 0.12 / 0.17)


def test_state_only_correction_reads_as_zero_pull():
    true, shift, mu = synthetic(np.random.default_rng(1), pull=0.0, state=0.15, a_m=0.05)
    (_, a_e, _), (_, k, c_e, _) = fit(true, shift, mu)
    assert abs(k) < 1e-9 and np.isclose(c_e, a_e) and np.isclose(a_e, 0.15)


def test_pure_pull_reads_as_full_share():
    true, shift, mu = synthetic(np.random.default_rng(2), pull=0.15, state=0.0, a_m=0.15)
    (_, a_e, _), (_, k, _, _) = fit(true, shift, mu)
    assert np.isclose(k / a_e, 1.0)


def test_per_recording_constants_do_not_change_the_gains():
    a = fit(*synthetic(np.random.default_rng(3)))
    b = fit(*synthetic(np.random.default_rng(3), offset=0.7))
    assert all(np.allclose(x, y) for x, y in zip(a, b))


def test_bootstrap_is_paired_reproducible_and_brackets_the_estimate():
    true, shift, mu = synthetic(np.random.default_rng(4), noise=0.05)
    d = [P.design(true, shift, "s", F, mu)]
    ft, fs, bt, bs = P.decompose(d, True, nboot=300, seed=7)
    ft2, fs2, bt2, bs2 = P.decompose(d, True, nboot=300, seed=7)
    assert np.array_equal(bt, bt2) and np.array_equal(bs, bs2)
    r = P.summarize(ft, fs, bt, bs)
    assert r["share_ci95"][0] <= r["share"] <= r["share_ci95"][1] and r["consistent"]


def test_shift_and_lag_mappings():
    assert [P.shifted_index(i, 10, 5) for i in (0, 4, 5, 9)] == [5, 9, 0, 4]
    assert P.shifted_index(3, 10, 0) == 3
    assert P.lagged_ms(1000, 0, 25) == 975 and P.lagged_ms(1000, 990, 25) == 990 and P.lagged_ms(1000, 1000, 25) == 1000


def test_classification_boundaries_and_consistency_tolerance():
    assert P.classify(0.5, 0.3, 0.9) == "event" and P.classify(0.49, 0.3, 0.9) == "uncertain"
    assert P.classify(0.6, 0.29, 0.9) == "uncertain"
    assert P.classify(0.3, 0.0, 0.5) == "state" and P.classify(0.3, 0.0, 0.51) == "uncertain"
    assert P.tol(0.05) == 0.04 and np.isclose(P.tol(0.2), 0.06)


def test_registered_jobs():
    assert len(P.JOBS) == 19 and ("n4", "e3_lag25") in P.JOBS and ("t1a", "e3") not in P.JOBS
    assert P.SHIFT_ORDER[0] == "e2_shift5" and P.TASKS["e2_true"] == ("e2", 0)
