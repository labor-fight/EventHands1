"""TPD EXT feasibility CPU tests (docs/EXPERIMENT_RECORDS.md#tpd-ext0-prereg-20261105): the pure pieces of
tools/tpd/ext_feas.py (finger of part, PCA axis, extents, Pearson, the written rule)."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd import ext_feas as X  # noqa: E402


def test_finger_of_part():
    assert X.finger_of_part([0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15]).tolist() == [-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_axis_and_extents():
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 40, 2000)
    xy = np.stack([10 + 0.6 * t, 20 + 0.8 * t], -1) + rng.normal(0, 0.5, (2000, 2))
    ax = X.pca_axis(xy)
    assert abs(abs(ax @ np.array([0.6, 0.8])) - 1) < 1e-3
    assert abs(X.extent(xy, ax) - 36.0) < 1.5
    chain = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0], [9.0, 12.0]])
    assert abs(X.span(chain, np.array([0.6, 0.8])) - 15.0) < 1e-9


def test_rule_and_pearson():
    assert X.rule(0.62, 0.40) == "TRAIN" and X.rule(0.62, 0.55) == "WEAK" and X.rule(0.45, 0.0) == "WEAK"
    assert X.rule(0.29, 0.0) == "KILL"
    a = np.arange(10.0)
    assert abs(X.pearson(a, 2 * a + 1) - 1) < 1e-12 and abs(X.pearson(a, -a) + 1) < 1e-12
