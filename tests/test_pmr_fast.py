"""PMR fast evaluation path, CPU (semkine/pmr_fast.py; docs/EXPERIMENT_RECORDS.md#pmr-s3c-prereg-20261109). Run with
CUDA_VISIBLE_DEVICES="".

The fast path is the reference function reassociated: frames, node statistics, node inputs and the increment equal the
reference within floating-point rounding for the three variants (evaluation mode, float64 statistics).
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import pmr_fast as F  # noqa: E402
from semkine import pmr_trunk as PM  # noqa: E402
from tests.test_pmr import _inputs, _net, _redraw_zero_layers  # noqa: E402


def test_frames_equal_reference():
    mano, prev, _ = _inputs(B=4)
    a = PM.node_frames(prev[:, 3:51].double(), mano.hands_mean.double())
    b = F.node_frames_fast(prev[:, 3:51].double(), mano.hands_mean.double())
    assert torch.allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("variant", PM.VARIANTS)
def test_fast_path_equals_reference(variant):
    _, _, args = _inputs(B=3, N=300)
    ref = _net(variant).eval()
    _redraw_zero_layers(ref)
    fast = _net(variant).eval()
    _redraw_zero_layers(fast)
    F.fastify(fast)
    with torch.no_grad():
        a, b = ref.node_inputs(*args), fast.node_inputs(*args)
        for k in ("x_part", "x_hub", "sp", "sh", "Mp", "Mh"):
            assert torch.allclose(a[k], b[k], rtol=1e-9, atol=1e-9), k
        da, _, _ = ref(*args)
        db, _, _ = fast(*args)
    assert torch.allclose(da, db, rtol=1e-4, atol=1e-6), float((da - db).abs().max())
    assert float(da.abs().max()) > 0
