"""PFO per-vertex "self" implementation (semkine/pfo_trunk5.py) equals pfo_trunk4's "self" variants: CPU tests."""
from __future__ import annotations

import pytest
import torch

from tests.test_pfo import _net_and_inputs
from tests.test_pfo_trunk4 import _build


def test_trunk5_matches_trunk4_self_variants():
    from semkine import pfo_trunk4 as T4, pfo_trunk5 as T5
    _, args = _net_and_inputs()
    with torch.no_grad():
        for v5 in ("vtx", "vtx+pos", "vtx+noprev", "vtx+hframe", "vtx+pos+noprev"):
            a, b = _build(T4, T5.to_trunk4(v5)), _build(T5, v5)
            sa, sb = a.state_dict(), b.state_dict()
            assert list(sa) == list(sb), v5
            assert all(torch.equal(sa[k], sb[k]) for k in sa), v5
            ya, yb = a(*args), b(*args)
            assert torch.allclose(ya[0], yb[0], atol=1e-6, rtol=1e-5), v5
            assert torch.equal(ya[2]["S"], yb[2]["S"]) and torch.equal(ya[2]["O"], yb[2]["O"]), v5
            assert torch.allclose(ya[2]["slots"], yb[2]["slots"], atol=1e-5, rtol=1e-5), v5


def test_trunk5_layers_touch_observed_vertices_only():
    from semkine import pfo_trunk5 as T5
    _, args = _net_and_inputs()
    net = _build(T5, "vtx")
    seen = []
    hook = net.layers[0].slf.register_forward_hook(lambda m, i, o: seen.append(i[0].shape[0]))
    with torch.no_grad():
        out = net(*args)
    hook.remove()
    assert seen == [int(out[2]["O"].sum())]


def test_trunk5_variant_strings():
    from semkine import pfo_trunk5 as T5
    assert T5.to_trunk4("vtx+pos") == "self+pos"
    assert T5.from_trunk4("self+noprev") == "vtx+noprev"
    assert T5.from_trunk4("nomp") is None
    with pytest.raises(ValueError):
        T5.to_trunk4("nomp")
