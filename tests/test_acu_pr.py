"""ACU PR router CPU tests (docs/EXPERIMENT_RECORDS.md#acu-s1c-prereg-20261104). Run with CUDA_VISIBLE_DEVICES="".

* causal stride: j % k == 0, cap, arrival order; k from the previous interval's rate only
* causality on real training packets with history (float64, untrained weights):
  - changing every window event after node m leaves the outputs of nodes <= m bitwise unchanged
  - the outputs of node m from the whole window equal those from the window cut right after it (prefix equivalence, up to
    float64 rounding of the windowed context sums, <= 1e-9)
  - other packets of the batch are unaffected
* routing decodes with S45-SM's tables
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
pytestmark = pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES", None) != "", reason="CPU tests: CUDA_VISIBLE_DEVICES=''")


def test_causal_stride_rule():
    import torch
    from semkine import acu_pr as PR
    ptr = torch.tensor([0, 10, 10, 6010])
    src, mask = PR.causal_stride(ptr, torch.tensor([3, 1, 2]), cap=2048)
    assert src[0][mask[0]].tolist() == [0, 3, 6, 9]
    assert int(mask[1].sum()) == 0
    s2 = src[2][mask[2]]
    assert len(s2) == 2048 and int(s2[0]) == 10 and int(s2[1]) == 12
    assert PR.stride_k(0, 0.0, 0.05).item() == 1
    assert PR.stride_k(4096, 0.05, 0.05).item() == 2 and PR.stride_k(4097, 0.05, 0.05).item() == 3
    assert PR.stride_k(4096, 0.025, 0.05).item() == 4


@pytest.fixture(scope="module")
def batch_and_net():
    import torch
    from semkine import acu_hist as AH
    from semkine import acu_pr as PR
    from tools.s45sm import router_train as S
    torch.set_num_threads(8)
    cfg = S._data({"dz": True})
    AH.install_dataset()
    try:
        ds = S.dataset(cfg, "ylf")
        b = AH.collate_hist([ds[i] for i in (3, 7000, 30000)])
    finally:
        AH.uninstall()
    torch.manual_seed(0)
    return b, PR.build(cfg["MANO"]["NPZ"]).double().eval()


def _run(net, b, ev):
    import torch
    with torch.no_grad():
        return net(ev, b.ptr, b.delta_t_s, b.hist_events.double(), b.hist_ptr, b.k_win(), b.k_hist)


def test_future_events_do_not_change_earlier_nodes(batch_and_net):
    import torch
    b, net = batch_and_net
    ev = b.events.double()
    out = _run(net, b, ev)
    for bi in range(b.batch_size):
        n = int(out["mask"][bi].sum())
        if n < 4:
            continue
        m = n // 3
        cut = int(out["src"][bi, m])
        ev2 = ev.clone()
        later = (torch.arange(ev.shape[0]) > cut) & (ev[:, 0] == bi)
        g = torch.Generator().manual_seed(bi)
        ev2[later, 1] = torch.randint(0, 240, (int(later.sum()),), generator=g).double()
        ev2[later, 2] = torch.randint(0, 180, (int(later.sum()),), generator=g).double()
        out2 = _run(net, b, ev2)
        for k in ("part", "coord", "onhand"):
            assert torch.equal(out[k][bi, : m + 1], out2[k][bi, : m + 1]), (bi, k)
        for j in range(b.batch_size):
            if j != bi:
                assert torch.equal(out["part"][j], out2["part"][j])


def test_prefix_equivalence(batch_and_net):
    import torch
    b, net = batch_and_net
    ev = b.events.double()
    out = _run(net, b, ev)
    bi = 0
    n = int(out["mask"][bi].sum())
    for m in (0, n // 2, n - 1):
        cut = int(out["src"][bi, m]) + 1
        sub = type(b)(**{**b.__dict__})
        sub.events, sub.ptr = ev[:cut], torch.tensor([0, cut])
        hp = b.hist_ptr
        sub.hist_events, sub.hist_ptr = b.hist_events[: int(hp[1])], hp[:2]
        sub.delta_t_s, sub.n_hist, sub.hist_len_s, sub.k_hist = b.delta_t_s[:1], b.n_hist[:1], b.hist_len_s[:1], b.k_hist[:1]
        with torch.no_grad():
            part = net(sub.events, sub.ptr, sub.delta_t_s, sub.hist_events.double(), sub.hist_ptr, b.k_win()[:1], sub.k_hist)
        for k in ("part", "coord", "onhand"):
            d = float((out[k][bi, m] - part[k][0, m]).abs().max())
            assert d <= 1e-9, (m, k, d)       # float64 rounding of the windowed context sums only


def test_route_decodes(batch_and_net):
    import torch
    b, net = batch_and_net
    out = _run(net, b, b.events.double())
    face, bary = net.route({k: v.float() for k, v in out.items() if k in ("part", "coord")}, out["mask"])
    assert face.shape == out["mask"].shape and bary.shape == (*out["mask"].shape, 3)
    assert torch.isfinite(bary).all()
