"""TPD causal deployment form CPU tests (docs/EXPERIMENT_RECORDS.md#tpd-causal-prereg-20261105).

- The causal tokens equal B2's tokens except where a pixel has not fired the polarity yet in the packet (B2: the packet's
  actual span / Δt; causal: exactly 1), and a token depends only on earlier events.
- The causal stride takes j % k == 0, j // k < cap in arrival order; k = max(1, ceil(n_prev / cap)).
- install / uninstall swap and restore both module attributes.
- The rule.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import tpd_causal as TC  # noqa: E402


def _events(n, seed=0, packets=2):
    g = torch.Generator().manual_seed(seed)
    b = torch.sort(torch.randint(0, packets, (n,), generator=g)).values
    ev = torch.zeros(n, 5)
    ev[:, 0] = b.float()
    ev[:, 1] = torch.randint(0, 6, (n,), generator=g).float()
    ev[:, 2] = torch.randint(0, 5, (n,), generator=g).float()
    t = torch.rand(n, generator=g) * 0.04
    for p in range(packets):
        m = b == p
        ev[m, 3] = torch.sort(t[m]).values
    ev[:, 4] = torch.randint(0, 2, (n,), generator=g).float()
    ptr = torch.zeros(packets + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(torch.bincount(b, minlength=packets), 0)
    return ev, ptr


def test_tokens():
    from semkine import encoder as ENC
    ev, ptr = _events(400)
    dt = torch.tensor([0.05, 0.05])
    a = ENC.event_tokens(ev, ptr, dt, 5, 6)
    c = TC.event_tokens_causal(ev, ptr, dt, 5, 6)
    assert torch.equal(a[:, :5], c[:, :5])
    for col in (5, 6):
        diff = a[:, col] != c[:, col]
        assert torch.all(c[diff, col] == 1.0) and diff.any()
    # prefix: replacing the second half of packet 0 leaves its first half's tokens unchanged
    n0 = int(ptr[1])
    ev2 = ev.clone()
    ev2[n0 // 2:n0, 1:3] = torch.flip(ev2[n0 // 2:n0, 1:3], [0])
    ev2[n0 // 2:n0, 3] = torch.linspace(float(ev[n0 // 2, 3]), 0.049, n0 - n0 // 2)
    c2 = TC.event_tokens_causal(ev2, ptr, dt, 5, 6)
    assert torch.equal(c2[: n0 // 2], c[: n0 // 2])
    a2 = ENC.event_tokens(ev2, ptr, dt, 5, 6)
    assert not torch.equal(a2[: n0 // 2], a[: n0 // 2])          # B2's fill reads the packet's last event


def test_stride_and_k():
    ptr = torch.tensor([0, 10, 10, 50])
    src, mask = TC.causal_stride(ptr, torch.tensor([3, 1, 4]), cap=5)
    assert src[0][mask[0]].tolist() == [0, 3, 6, 9] and mask[1].sum() == 0
    assert src[2][mask[2]].tolist() == [10, 14, 18, 22, 26]
    assert TC.k_from_prev([0, 2048, 2049, 10000]).tolist() == [1, 1, 2, 5]


def test_install_and_rule():
    from semkine import encoder as ENC
    from semkine import xyz_mesh as XM
    native_s, native_t = XM.stride_sample, ENC.event_tokens
    TC.install()
    try:
        assert XM.stride_sample is TC.stride_sample_causal and ENC.event_tokens is TC.event_tokens_causal
    finally:
        TC.uninstall()
    assert XM.stride_sample is native_s and ENC.event_tokens is native_t
    from tools.tpd import run_causal as RC
    assert RC.rule(0.15) == "TEST_ONLY" and RC.rule(0.16) == "TRAIN_CAUSAL"
