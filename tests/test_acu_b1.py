"""ACU B1 CPU tests (docs/EXPERIMENT_RECORDS.md#acu-s1b-prereg-20261104). Run with CUDA_VISIBLE_DEVICES="".

* the B1 stream: branch shares, prior inside the window, independent of B2's stream
* B1's training samples equal B2's except the prior on the B1 branches (pairing on real training indices)
* collate carries the offsets; the token wrapper shifts only the time column, only by the offset
* the age term starts at zero: the B1 trunk's output equals the base trunk's output at initialisation
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
pytestmark = pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES", None) != "", reason="CPU tests: CUDA_VISIBLE_DEVICES=''")


def test_b1_stream_shares():
    from semkine import acu_b1 as B1
    br = np.array([B1.draw_prior_ms(3407, i, 1000, 1049)[1] for i in range(20000)])
    ms = np.array([B1.draw_prior_ms(3407, i, 1000, 1049)[0] for i in range(2000)])
    assert abs((br == 0).mean() - 0.5) < 0.02 and abs((br == 1).mean() - 0.25) < 0.02
    assert ms.min() >= 1000 and ms.max() <= 1049


def test_b1_pairing_with_b2_samples():
    from tools.acu import run_b1 as R
    rows = R.pairing("preflight")
    assert all(r["ok"] and r["offset_ok"] for r in rows), [r for r in rows if not r["ok"]]


def test_collate_and_tokens():
    import torch
    from semkine import acu_b1 as B1
    from semkine import encoder as ENC
    from semkine.events import EventPacket
    def pk(n, off):
        ev = np.zeros((n, 4), np.float32)
        ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3] = 10, 20, np.linspace(0, 0.049, n), 1
        p = EventPacket(ev, 0, 0, 50000, False, False, np.zeros(51, np.float32), np.zeros(51, np.float32),
                        np.zeros(10, np.float32), np.eye(3, dtype=np.float32))
        p.meta["acu_prior_offset_s"] = off
        return p
    b = B1.collate_b1([pk(5, 0.0), pk(4, 0.02)])
    assert torch.allclose(b.prior_offset_s, torch.tensor([0.0, 0.02]))
    native = ENC.event_tokens
    B1._STATE["tokens"] = native
    B1._STATE["offset"] = b.prior_offset_s
    t0 = native(b.events, b.ptr, b.delta_t_s, 180, 240)
    t1 = B1._tokens(b.events, b.ptr, b.delta_t_s, 180, 240)
    B1._STATE["offset"] = None
    assert torch.equal(t0[:5], t1[:5])
    assert torch.allclose(t1[5:, 3], t0[5:, 3] - 0.02 / 0.05)
    assert torch.equal(torch.cat([t0[5:, :3], t0[5:, 4:]], 1), torch.cat([t1[5:, :3], t1[5:, 4:]], 1))


def test_age_term_starts_at_zero():
    import torch
    from semkine import acu_b1 as B1
    from semkine import pfo_trunk4 as T4
    import numpy as np_
    with np_.load(ROOT / "assets/mano_right.npz", allow_pickle=False) as z:
        w, f = torch.tensor(np_.asarray(z["weights"], np_.float32)), torch.tensor(np_.asarray(z["f"], np_.int64))
    base_cls = type("G", (T4.PFOMeshNet4,), {"variant": "gshare"})
    torch.manual_seed(5)
    base = base_cls(w, f)
    torch.manual_seed(5)
    mine = B1.b1_trunk(base_cls)(w, f)
    sd = {k: v for k, v in mine.state_dict().items() if not k.startswith("age.")}
    assert all(torch.equal(sd[k], v) for k, v in base.state_dict().items())
    x = torch.randn(3, base.fc1.in_features)
    B1._STATE["age"] = torch.tensor([1.0, 0.5, 0.0])
    try:
        assert torch.equal(base.fc1(x), mine.fc1(x))
    finally:
        B1._STATE["age"] = None
