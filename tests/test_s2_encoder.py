#!/usr/bin/env python3
r"""S2 gates for the raw-event encoder and its sparse-cell fallback.

These are mechanism gates, not tracking gates. The registered tracking parity (paired CI
upper bound < 0.5 mm versus the same-protocol domrand baseline) needs a trained student and
is deferred to the S2 training run. What can be falsified without training:

* tokens have the registered seven coordinates, in range
* the scan is causal and segmented (a later packet cannot change an earlier one)
* a zero-event packet is the zero feature, which with PREDICT_DELTA is bitwise prev
* the fallback never allocates a dense HxW convolution
* querying a render at event pixels matches direct indexing
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from semkine.encoder import (                                         # noqa: E402
    TOKEN_DIM, RawEventEncoder, SparseCellEncoder, event_tokens,
    inter_event_dt, query_render, sae_times, build_encoder,
)
from semkine.events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y            # noqa: E402
from model import MNISTModel                                          # noqa: E402

DEV = torch.device("cpu")


def _packet(events_xypt, dt=0.05):
    """`events_xypt` is a list of (x, y, t, p) tuples for a single packet."""
    n = len(events_xypt)
    ev = torch.zeros(n, 5)
    for i, (x, y, t, p) in enumerate(events_xypt):
        ev[i, EV_X], ev[i, EV_Y], ev[i, EV_T], ev[i, EV_P] = x, y, t, p
    ptr = torch.tensor([0, n], dtype=torch.long)
    return ev, ptr, torch.tensor([dt])


def test_token_layout_and_ranges():
    ev, ptr, dt = _packet([(10, 20, 0.00, 0), (10, 20, 0.01, 1), (11, 20, 0.02, 1)])
    tok = event_tokens(ev, ptr, dt, 180, 240)
    assert tok.shape == (3, TOKEN_DIM)
    assert torch.allclose(tok[0, 0], torch.tensor(10 / 240))
    assert torch.allclose(tok[0, 1], torch.tensor(20 / 180))
    assert torch.allclose(tok[:, 2], torch.tensor([-1.0, 1.0, 1.0]))
    assert float(tok[:, 3].max()) <= 1.0 + 1e-6
    assert float(tok[:, 5].min()) >= 0.0 and float(tok[:, 5].max()) <= 1.0
    assert float(tok[:, 6].min()) >= 0.0 and float(tok[:, 6].max()) <= 1.0


def test_inter_event_dt_is_intra_packet():
    # Two packets; the first event of packet 1 must not see packet 0's last timestamp.
    ev = torch.tensor([
        [0, 1, 1, 0.00, 1],
        [0, 1, 1, 0.02, 1],
        [1, 2, 2, 0.00, 1],
        [1, 2, 2, 0.01, 1],
    ], dtype=torch.float32)
    ptr = torch.tensor([0, 2, 4])
    dt = inter_event_dt(ev, ptr)
    assert float(dt[0]) == 0.0
    assert abs(float(dt[1]) - 0.02) < 1e-6
    assert float(dt[2]) == 0.0
    assert abs(float(dt[3]) - 0.01) < 1e-6


def test_sae_same_and_opposite_at_one_pixel():
    ev, ptr, dt = _packet([
        (5, 5, 0.00, 0),
        (5, 5, 0.01, 1),
        (5, 5, 0.03, 1),
    ])
    same, opp = sae_times(ev, ptr, 180, 240)
    # First event: no history, SAE = packet span 0.03
    assert abs(float(same[0]) - 0.03) < 1e-5
    assert abs(float(opp[0]) - 0.03) < 1e-5
    # Second: opposite of first (0.01), no same-polarity history
    assert abs(float(opp[1]) - 0.01) < 1e-5
    assert abs(float(same[1]) - 0.03) < 1e-5
    # Third: same polarity as second (0.02), opposite is first (0.03)
    assert abs(float(same[2]) - 0.02) < 1e-5
    assert abs(float(opp[2]) - 0.03) < 1e-5


def test_scan_is_causal():
    """Changing the last event of a packet must not change the first hidden state."""
    enc = RawEventEncoder(hidden=16, feat_dim=32).eval()
    ev, ptr, dt = _packet([(i, i, 0.001 * i, i % 2) for i in range(8)])
    with torch.no_grad():
        tok = enc.tokens(ev, ptr, dt)
        from semkine.encoder import gated_scan, inter_event_dt
        h0 = gated_scan(tok, inter_event_dt(ev, ptr), ptr, enc.w_forget, enc.w_input,
                        enc.log_decay)
        ev2 = ev.clone()
        ev2[-1, EV_X] = 100
        tok2 = enc.tokens(ev2, ptr, dt)
        h1 = gated_scan(tok2, inter_event_dt(ev2, ptr), ptr, enc.w_forget, enc.w_input,
                        enc.log_decay)
    assert torch.equal(h0[0], h1[0]), "the scan looked ahead"


def test_scan_is_segmented():
    """A second packet must not change the first packet's pooled feature."""
    enc = RawEventEncoder(hidden=16, feat_dim=32).to(DEV).eval()
    ev_a = torch.tensor([[0, 3, 3, 0.00, 1], [0, 4, 3, 0.02, 0]], dtype=torch.float32)
    ev_b = torch.tensor([[1, 8, 8, 0.00, 1], [1, 9, 8, 0.01, 1], [1, 9, 9, 0.04, 0]],
                        dtype=torch.float32)
    ev = torch.cat([ev_a, ev_b], 0)
    ptr = torch.tensor([0, 2, 5])
    dt = torch.tensor([0.05, 0.05])
    with torch.no_grad():
        both = enc(ev, ptr, dt)
        only_a = enc(ev_a, torch.tensor([0, 2]), dt[:1])
    assert torch.allclose(both[0], only_a[0], atol=1e-5), "packets leaked into each other"


def test_empty_packet_is_zero_feature():
    enc = RawEventEncoder(hidden=8, feat_dim=16).eval()
    ev = torch.zeros(0, 5)
    ptr = torch.tensor([0, 0, 0])
    dt = torch.tensor([0.05, 0.05])
    with torch.no_grad():
        feat = enc(ev, ptr, dt)
    assert feat.shape == (2, 16)
    assert torch.equal(feat, torch.zeros_like(feat))


def test_zero_event_gate_is_bitwise_prev():
    cfg = {
        "MODEL": {"POSE_REPR": "mano_full_axis_angle", "OUTPUT_DIM": 51,
                  "PREDICT_DELTA": True, "ZERO_EVENT_GATE": True,
                  "ENCODER": "raw_scan", "ENCODER_HIDDEN": 16, "ENCODER_FEAT": 32},
        "MANO": {"NPZ": str(REPO / "assets/mano_right.npz")},
        "LOSS": {"TYPE": "mse_51d", "LAMBDA_POSE": 60.0, "LAMBDA_T": 30000.0,
                 "LAMBDA_R": 60.0, "NORMALIZER": 51, "LOG10": True},
        "DATA": {"HEIGHT": 180, "WIDTH": 240},
    }
    m = MNISTModel(cfg).eval()
    prev = torch.randn(2, 51)
    from semkine.events import EventPacketBatch
    batch = EventPacketBatch(
        events=torch.zeros(0, 5),
        ptr=torch.tensor([0, 0, 0]),
        sequence_id=torch.zeros(2, dtype=torch.long),
        t_start_us=torch.zeros(2, dtype=torch.long),
        t_end_us=torch.full((2,), 50000, dtype=torch.long),
        delta_t_s=torch.tensor([0.05, 0.05]),
        is_sequence_start=torch.zeros(2, dtype=torch.bool),
        is_sequence_end=torch.zeros(2, dtype=torch.bool),
        target=prev.clone(),
        prev_state=prev,
        betas=torch.zeros(2, 10),
        camera_K=torch.eye(3).expand(2, 3, 3).contiguous(),
    )
    with torch.no_grad():
        out = m.forward_packet(batch)
    assert torch.equal(out, prev)


def test_fallback_has_no_spatial_conv():
    enc = SparseCellEncoder(cell=16, hidden=16, feat_dim=32)
    convs = [mod for mod in enc.modules() if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d))]
    assert convs == [], "the fallback allocated a dense convolution"


def test_query_render_matches_indexing():
    rend = torch.randn(2, 180, 240, 2)
    ev = torch.tensor([[0, 10, 20, 0.0, 1], [1, 3, 7, 0.0, 0]], dtype=torch.float32)
    q = query_render(rend, ev)
    assert torch.equal(q[0], rend[0, 20, 10])
    assert torch.equal(q[1], rend[1, 7, 3])


def test_build_encoder_names():
    assert isinstance(build_encoder("raw_scan", hidden=8, feat_dim=16), RawEventEncoder)
    assert isinstance(build_encoder("sparse_cell", hidden=8, feat_dim=16), SparseCellEncoder)
    with pytest.raises(ValueError):
        build_encoder("resnet")


def test_default_model_has_no_encoder():
    cfg = {
        "MODEL": {"POSE_REPR": "mano_full_axis_angle", "OUTPUT_DIM": 51},
        "MANO": {"NPZ": str(REPO / "assets/mano_right.npz")},
        "LOSS": {"TYPE": "mse_51d", "LAMBDA_POSE": 60.0, "LAMBDA_T": 30000.0,
                 "LAMBDA_R": 60.0, "NORMALIZER": 51, "LOG10": True},
        "DATA": {"HEIGHT": 180, "WIDTH": 240},
    }
    m = MNISTModel(cfg)
    assert m.encoder_name == ""
    assert m.rn is not None
    assert m.conv1 is not None
