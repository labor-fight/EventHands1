"""S22 / E5.5b contracts: the `(leading, main)` unroll pair and the mixed conditioning.

The debug-3eaac2 fixed-point analysis showed both frontends' recursive loops settling far above
their independent-noise fixed points because training never samples the loop's own conditioning
distribution (the model's drifted previous output). The pair is how that distribution enters
training; these tests pin the tiling, the shared augmentation, and the conditioning switch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from model import MNISTModel                                    # noqa: E402
from semkine import domrand as DR                               # noqa: E402
from semkine.dataset import SemKineDataset, sequences_for_split # noqa: E402
from semkine.events import EventPacket, collate_packets         # noqa: E402

H, W, SCALE = 180, 240, 0.375
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = REPO / "data/hand_data51"
MANO_NPZ = REPO / "assets/mano_right.npz"

pytestmark = pytest.mark.skipif(not ROOT.exists(), reason="hand_data51 not available")


def _dataset(**over):
    kw = dict(
        root=ROOT, sequences=sequences_for_split(ROOT, "train")[:2],
        components=np.load(MANO_NPZ)["hands_components"].astype(np.float32),
        mano_npz=str(MANO_NPZ), input_mode="raw_packed",
        window_min=5, window_max=50, width=W, height=H, train=True,
        domrand=DR.DomRandConfig(enabled=False), prev_noise=(0.0, 0.0, 0.0),
        seed=7, unroll_pair=True,
    )
    kw.update(over)
    return SemKineDataset(**kw)


def test_pair_tiles_like_the_recursive_evaluator():
    ds = _dataset()
    hits = 0
    for idx in np.linspace(0, len(ds) - 1, 40).astype(int):
        lead, main = ds[int(idx)]
        assert isinstance(lead, EventPacket) and isinstance(main, EventPacket)
        # The leading window ends exactly where the main one starts -- the evaluator's tiling.
        # An empty leading packet (main window at the run boundary) collapses to a zero-length
        # interval at that same instant, which the zero-event gate turns into the identity.
        assert lead.t_end_us == main.t_start_us
        assert lead.meta.get("leading") is True
        # With domrand off and zero conditioning noise, the lead's target *is* the main's
        # conditioning state: the model's lead prediction estimates exactly what the main
        # window is conditioned on.
        assert np.allclose(lead.target, main.prev_state)
        assert np.array_equal(lead.betas, main.betas)
        assert np.array_equal(lead.camera_K, main.camera_K)
        if lead.n_events:
            hits += 1
            t = lead.events[:, 2]
            assert float(t.min()) >= 0.0
            assert float(t.max()) <= lead.delta_t_s + 1e-6
    assert hits > 10, f"only {hits}/40 sampled pairs had a non-empty leading window"


def test_pair_shares_the_virtual_camera_under_domrand():
    ds = _dataset(domrand=DR.DomRandConfig(enabled=True, roll_deg=15.0, scale_min=0.8,
                                           scale_max=1.25, shift_px=14.0))
    for idx in np.linspace(0, len(ds) - 1, 12).astype(int):
        lead, main = ds[int(idx)]
        # One virtual camera per sample: the same domrand draw must transform both windows.
        assert np.array_equal(lead.camera_K, main.camera_K)


def _cfg():
    return {
        "MODEL": {
            "POSE_REPR": "mano_full_axis_angle", "OUTPUT_DIM": 51, "PREDICT_DELTA": True,
            "PREV_RENDER": True, "ZERO_EVENT_GATE": True, "ACTIVE_HEAD": True,
            # Any frontend does: these tests pin the unroll pair, not the encoder. `keg` stood
            # here until it was retired (`docs/GNN_ARMS_ARCHIVE_20260828.md`).
            "ENCODER": "event_gnn", "ENCODER_HIDDEN": 32, "ENCODER_FEAT": 64,
            "ENCODER_MAX_NODES": 64, "ENCODER_WINDOW": 8, "ENCODER_K": 4,
        },
        "DATA": {"HEIGHT": H, "WIDTH": W},
        "MANO": {"NPZ": str(MANO_NPZ)},
        "TRACK": {"UNROLL_PAIR": True, "UNROLL_P": 1.0},
        "LOSS": {"TYPE": "mse_51d", "LAMBDA_POSE": 1.0, "LAMBDA_T": 1.0, "LAMBDA_R": 1.0,
                 "NORMALIZER": 51, "LOG10": False},
        "TRAIN": {"LR": 1e-3, "WARMUP_STEPS": 0},
    }


def test_unroll_conditions_main_on_the_lead_prediction():
    ds = _dataset(prev_noise=(0.005, 0.05, 0.05))
    items = [ds[int(i)] for i in np.linspace(0, len(ds) - 1, 6).astype(int)]
    batch = tuple(collate_packets(list(col)) for col in zip(*items))
    batch = tuple(b.to(DEV) for b in batch)
    m = MNISTModel(_cfg()).to(DEV)

    m.train()
    assert m.unroll_p == 1.0
    before = batch[1].prev_state.clone()
    pred, target, betas, _packed = m._predict_batch(batch)
    # At UNROLL_P = 1 every main window must be re-conditioned on the lead prediction. The lead
    # windows here are real event packets, so the prediction differs from the noised GT prev.
    assert not torch.allclose(batch[1].prev_state, before)
    # The swapped-in state carries no graph: the unroll is stop-gradiented by contract.
    assert not batch[1].prev_state.requires_grad
    loss, _ = m._compute_loss(pred, target, betas)
    assert torch.isfinite(loss)
    loss.backward()

    # At UNROLL_P = 0 the pair degenerates to the plain teacher-forced sample.
    m.zero_grad(set_to_none=True)
    m.unroll_p = 0.0
    b2 = tuple(collate_packets(list(col)) for col in zip(*items))
    b2 = tuple(b.to(DEV) for b in b2)
    keep = b2[1].prev_state.clone()
    m._predict_batch(b2)
    assert torch.equal(b2[1].prev_state, keep)


def test_residual_unroll_adds_the_lead_error_on_top_of_the_noise():
    """E5.5c's contract. Replacement unroll conditioned on the near-clean lead prediction and
    thereby *narrowed* the error curriculum (probe 2026-08-25: g-slope 0.29 -> 0.38-0.42,
    recursive 27.8 -> 31-34, echo x1.85 -> x2.2). The residual form must keep the noised state
    and add exactly the lead's prediction error on top."""
    ds = _dataset(prev_noise=(0.005, 0.05, 0.05))
    items = [ds[int(i)] for i in np.linspace(0, len(ds) - 1, 6).astype(int)]
    batch = tuple(collate_packets(list(col)) for col in zip(*items))
    batch = tuple(b.to(DEV) for b in batch)
    lead, main = batch
    cfg = _cfg()
    cfg["TRACK"]["UNROLL_RESIDUAL"] = True
    m = MNISTModel(cfg).to(DEV)
    m.train()
    assert m.unroll_residual and m.unroll_p_now() == 1.0

    before = main.prev_state.clone()
    with torch.no_grad():
        est = m.forward_packet(lead)
    m._predict_batch(batch)
    # Conditioning = (GT + curriculum noise) + (lead prediction - lead target). The atol absorbs
    # scatter-order nondeterminism between our reference forward and the one inside the unroll.
    want = before + (est - lead.target).to(before.dtype)
    assert torch.allclose(main.prev_state, want, atol=2e-3)
    # The curriculum noise survives: the conditioning is NOT the near-clean prediction itself
    # (that was s22's measured mistake), and it is not the untouched noised state either.
    assert not torch.allclose(main.prev_state, est.to(before.dtype), atol=1e-3)
    assert not torch.allclose(main.prev_state, before, atol=1e-3)
    assert not main.prev_state.requires_grad


def test_unroll_probability_anneals_with_the_ramp():
    """The constant-P failure mode, pinned. Trained at a constant 0.5 from step 0 the network
    learned to ignore prev and collapsed to absolute-regression accuracy (TF RA 42.5 mm vs 12.1
    without unroll). The ramp must hold p at zero through the early garbage-prediction phase and
    reach the full value only at the configured step."""
    cfg = _cfg()
    cfg["TRACK"]["UNROLL_RAMP"] = [500, 2000]
    m = MNISTModel(cfg)
    s0, s1 = m.unroll_ramp
    assert (s0, s1) == (500, 2000)

    def p_at(step):
        frac = 1.0 if s1 <= s0 else min(max((step - s0) / float(s1 - s0), 0.0), 1.0)
        return m.unroll_p * frac

    assert p_at(0) == 0.0 and p_at(500) == 0.0
    assert abs(p_at(1250) - 0.5 * m.unroll_p) < 1e-9
    assert p_at(2000) == m.unroll_p == 1.0
    assert p_at(4000) == m.unroll_p
    # Default ramp (0, 0) keeps full mixing immediately -- the contract the tests above rely on.
    assert MNISTModel(_cfg()).unroll_ramp == (0, 0)
