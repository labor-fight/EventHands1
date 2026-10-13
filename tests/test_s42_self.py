"""S42 (trk6) candidate B: DATA.CLIP_* clips and TRACK.SELF_* deployed-loop self-conditioning
(docs/EXPERIMENT_RECORDS.md#s42-prereg). CPU only; reads the real training data."""
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "model"))
from config import load_config                     # noqa: E402
from model import MNISTModel                       # noqa: E402
from semkine import eval_track as ET               # noqa: E402
from semkine.anchored import AdaptiveFilter        # noqa: E402
from semkine.dataset import build_dataset          # noqa: E402

SPEC = REPO / "outputs/dt3/gate_specs/ramp1000.json"
BASE = load_config(str(REPO / "configs/dt/dt_dz_l3_2k.yaml"))
COMPS = np.load(BASE["MANO"]["NPZ"])["hands_components"].astype(np.float32)


def _ds(clip, fast, aug):
    c = copy.deepcopy(BASE)
    if clip:
        c["DATA"].update({"CLIP_K": 4, "CLIP_WINDOW": 200, "CLIP_FRAC": 1.0})
    if not aug:
        c["AUG"]["DOMRAND"]["ENABLED"] = False
        c["AUG"]["POLARITY_FLIP"] = False
        c["AUG"]["PIXEL_POLARITY_SWAP"] = False
    old = os.environ.get("SEMKINE_FAST_PIPE")
    os.environ["SEMKINE_FAST_PIPE"] = "1" if fast else "0"
    try:
        return build_dataset(c, "train", COMPS, train=True)
    finally:
        if old is None:
            os.environ.pop("SEMKINE_FAST_PIPE", None)
        else:
            os.environ["SEMKINE_FAST_PIPE"] = old


@pytest.mark.parametrize("fast", [False, True])
def test_clip_windows_are_the_evaluation_lnes(fast):
    ds = _ds(True, fast, aug=False)
    rng = np.random.default_rng(1)
    seen = 0
    for idx in rng.choice(len(ds), 40, replace=False):
        it = ds[int(idx)]
        if not it["valid"]:
            continue
        seen += 1
        h = ds._handle(int(ds.index[idx][0]))
        run_a = int(ds.index[idx][2])
        end = int(ds._window(int(idx), np.random.default_rng((ds.seed * 1_000_003 + int(idx)) & 0x7FFFFFFF))[1])
        for k, e in enumerate(end - 50 * (3 - k) for k in range(4)):
            w = min(200, e - run_a + 1)
            assert np.array_equal((it["clip_codes"][k].float() / it["clip_w"][k]).numpy(),
                                  ET.build_lnes(h.events, h.offsets, e, w, ("last",)))
            assert float(it["clip_n50"][k]) == float(h.offsets[e + 1] - h.offsets[e - 49])
    assert seen >= 20


@pytest.mark.parametrize("fast", [False, True])
def test_plain_samples_unchanged(fast):
    a, b = _ds(False, fast, aug=True), _ds(True, fast, aug=True)
    for idx in (7, 4321, 999_999):
        assert all(torch.equal(u, v) for u, v in zip(a[idx], b[idx]["main"]))


def _self_cfg():
    c = copy.deepcopy(BASE)
    c["DATA"]["CLIP_K"] = 4
    c["TRACK"].update({"SELF_P": 0.5, "SELF_RAMP": [0, 10], "SELF_FILTER": str(SPEC)})
    return c


def test_batch_filter_equals_adaptive_filter():
    m = MNISTModel(_self_cfg()).eval()
    torch.manual_seed(0)
    B = 32
    prev = torch.randn(B, 51) * 0.3
    z = prev + torch.randn(B, 51) * 0.1
    x = (torch.rand(B, 30, 40, 2) > 0.9).float() * torch.rand(B, 30, 40, 2)
    x[:2] = 0
    n = torch.tensor(np.exp(np.random.default_rng(0).uniform(0, 11, B)), dtype=torch.float64)
    n[4] = 0

    class Fixed(torch.nn.Module):
        def forward(self, x, p, betas=None, camera_K=None):
            return z

    ref = AdaptiveFilter.from_spec(Fixed(), AdaptiveFilter.canonical_spec(json.loads(SPEC.read_text())))(x, prev, n_events=n)
    assert float((ref - m._self_filter(prev, z, n, x)).abs().max()) < 1e-6


def test_self_keys_need_clips_and_a_filter():
    c = _self_cfg()
    c["DATA"].pop("CLIP_K")
    with pytest.raises(ValueError):
        MNISTModel(c)
    c = _self_cfg()
    c["TRACK"].pop("SELF_FILTER")
    with pytest.raises(ValueError):
        MNISTModel(c)


def test_maybe_self_replaces_only_taken_rows_and_keeps_shapes():
    m = MNISTModel(_self_cfg()).train()
    B, K = 6, 4
    x = torch.rand(B, 180, 240, 2) * (torch.rand(B, 180, 240, 2) > 0.97)
    prev = torch.zeros(B, 51); prev[:, 2] = 0.5
    y = prev.clone()
    batch = {"main": (x, prev, y, torch.zeros(B, 10), torch.tensor([[214.0, 0, 120], [0, 214, 90], [0, 0, 1]]).expand(B, 3, 3) / 0.375),
             "valid": torch.tensor([True, False, True, True, False, True])}
    nv = int(batch["valid"].sum())
    batch["clip_codes"] = (torch.rand(nv, K, 180, 240, 2) > 0.97).to(torch.uint8) * 100
    batch["clip_w"] = torch.full((nv, K), 200.0)
    batch["clip_tgt"] = torch.randn(nv, K, 51) * 0.01 + prev[:1]
    batch["clip_init"] = prev[:nv].clone()
    batch["clip_n50"] = torch.full((nv, K), 5000.0, dtype=torch.float64)
    torch.manual_seed(0)
    m.self_p_now = lambda: 1.0                     # global_step 0 would give p = 0 on any ramp
    out = m._maybe_self(batch)
    assert m.training
    assert [t.shape for t in out] == [t.shape for t in batch["main"]]
    changed = (out[2] != y).any(dim=1)
    assert not changed[~batch["valid"]].any()
    assert int(changed.sum()) == m._self_stats["rows"] > 0
