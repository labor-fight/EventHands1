"""MV2 M5 root-restricted self-prev arm (semkine/mv2_rootself.py): SLR path without the key; only the selected rows'
root block comes from the deployed rollout, the other blocks are GT(end - 50) + the recipe's noise; own generator."""
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import load_config  # noqa: E402
from semkine.slr import SLRModel  # noqa: E402
from semkine.mv2_rootself import RootSelfModel  # noqa: E402

SLR_CFG = REPO / "configs" / "mv2" / "mv2_slr.yaml"
RS_CFG = REPO / "configs" / "mv2" / "mv2_slr_rootself.yaml"


def _batch(B=4, K=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(B, 180, 240, 2, generator=g) > 0.97).float() * torch.rand(B, 180, 240, 2, generator=g)
    prev = torch.zeros(B, 51)
    prev[:, 2] = 0.55
    prev[:, 3:6] = torch.tensor([2.2, 0.1, -0.2])
    prev[:, 6:] = 0.05 * torch.randn(B, 45, generator=g)
    y = prev + 0.01 * torch.randn(B, 51, generator=g)
    betas = torch.zeros(B, 10)
    cam = torch.tensor([[[400.0, 0, 320], [0, 400.0, 240], [0, 0, 1]]]).repeat(B, 1, 1)
    codes = (torch.rand(B, K, 180, 240, 2, generator=g) > 0.98).to(torch.uint8) * 100
    tgt = prev.unsqueeze(1).repeat(1, K, 1) + 0.02 * torch.randn(B, K, 51, generator=g)
    batch = {"main": (x, prev, y, betas, cam), "valid": torch.ones(B, dtype=torch.bool),
             "clip_codes": codes, "clip_w": torch.full((B, K), 200.0), "clip_tgt": tgt,
             "clip_init": prev.clone(), "clip_n50": torch.full((B, K), 500.0, dtype=torch.float64)}
    return batch


def test_without_key_is_slr_bit_for_bit():
    torch.manual_seed(3407)
    a = SLRModel(load_config(SLR_CFG)).eval()
    torch.manual_seed(3407)
    b = RootSelfModel(load_config(SLR_CFG)).eval()
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_same_parameters_as_slr_arm():
    torch.manual_seed(3407)
    a = SLRModel(load_config(SLR_CFG))
    torch.manual_seed(3407)
    b = RootSelfModel(load_config(RS_CFG))
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_root_only_from_rollout_rest_gt_plus_noise():
    torch.manual_seed(0)
    m = RootSelfModel(load_config(RS_CFG)).train()
    m.self_ramp = (-1, 0)                                 # p = SELF_P = 1 at global_step 0
    batch = _batch()
    K = batch["clip_codes"].shape[1]
    torch.manual_seed(5)
    x, prev, y, _, _ = m._maybe_self(batch)
    assert m._self_stats["rows"] == 4
    gt_prev = batch["clip_tgt"][:, K - 2]
    assert torch.equal(y, batch["clip_tgt"][:, K - 1])
    # root block: the deployed rollout's (reproduce it with the parent's S42-B path from the same RNG state)
    torch.manual_seed(5)
    m.rootself = False
    _, prev_b, _, _, _ = m._maybe_self(batch)
    m.rootself = True
    assert torch.allclose(prev[:, 3:6], prev_b[:, 3:6], atol=1e-6)
    # other blocks: GT(end - 50) + noise of the recipe's scale (small 5 mm / 0.05, large 50 mm / 0.3)
    dt, df = (prev[:, :3] - gt_prev[:, :3]).abs(), (prev[:, 6:] - gt_prev[:, 6:]).abs()
    assert dt.max() < 0.3 and df.max() < 2.0 and dt.max() > 0
    assert not torch.allclose(prev[:, :3], prev_b[:, :3])


def test_noise_model_matches_dataset_scales():
    m = RootSelfModel(load_config(RS_CFG))
    nt, nf = m.rootself_noise(20000, 45)
    rt = nt.norm(dim=1)
    # mixed: 50 % small (sigma 5 mm), 30 % large (50 mm), 20 % directed (|t| ~ U[0, 50 mm])
    assert 0.40 < float((rt < 0.02).double().mean()) < 0.62
    assert float(rt.max()) < 0.4
    assert nf.shape == (20000, 45)


def test_stray_keys_rejected():
    cfg = load_config(SLR_CFG)
    cfg["MODEL"]["ROOTSELF_SEED"] = 1
    with pytest.raises(ValueError):
        RootSelfModel(cfg)
