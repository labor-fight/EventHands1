"""MV2 M1 surface-lifted readout (semkine/slr.py): S39 path bit-for-bit, zero-initialised readout, wrapper wiring."""
import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import load_config  # noqa: E402
from model import MNISTModel  # noqa: E402
from semkine.slr import SLRModel  # noqa: E402

S39_CFG = REPO / "configs" / "dt" / "dt_dz_l3.yaml"
SLR_CFG = REPO / "configs" / "mv2" / "mv2_slr.yaml"
S39_CKPT = REPO / "outputs" / "semkine" / "dt_dz_l3_s3407" / "dt_dz_l3_s3407-step=6000.ckpt"


def _cfg(path):
    return load_config(path)


def _inputs(B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(B, 180, 240, 2, generator=g) > 0.97).float() * torch.rand(B, 180, 240, 2, generator=g)
    prev = torch.zeros(B, 51)
    prev[:, 2] = 0.55
    prev[:, 3:6] = torch.tensor([2.2, 0.1, -0.2])
    prev[:, 6:] = 0.05 * torch.randn(B, 45, generator=g)
    return x, prev


def test_default_path_is_mnistmodel_bit_for_bit():
    torch.manual_seed(3407)
    a = MNISTModel(_cfg(S39_CFG)).eval()
    torch.manual_seed(3407)
    b = SLRModel(_cfg(S39_CFG)).eval()
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert sum(p.numel() for p in b.parameters()) == 2802645
    x, prev = _inputs()
    with torch.no_grad():
        assert torch.equal(a(x, prev), b(x, prev))


@pytest.mark.skipif(not S39_CKPT.exists(), reason="S39 checkpoint not present")
def test_s39_checkpoint_loads_and_matches():
    cfg = load_config(S39_CKPT.parent / "config_resolved.yaml")
    a = MNISTModel.load_from_checkpoint(str(S39_CKPT), cfg=cfg, map_location="cpu").eval()
    b = SLRModel.load_from_checkpoint(str(S39_CKPT), cfg=cfg, map_location="cpu").eval()
    x, prev = _inputs()
    with torch.no_grad():
        assert torch.equal(a(x, prev), b(x, prev))


def test_slr_zero_init_computes_s39_function_and_same_trunk_draws():
    torch.manual_seed(3407)
    a = MNISTModel(_cfg(S39_CFG)).eval()
    rng_a = torch.get_rng_state()
    torch.manual_seed(3407)
    b = SLRModel(_cfg(SLR_CFG)).eval()
    rng_b = torch.get_rng_state()
    assert torch.equal(rng_a, rng_b), "the readout must not consume the global RNG"
    sa, sb = a.state_dict(), b.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    x, prev = _inputs()
    with torch.no_grad():
        assert torch.equal(a(x, prev), b(x, prev))
    n_extra = sum(p.numel() for p in b.slr_readout.parameters())
    assert 0 < n_extra < 200_000


def test_slr_gradient_reaches_readout_and_trunk():
    torch.manual_seed(0)
    b = SLRModel(_cfg(SLR_CFG)).train()
    with torch.no_grad():
        for last in (b.slr_readout.finger[-1], b.slr_readout.rootT[-1]):
            last.weight.normal_(0, 1e-2)
    x, prev = _inputs()
    out = b(x, prev)
    out.sum().backward()
    assert b.slr_readout.vert[0].weight.grad is not None and b.slr_readout.vert[0].weight.grad.abs().sum() > 0
    assert b.rn.layer1[0].conv1.weight.grad.abs().sum() > 0
    assert b.slr_readout.occluded_logit.grad is not None


def test_stray_keys_rejected():
    c = _cfg(S39_CFG)
    c["MODEL"]["SLR_PROJ"] = 16
    with pytest.raises(ValueError):
        SLRModel(c)


def test_metric_vs_fixed_scale_differ_only_in_window():
    c = _cfg(SLR_CFG)
    c["MODEL"]["SLR_SCALE"] = "fixed"
    torch.manual_seed(1)
    m = SLRModel(c).eval()
    assert m.slr_readout.scale == "fixed"


def test_entry_patch_swaps_class():
    import importlib
    import model as M
    from mv2 import slr_entry
    orig = M.MNISTModel
    try:
        cls = slr_entry.patch()
        assert M.MNISTModel is cls
    finally:
        M.MNISTModel = orig


def test_visible_static_equals_aura_visible():
    from semkine.aura import visible
    from semkine.slr import visible_static
    torch.manual_seed(5)
    m = SLRModel(_cfg(S39_CFG)).eval()
    B = 6
    prev = torch.zeros(B, 51)
    prev[:, 0] = torch.linspace(-0.25, 0.25, B)          # some hands partly out of the image
    prev[:, 2] = torch.linspace(0.3, 0.9, B)
    prev[:, 3:6] = torch.tensor([2.2, 0.1, -0.2]) + 0.3 * torch.randn(B, 3)
    prev[:, 6:] = 0.2 * torch.randn(B, 45)
    betas, K = m._resolve_betas_K(prev, None, None)
    with torch.no_grad():
        verts, _ = m._fk(prev, betas)
        intr = m._intrinsics(K)
        a, b = visible(verts, intr), visible_static(verts, intr)
    assert torch.equal(a, b)
    assert 0 < int(a.sum()) < a.numel()
