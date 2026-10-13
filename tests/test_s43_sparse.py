"""S43 sparse trunk: the masked dense training form and the active-site executor are one function; the model wiring is
opt-in (the S39 config builds the same network as before)."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "model")]
from config import load_config                     # noqa: E402
from model import MNISTModel                       # noqa: E402
from sparse_trunk import SparseExecutor, SparseResNet, input_mask   # noqa: E402

S39_CFG = REPO / "configs" / "dt" / "dt_dz_l3.yaml"


def _cfg(backbone):
    cfg = copy.deepcopy(load_config(str(S39_CFG)))
    cfg["MODEL"]["CNN_BACKBONE"] = backbone
    return cfg


def _sparse_image(rng, h=180, w=240, c=4, frac=0.05):
    img = np.zeros((h, w, c), np.float32)
    # a blob of "events" and a sprinkle of isolated sites (hot pixels), plus border sites
    yy, xx = np.mgrid[60:110, 90:160]
    keep = rng.random(yy.shape) < 0.4
    img[yy[keep], xx[keep], rng.integers(0, 2, keep.sum())] = rng.random(keep.sum()).astype(np.float32)
    n = int(frac * h * w * 0.1)
    img[rng.integers(0, h, n), rng.integers(0, w, n), rng.integers(0, c, n)] = rng.random(n).astype(np.float32)
    img[0, 0, 0] = 0.5
    img[h - 1, w - 1, 1] = 0.25
    img[100:104, 120:126, 2:4] = 0.7               # "render" channels
    return torch.from_numpy(img)


def _randomise_bn(net, rng):
    for mod in net.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            c = mod.num_features
            mod.running_mean.copy_(torch.from_numpy(rng.normal(0, 0.3, c).astype(np.float32)))
            mod.running_var.copy_(torch.from_numpy(rng.uniform(0.5, 2.0, c).astype(np.float32)))
            mod.weight.data.copy_(torch.from_numpy(rng.uniform(0.5, 1.5, c).astype(np.float32)))
            mod.bias.data.copy_(torch.from_numpy(rng.normal(0, 0.2, c).astype(np.float32)))


@pytest.mark.parametrize("coord,pool,dilate", [(True, "mean_area", 0), (False, "mean", 0), (True, "mean_area", 1)])
def test_executor_equals_masked_dense(coord, pool, dilate):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    trunk = SparseResNet(51, coord=coord, pool=pool, dilate=dilate)
    adapter = torch.nn.Conv2d(4, 3, 3, padding=1)
    _randomise_bn(trunk, rng)
    trunk.eval()
    ex = SparseExecutor(adapter, trunk)
    for k in range(3):
        img = _sparse_image(np.random.default_rng(k))
        x = img.permute(2, 0, 1)[None]
        m = input_mask(x)
        with torch.no_grad():
            ref = trunk(adapter(x) * m, m)[0]
        out, macs, sizes = ex.run(img)
        assert torch.allclose(out, ref, atol=1e-4, rtol=0), (out - ref).abs().max()
        assert sizes["A0"] == int(m.sum())
        assert 0 < macs


def test_masked_bn_ignores_inactive_sites():
    """Train-mode output on the same content inside a larger empty frame is the same: batch statistics are taken over
    active sites only (coord / area pooling off, since those two depend on the frame size by design)."""
    torch.manual_seed(0)
    adapter = torch.nn.Conv2d(4, 3, 3, padding=1)
    t_small = SparseResNet(51, coord=False, pool="mean")
    t_big = copy.deepcopy(t_small)
    t_small.train()
    t_big.train()
    x = _sparse_image(np.random.default_rng(1)).permute(2, 0, 1)[None].repeat(2, 1, 1, 1)
    big = torch.zeros(2, 4, 180, 240 + 64)
    big[..., :240] = x
    m, mb = input_mask(x), input_mask(big)
    with torch.no_grad():
        a = t_small(adapter(x) * m, m)
        b = t_big(adapter(big) * mb, mb)
    assert torch.allclose(a, b, atol=1e-4, rtol=0), (a - b).abs().max()
    assert torch.allclose(t_small.bn1.running_var, t_big.bn1.running_var, atol=1e-5)


def test_model_wiring_and_backward():
    cfg = _cfg("sparse_resnet18_l3")
    torch.manual_seed(0)
    model = MNISTModel(cfg)
    assert model.sparse_trunk
    model.train()
    lnes = _sparse_image(np.random.default_rng(2), c=2)[None].repeat(2, 1, 1, 1)
    prev = torch.zeros(2, 51)
    prev[:, 2] = 0.55
    betas = torch.zeros(2, 10)
    K = torch.tensor([[200.0, 0, 120.0], [0, 200.0, 90.0], [0, 0, 1.0]]).expand(2, 3, 3).contiguous()
    out = model(lnes, prev, betas=betas, camera_K=K)
    assert out.shape == (2, 51) and torch.isfinite(out).all()
    out.square().sum().backward()
    assert model.rn.conv1.weight.grad is not None


def test_s39_config_unchanged():
    torch.manual_seed(0)
    model = MNISTModel(load_config(str(S39_CFG)))
    assert not model.sparse_trunk
    assert sum(p.numel() for p in model.parameters()) == 2802645


def test_sparse_rejects_incompatible():
    cfg = _cfg("sparse_resnet18_l3")
    cfg["MODEL"]["PREDICT_DELTA"] = False                  # (PREV_RENDER false is legal: ablation A1)
    with pytest.raises(ValueError):
        MNISTModel(cfg)


def test_transl_head_is_decoupled():
    """MODEL.TRANSL_HEAD: the translation comes from the head; the trunk gets no gradient from it."""
    cfg = _cfg("sparse_resnet18_l3")
    cfg["MODEL"]["TRANSL_HEAD"] = True
    torch.manual_seed(0)
    model = MNISTModel(cfg)
    model.train()
    lnes = _sparse_image(np.random.default_rng(3), c=2)[None].repeat(2, 1, 1, 1)
    prev = torch.zeros(2, 51)
    prev[:, 2] = 0.55
    betas = torch.zeros(2, 10)
    K = torch.tensor([[200.0, 0, 120.0], [0, 200.0, 90.0], [0, 0, 1.0]]).expand(2, 3, 3).contiguous()
    out = model(lnes, prev, betas=betas, camera_K=K)
    assert torch.equal(out[:, :3], prev[:, :3])                      # zero-initialised head: prev translation
    out[:, :3].square().sum().backward(retain_graph=True)
    assert all(p.grad is None or float(p.grad.abs().max()) == 0 for p in model.rn.parameters())
    assert model.transl_head[2].weight.grad is not None and float(model.transl_head[2].weight.grad.abs().max()) > 0


def test_sparse_noise_keeps_coarse_active_sets():
    """MODEL.SPARSE_NOISE adds events only inside active level-2 cells (levels 2..4 unchanged) and only in training."""
    from sparse_trunk import cell_mask
    cfg = _cfg("sparse_resnet18_l3")
    cfg["MODEL"]["SPARSE_NOISE"] = 0.3
    torch.manual_seed(0)
    model = MNISTModel(cfg)
    x = _sparse_image(np.random.default_rng(4)).permute(2, 0, 1)[None]
    model.train()
    y = model._sparse_noise(x)
    m, my = input_mask(x), input_mask(y)
    assert int(my.sum()) > int(m.sum())                                       # noise was added
    c, cy = m, my
    for lvl in range(1, 5):
        c, cy = cell_mask(c), cell_mask(cy)
        if lvl >= 2:
            assert torch.equal(c, cy), lvl
    assert torch.equal(y[:, 2:], x[:, 2:])                                     # render channels untouched
    model.eval()
    lnes = _sparse_image(np.random.default_rng(5), c=2)[None]
    prev = torch.zeros(1, 51)
    prev[:, 2] = 0.55
    K = torch.tensor([[[200.0, 0, 120.0], [0, 200.0, 90.0], [0, 0, 1.0]]])
    with torch.no_grad():
        assert torch.equal(model(lnes, prev, betas=torch.zeros(1, 10), camera_K=K),
                           model(lnes, prev, betas=torch.zeros(1, 10), camera_K=K))   # eval: no noise, deterministic


def test_static_trunk_equals_masked_dense_and_executor():
    """sparse_graph.StaticSparseTrunk (static shapes, bucketed) is the same function as the masked dense trunk."""
    from sparse_graph import StaticSparseTrunk
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    trunk = SparseResNet(51)
    adapter = torch.nn.Conv2d(4, 3, 3, padding=1)
    _randomise_bn(trunk, rng)
    trunk.eval()
    # the synthetic image sprinkles sites over the whole frame (every coarse cell is active): generous capacities
    st = StaticSparseTrunk(adapter, trunk, buckets=((64, 32, 16, 8, 4), (4096, 2048, 1024, 400, 180), (8192, 2048, 1024, 400, 180)))
    for k in range(3):
        img = _sparse_image(np.random.default_rng(10 + k))
        x = img.permute(2, 0, 1)[None]
        m = input_mask(x)
        with torch.no_grad():
            ref = trunk(adapter(x) * m, m)[0]
        cnt = st.counts(img).tolist()
        b = st.bucket_for(cnt)
        assert b is not None
        for bb in range(b, len(st.buckets)):                       # every bucket that holds the sets gives the same output
            out = st.run(img, bb)
            assert torch.allclose(out, ref, atol=1e-4, rtol=0), (bb, (out - ref).abs().max())
    assert st.bucket_for([3000, 10, 10, 10, 10]) == 1 and st.bucket_for([9000, 10, 10, 10, 10]) is None


def test_sparse_trunk_without_render():
    """S43 ablation A1: PREV_RENDER false -> the trunk reads the 2 LNES planes only; prev reaches the output through prev_mlp."""
    cfg = _cfg("sparse_resnet18_l3")
    cfg["MODEL"]["PREV_RENDER"] = False
    cfg["MODEL"]["ZERO_EVENT_GATE"] = True
    torch.manual_seed(0)
    model = MNISTModel(cfg)
    assert model.sparse_trunk and model.conv1.in_channels == 2 and hasattr(model, "mano")
    model.train()
    lnes = _sparse_image(np.random.default_rng(6), c=2)[None].repeat(2, 1, 1, 1)
    prev = torch.zeros(2, 51)
    prev[:, 2] = 0.55
    out = model(lnes, prev, betas=torch.zeros(2, 10))
    loss, _ = model._compute_loss(out, prev + 0.01, torch.zeros(2, 10))
    loss.backward()
    assert torch.isfinite(loss) and model.rn.conv1.weight.grad is not None
