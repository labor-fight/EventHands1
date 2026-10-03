"""Contracts of model/backbones.py (task `backbones`, lighter CNN trunks for the dense tracker): the default trunk is
torchvision's resnet18 bit for bit (state_dict keys, shapes, seeded values and RNG consumption); every registered
trunk maps (B, 3, 180, 240) to (B, 51) with a finite gradient on every parameter; parameter and MAC counts stay inside
the ranges measured in outputs/dt/reports/backbones.md; the width / depth variants keep the ResNet-18 structure; a
state_dict round trip, batch-1 BatchNorm behaviour and a one-batch overfit work for every trunk; unknown names raise.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python -m pytest tests/test_dt_backbone.py -q
"""
from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "model")]
import backbones as BB                                                       # noqa: E402
from backbones import BACKBONES, ScaledResNet, build_backbone, count_params  # noqa: E402

REQUIRED = ("resnet18", "resnet18_w0.75", "resnet18_w0.5", "resnet18_w0.25", "resnet18_l3", "resnet10",
            "regnet_x_400mf", "mobilenet_v3_small", "shufflenet_v2_x0_5")
OUT_DIM = 51
HW = (180, 240)

#: measured with torch 2.1.0 / torchvision 0.16.0 / thop 0.1.1 at out_dim 51 and input (1, 3, 180, 240)
#: (outputs/dt/reports/backbones.json): trunk parameters, Conv2d + Linear multiply-accumulates, thop MACs
MEASURED = {
    "resnet18": (11_202_675, 1_641_662_976, 1_650_397_184),
    "resnet18_w0.75": (6_309_987, 942_491_520, 949_042_176),
    "resnet18_w0.5": (2_811_987, 435_823_872, 440_190_976),
    "resnet18_w0.25": (708_675, 121_660_032, 123_843_584),
    "resnet18_l3": (2_795_891, 1_238_996_736, 1_247_260_672),
    "resnet10": (4_931_955, 800_279_040, 806_359_040),
    "regnet_x_400mf": (5_115_427, 388_774_320, 400_260_160),
    "mobilenet_v3_small": (1_570_131, 51_233_824, 56_591_096),
    "shufflenet_v2_x0_5": (394_067, 36_298_032, 39_956_976),
    "mobilenet_v3_small_nodrop": (1_570_131, 51_233_824, 56_591_096),
}
TOL = 0.01                                              # relative band around every measured count except resnet18


def _assert_same_state(a: nn.Module, b: nn.Module):
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    for k in sa:
        assert sa[k].shape == sb[k].shape and sa[k].dtype == sb[k].dtype, k
        assert torch.equal(sa[k], sb[k]), k


def _conv_fc_macs(model: nn.Module, chw=(3, *HW)) -> int:
    """Multiply-accumulates of every Conv2d and Linear at batch 1 (bias excluded), counted with forward hooks."""
    total = []

    def count(mod, _inp, out):
        if isinstance(mod, nn.Conv2d):
            total.append(out.numel() * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1])
        else:
            total.append(out.numel() * mod.in_features)

    hooks = [m.register_forward_hook(count) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model.eval()(torch.zeros(1, *chw))
    for h in hooks:
        h.remove()
    return sum(total)


# ------------------------------------------------------------------ registry and the default trunk
def test_registry_has_the_required_names_in_order():
    assert tuple(BACKBONES)[:len(REQUIRED)] == REQUIRED
    assert set(MEASURED) == set(BACKBONES)                # every registered name has a measured range below


@pytest.mark.parametrize("out_dim", [OUT_DIM, 12])
def test_default_name_is_torchvision_resnet18_bit_for_bit(out_dim):
    torch.manual_seed(1234)
    ref = torchvision.models.resnet18(num_classes=out_dim)
    ref_rng = torch.get_rng_state()
    torch.manual_seed(1234)
    got = build_backbone("resnet18", out_dim)
    assert type(got) is type(ref)
    assert [n for n, _ in got.named_parameters()] == [n for n, _ in ref.named_parameters()]
    _assert_same_state(ref, got)
    assert torch.equal(torch.get_rng_state(), ref_rng)    # the same number of RNG draws: later layers see the same stream
    x = torch.randn(2, 3, 64, 80)
    assert torch.equal(ref.eval()(x), got.eval()(x))


def test_scaled_resnet_at_full_width_is_resnet18_bit_for_bit():
    """The class behind the width variants reproduces torchvision's own construction and initialisation order."""
    torch.manual_seed(7)
    ref = torchvision.models.resnet18(num_classes=OUT_DIM)
    torch.manual_seed(7)
    got = ScaledResNet(OUT_DIM, width=1.0)
    _assert_same_state(ref, got)
    torch.manual_seed(7)
    ref10 = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [1, 1, 1, 1], num_classes=OUT_DIM)
    torch.manual_seed(7)
    _assert_same_state(ref10, ScaledResNet(OUT_DIM, layers=(1, 1, 1, 1)))
    torch.manual_seed(7)
    _assert_same_state(ref10, build_backbone("resnet10", OUT_DIM))


@pytest.mark.parametrize("bad", ["resnet50", "", "ResNet18", "resnet18 ", None, 18])
def test_unknown_name_raises_value_error_listing_the_names(bad):
    with pytest.raises(ValueError) as e:
        build_backbone(bad, OUT_DIM)
    for name in REQUIRED:
        assert name in str(e.value)


def test_out_dim_must_be_positive():
    with pytest.raises(ValueError):
        build_backbone("resnet18", 0)


def test_importing_the_module_does_not_touch_the_rng_and_needs_no_timm():
    torch.manual_seed(5)
    a = torch.rand(3)
    torch.manual_seed(5)
    spec = importlib.util.spec_from_file_location("backbones_fresh_copy", REPO / "model" / "backbones.py")
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)                       # a second, independent import of the file
    assert torch.equal(a, torch.rand(3))
    assert "timm" not in sys.modules and list(fresh.BACKBONES) == list(BACKBONES)


# ------------------------------------------------------------------ structure of the new trunks
@pytest.mark.parametrize("name,w", [("resnet18_w0.75", 0.75), ("resnet18_w0.5", 0.5), ("resnet18_w0.25", 0.25)])
def test_width_variants_keep_the_resnet18_structure(name, w):
    m, ref = build_backbone(name, OUT_DIM), torchvision.models.resnet18(num_classes=OUT_DIM)
    assert m.conv1.kernel_size == (7, 7) and m.conv1.stride == (2, 2) and m.conv1.padding == (3, 3)
    assert isinstance(m.maxpool, nn.MaxPool2d) and m.maxpool.kernel_size == 3 and m.maxpool.stride == 2
    assert isinstance(m, torchvision.models.resnet.ResNet)                  # a ResNet subclass, forward unchanged
    assert [n for n, _ in m.named_modules()] == [n for n, _ in ref.named_modules()]      # same layer structure
    for (n, a), (_, b) in zip(m.named_modules(), ref.named_modules()):
        assert n == "" or type(a) is type(b), n                              # every submodule has the same type
        if isinstance(a, nn.Conv2d):
            assert a.kernel_size == b.kernel_size and a.stride == b.stride and a.padding == b.padding, n
            assert (a.bias is None) == (b.bias is None), n
            assert a.out_channels == int(b.out_channels * w), n
            assert a.in_channels == (3 if n == "conv1" else int(b.in_channels * w)), n
    assert m.fc.in_features == int(512 * w) and m.fc.out_features == OUT_DIM
    assert not any(isinstance(x, (nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)) for x in m.modules())
    assert sum(isinstance(x, nn.BatchNorm2d) for x in m.modules()) == 20


def test_l3_drops_layer4_and_reads_256_features():
    m, ref = build_backbone("resnet18_l3", OUT_DIM), torchvision.models.resnet18(num_classes=OUT_DIM)
    assert isinstance(m.layer4, nn.Identity) and m.fc.in_features == 256
    assert [k for k in m.state_dict() if not k.startswith("fc.")] == \
        [k for k in ref.state_dict() if not k.startswith(("layer4.", "fc."))]
    assert m.state_dict()["fc.weight"].shape == (OUT_DIM, 256)
    seen = {}
    h = m.avgpool.register_forward_hook(lambda _m, i, o: seen.update(shape=tuple(i[0].shape)))
    m.eval()(torch.zeros(1, 3, *HW))
    h.remove()
    assert seen["shape"] == (1, 256, 12, 15)


def test_resnet10_has_one_block_per_stage():
    m = build_backbone("resnet10", OUT_DIM)
    assert [len(getattr(m, f"layer{i}")) for i in (1, 2, 3, 4)] == [1, 1, 1, 1]
    assert m.fc.in_features == 512


def test_scaled_resnet_rejects_bad_layer_specs():
    for layers in [(2, 2, 2), (0, 2, 2, 2), (2, 0, 2, 2), (0, 0, 0, 0), (2, 2, 2, -1)]:
        with pytest.raises(ValueError):
            ScaledResNet(OUT_DIM, layers=layers)


def test_mobilenet_nodrop_differs_only_in_the_dropout_rate():
    a, b = build_backbone("mobilenet_v3_small", OUT_DIM), build_backbone("mobilenet_v3_small_nodrop", OUT_DIM)
    assert [x.p for x in a.modules() if isinstance(x, nn.Dropout)] == [0.2]
    assert [x.p for x in b.modules() if isinstance(x, nn.Dropout)] == [0.0]
    assert list(a.state_dict()) == list(b.state_dict())


# ------------------------------------------------------------------ every trunk: shape, gradients, counts
@pytest.mark.parametrize("name", list(BACKBONES))
def test_every_backbone_builds_maps_to_out_dim_and_has_finite_grads(name):
    torch.manual_seed(0)
    m = build_backbone(name, OUT_DIM).train()
    x = torch.randn(2, 3, *HW)
    out = m(x)
    assert out.shape == (2, OUT_DIM) and torch.isfinite(out).all()
    F.mse_loss(out, torch.zeros_like(out)).backward()
    params = list(m.parameters())
    assert params and all(p.grad is not None for p in params)
    assert all(torch.isfinite(p.grad).all() for p in params)
    assert any(p.grad.abs().max() > 0 for p in params)


@pytest.mark.parametrize("name", list(BACKBONES))
def test_parameter_counts_are_in_the_measured_ranges(name):
    n = count_params(build_backbone(name, OUT_DIM))
    want = MEASURED[name][0]
    if name == "resnet18":
        assert n == want == 11_202_675                   # 11,176,512 trunk + 26,163 for fc(512 -> 51)
    else:
        assert abs(n - want) <= TOL * want, (n, want)


def test_parameter_counts_order_and_ratios():
    p = {n: count_params(build_backbone(n, OUT_DIM)) for n in REQUIRED}
    assert p["resnet18_w0.25"] < p["resnet18_w0.5"] < p["resnet18_w0.75"] < p["resnet18"]
    assert p["shufflenet_v2_x0_5"] < p["resnet18_w0.25"] < p["mobilenet_v3_small"] < p["resnet18_w0.5"]
    assert 0.24 < p["resnet18_l3"] / p["resnet18"] < 0.26          # layer4 holds 74.9 % of the parameters
    assert 0.24 < p["resnet18_w0.5"] / p["resnet18"] < 0.26
    assert 0.43 < p["resnet10"] / p["resnet18"] < 0.45


@pytest.mark.parametrize("name", list(BACKBONES))
def test_conv_and_fc_macs_are_in_the_measured_ranges(name):
    macs = _conv_fc_macs(build_backbone(name, OUT_DIM))
    want = MEASURED[name][1]
    if name == "resnet18":
        assert macs == want == 1_641_662_976             # derived by hand from the 180 x 240 feature-map sizes
    else:
        assert abs(macs - want) <= TOL * want, (macs, want)


@pytest.mark.parametrize("name", list(BACKBONES))
def test_thop_macs_are_in_the_measured_ranges(name):
    thop = pytest.importorskip("thop")
    import copy
    macs, _ = thop.profile(copy.deepcopy(build_backbone(name, OUT_DIM)), inputs=(torch.zeros(1, 3, *HW),),
                           verbose=False)
    want = MEASURED[name][2]
    assert abs(macs - want) <= TOL * want, (macs, want)


# ------------------------------------------------------------------ training behaviour
@pytest.mark.parametrize("name", list(BACKBONES))
def test_state_dict_round_trip_is_exact(name):
    torch.manual_seed(0)
    a = build_backbone(name, OUT_DIM).train()
    x = torch.randn(2, 3, 96, 128)
    for _ in range(2):                                    # move the BatchNorm running statistics off their init
        a(x)
    buf = io.BytesIO()
    torch.save(a.state_dict(), buf)
    buf.seek(0)
    torch.manual_seed(1)
    b = build_backbone(name, OUT_DIM)
    b.load_state_dict(torch.load(buf), strict=True)
    _assert_same_state(a, b)
    assert torch.equal(a.eval()(x), b.eval()(x))


@pytest.mark.parametrize("name", list(BACKBONES))
def test_eval_forward_does_not_depend_on_the_batch_size(name):
    """BatchNorm at batch 1 in eval uses running statistics only: sample 0 alone equals sample 0 of a batch of 2."""
    torch.manual_seed(0)
    m = build_backbone(name, OUT_DIM).train()
    x = torch.randn(2, 3, 96, 128)
    m(x)
    m.eval()
    with torch.no_grad():
        full, one = m(x), m(x[:1])
    assert torch.allclose(full[:1], one, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("name", list(BACKBONES))
def test_train_mode_batch_one_is_valid_at_the_real_input_size(name):
    """BatchNorm in train mode needs more than one value per channel: every final map at 180 x 240 is >= 6 x 8."""
    torch.manual_seed(0)
    m = build_backbone(name, OUT_DIM).train()
    with torch.no_grad():
        out = m(torch.randn(1, 3, *HW))
    assert out.shape == (1, OUT_DIM) and torch.isfinite(out).all()


@pytest.mark.parametrize("name", list(BACKBONES))
def test_one_batch_overfit_lowers_the_loss(name):
    torch.manual_seed(0)
    m = build_backbone(name, OUT_DIM).train()
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(2, 3, 96, 128, generator=g), 0.1 * torch.randn(2, OUT_DIM, generator=g)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    losses = []
    for _ in range(12):
        opt.zero_grad()
        loss = F.mse_loss(m(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-3:]) / 3 < 0.8 * losses[0], losses
