#!/usr/bin/env python3
"""Interchangeable CNN trunks for the dense tracker.

The dense arm of `MNISTModel` is `self.rn(self.conv1(x))`: a `Conv2d(in_ch, 3, 3, padding=1)` adapter (111 parameters
for the published 4-channel input) in front of `torchvision.models.resnet18(num_classes=OUT)`; the OUT = 51 values are
a delta added to the previous state. This module is the registry of drop-in replacements for `self.rn`:

    rn = build_backbone(name, out_dim)          # (B, 3, H, W) -> (B, out_dim); BatchNorm kept; no timm

    resnet18              torchvision.models.resnet18(num_classes=out_dim) itself: same state_dict keys, same values
                          under the same torch seed, same RNG consumption (the default; every recorded run)
    resnet18_w0.75/0.5/0.25   ResNet-18 with the stem and all four stage widths scaled (BasicBlock, 7x7 stride-2
                          stem, max-pool, [2, 2, 2, 2] blocks and BatchNorm unchanged; widths rounded to 8)
    resnet18_l3           layer4 replaced by Identity, the fc reads the 256-d pooled layer3 map
    resnet18_l3w128       DT2 arm 1: as resnet18_l3 but layer3 is 128 wide instead of 256 (stem / layer1 / layer2 stay
                          64 / 64 / 128; `ScaledResNet(stage_widths=(64, 128, 128), layers=(2, 2, 2, 0))`), the fc reads
                          the 128-d pooled layer3 map: 1.30 M parameters, 0.98 G MACs against 2.80 M / 1.24 G
    resnet18_l3w128s48    DT2 deviation 4: resnet18_l3w128 with the stem / layer1 / layer2 at 48 / 48 / 96 channels
                          (`stage_widths=(48, 96, 128)`): fewer MACs on the 90x120 and 45x60 maps, same depth
    resnet18_l3w128b1     DT2 deviation 4: resnet18_l3w128 with one BasicBlock per stage (`layers=(1, 1, 1, 0)`):
                          half the trunk's convolutions (eager batch-1 latency is kernel-launch bound)
    resnet10              torchvision ResNet(BasicBlock, [1, 1, 1, 1]), full width
    regnet_x_400mf, mobilenet_v3_small, shufflenet_v2_x0_5
                          torchvision classifiers built with num_classes=out_dim and nothing else changed
    mobilenet_v3_small_nodrop   extra (not in the requested list): the same net with the classifier dropout set to
                          0, because a regression head should not run Dropout(0.2) in train mode

Nothing in the repository imports this module yet: the new behaviour is opt-in and "resnet18" builds exactly what the
code base builds today (same call, same RNG draws). The parameter / MAC survey, CPU latency and trainability probes
that produced outputs/dt/reports/backbones.{json,md} live next to the results (backbones_survey.py).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import torch
from torch import nn
import torchvision
from torchvision.models.resnet import BasicBlock, ResNet

__all__ = ["BACKBONES", "STAGE_WIDTHS", "ScaledResNet", "build_backbone", "count_params", "resnet_forward_taps"]

#: stage widths of torchvision's ResNet-18 / ResNet-10; the 7x7 stem has the width of layer1
STAGE_WIDTHS = (64, 128, 256, 512)


def _scaled(c: int, width: float, divisor: int = 8) -> int:
    """`c * width` rounded to the nearest multiple of `divisor` (at least `divisor`): 64 * 0.75 -> 48, 64 * 0.25 -> 16."""
    return max(divisor, int(round(c * width / divisor)) * divisor)


class ScaledResNet(ResNet):
    """torchvision's ResNet-18 family with every stage width scaled and, optionally, trailing stages dropped.

    Built from torchvision's own `BasicBlock` and `ResNet` (`_make_layer`, `forward`): only `__init__` is rewritten,
    because `ResNet.__init__` hard-codes 64 / 128 / 256 / 512. Modules are created and initialised in the same order
    as in `ResNet.__init__`, so `ScaledResNet(n, width=1.0)` equals `torchvision.models.resnet18(num_classes=n)`
    bit for bit under the same seed (tests/test_dt_backbone.py).

    `layers` is the number of BasicBlocks per stage; trailing zeros replace those stages by `nn.Identity()` and the
    fc reads the last kept stage (`layers=(2, 2, 2, 0)` is the "l3" trunk: fc input 256 at width 1.0).

    `stage_widths` (default None: every stage is `_scaled(STAGE_WIDTHS[i], width)`, i.e. nothing changes) sets the
    channel widths of the first `len(stage_widths)` stages explicitly, one integer per stage (the stem has the width of
    stage 1, as in torchvision); the stages it does not cover keep their `width`-scaled size. `(64, 128, 128)` with
    `layers=(2, 2, 2, 0)` is the "l3w128" trunk: layer3 half as wide, so the layer that holds 75 % of the "l3" trunk's
    parameters shrinks 3.5x. Modules are still created and initialised in the order of `ResNet.__init__`.
    """

    def __init__(self, num_classes: int = 1000, width: float = 1.0, layers: Sequence[int] = (2, 2, 2, 2),
                 stage_widths: Optional[Sequence[int]] = None) -> None:
        nn.Module.__init__(self)                     # not ResNet.__init__: its stage widths cannot be changed
        layers = tuple(int(n) for n in layers)
        kept = sum(1 for n in layers if n > 0)
        if len(layers) != 4 or min(layers) < 0 or kept == 0 or any(n == 0 for n in layers[:kept]):
            raise ValueError(f"layers must be four block counts, positive up to the last kept stage and 0 after "
                             f"it; got {layers}")
        widths = tuple(_scaled(c, width) for c in STAGE_WIDTHS)
        if stage_widths is not None:
            given = tuple(stage_widths)
            if not 1 <= len(given) <= 4 or any(isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in given):
                raise ValueError(f"stage_widths must be one to four positive integers (channels of the first stages); "
                                 f"got {stage_widths!r}")
            widths = given + widths[len(given):]
        self.stage_widths, self.stage_blocks = widths, layers
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = widths[0]
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, widths[0], layers[0])
        self.layer2 = self._make_layer(BasicBlock, widths[1], layers[1], stride=2) if layers[1] else nn.Identity()
        self.layer3 = self._make_layer(BasicBlock, widths[2], layers[2], stride=2) if layers[2] else nn.Identity()
        self.layer4 = self._make_layer(BasicBlock, widths[3], layers[3], stride=2) if layers[3] else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[kept - 1] * BasicBlock.expansion, num_classes)
        for m in self.modules():                     # the initialisation loop of ResNet.__init__
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_forward_taps(rn: nn.Module, x, taps: Sequence[str] = ("layer2", "layer3")):
    """`rn(x)` of a torchvision-style ResNet (`ScaledResNet`, `resnet18`, `resnet10`) that also returns the feature maps
    after the stages named in `taps`: `(logits, {name: map})`.

    The operations are those of `torchvision.models.ResNet._forward_impl` in the same order (conv1, bn1, relu, maxpool,
    layer1 .. layer4, avgpool, flatten, fc), so `logits` equals `rn(x)` bit for bit (tests/test_dt_model3.py); a stage
    that is `nn.Identity()` (the "l3" trunk's layer4) is a no-op exactly as in `rn(x)`. Used by the DT2 root readout heads
    (`MODEL.ROOT_HEAD`), which read the layer2 / layer3 maps next to the global-average-pool readout.
    """
    bad = [t for t in taps if t not in ("layer1", "layer2", "layer3", "layer4")]
    if bad:
        raise ValueError(f"taps must be names of ResNet stages, got {bad}")
    x = rn.conv1(x)
    x = rn.bn1(x)
    x = rn.relu(x)
    x = rn.maxpool(x)
    got = {}
    for name in ("layer1", "layer2", "layer3", "layer4"):
        x = getattr(rn, name)(x)
        if name in taps:
            got[name] = x
    x = rn.avgpool(x)
    x = torch.flatten(x, 1)
    return rn.fc(x), got


def _resnet18(out_dim: int) -> nn.Module:
    # The line every recorded run depends on: exactly what model.py builds today (same call, same RNG draws).
    return torchvision.models.resnet18(num_classes=out_dim)


#: name -> builder(out_dim); insertion order is the order of the survey tables
BACKBONES: Dict[str, Callable[[int], nn.Module]] = {
    "resnet18": _resnet18,
    "resnet18_w0.75": lambda out_dim: ScaledResNet(out_dim, width=0.75),
    "resnet18_w0.5": lambda out_dim: ScaledResNet(out_dim, width=0.5),
    "resnet18_w0.25": lambda out_dim: ScaledResNet(out_dim, width=0.25),
    "resnet18_l3": lambda out_dim: ScaledResNet(out_dim, layers=(2, 2, 2, 0)),
    "resnet10": lambda out_dim: ResNet(BasicBlock, [1, 1, 1, 1], num_classes=out_dim),
    "regnet_x_400mf": lambda out_dim: torchvision.models.regnet_x_400mf(num_classes=out_dim),
    "mobilenet_v3_small": lambda out_dim: torchvision.models.mobilenet_v3_small(num_classes=out_dim),
    "shufflenet_v2_x0_5": lambda out_dim: torchvision.models.shufflenet_v2_x0_5(num_classes=out_dim),
    "mobilenet_v3_small_nodrop": lambda out_dim: torchvision.models.mobilenet_v3_small(num_classes=out_dim,
                                                                                       dropout=0.0),
    "resnet18_l3w128": lambda out_dim: ScaledResNet(out_dim, layers=(2, 2, 2, 0), stage_widths=(64, 128, 128)),
    "resnet18_l3w128s48": lambda out_dim: ScaledResNet(out_dim, layers=(2, 2, 2, 0), stage_widths=(48, 96, 128)),
    "resnet18_l3w128b1": lambda out_dim: ScaledResNet(out_dim, layers=(1, 1, 1, 0), stage_widths=(64, 128, 128)),
}


def build_backbone(name: str, out_dim: int) -> nn.Module:
    """Trunk `name` mapping (B, 3, H, W) to (B, out_dim); `name` is a key of BACKBONES, else ValueError."""
    if not isinstance(name, str) or name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; available: {', '.join(BACKBONES)}")
    if int(out_dim) < 1:
        raise ValueError(f"out_dim must be a positive integer, got {out_dim!r}")
    return BACKBONES[name](int(out_dim))


def count_params(model: nn.Module) -> int:
    """Number of parameters (trainable or not; BatchNorm running statistics are buffers and not counted)."""
    return sum(p.numel() for p in model.parameters())
