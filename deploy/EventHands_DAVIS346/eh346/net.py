"""The S39 network `dt_dz_l3` (2.80 M parameters), inference only.

Per 50 ms step:  x = [LNES (2) | render(prev): silhouette, inverse depth (2)]  (B, 180, 240, 4)
                 delta = ResNet18-to-layer3(conv3x3(4 -> 3)(x)) + prev_mlp(prev)     (51)
                 out = prev + delta   (delta = 0 when the LNES window holds no event)
Same modules, parameter names and operation order as the training repository (`model/model.py` dense forward,
`model/backbones.py` ScaledResNet layers (2, 2, 2, 0)), so a server checkpoint loads strictly and reproduces it.
"""
from __future__ import annotations

import torch
from torch import nn

from .mano import Mano

H, W = 180, 240
#: render-space scale: the model's K is in the 640 x 480 convention of the training camera, the image is 240 x 180
RENDER_SCALE = 240.0 / 640.0


class BasicBlock(nn.Module):
    def __init__(self, inp: int, out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.downsample = (nn.Sequential(nn.Conv2d(inp, out, 1, stride, bias=False), nn.BatchNorm2d(out))
                           if stride != 1 or inp != out else None)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNetL3(nn.Module):
    """torchvision ResNet-18 up to layer3 (64 / 128 / 256 channels), global average pool, fc 256 -> 51."""

    def __init__(self, out_dim: int = 51):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.fc(torch.flatten(self.avgpool(x), 1))


class S39Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 3, kernel_size=3, padding=1)
        self.rn = ResNetL3(51)
        self.prev_mlp = nn.Sequential(nn.Linear(51, 64), nn.ReLU(inplace=True), nn.Linear(64, 51))
        self.mano = Mano()

    @classmethod
    def load(cls, path, device="cpu") -> "S39Net":
        try:
            blob = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:                                   # torch < 1.13
            blob = torch.load(str(path), map_location="cpu")
        sd = blob["state_dict"] if "state_dict" in blob else blob
        net = cls()
        net.load_state_dict(sd, strict=True)
        net.meta = blob.get("meta", {})
        return net.to(device).eval()

    @torch.no_grad()
    def render(self, prev: torch.Tensor, betas: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Point-splat render of the previous state's 778 vertices: (B, 180, 240, 2) = [silhouette, inverse depth]."""
        verts, _ = self.mano.from_state(prev, betas)
        s = RENDER_SCALE
        fx, fy, cx, cy = K[:, 0, 0] * s, K[:, 1, 1] * s, K[:, 0, 2] * s, K[:, 1, 2] * s
        x, y, z = verts.unbind(-1)
        z_safe = z.clamp(min=1e-6)
        u = fx[:, None] * x / z_safe + cx[:, None]
        v = fy[:, None] * y / z_safe + cy[:, None]
        ui, vi = u.round().long(), v.round().long()
        B, n_v = ui.shape
        valid = (z > 1e-6) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        b_idx = torch.arange(B, device=verts.device)[:, None].expand(B, n_v)
        pix = (b_idx * (H * W) + vi * W + ui)[valid]
        z_valid = z_safe[valid]
        depth = torch.full((B * H * W,), 1.0e6, device=verts.device, dtype=torch.float32)
        sil = torch.zeros((B * H * W,), device=verts.device, dtype=torch.float32)
        if pix.numel() > 0:
            depth.scatter_reduce_(0, pix, z_valid, reduce="amin", include_self=True)
            sil.scatter_(0, pix, torch.ones_like(z_valid))
        sil = sil.view(B, H, W).clamp(0, 1)
        inv = ((1.0 / depth.view(B, H, W)).clamp(0.0, 5.0) / 5.0) * sil
        return torch.cat([sil.unsqueeze(-1), inv.unsqueeze(-1)], dim=-1)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.rn(self.conv1(x))

    @torch.no_grad()
    def forward(self, lnes: torch.Tensor, prev: torch.Tensor, betas: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """lnes (B, 180, 240, 2), prev (B, 51), betas (B, 10), K (B, 3, 3) -> the unfiltered new state (B, 51)."""
        B = lnes.shape[0]
        rend = self.render(prev.float(), betas, K).to(dtype=lnes.dtype, device=lnes.device)
        x = torch.cat([lnes, rend], dim=-1).permute(0, 3, 1, 2).contiguous()
        out = self.trunk(x)
        out = out + self.prev_mlp(prev.to(out.dtype))
        empty = lnes.reshape(B, -1).abs().sum(dim=1, keepdim=True) <= 0
        delta = torch.where(empty, torch.zeros_like(out), out)
        return delta + prev.to(out.dtype)


class GraphedTrunk(nn.Module):
    """Optional: `conv1 + rn` replayed from one CUDA graph per input shape (batch 1). Identical output, fewer kernel
    launches (about -1 ms per step on an L20)."""

    def __init__(self, net: S39Net):
        super().__init__()
        self.net = net
        self.graphs = {}

    def __call__(self, x):
        key = tuple(x.shape)
        if key not in self.graphs:
            sin = x.clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(3):
                    self.net.rn(self.net.conv1(sin))
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.no_grad():
                sout = self.net.rn(self.net.conv1(sin))
            self.graphs[key] = (g, sin, sout)
        g, sin, sout = self.graphs[key]
        sin.copy_(x)
        g.replay()
        return sout.clone()


def enable_cuda_graph(net: S39Net) -> S39Net:
    gt = GraphedTrunk(net)
    net.trunk = gt          # instance attribute shadows the method
    return net
