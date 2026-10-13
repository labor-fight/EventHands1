"""S39 tracker: network + adaptive filter + the deployed step contract.

One step ending at ms `end` (session clock), tracking started at ms `a` (`reset`):
  window  w = min(200, end - a + 1)             evidence window, clipped to the tracking start
  LNES    events of [end - w + 1, end]
  count   n = events of the last 50 ms [max(a, end - 49), end]   (x count_scale, see README)
  z       = net(LNES, prev, betas, K)           (several weight files: the seed-ensemble rule)
  state   = filter(prev, z, n)                  the filtered state is the output AND the next step's prev
The default step is 50 ms (20 Hz), the contract every reported S39 number was measured with.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .events import EventBuffer, build_lnes
from .filt import AdaptiveFilter, combine, per_step_spec
from .net import S39Net, enable_cuda_graph

ROOT = Path(__file__).resolve().parents[1]


class S39Tracker:
    def __init__(self, weights: List[str], device: str = "cuda", step_ms: int = 50, count_scale: float = 1.0,
                 config: Optional[str] = None, cuda_graph: bool = False):
        cfg = json.loads(Path(config or ROOT / "configs" / "s39.json").read_text())
        self.cfg = cfg
        self.device = torch.device(device)
        self.nets = [S39Net.load(w, self.device) for w in weights]
        if cuda_graph:
            if self.device.type != "cuda":
                raise ValueError("--cuda-graph needs a CUDA device")
            for n in self.nets:
                enable_cuda_graph(n)
        self.step_ms = int(step_ms)
        self.window_ms = int(cfg["window_ms"])
        self.count_ms = int(cfg["count_window_ms"])
        self.count_scale = float(count_scale)
        self.filter = AdaptiveFilter(per_step_spec(cfg["filter"], self.step_ms, self.count_ms)).to(self.device)
        self.mano = self.nets[0].mano
        self.state = None
        self.a = None
        self.betas = torch.zeros(1, 10, device=self.device)
        self.K = None

    # ------------------------------------------------------------------ set-up
    def set_camera(self, K_model: np.ndarray):
        """The network's intrinsics (`SensorMap.K_model()`: 640 x 480 convention)."""
        self.K = torch.as_tensor(np.asarray(K_model, np.float32).reshape(1, 3, 3), device=self.device)

    def reset(self, state51, start_ms: int, betas=None):
        """Start (or restart) tracking from `state51` (51,) at session ms `start_ms`; the first step ends at
        start_ms + step - 1 and its evidence window begins at start_ms."""
        self.state = torch.as_tensor(np.asarray(state51, np.float32).reshape(1, 51), device=self.device)
        if betas is not None:
            self.betas = torch.as_tensor(np.asarray(betas, np.float32).reshape(1, 10), device=self.device)
        self.a = int(start_ms)

    def first_end(self) -> int:
        return self.a + self.step_ms - 1

    @torch.no_grad()
    def warmup(self, n: int = 5):
        """A few dummy steps (CUDA context, cuDNN algorithm selection, CUDA graph capture) before tracking starts."""
        g = torch.Generator().manual_seed(0)
        lnes = (torch.rand(1, 180, 240, 2, generator=g) * (torch.rand(1, 180, 240, 2, generator=g) > 0.97)).to(self.device)
        state = torch.as_tensor(self.template_state(), device=self.device).view(1, 51)
        for _ in range(n):
            z = combine([net(lnes, state, self.betas, self.K) for net in self.nets])
            self.filter(lnes, state, z, 1000.0)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    # ------------------------------------------------------------------ one step
    @torch.no_grad()
    def step(self, buf: EventBuffer, end_ms: int) -> dict:
        assert self.state is not None and self.K is not None, "call set_camera() and reset() first"
        t0 = time.perf_counter()
        w = min(self.window_ms, end_ms - self.a + 1)
        xyp, ms_rel = buf.window(end_ms, w)
        lnes_np = build_lnes(xyp, ms_rel, w)
        lo = max(self.a, end_ms - self.count_ms + 1)
        n_raw = buf.count(lo, end_ms)
        if self.step_ms == self.count_ms and self.count_scale == 1.0:
            n = n_raw                                         # the official contract, exactly
        else:
            n = float(n_raw) * self.count_ms / (end_ms - lo + 1) * self.count_scale
        lnes = torch.from_numpy(lnes_np).unsqueeze(0).to(self.device)
        prev = self.state
        z = combine([net(lnes, prev, self.betas, self.K) for net in self.nets])
        self.state = self.filter(lnes, prev, z, n)
        out = self.state.cpu().numpy()[0]
        return {"end_ms": end_ms, "state": out, "n50": n_raw, "n_used": n, "window_ms": w, "lnes": lnes_np,
                "compute_ms": (time.perf_counter() - t0) * 1e3}

    # ------------------------------------------------------------------ mesh
    @torch.no_grad()
    def mesh(self, state51=None):
        """(verts (778, 3), joints (21, 3)) in the camera frame (metres) of `state51` (default: the current state)."""
        s = self.state if state51 is None else torch.as_tensor(np.asarray(state51, np.float32).reshape(1, 51),
                                                               device=self.device)
        v, j = self.mano.from_state(s, self.betas)
        return v[0].cpu().numpy(), j[0].cpu().numpy()

    def project(self, pts3d: np.ndarray) -> np.ndarray:
        """Camera-frame points -> LNES-image pixels (240 x 180)."""
        K = self.K[0].cpu().numpy() * 0.375
        z = np.clip(pts3d[:, 2], 1e-6, None)
        return np.stack([K[0, 0] * pts3d[:, 0] / z + K[0, 2], K[1, 1] * pts3d[:, 1] / z + K[1, 2]], -1)

    # ------------------------------------------------------------------ initial states
    def template_state(self, target_px: float = 72.0, center=(120.0, 90.0)) -> np.ndarray:
        """The template pose (configs/template_init.json: right hand open, palm to the camera, fingers up) placed so its
        projection is centred at `center` (LNES pixels) and `target_px` wide/high (training distribution: 60-90 px)."""
        tpl = json.loads((ROOT / "configs" / "template_init.json").read_text())
        s = np.asarray(tpl["state51"], np.float32).copy()
        s[:3] = 0.0
        v, _ = self.mano(self.betas, torch.as_tensor(s[3:6], device=self.device).view(1, 3),
                         torch.as_tensor(s[6:], device=self.device).view(1, 45) + self.mano.hands_mean.view(1, -1),
                         torch.zeros(1, 3, device=self.device))
        v = v[0].cpu().numpy()
        c = 0.5 * (v.min(0) + v.max(0))
        extent = float(max(np.ptp(v[:, 0]), np.ptp(v[:, 1])))
        K = self.K[0].cpu().numpy() * 0.375
        Z = float(K[0, 0]) * extent / float(target_px)
        X = (center[0] - K[0, 2]) * Z / K[0, 0]
        Y = (center[1] - K[1, 2]) * Z / K[1, 1]
        s[:3] = np.array([X, Y, Z], np.float32) - c
        return s
