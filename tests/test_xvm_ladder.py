"""XVM ladder helpers: endpoint grid parity with XCL, FD Jacobian parity with S44, GN recovers a known increment."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model"), str(ROOT / "tools")]

from xvm import common as C  # noqa: E402
from xvm.common import EF  # noqa: E402


def test_endpoint_grid_matches_xcl():
    from tools.xcl.evaluate import endpoint_grid
    runs = np.array([[10, 260], [400, 455], [600, 1000]])
    assert C.endpoint_grid(runs) == endpoint_grid({"valid_runs_ms": runs})


def _state(dev):
    g = torch.Generator().manual_seed(0)
    x = torch.zeros(1, 51, dtype=torch.float64)
    x[0, :3] = torch.tensor([0.0, 0.0, 0.7], dtype=torch.float64)
    x[0, 3:] = 0.1 * torch.randn(48, generator=g, dtype=torch.float64)
    return x.to(dev)


def test_fd_jacobian_matches_s44():
    dev = torch.device("cpu")
    mano = EF.Mano64(dev)
    x = _state(dev)
    betas = torch.zeros(1, 10, dtype=torch.float64)
    intr = (226.0, 226.0, 120.0, 90.0)
    _, _, uv, z, J2, _ = C.fd_jacobian(mano, x, betas, intr)
    _, uv_s, z_s, J_s = mano.jacobian(x, betas, intr)
    assert torch.allclose(J2, J_s, atol=1e-6) and torch.allclose(uv, uv_s)


def test_gn_recovers_known_increment_from_3d_and_2d():
    dev = torch.device("cpu")
    mano = EF.Mano64(dev)
    x0 = _state(dev)
    betas = torch.zeros(1, 10, dtype=torch.float64)
    intr = (226.0, 226.0, 120.0, 90.0)
    g = torch.Generator().manual_seed(1)
    d = torch.zeros(1, 51, dtype=torch.float64)
    d[0, 6:] = 0.05 * torch.randn(45, generator=g, dtype=torch.float64)
    xt = x0 + d
    vt, _ = mano.fk(xt, betas)
    P = torch.eye(51, dtype=torch.float64) * 1e-6
    for kind in ("3d", "2d"):
        x = x0.clone()
        v0, dX0, uv0, _, _, _ = C.fd_jacobian(mano, x0, betas, intr)
        ut, _ = EF.Mano64.project(vt, intr)
        for _ in range(4):
            v, dX, uv, _, J2, _ = C.fd_jacobian(mano, x, betas, intr)
            n = 778
            if kind == "3d":
                A, dy, m = dX[0], v[0] - v0[0], vt[0] - v0[0]
                Q = torch.eye(3, dtype=torch.float64).expand(n, 3, 3)
                W = torch.eye(3, dtype=torch.float64).expand(n, 3, 3) * 1e6
            else:
                A, dy, m = J2[0], uv[0] - uv0[0], ut[0] - uv0[0]
                Q = torch.eye(2, dtype=torch.float64).expand(n, 2, 2)
                W = torch.eye(2, dtype=torch.float64).expand(n, 2, 2) * 4.0
            x[0] += C.gn_step(A, dy, Q, m, W, P, 1.0, x[0] - x0[0])
        vx, _ = mano.fk(x, betas)
        assert (vx - vt).norm(dim=-1).max() < 1e-4, kind
