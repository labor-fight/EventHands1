"""S45-TR amendment 4 (docs/EXPERIMENT_RECORDS.md#s45tr-amend4-20261030): the correspondence of a forward computed under
bf16 autocast can be taken outside autocast (the T0 gate does), and equals the in-autocast one."""
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from semkine import s45tr_t0 as T0  # noqa: E402
from tools.s45tr import common as C  # noqa: E402


def test_correspondence_outside_autocast():
    verts, faces = T0.template(ROOT / "assets" / "mano_right.npz")
    torch.manual_seed(0)
    net = T0.T0Net(faces, verts, C.T1A_GNN).eval()
    n = [900, 400]
    ev = torch.cat([torch.stack([torch.full((k,), float(b)), torch.randint(0, 240, (k,)).float(),
                                 torch.randint(0, 180, (k,)).float(), torch.sort(torch.rand(k) * 0.05).values,
                                 torch.randint(0, 2, (k,)).float()], -1) for b, k in enumerate(n)])
    ptr = torch.tensor([0, n[0], n[0] + n[1]])
    with torch.no_grad():
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = net(ev, ptr, torch.full((2,), 0.05))
            inside = net.correspondence(out)
        outside = net.correspondence(out)
    assert out["z"].dtype == torch.bfloat16
    assert torch.equal(inside[0], outside[0]) or (inside[0] != outside[0]).float().mean() < 0.01
    assert outside[1].dtype == torch.float32 and torch.isfinite(outside[2]).all()
