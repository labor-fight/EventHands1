"""CPU tests: semkine/s45sm_hier.py (part-first decoding of the patch routers)."""
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import s45sm as SM  # noqa: E402
from semkine import s45sm_dw as DW  # noqa: E402
from semkine import s45sm_fast as FAST  # noqa: E402
from semkine import s45sm_hier as H  # noqa: E402
from semkine import s45sm_srp as SRP  # noqa: E402
from semkine import xyz_mesh as XM  # noqa: E402
from semkine.events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y  # noqa: E402

NPZ = ROOT / "assets/mano_right.npz"


def packets(seed=0, n_per=(900, 0, 400)):
    g = torch.Generator().manual_seed(seed)
    rows, ptr = [], [0]
    for b, n in enumerate(n_per):
        ev = torch.zeros(n, 5)
        ev[:, EV_BATCH] = b
        ev[:, EV_X] = torch.randint(80, 140, (n,), generator=g).float()
        ev[:, EV_Y] = torch.randint(60, 110, (n,), generator=g).float()
        ev[:, EV_T] = torch.sort(torch.rand(n, generator=g) * 0.05).values
        ev[:, EV_P] = (torch.rand(n, generator=g) > 0.5).float()
        rows.append(ev)
        ptr.append(ptr[-1] + n)
    return torch.cat(rows), torch.tensor(ptr), torch.full((len(n_per),), 0.05)


def test_part_first_head():
    torch.manual_seed(0)
    net = SRP.build(NPZ).eval()
    part = H.patch_part(net)
    assert part.shape == (net.n_patches,) and int(part.max()) + 1 == int(net.face_part.max()) + 1
    linear = net.patch
    head = H.PartFirst(linear, part, int(net.face_part.max()) + 1)
    z = torch.randn(3, 50, linear.in_features) * 4
    logits, masked = linear(z), head(z)
    prob = torch.softmax(logits, -1)
    part_prob = torch.zeros(3, 50, int(part.max()) + 1).index_add_(-1, part, prob)
    best_part = part_prob.argmax(-1)
    chosen = masked.argmax(-1)
    assert torch.equal(part[chosen], best_part)                                  # the patch lies in the MAP part
    inside = logits.masked_fill(part.view(1, 1, -1) != best_part.unsqueeze(-1), float("-inf"))
    assert torch.equal(chosen, inside.argmax(-1))                                # and is the best patch there
    same = part[logits.argmax(-1)] == best_part
    assert torch.equal(chosen[same], logits.argmax(-1)[same])                    # unchanged when argmax agrees
    assert sum(p.numel() for p in head.parameters()) == sum(p.numel() for p in linear.parameters())


@torch.no_grad()
def test_registration_routes_part_first(tmp_path):
    torch.manual_seed(1)
    ckpt = tmp_path / "srp.ckpt"
    SRP.save_checkpoint(SRP.build(NPZ), ckpt)
    native = (SM.ROUTERS, SM._load_router, SM.route_nodes, FAST.route)
    events, ptr, dt = packets(2)
    src, mask = XM.stride_sample(ptr, SM.MAX_NODES)
    sel = mask & (torch.rand(mask.shape, generator=torch.Generator().manual_seed(3)) > 0.4)
    try:
        SRP.register()
        DW.register()
        H.register()
        assert set(H.KINDS) <= set(SM.ROUTERS)
        base = SM._load_router("srp", ckpt)
        hier = SM._load_router("srph", ckpt)
        assert isinstance(hier.patch, H.PartFirst) and SM.n_params(hier) == SM.n_params(base) == 137787
        a = SM.route_nodes(hier, "srph", events, ptr, dt, 0.5)
        b = native[2](hier, "t0p", events, ptr, dt, 0.5)
        assert all(torch.equal(x, y) for x, y in zip(a[1:], b[1:]))
        out_h = FAST.router_forward(hier, events, ptr, dt, src, mask)
        out_b = FAST.router_forward(base, events, ptr, dt, src, mask)
        assert torch.equal(out_h["coord"], out_b["coord"]) and torch.equal(out_h["onhand"], out_b["onhand"])
        face_h, _ = FAST.route(hier, "srph", out_h, sel)
        part = H.patch_part(base)
        prob = torch.softmax(out_b["patch"].float(), -1)
        best = torch.zeros(*prob.shape[:-1], int(part.max()) + 1).index_add_(-1, part, prob).argmax(-1)
        assert torch.equal(base.face_part[face_h][sel], best[sel])               # routed faces lie in the MAP part
    finally:
        H.unregister()
        DW.unregister()
        SRP.unregister()
    assert (SM.ROUTERS, SM._load_router, SM.route_nodes, FAST.route) == native
