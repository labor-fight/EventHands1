"""CPU tests: semkine/s45sm_dw.py (depthwise patch router `dw`, docs/EXPERIMENT_RECORDS.md#s45sm-r8-prereg-20261031)."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import s45rt_router as RT  # noqa: E402
from semkine import s45sm as SM  # noqa: E402
from semkine import s45sm_dw as DW  # noqa: E402
from semkine import s45sm_fast as FAST  # noqa: E402
from semkine import s45sm_srp as SRP  # noqa: E402
from semkine import xag as XAG  # noqa: E402
from semkine import xyz_mesh as XM  # noqa: E402
from semkine.events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y  # noqa: E402
from tools.s45sm.router_train import kd_loss_patch as kd_loss  # noqa: E402

NPZ = ROOT / "assets/mano_right.npz"
T0P = ROOT / "outputs/s45rt/20261031/t0p/t0p_final.ckpt"


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


class Batch:
    def __init__(self, events, ptr, dt):
        self.events, self.ptr, self.delta_t_s = events, ptr, dt


def test_design_and_round_trip(tmp_path):
    torch.manual_seed(0)
    net = DW.build(NPZ)
    assert SM.n_params(net) == 137291 and 459059 + SM.n_params(net) <= 600000
    assert (RT.TRUNK, RT.FPN, RT.HEAD, RT.Block) == ((32, 64, 128, 256), 128, 256, RT.Block) and RT.Block is not DW.DWBlock
    assert all(isinstance(s, DW.DWBlock) for s in net.stages) and net.n_patches == 279
    path = tmp_path / "dw.ckpt"
    DW.save_checkpoint(net, path, step=1)
    back, obj = DW.load(path)
    assert obj["kind"] == "dw" and all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), back.state_dict().values()))
    with pytest.raises(ValueError):
        SRP.load(path)
    srp = tmp_path / "srp.ckpt"
    SRP.save_checkpoint(SRP.build(NPZ), srp)
    with pytest.raises(ValueError):
        DW.load(srp)


@torch.no_grad()
def test_registration_composes_with_srp(tmp_path):
    torch.manual_seed(1)
    net = DW.build(NPZ).eval()
    ckpt = tmp_path / "dw.ckpt"
    DW.save_checkpoint(net, ckpt)
    srp_ckpt = tmp_path / "srp.ckpt"
    SRP.save_checkpoint(SRP.build(NPZ), srp_ckpt)
    native = (SM.ROUTERS, SM._load_router, SM.route_nodes, FAST.route)
    events, ptr, dt = packets(2)
    src, mask = XM.stride_sample(ptr, SM.MAX_NODES)
    sel = mask & (torch.rand(mask.shape, generator=torch.Generator().manual_seed(3)) > 0.4)
    try:
        SRP.register()
        DW.register()
        assert {"srp", "dw"} <= set(SM.ROUTERS)
        loaded = SM._load_router("dw", ckpt)
        assert isinstance(loaded.stages[0], DW.DWBlock) and SM.n_params(loaded) == 137291
        assert SM.n_params(SM._load_router("srp", srp_ckpt)) == 137787
        a = SM.route_nodes(loaded, "dw", events, ptr, dt, 0.5)
        b = native[2](loaded, "t0p", events, ptr, dt, 0.5)
        assert all(torch.equal(x, y) for x, y in zip(a[1:], b[1:]))
        out = FAST.router_forward(loaded, events, ptr, dt, src, mask)
        assert all(torch.equal(x, y) for x, y in zip(FAST.route(loaded, "dw", out, sel), native[3](loaded, "t0p", out, sel)))
        FAST.install("dw", ckpt, "pfa", 0.5)
        try:
            got = XAG.sample_sources(Batch(events, ptr, dt), src, mask)
            routed = mask & (torch.sigmoid(out["onhand"].float()) >= 0.5)
            face, bary = native[3](loaded, "t0p", out, routed)
            assert torch.equal(got["xag_face"], face) and torch.equal(got["xag_valid"], routed)
        finally:
            FAST.uninstall()
    finally:
        DW.unregister()
        SRP.unregister()
    assert (SM.ROUTERS, SM._load_router, SM.route_nodes, FAST.route) == native


def test_distillation_step_runs():
    torch.manual_seed(2)
    student = DW.build(NPZ)
    teacher, _ = RT.load(T0P)
    events, ptr, dt = packets(4, n_per=(700, 500))
    out = student(events, ptr, dt)
    with torch.no_grad():
        tout = teacher.eval()(events, ptr, dt)
    assert torch.equal(out["src"], tout["src"])
    total, parts = kd_loss(out, tout, {"T": 2.0, "lambda_patch": 1.0, "lambda_coord": 0.5, "lambda_onhand": 0.5})
    total.backward()
    assert torch.isfinite(total) and set(parts) == {"kd_patch", "kd_coord", "kd_onhand"}
    assert all(p.grad is not None for p in student.parameters() if p.requires_grad)
