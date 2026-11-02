"""CPU tests: the slim patch router `srp` (semkine/s45sm_srp.py) and its training loss (tools/s45sm/router_train.py)."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import s45rt_router as RT  # noqa: E402
from semkine import s45sm as SM  # noqa: E402
from semkine import s45sm_fast as FAST  # noqa: E402
from semkine import s45sm_srp as SRP  # noqa: E402
from semkine import xag as XAG  # noqa: E402
from semkine import xyz_mesh as XM  # noqa: E402
from semkine.events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y  # noqa: E402

NPZ = ROOT / "assets/mano_right.npz"


def packets(seed=0, n_per=(700, 0, 1500)):
    g = torch.Generator().manual_seed(seed)
    rows, ptr = [], [0]
    for b, n in enumerate(n_per):
        ev = torch.zeros(n, 5)
        ev[:, EV_BATCH] = b
        ev[:, EV_X] = torch.randint(80, 160, (n,), generator=g).float()
        ev[:, EV_Y] = torch.randint(50, 120, (n,), generator=g).float()
        ev[:, EV_T] = torch.sort(torch.rand(n, generator=g) * 0.05).values
        ev[:, EV_P] = (torch.rand(n, generator=g) > 0.5).float()
        rows.append(ev)
        ptr.append(ptr[-1] + n)
    return torch.cat(rows), torch.tensor(ptr), torch.full((len(n_per),), 0.05)


@pytest.fixture()
def srp_ckpt(tmp_path):
    torch.manual_seed(5)
    net = SRP.build(NPZ)
    with torch.no_grad():
        net.onhand.bias.fill_(0.3)
    path = tmp_path / "srp.ckpt"
    SRP.save_checkpoint(net, path, step=0)
    return path


@pytest.fixture()
def registered():
    SRP.register()
    yield
    SRP.unregister()


def test_widths_params_and_round_trip(srp_ckpt):
    t0p_widths = (RT.TRUNK, RT.FPN, RT.HEAD)
    net = SRP.build(NPZ)
    assert (RT.TRUNK, RT.FPN, RT.HEAD) == t0p_widths == ((32, 64, 128, 256), 128, 256)
    assert SM.n_params(net) == 137787 and net.patch.out_features == 279
    assert SM.n_params(net) + 459059 <= 600000
    loaded, obj = SRP.load(srp_ckpt)
    assert obj["kind"] == "srp" and SM.n_params(loaded) == 137787
    ref = torch.load(str(srp_ckpt))["state_dict"]
    assert all(torch.equal(v, ref[k]) for k, v in loaded.state_dict().items())
    full = srp_ckpt.with_name("t0p_like.ckpt")
    RT.save_checkpoint(RT.build(NPZ), full)
    with pytest.raises(ValueError):
        SRP.load(full)


def test_register_routes_as_t0p(srp_ckpt, registered):
    assert "srp" in SM.ROUTERS and FAST.route is not SRP._NATIVE["fast_route"]
    with pytest.raises(RuntimeError):
        SRP._STATE["registered"] = False
        try:
            SRP.register()                                                       # already patched
        finally:
            SRP._STATE["registered"] = True
    net = SM._load_router("srp", srp_ckpt)
    assert isinstance(net, RT.T0PNet) and SM.n_params(net) == 137787
    events, ptr, dt = packets(1)
    out = net(events, ptr, dt)
    sel = out["mask"] & (torch.rand(out["mask"].shape, generator=torch.Generator().manual_seed(3)) > 0.4)
    assert sel.any() and (~sel & out["mask"]).any()
    f1, b1 = FAST.route(net, "srp", out, sel)
    f2, b2 = SRP._NATIVE["fast_route"](net, "t0p", out, sel)
    assert torch.equal(f1, f2) and torch.equal(b1, b2)
    r1 = SM.route_nodes(net, "srp", events, ptr, dt, 0.5)
    r2 = SRP._NATIVE["route_nodes"](net, "t0p", events, ptr, dt, 0.5)
    for a, b in zip(r1[1:], r2[1:]):
        assert torch.equal(a, b)


def test_fast_install_pfa_and_restore(srp_ckpt, registered):
    state = FAST.install("srp", srp_ckpt, "pfa", 0.5)
    try:
        assert state["router"] == "srp" and state["router_params"] == 137787 and state["fast"] is True
        events, ptr, dt = packets(2)

        class Batch:
            pass
        batch = Batch()
        batch.events, batch.ptr, batch.delta_t_s = events, ptr, dt
        src, mask = XM.stride_sample(ptr, SM.MAX_NODES)
        s = XAG.sample_sources(batch, src, mask)
        net = SM._net(src.device)
        out = FAST.router_forward(net, events, ptr, dt, src, mask)
        routed = mask & (torch.sigmoid(out["onhand"].float()) >= 0.5)
        face, _, _ = net.route(out, routed)
        assert torch.equal(s["xag_valid"], routed) and torch.equal(s["xag_face"][routed], face[routed])
        assert XAG.route_association is FAST.pfa_association
    finally:
        FAST.uninstall()
    assert SM.installed() is None and XAG.route_association is SM._NATIVE["route_association"]
    SRP.unregister()
    assert SM.ROUTERS == ("t0p", "sr") and SM._load_router is SRP._NATIVE["load"] and FAST.route is SRP._NATIVE["fast_route"]
    SRP.register()                                                               # the fixture unregisters


def test_kd_loss_terms():
    from tools.s45sm.router_train import kd_loss_patch as kd_loss
    g = torch.Generator().manual_seed(0)
    B, N = 2, 32
    mask = torch.rand(B, N, generator=g) > 0.2
    logits = torch.randn(B, N, 279, generator=g)
    t = {"patch": logits, "onhand": torch.randn(B, N, generator=g) * 3, "coord": torch.randn(B, N, 3, generator=g)}
    s = {"patch": logits.clone(), "onhand": t["onhand"].clone(), "coord": t["coord"].clone(), "mask": mask}
    kd = {"T": 2.0, "lambda_patch": 1.0, "lambda_coord": 0.5, "lambda_onhand": 0.5}
    _, parts = kd_loss(s, t, kd)
    assert abs(float(parts["kd_patch"])) < 1e-5 and float(parts["kd_coord"]) == 0.0
    s["patch"] = torch.randn(B, N, 279, generator=g)
    total, parts = kd_loss(s, t, kd)
    assert float(parts["kd_patch"]) > 0.01
    assert torch.isclose(total, parts["kd_patch"] + 0.5 * parts["kd_coord"] + 0.5 * parts["kd_onhand"])
