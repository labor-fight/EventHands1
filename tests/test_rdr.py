"""RDR routers and their hooks (docs/EXPERIMENT_RECORDS.md#rdr-prereg-20261110, #rdr-trunk-prereg-20261110).
CPU only: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_rdr.py (its own process: it patches router hooks)."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
NPZ = str(ROOT / "assets/mano_right.npz")


def packets(counts, seed=0):
    g = torch.Generator().manual_seed(seed)
    ev = []
    for b, n in enumerate(counts):
        x = torch.randint(60, 170, (n,), generator=g).float()
        y = torch.randint(40, 140, (n,), generator=g).float()
        t = torch.sort(torch.rand(n, generator=g) * 0.05).values
        ev.append(torch.stack([torch.full((n,), float(b)), x, y, t, torch.randint(0, 2, (n,), generator=g).float()], 1))
    ptr = torch.tensor([0] + list(torch.tensor(counts).cumsum(0)))
    return torch.cat(ev), ptr, torch.full((len(counts),), 0.05)


def test_budget_and_forward():
    from semkine import pfo_router as PR, rdr_router as RD, s45sm as SM
    assert SM.n_params(PR.build(NPZ, stem="s2d")) == RD.BUDGET_PARAMS
    ev, ptr, dt = packets([3000, 0, 700])
    for hier in (False, True):
        net = RD.build(NPZ, hier=hier)
        assert SM.n_params(net) <= RD.BUDGET_PARAMS
        out = net(ev, ptr, dt)
        assert out["coord"].shape == (3, SM.MAX_NODES, 3) and out["inside"][out["mask"]].all()
        x0, y0 = net.origin(ev, ptr)
        assert ((x0 >= 0) & (x0 <= SM.WIDTH - net.crop)).all() and ((y0 >= 0) & (y0 <= SM.HEIGHT - net.crop)).all()
        assert int(x0[1]) == SM.WIDTH // 2 - net.crop // 2                      # an empty packet gets the image centre
        face, bary = net.route(out, out["mask"])
        assert face.shape == out["mask"].shape and torch.allclose(bary[out["mask"]].sum(-1), torch.ones(int(out["mask"].sum())), atol=1e-4)
        assert (net.face_part[face][out["mask"]] == net.predicted_part(out)[out["mask"]]).all()


def test_crop_follows_the_events_and_outside_nodes_lose_image_features():
    from semkine import rdr_router as RD
    net = RD.build(NPZ)
    ev, ptr, dt = packets([2000])
    far = ev.clone()
    far[:40, 1] = 2.0                                                             # a few events at the left border
    x0, _ = net.origin(far, ptr)
    assert abs(int(x0[0]) - int(net.origin(ev, ptr)[0][0])) <= 3                 # 2 % outliers do not move the window
    out = net(far, ptr, dt)
    src = out["src"][0][out["mask"][0]]
    assert not out["inside"][0][out["mask"][0]][far[src, 1] == 2.0].any()


def test_set_term_is_the_group_mean_error():
    from semkine import rdr_router as RD
    B, N = 2, 12
    valid = torch.ones(B, N, dtype=torch.bool)
    part = torch.zeros(B, N, dtype=torch.long)
    part[:, 6:] = 3
    point = torch.zeros(B, N, 3)
    coord = torch.randn(B, N, 3) * 0.01
    coord[0, :6] += torch.tensor([2.0, 0.0, 0.0])                                 # one group shifted by 2 cm
    z = RD.set_term({"coord": coord - coord.mean(1, keepdim=True) * 0 }, {"valid": valid, "part": part, "point": point})
    base = RD.set_term({"coord": coord - torch.tensor([2.0, 0.0, 0.0]) * (torch.arange(N) < 6).float()[None, :, None] * torch.tensor([1.0, 0.0])[:, None, None]},
                       {"valid": valid, "part": part, "point": point})
    assert float(z) > float(base) + 0.1
    assert float(RD.set_term({"coord": torch.zeros(B, N, 3)}, {"valid": valid, "part": part, "point": point})) == 0.0


def test_pfr_kind_loads_an_rdr_checkpoint(tmp_path):
    from semkine import pfo_router as PR, rdr_hooks as RH, rdr_router as RD, s45sm_fast as FAST
    net = RD.build(NPZ, hier=True)
    p = tmp_path / "x.ckpt"
    RD.save_checkpoint(net, p)
    RH.enable()
    loaded, _ = PR.load(p)
    assert isinstance(loaded, RD.RDRNet) and loaded.design["hier"]
    ev, ptr, dt = packets([1500])
    out = FAST.router_forward(loaded, ev, ptr, dt, *__import__("semkine.xyz_mesh", fromlist=["stride_sample"]).stride_sample(ptr, 2048))
    face, bary = FAST.route(loaded, "sr", out, out["mask"])
    assert "patch" in out and face.shape == out["mask"].shape


def test_rule_and_registry():
    from tools.rdr import decide as D
    ref = {s: {"coherent_mm": v} for s, v in zip(D.SEQS, (7.8, 8.0, 3.4))}
    f = lambda a, b, c: D.rule(ref, {s: {"coherent_mm": v} for s, v in zip(D.SEQS, (a, b, c))})["verdict"]
    assert f(5.8, 5.8, 2.1) == "BETTER" and f(7.0, 7.2, 3.5) == "EQUIVALENT" and f(7.0, 7.0, 3.0) == "BETTER"
    assert f(7.7, 7.9, 3.3) == "EQUIVALENT" and f(8.3, 8.5, 3.6) == "WORSE"
    from tools.pmr import arms_any5  # noqa: F401
    from tools.pmr import arms as A, arms10 as A10, arms_rol2 as R2
    c, b = A10.build_config("d1rv3", 3407), A.build_config("d1r", 3407)
    assert c["PFO"]["ROUTER_KIND"] == "pfr" and c["PFO"]["ROUTER"] == "rdr_v3" and c["LOSS"] == b["LOSS"] and c["AUG"] == b["AUG"]
    assert {k for k in c["TRAIN"] if c["TRAIN"][k] != b["TRAIN"][k]} == {"OUTPUT_DIR", "RUN_NAME"}
    w = R2.build_config("d1rw3_rol5", 3407)
    assert w["MODEL"]["INIT_FROM"] == R2.build_config("d1r_rol5", 3407)["MODEL"]["INIT_FROM"] and w["PFO"]["ROUTER"] == "rdr_v3"
