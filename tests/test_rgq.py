"""MV2 M2 RGQ factor arm (semkine/rgq.py): SLR path bit-for-bit without the key, t_z decomposition, wrist-ray gauge
perturbation (training only, own generator), two Adam groups on the parent's schedule."""
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import load_config  # noqa: E402
from semkine.slr import SLRModel  # noqa: E402
from semkine.rgq import RGQModel  # noqa: E402

SLR_CFG = REPO / "configs" / "mv2" / "mv2_slr.yaml"
RGQ_CFG = REPO / "configs" / "mv2" / "mv2_slr_rgq.yaml"


def _inputs(B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(B, 180, 240, 2, generator=g) > 0.97).float() * torch.rand(B, 180, 240, 2, generator=g)
    prev = torch.zeros(B, 51)
    prev[:, 2] = 0.55
    prev[:, 3:6] = torch.tensor([2.2, 0.1, -0.2])
    prev[:, 6:] = 0.05 * torch.randn(B, 45, generator=g)
    return x, prev


def test_without_key_is_slr_bit_for_bit():
    torch.manual_seed(3407)
    a = SLRModel(load_config(SLR_CFG)).eval()
    torch.manual_seed(3407)
    b = RGQModel(load_config(SLR_CFG)).eval()
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    x, prev = _inputs()
    with torch.no_grad():
        assert torch.equal(a(x, prev), b(x, prev))


def test_rgq_same_draws_and_init_function_except_tz():
    torch.manual_seed(3407)
    a = SLRModel(load_config(SLR_CFG)).eval()
    ra = torch.get_rng_state()
    torch.manual_seed(3407)
    b = RGQModel(load_config(RGQ_CFG)).eval()
    assert torch.equal(ra, torch.get_rng_state()), "RGQ must not consume the global RNG"
    sa, sb = a.state_dict(), b.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert set(sb) - set(sa) == {"rgq_tz.weight", "rgq_tz.bias"}
    x, prev = _inputs()
    with torch.no_grad():
        oa, ob = a(x, prev), b(x, prev)
    i = b.rgq_tz_index
    assert i == 2
    keep = [j for j in range(51) if j != i]
    assert torch.equal(oa[:, keep], ob[:, keep])
    # zero-initialised t_z head, masked trunk / prev_mlp t_z: the t_z output is the previous t_z exactly
    assert torch.equal(ob[:, i], prev[:, i])


def test_perturbation_moves_wrist_along_its_ray_training_only():
    torch.manual_seed(0)
    b = RGQModel(load_config(RGQ_CFG))
    x, prev = _inputs(B=5)
    y = prev.clone()
    b.train()
    rng = torch.get_rng_state()
    out = b._unpack_batch((x, prev, y, None, None))
    assert torch.equal(rng, torch.get_rng_state()), "sigma must come from the model's own generator"
    p2 = out[1]
    assert torch.equal(p2[:, 3:], prev[:, 3:]) and torch.equal(out[2], y)
    betas_f, _ = b._resolve_betas_K(prev, None, None)
    _, j0 = b._fk(prev, betas_f)
    _, j1 = b._fk(p2, betas_f)
    w0, w1 = j0[:, 0], j1[:, 0]
    sig = w0.norm(dim=1) / w1.norm(dim=1)
    assert torch.all(sig >= 0.93 - 1e-4) and torch.all(sig <= 1.01 + 1e-4)
    cos = torch.nn.functional.cosine_similarity(w0, w1, dim=1)
    assert torch.all(cos > 1 - 1e-6)
    # rigid: every joint moves by the same vector
    d = (j1 - j0)
    assert torch.allclose(d, d[:, :1].expand_as(d), atol=1e-5)
    b.eval()
    out_e = b._unpack_batch((x, prev, y, None, None))
    assert torch.equal(out_e[1], prev)


def test_two_param_groups_follow_the_parent_schedule():
    b = RGQModel(load_config(RGQ_CFG))
    conf = b.configure_optimizers()
    opt, sched = conf["optimizer"], conf["lr_scheduler"]["scheduler"]
    g0, g1 = opt.param_groups
    assert len(g1["params"]) == 2 and abs(g1["lr"] / g0["lr"] - 0.1) < 1e-9
    n_all = sum(1 for p in b.parameters() if p.requires_grad)
    assert len(g0["params"]) + len(g1["params"]) == n_all
    for _ in range(700):
        opt.step()
        sched.step()
    assert abs(opt.param_groups[1]["lr"] / opt.param_groups[0]["lr"] - 0.1) < 1e-9


def test_gradient_reaches_tz_head():
    torch.manual_seed(0)
    b = RGQModel(load_config(RGQ_CFG)).train()
    x, prev = _inputs()
    out = b(x, prev)
    out[:, 2].sum().backward()
    assert b.rgq_tz.weight.grad is not None and b.rgq_tz.weight.grad.abs().sum() > 0
    assert b.prev_mlp[-1].weight.grad is None or b.prev_mlp[-1].weight.grad[2].abs().sum() == 0


def test_stray_keys_rejected():
    cfg = load_config(SLR_CFG)
    cfg["MODEL"]["RGQ_TZ_LR_MULT"] = 0.1
    with pytest.raises(ValueError):
        RGQModel(cfg)


def test_deepcopy_and_generator_advances():
    import copy
    b = RGQModel(load_config(RGQ_CFG)).train()
    c = copy.deepcopy(b)
    x, prev = _inputs(B=4)
    p1 = b._unpack_batch((x, prev, prev, None, None))[1]
    p2 = b._unpack_batch((x, prev, prev, None, None))[1]
    assert not torch.equal(p1, p2), "successive batches draw new sigmas"
    assert torch.equal(c._unpack_batch((x, prev, prev, None, None))[1], p1), "a copy replays the same stream"
