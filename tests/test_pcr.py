"""PCR CPU tests (docs/EXPERIMENT_RECORDS.md#pcr-prereg-20261105). Run with CUDA_VISIBLE_DEVICES="".

- PCRNet draws r_s2d's initial weights (the only extra parameter is the zero-initialised prior_in) and, while prior_in is
  zero, its outputs equal SlimSR's (r_s2d design) exactly on the same events; a non-zero prior_in changes them.
- prior_maps: the template hand at 0.45 m puts its mass where its vertices project, in each vertex's part channel,
  values in [0, 1); a translated prior moves the map by the projected offset.
- register / FAST.install with kind "pcr" installs the prior-aware sampler and is undone.
- The job list is well formed; the written rule on synthetic numbers.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
NPZ = ROOT / "assets/mano_right.npz"


def _events(n=600, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(80, 160, (n,), generator=g).float()
    y = torch.randint(50, 130, (n,), generator=g).float()
    t = torch.sort(torch.rand(n, generator=g) * 0.05).values
    p = torch.randint(0, 2, (n,), generator=g).float()
    return torch.stack([torch.zeros(n), x, y, t, p], -1), torch.tensor([0, n]), torch.tensor([0.05])


def _nets():
    from semkine import pcr_router as PCR
    from semkine import pfo_router as PR
    torch.manual_seed(0)
    a = PR.build(NPZ, stem="s2d").eval()
    torch.manual_seed(0)
    b = PCR.build(NPZ, stem="s2d").eval()
    return a, b


def _template_prior(shift=(0.0, 0.0, 0.0)):
    prev = torch.zeros(1, 51)
    prev[0, 0:3] = torch.tensor([0.0, 0.0, 0.45]) + torch.tensor(shift)
    K = torch.tensor([[[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]])
    return prev, torch.zeros(1, 10), K


def test_init_and_zero_prior_equivalence():
    from semkine import pcr_router as PCR
    a, b = _nets()
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sb) - set(sa) == {"prior_in.weight"} and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert torch.count_nonzero(sb["prior_in.weight"]) == 0
    ev, ptr, dt = _events()
    prev, betas, K = _template_prior()
    prior = PCR.prior_maps(prev, betas, K, b.vert_part)
    with torch.no_grad():
        oa, ob = a(ev, ptr, dt), b(ev, ptr, dt, prior)
        for k in ("part", "coord", "onhand"):
            assert torch.equal(oa[k], ob[k]), k
        torch.manual_seed(1)
        torch.nn.init.normal_(b.prior_in.weight, std=0.5)
        assert not torch.equal(oa["part"], b(ev, ptr, dt, prior)["part"])


def test_prior_maps_follow_the_projection():
    from semkine import pcr_router as PCR
    _, b = _nets()
    prev, betas, K = _template_prior()
    m = PCR.prior_maps(prev, betas, K, b.vert_part)
    assert m.shape == (1, PCR.N_PRIOR, PCR.H2, PCR.W2) and float(m.min()) >= 0 and float(m.max()) < 1
    u, v, z = PCR.project_prior(prev, betas, K)
    u2, v2 = ((u - 0.5) / 2).round().long()[0], ((v - 0.5) / 2).round().long()[0]
    inb = (u2 >= 0) & (u2 < PCR.W2) & (v2 >= 0) & (v2 < PCR.H2)
    own = m[0, b.vert_part[inb], v2[inb], u2[inb]]
    assert float((own > 0.05).float().mean()) > 0.95
    shifted = PCR.prior_maps(*_template_prior((0.02, 0.0, 0.0)), b.vert_part)
    cx = lambda mm: float((mm.sum(1)[0] * torch.arange(PCR.W2).view(1, -1)).sum() / mm.sum(1)[0].sum())  # noqa: E731
    expect = 600.0 * 0.375 * 0.02 / 0.45 / 2.0
    assert abs((cx(shifted) - cx(m)) - expect) < 1.0


def test_register_installs_prior_sampler(tmp_path):
    from semkine import pcr_router as PCR
    from semkine import s45sm as SM
    from semkine import s45sm_fast as FAST
    from semkine import xag as XAG
    _, b = _nets()
    ck = tmp_path / "pcr.ckpt"
    PCR.save_checkpoint(b, ck, router="pcr")
    PCR.register()
    try:
        st = FAST.install("pcr", str(ck), "pfa", 0.5)
        assert XAG.sample_sources is PCR.pcr_sources and st["sample_sources"] == "semkine.pcr_router.pcr_sources"
        FAST.uninstall()
    finally:
        PCR.unregister()
    assert XAG.sample_sources is SM._NATIVE["sample_sources"]


def test_jobs_and_rule():
    from tools.pcr import jobs as J
    from tools.pcr.analyze import rule
    jobs = J.r1()
    seen = set()
    for j in jobs:
        assert all(d in seen for d in j["deps"]), j["key"]
        seen.add(j["key"])
    final = [j for j in jobs if j["key"] == "final:analyze"][0]
    assert {j["key"] for j in jobs if j["key"].startswith(("read:", "probe:"))} <= set(final["deps"])
    ref = {"R15": 0.34, "F15": 0.41}
    assert rule(11.55, 10.6, ref, ref)["verdict"] == "PCR_WORKS"
    assert rule(11.55, 11.3, ref, ref)["verdict"] == "PCR_NULL"
    assert rule(11.55, 11.0, ref, {"R15": 0.40, "F15": 0.41}) == {"verdict": "PCR_WEAK", "delta_ra": 11.0 - 11.55,
                                                                   "prior_collapse": True}
