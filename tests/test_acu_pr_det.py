"""ACU engineering amendment 2 CPU tests (docs/EXPERIMENT_RECORDS.md#acu-amend2-20261104). Run with CUDA_VISIBLE_DEVICES="".

* the fixed-point window sums equal semkine/acu_pr.py's float64 ones up to the 2^-20 rounding of the summands
* with them installed, PR stays causal: changing every window event after node m leaves nodes <= m bitwise unchanged,
  and the whole window and the window cut right after node m give the same outputs for node m up to the float64 rounding
  of matrix products over different batch sizes (<= 1e-12; the window sums themselves are exact)
* the re-registered job list: every key prefixed and unique, dependencies inside the set renamed, every entry point that
  runs PR rewritten (scripts -> wrappers, modules -> detrun, stage-3 planner -> jobs_s3r2), nothing else changed
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
pytestmark = pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES", None) != "", reason="CPU tests: CUDA_VISIBLE_DEVICES=''")


def test_fixed_point_sums_match_native():
    import torch
    from semkine import acu_pr as PR
    from semkine import acu_pr_det as PD
    g = torch.Generator().manual_seed(0)
    n = 4000
    b = torch.sort(torch.randint(0, 3, (n,), generator=g)).values
    pos = torch.cat([torch.arange(int((b == i).sum())) for i in range(3)])
    t = torch.cat([torch.sort(torch.randint(0, 50000, (int((b == i).sum()),), generator=g)).values for i in range(3)])
    x = torch.randint(0, 240, (n,), generator=g).float()
    y = torch.randint(0, 180, (n,), generator=g).float()
    f = torch.rand(n, 48, generator=g) * 4
    det = PD.causal_cell_sums(b, pos, t, x, y, f, 30, 20000)
    nat = PD._native(b, pos, t, x, y, f, 30, 20000)
    for u, v in zip(det, nat):
        assert u.shape == v.shape and u.dtype == v.dtype
        assert float((u - v).abs().max()) <= 4e-3                 # <= n * 2^-21 per sum, plus float32 output rounding
    assert torch.equal(det[1], nat[1]) and torch.equal(det[3], nat[3])   # counts are exact either way
    assert PR.causal_cell_sums is not PD.causal_cell_sums or PD.install() == "acu_pr_det"
    PD.uninstall()
    assert PR.causal_cell_sums is PD._native


@pytest.fixture(scope="module")
def batch_and_net():
    import torch
    from semkine import acu_hist as AH
    from semkine import acu_pr as PR
    from tools.s45sm import router_train as S
    torch.set_num_threads(8)
    cfg = S._data({"dz": True})
    AH.install_dataset()
    try:
        ds = S.dataset(cfg, "ylf")
        b = AH.collate_hist([ds[i] for i in (3, 7000, 30000)])
    finally:
        AH.uninstall()
    torch.manual_seed(0)
    return b, PR.build(cfg["MANO"]["NPZ"]).double().eval()


def _run(net, b, ev, hist=None, ptr=None, sl=slice(None)):
    import torch
    with torch.no_grad():
        return net(ev, b.ptr if ptr is None else ptr, b.delta_t_s[sl], (b.hist_events if hist is None else hist).double(),
                   b.hist_ptr if hist is None else b.hist_ptr[:2], b.k_win()[sl], b.k_hist[sl])


def test_causal_and_prefix_exact_with_fixed_point(batch_and_net):
    import torch
    from semkine import acu_pr_det as PD
    b, net = batch_and_net
    PD.install()
    try:
        ev = b.events.double()
        out = _run(net, b, ev)
        for bi in range(b.batch_size):
            n = int(out["mask"][bi].sum())
            if n < 4:
                continue
            m = n // 3
            cut = int(out["src"][bi, m])
            ev2 = ev.clone()
            later = (torch.arange(ev.shape[0]) > cut) & (ev[:, 0] == bi)
            gen = torch.Generator().manual_seed(bi)
            ev2[later, 1] = torch.randint(0, 240, (int(later.sum()),), generator=gen).double()
            ev2[later, 2] = torch.randint(0, 180, (int(later.sum()),), generator=gen).double()
            out2 = _run(net, b, ev2)
            for k in ("part", "coord", "onhand"):
                assert torch.equal(out[k][bi, : m + 1], out2[k][bi, : m + 1]), (bi, k)
        n = int(out["mask"][0].sum())
        hist0 = b.hist_events[: int(b.hist_ptr[1])]
        for m in (0, n // 2, n - 1):
            cut = int(out["src"][0, m]) + 1
            part = _run(net, b, ev[:cut], hist=hist0, ptr=torch.tensor([0, cut]), sl=slice(0, 1))
            for k in ("part", "coord", "onhand"):
                d = float((out[k][0, m] - part[k][0, m]).abs().max())
                assert d <= 1e-12, (m, k, d)
    finally:
        PD.uninstall()


def test_reregistered_jobs():
    import json
    from tools.acu import jobs_amend2 as JA
    from tools.acu.common import OUT
    jobs = JA.build()
    keys = [j["key"] for j in jobs]
    assert len(keys) == len(set(keys)) and all(k.startswith(JA.PREFIX) for k in keys)
    old = []
    for s in JA.SOURCES:
        old += json.loads((OUT / "jobs" / f"{s}.json").read_text())
    assert len(jobs) == len(old) + 1 and jobs[0]["key"] == JA.FREEZE["key"]
    known = set(keys)
    for j, o in zip(jobs[1:], old):
        assert j["key"] == JA.PREFIX + o["key"] and j["kind"] == o["kind"] and j.get("pairs") == o.get("pairs")
        assert all(d in known for d in j["deps"]), j["key"]          # never an old (skipped) key
        assert JA.FREEZE["key"] in j["deps"]
        flat = " ".join(j["command"])
        assert "tools/acu/router_pr.py" not in flat and "tools/acu/train_s2.py" not in flat
        for mod in JA.DET_MODULES:
            if mod in o["command"]:
                i = j["command"].index(mod)
                assert j["command"][i - 1] == "tools.acu.detrun" and j["command"][i - 2] == "-m"
        if "tools.acu.jobs_s3" in o["command"]:
            assert "tools.acu.jobs_s3r2" in j["command"]
        stripped = [c for c in j["command"] if c != "tools.acu.detrun"]
        mapped = [c.replace("router_prd.py", "router_pr.py").replace("train_s2d.py", "train_s2.py")
                  .replace("tools.acu.jobs_s3r2", "tools.acu.jobs_s3") for c in stripped]
        assert mapped == o["command"], j["key"]
