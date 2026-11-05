"""ACU PR2 CPU tests (docs/EXPERIMENT_RECORDS.md#acu-pr2-prereg-20261104). Run with CUDA_VISIBLE_DEVICES="".

* the causal grid sums equal a brute-force loop (every cell of every scale, the global window)
* PR2 is causal: changing every window event after node m leaves nodes <= m bitwise unchanged; the whole window and the
  window cut right after node m give the same outputs for node m up to float64 matrix-product rounding (<= 1e-12)
* checkpoints round-trip through the dispatching loader (PR2 and PR kinds), install / uninstall restore PR's functions
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
pytestmark = pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES", None) != "", reason="CPU tests: CUDA_VISIBLE_DEVICES=''")


def test_grid_sums_match_brute_force():
    import torch
    from semkine import acu_pr2 as P2
    g = torch.Generator().manual_seed(1)
    n = 600
    b = torch.sort(torch.randint(0, 2, (n,), generator=g)).values
    pos = torch.cat([torch.arange(int((b == i).sum())) for i in range(2)])
    t = torch.cat([torch.sort(torch.randint(0, 60000, (int((b == i).sum()),), generator=g)).values for i in range(2)])
    x = torch.randint(0, 240, (n,), generator=g)
    y = torch.randint(0, 180, (n,), generator=g)
    f = torch.rand(n, 3, generator=g, dtype=torch.float64) * 2 - 1
    q = torch.arange(0, n, 7)
    tau = 40000
    for cell in (8, 20, 50):
        s, c = P2.causal_grid_sums(b, pos, t, x, y, f, cell, 2, tau, q)
        for r, i in enumerate(q.tolist()):
            k = 0
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    sel = ((b == b[i]) & (pos < pos[i]) & (t[i] - t <= tau) & (x // cell == x[i] // cell + dx)
                           & (y // cell == y[i] // cell + dy))
                    assert int(c[r, k]) == int(sel.sum())
                    assert float((s[r, k].double() - f[sel].sum(0)).abs().max()) <= 1e-4
                    k += 1
    s, c = P2.causal_grid_sums(b, pos, t, x, y, f, None, 0, tau, q)
    for r, i in enumerate(q.tolist()):
        sel = (b == b[i]) & (pos < pos[i]) & (t[i] - t <= tau)
        assert int(c[r, 0]) == int(sel.sum()) and float((s[r, 0].double() - f[sel].sum(0)).abs().max()) <= 1e-4


@pytest.fixture(scope="module")
def batch_and_net():
    import torch
    from semkine import acu_hist as AH
    from semkine import acu_pr2 as P2
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
    return b, P2.build(cfg["MANO"]["NPZ"]).double().eval(), cfg


def _run(net, b, ev, hist=None, ptr=None, sl=slice(None)):
    import torch
    with torch.no_grad():
        return net(ev, b.ptr if ptr is None else ptr, b.delta_t_s[sl], (b.hist_events if hist is None else hist).double(),
                   b.hist_ptr if hist is None else b.hist_ptr[:2], b.k_win()[sl], b.k_hist[sl])


def test_pr2_causal_and_prefix_exact(batch_and_net):
    import torch
    from semkine import acu_pr_det as PD
    b, net, _ = batch_and_net
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


def test_checkpoint_roundtrip_and_install(batch_and_net, tmp_path):
    import torch
    from semkine import acu_pr as PR
    from semkine import acu_pr2 as P2
    b, net, cfg = batch_and_net
    native = (PR.build, PR.load, PR.save_checkpoint, PR.n_params)
    assert P2.install() == "acu_pr2"
    try:
        assert PR.load is P2.load and PR.build is P2.build
        p2 = tmp_path / "pr2.ckpt"
        PR.save_checkpoint(net.float(), p2, step=1)
        back, obj = PR.load(p2)
        assert obj["kind"] == "acu_pr2" and isinstance(back, P2.PR2Net)
        for k, v in net.state_dict().items():
            assert torch.equal(v, back.state_dict()[k])
        torch.manual_seed(0)
        pr = P2._native["build"](cfg["MANO"]["NPZ"])
        p1 = tmp_path / "pr.ckpt"
        P2._native["save_checkpoint"](pr, p1, step=1)
        back1, obj1 = PR.load(p1)
        assert obj1["kind"] == "acu_pr" and type(back1) is PR.PRNet
        d, c = net.design, net.design["width"]
        same_head = P2._native["build"](cfg["MANO"]["NPZ"], head=d["head"])
        sc = d["sc_in"] * d["sc_hidden"] + d["sc_hidden"] + d["sc_hidden"] * c + c
        assert P2.n_params(net) == PR.n_params(same_head) + sc + c * d["head"]      # + shape context, + its head input
    finally:
        P2.uninstall()
        net.double()
    assert (PR.build, PR.load, PR.save_checkpoint, PR.n_params) == native


def _subprocess(code):
    import subprocess
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def test_namespace_rebinds_roots_and_router():
    out = _subprocess(
        "import sys; sys.path[:0] = ['.', 'model']\n"
        "from tools.acu import ns_q; ns_q.enter()\n"
        "from tools.acu import arms_s2, run_pr, router_pr, train_s2\n"
        "from tools.acu.common import OUT, CONFIGS\n"
        "from semkine import acu_pr as PR, acu_pr2 as P2, acu_pr_det as PD\n"
        "assert OUT.name == '20261104q' and CONFIGS.name == 'acu_20261104q' and OUT.parent.parent.name == 'outputs'\n"
        "assert str(arms_s2.PR_CHECKPOINT).endswith('outputs/acu/20261104q/routers/pr/pr_final.ckpt')\n"
        "assert router_pr.router_dir().parent.parent == OUT and run_pr.OUT == OUT and train_s2.OUT == OUT\n"
        "assert arms_s2.config_path('c00', 3407).parent == CONFIGS\n"
        "assert PR.load is P2.load and PR.build is P2.build and PR.causal_cell_sums is PD.causal_cell_sums\n"
        "assert ns_q.base_out().name == '20261104'\n"
        "print('ok')\n")
    assert out.strip().endswith("ok")
    bad = _subprocess(
        "import sys; sys.path[:0] = ['.', 'model']\n"
        "import tools.acu.run_pr\n"
        "from tools.acu import ns_q\n"
        "try:\n    ns_q.enter()\nexcept Exception as e:\n    print('refused', type(e).__name__)\n")
    assert "refused" in bad


def test_q_rewrite():
    out = _subprocess(
        "import sys; sys.path[:0] = ['.', 'model']\n"
        "from tools.acu import jobs_q as JQ\n"
        "from tools.acu.common import PY, ROOT\n"
        "r = JQ.rewrite(['env', 'X=1', PY, str(ROOT / 'tools/acu/router_pr.py'), '--phase', 'formal'])\n"
        "assert r[3] == str(ROOT / 'tools/acu/router_pr2.py') and r[4:] == ['--phase', 'formal']\n"
        "r = JQ.rewrite([PY, str(ROOT / 'tools/acu/train_s2.py'), '--config', 'c'])\n"
        "assert r[1] == str(ROOT / 'tools/acu/train_s2q.py')\n"
        "assert JQ.rewrite([PY, '-m', 'tools.acu.run_s2', 'eval']) == [PY, '-m', 'tools.acu.qrun', 'tools.acu.run_s2', 'eval']\n"
        "assert JQ.rewrite([PY, '-m', 'tools.acu.jobs_s3', 'plan_a']) == [PY, '-m', 'tools.acu.jobs_s3q', 'plan_a']\n"
        "assert JQ.rewrite([PY, '-m', 'tools.acu.freeze', 'verify']) == [PY, '-m', 'tools.acu.freeze', 'verify']\n"
        "jobs = JQ.rename([{'key': 'a', 'deps': [], 'command': [PY]}, {'key': 'b', 'deps': ['a', 'x'], 'command': [PY]}],\n"
        "                 extra_deps=('q:freeze',))\n"
        "assert [j['key'] for j in jobs] == ['q:a', 'q:b'] and jobs[1]['deps'] == ['q:a', 'x', 'q:freeze']\n"
        "print('ok')\n")
    assert out.strip().endswith("ok")
