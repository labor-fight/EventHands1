"""ACU stage-2 causal pipeline CPU tests (docs/EXPERIMENT_RECORDS.md#acu-s2-prereg-20261104). CUDA_VISIBLE_DEVICES="".

* training packets: the history is the H before the window (one realisation); b1 samples equal orig samples except the prior
  on the B1 branches
* evaluation packets: the history is the recording's H before the window (none at a run start)
* the whole observation path of one forward (causal-stride nodes, history-aware tokens, PR routes, CA association) is
  prefix-stable: randomising every window event after node m leaves every per-node quantity of nodes <= m unchanged
  (untrained PR weights and B2's trunk weights: the property does not depend on training)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
pytestmark = pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES", None) != "", reason="CPU tests: CUDA_VISIBLE_DEVICES=''")


def test_training_history_and_b1_pairing():
    from tools.acu import run_s2 as R
    rows = R.b1_pairing(3407, "preflight")
    assert all(r["ok"] for r in rows), [r for r in rows if not r["ok"]]
    assert len({r["branch"] for r in rows}) >= 2


def test_training_history_window():
    from semkine import acu_causal as AC
    from semkine.dataset import build_dataset
    from tools.acu import arms_s2 as A
    cfg = A.build_config("c00", 3407, "preflight")
    comps = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    ds = build_dataset(cfg, "train", comps, train=True)
    for idx in (0, 777, 31337):
        pk = AC._getitem(ds, idx)
        h = pk.meta["acu_hist"]
        if len(h):
            assert h[:, 2].max() < 0 and h[:, 2].min() >= -0.050 - 1e-6
            assert np.all(np.diff(h[:, 2]) >= 0)
        assert pk.events[:, 2].min() >= 0 and pk.meta["acu_k_hist"] >= 1


@pytest.fixture(scope="module")
def ca_net(tmp_path_factory):
    import torch
    from semkine import acu_ca as CA
    from semkine import acu_causal as AC
    from semkine import acu_pr as PR
    from semkine import pfo_router as PRR
    from tools.b2ub.run import deployed_router_for_mixes
    from tools.pfo import nets as N
    from model import MNISTModel
    torch.set_num_threads(8)
    torch.manual_seed(0)
    ck = tmp_path_factory.mktemp("pr") / "pr_untrained.ckpt"
    net0 = PR.build(str(ROOT / "assets/mano_right.npz"))
    with torch.no_grad():
        net0.onhand.bias.fill_(3.0)                                   # route every node, so CA has identities to use
    PR.save_checkpoint(net0, ck)
    PRR.register()
    deployed_router_for_mixes()
    CA.install("gshare")
    sp = json.loads((ROOT / "outputs/pfo2/20261103/specs/pfo2_fgs_s3407.json").read_text())
    sp = {**sp, "checkpoint": Path(sp["checkpoint"]), "config": Path(sp["config"])}
    net = N.load("acu_causal_test", device=torch.device("cpu"), spec=sp)
    AC.install_model(MNISTModel, ck, "ca")
    net.model.xyz_keep_aux = True
    yield net
    AC.uninstall()
    CA.uninstall()


def _forward(net, batch):
    import torch
    with torch.no_grad():
        net.model.forward_packet(batch)
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in net.model.xyz_last.items()}


def test_eval_history_and_prefix_stability(ca_net):
    import torch
    from semkine import acu_causal as AC
    from tools.pfo import engine as E
    from tools.pfo import nets as N
    net = ca_net
    root, seqs = E.sequences(net.cfg, ("zgz_local",))
    ctx = N.sequence(root, seqs[0][1], "zgz_local", torch.device("cpu"))
    net.bind(ctx)
    rows = E.endpoint_rows(ctx.aux)
    assert int(AC.eval_batch(ctx, rows[0][1], AC.run_start(ctx.aux, rows[0][1]), torch.zeros(1, 51),
                             torch.device("cpu")).hist_ptr[-1]) == 0
    _, end, prev_ms, _ = rows[700]
    prev = torch.from_numpy(np.asarray(ctx.gt[int(prev_ms)], np.float32)).view(1, -1)
    b = AC.eval_batch(ctx, end, AC.run_start(ctx.aux, end), prev, torch.device("cpu"))
    assert int(b.hist_ptr[-1]) > 0 and float(b.hist_events[:, 3].max()) < 0
    full = _forward(net, b)
    m_total = int(full["mask"].sum())
    assert m_total > 10 and int(full["xag_valid"].sum()) > 0
    m = m_total // 2
    cut = int(full["src"][0, m])
    ev2 = b.events.clone()
    later = torch.arange(ev2.shape[0]) > cut
    g = torch.Generator().manual_seed(3)
    ev2[later, 1] = torch.randint(0, 240, (int(later.sum()),), generator=g).float()
    ev2[later, 2] = torch.randint(0, 180, (int(later.sum()),), generator=g).float()
    ev2[later, 4] = torch.randint(0, 2, (int(later.sum()),), generator=g).float()
    b2 = type(b)(**{**b.__dict__, "events": ev2})
    alt = _forward(net, b2)
    keep = torch.zeros_like(full["mask"])
    keep[0, : m + 1] = True
    assert torch.equal(full["src"][keep], alt["src"][keep])
    for k in ("xag_part", "xag_face", "xag_bary", "xag_valid"):
        assert torch.equal(full[k][keep], alt[k][keep]), k
    for k in ("vid", "A", "q", "z", "ca_geo"):
        assert torch.equal(full["assoc"][k][keep], alt["assoc"][k][keep]), k
    assert torch.equal(full["tok"][keep], alt["tok"][keep])
