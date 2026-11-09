"""TPD LC router CPU tests (docs/EXPERIMENT_RECORDS.md#tpd-lc-prereg-20261105). Run with CUDA_VISIBLE_DEVICES="" in a
process of its own.

- LC built from r_s2d routes like r_s2d before fine-tuning (whatever the history); the history channels change the output
  once their stem columns are non-zero.
- history_image: the most recent event's normalised time and the count per pixel, polarities merged.
- The router dataset with the history: the current packet is the plain dataset's sample with the window fixed to 50 ms;
  without augmentation the history is the raw events of the 200 ms before the packet, times from the history origin; the
  control's history is empty; collate keeps the packets' rows.
- history_events (deployment) returns the same raw events from a sequence context.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import tpd_lc as LC  # noqa: E402

R_S2D = ROOT / "outputs/pfo/20261103/routers/r_s2d/r_s2d_final.ckpt"


def test_history_image():
    ev = torch.tensor([[0, 3, 2, 0.01, 1], [0, 3, 2, 0.15, 0], [1, 5, 4, 0.10, 1]], dtype=torch.float32)
    img = LC.history_image(ev, torch.tensor([0, 2, 3]), 0.2, height=8, width=8)
    assert img.shape == (2, 2, 8, 8)
    assert torch.isclose(img[0, 0, 2, 3], torch.tensor(0.75)) and torch.isclose(img[0, 1, 2, 3], torch.log1p(torch.tensor(2.0)))
    assert torch.isclose(img[1, 0, 4, 5], torch.tensor(0.5)) and img[0].sum() > 0 and img[1, :, 2, 3].sum() == 0


@pytest.fixture(scope="module")
def router_ds():
    from semkine.dataset import build_dataset
    from tools.s45sm import router_train as S
    cfg = S._data({"dz": True})
    comps = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    return cfg, comps, build_dataset


def test_dataset_and_collate(router_ds):
    import copy
    cfg, comps, build = router_ds
    plain = copy.deepcopy(cfg)
    plain["AUG"]["DOMRAND"]["ENABLED"] = False
    ref = build(plain, "train", comps, train=True)
    ref.fixed_window = 50
    ds = LC.upgrade(build(plain, "train", comps, train=True))
    for idx in (1000, 500000):
        pk, hist = ds[idx]
        rp = ref[idx]
        assert np.array_equal(pk.events, rp.events) and np.array_equal(pk.xag_face, rp.xag_face)
        assert pk.meta["window_ms"] == 50 and np.array_equal(pk.target, rp.target)
        start = pk.meta["end_idx"] - 49
        h = ds._handle(int(ds.index[idx][0]))
        a0, a1 = int(h.offsets[start - 200]), int(h.offsets[start])
        assert len(hist) == a1 - a0 and len(hist) > 0
        raw = np.asarray(h.events[a0:a1])
        assert np.array_equal(hist[:, 0], raw[:, 0]) and np.array_equal(hist[:, 1], raw[:, 1])
        assert hist[:, 2].min() >= 0 and hist[:, 2].max() < 0.2
    d0 = LC.upgrade(build(plain, "train", comps, train=True), empty=True)
    assert len(d0[1000][1]) == 0
    aug = LC.upgrade(build(cfg, "train", comps, train=True))
    items = [aug[i] for i in (10, 20000, 300000)]
    batch = LC.collate_hist(items)
    assert batch.batch_size == 3 and int(batch.hist_ptr[-1]) == sum(len(h) for _, h in items)
    assert torch.equal(batch.hist_events[:, 0].long(), torch.repeat_interleave(torch.arange(3), torch.diff(batch.hist_ptr)))


def test_lc_init_routes_like_r_s2d(router_ds):
    from semkine import pfo_router as PR
    cfg, comps, build = router_ds
    ds = LC.upgrade(build(cfg, "train", comps, train=True))
    batch = LC.collate_hist([ds[i] for i in (7, 70000, 700000, 1700000)])
    base, _ = PR.load(R_S2D)
    lc = LC.from_r_s2d(R_S2D).eval()
    b = batch.base
    with torch.no_grad():
        o0 = base(b.events, b.ptr, b.delta_t_s)
        o1 = lc(b.events, b.ptr, b.delta_t_s, batch.hist_events, batch.hist_ptr)
        for k in ("part", "coord", "onhand"):
            assert torch.allclose(o0[k], o1[k], atol=1e-5, rtol=1e-5), k
        assert torch.equal(o0["part"].argmax(-1), o1["part"].argmax(-1))
        lc.stem[0].weight[:, 16:] = 0.05
        o2 = lc(b.events, b.ptr, b.delta_t_s, batch.hist_events, batch.hist_ptr)
    assert not torch.allclose(o0["part"], o2["part"], atol=1e-3)


def test_history_events_from_context():
    from tools.pfo import nets as N
    from tools.pfo.engine import sequences
    import yaml
    cfg = yaml.safe_load((ROOT / "configs/pfo2_20261103/eval/pfo2_fgs_s3407.yaml").read_text())
    root, seqs = sequences(cfg, ("zgz_local",))
    ctx = N.sequence(root, seqs[0][1], "zgz_local", torch.device("cpu"))
    end = int(np.asarray(ctx.aux["valid_runs_ms"]).reshape(-1, 2)[0][0]) + 50 * 9 + 49
    LC._DEPLOY["empty"] = False
    ev, ptr = LC.history_events(ctx, end, 50, 200, torch.device("cpu"))
    start = end - 49
    a0, a1 = int(ctx.offsets[start - 200]), int(ctx.offsets[start])
    assert int(ptr[-1]) == a1 - a0 == len(ev) and float(ev[:, 3].min()) >= 0 and float(ev[:, 3].max()) < 0.2
    assert np.array_equal(ev[:, 1].numpy(), np.asarray(ctx.events[a0:a1])[:, 0].astype(np.float32))


def test_jobs_and_rules():
    from tools.tpd import analyze_lc as AN
    from tools.tpd import jobs_lc as J
    jobs = J.stage()
    keys = [j["key"] for j in jobs]
    assert len(keys) == len(set(keys)) and all(set(j) <= {"key", "kind", "command", "deps", "note", "mem_gib"} for j in jobs)
    assert AN.screen_rule(0.90) and not AN.screen_rule(0.899)
    assert AN.gate_rule(10.90, 11.55) and not AN.gate_rule(10.91, 11.55)
