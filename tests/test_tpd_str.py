"""TPD stage STR, CPU tests (semkine/tpd_str.py, tools/tpd/ns_str.py). Run with CUDA_VISIBLE_DEVICES="" in a process of
its own (the namespace patches the registries; routing / trunk / geometry hooks are module patches).

- The rule: the rate equals a brute-force count over the stream; kept events satisfy c % k == 0, no cap at evaluation.
- Prefix: replacing the events after position m (another count, later times) leaves the kept events before m and their
  tokens bit for bit; partition: `Stream` fed in random chunks over two update intervals equals the batched form.
- Data: the STR rollout samples equal the final configuration's (same draws) and add packet 0's history (without
  augmentation: the raw 50 ms before it) and the stream counters; plain packets carry the engine's history.
- Model: the STR model's nodes are `stream_nodes`; a rollout batch trains (gradient reaches every weight).
- Registry: the config is rolk's plus the STR block and names.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import copy  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd import ns_str  # noqa: E402

ns_str.enter()
from semkine import tpd_final as TF  # noqa: E402
from semkine import tpd_rol as R  # noqa: E402
from semkine import tpd_str as TS  # noqa: E402
from semkine.events import EV_T  # noqa: E402
from tools.tpd import arms_final as A  # noqa: E402

H, W = 180, 240


def _stream(gen, n_hist, n, ms_quant=0.7):
    """(N, 5) events of one sample, times sorted, many equal (ms-quantized), history times in [-0.05, 0)."""
    def part(m, lo, hi):
        t = np.sort(gen.uniform(lo, hi, m))
        q = gen.random(m) < ms_quant
        t[q] = np.floor(t[q] * 1000) / 1000
        t = np.maximum.accumulate(np.sort(t))
        ev = np.zeros((m, 5), np.float32)
        ev[:, 1], ev[:, 2] = gen.integers(0, W, m), gen.integers(0, H, m)
        ev[:, 3] = np.round(t * 1e6).astype(np.int64).astype(np.float32) * 1e-6
        ev[:, 4] = gen.integers(0, 2, m)
        return ev
    return part(n_hist, -0.05, 0.0), part(n, 0.0, 0.05)


def _batch(gen, sizes):
    hs, ps = zip(*[_stream(gen, a, b) for a, b in sizes])
    def cat(parts):
        ptr = np.zeros(len(parts) + 1, np.int64)
        ptr[1:] = np.cumsum([len(p) for p in parts])
        ev = np.concatenate(parts, 0)
        ev[:, 0] = np.repeat(np.arange(len(parts)), [len(p) for p in parts])
        return torch.from_numpy(ev), torch.from_numpy(ptr)
    return (*cat(ps), *cat(hs))


def test_rate_and_rule_match_brute_force():
    gen = np.random.default_rng(0)
    ev, ptr, hist, hp = _batch(gen, [(3000, 5000), (0, 700), (6000, 9000)])
    c0 = torch.tensor([11, 0, 5])
    src, mask, info = TS.stream_nodes(ev, ptr, hist, hp, c0, budget=2048)
    for b in range(3):
        t_all = np.concatenate([hist[hp[b]:hp[b + 1], EV_T].numpy(), ev[ptr[b]:ptr[b + 1], EV_T].numpy()])
        us = np.round(t_all.astype(np.float64) * 1e6).astype(np.int64)
        nh = int(hp[b + 1] - hp[b])
        keep = []
        for i in range(int(ptr[b + 1] - ptr[b])):
            g = nh + i
            n_i = int(np.sum(us[:g] >= us[g] - 50000))
            k = max(1, -(-n_i // 2048))
            if (int(c0[b]) + i) % k == 0:
                keep.append(int(ptr[b]) + i)
        assert src[b][mask[b]].tolist() == keep
    assert info["cap_hits"] == 0 and info["nodes_max"] == int(mask.sum(1).max())
    _, m2, i2 = TS.stream_nodes(ev, ptr, hist, hp, c0, budget=2048, cap=100)
    assert i2["cap_hits"] == 3 and int(m2.sum(1).max()) == 100


def test_prefix_stability():
    gen = np.random.default_rng(1)
    ev, ptr, hist, hp = _batch(gen, [(4000, 7000), (500, 900)])
    c0 = torch.tensor([3, 8])
    dts = torch.tensor([0.05, 0.05])
    src, mask, _ = TS.stream_nodes(ev, ptr, hist, hp, c0)
    tok = TS.stream_tokens(ev, ptr, dts, H, W, hist, hp)
    for frac in (0.25, 0.5, 0.75):
        m = int(ptr[0]) + int(frac * int(ptr[1] - ptr[0]))
        t_m = float(ev[m - 1, EV_T])
        n_new = int(gen.integers(10, 5000))
        new = np.zeros((n_new, 5), np.float32)
        new[:, 1], new[:, 2], new[:, 4] = gen.integers(0, W, n_new), gen.integers(0, H, n_new), gen.integers(0, 2, n_new)
        new[:, 3] = np.sort(gen.uniform(t_m, 0.05, n_new)).astype(np.float32)
        ev2 = torch.cat([ev[:m], torch.from_numpy(new), ev[ptr[1]:]], 0)
        ptr2 = torch.tensor([0, m + n_new, m + n_new + int(ptr[2] - ptr[1])])
        src2, mask2, _ = TS.stream_nodes(ev2, ptr2, hist, hp, c0)
        tok2 = TS.stream_tokens(ev2, ptr2, dts, H, W, hist, hp)
        before = src[0][mask[0]]
        before = before[before < m]
        after = src2[0][mask2[0]]
        assert torch.equal(after[after < m], before)
        assert torch.equal(tok2[before], tok[before])


def test_stream_chunks_equal_the_batched_form():
    gen = np.random.default_rng(2)
    h0, p0 = _stream(gen, 3000, 5000)
    _, p1 = _stream(gen, 0, 7000)
    dts = torch.tensor([0.05])
    s = TS.Stream(H, W, counter=17, history=torch.from_numpy(h0))
    outs = []
    for pk in (p0, p1):
        cuts = np.sort(gen.choice(np.arange(1, len(pk)), 40, replace=False))
        for chunk in np.split(pk, cuts):
            s.ingest(torch.from_numpy(chunk))
        outs.append(s.update())
    hist1 = np.concatenate([h0, p0], 0)
    hist1[:, EV_T] -= 0.05
    hist1 = hist1[hist1[:, EV_T] >= -0.05]
    for (kept, tok), pk, hist, c0 in ((outs[0], p0, h0, 17), (outs[1], p1, hist1, 17 + len(p0))):
        ev, hv = torch.from_numpy(pk), torch.from_numpy(hist)
        ptr, hp = torch.tensor([0, len(pk)]), torch.tensor([0, len(hist)])
        src, mask, _ = TS.stream_nodes(ev, ptr, hv, hp, torch.tensor([c0]))
        ref = TS.stream_tokens(ev, ptr, dts, H, W, hv, hp)
        assert torch.equal(kept, src[0][mask[0]]) and torch.equal(tok, ref[kept])


@pytest.fixture(scope="module")
def base():
    from semkine.dataset import build_dataset
    cfg = A.build_config("str", 3407, "formal")
    comps = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    return cfg, comps, build_dataset


def test_rollout_samples(base):
    cfg, comps, build = base
    fin = TS._FINAL["upgrade_rollout"](build(cfg, "train", comps, train=True), 5)
    st = TS.upgrade_rollout(build(cfg, "train", comps, train=True), 5)
    assert type(st) is TS.StrRolloutDataset and type(fin) is TF.FinalRolloutDataset
    for idx in (5, 123456, 2000000):
        a, b = fin[idx], st[idx]
        for i in range(5):
            assert np.array_equal(a.packets[i].events, b.packets[i].events)
            assert np.array_equal(a.packets[i].prev_state, b.packets[i].prev_state)
            assert a.packets[i].meta["n_prev"] == b.packets[i].meta["n_prev"]
        h0 = b.packets[0].meta["str_hist"]
        assert h0.ndim == 2 and h0.shape[1] == 4 and (len(h0) == 0 or (h0[:, 2] < 0).all() and (h0[:, 2] >= -0.05).all())
        c = [p.meta["str_c0"] for p in b.packets]
        assert c[0] == len(h0) and all(c[i] == c[i - 1] + b.packets[i - 1].n_events for i in range(1, 5))
        assert all("str_hist" not in p.meta for p in b.packets[1:])
    plain = copy.deepcopy(cfg)
    plain["AUG"]["DOMRAND"]["ENABLED"] = False
    plain["AUG"]["POLARITY_FLIP"] = False
    plain["AUG"]["PIXEL_POLARITY_SWAP"] = False
    fz = TS.upgrade_rollout(build(plain, "train", comps, train=True), 2)
    item = fz[777]
    si, ends, _, _ = fz.ends(777)
    h = fz._handle(si)
    ref = TS.history_events(h.events, h.offsets, h.tsub, ends[0] - 49)
    assert np.array_equal(item.packets[0].meta["str_hist"], ref[:, 1:])
    batch = TS.collate_str([st[i] for i in (1, 2)])
    assert isinstance(batch, R.RolloutBatch) and batch.packets[0].hist_ptr.shape == (3,)
    assert batch.packets[1].hist_events is None and batch.packets[3].c0.shape == (2,)
    TS.link_history(batch.packets)
    p0, p1 = batch.packets[0], batch.packets[1]
    assert torch.equal(p1.hist_ptr, p0.ptr)
    assert torch.allclose(p1.hist_events[:, EV_T], p0.events[:, EV_T] - 0.05)
    moved = batch.to("cpu")
    assert torch.equal(moved.packets[0].hist_events, batch.packets[0].hist_events)


def test_plain_samples_carry_the_engine_history(base):
    cfg, comps, build = base
    ev = TS.upgrade_plain(build(cfg, "val_core", comps, train=False))
    p = ev[1000]
    h = ev._handle(int(ev.index[1000][0]))
    s = p.meta["end_idx"] - 49
    ref = TS.history_events(h.events, h.offsets, h.tsub, s)
    assert np.array_equal(p.meta["str_hist"], ref[:, 1:]) and p.meta["str_c0"] == int(h.offsets[s])
    bb = TS.collate_str([p, ev[1001]])
    assert isinstance(bb, TS.StrPacketBatch) and bb.hist_ptr.tolist()[-1] == bb.hist_events.shape[0]


@pytest.fixture(scope="module")
def model_and_batch(base):
    from semkine import pfo_router as PR
    from semkine import pfo_trunk4 as T4
    from semkine import s45sm_fast as FAST
    from semkine import xag as XAG
    cfg, comps, build = base
    p = cfg["PFO"]
    PR.register()
    FAST.install(p["ROUTER_KIND"], p["ROUTER_CHECKPOINT"], p["ASSOC"], float(p["ONHAND_TAU"]))
    T4.install("gshare")
    R.install_grad_geometry()
    try:
        assert XAG.sample_sources is FAST.predicted_sources
        st = TS.upgrade_rollout(build(cfg, "train", comps, train=True), 5)
        yield cfg, TS.collate_str([st[i] for i in (11, 222222, 333333)])
    finally:
        R.uninstall_grad_geometry()
        T4.uninstall()
        FAST.uninstall()
        PR.unregister()


def test_model_nodes_and_training(model_and_batch, tmp_path):
    cfg, rb = model_and_batch
    c = copy.deepcopy(cfg)
    c["TRAIN"]["OUTPUT_DIR"] = str(tmp_path)
    torch.manual_seed(0)
    m = TF.final_model_class()(c)
    assert type(m).__name__ == "STRModel" and m.LR_FLOOR == 0.0
    m.init_from_abs_checkpoint(c["MODEL"]["INIT_FROM"])
    TS.link_history(rb.packets)
    pk = rb.packets[2]
    m.eval()
    src, mask = m._nodes(pk)
    ref, rmask, _ = TS.stream_nodes(pk.events, pk.ptr, pk.hist_events, pk.hist_ptr, pk.c0)
    assert torch.equal(src, ref) and torch.equal(mask, rmask)
    m.train()
    m.log = lambda *a, **k: None
    m.zero_grad(set_to_none=True)
    out = m.training_step(rb, 0)
    assert out.requires_grad
    out.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    rec = __import__("json").loads((tmp_path / "final_start_check_rank0.json").read_text())
    assert rec["stream"] and rec["causal"] and rec["init_equal"] and rec["nodes_mean"] > 0


def test_config_is_rolk_plus_the_str_block():
    for seed in (3407, 3408, 3409):
        for phase in ("preflight", "formal"):
            mine, rolk = A.build_config("str", seed, phase), A.build_config("rolk", seed, phase)
            assert mine["STR"] == {"BUDGET": 2048, "WINDOW_MS": 50, "TRAIN_CAP": 6144}
            a, b = copy.deepcopy(mine), copy.deepcopy(rolk)
            a.pop("STR")
            for x in (a, b):
                x["TRAIN"].pop("OUTPUT_DIR"); x["TRAIN"].pop("RUN_NAME"); x["EVAL"].pop("OUTPUT_DIR")
                x["PFO"].pop("ARM"); x["PFO"].pop("SECTION")
            assert a == b
    assert A.config_path("str", 3407).parent.name == "str" and A.run_name("str", 3408) == "tpd_str_s3408"
