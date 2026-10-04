"""DT3 package C: the inference-time probes of `tools/dt/infer_probe.py` (multi-window ensemble, AdaBN, polarity TTA, seed
ensemble) against tiny CPU stand-in models and a synthetic sequence patched into `semkine.eval_track` (no checkpoint, no data).

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_dt3_probe.py -q
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as Rot
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "dt"), str(REPO / "tools" / "tracking")]
from semkine.anchored import AdaptiveFilter, FilteredTracker           # noqa: E402
import infer_probe as IP                                                # noqa: E402

EX = IP.EX
DEV = torch.device("cpu")
NODES = [[300, 0.3], [3000, 0.5], [30000, 0.8]]


def _cfg():
    return {"TRACK": {"PREV_NOISE_T": 0.005, "PREV_NOISE_R": 0.05, "PREV_NOISE_POSE": 0.05}, "DATA": {}}


def _prev(b, seed=1):
    """(b, 51) states: translation near (0, 0, 0.4), root |phi| in [0.3, 2.3] (< pi), small finger angles."""
    g = torch.Generator().manual_seed(seed)
    p = 0.3 * torch.randn(b, 51, generator=g)
    p[:, :3] = 0.05 * torch.randn(b, 3, generator=g)
    p[:, 2] += 0.4
    axis = torch.randn(b, 3, generator=g)
    p[:, 3:6] = axis / axis.norm(dim=1, keepdim=True) * (0.3 + 2.0 * torch.rand(b, 1, generator=g))
    return p


class _Stand(nn.Module):
    """`prev + f(x) w + bias` with f a POLARITY-SENSITIVE scalar of the image (channel 0 minus half channel 1), row-wise."""

    def __init__(self, seed=0, scale=0.1, symmetric=False):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("w", scale * 0.05 * torch.randn(51, generator=g))
        self.register_buffer("bias", scale * 0.01 * torch.randn(51, generator=g))
        self.symmetric = symmetric
        self.ctx = None
        self.seen = []

    def set_hand_context(self, betas, camera_K):
        self.ctx = (betas, camera_K)

    def forward(self, x, prevpos, betas=None, camera_K=None):
        self.seen.append(x.clone())
        s = (x[..., 0] + x[..., 1] if self.symmetric else x[..., 0] - 0.5 * x[..., 1]).reshape(x.shape[0], -1).mean(dim=1, keepdim=True)
        return prevpos + s * self.w + self.bias


@pytest.fixture
def synth(monkeypatch):
    """A 2-segment (0-300 ms, 320-650 ms) synthetic sequence (as tests/test_dt_filter2.py's), with an LNES whose planes differ
    in polarity and whose content depends on the window; patched into `semkine.eval_track` (so evalx's loop and this file's see it).
    `calls` records every `build_lnes` window."""
    T = 700
    rng = np.random.default_rng(0)
    per_ms = rng.integers(20, 400, size=T).astype(np.int64)
    per_ms[0:100] = 0
    offsets = np.concatenate([[0], np.cumsum(per_ms)]).astype(np.int64)
    events = np.zeros((int(offsets[-1]), 3), np.float32)
    pos = np.tile(_prev(1, seed=9).numpy(), (T, 1)).astype(np.float32)
    pos[:, :3] += np.arange(T)[:, None] * np.array([1e-4, 0, 0], np.float32)
    aux = {"betas": np.zeros(10, np.float32), "camera_K": np.eye(3, dtype=np.float32),
           "valid_runs_ms": np.array([[0, 300], [320, 650]])}
    calls = []
    monkeypatch.setattr(IP.ET, "load_sequence", lambda root, d, seq: (events, offsets, aux, pos))

    def fake_lnes(ev, off, end, window, ch):
        calls.append((int(end), int(window)))
        n = int(off[end + 1] - off[end - window + 1])
        out = np.zeros((4, 5, 2), np.float32)
        out[..., 0] = 0.3 + 1e-4 * n
        out[..., 1] = 0.1 + 1e-3 * window
        out[0, 0, 1] += 0.2
        return out
    monkeypatch.setattr(IP.ET, "build_lnes", fake_lnes)
    return {"offsets": offsets, "pos": pos, "calls": calls}


def _loop(model, win=50, fn=None, **kw):
    fn = fn or IP.run_sequence
    return fn(model, _cfg(), Path("/x"), "d", "s", DEV, np.random.default_rng(0), "model", "fixed", win, 0, 300, **kw)


def _same(r1, r2, keys=("pred", "prev", "gt", "end", "run", "elapsed", "count")):
    for k in keys:
        assert r1[k].dtype == r2[k].dtype and np.array_equal(r1[k], r2[k]), k


# ----------------------------------------------------------------------------- the loop is evalx's loop
@pytest.mark.parametrize("kind", ["bare", "filtered", "adaptive"])
def test_loop_is_evalx_run_sequence_by_default(synth, kind):
    mk = {"bare": lambda: _Stand(),
          "filtered": lambda: FilteredTracker(_Stand(), 0.5, 1.0, 0.5),
          "adaptive": lambda: AdaptiveFilter(_Stand(), NODES, 0.8, 0.5, beta_root=0.2, beta_trans=0.2)}[kind]
    for win in (50, 200):
        ref = EX.run_sequence(mk().eval(), _cfg(), Path("/x"), "d", "s", DEV, np.random.default_rng(0), "model", "fixed", win, 0, 300)
        got = _loop(mk().eval(), win)
        _same(got, ref)
        assert set(got) == set(ref) - {"win"} and len(got["fstate"]) == len(ref["fstate"])    # DT3 evalx also returns the per-step window
        for a, b in zip(got["fstate"], ref["fstate"]):
            assert all(torch.equal(a[k], b[k]) for k in a)


def test_patched_loop_swaps_in_and_restores_evalx(synth):
    orig = EX.run_sequence
    with IP.patched_loop(clip_run_start=True):
        assert EX.run_sequence is not orig
        r = EX.run_sequence(_Stand().eval(), _cfg(), Path("/x"), "d", "s", DEV, np.random.default_rng(0), "model", "fixed", 200, 0, 300)
        assert len(r["pred"]) > 0
    assert EX.run_sequence is orig


# ----------------------------------------------------------------------------- windows
def test_clip_windows_are_50_100_150_200_at_a_segment_start(synth):
    ends = list(range(320 + 49, 650, 50))
    assert [IP.eff_window(e, 320, 200, True) for e in ends[:6]] == [50, 100, 150, 200, 200, 200]
    assert [IP.eff_window(e, 320, 200, False) for e in ends[:3]] == [200, 200, 200]      # evalx: only the recording start clips
    assert [IP.eff_window(e, 0, 200, False) for e in (49, 99, 149, 199, 249)] == [50, 100, 150, 200, 200]
    assert [IP.eff_window(e, 0, 200, True) for e in (49, 99, 149, 199, 249)] == [50, 100, 150, 200, 200]
    for clip, want2 in ((True, [50, 100, 150, 200, 200, 200]), (False, [200] * 6)):         # through the loop, segment 2 (a = 320)
        synth["calls"].clear()
        r = _loop(_Stand().eval(), 200, clip_run_start=clip)
        by_seg = {}
        for e, w in synth["calls"]:
            by_seg.setdefault(int(r["run"][list(r["end"]).index(e)]), []).append(w)
        assert by_seg[0][:5] == [50, 100, 150, 200, 200]                                      # segment 1 starts at the recording start
        assert by_seg[1][:6] == want2


def test_default_window_matches_evalx_window_of(synth):
    for end in (49, 99, 149, 399, 649):
        assert IP.eff_window(end, 320, 200, False) == EX.window_of(synth["offsets"], end, "fixed", 200, 0, 300)


# ----------------------------------------------------------------------------- tangent mean / combination
def test_tangent_mean_of_identical_rotations_is_that_rotation():
    aa = torch.tensor([0.4, -1.1, 0.7], dtype=torch.float64)
    for k in (1, 2, 5):
        m, rel = IP.tangent_mean(aa.expand(k, 3).clone())
        assert torch.allclose(m, aa, atol=1e-12) and float(rel.abs().max()) < 1e-12


def test_tangent_mean_of_two_rotations_is_the_geodesic_midpoint():
    g = np.random.default_rng(3)
    for _ in range(20):
        a = Rot.from_rotvec(g.standard_normal(3) * 0.8)
        b = Rot.from_rotvec(g.standard_normal(3) * 0.8)
        mid = a * Rot.from_rotvec(0.5 * (a.inv() * b).as_rotvec())                          # the shortest-arc midpoint
        m, _ = IP.tangent_mean(torch.from_numpy(np.stack([a.as_rotvec(), b.as_rotvec()])))
        assert (Rot.from_rotvec(m.numpy()).inv() * mid).magnitude() < 1e-10
        assert abs((a.inv() * Rot.from_rotvec(m.numpy())).magnitude() - 0.5 * (a.inv() * b).magnitude()) < 1e-10   # half way along the arc
    # a +-e about the first: the mean is the first
    a = Rot.from_rotvec([0.3, 0.2, -0.5])
    e = Rot.from_rotvec([0.1, -0.2, 0.05])
    rots = [a, a * e, a * e.inv()]
    m, _ = IP.tangent_mean(torch.from_numpy(np.stack([r.as_rotvec() for r in rots])))
    assert (Rot.from_rotvec(m.numpy()).inv() * a).magnitude() < 1e-12


def test_tangent_mean_is_hemisphere_safe_near_pi():
    a = Rot.from_rotvec(np.array([np.pi - 0.05, 0, 0]))
    b = Rot.from_rotvec(np.array([-(np.pi - 0.05), 0, 0]))                                    # 0.1 rad apart, across the seam
    m, rel = IP.tangent_mean(torch.from_numpy(np.stack([a.as_rotvec(), b.as_rotvec()])))
    assert abs(float(rel[1].norm()) - 0.1) < 1e-9
    assert (Rot.from_rotvec(m.numpy()).inv() * a).magnitude() < 0.051


def test_combine_single_output_is_returned_as_is_and_mean_of_two_is_the_rule():
    o = _prev(1, seed=5)
    out, sp = IP.combine([o])
    assert out is o and sp == {"spread_root_deg": 0.0, "spread_fing": 0.0, "spread_transl_mm": 0.0}
    o2 = o + 0.01 * torch.randn(1, 51, generator=torch.Generator().manual_seed(7))
    out, sp = IP.combine([o, o2])
    z = torch.cat([o, o2]).double()
    assert torch.allclose(out[:, :3].double(), z[:, :3].mean(0, keepdim=True), atol=1e-7)
    assert torch.allclose(out[:, 6:].double(), z[:, 6:].mean(0, keepdim=True), atol=1e-7)
    mid = Rot.from_rotvec(z[0, 3:6].numpy()) * Rot.from_rotvec(0.5 * (Rot.from_rotvec(z[0, 3:6].numpy()).inv() * Rot.from_rotvec(z[1, 3:6].numpy())).as_rotvec())
    assert (Rot.from_rotvec(out[0, 3:6].double().numpy()).inv() * mid).magnitude() < 1e-6
    half = float((Rot.from_rotvec(z[0, 3:6].numpy()).inv() * Rot.from_rotvec(z[1, 3:6].numpy())).magnitude())
    assert abs(sp["spread_root_deg"] - np.rad2deg(half) / 2) < 1e-4                          # two points: RMS deviation = half the distance
    assert abs(sp["spread_transl_mm"] - 1000 * float((z[0, :3] - z[1, :3]).norm()) / 2) < 1e-6
    assert out.dtype == o.dtype and out.shape == (1, 51)


# ----------------------------------------------------------------------------- ens-win
def test_single_window_ens_equals_the_plain_loop_bit_for_bit(synth):
    for win in (50, 200):
        for clip in (False, True):
            m = _Stand().eval()
            plain = _loop(m, win, clip_run_start=clip)
            ens = IP.MultiPass([_Stand().eval()], [(0, win, False)])
            sink = {}
            got = _loop(ens, win, clip_run_start=clip, probe=ens, sink=sink)
            _same(got, plain)
            assert set(sink["s"]) == {"count", "win_ms", *IP.DIAG_KEYS}
            assert not sink["s"]["spread_root_deg"].any() and not sink["s"]["spread_fing"].any()
    # the same single-window ensemble inside an AdaptiveFilter (the filter's gates see the primary window's image)
    mk = lambda inner: AdaptiveFilter(inner, NODES, 0.8, 0.5).eval()                          # noqa: E731
    plain = _loop(mk(_Stand().eval()), 200)
    ens = IP.MultiPass([_Stand().eval()], [(0, 200, False)])
    _same(_loop(mk(ens), 200, probe=ens), plain)


def test_ens_win_runs_one_batch1_forward_per_window_with_that_window_and_combines(synth):
    wins = [100, 200, 300]
    ens = IP.MultiPass([_Stand().eval()], [(0, w, False) for w in wins])
    sink = {}
    got = _loop(ens, 200, probe=ens, sink=sink)
    st = ens.members[0]
    assert len(st.seen) == len(wins) * len(got["end"]) and all(t.shape[0] == 1 for t in st.seen)   # sequential, batch 1
    i = 6                                                                                    # one step: recompute by hand
    end, prev = int(got["end"][i]), torch.from_numpy(got["prev"][i]).view(1, -1)
    ref = _Stand().eval()
    outs = [ref(torch.from_numpy(IP.ET.build_lnes(None, synth["offsets"], end, min(w, end + 1), ("last",))).unsqueeze(0), prev) for w in wins]
    want, sp = IP.combine(outs)
    assert torch.equal(torch.from_numpy(got["pred"][i]).view(1, -1), want)
    assert abs(sink["s"]["spread_root_deg"][i] - sp["spread_root_deg"]) < 1e-5 and sink["s"]["spread_root_deg"].max() > 0
    assert (sink["s"]["win_ms"] == [IP.eff_window(int(e), 320 if r else 0, 200) for e, r in zip(got["end"], got["run"])]).all()


def test_ens_win_innovation_and_counts_are_recorded(synth):
    class _Delta(nn.Module):
        def forward(self, x, prevpos, betas=None, camera_K=None):
            out = prevpos.clone()
            out[:, 0] += 0.002                                                            # 2 mm
            out[:, 6:] += 0.01                                                            # 45 fingers x 0.01 rad
            return out
    ens = IP.MultiPass([_Delta()], [(0, None, False)])
    sink = {}
    r = _loop(ens, 50, probe=ens, sink=sink)
    d = sink["s"]
    assert np.allclose(d["innov_transl_mm"], 2.0, atol=1e-4) and np.allclose(d["innov_root_deg"], 0.0, atol=1e-4)
    assert np.allclose(d["innov_fing"], 0.01 * np.sqrt(45), atol=1e-5)
    assert np.array_equal(d["count"], r["count"]) and len(d["count"]) == len(r["end"])


def test_windowed_pass_without_step_context_is_an_error():
    ens = IP.MultiPass([_Stand().eval()], [(0, 100, False)])
    with pytest.raises(RuntimeError, match="begin_step"):
        ens(torch.zeros(1, 4, 5, 2), _prev(1))


# ----------------------------------------------------------------------------- tta-pol
def test_swap_polarity_swaps_each_pair_and_only_the_event_planes():
    x = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).view(2, 3, 4, 6)                     # 3 channel sets (last, count, first)
    s = IP.swap_polarity(x)
    for k in range(3):
        assert torch.equal(s[..., 2 * k], x[..., 2 * k + 1]) and torch.equal(s[..., 2 * k + 1], x[..., 2 * k])
    assert torch.equal(IP.swap_polarity(s), x)
    x2 = torch.rand(1, 3, 4, 2)
    assert torch.equal(IP.swap_polarity(x2), x2[..., [1, 0]])
    with pytest.raises(AssertionError):
        IP.swap_polarity(torch.zeros(1, 3, 4, 3))
    # the order of semkine.events.splat_event_image: plane k holds polarity 0 / 1 in channels 2k / 2k + 1
    xs = np.array([3, 1], np.float32), np.array([2, 2], np.float32)
    img = IP.EV.splat_event_image(xs[0], xs[1], np.array([0, 1], np.float32), np.array([5, 9], np.float32), 10, 4, 6, ("last", "count"))
    assert img[2, 3, 0] > 0 and img[2, 3, 1] == 0 and img[2, 1, 1] > 0 and img[2, 1, 0] == 0
    assert img[2, 3, 2] > 0 and img[2, 3, 3] == 0 and img[2, 1, 3] > 0 and img[2, 1, 2] == 0


def test_tta_pol_with_a_polarity_symmetric_model_equals_a_single_pass(synth):
    single = _loop(_Stand(symmetric=True).eval(), 50)
    tta = IP.MultiPass([_Stand(symmetric=True).eval()], [(0, None, False), (0, None, True)])
    sink = {}
    got = _loop(tta, 50, probe=tta, sink=sink)
    for k in ("pred", "prev"):
        assert np.abs(got[k] - single[k]).max() < 1e-5
    assert np.array_equal(got["pred"][:, :3], single["pred"][:, :3]) and np.array_equal(got["pred"][:, 6:], single["pred"][:, 6:])
    assert sink["s"]["spread_root_deg"].max() < 1e-4 and sink["s"]["spread_fing"].max() < 1e-6
    assert len(tta.members[0].seen) == 2 * len(got["end"])
    a, b = tta.members[0].seen[0], tta.members[0].seen[1]                                    # plain, then swapped, same step
    assert torch.equal(b, a[..., [1, 0]]) and not torch.equal(a, b)


def test_tta_pol_with_a_polarity_sensitive_model_is_the_mean_of_both_passes(synth):
    tta = IP.MultiPass([_Stand().eval()], [(0, None, False), (0, None, True)])
    x = torch.rand(1, 4, 5, 2, generator=torch.Generator().manual_seed(4))
    p = _prev(1, seed=3)
    m = _Stand().eval()
    want, _ = IP.combine([m(x, p), m(x[..., [1, 0]], p)])
    got = tta(x, p)
    assert torch.equal(got, want) and not torch.allclose(got, m(x, p), atol=1e-7)


# ----------------------------------------------------------------------------- seed-ens
def test_seed_ens_is_the_combination_of_the_members_with_one_prev(synth):
    ms = [_Stand(seed=s, scale=0.2).eval() for s in (1, 2, 3)]
    ens = IP.MultiPass(ms, [(i, None, False) for i in range(3)])
    x = torch.rand(1, 4, 5, 2, generator=torch.Generator().manual_seed(4))
    p = _prev(1, seed=3)
    want, _ = IP.combine([m(x, p) for m in ms])
    assert torch.equal(ens(x, p), want)
    ens.set_hand_context(torch.zeros(1, 10), torch.eye(3).view(1, 3, 3))
    assert all(m.ctx is not None for m in ms)
    _loop(ens, 50, probe=ens)                                                                 # runs the closed loop end to end


# ----------------------------------------------------------------------------- AdaBN
class _BNStand(nn.Module):
    """x (B, H, W, 2) -> BatchNorm2d(2) -> per-row mean -> `prev + bn * w`; one non-BN buffer to check nothing else moves."""

    def __init__(self, c=2):
        super().__init__()
        self.bn = nn.BatchNorm2d(c)
        self.register_buffer("w", 0.01 * torch.ones(51))
        self.ctx = []

    def set_hand_context(self, betas, camera_K):
        self.ctx.append((betas.clone(), camera_K.clone()))

    def forward(self, x, prevpos, betas=None, camera_K=None):
        y = self.bn(x.permute(0, 3, 1, 2))
        return prevpos + y.mean(dim=(1, 2, 3))[:, None] * self.w


def _record(n, seed=0, c=2, hw=(4, 5)):
    g = np.random.default_rng(seed)
    return {"s": {"x": [g.random((*hw, c)).astype(np.float32) * (1 + k % 3) for k in range(n)],
                  "prev": [g.random(51).astype(np.float32) for _ in range(n)],
                  "betas": torch.zeros(1, 10), "K": torch.eye(3).view(1, 3, 3)}}


def test_adabn_stats_become_the_data_mean_and_unbiased_var():
    m = _BNStand().eval()
    m.bn.running_mean.fill_(5.0)
    m.bn.running_var.fill_(9.0)
    rec = _record(24)
    IP.adapt_bn(m, rec, bs=24, seed=0, device=DEV)                                            # one batch holding every step
    X = torch.from_numpy(np.stack(rec["s"]["x"])).permute(0, 3, 1, 2)
    assert torch.allclose(m.bn.running_mean, X.mean(dim=(0, 2, 3)), atol=1e-6)
    assert torch.allclose(m.bn.running_var, X.var(dim=(0, 2, 3), unbiased=True), atol=1e-6)
    assert not m.training and not m.bn.training and m.bn.momentum == 0.1                      # eval again, momentum restored
    assert torch.equal(m.w, 0.01 * torch.ones(51))
    assert len(m.ctx) == 1                                                                    # betas / K set once for the sequence


def test_adabn_with_several_batches_averages_the_batch_statistics_and_covers_every_step():
    m = _BNStand().eval()
    rec = _record(30)
    nb = IP.adapt_bn(m, rec, bs=10, seed=1, device=DEV)
    assert nb == 3 and int(m.bn.num_batches_tracked) == 3
    X = torch.from_numpy(np.stack(rec["s"]["x"])).permute(0, 3, 1, 2)
    assert torch.allclose(m.bn.running_mean, X.mean(dim=(0, 2, 3)), atol=1e-6)               # equal batches: mean of the batch means
    m2 = _BNStand().eval()
    IP.adapt_bn(m2, rec, bs=10, seed=1, device=DEV)
    assert torch.equal(m.bn.running_var, m2.bn.running_var)                                   # seeded shuffles reproduce
    nb7 = IP.adapt_bn(_BNStand().eval(), _record(15), bs=7, seed=0, device=DEV)
    assert nb7 == 2                                                                           # 7 + 7, the batch of 1 is skipped


def test_adabn_batches_stay_within_their_sequence_context():
    m = _BNStand().eval()
    rec = _record(8)
    rec["t"] = dict(_record(8, seed=1)["s"], betas=torch.ones(1, 10))
    IP.adapt_bn(m, rec, bs=4, seed=0, device=DEV)
    assert [float(b.sum()) for b, _ in m.ctx] == [0.0, 10.0]                                  # one set_hand_context per sequence, in order


def test_adabn_iters_zero_is_a_no_op_and_the_closed_loop_is_unchanged(synth):
    m = _BNStand().eval()
    m.bn.running_mean.fill_(0.3)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    args = argparse.Namespace(iters=0, bs=8, seed=0, window_ms=50, clip_run_start=False)
    before = _loop(m, 50)
    info, ra, orig = IP.run_adabn(m, m, _cfg(), Path("/x"), [("s", "d")], DEV, args)
    assert info == [] and ra == []
    assert all(torch.equal(sd[k], v) for k, v in m.state_dict().items()) and not m.bn.training
    _same(_loop(m, 50), before)


def test_adabn_rounds_record_the_closed_loop_and_adapt(synth):
    m = _BNStand().eval()
    args = argparse.Namespace(iters=2, bs=16, seed=0, window_ms=200, clip_run_start=True)
    stats0 = m.bn.running_var.clone()
    info, ra, orig = IP.run_adabn(m, m, _cfg(), Path("/x"), [("s", "d")], DEV, args)
    r = _loop(m, 200, clip_run_start=True)
    assert len(info) == 2 and ra == [None, None]
    assert info[0]["recorded_steps"] == {"s": len(r["end"])} and info[0]["n_batches"] >= 1
    assert info[0]["shift_vs_original"]["mean_d_mean_abs"] > 0 and not torch.equal(m.bn.running_var, stats0)
    assert len(orig) == 1 and not m.training
    # the recorder sees the same (x, prev) the loop feeds: recompute the first step's image
    rec = {}
    _loop(m, 200, clip_run_start=True, record=rec)
    assert len(rec["s"]["x"]) == len(r["end"]) and rec["s"]["x"][0].shape == (4, 5, 2)
    assert np.array_equal(np.stack(rec["s"]["prev"]), _loop(m, 200, clip_run_start=True)["prev"])


def test_adabn_leaves_a_model_without_batchnorm_alone(synth):
    m = _Stand().eval()
    args = argparse.Namespace(iters=1, bs=8, seed=0, window_ms=50, clip_run_start=False)
    before = _loop(m, 50)
    info, _, orig = IP.run_adabn(m, m, _cfg(), Path("/x"), [("s", "d")], DEV, args)
    assert orig == {} and info[0]["shift_vs_original"]["layers"] == []
    _same(_loop(m, 50), before)


# ----------------------------------------------------------------------------- statistics / io
def test_spearman_report_buckets_and_degenerate_cases():
    n = 90
    cnt = np.concatenate([np.full(30, 100), np.full(30, 1000), np.full(30, 5000)])
    err = np.arange(n, dtype=np.float32)
    arrays = {"model|a|count": cnt, "model|a|mpjpe_ra_mm": err, "probe|a|spread_root_deg": err * 2.0,
              "probe|a|spread_fing": np.zeros(n, np.float32)}
    rep = IP.spearman_report(arrays, [("a", "d")])
    assert rep["a"]["[0,500)"]["spread_root_deg"] == pytest.approx(1.0) and rep["a"]["[0,500)"]["n"] == 30
    assert rep["a"]["[2000,inf)"]["n"] == 30 and rep["a"]["all"]["n"] == n
    assert rep["a"]["[500,2000)"]["spread_fing"] is None                                      # a constant spread has no rank
    arrays["probe|a|spread_root_deg"] = -err
    assert IP.spearman_report(arrays, [("a", "d")])["all"]["all"]["spread_root_deg"] == pytest.approx(-1.0)
    arrays["model|a|count"] = np.concatenate([np.full(85, 100), np.full(5, 5000)])
    assert IP.spearman_report(arrays, [("a", "d")])["a"]["[2000,inf)"]["spread_root_deg"] is None   # fewer than 20 steps


def test_filter_tag_hash_is_filter_evals():
    spec = {"a_root": [[300.0, 0.3], [3000.0, 0.5], [30000.0, 0.8]], "a_rest": 0.8, "a_trans": 0.5, "beta_root": 0.0,
            "beta_trans": 0.0, "decay": 0.5, "host": False}
    canon = IP.load_one_spec(json.dumps(spec))
    assert canon == AdaptiveFilter.canonical_spec(spec)
    assert IP.spec_hash_tag(canon) == "afad1f21"                                              # the tag of the recorded dt2 wscan files
    assert IP.spec_hash_tag(canon) == "af" + hashlib.md5(json.dumps(canon, sort_keys=True).encode()).hexdigest()[:6]
    with pytest.raises(SystemExit):
        IP.load_one_spec(json.dumps([spec, spec]))


def test_cmp_reports_bit_identity(tmp_path, capsys):
    a = {"model|s|pred": np.arange(6, dtype=np.float32), "model|s|end": np.arange(3), "probe|s|x": np.zeros(2, np.float32)}
    b = dict(a)
    np.savez(tmp_path / "a.npz", **a)
    np.savez(tmp_path / "b.npz", **b)
    ns = argparse.Namespace(a=str(tmp_path / "a.npz"), b=str(tmp_path / "b.npz"), prefix="model|", json_a="", json_b="")
    with pytest.raises(SystemExit) as e:
        IP.cmd_cmp(ns)
    assert e.value.code == 0 and "2 arrays compared, 2 bit-identical" in capsys.readouterr().out
    b["model|s|pred"] = b["model|s|pred"].copy()
    b["model|s|pred"][3] += np.float32(1e-6)
    np.savez(tmp_path / "b.npz", **b)
    with pytest.raises(SystemExit) as e:
        IP.cmd_cmp(ns)
    assert e.value.code == 1
