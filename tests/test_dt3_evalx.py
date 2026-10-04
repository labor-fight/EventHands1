"""DT3 package A: the evidence-window options of `tools/tracking/evalx.py` (segment-clipped windows, the rate-normalized
event count, a quantized window set), their passthrough into `tools/dt/filter_eval.py` / `tools/dt/filter_sweep_2fold.py`,
and -- the point of the iron rule -- that with every option off nothing changed (window function, loop results, npz keys,
file names, json, old-signature callers).

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python -m pytest tests/test_dt3_evalx.py -q
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "dt"), str(REPO / "tools" / "tracking"), str(REPO / "tests")]
import filter_eval as FE                                                                  # noqa: E402
import filter_sweep_2fold as SW                                                           # noqa: E402
from semkine.anchored import AdaptiveFilter, FilteredTracker                              # noqa: E402
from test_dt_filter2 import NODES, _cfg, _Spy, _Stand, synth                              # noqa: E402,F401  (synth: a fixture)

EX = FE.EX
DEV = torch.device("cpu")
STEP = EX.STEP


# ------------------------------------------------------------------------------------- the old implementations, verbatim
def old_window_of(offsets, end, wmode, win, min_events, max_win):
    """`evalx.window_of` as it was before DT3."""
    if wmode == "fixed":
        return int(min(win, end + 1))
    w = int(min(win, end + 1))
    while w < max_win and w < end + 1 and offsets[end + 1] - offsets[end - w + 1] < min_events:
        w = min(w + 10, max_win, end + 1)
    return int(w)


def old_tag(a):
    """`evalx.cmd_eval`'s file-stem construction as it was before DT3."""
    tag = f"evalx_{a.split}_{a.ckpt.replace('=', '')}"
    tag += "_tf" if a.tf else ""
    tag += "_pert" if a.perturb else ""
    if a.window_mode != "fixed" or a.window_ms != STEP:
        tag += f"_w{a.window_mode}{a.window_ms}" + (f"_n{a.min_events}_x{a.max_window_ms}" if a.window_mode == "adaptive" else "")
    tag += f"_{a.suffix}" if a.suffix else ""
    return tag


def _rand_offsets(seed, T=3000, dead=True):
    rng = np.random.default_rng(seed)
    per_ms = rng.integers(0, 60, size=T).astype(np.int64)
    if dead:                                                # empty stretches, so the adaptive loop has something to grow over
        for s0 in rng.integers(0, T - 200, size=6):
            per_ms[s0:s0 + int(rng.integers(20, 180))] = 0
    return np.concatenate([[0], np.cumsum(per_ms)]).astype(np.int64)


def _uniform_offsets(per_ms, T=1200):
    return np.concatenate([[0], np.cumsum(np.full(T, per_ms, dtype=np.int64))]).astype(np.int64)


# ---------------------------------------------------------------------------------------------------------- window_of
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_window_of_default_is_the_old_function_on_random_offsets(seed):
    off = _rand_offsets(seed)
    rng = np.random.default_rng(100 + seed)
    n = 0
    for _ in range(3000):
        end = int(rng.integers(0, len(off) - 2))                    # includes ends closer to the start than the window
        win = int(rng.integers(1, 320))
        min_events = int(rng.integers(0, 6000))
        max_win = int(rng.integers(20, 400))
        for wmode in ("fixed", "adaptive"):
            ref = old_window_of(off, end, wmode, win, min_events, max_win)
            assert EX.window_of(off, end, wmode, win, min_events, max_win) == ref
            assert EX.window_of(off, end, wmode, win, min_events, max_win, floor=0, wset=()) == ref
            assert type(EX.window_of(off, end, wmode, win, min_events, max_win)) is int
            n += 1
    assert n == 6000


@pytest.mark.parametrize("wmode", ["fixed", "adaptive"])
def test_window_of_floor_is_the_old_function_on_the_offsets_cut_at_the_floor(wmode):
    """A window clipped at `floor` is exactly the old window on the recording that starts at `floor`."""
    off = _rand_offsets(7)
    rng = np.random.default_rng(8)
    for _ in range(2000):
        floor = int(rng.integers(0, 2500))
        end = int(rng.integers(floor + 49, min(floor + 700, len(off) - 2)))
        win, min_events, max_win = int(rng.integers(20, 320)), int(rng.integers(0, 6000)), int(rng.integers(20, 400))
        got = EX.window_of(off, end, wmode, win, min_events, max_win, floor=floor)
        assert got == old_window_of(off[floor:], end - floor, wmode, win, min_events, max_win)
        assert end - got + 1 >= floor                              # never reaches before the floor
        assert got <= end + 1 - floor


def test_window_set_picks_the_smallest_window_with_enough_events_else_the_largest():
    off = _uniform_offsets(10)                                      # 10 events per ms: m ms hold 10 m
    ws = (300, 150, 200)                                            # unsorted on purpose
    wo = lambda me, end=900, **kw: EX.window_of(off, end, "fixed", 50, me, 300, wset=ws, **kw)   # noqa: E731
    assert wo(1) == 150 and wo(1500) == 150                         # 150 ms hold exactly 1500
    assert wo(1501) == 200 and wo(2000) == 200
    assert wo(2001) == 300 and wo(3000) == 300
    assert wo(3001) == 300 and wo(10 ** 9) == 300                   # nobody reaches it: the largest member
    # the set replaces the window mode, win and max_win
    for wmode, win, mw in (("adaptive", 50, 60), ("fixed", 999, 10), ("adaptive", 1, 1)):
        assert EX.window_of(off, 900, wmode, win, 1800, mw, wset=ws) == 200
    # recent silence: only the older events count; the window grows over it
    gap = np.concatenate([[0], np.cumsum(np.where(np.arange(1200) >= 1100, 0, 10))]).astype(np.int64)   # the last 100 ms empty
    assert EX.window_of(gap, 1099 + 100, "fixed", 50, 1, 300, wset=ws) == 150      # 50 ms of events inside 150 ms: 500 >= 1
    assert EX.window_of(gap, 1099 + 100, "fixed", 50, 600, 300, wset=ws) == 200    # 150: 500, 200: 1000
    assert EX.window_of(gap, 1099 + 100, "fixed", 50, 1500, 300, wset=ws) == 300   # 200: 1000, 300: 2000


def test_window_set_is_clipped_to_the_available_past():
    off = _uniform_offsets(10)
    ws = (150, 200, 300)
    # 120 ms of past (end 599, floor 480): the counts are taken over the clipped windows, the answer is clipped to 120
    assert EX.window_of(off, 599, "fixed", 50, 1000, 300, floor=480, wset=ws) == 120       # 150 -> 120 ms = 1200 >= 1000
    assert EX.window_of(off, 599, "fixed", 50, 1300, 300, floor=480, wset=ws) == 120       # none reaches: largest, clipped
    # a clipped smaller member can reach the count only because of the clipping of the larger ones: never exceed the past
    for end in range(49, 700, 37):
        for floor in (0, 10, 400):
            if end + 1 - floor < 50:
                continue
            for me in (1, 800, 2500, 10 ** 6):
                w = EX.window_of(off, end, "fixed", 50, me, 300, floor=floor, wset=ws)
                assert 50 <= w <= min(300, end + 1 - floor) and w in set(ws) | {end + 1 - floor}


def test_parse_window_set_and_option_checks():
    assert EX.parse_window_set("") == () and EX.parse_window_set(None) == () and EX.parse_window_set(()) == ()
    assert EX.parse_window_set("300,150, 200,150") == (150, 200, 300) and EX.parse_window_set([200, 100]) == (100, 200)
    for bad in ("0,100", "-5", "abc"):
        with pytest.raises(ValueError):
            EX.parse_window_set(bad)
    ns = lambda **k: argparse.Namespace(**{"window_mode": "fixed", "window_ms": STEP, "min_events": 0, **k})   # noqa: E731
    EX.check_window_opts(ns())
    EX.check_window_opts(ns(window_set=(150, 200), min_events=5000, clip_run_start=True, count_mode="norm"))
    for bad in (ns(count_mode="mean"), ns(window_set=(150, 200)), ns(window_set=(150,), min_events=10, window_ms=200),
                ns(window_set=(150,), min_events=10, window_mode="adaptive")):
        with pytest.raises(ValueError):
            EX.check_window_opts(bad)


# ----------------------------------------------------------------------------------------------- run_sequence (synthetic)
@pytest.fixture
def wlog(synth, monkeypatch):
    """The (end, window) of every LNES the evaluator builds (the synthetic image function stays in place)."""
    log = []
    inner = EX.ET.build_lnes

    def rec(ev, off, end, window, ch):
        log.append((int(end), int(window)))
        return inner(ev, off, end, window, ch)
    monkeypatch.setattr(EX.ET, "build_lnes", rec)
    return log


def _run(model, *a, **kw):
    return EX.run_sequence(model, _cfg(), Path("/x"), "d", "s", DEV, np.random.default_rng(0), *a, **kw)


def _spy(**kw):
    return _Spy(_Stand(scale=0.1), NODES, 1.0, 0.5, **kw).eval()


def test_default_result_has_the_windows_and_nothing_else_new(synth, wlog):
    r = _run(_spy(beta_root=0.3, beta_trans=0.3))
    assert set(r) == {"pred", "gt", "end", "run", "elapsed", "count", "betas", "prev", "fstate", "win"}
    assert r["win"].dtype == np.int64 and (r["win"] == STEP).all() and len(r["win"]) == len(r["end"]) == 12
    assert [w for _, w in wlog] == [STEP] * 12


def test_clip_run_start_clips_the_windows_to_the_segment(synth, wlog):
    """Segments [0, 300) and [320, 650): the second one's windows grow 50, 100, 150, 200 (not 200 from the start)."""
    r0 = _run(_spy(), "model", "fixed", 200)                                    # default: clipped to the recording only
    assert list(r0["win"]) == [50, 100, 150, 200, 200, 200] + [200] * 6
    wlog.clear()
    r1 = _run(_spy(), "model", "fixed", 200, 0, 300, True)
    assert list(r1["win"]) == [50, 100, 150, 200, 200, 200] + [50, 100, 150, 200, 200, 200]
    assert [w for _, w in wlog] == list(r1["win"])                             # these are the windows the LNES was built over
    starts = {0: 0, 1: 320}
    assert all(int(e) - int(w) + 1 >= starts[int(rr)] for e, w, rr in zip(r1["end"], r1["win"], r1["run"]))
    assert (r0["end"] - r0["win"] + 1 < 320)[6:9].all()                         # the default one does reach over the gap
    assert np.array_equal(r0["count"], r1["count"])                            # the 50 ms count is the protocol's either way
    # the adaptive mode is bounded by the segment as well
    ra = _run(_spy(), "model", "adaptive", 50, 10 ** 9, 300, True)             # never enough events: grows to the cap / the past
    assert list(ra["win"]) == [50, 100, 150, 200, 250, 300] + [50, 100, 150, 200, 250, 300]
    ra0 = _run(_spy(), "model", "adaptive", 50, 10 ** 9, 300)
    assert list(ra0["win"]) == [50, 100, 150, 200, 250, 300] + [300] * 6


@pytest.mark.parametrize("mode", ["model", "hold", "noevents", "tf"])
@pytest.mark.parametrize("make", ["spy", "filtered", "bare"])
def test_clip_at_50_ms_is_the_default_loop_bit_for_bit(synth, mode, make):
    def mk():
        return {"spy": lambda: _spy(beta_root=0.3, beta_trans=0.3),
                "filtered": lambda: FilteredTracker(_Stand(scale=0.1), 0.5, 1.0, 0.5).eval(),
                "bare": lambda: _Stand(scale=0.1).eval()}[make]()
    a, b = _run(mk(), mode), _run(mk(), mode, "fixed", STEP, 0, 300, True)
    for k in ("pred", "gt", "end", "run", "elapsed", "count", "prev", "win"):
        assert np.array_equal(a[k], b[k]), k
        assert a[k].dtype == b[k].dtype
    assert len(a["fstate"]) == len(b["fstate"])


def test_count_mode_norm_is_the_window_count_at_a_50_ms_rate(synth, wlog):
    off = synth["offsets"]
    base = _spy()
    r_last = _run(base, "model", "fixed", 200, 0, 300, True)
    assert all(isinstance(n, int) for n in base.log["n"]) and "count_f" not in r_last
    spy = _spy()
    r = _run(spy, "model", "fixed", 200, 0, 300, True, "norm")
    want = [float(int(off[e + 1] - off[e - w + 1])) * (STEP / w) for e, w in zip(r["end"], r["win"])]
    assert spy.log["n"] == want and all(type(n) is float for n in spy.log["n"])
    assert r["count_f"].dtype == np.float64 and list(r["count_f"]) == want
    assert np.array_equal(r["count"], r_last["count"])                        # r["count"] stays the 50 ms packet's count
    assert [int(off[e + 1] - off[e - 49]) for e in r["end"]] == list(r["count"])
    # a device float64 computation of the same product is bit-equal -- with a true tensor / tensor division. (Not `50.0 / w_tensor`:
    # torch evaluates a python-scalar / tensor as `w.reciprocal() * 50.0`, which differs from `50 / w` in the last bit for ~1/3 of the
    # (n, w) pairs; with a python-int `w`, `50.0 / w` is exact.)
    n_win = torch.tensor([int(off[e + 1] - off[e - w + 1]) for e, w in zip(r["end"], r["win"])])
    got = (n_win.to(torch.float64) * (torch.tensor(50.0, dtype=torch.float64) / torch.tensor(r["win"]).to(torch.float64))).numpy()
    assert np.array_equal(got, r["count_f"])
    # the windows are 50 ms long only at a segment's first steps: there norm == last50
    first = r["win"] == STEP
    assert first.sum() == 2 and np.array_equal(r["count_f"][first], r["count"][first].astype(np.float64))
    assert not np.array_equal(r["count_f"], r["count"].astype(np.float64))
    # noevents still hands 0
    spy0 = _spy()
    _run(spy0, "noevents", "fixed", 200, 0, 300, True, "norm")
    assert set(spy0.log["n"]) == {0}


def test_count_mode_norm_equals_last50_at_50_ms(synth):
    sa, sb = _spy(beta_root=0.3, beta_trans=0.3), _spy(beta_root=0.3, beta_trans=0.3)
    a, b = _run(sa), _run(sb, "model", "fixed", STEP, 0, 300, False, "norm")
    assert [float(n) for n in sa.log["n"]] == sb.log["n"]
    assert np.array_equal(b["count_f"], a["count"].astype(np.float64))
    for k in ("pred", "gt", "end", "run", "elapsed", "count", "prev", "win"):
        assert np.array_equal(a[k], b[k]), k


def test_window_set_in_the_loop_follows_the_function(synth, wlog):
    off = synth["offsets"]
    ws = (50, 100, 200)
    r = _run(_spy(), "model", "fixed", STEP, 6000, 300, True, "last50", ws)
    assert set(r["win"]) <= set(ws) | {int(e + 1 - (0 if run == 0 else 320)) for e, run in zip(r["end"], r["run"])}
    for e, run, w in zip(r["end"], r["run"], r["win"]):
        lim = int(e) + 1 - (0 if run == 0 else 320)
        pick = next((m for m in ws if off[e + 1] - off[e - min(m, lim) + 1] >= 6000), ws[-1])
        assert w == min(pick, lim), (e, w, pick, lim)
    assert [w for _, w in wlog] == list(r["win"])
    assert len(set(r["win"])) > 1                                              # the test is not vacuous
    # an empty set is the fixed / adaptive window
    assert np.array_equal(_run(_spy(), "model", "fixed", 100, 0, 300, False, "last50", ())["win"],
                          _run(_spy(), "model", "fixed", 100)["win"])


# ----------------------------------------------------------------------------------------------------- perturb_trials
def test_perturb_branches_use_the_closed_loops_window_and_count(synth, wlog):
    spy = _spy(beta_root=0.3, beta_trans=0.3)
    r = _run(spy, "model", "fixed", 200, 0, 300, True, "norm")
    wlog.clear()
    spy.log["n"].clear()
    out = EX.perturb_trials(spy, _cfg(), Path("/x"), "d", "s", DEV, r, thetas=(10.0,), horizon=3, every=1, min_elapsed=100)
    n_trials = out[10.0]["div"].shape[0]
    assert n_trials > 0 and len(wlog) == n_trials * 3
    win_of = dict(zip(map(int, r["end"]), map(int, r["win"])))
    assert all(w == win_of[e] for e, w in wlog) and {w for _, w in wlog} - {STEP}        # branch windows are the loop's, not 50
    assert set(spy.log["n"]) <= {float(c) for c in r["count_f"]} and all(type(n) is float for n in spy.log["n"])
    # without a `win` entry (an older run_sequence result) the branches are 50 ms packets and get the plain count
    spy2 = _spy(beta_root=0.3, beta_trans=0.3)
    r2 = _run(spy2)
    assert (r2["win"] == STEP).all()
    wlog.clear()
    spy2.log["n"].clear()
    r3 = {k: v for k, v in r2.items() if k != "win"}
    EX.perturb_trials(spy2, _cfg(), Path("/x"), "d", "s", DEV, r3, thetas=(10.0,), horizon=3, every=1, min_elapsed=100)
    assert {w for _, w in wlog} == {STEP} and all(type(n) is int for n in spy2.log["n"])
    # ... and with the default `win` (all 50) the branch results are those of an `r` without it, bit for bit
    wlog.clear()
    o2 = EX.perturb_trials(spy2, _cfg(), Path("/x"), "d", "s", DEV, r2, thetas=(10.0,), horizon=3, every=1, min_elapsed=100)
    o3 = EX.perturb_trials(spy2, _cfg(), Path("/x"), "d", "s", DEV, r3, thetas=(10.0,), horizon=3, every=1, min_elapsed=100)
    for k in ("div", "excess"):
        assert np.array_equal(o2[10.0][k], o3[10.0][k])


# --------------------------------------------------------------------------------- evaluate: old callers, npz keys, json
def _fake_metrics(monkeypatch):
    def psm(mano, r, device):
        d = np.linalg.norm(r["pred"] - r["gt"], axis=1)
        return {"mpjpe_ra_mm": d * 1000, "mpvpe_ra_mm": d * 1100, "mpjpe_abs_mm": d * 1200, "mpvpe_abs_mm": d * 1300,
                "root_rot_deg": EX.rot_err_deg(r["pred"], r["gt"]), "transl_mm": d * 10}
    monkeypatch.setattr(EX, "per_step_metrics", psm)
    monkeypatch.setattr(EX, "jitter_decomp", lambda mano, r, device: {"n_steps": 1, "jit_gt_mm": 0.5})


def _ns(**kw):
    base = dict(controls=False, tf=False, perturb=False, window_mode="fixed", window_ms=STEP, min_events=0, max_window_ms=300)
    return argparse.Namespace(**{**base, **kw})


def test_evaluate_with_an_old_namespace_and_an_old_signature_run_sequence(synth, monkeypatch):
    """The probes monkeypatch `EX.run_sequence` with the 12-argument signature and the callers pass Namespaces that predate the
    options: neither may break, and the saved arrays are the historical key set."""
    _fake_metrics(monkeypatch)
    real, calls = EX.run_sequence, []

    def old_signature(model, cfg, root, d, seq, device, rng, mode="model", wmode="fixed", win=STEP, min_events=0, max_win=300):
        calls.append((mode, wmode, win, min_events, max_win))
        return real(model, cfg, root, d, seq, device, rng, mode, wmode, win, min_events, max_win)
    monkeypatch.setattr(EX, "run_sequence", old_signature)
    model = _Stand(scale=0.1).eval()
    summ, arrays = EX.evaluate(model, _cfg(), None, Path("/x"), [("s", "d")], DEV, _ns(controls=True, tf=True), label="t")
    assert calls == [(m, "fixed", STEP, 0, 300) for m in ("model", "hold", "noevents", "tf")]
    keys = {f"model|s|{k}" for k in ("pred", "gt", "end", "run", "elapsed", "count", "mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm",
                                     "mpvpe_abs_mm", "root_rot_deg", "transl_mm")}
    keys |= {f"{m}|s|{k}" for m in ("hold", "noevents", "tf") for k in ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm",
                                                                         "mpvpe_abs_mm", "root_rot_deg", "transl_mm")}
    assert set(arrays) == keys and not any(k.endswith(("|win", "|count_f")) for k in arrays)
    # an explicit all-default Namespace (what filter_eval / the new CLI pass) is the same evaluation
    calls.clear()
    summ2, arrays2 = EX.evaluate(model, _cfg(), None, Path("/x"), [("s", "d")], DEV,
                                 _ns(controls=True, tf=True, clip_run_start=False, count_mode="last50", window_set=()), label="t")
    assert set(arrays2) == set(arrays) and all(np.array_equal(arrays[k], arrays2[k]) for k in arrays)
    assert summ2["model"]["overall"] == summ["model"]["overall"] and len(calls) == 4
    # a monkeypatched old signature cannot take an option: asking for one is an error, not a silent default
    with pytest.raises(TypeError):
        EX.evaluate(model, _cfg(), None, Path("/x"), [("s", "d")], DEV, _ns(clip_run_start=True))


def test_run_sequence_refuses_an_unknown_count_mode(synth):
    with pytest.raises(ValueError):
        _run(_spy(), "model", "fixed", STEP, 0, 300, False, "mean")


def test_run_sequence_accepts_the_twelve_positional_arguments(synth):
    model = _spy()
    r = EX.run_sequence(model, _cfg(), Path("/x"), "d", "s", DEV, np.random.default_rng(0), "model", "fixed", STEP, 0, 300)
    r2 = _run(_spy())
    assert np.array_equal(r["pred"], r2["pred"])
    r3 = EX.run_sequence(_spy(), _cfg(), Path("/x"), "d", "s", DEV, np.random.default_rng(0), "model", "adaptive", 50, 3000, 300)
    assert len(r3["win"]) == 12


def test_evaluate_saves_the_window_arrays_only_with_an_option_on(synth, monkeypatch):
    _fake_metrics(monkeypatch)
    model = _spy()
    for kw, extra in ((dict(clip_run_start=True), {"win"}), (dict(count_mode="norm"), {"win", "count_f"}),
                      (dict(window_set=(100, 200), min_events=3000), {"win"}),
                      (dict(window_ms=200, clip_run_start=True, count_mode="norm"), {"win", "count_f"})):
        _, arrays = EX.evaluate(model, _cfg(), None, Path("/x"), [("s", "d")], DEV, _ns(**kw), label="t")
        assert {k.split("|")[2] for k in arrays if k.startswith("model|s|")} >= extra
        assert {k for k in arrays if k.endswith(("|win", "|count_f"))} == {f"model|s|{e}" for e in extra}
        assert arrays["model|s|win"].dtype == np.int64
    # window options that predate DT3 keep the historical keys
    for kw in (dict(window_ms=200), dict(window_mode="adaptive", window_ms=50, min_events=3000)):
        _, arrays = EX.evaluate(model, _cfg(), None, Path("/x"), [("s", "d")], DEV, _ns(**kw), label="t")
        assert not any(k.endswith(("|win", "|count_f")) for k in arrays)
    # the options reach the loop: a clipped W=200 run differs from the unclipped one only in the second segment
    _, a_un = EX.evaluate(model, _cfg(), None, Path("/x"), [("s", "d")], DEV, _ns(window_ms=200), label="t")
    _, a_cl = EX.evaluate(model, _cfg(), None, Path("/x"), [("s", "d")], DEV, _ns(window_ms=200, clip_run_start=True), label="t")
    assert list(a_cl["model|s|win"]) == [50, 100, 150, 200, 200, 200] * 2
    assert not np.array_equal(a_un["model|s|pred"][6:], a_cl["model|s|pred"][6:])


# ---------------------------------------------------------------------------------------------- file tags / json record
def test_eval_tag_default_names_are_the_old_names():
    n = 0
    for split in ("val_core", "val_x"):
        for ckpt in ("last", "selected", "step=1500"):
            for tf in (False, True):
                for pert in (False, True):
                    for wmode, wms, me in (("fixed", 50, 0), ("fixed", 200, 0), ("fixed", 75, 100), ("adaptive", 50, 2000),
                                           ("adaptive", 100, 0)):
                        for suffix in ("", "z0live"):
                            a = argparse.Namespace(split=split, ckpt=ckpt, tf=tf, perturb=pert, window_mode=wmode, window_ms=wms,
                                                   min_events=me, max_window_ms=300, suffix=suffix)       # no DT3 fields at all
                            assert EX.eval_tag(a) == old_tag(a)
                            a.clip_run_start, a.count_mode, a.window_set = False, "last50", ()
                            assert EX.eval_tag(a) == old_tag(a)
                            n += 1
    assert n == 2 * 3 * 2 * 2 * 5 * 2
    a = argparse.Namespace(split="val_core", ckpt="last", tf=True, perturb=True, window_mode="fixed", window_ms=STEP, min_events=0,
                           max_window_ms=300, suffix="")
    assert EX.eval_tag(a) == "evalx_val_core_last_tf_pert"                                            # the recorded name
    a.window_ms = 200
    a.suffix = "w200"
    assert EX.eval_tag(a) == "evalx_val_core_last_tf_pert_wfixed200_w200"                             # the recorded W=200 name


def test_eval_tag_new_options():
    mk = lambda **k: argparse.Namespace(**{"split": "val_core", "ckpt": "last", "tf": True, "perturb": True, "window_mode": "fixed",   # noqa: E731
                                           "window_ms": STEP, "min_events": 0, "max_window_ms": 300, "suffix": "", **k})
    assert EX.eval_tag(mk(clip_run_start=True)) == "evalx_val_core_last_tf_pert_clip"
    assert EX.eval_tag(mk(count_mode="norm")) == "evalx_val_core_last_tf_pert_cnorm"
    assert EX.eval_tag(mk(window_ms=200, clip_run_start=True, count_mode="norm", suffix="s")) == \
        "evalx_val_core_last_tf_pert_wfixed200_clip_cnorm_s"
    assert EX.eval_tag(mk(window_set=(150, 200, 300), min_events=5000, clip_run_start=True, suffix="dt3a_set")) == \
        "evalx_val_core_last_tf_pert_wset150-200-300_n5000_clip_dt3a_set"
    assert EX.eval_tag(mk(window_set=(300, 150), min_events=800, count_mode="norm")) == "evalx_val_core_last_tf_pert_wset150-300_n800_cnorm"
    # (adaptive) windows keep their own tag when the set is empty
    assert EX.eval_tag(mk(window_mode="adaptive", window_ms=100, min_events=2000, clip_run_start=True)) == \
        "evalx_val_core_last_tf_pert_wadaptive100_n2000_x300_clip"


def test_window_record_is_unchanged_by_default_and_complete_otherwise():
    d = _ns()
    assert EX.window_record(d) == {"mode": "fixed", "ms": 50, "min_events": 0, "max_ms": 300}
    assert list(EX.window_record(d)) == ["mode", "ms", "min_events", "max_ms"]
    assert EX.window_record(_ns(clip_run_start=False, count_mode="last50", window_set=())) == EX.window_record(d)
    assert EX.window_record(_ns(window_ms=200)) == {"mode": "fixed", "ms": 200, "min_events": 0, "max_ms": 300}
    r = EX.window_record(_ns(window_ms=200, clip_run_start=True))
    assert r == {"mode": "fixed", "ms": 200, "min_events": 0, "max_ms": 300, "clip_run_start": True, "count_mode": "last50",
                 "window_set": []}
    r = EX.window_record(_ns(window_set=(150, 200, 300), min_events=5000, count_mode="norm"))
    assert r["mode"] == "set" and r["window_set"] == [150, 200, 300] and r["count_mode"] == "norm" and r["clip_run_start"] is False
    json.dumps(r)


def test_evalx_cli_defaults_and_flags(monkeypatch):
    seen = []
    monkeypatch.setattr(EX, "cmd_eval", lambda a: seen.append(a))
    run = lambda *args: (monkeypatch.setattr(sys, "argv", ["evalx.py", "eval", "--run-dir", "R", *args]), EX.main())   # noqa: E731
    run()
    a = seen[-1]
    assert a.clip_run_start is False and a.count_mode == "last50" and a.window_set == () and a.window_ms == STEP
    run("--window-ms", "200", "--clip-run-start", "--count-mode", "norm")
    a = seen[-1]
    assert a.clip_run_start is True and a.count_mode == "norm" and a.window_ms == 200 and a.window_set == ()
    run("--window-set", "300,150,200", "--min-events", "5000")
    assert seen[-1].window_set == (150, 200, 300) and seen[-1].min_events == 5000
    for bad in (["--window-set", "150,200"], ["--window-set", "150,200", "--min-events", "10", "--window-ms", "200"],
                ["--count-mode", "mean"], ["--window-set", "0,5", "--min-events", "10"]):
        n = len(seen)
        with pytest.raises(SystemExit):
            run(*bad)
        assert len(seen) == n


def test_row_and_aggregate_runs_read_the_new_file_names(tmp_path, monkeypatch):
    """`row --variant` is the tail after `evalx_<split>_<ckpt>_`; every new name is such a tail."""
    monkeypatch.setattr(EX, "load_config", lambda p: {"cfg": str(p)})
    runs = []
    for seed in ("3407", "3408"):
        run = tmp_path / f"arm_s{seed}"
        run.mkdir()
        (run / "arm.yaml").write_text("x: 1\n")
        (run / "training_metadata.json").write_text(json.dumps({"seed": int(seed)}))
        runs.append(run)
    keys = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm", "root_rot_deg", "transl_mm")
    seq = {"n_frames": 10, **{k: [1.5, 1.0, 2.0] for k in keys}}
    js = {"step": 6000, "ckpt": "c", "model": {"n_frames": 20, "overall": {k: 1.5 for k in keys}, "zgz_global": seq,
                                               "zgz_local": {**seq, "n_frames": 10}, "jitter": {}}}
    a = argparse.Namespace(split="val_core", ckpt="last", tf=True, perturb=True, window_mode="fixed", window_ms=200, min_events=0,
                           max_window_ms=300, suffix="dt3a_set", clip_run_start=True, count_mode="norm", window_set=())
    tag = EX.eval_tag(a)
    variant = tag[len("evalx_val_core_last_"):]
    assert variant == "tf_pert_wfixed200_clip_cnorm_dt3a_set"
    for run in runs:
        (run / f"{tag}.json").write_text(json.dumps(js))
    per_seed, ext, mean, cfg = EX.aggregate_runs([str(r) for r in runs], "val_core", "last", variant)
    assert set(per_seed) == {"3407", "3408"} and mean["overall"]["mpjpe_ra_mm"] == 1.5
    assert per_seed["3407"]["global"]["mpjpe_ra_mm"] == 1.5 and per_seed["3407"]["local"]["mpjpe_ra_mm"] == 1.5
    with pytest.raises(FileNotFoundError):                                      # the plain name is another file
        EX.aggregate_runs([str(r) for r in runs], "val_core", "last", "tf_pert")


# ------------------------------------------------------------------------------------------------------- filter_eval
def test_filter_eval_eval_args_default_and_options():
    d = FE.eval_args(True, False, True)
    assert (d.window_ms, d.window_mode, d.min_events, d.max_window_ms) == (STEP, "fixed", 0, 300)
    assert (d.clip_run_start, d.count_mode, d.window_set) == (False, "last50", ())
    assert EX.run_options(d) == {}
    n = FE.eval_args(False, True, False, 200, True, "norm", "300,150", 5000)
    assert (n.window_ms, n.clip_run_start, n.count_mode, n.window_set, n.min_events) == (200, True, "norm", (150, 300), 5000)
    assert FE.eval_args(window_ms=100).window_ms == 100 and FE.eval_args(controls=True, tf=True, perturb=True, window_ms=EX.STEP).tf
    assert EX.run_options(n) == {"clip_run_start": True, "count_mode": "norm", "wset": (150, 300)}


@pytest.fixture
def fe_env(tmp_path, monkeypatch):
    """`filter_eval.cmd_eval` without a model / data: the names and jsons it writes, and the evaluate args it passes."""
    run = tmp_path / "arm_s3407"
    run.mkdir()
    (run / "training_metadata.json").write_text(json.dumps({"train_sequences": []}))
    seen = []

    class _Mano:
        def __init__(self, *a, **k):
            pass

        def to(self, d):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(FE, "build_run", lambda run_, ckpt, device, config=None: (
        {"MANO": {"NPZ": "x"}, "DATA": {"ROOT": "/x", "SPLITS_MANIFEST": "/m"}}, _Stand().eval(), Path("c.ckpt"), 6000, None))
    monkeypatch.setattr(FE, "ManoLayer", _Mano)
    monkeypatch.setattr(FE, "sequences_for_split", lambda root, split, mani: [("s", "d")])

    def fake_evaluate(net, cfg, mano, root, seqs, device, ns, label=""):
        seen.append((label, ns))
        return {"model": {"overall": {"mpjpe_ra_mm": 1.0}}}, {"model|s|pred": np.zeros((2, 51), np.float32)}
    monkeypatch.setattr(FE.EX, "evaluate", fake_evaluate)
    return run, tmp_path / "out", seen


def _fe_args(run, out, **kw):
    base = dict(run_dir=str(run), ckpt="last", gains="0.5,1,0.5", filter_spec="", raw=True, controls=False, tf=False, perturb=False,
                out_dir=str(out), split="val_core", manifest=None, config=None, suffix="", device="cpu", threads=0, no_npz=False,
                window_ms=STEP)
    return argparse.Namespace(**{**base, **kw})


def _fe_tags(out):
    return sorted(p.stem for p in out.glob("*.json"))


def test_filter_eval_tags_default_and_new_options(fe_env):
    run, out, seen = fe_env
    spec = {"a_root": [[300, 0.3], [3000, 0.5], [30000, 0.8]], "a_rest": 0.8, "a_trans": 0.5}
    st = FE.spec_tag(spec)
    FE.cmd_eval(_fe_args(run, out / "d", filter_spec=json.dumps(spec)))                 # an old-style Namespace: no DT3 fields
    assert _fe_tags(out / "d") == sorted(["raw", "r0.5_f1.0_t0.5", st])
    j = json.loads((out / "d" / "raw.json").read_text())
    assert j["window"] == {"mode": "fixed", "ms": 50, "min_events": 0, "max_ms": 300}
    assert sorted(np.load(out / "d" / "raw.npz").files) == ["model|s|pred"]
    FE.cmd_eval(_fe_args(run, out / "w", filter_spec=json.dumps(spec), window_ms=200))
    assert _fe_tags(out / "w") == sorted(["raw_w200", "r0.5_f1.0_t0.5_w200", st + "_w200"])
    assert json.loads((out / "w" / "raw_w200.json").read_text())["window"] == {"mode": "fixed", "ms": 200, "min_events": 0, "max_ms": 300}
    FE.cmd_eval(_fe_args(run, out / "c", filter_spec=json.dumps(spec), window_ms=200, clip_run_start=True, count_mode="norm",
                         suffix="x"))
    assert _fe_tags(out / "c") == sorted(["raw_w200_clip_cnorm_x", "r0.5_f1.0_t0.5_w200_clip_cnorm_x", st + "_w200_clip_cnorm_x"])
    jw = json.loads((out / "c" / (st + "_w200_clip_cnorm_x.json")).read_text())["window"]
    assert jw == {"mode": "fixed", "ms": 200, "min_events": 0, "max_ms": 300, "clip_run_start": True, "count_mode": "norm",
                  "window_set": []}
    FE.cmd_eval(_fe_args(run, out / "s", filter_spec=json.dumps(spec), window_ms=STEP, window_set=(150, 200, 300), min_events=5000,
                         clip_run_start=True))
    assert _fe_tags(out / "s") == sorted(["raw_wset150-200-300_n5000_clip", "r0.5_f1.0_t0.5_wset150-200-300_n5000_clip",
                                          st + "_wset150-200-300_n5000_clip"])
    assert json.loads((out / "s" / "raw_wset150-200-300_n5000_clip.json").read_text())["window"]["mode"] == "set"
    # --no-wtag: the tags are the plain spec hash / gain triple whatever the window options are
    FE.cmd_eval(_fe_args(run, out / "n", filter_spec=json.dumps(spec), window_ms=200, clip_run_start=True, count_mode="norm",
                         no_wtag=True))
    assert _fe_tags(out / "n") == sorted(["raw", "r0.5_f1.0_t0.5", st])
    assert json.loads((out / "n" / "raw.json").read_text())["window"]["clip_run_start"] is True        # the json still says it
    # the evaluate args the options become
    ns = seen[-1][1]
    assert (ns.window_ms, ns.clip_run_start, ns.count_mode, ns.window_set) == (200, True, "norm", ())
    # a contradictory window set is refused before anything is written
    with pytest.raises(ValueError):
        FE.cmd_eval(_fe_args(run, out / "bad", window_ms=200, window_set=(150, 200), min_events=100))
    assert not (out / "bad").exists()


def test_law_check_replays_on_the_stored_windows(monkeypatch, tmp_path):
    """`--law-check` builds the replayed packet over the window the stored evaluation used (its `win` array, written by a DT3
    window-option run); an evaluation without it is replayed on 50 ms packets, as before."""
    from test_dt_filter import _Contract, _fake_sequence, _fake_stored              # noqa: E402
    _fake_sequence(monkeypatch)
    log, inner = [], EX.ET.build_lnes
    monkeypatch.setattr(EX.ET, "build_lnes", lambda ev, off, end, w, ch: (log.append((int(end), int(w))), inner(ev, off, end, w, ch))[1])
    target, rho, a = np.array([0.3, -0.5, 0.2]), 0.3, 0.5
    npz = _fake_stored(tmp_path, target, (1.0 - a) + a * rho)
    out0 = FE.law_check_tag(_Contract(target, rho), (a, 1.0, 0.5), {}, tmp_path, [("s0", "d0")], npz)
    assert log and {w for _, w in log} == {STEP}
    z = dict(np.load(npz))
    z["model|s0|win"] = 100 + 50 * (np.arange(110) % 3)
    npz2 = tmp_path / "with_win.npz"
    np.savez_compressed(npz2, **z)
    log.clear()
    out1 = FE.law_check_tag(_Contract(target, rho), (a, 1.0, 0.5), {}, tmp_path, [("s0", "d0")], npz2)
    assert log and all(w == z["model|s0|win"][(e - 49) // 50] for e, w in log) and len({w for _, w in log}) > 1
    assert out1["10"]["n_trials"] == out0["10"]["n_trials"] == 3                     # the plan is unchanged


def test_filter_eval_cli_flags(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(FE, "cmd_eval", lambda a, post=None: seen.append(a))
    base = ["--run-dir", "R", "--out-dir", str(tmp_path), "--raw"]
    FE.main(base)
    a = seen[-1]
    assert (a.clip_run_start, a.count_mode, a.window_set, a.no_wtag, a.window_ms, a.min_events) == (False, "last50", (), False, STEP, 0)
    FE.main(base + ["--window-ms", "200", "--clip-run-start", "--count-mode", "norm", "--no-wtag"])
    a = seen[-1]
    assert (a.clip_run_start, a.count_mode, a.no_wtag, a.window_ms) == (True, "norm", True, 200)
    FE.main(base + ["--window-set", "300,150,200", "--min-events", "5000"])
    assert seen[-1].window_set == (150, 200, 300) and seen[-1].min_events == 5000
    n = len(seen)
    for bad in (["--window-set", "150,200"], ["--count-mode", "x"], ["--window-set", "150", "--min-events", "9", "--window-ms", "100"]):
        with pytest.raises(SystemExit):
            FE.main(base + bad)
    assert len(seen) == n
    assert "stay at 50 ms" not in FE.__doc__ and "always use 50 ms" not in FE.__doc__


# ------------------------------------------------------------------------------------------------------ filter_sweep_2fold
def _sw_ns(**kw):
    base = dict(run_dir="R", spec_file="", out_dir="", raw=True, primary=False, ckpt="last", device=None, threads=0, no_perturb=True)
    return argparse.Namespace(**{**base, **kw})


def test_sweep_eval_passes_the_window_options_and_drops_the_window_tag(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(FE, "cmd_eval", lambda ns, post=None: seen.append(ns))
    SW.cmd_eval(_sw_ns(out_dir=str(tmp_path / "a")))                            # the Namespace of an older caller: no window fields
    ns = seen[-1]
    assert (ns.window_ms, ns.clip_run_start, ns.count_mode, ns.window_set, ns.no_wtag) == (STEP, False, "last50", (), False)
    SW.cmd_eval(_sw_ns(out_dir=str(tmp_path / "b"), window_ms=STEP, clip_run_start=False, count_mode="last50", window_set=(), min_events=0))
    assert seen[-1].no_wtag is False
    for kw in (dict(window_ms=200), dict(clip_run_start=True), dict(count_mode="norm"), dict(window_set=(150, 200), min_events=100)):
        SW.cmd_eval(_sw_ns(out_dir=str(tmp_path / "c"), **kw))
        ns = seen[-1]
        assert ns.no_wtag is True and ns.out_dir == str(tmp_path / "c" / "R")
    SW.cmd_eval(_sw_ns(out_dir=str(tmp_path / "d"), window_ms=200, clip_run_start=True, count_mode="norm"))
    ns = seen[-1]
    assert (ns.window_ms, ns.clip_run_start, ns.count_mode, ns.perturb, ns.controls, ns.tf) == (200, True, "norm", False, False, False)
    with pytest.raises(ValueError):
        SW.cmd_eval(_sw_ns(out_dir=str(tmp_path / "e"), window_set=(150,), min_events=0))


def test_a_non_default_window_needs_its_own_sweep_directory(tmp_path, monkeypatch):
    rec = tmp_path / "recorded"                                                  # stands in for the recorded sweep's directory
    monkeypatch.setattr(SW, "DEFAULT_OUT", rec)
    for kw in (dict(window_ms=200), dict(clip_run_start=True), dict(count_mode="norm"), dict(window_set=(150,), min_events=10)):
        with pytest.raises(ValueError):
            SW.window_args(_sw_ns(out_dir=str(rec), **kw))
        with pytest.raises(ValueError):
            SW.stage_jobs(argparse.Namespace(stage="1a", runs=["r"], out_dir=str(rec), gpus="4", refs=False, **kw))
        assert not rec.exists()                                                  # refused before a spec file was written
    assert SW.window_args(_sw_ns(out_dir=str(rec)))["non_default"] is False      # the default sweep may use it
    assert SW.window_args(_sw_ns(out_dir=str(tmp_path / "other"), window_ms=200))["non_default"] is True


def _jobs_ns(tmp_path, **kw):
    base = dict(stage="1a", runs=["dt_dz_l3_s3407"], out_dir=str(tmp_path / "out"), gpus="4,5,6,7", refs=False)
    return argparse.Namespace(**{**base, **kw})


def test_stage_jobs_are_sched2_job_json_for_the_spec_chunks(tmp_path):
    jobs = SW.stage_jobs(_jobs_ns(tmp_path, refs=True, window_ms=200, clip_run_start=True, count_mode="norm"))
    assert [j["id"] for j in jobs] == [f"dt3_sw_1a_dt_dz_l3_s3407_c{k}" for k in range(4)]
    specs = SW.stage_specs("1a", tmp_path)
    for k, j in enumerate(jobs):
        assert set(j) == {"id", "kind", "cwd", "prio", "cores", "mem_mib", "cmd"}
        assert (j["kind"], j["cwd"], j["prio"], j["cores"], j["mem_mib"]) == ("eval", str(SW.REPO), 75, 1, 2500)
        c = j["cmd"]
        assert c[:3] == [SW.PY, "tools/dt/filter_sweep_2fold.py", "eval"]
        assert c[c.index("--run-dir") + 1] == "outputs/semkine/dt_dz_l3_s3407"
        assert c[c.index("--out-dir") + 1] == str((tmp_path / "out").resolve())
        sf = Path(c[c.index("--spec-file") + 1])
        assert sf.name == f"stage1a_dt_dz_l3_s3407_c{k}.json" and json.loads(sf.read_text()) == specs[k::4]
        assert ("--raw" in c and "--primary" in c) == (k == 0)
        assert c[c.index("--window-ms") + 1] == "200" and "--clip-run-start" in c and c[c.index("--count-mode") + 1] == "norm"
        assert "--window-set" not in c and "--min-events" not in c
    assert sum(len(json.loads(Path(j["cmd"][j["cmd"].index("--spec-file") + 1]).read_text())) for j in jobs) == 13
    json.dumps(jobs)
    # default window options: the command carries none; two runs; a window set carries --min-events
    d = SW.stage_jobs(_jobs_ns(tmp_path, runs=["dt_dz_l3_s3407", "dt_dz_l3_s3408"], gpus="4,7"))
    assert [j["id"] for j in d] == ["dt3_sw_1a_dt_dz_l3_s3407_c0", "dt3_sw_1a_dt_dz_l3_s3407_c1",
                                    "dt3_sw_1a_dt_dz_l3_s3408_c0", "dt3_sw_1a_dt_dz_l3_s3408_c1"]
    assert all(not {"--window-ms", "--clip-run-start", "--count-mode", "--window-set", "--min-events", "--raw"} & set(j["cmd"]) for j in d)
    s = SW.stage_jobs(_jobs_ns(tmp_path, gpus="5", window_set=(300, 150), min_events=5000))
    assert len(s) == 1 and s[0]["cmd"][s[0]["cmd"].index("--window-set") + 1] == "150,300" and "--min-events" in s[0]["cmd"]
    # without --refs an empty chunk is no job
    assert len(SW.stage_jobs(_jobs_ns(tmp_path, gpus="4,5,6,7", stage="1a"))) == 4


def test_stage_jobs_only_accept_gpus_4_to_7(tmp_path):
    for bad in ("3,4", "4,8", "0", "", "4,4", "a", "2,3,4,5"):
        with pytest.raises(SystemExit):
            SW.stage_jobs(_jobs_ns(tmp_path, gpus=bad))
    assert not (tmp_path / "out").exists()                                       # refused before anything is written


def test_stage_jobs_cli_prints_the_json(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(SW.subprocess, "Popen", lambda *a, **k: pytest.fail("stage-jobs must not launch anything"))
    SW.main(["stage-jobs", "--stage", "1a", "--runs", "dt_dz_l3_s3407", "--gpus", "4,5", "--out-dir", str(tmp_path / "o"),
             "--window-ms", "200", "--clip-run-start", "--refs"])
    jobs = json.loads(capsys.readouterr().out)
    assert len(jobs) == 2 and jobs[0]["id"] == "dt3_sw_1a_dt_dz_l3_s3407_c0" and "--raw" in jobs[0]["cmd"] and "--raw" not in jobs[1]["cmd"]
    with pytest.raises(SystemExit):                                              # the output directory is not defaulted
        SW.main(["stage-jobs", "--stage", "1a", "--runs", "r"])
    with pytest.raises(SystemExit):
        SW.main(["stage-jobs", "--stage", "1a", "--runs", "r", "--gpus", "2,3", "--out-dir", str(tmp_path / "o2")])
    with pytest.raises(SystemExit):                                              # a window set needs --min-events
        SW.main(["stage-jobs", "--stage", "1a", "--runs", "r", "--window-set", "150,200", "--out-dir", str(tmp_path / "o3")])
    # the sub-commands that exist keep their options: eval / run take the window options too
    seen = []
    monkeypatch.setattr(SW, "cmd_eval", lambda a: seen.append(a))
    SW.main(["eval", "--run-dir", "R", "--window-ms", "150", "--count-mode", "norm", "--out-dir", str(tmp_path / "o4")])
    assert seen[-1].window_ms == 150 and seen[-1].count_mode == "norm" and seen[-1].clip_run_start is False
    with pytest.raises(SystemExit):                                              # ... but not into the recorded sweep's directory
        SW.main(["eval", "--run-dir", "R", "--window-ms", "150"])
    SW.main(["eval", "--run-dir", "R"])                                          # the default evaluation keeps its default directory
    assert seen[-1].window_ms == STEP and seen[-1].out_dir == str(SW.DEFAULT_OUT)


def test_sweep_run_commands_are_the_historical_ones_by_default(tmp_path, monkeypatch):
    got = []
    monkeypatch.setattr(SW, "run_parallel", lambda jobs, gpus, cpus, log_dir: got.append((jobs, gpus, cpus)) or 0)
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        SW.main(["run", "--stage", "1a", "--runs", "dt_dz_l3_s3407", "dt_dz_l3_s3408", "--gpus", "4,7,5", "--cpus", "1:2:3",
                 "--out-dir", str(out), "--refs"])
    jobs, gpus, cpus = got[0]
    specs = SW.stage_specs("1a", out)
    exp = []
    for run in ("dt_dz_l3_s3407", "dt_dz_l3_s3408"):                             # the construction of `run` before DT3, verbatim
        for ci, ch in enumerate([specs[i::3] for i in range(3)]):
            if not ch and not (ci == 0 and True):
                continue
            sf = out / "_specs" / f"stage1a_{run}_c{ci}.json"
            argv = [str(Path(SW.__file__).resolve()), "eval", "--run-dir", f"outputs/semkine/{run}", "--spec-file", str(sf),
                    "--out-dir", str(out)]
            if ci == 0:
                argv += ["--raw", "--primary"]
            exp.append(argv)
    assert jobs == exp and gpus == ["4", "7", "5"] and cpus == ["1", "2", "3"]
    # the window options ride along on the same commands
    got.clear()
    with pytest.raises(SystemExit):
        SW.main(["run", "--stage", "1a", "--runs", "dt_dz_l3_s3407", "--gpus", "4", "--cpus", "1", "--out-dir", str(out),
                 "--window-ms", "200", "--clip-run-start"])
    assert got[0][0][0][-3:] == ["--window-ms", "200", "--clip-run-start"]


def test_sweep_tables_work_on_the_decoration_free_tags(tmp_path):
    """A window-option sweep keeps the tags = spec hashes in its own directory, so `load_table` / `ranked` need no change."""
    spec = SW.const_specs()[1]
    tag = SW._tag(spec)
    run_dir = tmp_path / "w200c" / "dt_dz_l3_s3407"
    run_dir.mkdir(parents=True)
    f = {"n": 10.0, "ra_sum": 100.0, "n2": 8.0, "acc_err_sum": 40.0, "acc_pred_sum": 30.0, "acc_gt_sum": 20.0, "n1": 9.0,
         "spd_pred_sum": 9.0, "spd_gt_sum": 18.0, "k1_n": 0.0, "k1_sum": 0.0}
    folds = {fold: {c: dict(f) for c in ("global", "local")} for fold in "AB"}
    one = {"n_frames": 40, "mpjpe_ra_mm": [10.0, 9.0, 11.0], "root_rot_deg": [5.0, 4.0, 6.0], "mpvpe_ra_mm": [1.0, 1.0, 1.0],
           "mpjpe_abs_mm": [1.0, 1.0, 1.0], "mpvpe_abs_mm": [1.0, 1.0, 1.0], "transl_mm": [1.0, 1.0, 1.0],
           "motion": {"root_speed_ratio": 1.0, "finger_speed_ratio": 1.0},
           "failure": {"episodes": 0, "bad_step_frac": 0.0, "fail_time_frac": 0.0, "longest_s": 0.0}}
    ov = {k: 10.0 for k in ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm", "root_rot_deg", "transl_mm")}
    for name, filt in ((tag, {"class": "semkine.anchored.AdaptiveFilter", "tag": tag, "spec": spec}), ("raw", None)):
        j = {"run": "dt_dz_l3_s3407", "filter": filt, "step": 1, "ckpt": "c", "model": {"n_frames": 80, "overall": ov, "zgz_global": one,
                                                                                         "zgz_local": one, "jitter": {}},
             "post": {"folds": folds, "n_blocks": {"A": 2, "B": 2}},
             "window": {"mode": "fixed", "ms": 200, "min_events": 0, "max_ms": 300, "clip_run_start": True, "count_mode": "last50",
                        "window_set": []}}
        (run_dir / f"{name}.json").write_text(json.dumps(j))
    table = SW.load_table(tmp_path / "w200c", ["dt_dz_l3_s3407"])
    assert set(table) == {tag, "raw"} and table[tag]["kind"] == "spec" and table["raw"]["kind"] == "raw"
    assert SW.ranked(table, ["dt_dz_l3_s3407"], "A") == [tag]
