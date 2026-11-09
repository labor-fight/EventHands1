"""TPD EvRealHands runner (tools/tpd/erh_eval.py): label grid, causal stepping, initial noise. CPU only."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path  # noqa: E402
import sys  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from semkine import tpd_causal as TC  # noqa: E402
from tools.tpd import erh_eval as R  # noqa: E402


def _labels(ids, t0_us=3700666.6666666665):
    ids = np.asarray(ids)
    t = t0_us + ids * 1e6 / 15
    return {"frame_id": ids, "label_ms": np.rint(t / 1000).astype(np.int64),
            "params51": np.arange(len(ids) * 51, dtype=np.float32).reshape(-1, 51)}


def test_scored_labels_fall_on_the_packet_grid():
    lab = _labels(list(range(10)) + [20, 21] + list(range(30, 34)))
    runs = R.scored_runs(lab)
    assert [len(ms) for ms, _ in runs] == [10, 4]                 # the two-label run has no scored label
    for ms, P in runs:
        k = np.arange(0, len(ms), R.STRIDE)
        assert np.array_equal(ms[k] - ms[0], k // R.STRIDE * R.SCORED_MS)
    assert np.array_equal(runs[0][1][0], lab["params51"][0])


def test_a_label_off_the_grid_is_refused():
    lab = _labels(range(7))
    lab["label_ms"] = lab["label_ms"].copy()
    lab["label_ms"][3] += 1
    with pytest.raises(RuntimeError):
        R.scored_runs(lab)


def test_causal_stepper_sets_the_packet_k_before_each_step():
    seen = []

    class Net:
        def step(self, ctx, end, prev):
            seen.append(int(TC._STATE["k"][0]))
            return prev

    class Ctx:
        offsets = np.arange(1000, dtype=np.int64) * 100            # 100 events per ms
    step = R.stepper(Net(), True)
    for end in (49, 99, 549):
        step(Ctx(), end, np.zeros(51, np.float32))
    TC.set_k(None)
    assert seen == [1, 3, 3]                                      # nothing before the first packet; then 5000 events
    assert R.stepper(Net(), False).__self__ is not None           # non-causal: the network's own step


def test_initial_noise_is_per_run_and_reproducible():
    cfg = {"TRACK": {"PREV_NOISE_T": 0.01, "PREV_NOISE_R": 0.05, "PREV_NOISE_POSE": 0.05}}
    a, b, c = R.init_noise(cfg, "2", 0), R.init_noise(cfg, "2", 0), R.init_noise(cfg, "2", 1)
    assert np.array_equal(a, b) and not np.array_equal(a, c) and a.shape == (51,)


@pytest.mark.skipif(not (R.ERH / "data" / "erh_2").exists(), reason="converted EvRealHands trial 2 missing")
def test_gt_mask_keeps_a_consistent_subset():
    from tools.tpd import erh_convert as EC
    ctx = EC.load("2", "cpu")
    m = R.gt_mask_context(ctx)
    off = np.asarray(m.offsets)
    assert len(off) == len(ctx.offsets) and off[0] == 0 and np.all(np.diff(off) >= 0)
    assert off[-1] == len(m.events) == len(m.tsub) and 0 < len(m.events) < int(ctx.offsets[-1])
    # every kept event of one labelled window lies in that label's dilated silhouette
    i = 10
    ms = int(ctx.labels["label_ms"][i])
    ev = np.asarray(m.events[off[ms]:off[ms + 1]])
    _, V = EC.fk51(ctx.labels["params51"][i:i + 1], ctx.betas.view(-1).numpy())
    K = ctx.K.view(3, 3).numpy().astype(np.float64) * EC.RENDER_SCALE
    uv = np.stack([K[0, 0] * V[0, :, 0] / V[0, :, 2] + K[0, 2], K[1, 1] * V[0, :, 1] / V[0, :, 2] + K[1, 2]], -1)
    sil = EC.silhouette(uv, EC.get_mano().f.numpy(), R.MASK_DILATE)
    assert len(ev) and sil[ev[:, 1].astype(int), ev[:, 0].astype(int)].all()
