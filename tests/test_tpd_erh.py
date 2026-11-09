"""TPD ERH (EvRealHands -> repo format) converter CPU tests (tools/tpd/erh_convert.py). Run with CUDA_VISIBLE_DEVICES="" in a
process of its own:  CUDA_VISIBLE_DEVICES="" python -m pytest -q -p no:cacheprovider tests/test_tpd_erh.py

- the camera-frame transform of MANO parameters equals rigidly transforming the FK output (our ManoLayer), and the about-root
  translation is what makes it so; with a trial-constant shape replacing per-frame shapes the wrist stays where it was;
- the stored `camera_K` times the model's 0.375 is the 240 x 180 projection, and equals floor-binning of the K projection on
  the 346 x 260 frame within 1 px;
- offsets / tsub reproduce the microsecond timestamps; ms indexing matches `semkine.eval_track._window_events`;
- events dropped by the undistortion bound are dropped consistently (offsets match the kept rows);
- the per-pixel undistortion table equals calling cv2 directly (other environment), and the stored events of a real trial
  equal an independent per-event recomputation;
- silhouette, cross-correlation lag sign, label runs, loader round trip.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd import erh_convert as E  # noqa: E402

RAW_OK = (E.RAW / "53" / "event.aedat4").exists() and Path(E.CV2_PY).exists()
CONVERTED_OK = (E.DATA / "erh_53" / "erh_53_events.npy").exists() and (E.DATA / "erh_0" / "erh_0_labels.npz").exists()


def _identity_und():
    ys, xs = np.mgrid[0:E.SRC_H, 0:E.SRC_W]
    return np.stack([xs, ys], -1).astype(np.float32)


def test_camera_transform_equals_rigid_transform():
    from scipy.spatial.transform import Rotation as Rot
    rng = np.random.default_rng(0)
    n = 6
    rot = rng.normal(size=(n, 3)) * 0.7
    hp = rng.normal(size=(n, 45)) * 0.3
    betas = np.tile(rng.normal(size=(1, 10)), (n, 1))
    trans = rng.normal(size=(n, 3)) * 0.1 + np.array([0.0, 0.0, 0.8])
    R = Rot.random(random_state=1).as_matrix()
    T = rng.normal(size=3)
    hm = E.hands_mean()
    J0 = E.root_joint_ours(betas)
    jw, vw = E.fk_raw(betas, rot, hp + hm, trans)                         # world frame
    rot_n, tr_n = E.mano_world_to_camera(rot, trans, J0, R, T)
    jc, vc = E.fk_raw(betas, rot_n, hp + hm, tr_n)                        # event camera frame
    assert np.abs(vc - (vw @ R.T + T)).max() < 2e-5
    assert np.abs(jc - (jw @ R.T + T)).max() < 2e-5
    # the naive translation (R trans + T, rotation not about the root joint) is wrong by centimetres
    _, vn = E.fk_raw(betas, rot_n, hp + hm, trans @ R.T + T)
    assert np.abs(vn - (vw @ R.T + T)).max() > 5e-3
    # through the 51-D state and the evaluator's decode the result is the same
    p51 = np.concatenate([tr_n, rot_n, hp], 1).astype(np.float32)
    j51, v51 = E.fk51(p51, betas[0])
    assert np.abs(v51 - vc).max() < 1e-5


def test_trial_constant_shape_keeps_the_wrist():
    from scipy.spatial.transform import Rotation as Rot
    rng = np.random.default_rng(2)
    n = 5
    rot, hp = rng.normal(size=(n, 3)) * 0.5, rng.normal(size=(n, 45)) * 0.3
    shapes_f = rng.normal(size=(n, 10)) * 1.5
    beta_t = np.median(shapes_f, 0)
    trans = rng.normal(size=(n, 3)) * 0.05 + np.array([0.1, 0.0, 0.9])
    R, T = Rot.random(random_state=3).as_matrix(), np.array([0.5, 0.04, -0.05])
    hm = E.hands_mean()
    jw, _ = E.fk_raw(shapes_f, rot, hp + hm, trans)                       # each frame with its own shape
    rj_f = E.root_joint_ours(shapes_f)
    j0_t = E.root_joint_ours(beta_t[None])[0]
    rot_n, tr_n = E.mano_world_to_camera(rot, trans, rj_f, R, T, target_root=j0_t)
    jc, _ = E.fk_raw(beta_t, rot_n, hp + hm, tr_n)                        # trial shape
    assert np.abs(jc[:, 0] - (jw[:, 0] @ R.T + T)).max() < 2e-5           # wrist is preserved exactly
    # the reference formula (own root joint) would move it
    rot_r, tr_r = E.mano_world_to_camera(rot, trans, rj_f, R, T)
    jr, _ = E.fk_raw(beta_t, rot_r, hp + hm, tr_r)
    assert np.abs(jr[:, 0] - (jw[:, 0] @ R.T + T)).max() > 1e-3


def test_root_joint_of_our_layer_is_the_posed_wrist_at_zero_translation():
    betas = np.random.default_rng(3).normal(size=(2, 10))
    j, _ = E.fk_raw(betas, np.array([[0.3, -0.2, 0.5]] * 2), np.zeros((2, 45)) + E.hands_mean(), np.zeros((2, 3)))
    assert np.abs(j[:, 0] - E.root_joint_ours(betas)).max() < 1e-6        # rotation is about J0 (smplx convention)


@pytest.mark.parametrize("K", [np.array([[298.67, 0, 172.22], [0, 301.0, 111.77], [0, 0, 1.0]]),
                               np.array([[311.93, 0, 174.14], [0, 313.81, 115.98], [0, 0, 1.0]])])
def test_stored_camera_K_times_render_scale_is_the_grid_projection(K):
    stored = E.stored_camera_K(K)
    Km = stored.astype(np.float64) * E.RENDER_SCALE
    assert abs(Km[0, 0] - K[0, 0] * 240 / 346) < 1e-3 and abs(Km[1, 1] - K[1, 1] * 180 / 260) < 1e-3
    assert abs(Km[0, 2] - K[0, 2] * 240 / 346) < 1e-3 and abs(Km[1, 2] - K[1, 2] * 180 / 260) < 1e-3
    rng = np.random.default_rng(4)
    P = rng.uniform([-0.2, -0.15, 0.7], [0.2, 0.15, 1.2], size=(2000, 3))
    u_model = Km[0, 0] * P[:, 0] / P[:, 2] + Km[0, 2]
    v_model = Km[1, 1] * P[:, 1] / P[:, 2] + Km[1, 2]
    xu = K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2]                            # 346 x 260 frame
    yu = K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]
    bx, by = np.floor(xu * E.SX), np.floor(yu * E.SY)
    assert np.all((u_model - bx > -1e-3) & (u_model - bx < 1 + 1e-3))     # within 1 px (the floor drops the fraction)
    assert np.all((v_model - by > -1e-3) & (v_model - by < 1 + 1e-3))


def test_binning_formula_floor_and_bounds():
    und = _identity_und()
    und[10, 10] = [345.0, 259.0]                                          # last legal position
    und[11, 11] = [345.4, 100.0]                                          # beyond the x bound
    und[12, 12] = [-0.2, 100.0]
    und[13, 13] = [100.0, 259.6]                                          # beyond the y bound (reference rule would keep it)
    tab = E.bin_tables(und)
    assert tab["bx"][10, 10] == 239 and tab["by"][10, 10] == 179 and tab["legal"][10, 10]
    assert not tab["legal"][11, 11] and not tab["legal"][12, 12] and not tab["legal"][13, 13]
    assert tab["ref_legal"][13, 13] and not tab["ref_legal"][11, 11]      # the reference keeps (and clips) y up to 345
    assert tab["bx"][50, 200] == int(np.floor(200 * 240 / 346)) and tab["by"][50, 200] == int(np.floor(50 * 180 / 260))


def test_offsets_tsub_roundtrip_and_window_events():
    from semkine import eval_track as ET
    rng = np.random.default_rng(1)
    n = 6000
    t = np.sort(rng.integers(5_000_000, 5_400_000, n)).astype(np.int64)
    x = rng.integers(0, E.SRC_W, n).astype(np.int16)
    y = rng.integers(0, E.SRC_H, n).astype(np.int16)
    p = rng.integers(0, 2, n).astype(np.int8)
    perm = np.arange(n)
    perm[100:140] = perm[100:140][::-1]                                   # packets out of order, as in the DAVIS files
    und = _identity_und()
    und[200:210, 300:310] = [400.0, 400.0]                                # an out-of-bounds patch: dropped
    tab = E.bin_tables(und)
    out = E.bin_events(t[perm], x[perm], y[perm], p[perm], tab)
    ev, off, tsub = out["events"], out["offsets"], out["tsub"]
    tp, xp, yp, pp = t[perm], x[perm], y[perm], p[perm]
    order = np.argsort(tp, kind="stable")
    ts, xs, ys, ps = tp[order], xp[order], yp[order], pp[order]
    keep = tab["legal"][ys, xs]
    assert out["stats"]["n_unsorted_diffs"] > 0 and (~keep).any() and len(ev) == keep.sum()
    t_rel = ts - ts[0]
    n_ms = int(t_rel[-1] // 1000) + 1
    assert len(off) == n_ms + 1 and off[0] == 0 and off[-1] == len(ev) and np.all(np.diff(off) >= 0)
    ms_of_row = np.repeat(np.arange(n_ms), np.diff(off))
    assert np.array_equal(ms_of_row * 1000 + tsub.astype(np.int64), t_rel[keep])
    assert np.array_equal(ev[:, 0], np.floor(xs[keep].astype(np.float64) * 240 / 346).astype(np.uint8))
    assert np.array_equal(ev[:, 1], np.floor(ys[keep].astype(np.float64) * 180 / 260).astype(np.uint8))
    assert np.array_equal(ev[:, 2], ps[keep].astype(np.uint8))
    # ms indexing agrees with the evaluator's packet [end - 49, end]
    for end in (60, 150, n_ms - 1):
        w = ET._window_events(ev, off, tsub, end, 50)
        start = end - 49
        sel = (t_rel[keep] >= start * 1000) & (t_rel[keep] < (end + 1) * 1000)
        assert len(w) == sel.sum()
        assert np.array_equal(w[:, 1], ev[sel, 0]) and np.array_equal(w[:, 2], ev[sel, 1])
        assert np.allclose(w[:, 3], (t_rel[keep][sel] - start * 1000) * 1e-6, atol=1e-7)
        assert np.array_equal(w[:, 4], ev[sel, 2])


def test_label_runs_and_true_runs():
    assert E.label_runs([1, 2, 3, 7, 8, 10]) == [(0, 2), (3, 4), (5, 5)]
    assert E.label_runs([]) == []
    assert E.true_runs([0, 1, 1, 0, 1]) == [(1, 2), (4, 4)] and E.true_runs([0, 0]) == []


def test_xcorr_lag_sign_and_speed_series():
    rng = np.random.default_rng(5)
    base = np.convolve(rng.normal(size=5000), np.ones(60) / 60, mode="same")
    rate = np.concatenate([np.zeros(25), base[:-25]])                     # rate[m] = speed[m - 25]: the events lag by 25 ms
    L, c = E.xcorr_lag(rate + 0.01 * rng.normal(size=5000), base)
    assert L[int(np.nanargmax(c))] == 25
    # speed series: constant motion -> constant speed over the interpolated span; gaps in frame ids stay NaN
    v = np.zeros((6, 4, 3))
    v[:, :, 0] = np.arange(6)[:, None] * 0.01                             # 1 cm per label step
    t_us = 1e6 / 15 * np.arange(6) + 3000.0
    ids = np.array([0, 1, 2, 4, 5, 6])                                    # a gap between the 3rd and 4th label
    sp = E.speed_series(v, t_us, ids, 500)
    assert np.isfinite(sp).any() and np.allclose(sp[np.isfinite(sp)][:5], 0.01 * 15, rtol=1e-6)
    gap_ms = int((t_us[2] + t_us[3]) / 2000)
    assert np.isnan(sp[gap_ms])


def test_silhouette_covers_the_projected_mesh():
    j, v = E.fk_raw(np.zeros(10), np.zeros((1, 3)), np.zeros((1, 45)) + E.hands_mean(), np.array([[0.0, 0.0, 0.9]]))
    K = E.stored_camera_K(np.array([[298.67, 0, 172.22], [0, 301.0, 111.77], [0, 0, 1.0]])).astype(np.float64) * E.RENDER_SCALE
    uv = E.project_grid(v[0], K)
    faces = E.get_mano().f.numpy()
    m0 = E.silhouette(uv, faces, dilate=0)
    ix, iy = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
    assert m0[iy, ix].all()                                               # every vertex bin is inside
    assert 100 < m0.sum() < 3000
    m2 = E.silhouette(uv, faces, dilate=2)
    assert m2[m0].all() and m2.sum() > m0.sum()
    ev = np.stack([ix, iy, np.ones_like(ix)], 1).astype(np.uint8)
    assert E.inside_fraction(m0, ev) == (len(ev), len(ev))
    assert E.inside_fraction(m0, ev, shift=(60, 0))[0] < len(ev) // 2     # a far shift loses most of them
    sh = E.shift_search(m0, ev)
    iy_, ix_ = np.unravel_index(int(np.argmax(sh)), sh.shape)
    assert (ix_ - E.SHIFT_RANGE, iy_ - E.SHIFT_RANGE) == (0, 0)


def test_loader_roundtrip_synthetic(tmp_path):
    n_ms = 400
    events = np.zeros((10, 3), np.uint8)
    offsets = np.zeros(n_ms + 1, np.int64)
    offsets[1:] = np.minimum(np.arange(1, n_ms + 1), 10)
    tsub = np.arange(10, dtype=np.uint16)
    label_ms = np.array([50, 117, 183])
    labels = {"frame_id": np.array([3, 4, 5]), "label_t_us": label_ms * 1000.0, "label_ms": label_ms, "tier": np.array("3d"),
              "params51": np.arange(3 * 51, dtype=np.float32).reshape(3, 51), "betas_trial": np.zeros(10, np.float32)}
    aux = {"betas": np.arange(10, dtype=np.float32), "camera_K": E.stored_camera_K(np.array([[300.0, 0, 170], [0, 300, 110], [0, 0, 1]])),
           "valid_runs_ms": np.array([[50, 184]]), "category": np.array("erh_3d")}
    E.write_trial_files(7, tmp_path, events, offsets, tsub, labels, aux, {"trial": 7})
    ctx = E.load(7, "cpu", root=tmp_path)
    for name in ("seq", "events", "offsets", "tsub", "aux", "gt", "betas", "K"):
        assert hasattr(ctx, name)
    assert ctx.seq == "erh_7" and ctx.gt.shape == (n_ms, 51) and ctx.betas.shape == (1, 10) and ctx.K.shape == (1, 3, 3)
    ok = ~np.isnan(ctx.gt).all(1)
    assert np.array_equal(np.flatnonzero(ok), label_ms) and np.array_equal(ctx.gt[label_ms], labels["params51"])
    assert np.isnan(ctx.gt[ok == 0]).all() and ctx.labels["tier"] == "3d"
    assert float(ctx.K[0, 0, 0]) == pytest.approx(300.0 * 240 / 346 / 0.375, rel=1e-5)


@pytest.mark.skipif(not RAW_OK, reason="raw ERH data or the cv2 environment is not available")
def test_lut_equals_direct_cv2_call():
    a = E.read_annot(53)
    cam = E.camera_of(a)
    lut = E.build_lut(cam["K_old"], cam["dist"], cam["K_new"])
    rng = np.random.default_rng(6)
    xs, ys = rng.integers(0, E.SRC_W, 300), rng.integers(0, E.SRC_H, 300)
    direct = E.undistort_points(np.stack([xs, ys], 1), cam["K_old"], cam["dist"], cam["K_new"])
    assert np.array_equal(direct, lut[ys, xs])                            # (row = y, col = x), float32 identical


@pytest.mark.skipif(not (RAW_OK and CONVERTED_OK), reason="converted ERH data is not available")
def test_stored_events_equal_an_independent_recomputation():
    a = E.read_annot(53)
    cam = E.camera_of(a)
    t, x, y, p = E.read_events(E.RAW / "53" / "event.aedat4")
    o = np.argsort(t, kind="stable")
    t, x, y, p = t[o], x[o], y[o], p[o]
    t_rel = t - t[0]
    b = E.trial_base(53)
    events = np.load(f"{b}_events.npy", mmap_mode="r")
    offsets = np.load(f"{b}_offsets.npy")
    tsub = np.load(f"{b}_tsub.npy", mmap_mode="r")
    sel = np.flatnonzero((t_rel >= 6000_000) & (t_rel < 6200_000))
    und = E.undistort_points(np.stack([x[sel], y[sel]], 1), cam["K_old"], cam["dist"], cam["K_new"]).astype(np.float64)
    ok = (und[:, 0] >= 0) & (und[:, 0] <= E.SRC_W - 1) & (und[:, 1] >= 0) & (und[:, 1] <= E.SRC_H - 1)
    exp = np.stack([np.clip(np.floor(und[ok, 0] * 240 / 346), 0, 239), np.clip(np.floor(und[ok, 1] * 180 / 260), 0, 179), np.clip(p[sel][ok], 0, 1)], 1)
    a0, a1 = int(offsets[6000]), int(offsets[6200])
    assert a1 - a0 == ok.sum()
    assert np.array_equal(np.asarray(events[a0:a1]).astype(np.int64), exp.astype(np.int64))
    us = np.repeat(np.arange(6000, 6200), np.diff(offsets[6000:6201])) * 1000 + np.asarray(tsub[a0:a1]).astype(np.int64)
    assert np.array_equal(us, t_rel[sel][ok])


@pytest.mark.skipif(not CONVERTED_OK, reason="converted ERH data is not available")
def test_converted_trial_invariants_and_loader():
    ctx = E.load(0)
    assert ctx.events.shape[1] == 3 and ctx.events.dtype == np.uint8 and ctx.tsub.dtype == np.uint16
    assert ctx.events[:, 0].max() < E.GRID_W and ctx.events[:, 1].max() < E.GRID_H and set(np.unique(ctx.events[:2000000, 2])) <= {0, 1}
    assert ctx.offsets.dtype == np.int64 and ctx.offsets[-1] == len(ctx.events) and np.all(np.diff(ctx.offsets) >= 0)
    lab = ctx.labels
    assert np.all(np.diff(lab["label_ms"]) > 0) and lab["label_ms"].max() < len(ctx.offsets) - 1
    assert ctx.gt.shape == (len(ctx.offsets) - 1, 51)
    assert np.isfinite(ctx.gt[lab["label_ms"]]).all() and np.isnan(ctx.gt).all(1).sum() == len(ctx.gt) - len(lab["label_ms"])
    # the stored 51-D state reproduces the stored annotation joints within the conversion tolerance (MANO fit error)
    j, _ = E.fk51(lab["params51"][::20], lab["betas_trial"])
    assert np.median(E.joint_error_mm(j, lab["joints3d_cam"][::20])) < 10.0
    # the loader's K reproduces the grid projection of the chosen annotation K
    Ks = ctx.aux["K_src"] if str(ctx.aux["camera_K_choice"]) == "K" else ctx.aux["K_new_src"]
    assert float(ctx.K[0, 0, 0]) * E.RENDER_SCALE == pytest.approx(float(Ks[0, 0]) * E.SX, rel=1e-5)
