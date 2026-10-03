"""DT round, jitter attribution: the contract of `evalx.jitter_decomp` on synthetic motions, the Shapley
attribution of the error acceleration to the three blocks of the state, the aggregation / CLI of
`tools/dt/jitter_offline.py` (diverged runs, identical evaluations, the old-evaluator comparison, the contrast table),
and a regression on the recorded closed-loop run `rt_cnntrack_s3407`.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python -m pytest tests/test_dt_jitter.py -q
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as Rot

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools"), str(REPO / "tools" / "tracking"),
                str(REPO / "tools" / "dt")]
import evalx as EX                                                            # noqa: E402
import jitter_offline as JO                                                   # noqa: E402
from mano_layer import ManoLayer                                              # noqa: E402

DEV = torch.device("cpu")
N = 240
BETAS = np.zeros(10, np.float32)
#: two segments of 120 steps: the boundary sits between step 119 and 120
RUN = np.concatenate([np.zeros(120, int), np.ones(120, int)])
BLOCKS = ("transl", "root", "fingers")
SL = {"transl": slice(0, 3), "root": slice(3, 6), "fingers": slice(6, 51)}

RECORDED_RUN = REPO / "outputs" / "semkine" / "rt_cnntrack_s3407"
RECORDED_NPZ = RECORDED_RUN / "evalx_val_core_last_tf_pert.npz"
HIST_RUN = REPO / "outputs" / "hand_data51" / "track_render51_dr_so3fk"


@pytest.fixture(scope="module")
def mano():
    return ManoLayer(str(REPO / "assets" / "mano_right.npz"), add_mean=False).to(DEV).eval()


def _gt():
    """A moving hand: constant velocity in x (2 mm / step), a slowly turning root, breathing fingers."""
    t = np.arange(N, dtype=np.float32)
    gt = np.zeros((N, 51), np.float32)
    gt[:, 2] = 0.45
    gt[:, 0] = 0.002 * t
    gt[:, 3:6] = np.array([1.5, 1.0, 1.3]) + 0.01 * t[:, None] * np.array([1.0, 0.0, 0.0])
    gt[:, 6:] = 0.05 * np.sin(0.05 * t)[:, None]
    return gt


def _decomp(mano, pred, gt=None, run=RUN):
    gt = _gt() if gt is None else gt
    return JO.add_shapley(EX.jitter_decomp(mano, {"pred": np.asarray(pred, np.float32), "gt": gt, "run": run,
                                                  "betas": BETAS}, DEV))


# ------------------------------------------------------------------ jitter_decomp contract (synthetic)
def test_perfect_tracking_has_zero_error_acceleration_and_unit_jitter_ratio(mano):
    j = _decomp(mano, _gt().copy())
    assert j["acc_err_mm"] == pytest.approx(0.0, abs=1e-4)
    assert j["jit_ratio"] == pytest.approx(1.0, abs=1e-6) and j["acc_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert j["jit_pred_mm"] > 1.0 and j["acc_gt_mm"] > 0.0           # the ground truth does move and accelerate
    for b in BLOCKS:
        assert j[f"acc_err_only_{b}_mm"] == pytest.approx(0.0, abs=1e-4)
        assert j[f"acc_err_allbut_{b}_mm"] == pytest.approx(0.0, abs=1e-4)
        assert j[f"shap_{b}_mm"] == pytest.approx(0.0, abs=1e-4)
    assert j["rot_acc_pred_deg"] == j["rot_acc_gt_deg"]


def test_constant_bias_adds_no_error_acceleration(mano):
    gt = _gt()
    ref = _decomp(mano, gt + np.random.default_rng(1).normal(0, 0.003, gt.shape).astype(np.float32) * (np.arange(51) < 3))
    assert ref["acc_err_mm"] > 5.0                                    # what real error acceleration looks like here
    # a translation offset
    p = gt.copy()
    p[:, :3] += np.array([0.01, -0.02, 0.03], np.float32)
    j = _decomp(mano, p)
    assert j["acc_err_mm"] < 1e-3 and j["jit_pred_mm"] == pytest.approx(j["jit_gt_mm"], abs=1e-3)
    # an offset in every block of a static hand: the joint error is constant, whatever the pose
    static = np.tile(gt[:1], (N, 1))
    p = static.copy()
    p[:, 0] += 0.01
    p[:, 3:6] += 0.05
    p[:, 6:] += 0.02
    j = _decomp(mano, p, gt=static)
    assert j["acc_err_mm"] < 1e-4 and j["jit_pred_mm"] < 1e-4
    # a finger offset on the moving hand: only the (small) pose dependence of the kinematics, 4e-4 mm against 12 mm of noise
    p = gt.copy()
    p[:, 0] += 0.01
    p[:, 6:] += 0.02
    j = _decomp(mano, p)
    assert j["acc_err_mm"] < 5e-3 and abs(j["jit_pred_mm"] - j["jit_gt_mm"]) < 0.05
    assert j["acc_err_mm"] < 1e-3 * ref["acc_err_mm"]


def test_translation_noise_is_attributed_to_the_translation_block(mano):
    rng = np.random.default_rng(0)
    p = _gt().copy()
    p[:, :3] += rng.normal(0, 0.003, (N, 3)).astype(np.float32)
    j = _decomp(mano, p)
    assert j["acc_err_mm"] > 5.0
    assert j["shap_transl_mm"] == pytest.approx(j["acc_err_mm"], rel=1e-6)
    assert abs(j["shap_root_mm"]) < 1e-6 and abs(j["shap_fingers_mm"]) < 1e-6
    assert j["shap_share_transl"] == pytest.approx(1.0, abs=1e-6)
    assert j["acc_err_only_transl_mm"] > 0.98 * j["acc_err_mm"]
    assert j["acc_err_only_root_mm"] < 1e-3 and j["acc_err_only_fingers_mm"] < 1e-3
    assert j["acc_err_allbut_transl_mm"] < 1e-3                       # take the translation back and the noise is gone
    assert JO.dominant(j) == "transl"


def test_root_noise_is_attributed_to_the_root_block(mano):
    rng = np.random.default_rng(1)
    p = _gt().copy()
    p[:, 3:6] += rng.normal(0, 0.02, (N, 3)).astype(np.float32)
    j = _decomp(mano, p)
    assert j["acc_err_mm"] > 1.0
    assert j["shap_root_mm"] == pytest.approx(j["acc_err_mm"], rel=1e-6)
    assert abs(j["shap_transl_mm"]) < 1e-6 and abs(j["shap_fingers_mm"]) < 1e-6
    assert JO.dominant(j) == "root"


def test_finger_noise_is_attributed_to_the_finger_block(mano):
    rng = np.random.default_rng(2)
    p = _gt().copy()
    p[:, 6:] += rng.normal(0, 0.05, (N, 45)).astype(np.float32)
    j = _decomp(mano, p)
    assert j["acc_err_mm"] > 1.0
    assert j["acc_err_only_fingers_mm"] > 0.9 * j["acc_err_mm"] and j["acc_err_only_root_mm"] < 1e-3
    assert j["shap_fingers_mm"] == pytest.approx(j["acc_err_mm"], rel=1e-6)
    assert abs(j["shap_transl_mm"]) < 1e-6 and abs(j["shap_root_mm"]) < 1e-6
    assert JO.dominant(j) == "fingers"


def test_a_jump_exactly_at_a_segment_boundary_changes_nothing(mano):
    gt = _gt()
    p = gt.copy()
    p[120:, :3] += 0.5                                                # the whole second segment is 0.5 m off
    j = _decomp(mano, p)
    assert j["acc_err_mm"] < 1e-3
    assert abs(j["jit_pred_mm"] - j["jit_gt_mm"]) < 1e-3
    assert j["n_steps"] == 238                                        # 240 frames, 2 segments
    # positive control: the same jump one step inside the segment is seen at once
    p = gt.copy()
    p[121:, :3] += 0.5
    k = _decomp(mano, p)
    assert k["acc_err_mm"] > 1.0 and k["jit_pred_mm"] > j["jit_gt_mm"] + 1.0


def test_one_segment_per_frame_leaves_nothing_to_measure(mano):
    j = EX.jitter_decomp(mano, {"pred": _gt(), "gt": _gt(), "run": np.arange(N), "betas": BETAS}, DEV)
    assert j["n_steps"] == 0 and math.isnan(j["acc_err_mm"]) and math.isnan(j["jit_gt_mm"])
    assert not any(k.startswith("shap_") for k in JO.add_shapley(dict(j)))   # no attribution, and no crash


def test_root_angular_acceleration_of_constant_angular_velocity_is_zero_and_of_a_kink_is_not():
    t = np.arange(N, dtype=np.float64)
    # R_t = R_0 Exp(t w): constant angular velocity. The second case starts at 3.0 rad and turns on through pi,
    # where `as_rotvec` flips to the opposite axis: the quaternion difference has no seam there
    for r0, w in ((Rot.from_rotvec([0.4, -0.3, 0.2]), np.array([0.01, 0.02, -0.015])),
                  (Rot.from_rotvec([3.0, 0.0, 0.0]), np.array([0.004, 0.0, 0.0]))):
        aa = (r0 * Rot.from_rotvec(t[:, None] * w)).as_rotvec()
        assert EX._rot_second_diff_deg(aa).max() < 1e-4
    # a rotation about one fixed axis through angles 2.9 .. 3.4 rad written as plain axis-angle (|aa| > pi allowed)
    aa = np.arange(2.9, 3.4, 0.005)[:, None] * np.array([0.0, 1.0, 0.0])
    assert EX._rot_second_diff_deg(aa).max() < 1e-4
    # a kink: the angular velocity changes from 1 to 3 deg per step between the steps 99 -> 100 and 100 -> 101
    ang = np.where(t < 100, t * 1.0, 100.0 + (t - 100) * 3.0)
    aa = np.deg2rad(ang)[:, None] * np.array([0.0, 0.0, 1.0])
    acc = EX._rot_second_diff_deg(aa)
    assert acc[99] == pytest.approx(2.0, abs=1e-6) and np.argmax(acc) == 99
    assert (np.delete(acc, 99) < 1e-6).all()


def test_root_angular_acceleration_reaches_the_decomposition(mano):
    t = np.arange(N, dtype=np.float64)
    gt = _gt()
    gt[:, 3:6] = (Rot.from_rotvec([0.4, -0.3, 0.2]) * Rot.from_rotvec(t[:, None] * np.array([0.01, 0.02, -0.015]))
                  ).as_rotvec().astype(np.float32)
    j = _decomp(mano, gt.copy(), gt=gt)
    assert j["rot_acc_gt_deg"] < 1e-3 and j["rot_acc_pred_deg"] == j["rot_acc_gt_deg"]
    kink = gt.copy()
    kink[100:, 3] += 0.2
    assert _decomp(mano, kink, gt=gt)["rot_acc_pred_deg"] > j["rot_acc_gt_deg"] + 0.01


# ------------------------------------------------------------------ Shapley
def _task_formula(v, x):
    """The attribution exactly as the task writes it, for the block x and the other two y, z."""
    y, z = [b for b in BLOCKS if b != x]
    f = frozenset
    return (1 / 3 * (v[f([x])] - v[f()]) + 1 / 6 * (v[f([x, y])] - v[f([y])]) + 1 / 6 * (v[f([x, z])] - v[f([z])])
            + 1 / 3 * (v[f(BLOCKS)] - v[f([y, z])]))


def test_shapley_of_known_games():
    def game(fn):
        return {frozenset(s): fn(frozenset(s)) for k in range(4) for s in itertools.combinations(BLOCKS, k)}
    additive = game(lambda s: sum({"transl": 1.5, "root": -0.5, "fingers": 4.0}[b] for b in s))
    assert JO.shapley(additive) == pytest.approx({"transl": 1.5, "root": -0.5, "fingers": 4.0})
    unanimity = game(lambda s: 1.0 if len(s) == 3 else 0.0)           # all three needed: equal split
    assert JO.shapley(unanimity) == pytest.approx({b: 1 / 3 for b in BLOCKS})
    dictator = game(lambda s: 1.0 if "root" in s else 0.0)            # one block does everything, the others are null
    assert JO.shapley(dictator) == pytest.approx({"transl": 0.0, "root": 1.0, "fingers": 0.0})
    pair = game(lambda s: 6.0 if {"transl", "fingers"} <= s else 0.0)  # two complements share, the third is null
    assert JO.shapley(pair) == pytest.approx({"transl": 3.0, "root": 0.0, "fingers": 3.0})
    rng = np.random.default_rng(0)
    rand = game(lambda s: 0.0 if not s else float(rng.uniform(0, 10)))
    phi = JO.shapley(rand)
    assert sum(phi.values()) == pytest.approx(rand[frozenset(BLOCKS)], abs=1e-12)            # efficiency
    for x in BLOCKS:
        assert phi[x] == pytest.approx(_task_formula(rand, x), abs=1e-12)                    # the task's formula


def test_coalition_values_follow_the_task_definition():
    j = {"acc_err_mm": 9.0}
    for i, b in enumerate(BLOCKS):
        j[f"acc_err_only_{b}_mm"] = 1.0 + i
        j[f"acc_err_allbut_{b}_mm"] = 5.0 + i
    v = JO.coalition_values(j)
    assert len(v) == 8 and v[frozenset()] == 0.0 and v[frozenset(BLOCKS)] == 9.0
    assert v[frozenset(["transl"])] == 1.0 and v[frozenset(["root"])] == 2.0 and v[frozenset(["fingers"])] == 3.0
    assert v[frozenset(["root", "fingers"])] == 5.0                    # all but transl
    assert v[frozenset(["transl", "fingers"])] == 6.0                  # all but root
    assert v[frozenset(["transl", "root"])] == 7.0                     # all but fingers


def test_shapley_values_of_a_real_decomposition_sum_to_the_error_acceleration(mano):
    rng = np.random.default_rng(3)
    p = _gt().copy()
    p[:, :3] += rng.normal(0, 0.002, (N, 3)).astype(np.float32)
    p[:, 3:6] += rng.normal(0, 0.03, (N, 3)).astype(np.float32)
    p[:, 6:] += rng.normal(0, 0.04, (N, 45)).astype(np.float32)
    j = _decomp(mano, p)
    phi = {b: j[f"shap_{b}_mm"] for b in BLOCKS}
    assert abs(sum(phi.values()) - j["acc_err_mm"]) < 1e-6
    assert sum(j[f"shap_share_{b}"] for b in BLOCKS) == pytest.approx(1.0, abs=1e-9)
    v = JO.coalition_values(j)
    for b in BLOCKS:
        assert phi[b] == pytest.approx(_task_formula(v, b), abs=1e-9)
    assert min(phi.values()) > 0.0                                      # all three blocks are noisy: all are charged


def test_shapley_efficiency_is_asserted(monkeypatch):
    j = {"acc_err_mm": 3.0}
    for b in BLOCKS:
        j[f"acc_err_only_{b}_mm"] = 1.0
        j[f"acc_err_allbut_{b}_mm"] = 2.0
    out = JO.add_shapley(dict(j))                                       # a genuine game: efficiency holds by construction
    assert [out[f"shap_{b}_mm"] for b in BLOCKS] == pytest.approx([1.0, 1.0, 1.0], abs=1e-12)
    assert [out[f"shap_share_{b}"] for b in BLOCKS] == pytest.approx([1 / 3] * 3, abs=1e-12)
    monkeypatch.setattr(JO, "shapley", lambda v, players=BLOCKS: {b: 2.0 for b in players})   # a broken attribution
    with pytest.raises(AssertionError):
        JO.add_shapley(dict(j))


# ------------------------------------------------------------------ aggregation and CLI
def test_combine_is_frame_weighted_and_leaves_out_sequences_lacking_a_key():
    def seq(n, jit, static=None):
        d = {"n_frames": n, "n_steps": n - 1, "n_static": 0, "n_segments": 1, "jit_gt_mm": jit, "jit_pred_mm": 2 * jit,
             "acc_gt_mm": 1.0, "acc_pred_mm": 2.0, "acc_err_mm": 3.0}
        for b in BLOCKS:
            d[f"acc_err_only_{b}_mm"] = 1.0
            d[f"acc_err_allbut_{b}_mm"] = 2.0
        if static is not None:
            d["jit_pred_static_mm"] = static
        return JO.add_shapley(d)
    o = JO.combine({"a": seq(300, 4.0, static=20.0), "b": seq(100, 8.0)})
    assert o["jit_gt_mm"] == pytest.approx((300 * 4.0 + 100 * 8.0) / 400)
    assert o["jit_pred_static_mm"] == 20.0                              # not diluted by the sequence without the key
    assert o["n_frames"] == 400 and o["n_steps"] == 398
    assert o["jit_ratio_pooled"] == pytest.approx(2.0) and o["acc_ratio_pooled"] == pytest.approx(2.0)
    assert sum(o[f"shap_{b}_mm"] for b in BLOCKS) == pytest.approx(o["acc_err_mm"], abs=1e-9)


def _fake_run(root, name, seed, drift=False, npz_name="evalx_val_core_last_tf_pert.npz"):
    """A run directory with the arrays of two sequences: A (two segments, noisy; with `drift` the predicted translation
    runs away at 200 mm per step) and B (one segment, perfect)."""
    d = root / f"{name}_s{seed}"
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    gt = _gt()
    pa = gt.copy()
    if drift:
        pa[:, 2] += 0.2 * np.arange(N, dtype=np.float32)
    else:
        pa[:, :3] += rng.normal(0, 0.002, (N, 3)).astype(np.float32)
        pa[:, 6:] += rng.normal(0, 0.03, (N, 45)).astype(np.float32)
    arrays = {"model|A|pred": pa, "model|A|gt": gt, "model|A|run": RUN, "model|A|mpjpe_ra_mm": np.full(N, 10.0 + seed, np.float32),
              "model|B|pred": gt[:120].copy(), "model|B|gt": gt[:120], "model|B|run": np.zeros(120, int),
              "model|B|mpjpe_ra_mm": np.full(120, 1.0, np.float32), "tf|A|pred": gt, "hold|A|pred": gt}
    np.savez_compressed(d / npz_name, **arrays)
    return d


def _stub_state(monkeypatch, mano, tmp_path):
    """The CPU MANO layer and zero betas instead of the validation data (the worker state of the tool)."""
    state = {"mano": mano, "device": DEV, "static_mm": EX.STATIC_MM, "betas": {}, "dirs": {}, "root": tmp_path}
    monkeypatch.setattr(JO, "_state", lambda *a, **k: state)
    monkeypatch.setattr(JO, "betas_for", lambda st, seq: BETAS)
    return state


def _poison(path, seq, key="pred", rows=slice(100, 110), col=7, value=np.nan):
    """Rewrite an npz with `value` in a block of one array of one sequence."""
    with np.load(path) as zf:
        z = dict(zf)
    a = z[f"model|{seq}|{key}"].copy()
    a[rows, col] = value
    z[f"model|{seq}|{key}"] = a
    np.savez_compressed(path, **z)


def test_cli_on_synthetic_runs(tmp_path, monkeypatch, mano):
    root = tmp_path / "semkine"
    _fake_run(root, "fake", 1)
    _fake_run(root, "fake", 2)
    _fake_run(root, "drift", 1, drift=True)
    (root / "empty_run").mkdir()                                        # a run without an evaluation: skipped
    (root / "fake_notes.json").write_text("{}")                         # a file matching the glob: ignored
    state = {"mano": mano, "device": DEV, "static_mm": EX.STATIC_MM, "betas": {}, "dirs": {}, "root": tmp_path}
    monkeypatch.setattr(JO, "_state", lambda *a, **k: state)
    monkeypatch.setattr(JO, "betas_for", lambda st, seq: BETAS)
    res = JO.main(["--runs", "fake_*", "drift_*", "empty_run", "--runs-root", str(root),
                   "--out-prefix", str(tmp_path / "out" / "rep"), "--jobs", "1", "--threads", "1"])
    assert sorted(res["runs"]) == ["drift_s1", "fake_s1", "fake_s2"] and list(res["meta"]["skipped"]) == ["empty_run"]
    text = (tmp_path / "out" / "rep.json").read_text()
    assert "NaN" not in text and "Infinity" not in text                 # strict JSON
    js = json.loads(text)
    md = (tmp_path / "out" / "rep.md").read_text()
    assert "fake_s1" in md and "fake_s2" in md and "| arm |" in md and "Summary" in md
    # the translation runaway: huge first-difference jitter, no error acceleration, flagged
    dr = js["runs"]["drift_s1"]
    assert dr["transl_runaway"] is True and not js["runs"]["fake_s1"]["transl_runaway"]
    assert dr["overall"]["n_transl_jump"] == 238 and dr["overall"]["transl_step_max_mm"] == pytest.approx(200.0, rel=1e-3)
    assert js["runs"]["fake_s1"]["overall"]["n_transl_jump"] == 0
    assert dr["sequences"]["A"]["jit_pred_mm"] > 100.0 and dr["sequences"]["A"]["acc_err_mm"] < 0.1
    assert js["arms"]["drift"]["transl_runaway_runs"] == ["drift_s1"] and "drift_s1 †" in md
    r1 = js["runs"]["fake_s1"]
    assert sorted(r1["sequences"]) == ["A", "B"] and r1["arm"] == "fake" and r1["npz_tag"] == "last_tf_pert"
    a, b, o = r1["sequences"]["A"], r1["sequences"]["B"], r1["overall"]
    assert a["n_frames"] == N and b["n_frames"] == 120 and o["n_frames"] == N + 120
    assert a["n_steps"] == 238 and b["n_steps"] == 119 and o["n_steps"] == 238 + 119
    assert b["acc_err_mm"] == pytest.approx(0.0, abs=1e-4) and b["jit_ratio"] == pytest.approx(1.0, abs=1e-6)
    for k in ("jit_gt_mm", "jit_pred_mm", "acc_err_mm", "shap_transl_mm", "ra_mpjpe_mm"):
        assert o[k] == pytest.approx((N * a[k] + 120 * b[k]) / (N + 120), rel=1e-9)                # frame-weighted
    assert o["ra_mpjpe_mm"] == pytest.approx((N * 11.0 + 120 * 1.0) / (N + 120))
    assert sum(o[f"shap_share_{k}"] for k in BLOCKS) == pytest.approx(1.0, abs=1e-9)
    arm = js["arms"]["fake"]
    assert arm["n_runs"] == 2 and arm["runs"] == ["fake_s1", "fake_s2"] and arm["nonfinite_runs"] == []
    assert arm["mean"]["acc_err_mm"] == pytest.approx(
        (js["runs"]["fake_s1"]["overall"]["acc_err_mm"] + js["runs"]["fake_s2"]["overall"]["acc_err_mm"]) / 2)
    # the 'tf' and 'hold' arrays of the npz are not analysed
    assert all(set(r["sequences"]) == {"A", "B"} for r in js["runs"].values())
    # nothing diverged, nothing identical: no flags
    assert res["meta"]["nonfinite_runs"] == [] and res["meta"]["identical_evaluations"] == []
    assert "‡" not in md and "NON-FINITE" not in md and not any(r["nonfinite"] for r in js["runs"].values())


def test_dominance_label_marks_near_ties():
    def sh(t, r, f):
        return {"shap_share_transl": t, "shap_share_root": r, "shap_share_fingers": f}
    assert JO.dominance(sh(0.5, 0.3, 0.2)) == "transl"
    assert JO.dominance(sh(0.40, 0.38, 0.22)) == "transl~root"
    assert JO.dominance(sh(0.1, 0.45, 0.45)) == "root~fingers"
    assert JO.dominance({"shap_share_transl": None, "shap_share_root": 0.5, "shap_share_fingers": 0.5}) == "n/a"
    assert JO.dominance({"nonfinite_sequences": ["A"]}) == "non-finite" and JO.dominance({"nonfinite": {"pred": {}}}) == "non-finite"
    assert JO.dominant({"shap_transl_mm": 1.0, "shap_root_mm": 3.0, "shap_fingers_mm": 2.0}) == "root"
    assert JO.dominant({}) is None
    assert JO.runaway({"n_steps": 100, "n_transl_jump": 5}) and not JO.runaway({"n_steps": 100, "n_transl_jump": 4})
    assert not JO.runaway({"n_steps": 100, "n_transl_jump": 50, "nonfinite_sequences": ["A"]})   # a diverged run is not a 'runaway'


def test_run_discovery_prefers_the_named_evaluation_and_names_resolve_under_outputs(tmp_path):
    runs = tmp_path / "outputs" / "semkine"
    (runs / "a_s1").mkdir(parents=True)
    (runs / "a_s2").mkdir()
    (tmp_path / "outputs" / "other" / "hist_run").mkdir(parents=True)
    for n in ("evalx_val_core_last.npz", "evalx_val_core_last_tf_pert.npz", "evalx_val_core_last_tf_pert_gpu.npz",
              "evalx_val_core_selected.npz"):
        (runs / "a_s1" / n).write_bytes(b"")
    (runs / "a_s2" / "evalx_val_core_last.npz").write_bytes(b"")
    assert [p.name for p in JO.resolve_runs(["a_s*"], runs)] == ["a_s1", "a_s2"]
    assert [p.name for p in JO.resolve_runs(["other/hist_run"], runs)] == ["hist_run"]      # under outputs/
    assert [p.name for p in JO.resolve_runs(["a_s1", "a_s*"], runs)] == ["a_s1", "a_s2"]    # no duplicates
    primary, others = JO.pick_npz(runs / "a_s1", JO.DEFAULT_NPZ)
    assert primary.name == "evalx_val_core_last_tf_pert.npz"
    assert [JO.variant_tag(p) for p in others] == ["last", "last_tf_pert_gpu", "selected"]
    assert JO.pick_npz(runs / "a_s2", JO.DEFAULT_NPZ)[0].name == "evalx_val_core_last.npz"   # the fallback
    assert JO.pick_npz(tmp_path / "outputs" / "other" / "hist_run", JO.DEFAULT_NPZ)[0] is None
    assert JO.arm_of("rt_cnn_s3407") == "rt_cnn" and JO.arm_of("track_render51_dr_so3fk_rep2") == "track_render51_dr_so3fk"
    assert JO.arm_of("rt_anchor_2k_at0.6_s3407") == "rt_anchor_2k_at0.6"


# ------------------------------------------------------------------ the recorded runs
#: what `jitter_offline.py --runs rt_cnntrack_s3407` printed on the recorded `evalx_val_core_last_tf_pert.npz` the
#: first time it ran (CPU; bit-identical for 1, 2, 4 and 8 torch threads)
REC = {
    "zgz_global": dict(n_frames=1386, n_steps=1385, jit_gt_mm=7.558133, jit_pred_mm=9.204060, acc_gt_mm=2.549364,
                       acc_pred_mm=6.402033, acc_err_mm=6.499862, jit_only_transl_mm=8.747573, jit_only_root_mm=8.751946,
                       jit_only_fingers_mm=7.793359, shap_transl_mm=2.865649, shap_root_mm=2.696763,
                       shap_fingers_mm=0.937451, rot_acc_gt_deg=1.821894, rot_acc_pred_deg=3.068544, ra_mpjpe_mm=9.596321),
    "zgz_local": dict(n_frames=1204, n_steps=1163, jit_gt_mm=3.789076, jit_pred_mm=6.065321, acc_gt_mm=5.525745,
                      acc_pred_mm=8.106129, acc_err_mm=10.276837, jit_only_transl_mm=6.722000, jit_only_root_mm=6.450057,
                      jit_only_fingers_mm=4.470967, shap_transl_mm=3.646067, shap_root_mm=3.909779,
                      shap_fingers_mm=2.720991, rot_acc_gt_deg=4.922189, rot_acc_pred_deg=4.307158, ra_mpjpe_mm=18.412826),
    "overall": dict(n_frames=2590, n_steps=2548, jit_gt_mm=5.806031, jit_pred_mm=7.744971, acc_gt_mm=3.932979,
                    acc_pred_mm=7.194207, acc_err_mm=8.255645, jit_only_transl_mm=7.805955, jit_only_root_mm=7.681879,
                    jit_only_fingers_mm=6.248896, shap_transl_mm=3.228438, shap_root_mm=3.260652,
                    shap_fingers_mm=1.766556, rot_acc_gt_deg=3.263112, rot_acc_pred_deg=3.644332, ra_mpjpe_mm=13.694804,
                    jit_ratio_pooled=7.744971 / 5.806031, acc_ratio_pooled=7.194207 / 3.932979),
}


@pytest.mark.skipif(not RECORDED_NPZ.exists(), reason="the recorded rt_cnntrack_s3407 evaluation is not on this machine")
def test_jitter_offline_reproduces_the_first_run_on_the_recorded_rt_cnntrack_evaluation():
    from config import load_config
    cfg = load_config(str(REPO / JO.DEFAULT_CONFIG))
    root = Path(cfg["DATA"]["ROOT"])
    if not (root / "val" / "zgz_global_aux.npz").exists():
        pytest.skip("the validation data is not on this machine")
    job = {"name": "rt_cnntrack_s3407", "variant": None, "npz": str(RECORDED_NPZ), "mano_npz": cfg["MANO"]["NPZ"],
           "root": str(root), "manifest": cfg["DATA"]["SPLITS_MANIFEST"], "threads": 2, "static_mm": EX.STATIC_MM}
    entry = JO.analyse_job(job)["entry"]
    got = dict(entry["sequences"], overall=entry["overall"])
    for name, ref in REC.items():
        for k, v in ref.items():
            assert got[name][k] == pytest.approx(v, rel=1e-5, abs=2e-6), (name, k)
    o = entry["overall"]
    assert sum(o[f"shap_{b}_mm"] for b in BLOCKS) == pytest.approx(o["acc_err_mm"], abs=1e-6)
    assert sum(o[f"shap_share_{b}"] for b in BLOCKS) == pytest.approx(1.0, abs=1e-9)
    # the npz is the recorded evaluation: its RA is the recorded RA
    rec = json.loads((RECORDED_RUN / "evalx_val_core_last_tf_pert.json").read_text())["model"]["overall"]["mpjpe_ra_mm"]
    assert o["ra_mpjpe_mm"] == pytest.approx(rec, abs=1e-3)
    assert "nonfinite" not in got["zgz_global"] and "nonfinite_sequences" not in o


@pytest.mark.skipif(not (HIST_RUN / "evalx_val_core_step3000_tf_pert_dt.npz").exists(),
                    reason="the historical dr_so3fk evaluation (evalx, suffix dt) has not been run on this machine")
def test_evalx_reproduces_the_old_evaluators_recursive_ra_of_the_historical_arm():
    old = json.loads((HIST_RUN / "eval_step3000" / "track_metrics_step50.json").read_text())["overall"]["mpjpe_ra_mm"]
    z = np.load(HIST_RUN / "evalx_val_core_step3000_tf_pert_dt.npz")
    ra = np.concatenate([z[f"model|{s}|mpjpe_ra_mm"] for s in ("zgz_global", "zgz_local")]).astype(np.float64).mean()
    assert old == pytest.approx(12.254, abs=1e-3)
    assert abs(ra - old) < 0.05
    # the tool's own number for the run is the same quantity: its difference to the old evaluator is the one printed
    assert JO._old_eval_ra(HIST_RUN / "evalx_val_core_step3000_tf_pert_dt.npz") == pytest.approx(old, abs=0)


def test_variants_doc_and_noise_floor_outputs(tmp_path, monkeypatch, mano):
    root = tmp_path / "semkine"
    d = _fake_run(root, "fake", 1)
    z = dict(np.load(d / "evalx_val_core_last_tf_pert.npz"))
    z["model|A|pred"] = z["model|A|pred"].copy()
    z["model|A|pred"][:, :3] += np.random.default_rng(9).normal(0, 0.0005, (N, 3)).astype(np.float32)
    np.savez_compressed(d / "evalx_val_core_last_tf_pert_gpu.npz", **z)          # a second evaluation, slightly different
    np.savez_compressed(d / "evalx_val_core_selected.npz", **dict(np.load(d / "evalx_val_core_last_tf_pert.npz")))
    state = {"mano": mano, "device": DEV, "static_mm": EX.STATIC_MM, "betas": {}, "dirs": {}, "root": tmp_path}
    monkeypatch.setattr(JO, "_state", lambda *a, **k: state)
    monkeypatch.setattr(JO, "betas_for", lambda st, seq: BETAS)
    res = JO.main(["--runs", "fake_*", "--runs-root", str(root), "--out-prefix", str(tmp_path / "rep"), "--variants",
                   "--headline", "fake_*", "--doc", str(tmp_path / "doc.md"), "--jobs", "1"])
    v = res["runs"]["fake_s1"]["variants"]
    assert sorted(v) == ["last_tf_pert_gpu", "selected"]
    assert v["selected"]["overall"]["acc_err_mm"] == res["runs"]["fake_s1"]["overall"]["acc_err_mm"]      # same arrays
    assert v["selected"]["pred_sha1"] == res["runs"]["fake_s1"]["pred_sha1"]
    assert v["last_tf_pert_gpu"]["pred_sha1"] != res["runs"]["fake_s1"]["pred_sha1"]
    dacc = abs(v["last_tf_pert_gpu"]["overall"]["acc_err_mm"] - res["runs"]["fake_s1"]["overall"]["acc_err_mm"])
    assert dacc > 1e-3                                                                                       # different arrays
    line = JO.noise_floor_line(res)
    assert line.startswith("1 pairs, 0 of them identical") and f"{dacc:.2f} mm in `acc_err`" in line
    assert "(0 with bitwise identical per-step predictions)" in line
    md, doc = (tmp_path / "rep.md").read_text(), (tmp_path / "doc.md").read_text()
    assert "## 7. Other recorded evaluations" in md and "last_tf_pert_gpu" in md and "selected" in md
    assert "## Headline runs" in doc and "| fake_s1 |" in doc and "## Caveats" in doc and "## Reproduce" in doc
    assert "1 recorded closed-loop evaluations" in doc and "plus 0 new" in doc
    assert "(1 from `evalx_val_core_last_tf_pert.npz`;" in doc               # the source of the arrays is named
    assert "## Arm contrasts" not in doc                                       # none of the contrast arms is present


def test_noise_floor_line_counts_bitwise_identical_pairs_and_skips_diverged_runs():
    def ov(acc, jit, ra, share=0.5):
        return {"acc_err_mm": acc, "jit_pred_mm": jit, "ra_mpjpe_mm": ra, **{f"shap_share_{b}": share for b in BLOCKS}}
    res = {"runs": {
        "a": {"overall": ov(10, 5, 13), "pred_sha1": "x", "variants": {"last_tf_pert_gpu": {"overall": ov(10, 5, 13), "pred_sha1": "x"}}},
        "b": {"overall": ov(10, 5, 13), "pred_sha1": "y", "variants": {"last_tf_pert_gpu": {"overall": ov(10.2, 5.1, 13.3, 0.55), "pred_sha1": "z"}}},
        # a diverged primary evaluation has no numbers: its pair is left out instead of crashing the line
        "c": {"overall": {"nonfinite_sequences": ["A"]}, "pred_sha1": "w", "variants": {"last_tf_pert_gpu": {"overall": ov(1, 1, 1), "pred_sha1": "w"}}},
    }}
    line = JO.noise_floor_line(res)
    assert line.startswith("2 pairs, 1 of them identical to the printed precision (1 with bitwise identical per-step predictions)")
    assert "0.20 mm in `acc_err`" in line and "0.10 mm in `jit_pred`" in line and "0.30 mm in RA" in line
    assert "5.0 points in a share" in line
    assert JO.noise_floor_line({"runs": {}}) == "no pair of evaluations of the same checkpoint in this run."


# ------------------------------------------------------------------ diverged runs (NaN / inf)
def test_nonfinite_info_reports_where_and_undefined_keys_flag_overflow():
    ok = np.zeros((10, 51), np.float32)
    assert JO.nonfinite_info({"pred": ok, "gt": ok, "run": np.zeros(10, int)}) is None
    bad = ok.copy()
    bad[3, 5] = np.nan
    bad[7:9, :] = np.inf
    info = JO.nonfinite_info({"pred": bad, "gt": ok, "mpjpe_ra_mm": np.array([1.0, np.nan] + [1.0] * 8)})
    assert set(info) == {"pred", "mpjpe_ra_mm"}
    assert info["pred"] == {"n_values": 1 + 2 * 51, "n_frames": 3, "first_frame": 3}
    assert info["mpjpe_ra_mm"] == {"n_values": 1, "n_frames": 1, "first_frame": 1}
    same = np.ones(9, bool)
    j = {k: 1.0 for k in JO.FIRST_DIFF_KEYS + JO.SECOND_DIFF_KEYS}
    assert JO.undefined_keys(j, same) == []
    j["acc_err_mm"] = float("nan")
    j.pop("jit_pred_mm")
    assert sorted(JO.undefined_keys(j, same)) == ["acc_err_mm", "jit_pred_mm"]
    assert JO.undefined_keys({}, np.zeros(9, bool)) == []                    # nothing to measure: nothing is required
    two = np.array([True, False] * 4 + [True])                               # segments of two frames: first differences only
    assert JO.undefined_keys({k: 1.0 for k in JO.FIRST_DIFF_KEYS}, two) == []
    assert sorted(JO.undefined_keys({}, two)) == sorted(JO.FIRST_DIFF_KEYS)


def test_a_nonfinite_sequence_is_flagged_and_the_run_gets_no_overall(tmp_path, monkeypatch, mano):
    d = _fake_run(tmp_path, "bad", 1)
    p = d / "evalx_val_core_last_tf_pert.npz"
    clean = JO.analyse_npz(p, _stub_state(monkeypatch, mano, tmp_path))
    _poison(p, "A")                                                          # NaN in 10 frames of the first sequence
    per = JO.analyse_npz(p, _stub_state(monkeypatch, mano, tmp_path))
    assert set(per) == {"A", "B"}
    a, b = per["A"], per["B"]
    assert a["nonfinite"] == {"pred": {"n_values": 10, "n_frames": 10, "first_frame": 100}}
    assert a["n_frames"] == N and a["n_steps"] == 238 and a["n_segments"] == 2
    assert not any(isinstance(v, float) for v in a.values())                 # a diverged sequence has no values at all
    assert a["pred_sha1"] != clean["A"]["pred_sha1"] and "nonfinite" not in b
    assert b == clean["B"]                                                   # the healthy sequence is untouched by its neighbour
    o = JO.combine(per)
    assert o == {"n_frames": N + 120, "n_steps": 238 + 119, "n_segments": 3, "nonfinite_sequences": ["A"]}
    assert JO.dominance(o) == "non-finite" and not JO.runaway(o)
    assert JO.finish(dict(o)) == o                                           # nothing is invented for it afterwards
    # without the diverged sequence the overall is the other sequence's, and it is defined: no silent fallback to that
    ok = JO.combine({"B": b})
    assert ok["acc_err_mm"] == pytest.approx(b["acc_err_mm"]) and "nonfinite_sequences" not in ok


def test_nonfinite_gt_and_nonfinite_recorded_errors_are_flagged_too(tmp_path, monkeypatch, mano):
    d = _fake_run(tmp_path, "bad", 2)
    p = d / "evalx_val_core_last_tf_pert.npz"
    _poison(p, "A", key="gt", rows=slice(5, 6), col=0, value=np.inf)         # the ground truth itself is not finite
    with np.load(p) as zf:
        z = dict(zf)
    ra = z["model|B|mpjpe_ra_mm"].copy()
    ra[3] = np.nan
    z["model|B|mpjpe_ra_mm"] = ra
    np.savez_compressed(p, **z)
    per = JO.analyse_npz(p, _stub_state(monkeypatch, mano, tmp_path))
    assert per["A"]["nonfinite"] == {"gt": {"n_values": 1, "n_frames": 1, "first_frame": 5}}
    assert per["B"]["nonfinite"] == {"mpjpe_ra_mm": {"n_values": 1, "n_frames": 1, "first_frame": 3}}
    assert JO.combine(per)["nonfinite_sequences"] == ["A", "B"]


@pytest.mark.filterwarnings("ignore")
def test_finite_predictions_that_overflow_the_kinematics_are_flagged_too(tmp_path, monkeypatch, mano):
    d = _fake_run(tmp_path, "huge", 1)
    p = d / "evalx_val_core_last_tf_pert.npz"
    with np.load(p) as zf:
        z = dict(zf)
    pa = z["model|A|pred"].copy()
    pa[::2, 2], pa[1::2, 2] = 3.0e38, -3.0e38                                # finite float32, but a step of 6e38 overflows
    z["model|A|pred"] = pa
    np.savez_compressed(p, **z)
    per = JO.analyse_npz(p, _stub_state(monkeypatch, mano, tmp_path))
    assert np.isfinite(pa).all()
    assert "jit_pred_mm" in per["A"]["nonfinite"]["jitter_decomp"]["undefined_keys"]
    assert JO.combine(per)["nonfinite_sequences"] == ["A"] and "nonfinite" not in per["B"]


def test_sequences_without_steps_are_not_diverged_just_empty(tmp_path, monkeypatch, mano):
    d = _fake_run(tmp_path, "single", 3)
    p = d / "evalx_val_core_last_tf_pert.npz"
    with np.load(p) as zf:
        z = dict(zf)
    z["model|B|run"] = np.arange(120)                                        # every frame its own segment: nothing to measure
    np.savez_compressed(p, **z)
    per = JO.analyse_npz(p, _stub_state(monkeypatch, mano, tmp_path))
    assert "nonfinite" not in per["B"] and per["B"]["n_steps"] == 0 and math.isnan(per["B"]["acc_err_mm"])
    o = JO.combine(per)
    assert "nonfinite_sequences" not in o and o["acc_err_mm"] == pytest.approx(per["A"]["acc_err_mm"])   # the other sequence's
    assert o["n_frames"] == N + 120


def test_cli_marks_diverged_runs_instead_of_averaging_around_them(tmp_path, monkeypatch, mano, capsys):
    root = tmp_path / "semkine"
    _fake_run(root, "ok", 1)
    _fake_run(root, "mix", 1)
    _fake_run(root, "mix", 2)
    _fake_run(root, "nan", 1)
    _fake_run(root, "inf", 1)
    _poison(root / "mix_s2" / "evalx_val_core_last_tf_pert.npz", "A")                           # NaN in the first sequence only
    _poison(root / "inf_s1" / "evalx_val_core_last_tf_pert.npz", "B", rows=slice(5, 6), col=0, value=np.inf)   # inf in the second
    for seq in ("A", "B"):                                                                       # nan_s1: every sequence diverged
        _poison(root / "nan_s1" / "evalx_val_core_last_tf_pert.npz", seq)
    _stub_state(monkeypatch, mano, tmp_path)
    res = JO.main(["--runs", "*", "--runs-root", str(root), "--out-prefix", str(tmp_path / "rep"), "--doc", str(tmp_path / "doc.md"),
                   "--detail", "inf_s*"])
    err = capsys.readouterr().err
    assert "WARNING: non-finite predictions (NaN / inf) in 3 run(s): inf_s1, mix_s2, nan_s1" in err
    assert res["meta"]["nonfinite_runs"] == ["inf_s1", "mix_s2", "nan_s1"]
    text = (tmp_path / "rep.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    js = json.loads(text)
    R = js["runs"]
    assert [R[n]["nonfinite"] for n in ("ok_s1", "mix_s1", "mix_s2", "nan_s1", "inf_s1")] == [False, False, True, True, True]
    # a run with a diverged sequence has no overall values -- the healthy sequence does NOT stand in for the run
    assert R["inf_s1"]["overall"]["nonfinite_sequences"] == ["B"] and "acc_err_mm" not in R["inf_s1"]["overall"]
    assert math.isfinite(R["inf_s1"]["sequences"]["A"]["acc_err_mm"])             # but its healthy sequence keeps its numbers
    assert R["nan_s1"]["overall"]["nonfinite_sequences"] == ["A", "B"] and "ra_mpjpe_mm" not in R["nan_s1"]["overall"]
    assert R["mix_s2"]["overall"]["nonfinite_sequences"] == ["A"] and "shap_share_transl" not in R["mix_s2"]["overall"]
    # the arms: a mixed arm averages its finite runs and says so; an arm with no finite run has no mean
    mix = js["arms"]["mix"]
    assert mix["n_runs"] == 2 and mix["nonfinite_runs"] == ["mix_s2"] and mix["sd_over_runs"] == {}
    assert mix["mean"]["acc_err_mm"] == pytest.approx(R["mix_s1"]["overall"]["acc_err_mm"])
    assert js["arms"]["nan"]["mean"] == {} and js["arms"]["nan"]["nonfinite_runs"] == ["nan_s1"]
    assert js["arms"]["ok"]["nonfinite_runs"] == []
    md, doc = (tmp_path / "rep.md").read_text(), (tmp_path / "doc.md").read_text()
    assert "2 runs with an attribution" in md                                     # ok_s1 and mix_s1 only
    assert "NON-FINITE predictions" in md and "inf_s1, mix_s2, nan_s1" in md
    row = next(line for line in md.splitlines() if line.startswith("| nan_s1 ‡ |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1:-1] == ["n/a"] * (len(cells) - 2) and cells[-1] == "non-finite"
    assert "| mix ‡ | 2 |" in md and "| nan ‡ | 1 |" in md and "‡ non-finite predictions" in md
    lowest = next(line for line in md.splitlines() if line.startswith("- Lowest `acc_err`"))
    assert "mix " not in lowest and "nan " not in lowest and "ok " in lowest      # an arm with a diverged run is not ranked
    assert "NON-FINITE predictions" in doc and "| nan ‡ | 1 |" in doc
    # the sequence table lists the healthy sequence of a diverged run with its numbers and the diverged one without
    cells5 = [[x.strip() for x in line.strip().strip("|").split("|")] for line in md.splitlines() if line.startswith("| inf_s1 ‡ | ")]
    seq_rows = {c[1]: c for c in cells5 if c[1] in ("A", "B")}
    assert set(seq_rows) == {"A", "B"}
    assert "n/a" not in seq_rows["A"][4:] and seq_rows["A"][2:4] == [str(N), "2"]
    assert seq_rows["B"][2:4] == ["120", "1"] and seq_rows["B"][4:14] == ["n/a"] * 10 and seq_rows["B"][14] == "non-finite"


def test_a_diverged_second_evaluation_is_listed_without_values_and_not_paired(tmp_path, monkeypatch, mano, capsys):
    root = tmp_path / "semkine"
    d = _fake_run(root, "fake", 1)
    with np.load(d / "evalx_val_core_last_tf_pert.npz") as zf:
        np.savez_compressed(d / "evalx_val_core_last_tf_pert_gpu.npz", **dict(zf))
    _poison(d / "evalx_val_core_last_tf_pert_gpu.npz", "A")                     # the second evaluation diverged, the primary did not
    _stub_state(monkeypatch, mano, tmp_path)
    res = JO.main(["--runs", "fake_*", "--runs-root", str(root), "--out-prefix", str(tmp_path / "rep"), "--variants"])
    err = capsys.readouterr().err
    v = res["runs"]["fake_s1"]["variants"]["last_tf_pert_gpu"]
    assert v["overall"]["nonfinite_sequences"] == ["A"] and "acc_err_mm" not in v["overall"]
    assert res["runs"]["fake_s1"]["nonfinite"] is False and res["meta"]["nonfinite_runs"] == []
    assert res["meta"]["nonfinite_variants"] == ["fake_s1:last_tf_pert_gpu"]
    assert "other recorded evaluation(s) (--variants): fake_s1:last_tf_pert_gpu" in err and "in 1 run(s)" not in err
    md = (tmp_path / "rep.md").read_text()
    assert "NON-FINITE predictions also in 1 other recorded evaluation(s)" in md and "fake_s1:last_tf_pert_gpu" in md
    row = next(line for line in md.splitlines() if line.startswith("| fake_s1 | last_tf_pert_gpu |"))
    assert row.endswith("| non-finite |") and row.count("n/a") == 10
    assert JO.noise_floor_line(res) == "no pair of evaluations of the same checkpoint in this run."
    assert math.isfinite(res["runs"]["fake_s1"]["overall"]["acc_err_mm"])           # the primary is untouched


def test_runs_with_byte_identical_predictions_are_reported(tmp_path, monkeypatch, mano):
    root = tmp_path / "semkine"
    _fake_run(root, "dupa", 5)
    _fake_run(root, "dupb", 5)                                                   # the same seed: the same arrays
    _fake_run(root, "other", 6)
    _stub_state(monkeypatch, mano, tmp_path)
    res = JO.main(["--runs", "*", "--runs-root", str(root), "--out-prefix", str(tmp_path / "rep")])
    R = res["runs"]
    assert res["meta"]["identical_evaluations"] == [["dupa_s5", "dupb_s5"]]
    assert R["dupa_s5"]["same_arrays_as"] == ["dupb_s5"] and R["dupb_s5"]["same_arrays_as"] == ["dupa_s5"]
    assert "same_arrays_as" not in R["other_s6"]
    assert R["dupa_s5"]["pred_sha1"] == R["dupb_s5"]["pred_sha1"] != R["other_s6"]["pred_sha1"]
    assert R["dupa_s5"]["sequences"]["A"]["pred_sha1"] == R["dupb_s5"]["sequences"]["A"]["pred_sha1"]
    md = (tmp_path / "rep.md").read_text()
    assert "3 runs with an attribution" in md
    assert "dupa_s5 = dupb_s5" in md and "Counting each distinct evaluation once (2 runs)" in md


# ------------------------------------------------------------------ the old evaluator's recursive RA
def test_old_evaluator_ra_is_read_from_the_matching_eval_step_directory(tmp_path):
    run = tmp_path / "hist"
    for step, ra in ((3000, 12.254020), (4000, 11.5)):
        (run / f"eval_step{step}").mkdir(parents=True)
        (run / f"eval_step{step}" / "track_metrics_step50.json").write_text(json.dumps({"overall": {"mpjpe_ra_mm": ra}}))
    (run / "eval_step5000").mkdir()                                              # a directory without the metrics file
    assert JO._old_eval_ra(run / "evalx_val_core_step3000_tf_pert_dt.npz") == 12.254020
    assert JO._old_eval_ra(run / "evalx_val_core_step4000_tf_pert_dt.npz") == 11.5       # the step number picks the directory
    assert JO._old_eval_ra(run / "evalx_val_core_step5000_tf_pert_dt.npz") is None
    assert JO._old_eval_ra(run / "evalx_val_core_step6000_tf_pert_dt.npz") is None
    assert JO._old_eval_ra(run / "evalx_val_core_last_tf_pert.npz") is None            # not a step-checkpoint evaluation


def test_historical_run_enters_the_report_with_its_signed_difference_to_the_old_evaluator(tmp_path, monkeypatch, mano):
    ra = (N * 11.0 + 120 * 1.0) / (N + 120)                                      # the RA the arrays of a seed-1 fake run give
    hist = tmp_path / "outputs" / "hand_data51"
    dirs = {}
    for name, old in (("hista", ra - 0.02), ("histb", ra + 0.30)):                # evalx 0.02 above / 0.30 below the old evaluator
        d = _fake_run(hist, name, 1, npz_name="evalx_val_core_step3000_tf_pert_dt.npz")
        (d / "eval_step3000").mkdir()
        (d / "eval_step3000" / "track_metrics_step50.json").write_text(json.dumps({"overall": {"mpjpe_ra_mm": old}}))
        dirs[name] = d
    _fake_run(tmp_path / "outputs" / "semkine", "plain", 1)                       # a recorded run: no old evaluator
    _stub_state(monkeypatch, mano, tmp_path)
    res = JO.main(["--runs", str(dirs["hista"]), str(dirs["histb"]), "plain_s1", "--runs-root", str(tmp_path / "outputs" / "semkine"),
                   "--out-prefix", str(tmp_path / "rep"), "--doc", str(tmp_path / "doc.md")])
    R = res["runs"]
    assert R["hista_s1"]["npz_tag"] == "step3000_tf_pert_dt" and R["hista_s1"]["overall"]["ra_mpjpe_mm"] == pytest.approx(ra)
    assert R["hista_s1"]["ra_old_evaluator_mm"] == pytest.approx(ra - 0.02)
    assert R["hista_s1"]["ra_diff_to_old_evaluator_mm"] == pytest.approx(+0.02, abs=1e-6)       # evalx minus old
    assert R["histb_s1"]["ra_diff_to_old_evaluator_mm"] == pytest.approx(-0.30, abs=1e-6)
    assert "ra_old_evaluator_mm" not in R["plain_s1"] and "ra_diff_to_old_evaluator_mm" not in R["plain_s1"]
    md, doc = (tmp_path / "rep.md").read_text(), (tmp_path / "doc.md").read_text()
    assert "## 6. Historical arm" in md
    ra_a = next(line for line in md.splitlines() if line.startswith("| hista_s1 | step3000_tf_pert_dt |"))
    ra_b = next(line for line in md.splitlines() if line.startswith("| histb_s1 | step3000_tf_pert_dt |"))
    assert f"| {ra:.6f} | {ra - 0.02:.6f} | +0.020000 | yes |" in ra_a
    assert f"| {ra:.6f} | {ra + 0.30:.6f} | -0.300000 | NO |" in ra_b
    assert "## Historical arm: evalx against the old evaluator" in doc and "| hista_s1 |" in doc
    da = next(line for line in doc.splitlines() if line.startswith("| hista_s1 |") and "yes" in line)
    db = next(line for line in doc.splitlines() if line.startswith("| histb_s1 |") and line.endswith("| NO |"))
    assert "+2.00e-02" in da and "-3.00e-01" in db
    # the two historical evaluations are counted apart from the recorded ones, and their tag is not among the sources
    assert ("of 1 recorded closed-loop evaluations (1 from `evalx_val_core_last_tf_pert.npz`; CPU only, no new closed loop) "
            "plus 2 new `evalx.py eval") in doc


def test_a_diverged_historical_run_has_no_difference_to_the_old_evaluator(tmp_path, monkeypatch, mano):
    hist = tmp_path / "outputs" / "hand_data51"
    d = _fake_run(hist, "histc", 1, npz_name="evalx_val_core_step3000_tf_pert_dt.npz")
    (d / "eval_step3000").mkdir()
    (d / "eval_step3000" / "track_metrics_step50.json").write_text(json.dumps({"overall": {"mpjpe_ra_mm": 12.0}}))
    _poison(d / "evalx_val_core_step3000_tf_pert_dt.npz", "A")
    _stub_state(monkeypatch, mano, tmp_path)
    res = JO.main(["--runs", str(d), "--runs-root", str(tmp_path / "outputs" / "semkine"), "--out-prefix", str(tmp_path / "rep"),
                   "--doc", str(tmp_path / "doc.md")])
    r = res["runs"]["histc_s1"]
    assert r["ra_old_evaluator_mm"] == 12.0 and "ra_diff_to_old_evaluator_mm" not in r
    md, doc = (tmp_path / "rep.md").read_text(), (tmp_path / "doc.md").read_text()
    assert any(line.startswith("| histc_s1 | step3000_tf_pert_dt | n/a | 12.000000 | n/a | non-finite |") for line in md.splitlines())
    assert any(line.startswith("| histc_s1 | n/a | 12.000000 | n/a | non-finite |") for line in doc.splitlines())


# ------------------------------------------------------------------ the documentation page
def test_doc_names_its_sources_and_prints_the_contrasts_of_the_arms_present(tmp_path, monkeypatch, mano):
    root = tmp_path / "semkine"
    _fake_run(root, "rt_cnn", 1)
    _fake_run(root, "rt_cnndelta", 2)
    _fake_run(root, "rt_cnntrack", 3)
    _fake_run(root, "rt_cnnf", 4, npz_name="evalx_val_core_last.npz")           # the only recorded evaluation is the 'last' one
    _stub_state(monkeypatch, mano, tmp_path)
    res = JO.main(["--runs", "*", "--runs-root", str(root), "--out-prefix", str(tmp_path / "rep"), "--doc", str(tmp_path / "doc.md")])
    doc = (tmp_path / "doc.md").read_text()
    md = (tmp_path / "rep.md").read_text()
    assert res["runs"]["rt_cnnf_s4"]["npz_tag"] == "last"
    assert ("of 4 recorded closed-loop evaluations (3 from `evalx_val_core_last_tf_pert.npz`, "
            "1 from `evalx_val_core_last.npz` (marked `(last)` in the tables); CPU only") in doc
    assert "| rt_cnnf_s4 (last) |" in doc and "| rt_cnnf_s4 (last) |" in md
    assert "## Arm contrasts: what differs, and what the numbers can say" in doc
    m = {a: res["arms"][a]["mean"] for a in ("rt_cnn", "rt_cnndelta", "rt_cnntrack", "rt_cnnf")}
    rows = {line.split("|")[1].strip(): [c.strip() for c in line.strip().strip("|").split("|")]
             for line in doc.splitlines() if line.startswith("| rt_cnn")}
    row = rows["rt_cnndelta -> rt_cnntrack"]
    assert "`MODEL.PREV_RENDER` false -> true" in row[1]
    assert row[2] == f"{m['rt_cnndelta']['acc_err_mm']:.2f} -> {m['rt_cnntrack']['acc_err_mm']:.2f}"
    assert row[5] == f"{m['rt_cnndelta']['shap_transl_mm']:.2f} -> {m['rt_cnntrack']['shap_transl_mm']:.2f}"
    assert row[6] == f"{m['rt_cnndelta']['acc_err_only_transl_mm']:.2f} -> {m['rt_cnntrack']['acc_err_only_transl_mm']:.2f}"
    assert row[7] == f"{JO._pct(m['rt_cnndelta']['shap_share_transl'])} -> {JO._pct(m['rt_cnntrack']['shap_share_transl'])}"
    assert "rt_cnn -> rt_cnndelta" in rows and "rt_cnn -> rt_cnnf" in rows
    assert "rt_cnntrack -> track_render51_dr_so3fk" not in rows                   # an arm that is not in the report: no row
    assert "A Shapley value compares the blocks within one run" in doc
    assert f"{m['rt_cnnf']['acc_err_only_transl_mm']:.2f} against {m['rt_cnn']['acc_err_only_transl_mm']:.2f}" in doc
    # an arm with no usable mean (all of its runs diverged) gets no contrast row, and nothing crashes
    assert JO.contrast_rows({"rt_cnn": {"mean": {}}, "rt_cnndelta": {"mean": m["rt_cnndelta"]}}) == []
    assert JO.standalone_line({}) == "" and JO.contrast_rows({}) == []


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def test_contrast_descriptions_match_the_configs():
    """The 'what differs' text of the first two contrasts is what the YAML files say (nothing else differs)."""
    from config import load_config
    names = ("rt_cnn", "rt_cnndelta", "rt_cnntrack")
    cfg = {n: _flat(load_config(str(REPO / "configs" / "rt" / f"{n}.yaml"))) for n in names}
    ignore = {"TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR", "_config_path"}

    def diff(a, b):
        return {k for k in set(cfg[a]) | set(cfg[b]) if cfg[a].get(k) != cfg[b].get(k)} - ignore

    d1, d2 = diff("rt_cnn", "rt_cnndelta"), diff("rt_cnndelta", "rt_cnntrack")
    assert d1 == {"MODEL.PREDICT_DELTA", "MODEL.PREVPOS_EMBED", "MODEL.ZERO_EVENT_GATE"}
    assert d2 == {"MODEL.PREV_RENDER", "MODEL.RENDER_H", "MODEL.RENDER_W", "MODEL.RENDER_SCALE", "MODEL.RENDER_CHUNK"}
    assert [cfg[n][k] for n in ("rt_cnn", "rt_cnndelta") for k in sorted(d1)] == [False] * 3 + [True] * 3
    assert cfg["rt_cnndelta"]["MODEL.PREV_RENDER"] is False and cfg["rt_cnntrack"]["MODEL.PREV_RENDER"] is True
    text = {(a, b): w for a, b, w in JO.CONTRASTS}
    assert all(k.split(".")[1] in text[("rt_cnn", "rt_cnndelta")] for k in d1)
    assert "PREV_RENDER" in text[("rt_cnndelta", "rt_cnntrack")] and "RENDER_H/W/SCALE/CHUNK" in text[("rt_cnndelta", "rt_cnntrack")]


HIST_CFG = Path("/data1/lyq/code/mesh/EventHands/configs/eventhands_track_render51_dr_so3fk.yaml")


@pytest.mark.skipif(not HIST_CFG.exists(), reason="the historical config is not on this machine")
def test_historical_contrast_text_matches_the_two_configs():
    """The 'what differs' text of the last contrast (historical arm vs rt_cnntrack) is what the two YAML files say."""
    from config import load_config
    h = _flat(load_config(str(HIST_CFG)))
    t = _flat(load_config(str(REPO / "configs" / "rt" / "rt_cnntrack.yaml")))
    assert h["LOSS.TYPE"] == "so3_trans_fk" and "LOSS.TYPE" not in t                  # rt_cnntrack: the default mse_51d
    assert (h["LOSS.ROT_WEIGHT"], h["LOSS.TRANS_WEIGHT"], h["LOSS.FK_WEIGHT"]) == (1.0, 1.0, 2.0)
    assert "abs_full51" in h["MODEL.INIT_FROM"] and "step=6000" in h["MODEL.INIT_FROM"] and "MODEL.INIT_FROM" not in t
    assert (h["TRAIN.MAX_STEPS"], t["TRAIN.MAX_STEPS"]) == (3000, 6000)
    assert (h["TRAIN.BATCH_SIZE_PER_GPU"], h["TRAIN.ACCUMULATE_GRAD_BATCHES"]) == (2048, 1)
    assert (t["TRAIN.BATCH_SIZE_PER_GPU"], t["TRAIN.ACCUMULATE_GRAD_BATCHES"]) == (512, 2)
    assert (h["TRAIN.LR"], t["TRAIN.LR"]) == (0.005656, 0.004)
    assert "TRAIN.LR_SCHEDULE" not in h and t["TRAIN.LR_SCHEDULE"] == "cosine"
    assert h["TRAIN.WARMUP_STEPS"] == t["TRAIN.WARMUP_STEPS"]
    # augmentation: the older flat keys and the DOMRAND block carry the same numbers
    assert (h["AUG.ROLL_DEG"], h["AUG.SHIFT_PX"], h["AUG.HOT_PIXEL"], h["AUG.SCALE_RANGE"], h["AUG.EVENT_KEEP"]) == (
        15.0, 14.0, 0.0002, [0.8, 1.25], [0.25, 1.0])
    assert tuple(t[f"AUG.DOMRAND.{k}"] for k in ("ROLL_DEG", "SHIFT_PX", "HOT_PIXEL_RATE", "SCALE_MIN", "SCALE_MAX", "KEEP_MIN", "KEEP_MAX")) == (
        15.0, 14.0, 0.0002, 0.8, 1.25, 0.25, 1.0)
    # nothing else of the MODEL block differs except what the text names (render channels, chunk size)
    names = {k for k in set(h) | set(t) if k.startswith("MODEL.") and h.get(k) != t.get(k)}
    assert names == {"MODEL.INIT_FROM", "MODEL.RENDER_CHANNELS", "MODEL.RENDER_CHUNK"}
    text = {(a, b): w for a, b, w in JO.CONTRASTS}[("rt_cnntrack", "track_render51_dr_so3fk")]
    for needle in ("so3_trans_fk", "mse_51d", "abs_full51", "3000 vs 6000 steps", "batch 2048 vs 512 x 2", "0.005656", "0.004", "cosine"):
        assert needle in text


@pytest.mark.skipif(not (REPO / "outputs" / "semkine" / "rt_cnnf_s3409" / "training_metadata.json").exists(),
                    reason="the recorded rt_cnnf runs are not on this machine")
def test_the_filter_contrast_wraps_the_cnn_checkpoints_with_an_unfiltered_translation():
    for seed in (3407, 3408, 3409):
        meta = json.loads((REPO / "outputs" / "semkine" / f"rt_cnnf_s{seed}" / "training_metadata.json").read_text())
        assert meta["kind"] == "causal_filter" and meta["abs_run"].endswith(f"rt_cnn_s{seed}")
        assert (meta["a_root"], meta["a_rest"], meta["a_trans"]) == (0.5, 0.5, 1.0)       # gain 1.0 = no filtering
    text = {(a, b): w for a, b, w in JO.CONTRASTS}[("rt_cnn", "rt_cnnf")]
    assert "root 0.5, fingers 0.5 and translation 1.0" in text
