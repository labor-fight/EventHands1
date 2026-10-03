"""DT round, paired-gate report (`tools/dt/screen_report.py`): every row of the section-5 gate table with a
passing and a failing case, seeds with opposite signs, results exactly at the tie-band edge, a guardrail that trips
while the gate passes, a missing seed 3409, a missing jitter block, the dt_trnos and dt_trdz (section 9.6)
interaction arithmetic, the wording of an efficiency pass, malformed latency / step inputs, mixed jitter sources,
the shape of every markdown table, the
parameter count, determinism, and a smoke test on recorded runs. The synthetic evaluations are written in the
layout of `evalx.cmd_eval` and every expectation is computed by hand in the comments.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python -m pytest tests/test_dt_screen.py -q
"""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "tools" / "dt")]
import screen_report as SR                                                    # noqa: E402

SEEDS = (3407, 3408, 3409)
BASE_NAME = "dt_base"
PENDING = "pending 3409 confirmation"


# ------------------------------------------------------------------------------------ synthetic evaluations
def base_spec(i):
    """The baseline of seed index i. Gate means over seeds 3407 / 3408: RA 14.2, local RA 17.3, ABS 61.0, ROT 8.2,
    amplification 1.405, finger ratio 0.355, jit_pred 9.2, acc_err 8.2, global root ratio 1.45."""
    return dict(
        ra=14.0 + 0.4 * i, ra_g=11.0 + 0.2 * i, ra_l=17.0 + 0.6 * i, rot=8.0 + 0.4 * i, transl=50.0 + 2.0 * i,
        abs=60.0 + 2.0 * i, abs_g=30.0 + i, abs_l=90.0 + 3.0 * i, tf=10.0 + 0.2 * i, amp=1.40 + 0.01 * i,
        ep_g=0, ep_l=0, root_ratio=(1.5, 1.4, 1.45)[i], fing_g=0.30 + 0.01 * i, fing_l=0.40 + 0.01 * i,
        ret=(0.20, 0.10, 0.07, 0.06, 0.05), grid=17.0 + i, params=1000,
        jit=dict(jit_gt_mm=6.0, jit_pred_mm=9.0 + 0.4 * i, jit_ratio=1.5, acc_gt_mm=4.0, acc_pred_mm=7.0,
                 acc_err_mm=8.0 + 0.4 * i, acc_ratio=1.75, rot_acc_pred_deg=3.0, rot_acc_gt_deg=2.0))


BASE = {s: base_spec(i) for i, s in enumerate(SEEDS)}


def var(deltas=None, sets=None, jit_mul=None, drop_jitter=False, params=None, seeds=(3407, 3408)):
    """Per-seed specs of a candidate: the baseline plus per-seed deltas (tuple per seed, or a scalar for all)."""
    out = {}
    pick = lambda v, i: v[i] if isinstance(v, (tuple, list)) else v                   # noqa: E731
    for i, s in enumerate(seeds):
        sp = copy.deepcopy(BASE[s])
        for k, v in (deltas or {}).items():
            sp[k] = sp[k] + pick(v, i)
        for k, v in (sets or {}).items():
            sp[k] = pick(v, i)
        for k, f in (jit_mul or {}).items():
            sp["jit"][k] = sp["jit"][k] * f
        if drop_jitter:
            sp["jit"] = None
        if params is not None:
            sp["params"] = params
        out[s] = sp
    return out


def _seq(nf, ra, ab, rot, tr, ep, rr, fing):
    ci = lambda v: [v, v - 0.5, v + 0.5]                                                # noqa: E731
    return {"mpjpe_ra_mm": ci(ra), "mpvpe_ra_mm": ci(ra - 3), "mpjpe_abs_mm": ci(ab), "mpvpe_abs_mm": ci(ab - 0.1),
            "root_rot_deg": ci(rot), "transl_mm": ci(tr), "n_frames": nf,
            "failure": {"bad_step_frac": 0.0, "episodes": ep, "fail_time_frac": 0.0, "longest_s": 0.0},
            "by_events": {}, "motion": {"root_speed_ratio": rr, "finger_speed_ratio": fing,
                                        "root_speed_pred_deg": 2.0, "root_err_slope_deg_per_10s": []}}


def eval_json(arm, seed, sp, step=6000):
    model = {"n_frames": 100,
             "overall": {"mpjpe_ra_mm": sp["ra"], "mpvpe_ra_mm": sp["ra"] - 3, "mpjpe_abs_mm": sp["abs"],
                         "mpvpe_abs_mm": sp["abs"] - 0.1, "root_rot_deg": sp["rot"], "transl_mm": sp["transl"]},
             "zgz_global": _seq(60, sp["ra_g"], sp["abs_g"], sp["rot"], sp["transl"], sp["ep_g"], sp["root_ratio"],
                                sp["fing_g"]),
             "zgz_local": _seq(40, sp["ra_l"], sp["abs_l"], sp["rot"], sp["transl"], sp["ep_l"], 1.0, sp["fing_l"])}
    if sp.get("jit"):
        model["jitter"] = dict(sp["jit"])
    return {"run": f"{arm}_s{seed}", "ckpt": "x.ckpt", "step": step, "split": "val_core", "manifest": "m.json",
            "model": model,
            "hold": {"n_frames": 100, "overall": {"mpjpe_ra_mm": 99.0}},
            "tf": {"n_frames": 100, "overall": {"mpjpe_ra_mm": sp["tf"], "root_rot_deg": 5.0}},
            "amplification": {"mpjpe_ra_mm": sp["amp"], "root_rot_deg": 1.1},
            "perturb": {"10": {"trials": 10, "retention_k1_k2_k5_k10_k20": list(sp["ret"]), "half_life_steps": 0},
                        "20": {"trials": 10, "retention_k1_k2_k5_k10_k20": [0.5] * 5, "half_life_steps": 0}},
            "window": {"mode": "fixed", "ms": 50, "min_events": 0, "max_ms": 300}}


def write_ckpt(path, n_params):
    """A Lightning-style checkpoint whose parameter count is n_params; the BN statistics, the MANO buffers and
    a distillation teacher are in the state_dict too and must not be counted."""
    sd = {"net.w": torch.zeros(n_params - 3), "net.b": torch.zeros(3),
          "net.bn.running_mean": torch.zeros(11), "net.bn.running_var": torch.zeros(11),
          "net.bn.num_batches_tracked": torch.tensor(5), "mano.v_template": torch.zeros(4, 3),
          "teacher.net.w": torch.zeros(77)}
    torch.save({"state_dict": sd, "epoch": 1}, path)


def write_run(root, arm, seed, sp, step=6000, ckpt=True):
    d = Path(root) / f"{arm}_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / SR.EVAL_NAME).write_text(json.dumps(eval_json(arm, seed, sp, step)))
    (d / "selection_val_core_step50_splits_semkine.json").write_text(
        json.dumps({"run": d.name, "grid_median": sp["grid"], "selected": {"step": step}}))
    if ckpt:
        write_ckpt(d / f"{d.name}-step={step}.ckpt", sp["params"])
    return d


def world(tmp_path, arms=None, base_seeds=SEEDS, base_jit=True, base_name=BASE_NAME, ckpt=True):
    root = Path(tmp_path) / "semkine"
    root.mkdir(parents=True, exist_ok=True)
    for s in base_seeds:
        sp = copy.deepcopy(BASE[s])
        if not base_jit:
            sp["jit"] = None
        write_run(root, base_name, s, sp, ckpt=ckpt)
    for arm, per in (arms or {}).items():
        for s, sp in per.items():
            write_run(root, arm, s, sp, ckpt=ckpt)
    return root


def report(root, arms, **kw):
    kw.setdefault("offline_json", None)
    kw.setdefault("want_params", True)
    return SR.build_report(root, BASE_NAME, arms, **kw)


def one(tmp_path, arm, spec, **kw):
    """Build a world with one candidate and return its gate record."""
    rep = report(world(tmp_path, {arm: spec}), [arm], **kw)
    return rep["gates"][arm]


def clause(g, cid):
    return next(c for c in g["clauses"] if c["id"] == cid)


def md_tables(md):
    """[(header cells, [row cells ...])] of every markdown table of md. Fails when a row has another number of cells
    than its header (a dropped or an extra cell) or when a table is followed directly by text."""
    lines, out, i = md.splitlines(), [], 0
    while i < len(lines):
        if not lines[i].startswith("|"):
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].startswith("|"):
            j += 1
        rows = [[c.strip() for c in ln.rstrip()[1:-1].split("|")] for ln in lines[i:j]]
        assert all(len(r) == len(rows[0]) for r in rows), (
            f"table at line {i + 1} ({rows[0][:3]}): cell counts {sorted({len(r) for r in rows})}")
        assert j == len(lines) or lines[j] == "", f"table at line {i + 1} is followed directly by text"
        out.append((rows[0], rows[2:]))
        i = j
    return out


def table_with(md, column):
    """(header cells, rows) of the first table of md that has a column of that name."""
    return next(t for t in md_tables(md) if column in t[0])


def table_row(md, column, first_cell):
    """{column: cell} of the row whose first cell is first_cell, in the first table that has the column."""
    heads, rows = table_with(md, column)
    return dict(zip(heads, next(r for r in rows if r[0] == first_cell)))


# ------------------------------------------------------------------------------------ extraction
def test_extraction_of_every_column_from_an_evalx_json(tmp_path):
    sp = copy.deepcopy(BASE[3407])
    sp["ep_g"], sp["ep_l"] = 1, 2
    d = write_run(tmp_path, "dt_base", 3407, sp)
    rep = SR.build_report(tmp_path, "dt_base", [], offline_json=None)
    r = rep["runs"]["dt_base_s3407"]
    m = r["m"]
    assert (r["arm"], r["seed"], r["step"], r["n_frames"], r["jitter_src"]) == ("dt_base", 3407, 6000, 100, "eval")
    assert m["ra"] == 14.0 and m["ra_global"] == pytest.approx(11.0) and m["ra_local"] == pytest.approx(17.0)
    assert m["rot"] == 8.0 and m["transl"] == 50.0
    assert m["abs"] == 60.0 and m["abs_global"] == 30.0 and m["abs_local"] == 90.0
    assert m["tf_ra"] == 10.0 and m["amp"] == 1.4
    assert m["fail"] == 3                                               # episodes of both sequences added: 1 + 2
    assert m["root_ratio"] == 1.5                                       # the GLOBAL sequence's (local carries 1.0)
    assert m["finger_ratio_global"] == pytest.approx(0.30) and m["finger_ratio_local"] == pytest.approx(0.40)
    assert m["finger_ratio"] == pytest.approx(0.35)                     # mean of global and local
    assert (m["ret10_k1"], m["ret10_k5"], m["ret10_k20"]) == (0.20, 0.07, 0.05)   # indices 0 / 2 / 4 of the 10 deg list
    assert r["jitter"] == pytest.approx(BASE[3407]["jit"]) and r["params"] == 1000
    assert r["grid_median"] == 17.0 and r["ckpt_step"] == 6000 and r["ckpt"].endswith("dt_base_s3407-step=6000.ckpt")
    assert d.name == "dt_base_s3407"


def test_global_and_local_follow_the_evalx_naming_rule_and_are_frame_weighted():
    seq = lambda nf, ra: {"n_frames": nf, "mpjpe_ra_mm": [ra, 0, 0], "mpjpe_abs_mm": [ra, 0, 0],       # noqa: E731
                          "failure": {"episodes": 0}, "motion": {"root_speed_ratio": ra / 10,
                                                                 "finger_speed_ratio": ra / 100}}
    js = {"model": {"n_frames": 100, "overall": {"mpjpe_ra_mm": 5.0},
                    "a_global": seq(60, 10.0), "b_global": seq(20, 20.0), "a_local": seq(20, 30.0),
                    "jitter": {"jit_pred_mm": 1.0, "acc_err_mm": 2.5, "jit_gt_mm": float("nan")}}}
    m, jit = SR.extract_metrics(js)
    assert m["ra_global"] == pytest.approx((60 * 10 + 20 * 20) / 80)             # 12.5, frame-weighted
    assert m["ra_local"] == 30.0 and m["root_ratio"] == pytest.approx((60 * 1.0 + 20 * 2.0) / 80)
    assert m["finger_ratio_global"] == pytest.approx((60 * 0.1 + 20 * 0.2) / 80)
    assert m["finger_ratio"] == pytest.approx((m["finger_ratio_global"] + m["finger_ratio_local"]) / 2)
    assert jit == {"jit_pred_mm": 1.0, "acc_err_mm": 2.5}              # "jitter" is no sequence; a NaN value is dropped
    with pytest.raises(SR.ScreenError):
        SR.extract_metrics({"model": {"overall": {}}})


def test_a_json_without_tf_perturb_or_jitter_still_loads_with_na(tmp_path):
    js = eval_json("dt_base", 3407, BASE[3407])
    for k in ("tf", "amplification", "perturb"):
        del js[k]
    del js["model"]["jitter"]
    d = tmp_path / "dt_base_s3407"
    d.mkdir()
    (d / SR.EVAL_NAME).write_text(json.dumps(js))
    rep = SR.build_report(tmp_path, "dt_base", [], offline_json=None, want_params=False)
    m = rep["runs"]["dt_base_s3407"]["m"]
    assert m["tf_ra"] is None and m["amp"] is None and m["ret10_k1"] is None
    assert rep["runs"]["dt_base_s3407"]["jitter"] is None and rep["runs"]["dt_base_s3407"]["params"] is None
    md = SR.render_markdown(rep)
    assert "n/a" in md and "| nan |" not in md and "None" not in md


# ------------------------------------------------------------------------------------ dt_tr
TR_CASES = [
    # id, deltas, gate_pass, (C1 value, C2 value) when decidable
    ("pass", {"abs": (-6.0, -5.0), "ra": (0.2, 0.4)}, True, (-5.5, 0.3)),                  # mean -5.5 <= -5; +0.3 <= +0.5
    ("pass_at_both_edges", {"abs": (-5.0, -5.0), "ra": (0.5, 0.5)}, True, (-5.0, 0.5)),    # exactly -5 and +0.5
    ("fail_just_short", {"abs": (-4.9, -4.9), "ra": (0.0, 0.0)}, False, (-4.9, 0.0)),      # -4.9 > -5
    ("fail_opposite_signs", {"abs": (-12.0, 1.0), "ra": (0.0, 0.0)}, False, (-5.5, 0.0)),  # mean -5.5 but seed 3408 +1
    ("fail_ra_bound", {"abs": (-6.0, -6.0), "ra": (0.6, 0.6)}, False, (-6.0, 0.6)),        # +0.6 > +0.5
]


@pytest.mark.parametrize("case", TR_CASES, ids=[c[0] for c in TR_CASES])
def test_dt_tr_gate(tmp_path, case):
    _, deltas, ok, (v1, v2) = case
    g = one(tmp_path, "dt_tr", var(deltas))
    assert g["gate_pass"] is ok
    assert clause(g, "C1")["value"] == pytest.approx(v1) and clause(g, "C2")["value"] == pytest.approx(v2)
    assert g["pending_3409"] is False                                   # |dABS| >= 5 mm never lies in [1.1, 2.5)
    assert g["verdict"] == ("PASS; no guardrail tripped" if ok else "FAIL")
    assert g["better_than_base"] is ok


def test_dt_tr_opposite_signs_fail_through_the_sign_rule_not_the_mean(tmp_path):
    g = one(tmp_path, "dt_tr", var({"abs": (-12.0, 1.0)}))
    c1 = clause(g, "C1")
    assert c1["met"] is False and c1["value"] <= -5.0                   # the mean alone would have passed
    assert c1["per_seed"] == {"3407": pytest.approx(-12.0), "3408": pytest.approx(1.0)}
    assert g["wording"] == "tie"                                        # seeds disagree: preregistration says tie


# ------------------------------------------------------------------------------------ dt_nos family
NOS_ARMS = ["dt_nos", "dt_cam", "dt_dz", "dt_nopm", "dt_pmt"]
NOS_CASES = [
    # id, deltas, gate_pass, pending
    ("abs_at_edge_passes", {"abs": (-3.0, -3.0)}, True, False),                  # -3.0 <= -3
    ("abs_just_short", {"abs": (-2.9, -2.9)}, False, False),
    ("ra_at_tie_band_edge_passes_pending", {"ra": (-1.2, -1.0)}, True, True),    # mean -1.1 exactly: in [1.1, 2.5)
    ("ra_inside_tie_band_fails", {"ra": (-1.0, -1.0)}, False, False),            # -1.0 > -1.1
    ("ra_at_upper_band_edge_is_resolved", {"ra": (-2.5, -2.5)}, True, False),    # |-2.5| is not < 2.5
    ("ra_inside_band_pending", {"ra": (-2.4, -2.4)}, True, True),
    ("ra_resolved", {"ra": (-4.0, -4.0)}, True, False),
    ("abs_opposite_signs", {"abs": (-8.0, 1.0)}, False, False),                  # mean -3.5 <= -3 but one seed positive
    ("ra_opposite_signs", {"ra": (-3.4, 1.2)}, False, False),                    # mean -1.1 but one seed positive
    ("abs_clause_resolves_an_in_band_ra", {"abs": (-4.0, -4.0), "ra": (-1.5, -1.5)}, True, False),
]


@pytest.mark.parametrize("arm", NOS_ARMS)
@pytest.mark.parametrize("case", NOS_CASES, ids=[c[0] for c in NOS_CASES])
def test_dt_nos_family_gate(tmp_path, arm, case):
    _, deltas, ok, pending = case
    g = one(tmp_path, arm, var(deltas))
    assert g["family"] == "nos" and g["gate_pass"] is ok and g["pending_3409"] is pending
    assert g["label_text"] == (PENDING if pending else None)
    if ok:
        assert g["verdict"] == ("PASS, " + PENDING + "; no guardrail tripped" if pending
                                else "PASS; no guardrail tripped")
        assert g["better_than_base"] is (None if pending else True)
    else:
        assert g["verdict"] == "FAIL" and g["better_than_base"] is False


def test_dt_nos_clause_values_are_the_hand_computed_means(tmp_path):
    g = one(tmp_path, "dt_nos", var({"abs": (-4.0, -2.0), "ra": (-1.2, -1.0)}))
    assert clause(g, "C1")["value"] == pytest.approx(-3.0) and clause(g, "C1")["met"] is True
    assert clause(g, "C2")["value"] == pytest.approx(-1.1) and clause(g, "C2")["met"] is True
    assert g["paired"]["abs"]["per_seed"] == {"3407": pytest.approx(-4.0), "3408": pytest.approx(-2.0)}
    assert g["paired"]["abs"]["base_mean"] == pytest.approx(61.0)           # (60 + 62) / 2
    assert g["paired"]["abs"]["arm_mean"] == pytest.approx(58.0)            # (56 + 60) / 2
    assert g["paired"]["ra"]["reading"] == "better, in the 1.1-2.5 mm band"
    assert g["paired"]["abs"]["reading"] == "better"                        # |-3.0| >= 2.5


def test_a_band_sized_ra_gain_next_to_a_resolved_abs_pass_is_noted_not_labelled(tmp_path):
    g = one(tmp_path, "dt_cam", var({"abs": (-4.0, -4.0), "ra": (-1.5, -1.5)}))
    assert g["pending_3409"] is False and g["gate_pass"] is True
    assert any("C2 (RA)" in n and "-1.50 mm" in n for n in g["band"]["notes"])


READING_CASES = [
    # mean dRA of the two seeds, expected reading of the paired table (tie band 1.1 mm, pending band up to 2.5 mm)
    ((-1.0, -1.0), "tie"), ((1.0, 1.0), "tie"), ((0.0, 0.0), "tie"),
    ((-1.1, -1.1), "better, in the 1.1-2.5 mm band"), ((1.1, 1.1), "worse, in the 1.1-2.5 mm band"),
    ((-2.4, -2.4), "better, in the 1.1-2.5 mm band"), ((-2.5, -2.5), "better"), ((2.5, 2.5), "worse"),
    ((-5.0, 0.2), "tie (seeds disagree)"),                       # mean -2.4 but one seed went the other way
    ((-0.2, -2.2), "better, in the 1.1-2.5 mm band"),            # both negative, one of them inside the tie band
]


@pytest.mark.parametrize("deltas,expected", READING_CASES, ids=[str(c[0]) for c in READING_CASES])
def test_tie_band_reading_of_the_paired_table(tmp_path, deltas, expected):
    g = one(tmp_path, "dt_nos", var({"ra": deltas}))
    assert g["paired"]["ra"]["reading"] == expected


def test_nos_failures_are_worded_tie_or_worse_or_not_met(tmp_path):
    tie = one(tmp_path / "a", "dt_nos", var({"abs": (-0.5, -0.5), "ra": (-0.5, -0.5)}))
    worse = one(tmp_path / "b", "dt_nos", var({"abs": (4.0, 4.0), "ra": (3.0, 3.0)}))
    short = one(tmp_path / "c", "dt_nos", var({"abs": (-2.9, -2.9)}))
    assert (tie["gate_pass"], tie["wording"]) == (False, "tie")                 # both inside +-1.1 mm
    assert (worse["gate_pass"], worse["wording"]) == (False, "worse")
    assert (short["gate_pass"], short["wording"]) == (False, "gate not met")    # better by 2.9 mm but needs 3


# ------------------------------------------------------------------------------------ dt_so3c
SO3C_CASES = [
    # id, deltas, sets, gate_pass, pending, tripped
    ("pass_pending", {"rot": (-0.6, -0.4), "ra": (-1.3, -1.5)}, {}, True, True, []),          # ROT -0.5 edge, RA -1.4
    ("pass_resolved", {"rot": (-0.6, -0.4), "ra": (-3.0, -3.0)}, {}, True, False, []),
    ("fail_rot_short", {"rot": (-0.5, -0.3), "ra": (-3.0, -3.0)}, {}, False, False, []),      # ROT mean -0.4
    ("fail_ra_short", {"rot": (-0.6, -0.4), "ra": (-1.0, -1.0)}, {}, False, False, []),       # RA mean -1.0
    ("fail_rot_opposite_signs", {"rot": (-1.2, 0.2), "ra": (-3.0, -3.0)}, {}, False, False, []),
    ("fail_ra_opposite_signs", {"rot": (-0.6, -0.4), "ra": (-3.4, 1.2)}, {}, False, False, []),    # RA mean -1.1, seed 3408 +1.2
    ("pass_but_root_ratio_guardrail", {"rot": (-0.6, -0.4), "ra": (-3.0, -3.0)}, {"root_ratio": (0.85, 0.85)},
     True, False, ["G4a"]),
    ("pass_root_ratio_at_edge_ok", {"rot": (-0.6, -0.4), "ra": (-3.0, -3.0)}, {"root_ratio": (0.9, 0.9)},
     True, False, []),
]


@pytest.mark.parametrize("case", SO3C_CASES, ids=[c[0] for c in SO3C_CASES])
def test_dt_so3c_gate(tmp_path, case):
    _, deltas, sets, ok, pending, tripped = case
    g = one(tmp_path, "dt_so3c", var(deltas, sets))
    assert g["gate_pass"] is ok and g["pending_3409"] is pending and g["guardrails_tripped"] == tripped
    if ok and tripped:
        assert g["verdict"] == "PASS gate, NOT ADOPTED: guardrail G4a" and g["better_than_base"] is False
    elif ok:
        assert g["verdict"].startswith("PASS") and (PENDING in g["verdict"]) is pending
    else:
        assert g["verdict"] == "FAIL"


def test_dt_so3c_clause_values(tmp_path):
    g = one(tmp_path, "dt_so3c", var({"rot": (-0.6, -0.4), "ra": (-1.3, -1.5)}))
    assert clause(g, "C1")["value"] == pytest.approx(-0.5) and clause(g, "C1")["unit"] == "deg"
    assert clause(g, "C2")["value"] == pytest.approx(-1.4)
    assert "RA difference" in g["band"]["rule"]                       # the rule used for the angle gate is stated


# ------------------------------------------------------------------------------------ dt_acc
def acc(deltas=None, mul=0.8, **kw):
    """A dt_acc candidate: acc_err scaled by `mul` (base 8.0 / 8.4, mean 8.2)."""
    return var(deltas, jit_mul={"acc_err_mm": mul} if mul is not None else None, **kw)


ACC_CASES = [
    # id, kwargs of acc(), gate_pass, tripped
    ("pass_minus_20pct", dict(deltas={"ra": (0.3, 0.3), "abs": (1.1, 1.1)}), True, []),   # both bounds at their edge
    ("pass_at_minus_15pct_edge", dict(mul=0.85), True, []),                  # (6.8 + 7.14) / 2 / 8.2 - 1 = -0.15
    ("fail_minus_14pct", dict(mul=0.86), False, []),
    ("fail_ra_bound", dict(deltas={"ra": (0.4, 0.4)}), False, []),            # +0.4 > +0.3
    ("fail_abs_bound", dict(deltas={"abs": (1.2, 1.2)}), False, []),          # +1.2 > +1.1
    ("pass_but_finger_guardrail", dict(deltas={"fing_g": (-0.06, -0.06), "fing_l": (-0.06, -0.06)}), True, ["G4b"]),
]


@pytest.mark.parametrize("case", ACC_CASES, ids=[c[0] for c in ACC_CASES])
def test_dt_acc_gate(tmp_path, case):
    _, kw, ok, tripped = case
    g = one(tmp_path, "dt_acc", acc(**kw))
    assert g["gate_pass"] is ok and g["guardrails_tripped"] == tripped
    assert g["pending_3409"] is False                                          # relative gate: no mm band applies
    assert "relative gate" in g["band"]["rule"] and "never labelled pending" in g["band"]["rule"]
    if ok and tripped:
        assert g["verdict"] == "PASS gate, NOT ADOPTED: guardrail G4b"
    elif ok:
        assert g["verdict"] == "PASS; no guardrail tripped"


def test_dt_acc_clause_values_and_opposite_signs(tmp_path):
    g = one(tmp_path, "dt_acc", acc(mul=0.8, deltas={"ra": (0.3, 0.3)}))
    c1 = clause(g, "C1")
    assert c1["relative"] is True and c1["value"] == pytest.approx(-0.2)        # 0.8 x base: -20 %
    assert c1["per_seed"] == {"3407": pytest.approx(-1.6), "3408": pytest.approx(-1.68)}    # 6.4 - 8.0; 6.72 - 8.4
    assert clause(g, "C2")["value"] == pytest.approx(0.3)
    # acc_err 4.0 / 9.0 against 8.0 / 8.4: d = -4.0 / +0.6, mean -1.7 = -20.7 % meets -15 %, but seed 3408 got worse
    spec = var()
    spec[3407]["jit"]["acc_err_mm"] = 4.0
    spec[3408]["jit"]["acc_err_mm"] = 9.0
    g2 = one(tmp_path / "x", "dt_acc", spec)
    c = clause(g2, "C1")
    assert c["value"] == pytest.approx(-1.7 / 8.2) and c["met"] is False and g2["gate_pass"] is False
    assert c["per_seed"] == {"3407": pytest.approx(-4.0), "3408": pytest.approx(0.6)}


def test_dt_acc_needs_the_jitter_block(tmp_path):
    g = one(tmp_path, "dt_acc", var(drop_jitter=True))
    assert g["gate_pass"] is None and g["verdict"].startswith("INCOMPLETE")
    assert "dt_acc seed 3407: jitter.acc_err_mm n/a" in g["verdict"]
    assert g["better_than_base"] is None and clause(g, "C1")["value"] is None


# ------------------------------------------------------------------------------------ efficiency gates
EFF_ARMS = ["dt_w05", "dt_l3", "dt_w05_kd"]
EFF_CASES = [
    # id, params, deltas, latency of the arm in ms (base 10.0; None: no --latency-json), gate_pass, pending
    ("params_at_half_passes", 500, {"ra": (0.2, 0.2)}, None, True, False),         # 500 <= 0.5 x 1000; RA 14.4 <= 14.484
    ("params_501_is_undecided_without_latency", 501, {"ra": (0.2, 0.2)}, None, None, False),
    ("params_501_latency_90pct_fails", 501, {"ra": (0.2, 0.2)}, 9.0, False, False),
    ("params_501_latency_at_80pct_passes", 501, {"ra": (0.2, 0.2)}, 8.0, True, False),
    ("ra_at_1p02_edge_passes", 500, {"ra": (0.284, 0.284)}, None, True, False),     # 14.484 <= 1.02 x 14.2
    ("ra_above_1p02_fails", 500, {"ra": (0.3, 0.3)}, None, False, False),           # 14.5 > 14.484
    ("improvement_branch_pending", 900, {"ra": (-1.2, -1.0)}, None, True, True),    # cost None, but E1 passes
    ("improvement_branch_resolved", 900, {"ra": (-3.0, -3.0)}, None, True, False),
    ("improvement_with_opposite_signs_fails", 900, {"ra": (-3.4, 1.2)}, 9.0, False, False),
    ("cost_branch_resolves_an_in_band_gain", 500, {"ra": (-1.5, -1.5)}, None, True, False),
]


@pytest.mark.parametrize("arm", EFF_ARMS)
@pytest.mark.parametrize("case", EFF_CASES, ids=[c[0] for c in EFF_CASES])
def test_efficiency_gate(tmp_path, arm, case):
    _, params, deltas, lat, ok, pending = case
    kw = {}
    if lat is not None:
        p = tmp_path / "lat.json"
        p.write_text(json.dumps({BASE_NAME: 10.0, arm: lat, "other": 1.0}))
        kw["latency_json"] = p
    g = one(tmp_path, arm, var(deltas, params=params), **kw)
    assert g["family"] == "eff" and g["gate_pass"] is ok and g["pending_3409"] is pending
    assert g["efficiency"]["params_arm"] == params and g["efficiency"]["params_base"] == 1000
    assert g["efficiency"]["latency"] == ("measured" if lat is not None else "not measured")
    if ok is None:
        assert g["verdict"].startswith("INCOMPLETE") and "latency not measured" in g["verdict"]
        assert clause(g, "E4")["met"] is None
    elif ok is False:
        assert g["verdict"] == "FAIL"


def test_efficiency_clause_values(tmp_path):
    p = tmp_path / "lat.json"
    p.write_text(json.dumps({BASE_NAME: 10.0, "dt_l3": 7.0}))
    g = one(tmp_path, "dt_l3", var({"ra": (0.2, 0.2)}, params=250), latency_json=p)
    assert clause(g, "E2")["value"] == pytest.approx(14.4 / 14.2)               # RA mean ratio
    assert clause(g, "E3")["value"] == pytest.approx(0.25) and clause(g, "E3")["met"] is True
    assert clause(g, "E4")["value"] == pytest.approx(0.7) and clause(g, "E4")["met"] is True
    assert clause(g, "E1")["met"] is False and g["gate_pass"] is True


def test_efficiency_failure_wording(tmp_path):
    lat = tmp_path / "lat.json"
    lat.write_text(json.dumps({BASE_NAME: 10.0, "dt_l3": 9.5}))
    costly = one(tmp_path / "a", "dt_l3", var({"ra": (0.2, 0.2)}, params=600), latency_json=lat)
    worse = one(tmp_path / "b", "dt_l3", var({"ra": (3.0, 3.0)}, params=600), latency_json=lat)
    assert (costly["gate_pass"], costly["wording"]) == (False, "gate not met")      # a tied RA is not a tie here
    assert (worse["gate_pass"], worse["wording"]) == (False, "worse")


def test_cost_branch_pass_notes_the_in_band_gain(tmp_path):
    g = one(tmp_path, "dt_w05", var({"ra": (-1.5, -1.5)}, params=500))
    assert g["gate_pass"] is True and g["pending_3409"] is False
    assert any("E1" in n for n in g["band"]["notes"])


# ------------------------------------------------------------------------------------ universal guardrails
GUARD_CASES = [
    # id, extra spec (the gate itself passes through dRA = -3.0 mm), expected tripped ids
    ("G1_abs_plus_2p5", dict(deltas={"abs": (2.5, 2.5)}), ["G1"]),
    ("G1_abs_plus_2_at_edge_ok", dict(deltas={"abs": (2.0, 2.0)}), []),
    ("G2_local_ra_plus_1p2", dict(deltas={"ra_l": (1.2, 1.2)}), ["G2"]),
    ("G2_local_ra_plus_1_at_edge_ok", dict(deltas={"ra_l": (1.0, 1.0)}), []),
    ("G3a_jit_pred_plus_6pct", dict(jit_mul={"jit_pred_mm": 1.06}), ["G3a"]),
    ("G3a_jit_pred_plus_5pct_at_edge_ok", dict(jit_mul={"jit_pred_mm": 1.05}), []),
    ("G3b_acc_err_plus_6pct", dict(jit_mul={"acc_err_mm": 1.06}), ["G3b"]),
    ("G3b_acc_err_plus_5pct_at_edge_ok", dict(jit_mul={"acc_err_mm": 1.05}), []),
    ("G3_acc_err_decrease_ok", dict(jit_mul={"acc_err_mm": 0.5, "jit_pred_mm": 0.9}), []),
    ("G4a_root_ratio_0p85", dict(sets={"root_ratio": (0.85, 0.85)}), ["G4a"]),
    ("G4a_root_ratio_0p9_at_edge_ok", dict(sets={"root_ratio": (0.9, 0.9)}), []),
    ("G4b_finger_drop_0p06", dict(deltas={"fing_g": (-0.06, -0.06), "fing_l": (-0.06, -0.06)}), ["G4b"]),
    ("G4b_finger_drop_0p05_at_edge_ok", dict(deltas={"fing_g": (-0.05, -0.05), "fing_l": (-0.05, -0.05)}), []),
    ("G4b_finger_rise_ok", dict(deltas={"fing_g": (0.2, 0.2), "fing_l": (0.2, 0.2)}), []),
    ("G5_one_more_failure_episode", dict(deltas={"ep_g": (1, 0)}), ["G5"]),
    ("G6_amplification_plus_0p06", dict(deltas={"amp": (0.06, 0.06)}), ["G6"]),
    ("G6_amplification_plus_0p05_at_edge_ok", dict(deltas={"amp": (0.05, 0.05)}), []),
    ("two_guardrails_at_once", dict(deltas={"abs": (3.0, 3.0), "amp": (0.5, 0.5)}), ["G1", "G6"]),
]


@pytest.mark.parametrize("case", GUARD_CASES, ids=[c[0] for c in GUARD_CASES])
def test_universal_guardrail_trips_while_the_gate_passes(tmp_path, case):
    _, extra, tripped = case
    deltas = {"ra": (-3.0, -3.0)}
    deltas.update(extra.get("deltas", {}))
    spec = var(deltas, extra.get("sets"), extra.get("jit_mul"))
    g = one(tmp_path, "dt_nos", spec)
    assert g["gate_pass"] is True                                  # dRA = -3 mm: passes whatever the guardrails say
    assert g["guardrails_tripped"] == tripped
    assert g["guardrail_status"] == ("tripped" if tripped else "ok")
    if tripped:
        assert g["verdict"] == "PASS gate, NOT ADOPTED: guardrail " + ", ".join(tripped)
        assert g["better_than_base"] is False
    else:
        assert g["verdict"] == "PASS; no guardrail tripped" and g["better_than_base"] is True


def test_guardrail_values_are_the_hand_computed_ones(tmp_path):
    spec = var({"ra": (-3.0, -3.0), "abs": (2.5, 2.5), "ra_l": (1.2, 1.2), "ep_g": (1, 0), "amp": (0.06, 0.06)},
               {"root_ratio": (0.85, 0.95)}, {"jit_pred_mm": 1.06})
    g = one(tmp_path, "dt_nos", spec)
    gs = {x["id"]: x for x in g["guardrails"]}
    assert gs["G1"]["value"] == pytest.approx(2.5) and gs["G2"]["value"] == pytest.approx(1.2)
    assert gs["G3a"]["value"] == pytest.approx(0.06)                       # 9.2 x 1.06 / 9.2 - 1
    assert gs["G3b"]["value"] == pytest.approx(0.0)
    assert gs["G4a"]["value"] == pytest.approx(0.9) and gs["G4a"]["tripped"] is False   # mean(0.85, 0.95) = 0.9
    assert gs["G4a"]["per_seed"] == {"3407": 0.85, "3408": 0.95}
    assert gs["G5"]["value"] == pytest.approx(0.5) and gs["G5"]["tripped"] is True      # (1 + 0) / 2
    assert gs["G6"]["value"] == pytest.approx(0.06)


def test_fewer_failure_episodes_do_not_trip_the_failure_guardrail(tmp_path):
    root = world(tmp_path, {"dt_nos": var({"ra": (-3.0, -3.0)})})
    for s in (3407, 3408):                                                # the baseline had one episode on each seed
        sp = copy.deepcopy(BASE[s])
        sp["ep_l"] = 1
        write_run(root, BASE_NAME, s, sp)
    g = report(root, ["dt_nos"])["gates"]["dt_nos"]
    assert {x["id"]: x for x in g["guardrails"]}["G5"]["value"] == pytest.approx(-1.0)     # 0 - 1 on both seeds
    assert g["guardrails_tripped"] == [] and g["verdict"] == "PASS; no guardrail tripped"


# ------------------------------------------------------------------------------------ seeds
def test_pairing_is_by_seed_not_by_position(tmp_path):
    # the candidate lacks 3407: its 3408 run must be paired with the baseline's 3408 (RA 14.4), not 3407 (14.0)
    arm = {3408: copy.deepcopy(BASE[3408])}
    arm[3408]["ra"] = 13.0
    rep = report(world(tmp_path, {"dt_nos": arm}), ["dt_nos"])
    g = rep["gates"]["dt_nos"]
    assert g["paired"]["ra"]["per_seed"] == {"3408": pytest.approx(13.0 - 14.4)}
    assert g["seeds"]["missing_gate_seeds"] == ["3407"] and g["complete"] is False
    assert g["gate_pass"] is None and g["verdict"].startswith("INCOMPLETE")
    assert "dt_nos seed 3407: no evaluated run" in g["verdict"]
    assert g["paired"]["ra"]["gate_mean"] is None


def test_seed_3409_never_enters_the_gate_and_is_reported_separately(tmp_path):
    spec = var({"ra": (-1.2, -1.0, 10.0)}, seeds=SEEDS)                    # seed 3409 is 10 mm worse
    g = one(tmp_path, "dt_nos", spec)
    assert g["gate_pass"] is True and g["pending_3409"] is True            # decided by 3407 / 3408 only: mean -1.1
    c = g["confirmation"]
    assert c["available"] is True and c["seed"] == 3409
    assert c["diffs"]["ra"] == pytest.approx(10.0)                         # 3409: 14.8 + 10 - 14.8
    assert c["same_sign_as_mean"]["ra"] is False                           # +10 against the 2-seed mean -1.1
    assert c["three_seed_mean"]["ra"] == pytest.approx((-1.2 - 1.0 + 10.0) / 3)
    assert c["supports"] is False and g["label_text"] == PENDING           # the label is kept: no numeric rule
    assert g["paired"]["ra"]["per_seed"]["3409"] == pytest.approx(10.0)


def test_confirmation_that_agrees_in_sign(tmp_path):
    g = one(tmp_path, "dt_nos", var({"ra": (-1.2, -1.0, -1.0)}, seeds=SEEDS))
    c = g["confirmation"]
    assert c["supports"] is True and c["same_sign_as_mean"]["ra"] is True
    assert c["three_seed_mean"]["ra"] == pytest.approx(-3.2 / 3)


def test_missing_3409_is_reported_and_the_pending_label_stays(tmp_path):
    rep = report(world(tmp_path, {"dt_nos": var({"ra": (-1.2, -1.0)})}), ["dt_nos"])
    g = rep["gates"]["dt_nos"]
    c = g["confirmation"]
    assert g["pending_3409"] is True and g["label_text"] == PENDING
    assert (c["available"], c["arm_has_seed"], c["base_has_seed"]) == (False, False, True)
    assert c["diffs"]["ra"] is None and c["three_seed_mean"]["ra"] is None and c["supports"] is None
    assert "not available (arm: no, base: yes)" in SR.render_markdown(rep)
    assert "3409" not in g["paired"]["ra"]["per_seed"]


def test_a_baseline_without_3409_makes_the_confirmation_unavailable_too(tmp_path):
    root = world(tmp_path, {"dt_nos": var({"ra": (-1.2, -1.0, -1.0)}, seeds=SEEDS)}, base_seeds=(3407, 3408))
    c = report(root, ["dt_nos"])["gates"]["dt_nos"]["confirmation"]
    assert (c["available"], c["arm_has_seed"], c["base_has_seed"]) == (False, True, False)


# ------------------------------------------------------------------------------------ jitter
def test_missing_jitter_block_is_na_and_the_guardrail_check_incomplete(tmp_path):
    root = world(tmp_path, {"dt_nos": var({"ra": (-3.0, -3.0)}, drop_jitter=True)}, base_jit=False)
    rep = report(root, ["dt_nos"])
    r = rep["runs"]["dt_nos_s3407"]
    assert r["jitter"] is None and r["jitter_src"] is None
    g = rep["gates"]["dt_nos"]
    assert g["gate_pass"] is True and g["guardrail_status"] == "incomplete"
    assert g["guardrails_unknown"] == ["G3a", "G3b"] and g["guardrails_tripped"] == []
    assert g["verdict"] == "PASS; guardrail check incomplete (n/a: G3a, G3b)" and g["better_than_base"] is None
    assert g["paired"]["jitter.acc_err_mm"]["gate_mean"] is None
    md = SR.render_markdown(rep)
    jit_rows = [ln for ln in md.splitlines() if ln.startswith("| dt_nos_s3407 |") and "n/a | n/a | n/a" in ln]
    assert jit_rows, "the jitter table must render n/a for a run without a jitter block"


def test_jitter_missing_only_in_the_candidate_is_incomplete_too(tmp_path):
    g = one(tmp_path, "dt_nos", var({"ra": (-3.0, -3.0)}, drop_jitter=True))
    assert g["guardrails_unknown"] == ["G3a", "G3b"] and g["gate_pass"] is True
    assert "dt_nos seed 3407: jitter.jit_pred_mm n/a" in " ".join(next(
        x for x in g["guardrails"] if x["id"] == "G3a")["missing"])


def test_offline_jitter_fills_a_run_without_a_block_and_is_marked(tmp_path):
    root = world(tmp_path, {"dt_nos": var({"ra": (-3.0, -3.0)})}, base_jit=False)
    off = {"runs": {
        "dt_base_s3407": {"npz_tag": "last_tf_pert", "overall": {
            "n_frames": 100, "jit_pred_mm": 9.0, "acc_err_mm": 8.0, "acc_ratio": 2.0, "rot_acc_pred_deg": 3.0,
            "shap_transl_mm": 1.0, "ra_mpjpe_mm": 14.0, "jit_ratio_pooled": 1.2, "n_steps": 5}},
        "dt_base_s3408": {"npz_tag": "last_tf_pert", "overall": {"n_frames": 100, "jit_pred_mm": 9.4,
                                                                  "acc_err_mm": 8.4}},
        "dt_base_s3409": {"npz_tag": "selected", "overall": {"n_frames": 100, "jit_pred_mm": 1.0}},   # not `last`
    }}
    p = tmp_path / "off.json"
    p.write_text(json.dumps(off))
    rep = SR.build_report(root, BASE_NAME, ["dt_nos"], offline_json=p)
    r = rep["runs"]["dt_base_s3407"]
    assert r["jitter_src"] == "offline"
    assert set(r["jitter"]) == {"jit_pred_mm", "acc_err_mm", "acc_ratio", "rot_acc_pred_deg"}   # only jitter_decomp keys
    assert rep["runs"]["dt_base_s3409"]["jitter"] is None                  # a `selected` attribution is not `last`
    assert rep["runs"]["dt_nos_s3407"]["jitter_src"] == "eval"             # the evaluation's own block wins
    g = rep["gates"]["dt_nos"]
    assert g["guardrail_status"] == "ok"                                   # arm and base jitter now both known
    # a frame-count mismatch disables the fallback
    off["runs"]["dt_base_s3407"]["overall"]["n_frames"] = 99
    p.write_text(json.dumps(off))
    assert SR.build_report(root, BASE_NAME, ["dt_nos"], offline_json=p)["runs"]["dt_base_s3407"]["jitter"] is None


# ------------------------------------------------------------------------------------ descriptive blocks
def test_dt_trnos_interaction_arithmetic(tmp_path):
    # RA   base 14.0 / 14.4, tr 13.0 / 13.2, nos 13.5 / 14.0, trnos 12.0 / 12.4
    #      seed 3407: (12.0 - 13.5) - (13.0 - 14.0) = -1.5 + 1.0 = -0.5;  seed 3408: (12.4 - 14.0) - (13.2 - 14.4) = -0.4
    # ABS  base 60 / 62, tr 52 / 55, nos 57 / 58, trnos 47 / 51
    #      seed 3407: (47 - 57) - (52 - 60) = -2;  seed 3408: (51 - 58) - (55 - 62) = 0
    arms = {"dt_tr": var(sets={"ra": (13.0, 13.2), "abs": (52.0, 55.0)}),
            "dt_nos": var(sets={"ra": (13.5, 14.0), "abs": (57.0, 58.0)}),
            "dt_trnos": var(sets={"ra": (12.0, 12.4), "abs": (47.0, 51.0)})}
    rep = report(world(tmp_path, arms), ["dt_trnos"])                  # dt_tr / dt_nos are loaded as helpers
    assert rep["meta"]["helper_arms"] == ["dt_tr", "dt_nos"] and "dt_tr_s3407" not in rep["runs"]
    f = rep["factorial"]
    ra, ab = f["metrics"]["ra"], f["metrics"]["abs"]
    assert ra["per_seed"]["3407"]["interaction"] == pytest.approx(-0.5)
    assert ra["per_seed"]["3408"]["interaction"] == pytest.approx(-0.4)
    assert ra["gate_mean"]["interaction"] == pytest.approx(-0.45)
    assert ab["per_seed"]["3407"]["interaction"] == pytest.approx(-2.0)
    assert ab["per_seed"]["3408"]["interaction"] == pytest.approx(0.0)
    assert ab["gate_mean"]["interaction"] == pytest.approx(-1.0)
    s = ra["per_seed"]["3407"]
    assert (s["effect_tr"], s["effect_nos"], s["effect_both"]) == pytest.approx((-1.0, -0.5, -2.0))
    assert s["interaction"] == pytest.approx(s["effect_both"] - s["effect_tr"] - s["effect_nos"])
    assert ra["gate_mean"]["effect_tr"] == pytest.approx(-1.1) and ra["gate_mean"]["effect_both"] == pytest.approx(-2.0)
    g = rep["gates"]["dt_trnos"]
    assert g["kind"] == "descriptive" and "gate_pass" not in g and "no gate" in g["verdict"]
    md = SR.render_markdown(rep)
    assert "2 x 2 factorial" in md and "-0.450" in md and "-1.000" in md


def test_dt_trnos_needs_all_four_arms_per_seed_and_never_uses_3409_for_the_mean(tmp_path):
    arms = {"dt_tr": var(sets={"ra": (13.0, 13.2, 99.0)}, seeds=SEEDS),
            "dt_nos": var(sets={"ra": (13.5, 14.0)}),                      # no 3409: that seed cannot be formed
            "dt_trnos": var(sets={"ra": (12.0, 12.4, 99.0)}, seeds=SEEDS)}
    f = report(world(tmp_path, arms), ["dt_trnos"])["factorial"]
    assert sorted(f["metrics"]["ra"]["per_seed"]) == ["3407", "3408"]
    assert f["metrics"]["ra"]["gate_mean"]["interaction"] == pytest.approx(-0.45)
    # a missing gate seed in one cell leaves the mean undefined
    arms2 = {"dt_tr": var(sets={"ra": (13.0,)}, seeds=(3407,)), "dt_nos": var(sets={"ra": (13.5, 14.0)}),
             "dt_trnos": var(sets={"ra": (12.0, 12.4)})}
    f2 = report(world(tmp_path / "b", arms2), ["dt_trnos"])["factorial"]
    assert sorted(f2["metrics"]["ra"]["per_seed"]) == ["3407"] and f2["metrics"]["ra"]["gate_mean"] is None


def test_dt_cam_is_compared_with_dt_nos_descriptively(tmp_path):
    arms = {"dt_nos": var({"ra": (-1.0, -1.0), "abs": (-3.0, -3.0)}), "dt_cam": var({"ra": (-2.0, -2.4), "abs": (-3.0, -3.0)})}
    rep = report(world(tmp_path, arms), ["dt_cam", "dt_nos"])
    c = rep["cam_vs_nos"]["metrics"]
    assert c["ra"]["per_seed"] == {"3407": pytest.approx(-1.0), "3408": pytest.approx(-1.4)}
    assert c["ra"]["gate_mean"] == pytest.approx(-1.2) and c["ra"]["reading"] == "better, in the 1.1-2.5 mm band"
    assert c["abs"]["gate_mean"] == pytest.approx(0.0) and c["abs"]["reading"] == "tie"
    assert "dt_cam against dt_nos" in SR.render_markdown(rep)


def test_an_arm_without_a_gate_is_reported_with_paired_differences_only(tmp_path):
    arms = {"rt_foo": var({"ra": (-5.0, -5.0), "abs": (-9.0, -9.0)})}
    rep = report(world(tmp_path, arms), ["rt_foo"])
    g = rep["gates"]["rt_foo"]
    assert g["kind"] == "none" and "clauses" not in g and g["rule"] is None
    assert "no gate registered" in g["verdict"]
    assert g["paired"]["abs"]["gate_mean"] == pytest.approx(-9.0)
    assert "rt_foo_s3407" in rep["runs"]
    md = SR.render_markdown(rep)
    assert "| rt_foo_s3407 |" in md and "no gate registered" in md and "dABS -9.00 mm" in md


# ------------------------------------------------------------------------------------ failing loudly, determinism
def test_missing_baseline_fails_loudly(tmp_path, capsys):
    root = Path(tmp_path) / "semkine"
    write_run(root, "dt_nos", 3407, BASE[3407])
    (root / "dt_base_s3407").mkdir()                                  # a baseline directory without its evaluation
    with pytest.raises(SR.ScreenError) as e:
        SR.build_report(root, "dt_base", ["dt_nos"], offline_json=None)
    msg = str(e.value)
    assert "baseline arm 'dt_base' has no evaluated run" in msg and "dt_base_s<seed>/" + SR.EVAL_NAME in msg
    assert "[3407]" in msg                                           # the directory that lacks the json is named
    code = SR.main(["--runs-dir", str(root), "--arms", "dt_nos", "--out-prefix", str(tmp_path / "o" / "r")])
    assert code == 2 and "screen_report: ERROR: baseline arm 'dt_base' has no evaluated run" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()                             # nothing is written


def test_missing_runs_directory_and_unreadable_json_fail_loudly(tmp_path):
    with pytest.raises(SR.ScreenError, match="does not exist"):
        SR.build_report(tmp_path / "nope", "dt_base", [])
    root = world(tmp_path)
    (root / "dt_nos_s3407").mkdir()
    (root / "dt_nos_s3407" / SR.EVAL_NAME).write_text("{ truncated")
    with pytest.raises(SR.ScreenError, match="cannot read"):
        report(root, ["dt_nos"])
    (root / "dt_nos_s3407" / SR.EVAL_NAME).write_text(json.dumps({"model": {}}))
    with pytest.raises(SR.ScreenError, match="not an evalx evaluation json"):
        report(root, ["dt_nos"])


def test_partial_baseline_warns_and_the_gates_are_incomplete(tmp_path):
    root = world(tmp_path, {"dt_nos": var({"ra": (-3.0, -3.0)})}, base_seeds=(3407,))
    rep = report(root, ["dt_nos"])
    assert any("lacks gate seed(s) [3408]" in w for w in rep["warnings"])
    assert rep["gates"]["dt_nos"]["gate_pass"] is None


def test_output_is_deterministic_and_valid_json(tmp_path):
    root = world(tmp_path, {"dt_nos": var({"ra": (-1.2, -1.0)}), "dt_tr": var({"abs": (-6.0, -5.0)})})
    outs = []
    for i in range(2):
        rep = report(root, ["dt_nos", "dt_tr"])
        md, js = SR.write_outputs(rep, tmp_path / f"out{i}" / "rep")
        outs.append((md.read_bytes(), js.read_bytes()))
    assert outs[0] == outs[1]
    d = json.loads(outs[0][1])                                        # strict JSON: no NaN / Infinity
    assert d["gates"]["dt_nos"]["pending_3409"] is True and d["meta"]["gate_seeds"] == ["3407", "3408"]
    assert "NaN" not in outs[0][1].decode() and "Infinity" not in outs[0][1].decode()
    assert "202" not in outs[0][0].decode().split("Reading guide")[0]  # no timestamp in the header


def test_default_arm_discovery_skips_2k_arms_and_lists_missing_arms_once(tmp_path):
    root = world(tmp_path, {"dt_tr": var({"abs": (-6.0, -6.0)}), "dt_foo": var(), "dt_tr_2k": var()})
    write_run(root, "dt_nos", 3407, BASE[3407], ckpt=False)
    (root / "dt_cam_s3407").mkdir()                                  # trained but not evaluated yet
    assert SR.discover_dt_arms(root) == ["dt_base", "dt_foo", "dt_nos", "dt_tr"]      # no *_2k, no unevaluated dt_cam
    rep = SR.build_report(root, BASE_NAME, None, offline_json=None, want_params=False)
    arms = rep["meta"]["arms"]
    n = len(SR.DEFAULT_ARMS)
    assert n == 13 and SR.DEFAULT_ARMS[6:8] == ("dt_trnos", "dt_trdz")          # section 9.6 added dt_trdz
    assert arms[:n] == list(SR.DEFAULT_ARMS) and arms[n:] == ["dt_foo"] and "dt_tr_2k" not in arms
    warn = [w for w in rep["warnings"] if w.startswith("arms without an evaluated run")]
    assert len(warn) == 1 and "dt_cam (directories without the json: s3407)" in warn[0] and "dt_w05" in warn[0]
    assert rep["gates"]["dt_tr"]["gate_pass"] is True and rep["gates"]["dt_foo"]["kind"] == "none"
    assert rep["gates"]["dt_nos"]["verdict"].startswith("INCOMPLETE")
    assert "dt_cam | " in SR.render_markdown(rep) and "no evaluated run" in SR.render_markdown(rep)


def test_the_baseline_is_not_its_own_candidate_and_a_step_other_than_6000_warns(tmp_path):
    root = world(tmp_path)
    write_run(root, "dt_nos", 3407, BASE[3407], step=2000)
    rep = report(root, ["dt_base", "dt_nos", "dt_nos"])
    assert rep["meta"]["arms"] == ["dt_nos"]
    assert any("dt_nos_s3407 was evaluated at step 2000" in w for w in rep["warnings"])
    assert any("is the baseline and is not a candidate" in w for w in rep["warnings"])


# ------------------------------------------------------------------------------------ parameters
def test_parameter_count_excludes_bn_statistics_mano_and_teacher(tmp_path):
    ck = tmp_path / "x.ckpt"
    write_ckpt(ck, 1234)
    sd = torch.load(ck, map_location="cpu")["state_dict"]
    assert sum(v.numel() for v in sd.values()) == 1234 + 11 + 11 + 1 + 12 + 77
    assert SR.count_params(sd) == 1234
    root = world(tmp_path, {"dt_w05": var(params=321)})
    assert report(root, ["dt_w05"])["runs"]["dt_w05_s3407"]["params"] == 321


def test_checkpoint_is_found_the_way_evalx_find_ckpt_does(tmp_path):
    d = tmp_path / "r_s1"
    d.mkdir()
    for n in ("r_s1-step=500.ckpt", "r_s1-step=1000.ckpt", "r_s1-step=6000.ckpt", "r_s1-step=6000-v1.ckpt", "last.ckpt"):
        write_ckpt(d / n, 10)
    ck, st = SR.find_ckpt_last(d)
    assert (ck.name, st) == ("r_s1-step=6000.ckpt", 6000)             # the largest step, not the lexicographic last
    sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools"), str(REPO / "tools" / "tracking")]
    try:
        import evalx as EX
    except Exception:                                                  # noqa: BLE001
        EX = None
    if EX is not None:
        assert EX.find_ckpt(d, "last")[0].name == ck.name and EX.find_ckpt(d, "last")[1] == st
    only_last = tmp_path / "o_s1"
    only_last.mkdir()
    write_ckpt(only_last / "last.ckpt", 10)
    assert SR.find_ckpt_last(only_last) == (only_last / "last.ckpt", None)
    assert SR.find_ckpt_last(tmp_path / "nothing") == (None, None)


def test_a_run_without_a_readable_checkpoint_has_na_parameters_and_a_warning(tmp_path):
    root = world(tmp_path, {"dt_nos": var({"ra": (-3.0, -3.0)})}, ckpt=False)
    (root / "dt_nos_s3407" / "dt_nos_s3407-step=6000.ckpt").write_text("not a checkpoint")
    rep = report(root, ["dt_nos"])
    assert rep["runs"]["dt_nos_s3407"]["params"] is None and rep["runs"]["dt_base_s3407"]["params"] is None
    assert any("dt_nos_s3407: parameters n/a (checkpoint unreadable" in w for w in rep["warnings"])
    assert any("dt_base_s3407: parameters n/a (no checkpoint" in w for w in rep["warnings"])
    assert report(root, ["dt_nos"], want_params=False)["warnings"] == []


# ------------------------------------------------------------------------------------ markdown and CLI
def test_cli_writes_markdown_and_json_with_the_reading_guide_and_all_columns(tmp_path, capsys):
    root = world(tmp_path, {"dt_nos": var({"ra": (-1.2, -1.0)}), "rt_foo": var()})
    lat = tmp_path / "lat.json"
    lat.write_text(json.dumps({BASE_NAME: 10.0, "dt_nos": 9.5}))
    code = SR.main(["--runs-dir", str(root), "--arms", "dt_nos", "rt_foo", "--out-prefix", str(tmp_path / "out" / "rep"),
                    "--no-offline", "--latency-json", str(lat)])
    out = capsys.readouterr().out
    assert code == 0 and "rep.md" in out and "rep.json" in out
    assert "PASS, pending 3409 confirmation; no guardrail tripped" in out and "rt_foo" in out
    md = (tmp_path / "out" / "rep.md").read_text()
    head = md.split("## Summary")[0]
    for phrase in ("Reading guide", "d(m, s) = m(arm, s) - m(base, s)", "seeds 3407 and 3408", "seed 3409",
                   "|difference| < 1.1 mm is a tie", "2.5-3 mm", "[1.1, 2.5)", PENDING,
                   "both the development set and the test set"):
        assert phrase in head, phrase
    for col in ("RA | RA glob | RA loc", "root rot (deg)", "abs transl (mm)", "ABS | ABS glob | ABS loc", "TF RA",
                "amp (RA)", "fail eps", "root spd ratio (glob)", "finger spd ratio (g/l mean)", "ret k1", "ret k5",
                "ret k20", "params", "grid median RA", "latency (ms)", "jit gt", "jit pred", "acc err",
                "acc err only transl", "acc err allbut transl", "rot acc pred (deg)"):
        assert col in md, col
    assert "| dt_base_s3407 (base) |" in md and "**dt_nos mean(3407, 3408)**" in md
    assert "Verdict: PASS, pending 3409 confirmation; no guardrail tripped" in md
    js = json.loads((tmp_path / "out" / "rep.json").read_text())
    assert js["meta"]["latency_ms"] == {BASE_NAME: 10.0, "dt_nos": 9.5} and js["meta"]["offline_json"] is None
    assert md.endswith("\n") and "None" not in md
    heads, rows = md_tables(md)[0]                                                 # the summary table; every table is rectangular
    assert heads[-2:] == ["verdict", "wording"] and {r[0]: r[-1] for r in rows}["dt_nos"] == PENDING


def test_markdown_tables_are_separated_from_following_text(tmp_path):
    rep = report(world(tmp_path, {"dt_nos": var({"ra": (-3.0, -3.0)})}), ["dt_nos"])
    lines = SR.render_markdown(rep).splitlines()
    for i, ln in enumerate(lines[:-1]):
        if ln.startswith("|") and not lines[i + 1].startswith("|"):
            assert lines[i + 1] == "", f"a table row is followed directly by text: {lines[i + 1][:60]!r}"


# ------------------------------------------------------------------------------------ recorded runs (smoke)
RECORDED = REPO / "outputs" / "semkine"
needs_recorded = pytest.mark.skipif(not (RECORDED / "rt_cnntrack_s3407" / SR.EVAL_NAME).is_file()
                                    or not (RECORDED / "rt_cnn_s3407" / SR.EVAL_NAME).is_file()
                                    or not (RECORDED / "rt_s37_s3407" / SR.EVAL_NAME).is_file(),
                                    reason="recorded evaluations are not on this machine")


@needs_recorded
def test_smoke_on_recorded_runs_renders_every_column_without_gates(tmp_path):
    rep = SR.build_report(RECORDED, "rt_cnntrack", ["rt_cnn", "rt_s37"])
    assert rep["meta"]["arms"] == ["rt_cnn", "rt_s37"]
    assert all(g["kind"] == "none" for g in rep["gates"].values())          # not DT arms: no gate
    r = rep["runs"]["rt_cnntrack_s3407"]
    assert r["m"]["ra"] == 13.69480423503861                              # the preregistration's reproducibility constant
    assert r["params"] == 11209429 and r["m"]["amp"] == pytest.approx(1.3649804799377536)
    assert rep["runs"]["rt_s37_s3407"]["params"] == 733830 and rep["runs"]["rt_cnn_s3407"]["params"] == 11202732
    assert r["m"]["root_ratio"] == pytest.approx(1.5246732722199778)
    assert r["m"]["finger_ratio"] == pytest.approx((0.3267323970794678 + 0.4565676748752594) / 2)
    assert r["jitter"] is None or r["jitter_src"] in ("eval", "offline")      # old run: offline fallback or n/a
    assert {"rt_cnntrack_s3407", "rt_cnntrack_s3408"} <= set(rep["runs"])
    assert all(rep["runs"][k]["step"] == 6000 for k in rep["runs"])
    md_path, js_path = SR.write_outputs(rep, tmp_path / "smoke")
    md = md_path.read_text()
    for col in ("RA glob", "ABS loc", "TF RA", "amp (RA)", "fail eps", "root spd ratio", "finger spd ratio", "ret k20",
                "params", "grid median RA", "jit pred", "acc err"):
        assert col in md
    assert "| rt_cnn_s3409 |" in md and "| rt_s37_s3409 |" in md
    json.loads(js_path.read_text())
    md_tables(md)                                                          # every table of the real report is rectangular
    raw10 = json.loads((RECORDED / "rt_cnntrack_s3407" / SR.EVAL_NAME).read_text())[
        "perturb"]["10"]["retention_k1_k2_k5_k10_k20"]
    want = (raw10[0], raw10[2], raw10[4])                                   # k1 / k5 / k20 of the 10 degree list
    assert len(set(want)) == 3 and (r["m"]["ret10_k1"], r["m"]["ret10_k5"], r["m"]["ret10_k20"]) == want
    row = table_row(md, "ret k5", "rt_cnntrack_s3407 (base)")
    assert (row["ret k1 (10deg)"], row["ret k5"], row["ret k20"]) == tuple(f"{v:.3f}" for v in want)


@needs_recorded
def test_smoke_values_agree_with_evalx_aggregate_runs_and_the_recorded_main_row(tmp_path):
    runs = [RECORDED / "rt_cnntrack_s3407", RECORDED / "rt_cnntrack_s3408"]
    sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools"), str(REPO / "tools" / "tracking")]
    try:
        import evalx as EX
        per_seed, _, mean, _ = EX.aggregate_runs([str(r) for r in runs], "val_core", "last", "tf_pert")
    except Exception as e:                                                  # noqa: BLE001
        pytest.skip(f"evalx.aggregate_runs not usable here: {type(e).__name__}")
    rep = SR.build_report(RECORDED, "rt_cnntrack", [], want_params=False)
    for seed, ref in per_seed.items():
        m = rep["runs"][f"rt_cnntrack_s{seed}"]["m"]
        assert m["ra"] == pytest.approx(ref["overall"]["mpjpe_ra_mm"], rel=1e-12)
        assert m["ra_global"] == pytest.approx(ref["global"]["mpjpe_ra_mm"], rel=1e-12)
        assert m["ra_local"] == pytest.approx(ref["local"]["mpjpe_ra_mm"], rel=1e-12)
        assert m["abs_global"] == pytest.approx(ref["global"]["mpjpe_abs_mm"], rel=1e-12)
        assert m["abs_local"] == pytest.approx(ref["local"]["mpjpe_abs_mm"], rel=1e-12)
        assert m["rot"] == pytest.approx(ref["overall"]["root_rot_deg"], rel=1e-12)
        assert m["transl"] == pytest.approx(ref["overall"]["transl_mm"], rel=1e-12)
    am = rep["arm_means"]["rt_cnntrack"]["m"]
    assert am["ra"] == pytest.approx(mean["overall"]["mpjpe_ra_mm"], rel=1e-12)
    assert am["abs_local"] == pytest.approx(mean["local"]["mpjpe_abs_mm"], rel=1e-12)
    row = RECORDED / "rt_cnntrack_tf_pert_main_row.json"
    if row.is_file():
        r = json.loads(row.read_text())
        assert am["ra"] == pytest.approx(r["two_seed_mean"]["overall"]["mpjpe_ra_mm"], rel=1e-12)
        full = SR.build_report(RECORDED, "rt_cnntrack", [], want_params=True)
        assert full["runs"]["rt_cnntrack_s3407"]["params"] == r["params_total"]   # == sum(p.numel() for p in parameters())


DTCHK = "evalx_val_core_last_dtchk.json"          # a real new-style evaluation (jitter block, no --tf / --perturb)
needs_dtchk = pytest.mark.skipif(not (RECORDED / "rt_cnntrack_s3407" / DTCHK).is_file(),
                                 reason="the recorded new-style evaluation is not on this machine")


@needs_dtchk
def test_a_recorded_new_style_json_with_a_jitter_block_is_read_as_the_evaluations_own(tmp_path):
    rep = SR.build_report(RECORDED, "rt_cnntrack", [], eval_name=DTCHK, want_params=False)
    r = rep["runs"]["rt_cnntrack_s3407"]
    raw = json.loads((RECORDED / "rt_cnntrack_s3407" / DTCHK).read_text())["model"]["jitter"]
    assert r["jitter_src"] == "eval" and r["jitter"]["acc_err_mm"] == raw["acc_err_mm"]
    assert r["jitter"]["jit_pred_mm"] == raw["jit_pred_mm"] and set(r["jitter"]) == set(raw)
    assert r["m"]["tf_ra"] is None and r["m"]["amp"] is None and r["m"]["ret10_k1"] is None   # no --tf / --perturb
    off = SR.load_offline(SR.DEFAULT_OFFLINE)
    if off is not None and "rt_cnntrack_s3407" in off["runs"]:           # the offline attribution agrees on the shared keys
        o = SR.offline_jitter(off, "rt_cnntrack_s3407", r["n_frames"])
        assert o["acc_err_mm"] == pytest.approx(raw["acc_err_mm"], abs=1e-3)
        assert o["jit_pred_mm"] == pytest.approx(raw["jit_pred_mm"], abs=1e-3)
    md_path, _ = SR.write_outputs(rep, tmp_path / "dtchk")
    assert "| rt_cnntrack_s3407 (base) |" in md_path.read_text() and "eval" in md_path.read_text()


@needs_dtchk
def test_cli_eval_json_option_selects_the_evaluation_file(tmp_path, capsys):
    code = SR.main(["--runs-dir", str(RECORDED), "--base", "rt_cnntrack", "--arms", "--eval-json", DTCHK,
                    "--out-prefix", str(tmp_path / "x"), "--no-params"])
    assert code == 0 and (tmp_path / "x.md").is_file()
    assert json.loads((tmp_path / "x.json").read_text())["meta"]["eval_json"] == DTCHK


# ------------------------------------------------------------------------------------ dt_trdz (preregistration 9.6)
# RA   base 14.0 / 14.4 / 14.8 (seeds 3407 / 3408 / 3409);  tr 13.0 / 14.0 / 13.5;  dz 13.6 / 14.1 / 13.8;  trdz 12.6 / 13.4 / 13.0
# ABS  base 60 / 62 / 64;                                   tr 52 / 56 / 55;        dz 57 / 60 / 59;        trdz 50 / 54 / 52
TRDZ_ARMS = {
    "dt_tr": var(sets={"ra": (13.0, 14.0, 13.5), "abs": (52.0, 56.0, 55.0)}, seeds=SEEDS),
    "dt_dz": var(sets={"ra": (13.6, 14.1, 13.8), "abs": (57.0, 60.0, 59.0)}, seeds=SEEDS),
    "dt_trdz": var(sets={"ra": (12.6, 13.4, 13.0), "abs": (50.0, 54.0, 52.0)}, seeds=SEEDS),
}


def test_the_factorial_registry_follows_the_preregistration():
    assert SR.FACTORIALS == {"dt_trnos": ("dt_tr", "dt_nos"), "dt_trdz": ("dt_tr", "dt_dz")}
    assert set(SR.FACTORIALS) == set(SR.DESCRIPTIVE) and not set(SR.FACTORIALS) & set(SR.GATES)    # no gate for either
    assert SR.DEFAULT_ARMS.index("dt_trdz") == SR.DEFAULT_ARMS.index("dt_trnos") + 1
    assert SR.GATES["dt_nopm"] == ("nos", "C9a") and SR.GATES["dt_pmt"] == ("nos", "C9b")   # section 9.2: the C2 gate


def test_dt_trdz_interaction_arithmetic(tmp_path):
    rep = report(world(tmp_path, TRDZ_ARMS), ["dt_trdz"])             # dt_tr and dt_dz are loaded as helpers
    assert rep["meta"]["helper_arms"] == ["dt_tr", "dt_dz"] and list(rep["factorials"]) == ["dt_trdz"]
    f = rep["factorials"]["dt_trdz"]
    assert rep["factorial"] == f                                      # the earlier layout: the first (here only) block
    assert f["arms"] == {"base": "dt_base", "tr": "dt_tr", "nos": "dt_dz", "both": "dt_trdz"}
    assert f["labels"] == {"base": "base", "tr": "tr", "nos": "dz", "both": "trdz"}
    assert "(dt_trdz - dt_dz) - (dt_tr - dt_base)" in f["definition"]
    ra, ab = f["metrics"]["ra"], f["metrics"]["abs"]
    # RA 3407: (12.6 - 13.6) - (13.0 - 14.0) = -1.0 + 1.0 = 0.0;  3408: (13.4 - 14.1) - (14.0 - 14.4) = -0.7 + 0.4 = -0.3
    # RA 3409: (13.0 - 13.8) - (13.5 - 14.8) = -0.8 + 1.3 = +0.5 (reported, never in the mean)
    assert [ra["per_seed"][s]["interaction"] for s in ("3407", "3408", "3409")] == pytest.approx([0.0, -0.3, 0.5])
    assert ra["gate_mean"]["interaction"] == pytest.approx(-0.15)
    # ABS 3407: (50 - 57) - (52 - 60) = +1;  3408: (54 - 60) - (56 - 62) = 0;  3409: (52 - 59) - (55 - 64) = +2
    assert [ab["per_seed"][s]["interaction"] for s in ("3407", "3408", "3409")] == pytest.approx([1.0, 0.0, 2.0])
    assert ab["gate_mean"]["interaction"] == pytest.approx(0.5)
    s = ra["per_seed"]["3407"]
    assert (s["effect_tr"], s["effect_nos"], s["effect_both"]) == pytest.approx((-1.0, -0.4, -1.4))
    gm = ra["gate_mean"]
    assert (gm["effect_tr"], gm["effect_nos"], gm["effect_both"]) == pytest.approx((-0.7, -0.35, -1.2))
    assert gm["interaction"] == pytest.approx(gm["effect_both"] - gm["effect_tr"] - gm["effect_nos"])
    g = rep["gates"]["dt_trdz"]
    assert g["kind"] == "descriptive" and "gate_pass" not in g and "no gate" in g["verdict"]
    md = SR.render_markdown(rep)
    assert "### dt_trdz: 2 x 2 factorial (descriptive, no gate)" in md and "dt_trnos: 2 x 2" not in md
    heads, rows = table_with(md, "effect dz")
    assert heads == ["metric", "seed", "base", "tr", "dz", "trdz", "effect tr", "effect dz", "effect trdz", "interaction"]
    cells = {(r[0], r[1]): r[-1] for r in rows}
    assert cells[("RA", "**mean(3407, 3408)**")] == "**-0.150**" and cells[("ABS", "**mean(3407, 3408)**")] == "**+0.500**"
    assert cells[("RA", "3409")] == "+0.500" and cells[("ABS", "3409")] == "+2.000"
    assert "dt_dz: 3407, 3408, 3409" in md                            # the seeds line names the arms, not the generic keys


def test_both_factorials_are_reported_together_and_share_dt_tr(tmp_path):
    arms = {"dt_tr": var(sets={"ra": (13.0, 13.2), "abs": (52.0, 55.0)}),
            "dt_nos": var(sets={"ra": (13.5, 14.0), "abs": (57.0, 58.0)}),
            "dt_trnos": var(sets={"ra": (12.0, 12.4), "abs": (47.0, 51.0)}),
            "dt_dz": var(sets={"ra": (13.6, 14.1), "abs": (57.0, 60.0)}),
            "dt_trdz": var(sets={"ra": (12.6, 13.4), "abs": (50.0, 54.0)})}
    rep = report(world(tmp_path, arms), ["dt_trnos", "dt_trdz"])
    assert rep["meta"]["helper_arms"] == ["dt_tr", "dt_nos", "dt_dz"]
    assert list(rep["factorials"]) == ["dt_trnos", "dt_trdz"] and rep["factorial"] == rep["factorials"]["dt_trnos"]
    fn, fz = (rep["factorials"][a]["metrics"] for a in ("dt_trnos", "dt_trdz"))
    # trnos RA: (12.0 - 13.5) - (13.0 - 14.0) = -0.5 and (12.4 - 14.0) - (13.2 - 14.4) = -0.4;  ABS: -2 and 0
    assert fn["ra"]["gate_mean"]["interaction"] == pytest.approx(-0.45)
    assert fn["abs"]["gate_mean"]["interaction"] == pytest.approx(-1.0)
    # trdz RA: (12.6 - 13.6) - (13.0 - 14.0) = 0.0 and (13.4 - 14.1) - (13.2 - 14.4) = +0.5;
    # trdz ABS: (50 - 57) - (52 - 60) = +1 and (54 - 60) - (55 - 62) = +1
    assert fz["ra"]["gate_mean"]["interaction"] == pytest.approx(0.25)
    assert fz["abs"]["gate_mean"]["interaction"] == pytest.approx(1.0)
    md = SR.render_markdown(rep)
    assert md.index("### dt_trnos: 2 x 2") < md.index("### dt_trdz: 2 x 2")
    md_tables(md)


def test_dt_trdz_with_a_missing_cell_or_no_common_seed_has_no_mean_and_no_crash(tmp_path):
    arms = dict(TRDZ_ARMS, dt_dz=var(sets={"ra": (13.6,), "abs": (57.0,)}, seeds=(3407,)))     # no dt_dz seed 3408
    rep = report(world(tmp_path / "a", arms), ["dt_trdz"])
    ra = rep["factorials"]["dt_trdz"]["metrics"]["ra"]
    assert sorted(ra["per_seed"]) == ["3407"] and ra["gate_mean"] is None
    md = SR.render_markdown(rep)
    assert "dt_dz: 3407;" in md and "No seed is present in all four cells" not in md
    apart = dict(TRDZ_ARMS, dt_tr=var(sets={"ra": (13.0,), "abs": (52.0,)}, seeds=(3407,)),
                 dt_dz=var(sets={"ra": (14.1,), "abs": (60.0,)}, seeds=(3408,)))              # the cells never overlap
    rep2 = report(world(tmp_path / "b", apart), ["dt_trdz"])
    f2 = rep2["factorials"]["dt_trdz"]["metrics"]["ra"]
    assert f2["per_seed"] == {} and f2["gate_mean"] is None
    assert "No seed is present in all four cells" in SR.render_markdown(rep2)


def test_arm_discovery_matches_exact_run_directory_names_only(tmp_path):
    root = Path(tmp_path) / "semkine"
    for name in ("dt_tr_s3407", "dt_tr_s3407_bak", "dt_tr_s3408", "dt_trnos_s3407", "dt_trdz_s3407", "dt_tr_2k_s3407"):
        (root / name).mkdir(parents=True)
        (root / name / SR.EVAL_NAME).write_text("{}")
    have, without = SR.discover(root, "dt_tr")
    assert sorted(have) == [3407, 3408] and have[3407].name == "dt_tr_s3407" and without == []   # a backup copy is no run
    assert [sorted(SR.discover(root, a)[0]) for a in ("dt_trnos", "dt_trdz", "dt_tr_2k")] == [[3407]] * 3
    assert SR.discover_dt_arms(root) == ["dt_tr", "dt_trdz", "dt_trnos"]         # no *_2k arm, no `_bak` arm


# ------------------------------------------------------------------------------------ wording of an efficiency pass
def test_an_efficiency_arm_that_passes_on_cost_alone_claims_no_accuracy_gain(tmp_path):
    # RA 14.4 <= 1.02 x 14.2 = 14.484 and 250 <= 0.5 x 1000 parameters: E2 and E3 hold, E1 (dRA +0.2) does not
    rep = report(world(tmp_path, {"dt_w05": var({"ra": (0.2, 0.2)}, params=250)}), ["dt_w05"])
    g = rep["gates"]["dt_w05"]
    assert g["gate_pass"] is True and g["verdict"] == "PASS; no guardrail tripped" and g["pending_3409"] is False
    assert g["wording"].startswith("efficiency gate met") and "params <= 0.5x" in g["wording"]
    assert "RA non-inferior (<= 1.02x)" in g["wording"] and clause(g, "E2")["threshold"] == 1.02    # the margin of E2
    assert "latency" not in g["wording"] and "no accuracy gain is claimed" in g["wording"]
    assert g["wording"] != "better than base" and g["better_than_base"] is False
    assert "**Verdict: PASS; no guardrail tripped** (wording: efficiency gate met: RA non-inferior" in SR.render_markdown(rep)


def test_a_latency_pass_names_the_latency_clause(tmp_path):
    lat = tmp_path / "lat.json"
    lat.write_text(json.dumps({BASE_NAME: 10.0, "dt_l3": 7.0}))
    g = one(tmp_path, "dt_l3", var({"ra": (0.2, 0.2)}, params=600), latency_json=lat)     # params 0.6x fail, latency 0.7x
    assert g["gate_pass"] is True and "latency <= 0.8x" in g["wording"] and "params" not in g["wording"]
    both = one(tmp_path / "b", "dt_l3", var({"ra": (0.2, 0.2)}, params=400), latency_json=lat)
    assert "params <= 0.5x and latency <= 0.8x" in both["wording"]


def test_a_cost_pass_with_an_in_band_ra_gain_keeps_the_pending_label_in_the_wording(tmp_path):
    g = one(tmp_path, "dt_l3", var({"ra": (-1.5, -1.5)}, params=250))
    assert g["gate_pass"] is True and g["pending_3409"] is False         # the pass itself rests on the cost branch
    assert g["verdict"] == "PASS; no guardrail tripped"
    assert g["wording"].startswith("efficiency gate met") and PENDING in g["wording"] and "-1.50 mm" in g["wording"]
    assert g["better_than_base"] is None                                 # the accuracy claim is pending
    assert any("E1" in n for n in g["band"]["notes"])


def test_a_cost_pass_with_a_resolved_ra_gain_is_also_better(tmp_path):
    g = one(tmp_path, "dt_w05_kd", var({"ra": (-3.0, -3.0)}, params=250))
    assert g["wording"].startswith("efficiency gate met") and "RA also improved (mean dRA -3.00 mm)" in g["wording"]
    assert PENDING not in g["wording"] and g["better_than_base"] is True and g["pending_3409"] is False


def test_an_efficiency_arm_that_passes_on_its_ra_gain_alone_is_worded_like_the_other_families(tmp_path):
    pend = one(tmp_path / "a", "dt_w05", var({"ra": (-1.2, -1.0)}, params=900))        # cost clauses undecided or failed
    done = one(tmp_path / "b", "dt_w05", var({"ra": (-3.0, -3.0)}, params=900))
    assert (pend["gate_pass"], pend["wording"], pend["pending_3409"], pend["better_than_base"]) == (True, PENDING, True, None)
    assert (done["gate_pass"], done["wording"], done["pending_3409"], done["better_than_base"]) == (
        True, "better than base", False, True)


def test_a_cost_pass_with_a_tripped_or_unknown_guardrail(tmp_path):
    tripped = one(tmp_path / "a", "dt_l3", var({"ra": (0.2, 0.2), "abs": (3.0, 3.0)}, params=250))   # dABS +3 > +2: G1
    assert tripped["verdict"] == "PASS gate, NOT ADOPTED: guardrail G1" and tripped["wording"] == "not adopted (guardrail)"
    assert tripped["better_than_base"] is False
    unknown = one(tmp_path / "b", "dt_l3", var({"ra": (0.2, 0.2)}, drop_jitter=True, params=250))
    assert unknown["guardrail_status"] == "incomplete" and unknown["wording"].endswith("; guardrail check incomplete")
    assert unknown["better_than_base"] is False


def test_the_summary_table_shows_the_wording(tmp_path):
    arms = {"dt_nos": var({"ra": (-1.2, -1.0)}), "dt_w05": var({"ra": (0.2, 0.2)}, params=250),
            "dt_tr": var({"abs": (-6.0, -6.0)}), "dt_cam": var()}
    rep = report(world(tmp_path, arms), ["dt_nos", "dt_w05", "dt_tr", "dt_cam", "dt_so3c"])
    heads, rows = md_tables(SR.render_markdown(rep))[0]
    by = {r[0]: dict(zip(heads, r)) for r in rows}
    assert heads[-1] == "wording" and len(rows) == 5
    assert by["dt_nos"]["wording"] == PENDING and by["dt_tr"]["wording"] == "better than base"
    assert by["dt_w05"]["wording"].startswith("efficiency gate met") and by["dt_cam"]["wording"] == "tie"
    assert by["dt_so3c"]["seeds"] == "no evaluated run" and by["dt_so3c"]["wording"] == ""


# ------------------------------------------------------------------------------------ inputs that are not what they should be
def test_latency_json_must_be_an_object_and_bad_entries_are_reported(tmp_path, capsys):
    root = world(tmp_path, {"dt_w05": var({"ra": (0.2, 0.2)}, params=600)})
    lst = tmp_path / "lat_list.json"
    lst.write_text("[1, 2, 3]")
    with pytest.raises(SR.ScreenError, match="must hold an object"):
        report(root, ["dt_w05"], latency_json=lst)
    code = SR.main(["--runs-dir", str(root), "--arms", "dt_w05", "--out-prefix", str(tmp_path / "o" / "r"),
                    "--no-offline", "--latency-json", str(lst)])
    err = capsys.readouterr().err
    assert code == 2 and "screen_report: ERROR: the latency json" in err and "Traceback" not in err
    assert not (tmp_path / "o").exists()
    bad = tmp_path / "lat_bad.json"
    bad.write_text(json.dumps({BASE_NAME: "10 ms", "dt_w05": 7.9, "dt_l3": 0, "dt_x": -1.0, "dt_y": True, "dt_z": None}))
    rep = report(root, ["dt_w05"], latency_json=bad)
    assert rep["meta"]["latency_ms"] == {"dt_w05": 7.9}
    w = [x for x in rep["warnings"] if x.startswith("latency json")]
    assert len(w) == 1 and "dt_base, dt_l3, dt_x, dt_y, dt_z" in w[0] and "positive number" in w[0]
    g = rep["gates"]["dt_w05"]                                           # params 0.6x fails; the base latency was dropped
    assert g["efficiency"]["latency"] == "not measured" and g["gate_pass"] is None
    assert "latency not measured" in g["verdict"]
    bench = tmp_path / "bench.json"                                      # the nested layout of tools/s38/bench.py
    bench.write_text(json.dumps({"env": {}, "anchor_ms": [1.7, 1.8], "arms": {"dt_base": {"params": 1}}}))
    w = [x for x in report(root, ["dt_w05"], latency_json=bench)["warnings"] if x.startswith("latency json")]
    assert len(w) == 1 and "anchor_ms, arms, env" in w[0] and "bench.py" in w[0]


def test_a_missing_or_non_finite_step_is_unknown_not_a_crash(tmp_path):
    root = world(tmp_path, {"dt_nos": var({"ra": (-3.0, -3.0)})})
    for name, val in (("dt_nos_s3407", float("nan")), ("dt_nos_s3408", None)):
        p = root / name / SR.EVAL_NAME
        js = json.loads(p.read_text())
        js["step"] = val                                                 # NaN is written as the token NaN
        p.write_text(json.dumps(js))
    rep = report(root, ["dt_nos"])
    assert rep["runs"]["dt_nos_s3407"]["step"] is None and rep["runs"]["dt_nos_s3408"]["step"] is None
    assert any(w.startswith("dt_nos_s3407 was evaluated at step n/a") for w in rep["warnings"])
    md_path, js_path = SR.write_outputs(rep, tmp_path / "o")             # strict JSON (no NaN) can still be written
    md = md_path.read_text()
    json.loads(js_path.read_text())
    assert "None" not in md and "NaN" not in js_path.read_text()
    assert table_row(md, "ret k5", "dt_nos_s3407")["step"] == "n/a"
    # an integer-valued float is the step; a fractional one is not
    for val, want in ((6000.0, 6000), (5999.5, None), ("6000", None)):
        p = root / "dt_nos_s3407" / SR.EVAL_NAME
        js = json.loads(p.read_text())
        js["step"] = val
        p.write_text(json.dumps(js))
        assert report(root, ["dt_nos"])["runs"]["dt_nos_s3407"]["step"] == want


def test_missing_frame_counts_are_named_n_a_in_the_warnings(tmp_path):
    root = world(tmp_path, {"dt_nos": var({"ra": (-3.0, -3.0)})})
    p = root / "dt_nos_s3407" / SR.EVAL_NAME
    js = json.loads(p.read_text())
    del js["model"]["n_frames"]
    p.write_text(json.dumps(js))
    w = [x for x in report(root, ["dt_nos"])["warnings"] if "number of evaluated frames" in x]
    assert len(w) == 1 and "(n/a vs 100)" in w[0] and "None" not in w[0]


def test_unknown_parameters_print_na_not_none(tmp_path):
    root = world(tmp_path)                                               # the baseline has checkpoints: 1000 parameters
    for s, sp in var({"ra": (0.2, 0.2)}, params=250).items():
        write_run(root, "dt_w05", s, sp, ckpt=False)                     # the arm has none
    rep = report(root, ["dt_w05"])
    g = rep["gates"]["dt_w05"]
    assert g["efficiency"]["params_arm"] is None and g["efficiency"]["params_base"] == 1000
    assert g["gate_pass"] is None                                        # RA is within 1.02x but no cost clause is decided
    md = SR.render_markdown(rep)
    assert "- parameters: arm n/a, base 1000; latency: not measured" in md and "None" not in md


def test_md_table_prints_na_for_a_missing_cell():
    assert SR.md_table(["a", "b"], [["x", None]]).splitlines()[-1] == "| x | n/a |"
    assert SR.na(None) == "n/a" and SR.na(0) == 0 and SR.na("") == ""


def test_mixed_jitter_sources_are_warned_about(tmp_path):
    def offline(arms):
        return {"runs": {f"{a}_s{s}": {"npz_tag": "last", "overall": {"n_frames": 100, "jit_pred_mm": 9.0,
                                                                       "acc_err_mm": 8.0}}
                         for a in arms for s in (3407, 3408)}}

    p = tmp_path / "off.json"
    p.write_text(json.dumps(offline([BASE_NAME])))
    root = world(tmp_path / "a", {"dt_nos": var({"ra": (-3.0, -3.0)})}, base_jit=False)     # base: offline, arm: eval
    rep = SR.build_report(root, BASE_NAME, ["dt_nos"], offline_json=p)
    w = [x for x in rep["warnings"] if "different sources" in x]
    assert len(w) == 1 and w[0].startswith("dt_nos: ")
    assert "seed 3407: arm eval, base offline; seed 3408: arm eval, base offline" in w[0] and "G3a / G3b" in w[0]
    assert rep["runs"]["dt_base_s3407"]["jitter_src"] == "offline" and rep["runs"]["dt_nos_s3407"]["jitter_src"] == "eval"
    # the same source on both sides: no warning (both offline, then both from the evaluations)
    p2 = tmp_path / "off2.json"
    p2.write_text(json.dumps(offline([BASE_NAME, "dt_nos"])))
    root2 = world(tmp_path / "b", {"dt_nos": var({"ra": (-3.0, -3.0)}, drop_jitter=True)}, base_jit=False)
    rep2 = SR.build_report(root2, BASE_NAME, ["dt_nos"], offline_json=p2)
    assert rep2["runs"]["dt_nos_s3407"]["jitter_src"] == "offline"
    assert not any("different sources" in x for x in rep2["warnings"])
    rep3 = report(world(tmp_path / "c", {"dt_nos": var({"ra": (-3.0, -3.0)})}), ["dt_nos"])
    assert not any("different sources" in x for x in rep3["warnings"])


# ------------------------------------------------------------------------------------ what the gates and tables are made of
def test_arm_means_are_over_the_gate_seeds_only(tmp_path):
    rep = report(world(tmp_path, {"dt_nos": var({"ra": (-1.0, -1.0, -50.0)}, seeds=SEEDS)}), ["dt_nos"])
    b, a = rep["arm_means"][BASE_NAME], rep["arm_means"]["dt_nos"]
    assert b["m"]["ra"] == pytest.approx(14.2)                           # (14.0 + 14.4) / 2; with seed 3409 it would be 14.4
    assert b["jitter"]["jit_pred_mm"] == pytest.approx(9.2)              # (9.0 + 9.4) / 2; with seed 3409: 9.4
    assert b["jitter"]["acc_err_mm"] == pytest.approx(8.2)               # (8.0 + 8.4) / 2; with seed 3409: 8.4
    assert b["grid_median"] == pytest.approx(17.5) and b["params"] == 1000
    assert a["m"]["ra"] == pytest.approx(13.2)                           # (13.0 + 13.4) / 2; seed 3409 is 50 mm better
    assert a["jitter"]["jit_pred_mm"] == pytest.approx(9.2) and a["jitter_src"] == "eval"


def test_retention_columns_read_entries_k1_k5_k20_of_the_10_degree_list(tmp_path):
    rep = report(world(tmp_path), [])
    m = rep["runs"]["dt_base_s3407"]["m"]
    assert (m["ret10_k1"], m["ret10_k5"], m["ret10_k20"]) == (0.20, 0.07, 0.05)     # entries 0 / 2 / 4 of (.20 .10 .07 .06 .05)
    row = table_row(SR.render_markdown(rep), "ret k5", "dt_base_s3407 (base)")
    assert (row["ret k1 (10deg)"], row["ret k5"], row["ret k20"]) == ("0.200", "0.070", "0.050")   # the 20 deg list is .5


def test_a_full_round_renders_every_arm_and_every_table_is_rectangular(tmp_path, capsys):
    arms = {
        "dt_tr": var({"abs": (-6.0, -6.0)}), "dt_nos": var({"ra": (-1.2, -1.0)}), "dt_cam": var({"abs": (-3.5, -3.5)}),
        "dt_dz": var({"ra": (-3.0, -3.0)}), "dt_nopm": var({"ra": (0.5, 0.5)}), "dt_pmt": var({"abs": (-2.0, -2.0)}),
        "dt_trnos": var({"abs": (-8.0, -8.0)}), "dt_trdz": var({"abs": (-7.0, -7.0)}),
        "dt_so3c": var({"rot": (-1.0, -1.0), "ra": (-2.0, -2.0)}), "dt_acc": acc(),
        "dt_w05": var({"ra": (0.1, 0.1)}, params=400), "dt_l3": var({"ra": (0.1, 0.1)}, params=450),
        "dt_w05_kd": var({"ra": (0.1, 0.1)}, params=700),
    }
    root = world(tmp_path, arms)
    lat = tmp_path / "lat.json"
    lat.write_text(json.dumps({BASE_NAME: 10.0, "dt_w05_kd": 7.0}))      # dt_w05_kd: 0.7x latency, 1.4x the half parameters
    code = SR.main(["--runs-dir", str(root), "--out-prefix", str(tmp_path / "o" / "r"), "--no-offline",
                    "--latency-json", str(lat)])
    out = capsys.readouterr().out
    assert code == 0 and len(SR.DEFAULT_ARMS) == 13
    md = (tmp_path / "o" / "r.md").read_text()
    md_tables(md)                                                        # rectangular, each followed by a blank line
    for arm in SR.DEFAULT_ARMS:
        assert re.search(rf"^### {arm}( \(|$)", md, re.M), arm           # a section for every arm of the round
        assert re.search(rf"^  {arm}\s", out, re.M), arm                 # and a line on stdout
    heads, rows = md_tables(md)[0]
    assert [r[0] for r in rows] == list(SR.DEFAULT_ARMS)
    by = {r[0]: dict(zip(heads, r)) for r in rows}
    assert by["dt_tr"]["wording"] == "better than base" and by["dt_nos"]["wording"] == PENDING
    assert by["dt_nopm"]["gate"] == "FAIL" and by["dt_pmt"]["gate"] == "FAIL"
    assert by["dt_so3c"]["wording"] == PENDING and by["dt_acc"]["gate"] == "PASS"
    assert "params <= 0.5x" in by["dt_w05"]["wording"] and "latency <= 0.8x" in by["dt_w05_kd"]["wording"]
    assert by["dt_l3"]["gate"] == "PASS" and by["dt_trnos"]["verdict"].startswith("descriptive only")
    q = "gate quantities (mean over 3407, 3408)"
    # trnos RA: (0 - (-1.1)) - 0 = +1.1 (dt_nos is -1.2 / -1.0, the others 0), ABS: (-8 - 0) - (-6) = -2
    assert by["dt_trnos"][q].endswith("interaction RA +1.10 mm; interaction ABS -2.00 mm")
    # trdz RA: (0 - (-3)) - 0 = +3, ABS: (-7 - 0) - (-6) = -1
    assert by["dt_trdz"][q] == "dRA +0.00 mm; dABS -7.00 mm; dROT +0.00 deg; interaction RA +3.00 mm; interaction ABS -1.00 mm"
    assert "None" not in md and "### dt_trnos: 2 x 2" in md and "### dt_trdz: 2 x 2" in md
