#!/usr/bin/env python3
"""Paired-gate report of the DT round (docs/DT_RENDER_TRACK_PREREG.md; section 5 is the contract).

    python tools/dt/screen_report.py [--base dt_base] [--arms ARM ...] [--out-prefix PATH]
                                     [--runs-dir DIR] [--latency-json FILE] [--offline-json FILE | --no-offline]
                                     [--no-params] [--expect-step 6000] [--eval-json NAME]

Reads every run directory `<runs-dir>/<arm>_s<seed>/` that holds `evalx_val_core_last_tf_pert.json` (written by
`tools/tracking/evalx.py eval --ckpt last --controls --tf --perturb`; only the last checkpoint is ever used, no
selection) and writes `<out-prefix>.md` and `<out-prefix>.json` (default `outputs/dt/reports/screen_6k`).
Nothing numeric is hard-coded: every number of the report comes from those json files (plus, optionally, the
offline jitter attribution file, the selection json and the checkpoint's state_dict); the only constants are the
preregistered thresholds. The output is a pure function of the inputs (no timestamps).

What is read from one evalx json (`summary = json`; `evalx.evaluate` / `cmd_eval` write it):

    RA / ABS / ROT / transl   summary["model"]["overall"]["mpjpe_ra_mm" / "mpjpe_abs_mm" / "root_rot_deg" / "transl_mm"]
    global / local            the sequences of summary["model"] whose name has no / has "_local" (evalx.aggregate_runs'
                              rule), frame-weighted mean of the first element of the [mean, ci_lo, ci_hi] lists
    TF RA, amplification      summary["tf"]["overall"]["mpjpe_ra_mm"], summary["amplification"]["mpjpe_ra_mm"]
    failure episodes          sum over the sequences of summary["model"][seq]["failure"]["episodes"]
    root speed ratio          global sequences' ["motion"]["root_speed_ratio"]
    finger speed ratio        MEAN OF the global and the local ["motion"]["finger_speed_ratio"] (unweighted)
    retention k1 / k5 / k20   summary["perturb"]["10"]["retention_k1_k2_k5_k10_k20"][0 / 2 / 4]  (10 deg)
    jitter keys               summary["model"]["jitter"] (frame-weighted `evalx.jitter_decomp`) when present; if absent
                              and the run is in the offline attribution file (tools/dt/jitter_offline.py, same
                              checkpoint, same frame count) its "overall" values are used and marked "offline";
                              otherwise "n/a"
    parameters                numel of the last checkpoint's state_dict (found as `evalx.find_ckpt(run, "last")`
                              does) without the batch-norm running statistics, the frozen MANO layer buffers
                              ("mano.*") and a distillation teacher ("teacher.*"): exactly `sum(p.numel() for p in
                              model.parameters())` (checked equal to `params_total` of nine recorded arms)
    grid median               selection_val_core_step50*.json["grid_median"] (robustness reference only)

Gates (preregistration section 5), on the paired differences d(m, s) = m(arm, s) - m(base, s) of the same seed;
`mean d` is over the seeds 3407 and 3408 only; seed 3409 is reported separately as confirmation. Interpretations
that the preregistration leaves open, as implemented (also printed in the report):

  * "both seeds the same sign as the mean" is applied to every IMPROVEMENT clause (negative threshold): the mean
    must meet the threshold and both seeds' differences must be < 0. A bound clause (non-negative threshold, a
    non-inferiority margin) is tested on the mean only.
  * 1.1 mm is the tie band (|d| < 1.1 mm is a tie), compared with a 1e-9 tolerance so that a result at the edge
    is decided by the rule and not by floating point. A PASSING result is labelled "pending 3409 confirmation"
    when every clause it passes through rests on an mm effect with 1.1 <= |mean d| < 2.5 mm (the ABS gates: the
    band is applied to the ABS difference; so3c: to the RA difference; efficiency: to the RA difference of its
    improvement branch). Relative (acc) and cost-only passes carry no mm effect and are never labelled; the report
    says so in each gate's "band rule" line. The preregistration has no numeric confirmation rule: seed 3409 is
    reported (difference, same sign as the 2-seed mean, 3-seed mean) and the label stays until a person clears it.
  * guardrails use the mean over seeds 3407 / 3408 (jit_pred / acc_err: relative rise of the arm's seed mean over
    the base's seed mean; global root speed ratio: the arm's own seed mean; failure episodes: the seed-mean paired
    difference of the episode count of both sequences, tripped when > 0). A guardrail that cannot be computed
    (jitter block missing) is "n/a": the verdict then says the guardrail check is incomplete.
  * a relative clause (dt_acc's ACC) is the mean paired difference divided by the baseline's seed mean.
  * a gate that needs a number that is not there (a missing seed, a missing jitter block, a latency that was not
    measured) is three-valued: it is decided only if the available clauses decide it; otherwise it is INCOMPLETE.
  * arms: dt_tr; dt_nos / dt_cam / dt_dz / dt_nopm / dt_pmt (one gate; section 9.2 registers dt_nopm and dt_pmt
    as C9a / C9b with "the gate of the C2 family"); dt_so3c; dt_acc; dt_w05 / dt_l3 / dt_w05_kd (efficiency);
    dt_trnos (section 5) and dt_trdz (section 9.6) are descriptive: 2 x 2 factorials with dt_base, interaction
    (both - second factor) - (first factor - base), no gate; any other arm is reported with its paired
    differences and no gate.
  * the conclusion wording follows section 5: "better than base" needs a gate pass that claims better accuracy, no
    tripped guardrail and, for a 1.1-2.5 mm effect, the 3409 confirmation. An efficiency arm that passes only
    through its cost branch (E2 and (E3 or E4): RA within 1.02x at <= 0.5x parameters / <= 0.8x latency) claims
    no accuracy gain: it is worded "efficiency gate met" and better_than_base is false; an E1 gain next to it is
    worded with its own band status.

Output `<prefix>.json` (all keys are strings, seeds as "3407"): `meta` (thresholds, the input files with their
sha1), `warnings`, `runs` (per run: metrics `m`, the jitter block and its source, parameters, grid median),
`arm_means` (mean over the gate seeds), `base_spread`, `gates` (per arm: the paired differences, the clauses, the
guardrails, the band rule, the seed-3409 confirmation, the verdict, the wording), `factorials` ({factorial arm:
block} for dt_trnos and dt_trdz; the generic keys tr / nos / both of a block are the first factor, the second factor
and the arm with both, named in its `arms` and `labels`), `factorial` (the first block of `factorials`, kept for the
earlier layout), `cam_vs_nos`.
Exit code 0 when a report was written (whatever the gates say), 2 for a missing baseline or an unreadable input
(including a latency json that is not an object {arm: ms}).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = REPO / "outputs" / "semkine"
DEFAULT_OUT = REPO / "outputs" / "dt" / "reports" / "screen_6k"
DEFAULT_OFFLINE = REPO / "outputs" / "dt" / "reports" / "jitter_offline.json"
EVAL_NAME = "evalx_val_core_last_tf_pert.json"
SELECTION_GLOB = "selection_val_core_step50*.json"

#: section 5: the gate seeds, the confirmation seed, the tie band, the upper edge of the "pending" band (mm)
GATE_SEEDS = (3407, 3408)
CONFIRM_SEED = 3409
TIE_MM = 1.1
BAND_HI_MM = 2.5
#: tolerance of every threshold comparison: a value at the edge is decided by the rule, not by float noise
EPS = 1e-9
#: the round reports 6000-step runs only (section 2)
EXPECT_STEP = 6000
PENDING_LABEL = "pending 3409 confirmation"

#: arm -> (gate family, registered name). C1-C7: sections 2 and 5. dt_nopm / dt_pmt (C9a / C9b): section 9.2
#: registers both with "the gate of the C2 family" (dABS <= -3 mm or dRA <= -1.1 mm, both seeds the same sign)
GATES = {
    "dt_tr": ("tr", "C1"),
    "dt_nos": ("nos", "C2a"),
    "dt_cam": ("nos", "C2b"),
    "dt_dz": ("nos", "C2c"),
    "dt_nopm": ("nos", "C9a"),
    "dt_pmt": ("nos", "C9b"),
    "dt_so3c": ("so3c", "C3"),
    "dt_acc": ("acc", "C4"),
    "dt_w05": ("eff", "C5"),
    "dt_l3": ("eff", "C6"),
    "dt_w05_kd": ("eff", "C7"),
}
#: descriptive arms: 2 x 2 factorials with the baseline, no gate (section 5: dt_trnos; section 9.6: dt_trdz)
DESCRIPTIVE = {"dt_trnos": "2 x 2 factorial of dt_tr and dt_nos with the baseline (no gate)",
               "dt_trdz": "2 x 2 factorial of dt_tr and dt_dz with the baseline (section 9.6; no gate)"}
#: factorial arm -> (arm with the first factor, arm with the second factor); the interaction is
#: (both - second) - (first - base): (trnos - nos) - (tr - base) and (trdz - dz) - (tr - base) (section 9.6)
FACTORIALS = {"dt_trnos": ("dt_tr", "dt_nos"), "dt_trdz": ("dt_tr", "dt_dz")}
DEFAULT_ARMS = ("dt_tr", "dt_nos", "dt_cam", "dt_dz", "dt_nopm", "dt_pmt", "dt_trnos", "dt_trdz", "dt_so3c",
                "dt_acc", "dt_w05", "dt_l3", "dt_w05_kd")

GATE_RULES = {
    "tr": "mean dABS <= -5 mm and both seeds < 0; mean dRA <= +0.5 mm",
    "nos": "mean dABS <= -3 mm or mean dRA <= -1.1 mm (both seeds < 0 on the clause that passes)",
    "so3c": "mean dROT <= -0.5 deg and mean dRA <= -1.1 mm (both seeds < 0 on both); "
            "guardrail: global root speed ratio >= 0.9",
    "acc": "ACC (jitter.acc_err_mm) vs base <= -15% (both seeds < 0); mean dRA <= +0.3 mm; mean dABS <= +1.1 mm; "
           "guardrail: finger speed ratio drop <= 0.05",
    "eff": "[RA mean <= 1.02 x base RA mean and (params <= 0.5 x base or measured latency <= 0.8 x base)] "
           "or mean dRA <= -1.1 mm (both seeds < 0)",
}
BAND_RULES = {
    "tr": "ABS gate: the band is applied to the ABS difference; passing needs mean dABS <= -5 mm, which is above "
          "2.5 mm, so a pass is never labelled pending.",
    "nos": "ABS/RA gate: the band is applied to the difference of the clause that passes; the ABS clause needs "
           "<= -3 mm (never in the band), the RA clause is pending when 1.1 <= |mean dRA| < 2.5 mm; a pass through "
           "the ABS clause is resolved whatever the RA difference is.",
    "so3c": "angle + mm gate: the band is applied to the RA difference (the mm clause); the 0.5 deg root-rotation "
            "clause has no band of its own.",
    "acc": "relative gate (ACC %): the preregistered band is defined in mm and the mm clauses (dRA <= +0.3, "
           "dABS <= +1.1) are non-inferiority bounds, so no band applies and a pass is never labelled pending; "
           "seed 3409 is reported for information.",
    "eff": "efficiency gate: the band is applied to the RA difference of the improvement branch (mean dRA <= -1.1 "
           "mm); a pass through the cost branch (exact parameter count / measured latency with a non-inferiority "
           "bound) has no mm effect and is never labelled pending.",
}
#: mm accuracy metrics for which the tie band (1.1 mm) / pending band (2.5 mm) reading is printed
#: primary mm effects per gate family (for the wording of a failed gate)
PRIMARY_MM = {"tr": ("abs",), "nos": ("abs", "ra"), "so3c": ("ra",), "acc": (), "eff": ("ra",)}
CONFIRM_KEYS = {"tr": ("abs", "ra"), "nos": ("abs", "ra"), "so3c": ("rot", "ra"),
                "acc": ("jitter.acc_err_mm", "ra", "abs"), "eff": ("ra",)}

#: (key, header, decimals) of the per-run accuracy / dynamics / robustness table
RUN_COLS = (
    ("ra", "RA", 2), ("ra_global", "RA glob", 2), ("ra_local", "RA loc", 2),
    ("rot", "root rot (deg)", 2), ("transl", "abs transl (mm)", 2),
    ("abs", "ABS", 2), ("abs_global", "ABS glob", 2), ("abs_local", "ABS loc", 2),
    ("tf_ra", "TF RA", 2), ("amp", "amp (RA)", 3), ("fail", "fail eps", 0),
    ("root_ratio", "root spd ratio (glob)", 3), ("finger_ratio", "finger spd ratio (g/l mean)", 3),
    ("ret10_k1", "ret k1 (10deg)", 3), ("ret10_k5", "ret k5", 3), ("ret10_k20", "ret k20", 3),
)
#: the jitter keys of the per-run jitter table (the json keeps all of them)
JIT_COLS = (
    ("jit_gt_mm", "jit gt", 2), ("jit_pred_mm", "jit pred", 2), ("jit_ratio", "jit ratio", 3),
    ("acc_gt_mm", "acc gt", 2), ("acc_pred_mm", "acc pred", 2), ("acc_err_mm", "acc err", 2),
    ("acc_ratio", "acc ratio", 3),
    ("acc_err_only_transl_mm", "acc err only transl", 2), ("acc_err_only_root_mm", "only root", 2),
    ("acc_err_only_fingers_mm", "only fingers", 2),
    ("acc_err_allbut_transl_mm", "acc err allbut transl", 2), ("acc_err_allbut_root_mm", "allbut root", 2),
    ("acc_err_allbut_fingers_mm", "allbut fingers", 2),
    ("rot_acc_pred_deg", "rot acc pred (deg)", 2), ("rot_acc_gt_deg", "rot acc gt (deg)", 2),
)
#: (key, label, decimals, mm accuracy metric?) of the paired-difference table of every arm
PAIRED_ROWS = (
    ("ra", "RA overall (mm)", 2, True), ("ra_global", "RA global (mm)", 2, True), ("ra_local", "RA local (mm)", 2, True),
    ("rot", "root rot (deg)", 2, False), ("transl", "abs transl (mm)", 2, True),
    ("abs", "ABS overall (mm)", 2, True), ("abs_global", "ABS global (mm)", 2, True),
    ("abs_local", "ABS local (mm)", 2, True),
    ("tf_ra", "TF RA (mm)", 2, True), ("amp", "amplification (RA)", 3, False), ("fail", "failure episodes", 1, False),
    ("root_ratio", "global root speed ratio", 3, False), ("finger_ratio", "finger speed ratio (g/l mean)", 3, False),
    ("ret10_k1", "retention k1 (10 deg)", 3, False), ("ret10_k5", "retention k5", 3, False),
    ("ret10_k20", "retention k20", 3, False),
    ("jitter.jit_pred_mm", "jit_pred (mm/step)", 2, False), ("jitter.acc_err_mm", "acc_err (mm/step^2)", 2, False),
)
PAIRED_NAMES = {k: lab for k, lab, _, _ in PAIRED_ROWS}
JITTER_KEY = re.compile(r"^(jit_|acc_|rot_acc_)")


class ScreenError(RuntimeError):
    """A fatal input problem (missing baseline, unreadable json): `main` prints it and exits 2."""


# ---------------------------------------------------------------------------------------------- small helpers
def num(x):
    """A finite float or None."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def first(x):
    """The point estimate of a `[mean, ci_lo, ci_hi]` list, or a plain number."""
    if isinstance(x, (list, tuple)):
        x = x[0] if x else None
    return num(x)


def dig(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def fmt(x, nd=2, signed=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    if round(x, nd) == 0:
        x = 0.0
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def na(x):
    """A value for running text, or "n/a" when it is not there (never the word None)."""
    return "n/a" if x is None else x


def k_and(*xs):
    """Three-valued AND (None = unknown)."""
    if any(x is False for x in xs):
        return False
    return None if any(x is None for x in xs) else True


def k_or(*xs):
    """Three-valued OR (None = unknown)."""
    if any(x is True for x in xs):
        return True
    return None if any(x is None for x in xs) else False


def mean(xs):
    return math.fsum(xs) / len(xs)


def in_band(x):
    """1.1 <= |x| < 2.5 (mm), the results that are labelled pending 3409 confirmation."""
    return x is not None and TIE_MM - EPS <= abs(x) < BAND_HI_MM - EPS


def seed_key(s):
    return str(int(s))


# ---------------------------------------------------------------------------------------------- reading a run
def _sequences(model):
    """(global, local) sequence entries of summary["model"], by evalx.aggregate_runs' naming rule."""
    seqs = [(k, v) for k, v in model.items()
            if k not in ("overall", "jitter") and isinstance(v, dict) and "n_frames" in v]
    return [kv for kv in seqs if "_local" not in kv[0]], [kv for kv in seqs if "_local" in kv[0]]


def _wavg(parts, fn):
    """Frame-weighted mean of fn(entry) over the sequences; None if there is none or one lacks the value."""
    tot = acc = 0.0
    for _, v in parts:
        x, n = fn(v), num(v.get("n_frames"))
        if x is None or not n:
            return None
        acc += x * n
        tot += n
    return acc / tot if tot else None


def extract_metrics(js):
    """The flat metric dict (None = not in the json) and the jitter dict (None = no jitter block)."""
    model = js.get("model")
    ov = dig(model, "overall")
    if not isinstance(ov, dict) or num(ov.get("mpjpe_ra_mm")) is None:
        raise ScreenError("not an evalx evaluation json: summary['model']['overall']['mpjpe_ra_mm'] is missing")
    glo, loc = _sequences(model)
    m = {"ra": num(ov.get("mpjpe_ra_mm")),
         "ra_global": _wavg(glo, lambda v: first(v.get("mpjpe_ra_mm"))),
         "ra_local": _wavg(loc, lambda v: first(v.get("mpjpe_ra_mm"))),
         "rot": num(ov.get("root_rot_deg")), "transl": num(ov.get("transl_mm")),
         "abs": num(ov.get("mpjpe_abs_mm")),
         "abs_global": _wavg(glo, lambda v: first(v.get("mpjpe_abs_mm"))),
         "abs_local": _wavg(loc, lambda v: first(v.get("mpjpe_abs_mm"))),
         "tf_ra": num(dig(js, "tf", "overall", "mpjpe_ra_mm")),
         "amp": num(dig(js, "amplification", "mpjpe_ra_mm"))}
    eps = [num(dig(v, "failure", "episodes")) for _, v in glo + loc]
    m["fail"] = int(sum(eps)) if eps and all(e is not None for e in eps) else None
    m["root_ratio"] = _wavg(glo, lambda v: num(dig(v, "motion", "root_speed_ratio")))
    fg = _wavg(glo, lambda v: num(dig(v, "motion", "finger_speed_ratio")))
    fl = _wavg(loc, lambda v: num(dig(v, "motion", "finger_speed_ratio")))
    m["finger_ratio_global"], m["finger_ratio_local"] = fg, fl
    m["finger_ratio"] = (fg + fl) / 2 if fg is not None and fl is not None else None
    ret = dig(js, "perturb", "10", "retention_k1_k2_k5_k10_k20")
    ok = isinstance(ret, list) and len(ret) >= 5
    m["ret10_k1"], m["ret10_k5"], m["ret10_k20"] = ((num(ret[0]), num(ret[2]), num(ret[4])) if ok
                                                    else (None, None, None))
    jit = dig(model, "jitter")
    jit = {k: num(v) for k, v in jit.items() if num(v) is not None} if isinstance(jit, dict) else None
    return m, (jit or None)


def mval(run, key):
    """One metric of a loaded run (`jitter.<k>` reads the jitter block); None if the run or the value is missing."""
    if run is None:
        return None
    if key.startswith("jitter."):
        j = run.get("jitter")
        return None if j is None else j.get(key[len("jitter."):])
    return run["m"].get(key)


# ---------------------------------------------------------------------------------------------- checkpoint
def find_ckpt_last(run: Path):
    """`evalx.find_ckpt(run, "last")`: the `*-step=<N>.ckpt` with the largest N; `last.ckpt` if there is no
    step checkpoint. Returns (path, step) or (None, None)."""
    steps = sorted(int(m.group(1)) for p in run.glob("*step=*.ckpt") for m in [re.search(r"step=(\d+)", p.name)] if m)
    if steps:
        st = steps[-1]
        cands = sorted(run.glob(f"*-step={st}.ckpt")) or sorted(
            p for p in run.glob("*step=*.ckpt") if re.search(rf"step={st}(?!\d)", p.name))
        if cands:
            return cands[0], st
    last = run / "last.ckpt"
    return (last, None) if last.is_file() else (None, None)


def count_params(state_dict):
    """Number of parameters in a Lightning state_dict: everything except the batch-norm running statistics, the
    frozen MANO layer buffers and a distillation teacher (not parameters of the evaluated network)."""
    n = 0
    for k, v in state_dict.items():
        if k.endswith((".running_mean", ".running_var", ".num_batches_tracked")):
            continue
        if k.startswith(("mano.", "teacher.")) or ".mano." in k:
            continue
        n += int(v.numel())
    return n


def run_params(run: Path):
    """(params, ckpt path, ckpt step, note) of the run's last checkpoint; params None when there is none or it
    cannot be read (a checkpoint that is being written, a composite tracker without a checkpoint)."""
    ck, st = find_ckpt_last(run)
    if ck is None:
        return None, None, None, "no checkpoint in the run directory"
    try:
        import torch
        obj = torch.load(str(ck), map_location="cpu", weights_only=False)
        return count_params(obj["state_dict"]), str(ck), st, None
    except Exception as e:                                                       # noqa: BLE001
        return None, str(ck), st, f"checkpoint unreadable: {type(e).__name__}: {str(e)[:120]}"


def grid_median(run: Path):
    for p in sorted(run.glob(SELECTION_GLOB)):
        try:
            return num(json.loads(p.read_text()).get("grid_median"))
        except (OSError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------------------------- offline jitter
def load_offline(path):
    """The offline attribution file (tools/dt/jitter_offline.py) or None."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise ScreenError(f"cannot read the offline jitter file {p}: {e}")
    return d if isinstance(d.get("runs"), dict) else None


def offline_jitter(offline, run_name, n_frames):
    """The frame-weighted `jitter_decomp` values of a run from the offline file; None unless the run is in it,
    was read from the `last` checkpoint and has the same number of frames as the evaluation json."""
    r = dig(offline, "runs", run_name)
    if not isinstance(r, dict) or not str(r.get("npz_tag", "")).startswith("last"):
        return None
    ov = r.get("overall")
    if not isinstance(ov, dict) or (n_frames is not None and num(ov.get("n_frames")) != n_frames):
        return None
    out = {k: num(v) for k, v in ov.items()
           if JITTER_KEY.match(k) and not k.endswith("_pooled") and num(v) is not None}
    return out or None


def load_latency(path, warnings):
    """{arm: ms} of the optional --latency-json. Only positive finite numbers count; every other entry is left out
    with a warning (the efficiency gate then reports the latency as not measured). A json that is not an object is a
    fatal input error."""
    if not path:
        return {}
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise ScreenError(f"cannot read the latency json {path}: {e}")
    if not isinstance(raw, dict):
        raise ScreenError(f"the latency json {path} must hold an object {{arm: ms}}, not a {type(raw).__name__}")
    out = {k: num(v) for k, v in raw.items() if num(v) is not None and num(v) > 0}
    left = sorted(k for k in raw if k not in out)
    if left:
        hint = (" (this looks like a tools/s38/bench.py json: reduce its per-arm forward times to {arm: ms})"
                if isinstance(raw.get("arms"), dict) else "")
        warnings.append(f"latency json {path}: entries left out because they are not a positive number of ms: "
                        f"{', '.join(left)}{hint}")
    return out


def jitter_source_mix(arm_runs, base_runs):
    """[(seed, arm source, base source)] of the gate seeds whose jitter values come from different sources (an
    evaluation's own block / the offline attribution file) for the arm and for the baseline."""
    out = []
    for s in GATE_SEEDS:
        a, b = arm_runs.get(s), base_runs.get(s)
        if a and b and a["jitter_src"] and b["jitter_src"] and a["jitter_src"] != b["jitter_src"]:
            out.append((s, a["jitter_src"], b["jitter_src"]))
    return out


# ---------------------------------------------------------------------------------------------- loading
def discover(runs_dir: Path, arm: str, eval_name=EVAL_NAME):
    """({seed: run dir} of the run directories `<arm>_s<seed>` that hold the evaluation json, [seeds whose
    directory exists without it])."""
    pat = re.compile(rf"^{re.escape(arm)}_s(\d+)$")
    have, without = {}, []
    if runs_dir.is_dir():
        for p in sorted(runs_dir.iterdir()):
            mt = pat.match(p.name)
            if mt and p.is_dir():
                if (p / eval_name).is_file():
                    have[int(mt.group(1))] = p
                else:
                    without.append(int(mt.group(1)))
    return have, sorted(without)


def discover_dt_arms(runs_dir: Path, eval_name=EVAL_NAME):
    """Names of the `dt_*` arms (not the 2000-step `*_2k` information arms) with at least one evaluated run."""
    pat = re.compile(r"^(dt_.+)_s\d+$")
    out = set()
    if runs_dir.is_dir():
        for p in runs_dir.iterdir():
            mt = pat.match(p.name)
            if mt and not mt.group(1).endswith("_2k") and (p / eval_name).is_file():
                out.add(mt.group(1))
    return sorted(out)


def load_run(rdir: Path, arm, seed, offline=None, want_params=True, eval_name=EVAL_NAME):
    p = rdir / eval_name
    try:
        raw = p.read_bytes()
        js = json.loads(raw)
    except (OSError, ValueError) as e:
        raise ScreenError(f"cannot read {p}: {e}")
    try:
        metrics, jit = extract_metrics(js)
    except ScreenError as e:
        raise ScreenError(f"{p}: {e}")
    n_frames = num(dig(js, "model", "n_frames"))
    n_frames = int(n_frames) if n_frames is not None else None
    step = num(js.get("step"))                      # evalx writes an int; null, NaN or text mean "unknown"
    step = int(step) if step is not None and step == int(step) else None
    src = "eval" if jit else None
    if jit is None and offline is not None:
        jit = offline_jitter(offline, rdir.name, n_frames)
        src = "offline" if jit else None
    params, ck, ck_step, note = (run_params(rdir) if want_params else (None, None, None, "not counted (--no-params)"))
    return {"arm": arm, "seed": int(seed), "run": rdir.name, "json": str(p),
            "sha1": hashlib.sha1(raw).hexdigest()[:12], "step": step, "n_frames": n_frames,
            "m": metrics, "jitter": jit, "jitter_src": src, "params": params, "ckpt": ck, "ckpt_step": ck_step,
            "params_note": note, "grid_median": grid_median(rdir)}


# ---------------------------------------------------------------------------------------------- pairing
class Pair:
    """Seed-paired view of one arm against the baseline: d(m, s) = m(arm, s) - m(base, s)."""

    def __init__(self, arm, base, arm_runs, base_runs):
        self.arm, self.base, self.a, self.b = arm, base, arm_runs, base_runs

    def val(self, which, key, seed):
        return mval((self.a if which == "arm" else self.b).get(seed), key)

    def d(self, key, seed):
        x, y = self.val("arm", key, seed), self.val("base", key, seed)
        return None if x is None or y is None else x - y

    def gate_d(self, key):
        return [self.d(key, s) for s in GATE_SEEDS]

    def mean_d(self, key):
        ds = self.gate_d(key)
        return None if any(v is None for v in ds) else mean(ds)

    def mean_val(self, which, key):
        vs = [self.val(which, key, s) for s in GATE_SEEDS]
        return None if any(v is None for v in vs) else mean(vs)

    def why_missing(self, key, which=None):
        """Why the gate-seed values of `key` are not all there (`which`: only "arm" or only "base")."""
        out = []
        for s in GATE_SEEDS:
            for w, name in (("arm", self.arm), ("base", self.base)):
                if which not in (None, w):
                    continue
                runs = self.a if w == "arm" else self.b
                if s not in runs:
                    out.append(f"{name} seed {s}: no evaluated run")
                elif mval(runs[s], key) is None:
                    out.append(f"{name} seed {s}: {key} n/a")
        return out

    def common_seeds(self):
        return sorted(set(self.a) & set(self.b))


# ---------------------------------------------------------------------------------------------- clauses
def _per_seed(P, key):
    return {seed_key(s): d for s, d in zip(GATE_SEEDS, P.gate_d(key))}


def clause_improve(P, cid, key, thr, unit, short, text, rel=False):
    """Improvement clause: mean d (or mean d / base mean when `rel`) <= thr (< 0) AND every gate seed's d < 0."""
    ds, miss = P.gate_d(key), P.why_missing(key)
    value = met = None
    rel_seed = {}
    if not miss:
        bm = P.mean_val("base", key)
        if rel and (bm is None or abs(bm) < 1e-12):
            miss = [f"base {key} mean is 0"]
        else:
            value = mean(ds) / bm if rel else mean(ds)
            met = bool(value <= thr + EPS and all(d < 0 for d in ds))
            if rel:                                       # per-seed change relative to that seed's baseline value
                rel_seed = {seed_key(s): d / P.val("base", key, s) for s, d in zip(GATE_SEEDS, ds)
                            if P.val("base", key, s)}
    return {"id": cid, "kind": "improve", "metric": key, "short": short, "text": text, "threshold": thr,
            "unit": unit, "relative": rel, "value": value, "per_seed": _per_seed(P, key), "met": met,
            "missing": miss, "per_seed_relative": rel_seed}


def clause_bound(P, cid, key, thr, unit, short, text):
    """Bound clause (non-inferiority margin): mean d <= thr (>= 0), on the mean only."""
    ds, miss = P.gate_d(key), P.why_missing(key)
    value = mean(ds) if not miss else None
    return {"id": cid, "kind": "bound", "metric": key, "short": short, "text": text, "threshold": thr,
            "unit": unit, "relative": False, "value": value, "per_seed": _per_seed(P, key),
            "met": None if value is None else bool(value <= thr + EPS), "missing": miss}


def clause_noninf(P, cid, key, factor, short, text):
    """Non-inferiority on the level: arm mean <= factor x base mean (seeds 3407 / 3408)."""
    am, bm = P.mean_val("arm", key), P.mean_val("base", key)
    miss = P.why_missing(key)
    value = am / bm if not miss and bm else None
    return {"id": cid, "kind": "noninf", "metric": key, "short": short, "text": text, "threshold": factor,
            "unit": "x", "relative": True, "value": value, "arm_mean": am, "base_mean": bm,
            "per_seed": {}, "met": None if value is None else bool(am <= factor * bm + EPS), "missing": miss}


def clause_ratio(cid, short, text, arm_val, base_val, factor, unit, what):
    """A cost clause: arm value <= factor x base value (parameters, latency)."""
    miss = []
    if arm_val is None:
        miss.append(f"arm {what} n/a")
    if base_val is None:
        miss.append(f"base {what} n/a")
    value = arm_val / base_val if not miss and base_val else None
    return {"id": cid, "kind": "cost", "metric": what, "short": short, "text": text, "threshold": factor,
            "unit": unit, "relative": True, "value": value, "arm_value": arm_val, "base_value": base_val,
            "per_seed": {}, "met": None if value is None else bool(arm_val <= factor * base_val + EPS * base_val),
            "missing": miss}


def branch(name, met, effect_mm, effect_key):
    return {"name": name, "met": met, "effect_key": effect_key, "effect_mm": effect_mm,
            "in_band": None if effect_mm is None else in_band(effect_mm)}


# ---------------------------------------------------------------------------------------------- the gates
def gate_tr(P, ctx):
    c1 = clause_improve(P, "C1", "abs", -5.0, "mm", "dABS", "mean dABS <= -5 mm and both seeds < 0")
    c2 = clause_bound(P, "C2", "ra", 0.5, "mm", "dRA", "mean dRA <= +0.5 mm")
    ok = k_and(c1["met"], c2["met"])
    return [c1, c2], "C1 and C2", ok, [branch("C1 and C2", ok, P.mean_d("abs"), "abs")]


def gate_nos(P, ctx):
    c1 = clause_improve(P, "C1", "abs", -3.0, "mm", "dABS", "mean dABS <= -3 mm and both seeds < 0")
    c2 = clause_improve(P, "C2", "ra", -TIE_MM, "mm", "dRA", "mean dRA <= -1.1 mm and both seeds < 0")
    return ([c1, c2], "C1 or C2", k_or(c1["met"], c2["met"]),
            [branch("C1 (ABS)", c1["met"], P.mean_d("abs"), "abs"), branch("C2 (RA)", c2["met"], P.mean_d("ra"), "ra")])


def gate_so3c(P, ctx):
    c1 = clause_improve(P, "C1", "rot", -0.5, "deg", "dROT", "mean dROT <= -0.5 deg and both seeds < 0")
    c2 = clause_improve(P, "C2", "ra", -TIE_MM, "mm", "dRA", "mean dRA <= -1.1 mm and both seeds < 0")
    ok = k_and(c1["met"], c2["met"])
    return [c1, c2], "C1 and C2", ok, [branch("C1 and C2", ok, P.mean_d("ra"), "ra")]


def gate_acc(P, ctx):
    c1 = clause_improve(P, "C1", "jitter.acc_err_mm", -0.15, "%", "ACC", "ACC (acc_err) vs base <= -15% and both "
                        "seeds < 0", rel=True)
    c2 = clause_bound(P, "C2", "ra", 0.3, "mm", "dRA", "mean dRA <= +0.3 mm")
    c3 = clause_bound(P, "C3", "abs", 1.1, "mm", "dABS", "mean dABS <= +1.1 mm")
    ok = k_and(c1["met"], c2["met"], c3["met"])
    return [c1, c2, c3], "C1 and C2 and C3", ok, [branch("C1 and C2 and C3", ok, None, None)]


def gate_eff(P, ctx):
    e1 = clause_improve(P, "E1", "ra", -TIE_MM, "mm", "dRA", "mean dRA <= -1.1 mm and both seeds < 0")
    e2 = clause_noninf(P, "E2", "ra", 1.02, "RA ratio", "RA mean <= 1.02 x base RA mean")
    e3 = clause_ratio("E3", "params ratio", "parameters <= 0.5 x base (exact)", ctx.get("arm_params"),
                      ctx.get("base_params"), 0.5, "x", "params")
    lat, base_lat = ctx.get("arm_latency"), ctx.get("base_latency")
    e4 = clause_ratio("E4", "latency ratio", "measured latency <= 0.8 x base", lat, base_lat, 0.8, "x", "latency")
    if lat is None or base_lat is None:
        e4["missing"] = ["latency not measured"]
    cost = k_or(e3["met"], e4["met"])
    eff = k_and(e2["met"], cost)
    return ([e1, e2, e3, e4], "E1 or (E2 and (E3 or E4))", k_or(e1["met"], eff),
            [branch("E1 (RA improvement)", e1["met"], P.mean_d("ra"), "ra"),
             branch("E2 and (E3 or E4) (efficiency)", eff, None, None)])


GATE_FUNCS = {"tr": gate_tr, "nos": gate_nos, "so3c": gate_so3c, "acc": gate_acc, "eff": gate_eff}


# ---------------------------------------------------------------------------------------------- guardrails
def _g_delta(P, gid, text, key, thr, op):
    v = P.mean_d(key)
    trip = None if v is None else (v > thr + EPS if op == ">" else v < thr - EPS)
    return {"id": gid, "text": text, "metric": key, "kind": "mean_delta", "value": v, "threshold": thr, "op": op,
            "per_seed": _per_seed(P, key), "tripped": trip, "missing": [] if v is not None else P.why_missing(key)}


def _g_rise(P, gid, text, key, thr):
    am, bm = P.mean_val("arm", key), P.mean_val("base", key)
    miss = P.why_missing(key)
    v = am / bm - 1.0 if not miss and bm else None
    return {"id": gid, "text": text, "metric": key, "kind": "relative_rise", "value": v, "threshold": thr, "op": ">",
            "arm_mean": am, "base_mean": bm, "per_seed": {},
            "tripped": None if v is None else bool(v > thr + EPS), "missing": miss if v is None else []}


def _g_level(P, gid, text, key, thr):
    v = P.mean_val("arm", key)
    return {"id": gid, "text": text, "metric": key, "kind": "arm_level", "value": v, "threshold": thr, "op": "<",
            "per_seed": {seed_key(s): P.val("arm", key, s) for s in GATE_SEEDS},
            "tripped": None if v is None else bool(v < thr - EPS),
            "missing": P.why_missing(key, "arm") if v is None else []}


def universal_guardrails(P):
    return [
        _g_delta(P, "G1", "mean dABS > +2 mm", "abs", 2.0, ">"),
        _g_delta(P, "G2", "mean dRA (local) > +1 mm", "ra_local", 1.0, ">"),
        _g_rise(P, "G3a", "jit_pred seed mean rises by more than 5%", "jitter.jit_pred_mm", 0.05),
        _g_rise(P, "G3b", "acc_err seed mean rises by more than 5%", "jitter.acc_err_mm", 0.05),
        _g_level(P, "G4a", "global root speed ratio < 0.9", "root_ratio", 0.9),
        _g_delta(P, "G4b", "finger speed ratio drops by more than 0.05", "finger_ratio", -0.05, "<"),
        _g_delta(P, "G5", "failure episodes increase", "fail", 0.0, ">"),
        _g_delta(P, "G6", "closed-loop / TF amplification (RA) exceeds the base by more than 0.05", "amp", 0.05, ">"),
    ]


def guardrail_status(gs):
    tripped = [g["id"] for g in gs if g["tripped"] is True]
    unknown = [g["id"] for g in gs if g["tripped"] is None]
    return ("tripped" if tripped else "incomplete" if unknown else "ok"), tripped, unknown


# ---------------------------------------------------------------------------------------------- readings
def reading(P, key):
    """The tie-band reading of a mm accuracy metric: tie / tie (seeds disagree) / better|worse [, 1.1-2.5 band]."""
    ds = P.gate_d(key)
    if any(d is None for d in ds):
        return None
    m = mean(ds)
    if abs(m) < TIE_MM - EPS:
        return "tie"
    sgn = -1.0 if m < 0 else 1.0
    if any(d * sgn <= 0 for d in ds):
        return "tie (seeds disagree)"
    word = "better" if m < 0 else "worse"
    return f"{word}, in the 1.1-2.5 mm band" if abs(m) < BAND_HI_MM - EPS else word


def paired_block(P):
    """Per metric: base / arm means over the gate seeds, d per common seed, the gate mean, the sign agreement."""
    out = {}
    for key, label, nd, mm in PAIRED_ROWS:
        ds = P.gate_d(key)
        gm = None if any(d is None for d in ds) else mean(ds)
        bm, am = P.mean_val("base", key), P.mean_val("arm", key)
        same = None
        if gm is not None:
            same = bool(gm != 0 and all(d * gm > 0 for d in ds))
        per = {seed_key(s): P.d(key, s) for s in P.common_seeds()}
        out[key] = {"label": label, "decimals": nd, "base_mean": bm, "arm_mean": am, "per_seed": per,
                    "gate_mean": gm, "same_sign": same,
                    "rel_gate_mean": gm / bm if gm is not None and bm else None,
                    "reading": reading(P, key) if mm else None}
    return out


def confirmation_block(P, keys, improve_keys):
    avail_arm, avail_base = CONFIRM_SEED in P.a, CONFIRM_SEED in P.b
    out = {"seed": CONFIRM_SEED, "available": bool(avail_arm and avail_base), "arm_has_seed": avail_arm,
           "base_has_seed": avail_base, "diffs": {}, "same_sign_as_mean": {}, "three_seed_mean": {}}
    for k in keys:
        d = P.d(k, CONFIRM_SEED)
        gm = P.mean_d(k)
        out["diffs"][k] = d
        out["same_sign_as_mean"][k] = (None if d is None or gm is None or gm == 0 else bool(d * gm > 0))
        allv = P.gate_d(k) + [d]
        out["three_seed_mean"][k] = None if any(v is None for v in allv) else mean(allv)
    out["supports"] = None
    if out["available"]:
        flags = [out["same_sign_as_mean"][k] for k in improve_keys]
        out["supports"] = None if not flags or any(f is None for f in flags) else bool(all(flags))
    return out


# ---------------------------------------------------------------------------------------------- one arm
def efficiency_wording(clauses, branches, gstat):
    """Wording of an efficiency arm that passes through its cost branch (E2 and (E3 or E4)): accuracy within the
    non-inferiority margin at a lower cost, which is not "better than base". An E1 gain is added with its band status."""
    e1, e2, e3, e4 = clauses[:4]
    costs = [f"{t} <= {c['threshold']:g}x" for t, c in (("params", e3), ("latency", e4)) if c["met"] is True]
    text = (f"efficiency gate met: RA non-inferior (<= {e2['threshold']:g}x) at lower cost ("
            + " and ".join(costs) + ")")
    if e1["met"] is True:
        mm = fmt(branches[0]["effect_mm"], 2, True)
        text += (f"; the RA gain of {mm} mm lies in the 1.1-2.5 mm band: {PENDING_LABEL}" if branches[0]["in_band"]
                 else f"; RA also improved (mean dRA {mm} mm)")
    else:
        text += "; no accuracy gain is claimed (not \"better than base\")"
    return text + ("; guardrail check incomplete" if gstat == "incomplete" else "")


def evaluate_arm(arm, base, arm_runs, base_runs, latency):
    """The full verdict record of one candidate arm (gate, guardrails, band, confirmation)."""
    P = Pair(arm, base, arm_runs, base_runs)
    fam = GATES.get(arm, (None, None))[0]
    rec = {"arm": arm, "label": GATES.get(arm, (None, None))[1], "family": fam,
           "kind": "gate" if fam else "descriptive" if arm in DESCRIPTIVE else "none",
           "rule": GATE_RULES.get(fam) if fam else DESCRIPTIVE.get(arm),
           "seeds": {"arm": [seed_key(s) for s in sorted(arm_runs)], "base": [seed_key(s) for s in sorted(base_runs)],
                     "missing_gate_seeds": [seed_key(s) for s in GATE_SEEDS if s not in arm_runs or s not in base_runs]},
           "paired": paired_block(P)}
    rec["complete"] = not rec["seeds"]["missing_gate_seeds"]
    if not fam:
        rec["verdict"] = ("descriptive only (no gate): see the factorial block" if arm in DESCRIPTIVE
                          else "no gate registered for this arm: paired differences only")
        return rec
    ctx = {"arm_params": _arm_params(arm_runs), "base_params": _arm_params(base_runs),
           "arm_latency": latency.get(arm), "base_latency": latency.get(base)}
    clauses, expr, ok, branches = GATE_FUNCS[fam](P, ctx)
    gs = universal_guardrails(P)
    gstat, tripped, unknown = guardrail_status(gs)
    passing = [b for b in branches if b["met"] is True]
    pending = bool(ok is True and passing and all(b["in_band"] is True for b in passing))
    notes = [f"{b['name']}: mean d{b['effect_key'].upper()} = {fmt(b['effect_mm'], 2, True)} mm lies in the "
             f"1.1-2.5 mm band" for b in branches if b["in_band"] is True and b["met"] is True and not pending]
    missing = sorted({m for c in clauses if c["met"] is None for m in c["missing"]})
    rec.update({"clauses": clauses, "expression": expr, "gate_pass": ok, "guardrails": gs,
                "guardrail_status": gstat, "guardrails_tripped": tripped, "guardrails_unknown": unknown,
                "band": {"rule": BAND_RULES[fam], "branches": branches, "pending": pending, "notes": notes},
                "pending_3409": pending, "label_text": PENDING_LABEL if pending else None, "missing": missing})
    if fam == "eff":
        rec["efficiency"] = {"params_arm": ctx["arm_params"], "params_base": ctx["base_params"],
                             "latency_arm_ms": ctx["arm_latency"], "latency_base_ms": ctx["base_latency"],
                             "latency": "measured" if ctx["arm_latency"] is not None and ctx["base_latency"] is not None
                             else "not measured"}
    # the verdict
    if ok is None:
        verdict = "INCOMPLETE: " + ("; ".join(missing) if missing else "a gate input is missing")
    elif ok is False:
        verdict = "FAIL"
    elif gstat == "tripped":
        verdict = "PASS gate, NOT ADOPTED: guardrail " + ", ".join(tripped)
    else:
        verdict = "PASS" + (", " + PENDING_LABEL if pending else "") + "; no guardrail tripped"
        if gstat == "incomplete":
            verdict = ("PASS" + (", " + PENDING_LABEL if pending else "") +
                       "; guardrail check incomplete (n/a: " + ", ".join(unknown) + ")")
    rec["verdict"] = verdict
    # the conclusion rule ("better than base" needs a gate pass, no tripped guardrail and, for a 1.1-2.5 mm effect, the
    # 3409 confirmation). Does the pass claim better accuracy? Every family but the efficiency one: yes. An efficiency
    # arm can pass through its cost branch (E2 and (E3 or E4): non-inferior accuracy at lower cost), which claims no
    # accuracy gain; only its E1 clause does, and that gain has its own band status (claim_pending).
    cost_pass = fam == "eff" and branches[1]["met"] is True
    acc_claim = (branches[0]["met"] is True) if fam == "eff" else True
    claim_pending = bool(branches[0]["in_band"]) if fam == "eff" else pending
    rec["better_than_base"] = (None if ok is None else False if ok is False or gstat == "tripped" or not acc_claim
                               else True if gstat == "ok" and not claim_pending else None)
    # wording of the conclusion rule
    if ok is None:
        wording = None
    elif ok is True:
        if gstat == "tripped":
            wording = "not adopted (guardrail)"
        elif cost_pass:
            wording = efficiency_wording(clauses, branches, gstat)
        else:
            wording = (PENDING_LABEL if pending else "better than base" if gstat == "ok"
                       else "pass, guardrails incomplete")
    else:
        rd = [reading(P, k) for k in PRIMARY_MM[fam]]
        # "tie" is a statement about the effect the gate asks for; an efficiency arm that fails on cost with a
        # tied RA has not tied anything, it has not met the gate
        if fam != "eff" and rd and all(r is not None and r.startswith("tie") for r in rd):
            wording = "tie"
        elif any(r is not None and r.startswith("worse") for r in rd):
            wording = "worse"
        else:
            wording = "gate not met"
    rec["wording"] = wording
    # the sign check of seed 3409 concerns the improvement clauses the pass goes through
    rec["confirmation"] = confirmation_block(
        P, CONFIRM_KEYS[fam], [c["metric"] for c in clauses if c["kind"] == "improve" and c["met"] is True]
        if ok is True else [])
    return rec


def _arm_params(runs):
    """Parameter count of an arm: that of its lowest seed that has one (the architecture is the same for all)."""
    for s in sorted(runs):
        if runs[s]["params"] is not None:
            return runs[s]["params"]
    return None


# ---------------------------------------------------------------------------------------------- descriptive blocks
def factorial(runs, base, tr="dt_tr", nos="dt_nos", both="dt_trnos"):
    """2 x 2 factorial with the baseline: interaction = (both - nos) - (tr - base), RA and ABS. The keys of the result
    are generic: tr = the first factor's arm, nos = the second factor's arm, both = the arm with both factors
    (dt_trnos: tr = dt_tr, nos = dt_nos; dt_trdz: tr = dt_tr, nos = dt_dz); `arms` and `labels` name them."""
    cells = {"base": runs.get(base, {}), "tr": runs.get(tr, {}), "nos": runs.get(nos, {}), "both": runs.get(both, {})}
    seeds = sorted(set.intersection(*[set(c) for c in cells.values()])) if all(cells.values()) else []
    short = lambda a: a[3:] if a.startswith("dt_") else a                              # noqa: E731
    out = {"definition": f"interaction = ({both} - {nos}) - ({tr} - {base}) = effect({both}) - effect({tr}) - "
                         f"effect({nos}); negative: combining is better than adding the two effects, positive: the "
                         f"effects overlap", "arms": {"base": base, "tr": tr, "nos": nos, "both": both},
           "labels": {"base": short(base), "tr": short(tr), "nos": short(nos), "both": short(both)},
           "seeds_present": {k: [seed_key(s) for s in sorted(c)] for k, c in cells.items()}, "metrics": {}}
    for key in ("ra", "abs"):
        per = {}
        for s in seeds:
            v = {k: mval(c[s], key) for k, c in cells.items()}
            if any(x is None for x in v.values()):
                continue
            per[seed_key(s)] = {"base": v["base"], "tr": v["tr"], "nos": v["nos"], "both": v["both"],
                                "effect_tr": v["tr"] - v["base"], "effect_nos": v["nos"] - v["base"],
                                "effect_both": v["both"] - v["base"],
                                "interaction": (v["both"] - v["nos"]) - (v["tr"] - v["base"])}
        gm = None
        if all(seed_key(s) in per for s in GATE_SEEDS):
            gm = {f: mean([per[seed_key(s)][f] for s in GATE_SEEDS])
                  for f in ("base", "tr", "nos", "both", "effect_tr", "effect_nos", "effect_both", "interaction")}
        out["metrics"][key] = {"per_seed": per, "gate_mean": gm}
    return out


def cam_vs_nos(runs, cam="dt_cam", nos="dt_nos"):
    """Descriptive: does the ray-plane input add anything beyond removing the scale augmentation (cam - nos)?"""
    a, b = runs.get(cam, {}), runs.get(nos, {})
    if not a or not b:
        return None
    P = Pair(cam, nos, a, b)
    out = {"definition": f"paired difference {cam} - {nos} (descriptive, no gate)", "metrics": {}}
    for key in ("ra", "abs"):
        gm = P.mean_d(key)
        out["metrics"][key] = {"per_seed": {seed_key(s): P.d(key, s) for s in P.common_seeds()}, "gate_mean": gm,
                               "reading": reading(P, key)}
    return out


# ---------------------------------------------------------------------------------------------- the report
def mean_row(runs):
    """Mean over the gate seeds of every metric, or None unless every gate seed has a run."""
    if not all(s in runs for s in GATE_SEEDS):
        return None
    out = {"m": {}, "jitter": None, "jitter_src": None, "params": None, "grid_median": None}
    for k in runs[GATE_SEEDS[0]]["m"]:
        vs = [runs[s]["m"].get(k) for s in GATE_SEEDS]
        out["m"][k] = None if any(v is None for v in vs) else mean(vs)
    js = [runs[s]["jitter"] for s in GATE_SEEDS]
    if all(j is not None for j in js):
        out["jitter"] = {k: mean([j[k] for j in js]) for k in js[0] if all(k in j for j in js)}
        out["jitter_src"] = "+".join(sorted({runs[s]["jitter_src"] for s in GATE_SEEDS}))
    out["params"] = _arm_params(runs)
    gm = [runs[s]["grid_median"] for s in GATE_SEEDS]
    out["grid_median"] = None if any(v is None for v in gm) else mean(gm)
    return out


def build_report(runs_dir=None, base="dt_base", arms=None, latency_json=None, offline_json=DEFAULT_OFFLINE,
                 want_params=True, expect_step=EXPECT_STEP, eval_name=EVAL_NAME):
    """Load, pair and judge. Raises ScreenError on a missing baseline or an unreadable json."""
    runs_dir = Path(runs_dir) if runs_dir else DEFAULT_RUNS
    if not runs_dir.is_dir():
        raise ScreenError(f"runs directory {runs_dir} does not exist")
    warnings = []
    latency = load_latency(latency_json, warnings)
    offline = load_offline(offline_json)
    if arms is None:
        arms = list(DEFAULT_ARMS) + [a for a in discover_dt_arms(runs_dir, eval_name)
                                     if a not in DEFAULT_ARMS and a != base]
    if base in arms:
        warnings.append(f"{base} is the baseline and is not a candidate: left out of --arms")
    arms = [a for i, a in enumerate(arms) if a != base and a not in arms[:i]]
    helpers = []
    for both, parts in FACTORIALS.items():
        if both in arms:
            helpers += [a for a in parts if a not in arms and a not in helpers]
    if "dt_cam" in arms and "dt_nos" not in arms and "dt_nos" not in helpers:
        helpers.append("dt_nos")

    runs, no_json = {}, {}
    for arm in [base] + arms + helpers:
        have, without = discover(runs_dir, arm, eval_name)
        no_json[arm] = without
        runs[arm] = {s: load_run(d, arm, s, offline, want_params, eval_name) for s, d in sorted(have.items())}
    if not runs[base]:
        raise ScreenError(
            f"baseline arm '{base}' has no evaluated run: expected {runs_dir}/{base}_s<seed>/{eval_name} "
            f"(written by tools/tracking/evalx.py eval --ckpt last --controls --tf --perturb); "
            f"run directories without that file: {no_json[base] or 'none'}")
    miss_base = [s for s in GATE_SEEDS if s not in runs[base]]
    if miss_base:
        warnings.append(f"baseline {base} lacks gate seed(s) {miss_base}: the gates of every arm are incomplete")
    for arm, rs in runs.items():
        for s, r in rs.items():
            if expect_step and r["step"] != expect_step:
                warnings.append(f"{r['run']} was evaluated at step {na(r['step'])}, the gates are defined at "
                                f"{expect_step}")
            if r["params_note"] and want_params:
                warnings.append(f"{r['run']}: parameters n/a ({r['params_note']})")
        for s in sorted(set(rs) & set(runs[base])):
            if arm != base and rs[s]["n_frames"] != runs[base][s]["n_frames"]:
                warnings.append(f"{rs[s]['run']} and {runs[base][s]['run']} differ in the number of evaluated frames "
                                f"({na(rs[s]['n_frames'])} vs {na(runs[base][s]['n_frames'])}): not the same protocol")
    for arm in arms:
        mix = jitter_source_mix(runs[arm], runs[base])
        if mix:
            warnings.append(f"{arm}: the jitter values of the paired runs come from different sources ("
                            + "; ".join(f"seed {s}: arm {a}, base {b}" for s, a, b in mix) + "): the jitter "
                            "guardrails G3a / G3b and the jitter differences compare an evaluation's own jitter block "
                            "with the offline attribution file (two different programs); read them as approximate")
    shown = [base] + arms
    absent = [f"{a} (directories without the json: {', '.join(f's{s}' for s in no_json[a])})" if no_json[a] else a
              for a in arms if not runs[a]]
    if absent:
        warnings.append("arms without an evaluated run: " + "; ".join(absent))
    gates = {arm: evaluate_arm(arm, base, runs[arm], runs[base], latency) for arm in arms if runs[arm]}
    means = {a: mean_row(runs[a]) for a in shown if runs[a]}
    factorials = {both: factorial(runs, base, tr=parts[0], nos=parts[1], both=both)
                  for both, parts in FACTORIALS.items() if both in arms and runs.get(both)}
    rep = {
        "meta": {"tool": "tools/dt/screen_report.py", "runs_dir": str(runs_dir), "eval_json": eval_name,
                 "base": base, "arms": arms, "helper_arms": helpers,
                 "gate_seeds": [seed_key(s) for s in GATE_SEEDS], "confirm_seed": seed_key(CONFIRM_SEED),
                 "tie_band_mm": TIE_MM, "pending_band_hi_mm": BAND_HI_MM, "tolerance": EPS,
                 "expect_step": expect_step, "latency_json": str(latency_json) if latency_json else None,
                 "latency_ms": latency, "offline_json": str(offline_json) if offline else None,
                 "params": "numel of the last checkpoint's state_dict without BN statistics, mano.* and teacher.*"
                           if want_params else "not counted",
                 "run_dirs_without_eval": {a: v for a, v in no_json.items() if v},
                 "inputs": {r["run"]: {"json": r["json"], "sha1": r["sha1"]}
                            for a in [base] + arms + helpers for r in runs[a].values()}},
        "warnings": warnings,
        "runs": {r["run"]: {k: v for k, v in r.items() if k not in ("run",)}
                 for a in shown for r in runs[a].values()},
        "arm_means": {a: v for a, v in means.items() if v},
        "base_spread": _spread(runs[base]),
        "gates": gates,
        "factorials": factorials,
        "factorial": next(iter(factorials.values()), None),
        "cam_vs_nos": cam_vs_nos(runs) if "dt_cam" in arms and runs.get("dt_cam") else None,
    }
    return rep


def _spread(base_runs):
    out = {}
    for key in ("ra", "abs", "ra_local", "rot"):
        vs = {seed_key(s): r["m"].get(key) for s, r in sorted(base_runs.items())}
        have = [v for v in vs.values() if v is not None]
        out[key] = {"per_seed": vs, "range": (max(have) - min(have)) if len(have) > 1 else None}
    return out


# ---------------------------------------------------------------------------------------------- markdown
READING_GUIDE = (
    "**Reading guide.** Pairing rule: the paired difference of a metric m for seed s is d(m, s) = m(arm, s) - "
    "m(base, s) (same seed, last checkpoint, no checkpoint selection); a gate uses the mean over seeds {gs} only, "
    "and seed {cs} is reported separately as confirmation when both arms have it. Tie band: |difference| < {tie} mm "
    "is a tie. Two seeds resolve only differences of {hi}-3 mm or more reliably, so a passing result whose primary "
    "effect lies in [{tie}, {hi}) mm is labelled \"{pend}\". zgz is both the development set and the test set "
    "(AGENTS.md): every conclusion carries that qualifier."
)
CONVENTIONS = (
    "**Conventions.** \"Both seeds the same sign as the mean\" is required of every improvement clause (a negative "
    "threshold); a bound clause (a non-negative margin) is tested on the mean. The absolute-MPJPE gates apply the "
    "band to the ABS difference; each gate's \"band rule\" line says what is done for relative and angle gates. Any "
    "universal guardrail that trips blocks adoption even when the gate passes. The absolute errors of zgz_local "
    "reflect depth extrapolation (preregistration F4), not only tracking accuracy. n/a = the value is not in the "
    "evaluation json (older runs carry no jitter block); source \"offline\" marks jitter values taken from the "
    "offline attribution file (same checkpoint and frame count). An efficiency arm that passes only through its "
    "cost branch is worded \"efficiency gate met\" (non-inferior accuracy at lower cost): it claims no accuracy gain."
)


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(na(c)) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _run_rows(rep):
    base, arms = rep["meta"]["base"], rep["meta"]["arms"]
    lat = rep["meta"]["latency_ms"]
    rows, jrows = [], []
    for arm in [base] + arms:
        rs = sorted((r for r in rep["runs"].values() if r["arm"] == arm), key=lambda r: r["seed"])
        for r in rs:
            tag = f"{r['arm']}_s{r['seed']}" + (" (base)" if arm == base else "")
            row = [tag, na(r["step"])] + [fmt(r["m"].get(k), nd) for k, _, nd in RUN_COLS]
            row += [r["params"] if r["params"] is not None else "n/a", fmt(r["grid_median"], 2)]
            if lat:
                row.append(fmt(lat.get(arm), 2))
            rows.append(row)
            j = r["jitter"] or {}
            jrows.append([tag, r["jitter_src"] or "n/a"] + [fmt(j.get(k), nd) for k, _, nd in JIT_COLS])
        am = rep["arm_means"].get(arm)
        if am:
            row = [f"**{arm} mean(3407, 3408)**", ""] + [fmt(am["m"].get(k), nd) for k, _, nd in RUN_COLS]
            row += [am["params"] if am["params"] is not None else "n/a", fmt(am["grid_median"], 2)]
            if lat:
                row.append(fmt(lat.get(arm), 2))
            rows.append(row)
            j = am["jitter"] or {}
            jrows.append([f"**{arm} mean(3407, 3408)**", am["jitter_src"] or "n/a"] +
                         [fmt(j.get(k), nd) for k, _, nd in JIT_COLS])
    return rows, jrows


def _val_str(c, nd):
    """The measured value of a clause with its unit."""
    if c["relative"] and c["kind"] == "improve":
        return f"{fmt(c['value'] * 100, nd, True)} %"
    if c["kind"] in ("noninf", "cost"):
        return f"{c['value']:.4f} x"
    return f"{fmt(c['value'], nd, True)} {c['unit']}"


def _qty(c):
    return f"{c['short']} {_val_str(c, 2)}"


def _paired_qty(rec):
    """Mean paired differences of an arm without a gate (RA, ABS in mm, ROT in deg), for the summary table."""
    p = rec["paired"]
    return "; ".join(f"d{n} {fmt(p[k]['gate_mean'], 2, True)} {u}"
                     for k, n, u in (("ra", "RA", "mm"), ("abs", "ABS", "mm"), ("rot", "ROT", "deg"))
                     if p[k]["gate_mean"] is not None)


def _summary_qty(g, f):
    """The quantities of a summary row: the gate clauses, or the paired differences (plus, for a factorial arm `f`,
    the interaction of RA and ABS over the gate seeds)."""
    if "clauses" in g:
        return "; ".join(_qty(c) for c in g["clauses"] if c["value"] is not None)
    inter = [f"interaction {k.upper()} {fmt(f['metrics'][k]['gate_mean']['interaction'], 2, True)} mm"
             for k in ("ra", "abs") if f and f["metrics"][k]["gate_mean"]]
    return "; ".join([q for q in [_paired_qty(g)] if q] + inter)


def _clause_line(c):
    if c["value"] is None:
        val = "n/a (" + "; ".join(c["missing"]) + ")" if c["missing"] else "n/a"
    else:
        val = _val_str(c, 3)
    per = ""
    if c.get("per_seed_relative"):
        per = " (" + "; ".join(f"{s}: {fmt(d * 100, 2, True)} %" for s, d in c["per_seed_relative"].items()) + ")"
    elif c["per_seed"]:
        per = " (" + "; ".join(f"{s}: {fmt(d, 3, True)}" for s, d in c["per_seed"].items()) + ")"
    met = {True: "met", False: "NOT met", None: "n/a"}[c["met"]]
    return f"- {c['id']}: {c['text']}: {c['short']} = {val}{per} -> {met}"


def _guard_line(g):
    if g["value"] is None:
        val = "n/a (" + "; ".join(g["missing"]) + ")" if g["missing"] else "n/a"
    elif g["kind"] == "relative_rise":
        val = f"{fmt(g['value'] * 100, 2, True)} %"
    else:
        val = fmt(g["value"], 3, g["kind"] == "mean_delta")
    st = {True: "TRIPPED", False: "ok", None: "n/a"}[g["tripped"]]
    return f"- {g['id']}: {g['text']}: {val} -> {st}"


def _gate_md(rec):
    L = [f"### {rec['arm']}" + (f" ({rec['label']})" if rec.get("label") else ""), ""]
    s = rec["seeds"]
    L.append(f"Seeds with an evaluation: arm {', '.join(s['arm']) or 'none'}; base {', '.join(s['base']) or 'none'}. "
             + (f"Gate seeds missing: {', '.join(s['missing_gate_seeds'])}." if s["missing_gate_seeds"] else
                "Both gate seeds are present."))
    if rec["rule"]:
        L += ["", f"Rule: {rec['rule']}."]
    rows = []
    for key, p in rec["paired"].items():
        ps = p["per_seed"]
        rows.append([p["label"], fmt(p["base_mean"], p["decimals"]), fmt(p["arm_mean"], p["decimals"]),
                     fmt(ps.get("3407"), p["decimals"], True), fmt(ps.get("3408"), p["decimals"], True),
                     fmt(p["gate_mean"], p["decimals"], True),
                     "n/a" if p["gate_mean"] is None else "-" if p["gate_mean"] == 0
                     else "yes" if p["same_sign"] else "no",
                     fmt(ps.get("3409"), p["decimals"], True), p["reading"] or ""])
    L += ["", md_table(["metric", "base mean", "arm mean", "d 3407", "d 3408", "mean d (3407, 3408)", "same sign",
                        "d 3409 (confirmation)", "reading (tie band)"], rows)]
    if rec["kind"] == "gate":
        L += ["", f"Gate clauses ({rec['expression']}):", ""] + [_clause_line(c) for c in rec["clauses"]]
        if "efficiency" in rec:
            e = rec["efficiency"]
            L.append(f"- parameters: arm {na(e['params_arm'])}, base {na(e['params_base'])}; latency: {e['latency']}"
                     + (f" (arm {e['latency_arm_ms']} ms, base {e['latency_base_ms']} ms)"
                        if e["latency"] == "measured" else " (no --latency-json entry for the arm and the base)"))
        L += ["", f"Gate: {'PASS' if rec['gate_pass'] else 'FAIL' if rec['gate_pass'] is False else 'INCOMPLETE'}"]
        L += ["", "Universal guardrails (any trip blocks adoption):", ""] + [_guard_line(g) for g in rec["guardrails"]]
        b = rec["band"]
        L += ["", f"Band rule: {b['rule']}"]
        if b["notes"]:
            L.append("")
        for n in b["notes"]:
            L.append(f"- note: {n} (that claim is pending 3409; the pass does not rest on it)")
        c = rec["confirmation"]
        L.append("")
        if c["available"]:
            L.append(f"Seed {c['seed']} (confirmation): " + "; ".join(
                f"d{k} = {fmt(v, 3, True)} (same sign as the 2-seed mean: "
                f"{'n/a' if c['same_sign_as_mean'][k] is None else 'yes' if c['same_sign_as_mean'][k] else 'no'}; "
                f"3-seed mean {fmt(c['three_seed_mean'][k], 3, True)})" for k, v in c["diffs"].items())
                + ". The preregistration gives no numeric confirmation rule: judged by the verdict author.")
        else:
            L.append(f"Seed {c['seed']} (confirmation): not available (arm: {'yes' if c['arm_has_seed'] else 'no'}, "
                     f"base: {'yes' if c['base_has_seed'] else 'no'}).")
        L += ["", f"**Verdict: {rec['verdict']}**" + (f" (wording: {rec['wording']})" if rec["wording"] else "")]
    else:
        L += ["", f"**{rec['verdict']}**"]
    return "\n".join(L)


def _factorial_md(both, f):
    """The markdown block of one descriptive 2 x 2 factorial (dt_trnos, dt_trdz)."""
    lab = f["labels"]
    L = [f"### {both}: 2 x 2 factorial (descriptive, no gate)", "", f["definition"] + ".", ""]
    fr = []
    for key in ("ra", "abs"):
        for s, v in f["metrics"][key]["per_seed"].items():
            fr.append([key.upper(), s, fmt(v["base"]), fmt(v["tr"]), fmt(v["nos"]), fmt(v["both"]),
                       fmt(v["effect_tr"], 2, True), fmt(v["effect_nos"], 2, True), fmt(v["effect_both"], 2, True),
                       fmt(v["interaction"], 3, True)])
        gm = f["metrics"][key]["gate_mean"]
        if gm:
            fr.append([key.upper(), "**mean(3407, 3408)**", fmt(gm["base"]), fmt(gm["tr"]), fmt(gm["nos"]),
                       fmt(gm["both"]), fmt(gm["effect_tr"], 2, True), fmt(gm["effect_nos"], 2, True),
                       fmt(gm["effect_both"], 2, True), f"**{fmt(gm['interaction'], 3, True)}**"])
    L.append(md_table(["metric", "seed", lab["base"], lab["tr"], lab["nos"], lab["both"], f"effect {lab['tr']}",
                       f"effect {lab['nos']}", f"effect {lab['both']}", "interaction"], fr))
    L += ["", "Seeds present: " + "; ".join(f"{f['arms'][k]}: {', '.join(v) or 'none'}"
                                           for k, v in f["seeds_present"].items())
          + ". An |interaction| below the 1.1 mm tie band is not resolvable with two seeds."]
    if not any(f["metrics"][k]["per_seed"] for k in ("ra", "abs")):
        L.append("No seed is present in all four cells: no interaction can be formed.")
    return L + [""]


def render_markdown(rep):
    m = rep["meta"]
    L = [f"# DT round screen report: paired gates against {m['base']}", "",
         READING_GUIDE.format(gs=" and ".join(m["gate_seeds"]), cs=m["confirm_seed"], tie=m["tie_band_mm"],
                              hi=m["pending_band_hi_mm"], pend=PENDING_LABEL), "", CONVENTIONS, ""]
    L.append(f"Runs: `{m['runs_dir']}`; evaluation file `{m['eval_json']}`; baseline `{m['base']}`; "
             f"parameters: {m['params']}; jitter fallback: "
             f"{('offline file ' + m['offline_json']) if m['offline_json'] else 'off'}.")
    if rep["warnings"]:
        L += ["", "**Warnings**"] + [f"- {w}" for w in rep["warnings"]]
    L += ["", "## Summary", ""]
    rows = []
    for arm in m["arms"]:
        g = rep["gates"].get(arm)
        if g is None:
            rows.append([arm, GATES.get(arm, (None, ""))[1], "no evaluated run", "", "", "", "", ""])
            continue
        gp = g.get("gate_pass")
        rows.append([arm, g.get("label") or "",
                     f"arm {','.join(g['seeds']['arm'])} / base {','.join(g['seeds']['base'])}",
                     _summary_qty(g, rep["factorials"].get(arm)),
                     "" if g["kind"] != "gate" else "PASS" if gp else "FAIL" if gp is False else "INCOMPLETE",
                     g.get("guardrail_status", ""), g["verdict"], g.get("wording") or ""])
    L.append(md_table(["arm", "registered", "seeds", "gate quantities (mean over 3407, 3408)", "gate",
                       "guardrails", "verdict", "wording"], rows))
    sp = rep["base_spread"]
    L += ["", f"Baseline seed-to-seed spread ({m['base']}, range over its seeds): "
          + "; ".join(f"{PAIRED_NAMES[k]} {fmt(v['range'], 2)} "
                      f"({', '.join(f'{s}: {fmt(x, 2)}' for s, x in v['per_seed'].items())})"
                      for k, v in sp.items()) + "."]
    L += ["", "## 1. Per-run table", "", "### 1a. Accuracy, dynamics, robustness (last checkpoint, val_core)", ""]
    rows, jrows = _run_rows(rep)
    heads = ["run", "step"] + [h for _, h, _ in RUN_COLS] + ["params", "grid median RA"]
    if m["latency_ms"]:
        heads.append("latency (ms)")
    L.append(md_table(heads, rows))
    L += ["", "### 1b. Jitter and acceleration (`evalx.jitter_decomp`, frame-weighted, mm per step / step^2)", ""]
    L.append(md_table(["run", "source"] + [h for _, h, _ in JIT_COLS], jrows))
    L += ["", "## 2. Gates (seeds 3407 and 3408; d = arm - base)", ""]
    for arm in m["arms"]:
        if arm in rep["gates"]:
            L += [_gate_md(rep["gates"][arm]), ""]
    if rep["cam_vs_nos"]:
        c = rep["cam_vs_nos"]
        L += ["### dt_cam against dt_nos (descriptive)", "", c["definition"] + ": " + "; ".join(
            f"{k}: mean {fmt(v['gate_mean'], 2, True)} mm ({', '.join(f'{s}: {fmt(x, 2, True)}' for s, x in v['per_seed'].items())})"
            f"{', ' + v['reading'] if v['reading'] else ''}" for k, v in c["metrics"].items())
              + ". Negative: the ray-plane input adds something beyond removing the scale augmentation.", ""]
    for both, f in rep["factorials"].items():
        L += _factorial_md(both, f)
    return "\n".join(L).rstrip() + "\n"


def write_outputs(rep, prefix):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    md, js = prefix.with_name(prefix.name + ".md"), prefix.with_name(prefix.name + ".json")
    md.write_text(render_markdown(rep), encoding="utf-8")
    js.write_text(json.dumps(rep, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    return md, js


def main(argv=None):
    ap = argparse.ArgumentParser(description="Paired-gate report of the DT round (preregistration section 5).")
    ap.add_argument("--base", default="dt_base", help="baseline arm (default dt_base)")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="candidate arms (default: every registered DT arm with an evaluated run, plus any other "
                         "dt_* arm, which gets no gate)")
    ap.add_argument("--out-prefix", default=str(DEFAULT_OUT), help="writes <prefix>.md and <prefix>.json")
    ap.add_argument("--runs-dir", default=str(DEFAULT_RUNS), help="directory of the <arm>_s<seed> run directories")
    ap.add_argument("--latency-json", default=None,
                    help="optional {arm: ms} (positive numbers; the latency clause needs the arm and the baseline): "
                         "formal latency for the efficiency gate")
    ap.add_argument("--offline-json", default=str(DEFAULT_OFFLINE),
                    help="offline jitter attribution file used when a run has no jitter block")
    ap.add_argument("--no-offline", action="store_true", help="do not use the offline jitter file")
    ap.add_argument("--no-params", action="store_true", help="do not read checkpoints (parameters: n/a)")
    ap.add_argument("--expect-step", type=int, default=EXPECT_STEP, help="warn for runs evaluated at another step "
                    "(0: off)")
    ap.add_argument("--eval-json", default=EVAL_NAME, help="name of the evaluation json inside every run directory "
                    "(default: the one `evalx.py eval --ckpt last --controls --tf --perturb` writes)")
    a = ap.parse_args(argv)
    try:
        rep = build_report(a.runs_dir, a.base, a.arms, a.latency_json, None if a.no_offline else a.offline_json,
                           not a.no_params, a.expect_step, a.eval_json)
    except ScreenError as e:
        print(f"screen_report: ERROR: {e}", file=sys.stderr)
        return 2
    md, js = write_outputs(rep, a.out_prefix)
    for w in rep["warnings"]:
        print(f"screen_report: warning: {w}", file=sys.stderr)
    print(f"wrote {md}\nwrote {js}")
    for arm in rep["meta"]["arms"]:
        g = rep["gates"].get(arm)
        print(f"  {arm:10s} {g['verdict'] if g else 'no evaluated run'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
