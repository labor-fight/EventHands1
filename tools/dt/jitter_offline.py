#!/usr/bin/env python3
"""Offline attribution of the frame-to-frame jitter of every recorded closed-loop run (DT round).

Every closed-loop evaluation of this project saved its per-step arrays in a compressed npz next to its
json (`evalx_val_core_<ckpt>[_tf][_pert][_<suffix>].npz`, keys `model|<seq>|pred / gt / run / ...`). This tool
replays `evalx.jitter_decomp` on those arrays -- no model, no GPU, no new closed loop -- and writes

    <prefix>.json   per run: per-sequence and frame-weighted overall values of every float key of
                    `jitter_decomp` (+ n_steps), the Shapley attribution of the error acceleration to the
                    three blocks of the state, the recursive RA of the same arrays, per-arm seed means
    <prefix>.md     the same as tables

    python tools/dt/jitter_offline.py --runs 'rt_cnntrack_s*' 'rt_cnn_s*' hand_data51/track_render51_dr_so3fk \
        --out-prefix outputs/dt/reports/jitter_offline [--variants] [--jobs 4]

Shapley attribution of `acc_err_mm` (the acceleration of the error pred - gt, mm per step^2). The game is
played by the three blocks X in {transl, root, fingers}; the value of a coalition S is the error acceleration
of the state in which the blocks of S come from the prediction and the others from the ground truth:

    v({})      = 0                       (the ground truth itself)
    v({X})     = acc_err_only_X          (ground truth with block X from the prediction)
    v({Y, Z})  = acc_err_allbut_X        (prediction with block X from the ground truth)
    v(all)     = acc_err                 (the prediction)

    phi_X = 1/3 [v({X}) - v({})] + 1/6 [v({X,Y}) - v({Y})] + 1/6 [v({X,Z}) - v({Z})] + 1/3 [v(all) - v({Y,Z})]

The three phi add up to acc_err (efficiency; asserted to 1e-6) and `share_X = phi_X / acc_err`. The value is a
mean of norms of a nonlinear (forward-kinematics) function of the state, so the split is a fair accounting of
the measured number, not a linear decomposition: a share can be negative or above 100 % when two blocks'
errors partly cancel, and a block's phi moves when the OTHER blocks' errors change even if the block itself did
not (compare one block across arms with its stand-alone counterfactual `acc_err_only_X` / `jit_only_X`). The
first-difference jitter is NOT additive: `jit_only_X` is reported raw.

The overall value of every float key is the mean over the sequences that have the key, weighted by the
sequence's frame count (what `evalx.evaluate` does for its own `jitter` entry); `jit_ratio_pooled` /
`acc_ratio_pooled` are ratios of the overall means.

Diverged runs. A sequence whose `pred` / `gt` (or recorded error arrays) hold NaN / inf is not analysed: it carries
`nonfinite` (which arrays, how many values / frames, the first frame) instead of values, and its run gets NO overall
values (never a mean over the remaining sequences): the run is flagged `nonfinite`, marked with a double dagger in
the tables, left out of the counts, medians and arm means, and a warning goes to stderr. A sequence whose
`jitter_decomp` values turn out non-finite although it has steps to measure (overflow) is treated the same way.
Runs whose per-step predictions are byte-identical (`pred_sha1`) are listed in `meta.identical_evaluations`.
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import itertools
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "tools" / "tracking", REPO / "tools", REPO / "model", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import evalx as EX                                                            # noqa: E402
from config import load_config                                                # noqa: E402
from mano_layer import ManoLayer                                              # noqa: E402
from semkine import eval_track as ET                                          # noqa: E402
from semkine.dataset import sequences_for_split                               # noqa: E402

BLOCKS = tuple(EX.JIT_PARTS)                      # ("transl", "root", "fingers")
#: the recorded evaluation a run is read from: the first pattern that matches a file in the run directory
DEFAULT_NPZ = ("evalx_val_core_last_tf_pert.npz", "evalx_val_core_last.npz", "evalx_val_core_step*_tf_pert_dt.npz")
DEFAULT_CONFIG = "configs/rt/rt_cnntrack.yaml"
#: runs whose per-sequence rows go into the markdown (the JSON always has every sequence)
DEFAULT_DETAIL = ("rt_cnntrack_s*", "rt_cnn_s*", "rt_cnnf_s*", "rt_s37_s*", "track_render51_dr_so3fk*")
SHAPLEY_TOL = 1e-6
#: a predicted translation step above this (mm per 50 ms step) is a jump, not jitter: the ground truth of the two
#: validation sequences never moves more than 21.7 mm per step. A run with >= RUNAWAY_FRAC of its steps jumping is
#: marked as a translation runaway: its first-difference jitter is the runaway's speed.
JUMP_MM = 100.0
RUNAWAY_FRAC = 0.05
#: keys whose overall value is the maximum over the sequences, not a frame-weighted mean
MAX_KEYS = ("transl_step_max_mm",)
#: arrays recorded next to the predictions: (key in the report, key in the npz)
RECORDED = (("ra_mpjpe_mm", "mpjpe_ra_mm"), ("root_rot_deg", "root_rot_deg"), ("transl_err_mm", "transl_mm"))
#: keys of `jitter_decomp` that are finite whenever the sequence has first / second differences to measure
FIRST_DIFF_KEYS = ("jit_gt_mm", "jit_pred_mm") + tuple(f"jit_only_{b}_mm" for b in BLOCKS)
SECOND_DIFF_KEYS = (("acc_gt_mm", "acc_pred_mm", "acc_err_mm", "rot_acc_pred_deg", "rot_acc_gt_deg")
                    + tuple(f"acc_err_{kind}_{b}_mm" for b in BLOCKS for kind in ("only", "allbut")))
#: arm pairs of the documentation page: (from, to, what differs). The first two are single-setting contrasts and are
#: checked against the YAML files by tests/test_dt_jitter.py; the filter row against the runs' training_metadata.json.
CONTRASTS = (
    ("rt_cnn", "rt_cnndelta",
     "`MODEL.PREDICT_DELTA`, `PREVPOS_EMBED`, `ZERO_EVENT_GATE` false -> true: the previous state enters through an MLP and "
     "the net predicts a delta; no rendered state (configs/rt/rt_cnn.yaml vs rt_cnndelta.yaml, nothing else differs)"),
    ("rt_cnndelta", "rt_cnntrack",
     "`MODEL.PREV_RENDER` false -> true (with the `RENDER_H/W/SCALE/CHUNK` sizes it needs): the rendered silhouette and "
     "inverse depth of the previous state are extra input channels (configs/rt/rt_cnndelta.yaml vs rt_cnntrack.yaml, "
     "nothing else differs)"),
    ("rt_cnn", "rt_cnnf",
     "the same `rt_cnn` checkpoints behind a causal filter with gains root 0.5, fingers 0.5 and translation 1.0 "
     "(`outputs/semkine/rt_cnnf_s*/training_metadata.json`): the translation passes the filter unchanged"),
    ("rt_cnntrack", "track_render51_dr_so3fk",
     "same architecture (4-channel ResNet18, 11,209,429 parameters; both historical checkpoints load strictly under both "
     "configs), different training: `LOSS.TYPE` so3_trans_fk vs the default mse_51d, `MODEL.INIT_FROM` the abs_full51 "
     "step-6000 checkpoint vs none, 3000 vs 6000 steps, batch 2048 vs 512 x 2 accumulation, LR 0.005656 with no schedule "
     "key (constant after the warmup) vs 0.004 with cosine decay, the older flat `AUG.*` keys vs `AUG.DOMRAND.*` (same "
     "numbers; whether both reach the same augmentation was not checked), seed; which of these matters was not tested"),
)


# ------------------------------------------------------------------------------------ Shapley
def coalition_values(j: dict) -> dict:
    """The 8 coalition values of the error-acceleration game from the keys `jitter_decomp` returns."""
    v = {frozenset(): 0.0, frozenset(BLOCKS): j["acc_err_mm"]}
    for b in BLOCKS:
        v[frozenset([b])] = j[f"acc_err_only_{b}_mm"]
        v[frozenset(set(BLOCKS) - {b})] = j[f"acc_err_allbut_{b}_mm"]
    return v


def shapley(v: dict, players=BLOCKS) -> dict:
    """Exact Shapley values of a cooperative game given as {frozenset(coalition): value} (all subsets)."""
    n = len(players)
    phi = {}
    for x in players:
        others = [p for p in players if p != x]
        tot = 0.0
        for k in range(n):
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for s in itertools.combinations(others, k):
                s = frozenset(s)
                tot += w * (v[s | {x}] - v[s])
        phi[x] = tot
    return phi


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def add_shapley(j: dict, tol: float = SHAPLEY_TOL) -> dict:
    """Add `shap_<block>_mm` and `shap_share_<block>` to a jitter_decomp result (in place); asserts efficiency."""
    need = ["acc_err_mm"] + [f"acc_err_{kind}_{b}_mm" for b in BLOCKS for kind in ("only", "allbut")]
    if not all(_finite(j.get(k)) for k in need):
        return j
    phi = shapley(coalition_values(j))
    total = sum(phi.values())
    assert abs(total - j["acc_err_mm"]) < tol, f"Shapley values sum to {total}, acc_err_mm is {j['acc_err_mm']}"
    for b in BLOCKS:
        j[f"shap_{b}_mm"] = float(phi[b])
        j[f"shap_share_{b}"] = float(phi[b] / j["acc_err_mm"]) if j["acc_err_mm"] > 1e-9 else None
    return j


def dominant(j: dict):
    """The block with the largest Shapley value (None when the attribution is not available)."""
    if any(f"shap_{b}_mm" not in j for b in BLOCKS):
        return None
    return max(BLOCKS, key=lambda b: j[f"shap_{b}_mm"])


def dominance(j: dict, margin: float = 0.10) -> str:
    """`dominant` as a label: `a~b` when the two largest shares are within `margin` of each other; `non-finite`
    for a diverged sequence / run."""
    if j.get("nonfinite") or j.get("nonfinite_sequences"):
        return "non-finite"
    sh = {b: j.get(f"shap_share_{b}") for b in BLOCKS}
    if any(not _finite(v) for v in sh.values()):
        return "n/a"
    first, second = sorted(BLOCKS, key=lambda b: -sh[b])[:2]
    return first if sh[first] - sh[second] >= margin else f"{first}~{second}"


def runaway(o: dict) -> bool:
    """True when >= RUNAWAY_FRAC of the steps move the predicted translation by more than JUMP_MM."""
    return (o.get("n_steps", 0) > 0 and not o.get("nonfinite_sequences")
            and o.get("n_transl_jump", 0) >= RUNAWAY_FRAC * o["n_steps"])


def nonfinite_info(arrays: dict):
    """None when every array is finite, else {name: {"n_values", "n_frames", "first_frame"}} for those that are not
    (the first axis of an array is the frame)."""
    bad = {}
    for name, a in arrays.items():
        ok = np.isfinite(np.asarray(a))
        if not ok.all():
            frames = ~ok.reshape(len(ok), -1).all(axis=1)
            bad[name] = {"n_values": int((~ok).sum()), "n_frames": int(frames.sum()),
                         "first_frame": int(np.flatnonzero(frames)[0])}
    return bad or None


def undefined_keys(j: dict, same: np.ndarray) -> list:
    """Keys of a `jitter_decomp` result that must be finite (the sequence has first / second differences to measure,
    `same` = steps inside a segment) but are not: an overflow in the kinematics, say."""
    same2 = same[1:] & same[:-1]
    need = (FIRST_DIFF_KEYS if same.any() else ()) + (SECOND_DIFF_KEYS if same2.any() else ())
    return [k for k in need if not _finite(j.get(k))]


# ------------------------------------------------------------------------------------ aggregation
def combine(per_seq: dict) -> dict:
    """Frame-weighted overall of per-sequence results (see the module docstring). If any sequence is non-finite the
    run has no overall values: only the counts and `nonfinite_sequences`."""
    bad = sorted(s for s, d in per_seq.items() if d.get("nonfinite"))
    if bad:
        out = {k: int(sum(d[k] for d in per_seq.values() if k in d)) for k in ("n_frames", "n_steps", "n_segments")}
        out["nonfinite_sequences"] = bad
        return out
    out = {}
    keys = []
    for d in per_seq.values():
        keys += [k for k, v in d.items() if isinstance(v, float) and k not in keys]
    for k in keys:
        pairs = [(d["n_frames"], d[k]) for d in per_seq.values() if _finite(d.get(k))]
        if pairs and k in MAX_KEYS:
            out[k] = float(max(x for _, x in pairs))
        elif pairs:
            out[k] = float(sum(n * x for n, x in pairs) / sum(n for n, _ in pairs))
    for k in ("n_frames", "n_steps", "n_static", "n_segments", "n_transl_jump"):
        out[k] = int(sum(d[k] for d in per_seq.values() if k in d))
    finish(out)
    # linearity check: the Shapley values of the overall game are the frame-weighted mean of the sequences' own
    for b in BLOCKS:
        if f"shap_{b}_mm" in out and all(f"shap_{b}_mm" in d for d in per_seq.values()):
            mean = sum(d["n_frames"] * d[f"shap_{b}_mm"] for d in per_seq.values()) / out["n_frames"]
            assert abs(mean - out[f"shap_{b}_mm"]) < 1e-9, (b, mean, out[f"shap_{b}_mm"])
    return out


def finish(o: dict) -> dict:
    """Pooled ratios and the Shapley attribution of an overall (or arm-mean) dict, from its own means."""
    if _finite(o.get("jit_gt_mm")) and _finite(o.get("jit_pred_mm")):
        o["jit_ratio_pooled"] = o["jit_pred_mm"] / max(o["jit_gt_mm"], 1e-9)
    if _finite(o.get("acc_gt_mm")) and _finite(o.get("acc_pred_mm")):
        o["acc_ratio_pooled"] = o["acc_pred_mm"] / max(o["acc_gt_mm"], 1e-9)
    add_shapley(o)
    return o


def arm_of(name: str) -> str:
    """Run name without its seed / replicate suffix (`rt_cnn_s3407` -> `rt_cnn`)."""
    return re.sub(r"(_s\d+|_rep\d+)$", "", name)


def arm_means(runs: dict) -> dict:
    """Seed means (plain mean over the runs of an arm) of the overall values; shares from the mean Shapley values.
    Non-finite runs have no overall values: they are listed in `nonfinite_runs` and left out of the mean."""
    arms = {}
    for name, r in runs.items():
        arms.setdefault(arm_of(name), []).append(name)
    out = {}
    for arm, names in sorted(arms.items()):
        bad = [n for n in names if runs[n]["overall"].get("nonfinite_sequences")]
        ov = [runs[n]["overall"] for n in names if n not in bad]
        mean, sd = {}, {}
        if ov:
            for k in sorted({k for o in ov for k, v in o.items() if isinstance(v, float)}):
                xs = [o[k] for o in ov if _finite(o.get(k))]
                if xs and not k.startswith("shap_share_") and not k.endswith("_pooled") and k not in MAX_KEYS:
                    mean[k] = float(np.mean(xs))
            mean["n_steps"] = int(round(np.mean([o["n_steps"] for o in ov])))
            mean["n_transl_jump"] = float(np.mean([o.get("n_transl_jump", 0) for o in ov]))
            finish(mean)
            sd = {k: float(np.std([o[k] for o in ov], ddof=1)) for k in ("ra_mpjpe_mm", "jit_pred_mm", "acc_err_mm")
                  if len(ov) > 1 and all(_finite(o.get(k)) for o in ov)}
        out[arm] = {"runs": names, "n_runs": len(names), "mean": mean, "sd_over_runs": sd,
                    "transl_runaway_runs": [n for n in names if runaway(runs[n]["overall"])],
                    "nonfinite_runs": bad}
    return out


# ------------------------------------------------------------------------------------ one run
_STATE = {}


def _state(mano_npz: str, root: str, manifest: str, threads: int, static_mm: float) -> dict:
    key = (mano_npz, root, manifest, threads, static_mm)
    if _STATE.get("key") != key:
        torch.set_num_threads(threads)
        dev = torch.device("cpu")
        mano = ManoLayer(mano_npz, add_mean=False).to(dev).eval()
        dirs = dict(sequences_for_split(Path(root), "val_core", Path(manifest)))
        _STATE.clear()
        _STATE.update(key=key, mano=mano, device=dev, root=Path(root), dirs=dirs, betas={}, static_mm=static_mm)
    return _STATE


def betas_for(st: dict, seq: str) -> np.ndarray:
    """MANO shape of a sequence: `semkine.eval_track.load_sequence(root, dir, seq)[2]["betas"]` (cached)."""
    if seq not in st["betas"]:
        st["betas"][seq] = np.asarray(ET.load_sequence(st["root"], st["dirs"].get(seq, "val"), seq)[2]["betas"],
                                      np.float32)
    return st["betas"][seq]


def analyse_npz(npz_path, st: dict) -> dict:
    """{sequence: jitter_decomp result + n_frames, n_segments, RA / root rotation / translation error + Shapley}.
    A sequence with non-finite values gets `nonfinite` (where) and no values."""
    per = {}
    with np.load(npz_path) as z:
        seqs = sorted({k.split("|")[1] for k in z.files if k.startswith("model|") and k.endswith("|pred")})
        if not seqs:
            raise KeyError(f"{npz_path} has no 'model|<seq>|pred' arrays")
        for s in seqs:
            r = {"pred": z[f"model|{s}|pred"], "gt": z[f"model|{s}|gt"], "run": z[f"model|{s}|run"],
                 "betas": betas_for(st, s)}
            rec = {src: z[f"model|{s}|{src}"] for _, src in RECORDED if f"model|{s}|{src}" in z.files}
            same = r["run"][1:] == r["run"][:-1]
            digest = hashlib.sha1(np.ascontiguousarray(r["pred"]).tobytes()
                                  + np.ascontiguousarray(r["run"]).tobytes()).hexdigest()[:12]
            bad = nonfinite_info({"pred": r["pred"], "gt": r["gt"], **rec})
            if bad is None:
                j = EX.jitter_decomp(st["mano"], r, st["device"], st["static_mm"])
                missing = undefined_keys(j, same)
                if missing:
                    bad = {"jitter_decomp": {"undefined_keys": missing}}
            if bad is not None:
                per[s] = {"n_frames": int(len(r["pred"])), "n_segments": int(len(np.unique(r["run"]))),
                          "n_steps": int(same.sum()), "nonfinite": bad, "pred_sha1": digest}
                continue
            j["n_frames"] = int(len(r["pred"]))
            j["n_segments"] = int(len(np.unique(r["run"])))
            step = np.linalg.norm(r["pred"][1:, :3].astype(np.float64) - r["pred"][:-1, :3], axis=-1)[same] * 1000
            j["transl_step_max_mm"] = float(step.max()) if step.size else None
            j["n_transl_jump"] = int((step > JUMP_MM).sum())
            for key, src in RECORDED:
                if src in rec:
                    j[key] = float(np.mean(rec[src], dtype=np.float64))
            j["pred_sha1"] = digest
            per[s] = add_shapley(j)
    return per


def _old_eval_ra(npz_path: Path):
    """RA recorded by the old evaluator for a historical run (`eval_step<N>/track_metrics_step50.json`), if any."""
    m = re.match(r"evalx_val_core_step(\d+)_", npz_path.name)
    if not m:
        return None
    p = npz_path.parent / f"eval_step{m.group(1)}" / "track_metrics_step50.json"
    return float(json.loads(p.read_text())["overall"]["mpjpe_ra_mm"]) if p.exists() else None


def analyse_job(job: dict) -> dict:
    """Worker: one npz -> {sequences, overall, npz, pred_sha1}."""
    st = _state(job["mano_npz"], job["root"], job["manifest"], job["threads"], job["static_mm"])
    t0 = time.time()
    per = analyse_npz(job["npz"], st)
    entry = {"npz": job["npz"], "sequences": per, "overall": combine(per), "seconds": round(time.time() - t0, 2),
             "pred_sha1": hashlib.sha1("|".join(f"{s}:{per[s]['pred_sha1']}" for s in sorted(per)).encode()
                                       ).hexdigest()[:12]}
    old = _old_eval_ra(Path(job["npz"]))
    if old is not None:
        entry["ra_old_evaluator_mm"] = old
        if _finite(entry["overall"].get("ra_mpjpe_mm")):
            entry["ra_diff_to_old_evaluator_mm"] = entry["overall"]["ra_mpjpe_mm"] - old
    return {"name": job["name"], "variant": job["variant"], "entry": entry}


# ------------------------------------------------------------------------------------ discovery
def resolve_runs(patterns, runs_root: Path) -> list:
    """Run directories named by `patterns`: names / globs under `runs_root` (outputs/semkine), else under its
    parent (outputs/), else as given (absolute or relative to the working directory)."""
    out, seen = [], set()
    for pat in patterns:
        bases = [None] if Path(pat).is_absolute() else [runs_root, runs_root.parent, None]
        hits = []
        for base in bases:
            hits = sorted(glob.glob(pat if base is None else str(base / pat)))
            if hits:
                break
        for h in hits:
            h = Path(h)
            if h.is_dir() and h.resolve() not in seen:
                seen.add(h.resolve())
                out.append(h)
    return out


def pick_npz(run_dir: Path, patterns) -> tuple:
    """(primary npz or None, other evalx_val_core_*.npz files of the directory)."""
    primary = None
    for pat in patterns:
        hit = sorted(run_dir.glob(pat))
        if hit:
            primary = hit[-1]
            break
    others = [p for p in sorted(run_dir.glob("evalx_val_core_*.npz")) if p != primary]
    return primary, others


def variant_tag(p: Path) -> str:
    return re.sub(r"^evalx_val_core_", "", p.stem)


# ------------------------------------------------------------------------------------ report
def _clean(o):
    """JSON-safe: NaN / inf -> null, numpy scalars -> python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def _f(v, nd=2):
    return "n/a" if not _finite(v) else f"{v:.{nd}f}"


def _pct(v):
    return "n/a" if not _finite(v) else f"{100 * v:.0f}%"


def _n(v):
    return "n/a" if v is None else str(v)


def _row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(header, rows):
    return "\n".join([_row(header), _row(["---"] * len(header))] + [_row(r) for r in rows])


def _head_cells(o):
    ratio = o.get("jit_ratio_pooled", o.get("jit_ratio"))      # per sequence the two are the same number
    return [_f(o.get("ra_mpjpe_mm")), _f(o.get("jit_gt_mm")), _f(o.get("jit_pred_mm")), _f(ratio, 3),
            _f(o.get("acc_gt_mm")), _f(o.get("acc_pred_mm")), _f(o.get("acc_err_mm"))]


def _share_cells(o):
    return [_pct(o.get(f"shap_share_{b}")) for b in BLOCKS]


def _nonfinite_runs(runs: dict) -> list:
    return sorted(n for n, r in runs.items() if r["overall"].get("nonfinite_sequences"))


def _nonfinite_variants(runs: dict) -> list:
    """`run:evaluation` of the other recorded evaluations (--variants) that hold non-finite values."""
    return sorted(f"{n}:{t}" for n, r in runs.items() for t, v in r.get("variants", {}).items()
                  if v["overall"].get("nonfinite_sequences"))


def _reference(runs: dict) -> dict:
    """Overall values of the first run with usable numbers (for facts that are the same for every run)."""
    return next((r["overall"] for r in runs.values() if _finite(r["overall"].get("jit_gt_mm"))), {})


def summary_lines(res: dict) -> list:
    """Statements generated from the numbers (so they cannot drift from them)."""
    runs, arms = res["runs"], res["arms"]
    ok = {n: r["overall"] for n, r in runs.items() if r["overall"].get("shap_share_transl") is not None}
    if not ok:
        return ["- no run has an attribution."]
    cnt = {b: sum(1 for o in ok.values() if dominant(o) == b) for b in BLOCKS}
    med = {b: float(np.median([o[f"shap_share_{b}"] for o in ok.values()])) for b in BLOCKS}
    lo = {b: min(o[f"shap_share_{b}"] for o in ok.values()) for b in BLOCKS}
    hi = {b: max(o[f"shap_share_{b}"] for o in ok.values()) for b in BLOCKS}
    L = [f"- {len(ok)} runs with an attribution. Largest Shapley block of `acc_err`: translation in {cnt['transl']} runs, "
         f"root rotation in {cnt['root']}, fingers in {cnt['fingers']}. Median share (min - max): "
         + "; ".join(f"{b} {100 * med[b]:.0f}% ({100 * lo[b]:.0f} - {100 * hi[b]:.0f}%)" for b in BLOCKS) + "."]
    ident = res["meta"].get("identical_evaluations", [])
    if ident:
        later = {n for g in ident for n in g[1:]}
        distinct = {n: o for n, o in ok.items() if n not in later}
        cnt2 = {b: sum(1 for o in distinct.values() if dominant(o) == b) for b in BLOCKS}
        L.append(f"- Runs with byte-identical per-step predictions (the same evaluation recorded under two names; counted separately "
                 f"above): {'; '.join(' = '.join(g) for g in ident)}. Counting each distinct evaluation once ({len(distinct)} runs): "
                 f"translation {cnt2['transl']}, root rotation {cnt2['root']}, fingers {cnt2['fingers']}.")
    gt = next(iter(ok.values()))
    L.append(f"- The ground truth's own motion on these two sequences: `jit_gt` {gt['jit_gt_mm']:.2f} mm per step, "
             f"`acc_gt` {gt['acc_gt_mm']:.2f} mm per step^2 (the same for every run).")
    ranked = sorted(((a, d["mean"]) for a, d in arms.items() if _finite(d["mean"].get("acc_err_mm"))
                     and not d["transl_runaway_runs"] and not d.get("nonfinite_runs")), key=lambda t: t[1]["acc_err_mm"])
    L.append("- Lowest `acc_err` (arm means, no translation runaway): "
             + ", ".join(f"{a} {m['acc_err_mm']:.2f} (RA {_f(m.get('ra_mpjpe_mm'))})" for a, m in ranked[:5]) + ".")
    ran = sorted(n for n, r in runs.items() if runaway(r["overall"]))
    if ran:
        L.append(f"- Translation runaway (>= {100 * RUNAWAY_FRAC:.0f}% of the steps move the predicted translation by more than "
                 f"{JUMP_MM:.0f} mm; the ground truth never moves more than 21.7 mm per step): {', '.join(ran)} (marked with a dagger). "
                 "Their `jit_pred` is the runaway's speed, not jitter; `acc_err` is not affected by a runaway at constant velocity.")
    bad = _nonfinite_runs(runs)
    if bad:
        L.append(f"- NON-FINITE predictions (NaN / inf in `pred`, `gt` or the recorded errors) in {len(bad)} run(s): "
                 f"{', '.join(bad)} (marked with a double dagger). The affected sequences have no jitter values, the runs have no "
                 "overall values and no attribution, and they are left out of the counts, medians, rankings and arm means.")
    bad_v = res["meta"].get("nonfinite_variants", [])
    if bad_v:
        L.append(f"- NON-FINITE predictions also in {len(bad_v)} other recorded evaluation(s) (section 7 of the md): "
                 f"{', '.join(bad_v)}; they are listed without values and not paired in the re-evaluation line.")
    return L


def noise_floor_line(res: dict) -> str:
    """How far a second evaluation of the same checkpoint (`*_gpu` files) moves the numbers (from the data)."""
    keys = ("acc_err_mm", "jit_pred_mm", "ra_mpjpe_mm")
    pairs = [(r["overall"], r["variants"]["last_tf_pert_gpu"]["overall"], r["pred_sha1"] == r["variants"]["last_tf_pert_gpu"].get("pred_sha1"))
             for r in res["runs"].values() if "last_tf_pert_gpu" in r.get("variants", {})]
    pairs = [p for p in pairs if all(_finite(o.get(k)) for o in p[:2] for k in keys)]
    if not pairs:
        return "no pair of evaluations of the same checkpoint in this run."
    d = lambda k: [abs(b[k] - a[k]) for a, b, _ in pairs if _finite(a.get(k)) and _finite(b.get(k))]      # noqa: E731
    same = sum(1 for a, b, _ in pairs if abs(b["acc_err_mm"] - a["acc_err_mm"]) < 5e-3 and abs(b["ra_mpjpe_mm"] - a["ra_mpjpe_mm"]) < 5e-3)
    exact = sum(1 for p in pairs if p[2])
    return (f"{len(pairs)} pairs, {same} of them identical to the printed precision ({exact} with bitwise identical "
            f"per-step predictions); over all pairs the largest differences are "
            f"{max(d('acc_err_mm')):.2f} mm in `acc_err`, {max(d('jit_pred_mm')):.2f} mm in `jit_pred`, "
            f"{max(d('ra_mpjpe_mm')):.2f} mm in RA, {100 * max([0.0] + [x for b in BLOCKS for x in d(f'shap_share_{b}')]):.1f} points in a share.")


def contrast_rows(arms: dict) -> list:
    """Rows of the contrast table for the arm pairs of CONTRASTS that are present (arm means, 'a -> b')."""
    rows = []
    for a, b, what in CONTRASTS:
        if a in arms and b in arms and arms[a]["mean"] and arms[b]["mean"]:
            ma, mb = arms[a]["mean"], arms[b]["mean"]
            two = lambda k, nd=2: f"{_f(ma.get(k), nd)} -> {_f(mb.get(k), nd)}"                    # noqa: E731
            rows.append([f"{a} -> {b}", what, two("acc_err_mm"), two("jit_pred_mm"), two("ra_mpjpe_mm"), two("shap_transl_mm"),
                         two("acc_err_only_transl_mm"), f"{_pct(ma.get('shap_share_transl'))} -> {_pct(mb.get('shap_share_transl'))}"])
    return rows


def standalone_line(arms: dict) -> str:
    """Why a Shapley value is not a property of the block alone, from the rt_cnn / rt_cnnf pair (same translation)."""
    a, b = arms.get("rt_cnn", {}).get("mean", {}), arms.get("rt_cnnf", {}).get("mean", {})
    if not all(_finite(m.get(k)) for m in (a, b) for k in ("acc_err_only_transl_mm", "shap_transl_mm")):
        return ""
    return (f"A Shapley value compares the blocks within one run; it moves when the other blocks' errors change even if the block "
            f"did not. In this data the `rt_cnnf` filter leaves the translation unchanged (gain 1.0; stand-alone translation "
            f"counterfactual `acc_err_only_transl` {b['acc_err_only_transl_mm']:.2f} against {a['acc_err_only_transl_mm']:.2f} for "
            f"`rt_cnn`, seed means) but `phi_transl` is {b['shap_transl_mm']:.2f} against {a['shap_transl_mm']:.2f} mm: a Shapley value "
            "averages the block's marginal contribution over the coalitions that contain the other blocks' errors, and the filter "
            "shrank those. Compare one block across arms with `acc_err_only_X` / `jit_only_X`, which are stand-alone.")


def render_doc(res: dict, headline) -> str:
    """A compact version of the report for docs/ (the full tables stay in the .md / .json)."""
    runs, arms, meta = res["runs"], res["arms"], res["meta"]
    names = sorted(runs)
    hist = [n for n in names if "ra_old_evaluator_mm" in runs[n]]
    rec = [n for n in names if n not in hist]
    tags = Counter(runs[n]["npz_tag"] for n in rec)
    src = ", ".join(f"{c} from `evalx_val_core_{t}.npz`" + ("" if t == "last_tf_pert" else f" (marked `({t})` in the tables)")
                    for t, c in sorted(tags.items(), key=lambda kv: (-kv[1], kv[0])))
    ref = _reference(runs)
    L = ["# DT: offline jitter attribution of the recorded closed-loop runs", ""]
    L += [f"Generated {meta['created']} by `tools/dt/jitter_offline.py` (tests: `tests/test_dt_jitter.py`) from the per-step arrays "
          f"of {len(rec)} recorded closed-loop evaluations ({src}; CPU only, no new closed loop) "
          f"plus {len(hist)} new `evalx.py eval --controls --tf --perturb --suffix dt` evaluations of the historical EventHands-Track "
          "(render + SO3 + FK + DomRand) checkpoints (step 3000, seeds 0 and 1). Every run, every sequence, the counterfactual "
          "mixes and the other recorded evaluations: `outputs/dt/reports/jitter_offline.md` / `.json`.", ""]
    L += ["## Quantities", "",
          "- `jit_*` first-difference joint jitter, mm per 50 ms step (mean over the 21 absolute joints, within segments; "
          "what `model/eval_track.py` calls jitter); `jit_gt` is the ground truth's own motion. `acc_*` the second difference, "
          "mm per step^2. `acc_err` is the acceleration of the error pred - gt: zero for perfect tracking however fast the hand "
          "moves and for any constant offset.",
          "- Shapley attribution of `acc_err` to translation (3), root axis-angle (3) and fingers (45): v(S) is `acc_err` of the "
          "ground truth with the blocks of S taken from the prediction; phi_X = 1/3 [v(X) - v()] + 1/6 [v(XY) - v(Y)] + "
          "1/6 [v(XZ) - v(Z)] + 1/3 [v(all) - v(YZ)]; the three phi add up to `acc_err` (asserted to 1e-6). The split is an "
          "accounting of a mean of norms of a nonlinear function, not a linear decomposition; a share can be negative.",
          "- `jit_only_X` (ground truth with block X from the prediction) is reported raw: first-difference jitter is not additive.", ""]
    L += ["## Result", ""] + summary_lines(res) + [""]
    L += ["## Headline runs (overall = frame-weighted over zgz_global and zgz_local)", ""]
    rows = []
    for n in names:
        if any(fnmatch.fnmatchcase(n, p) for p in headline):
            o = runs[n]["overall"]
            tag = runs[n]["npz_tag"]
            rows.append([n + ("" if tag == "last_tf_pert" else f" ({tag})") + (" ‡" if o.get("nonfinite_sequences") else "")]
                        + _head_cells(o) + [_f(o.get(f"shap_{b}_mm")) for b in BLOCKS] + _share_cells(o) + [dominance(o)])
    L += [_table(["run", "RA", "jit_gt", "jit_pred", "jit_pred/jit_gt", "acc_gt", "acc_pred", "acc_err", "phi_transl", "phi_root",
                  "phi_fingers", "share_transl", "share_root", "share_fingers", "dom"], rows), ""]
    L += ["## All arms (mean over seeds / replicates; sd = standard deviation over the runs of the arm)", ""]
    rows = []
    for a, d in arms.items():
        m, sd = d["mean"], d["sd_over_runs"]
        rows.append([a + (" †" if d["transl_runaway_runs"] else "") + (" ‡" if d.get("nonfinite_runs") else ""), d["n_runs"],
                     _f(m.get("ra_mpjpe_mm")), _f(m.get("jit_pred_mm")), _f(m.get("acc_err_mm")),
                     _f(sd.get("acc_err_mm")) if sd else "n/a"] + _share_cells(m) + [dominance(m) if m else "non-finite"])
    L += [_table(["arm", "n", "RA", "jit_pred", "acc_err", "sd acc_err", "share_transl", "share_root", "share_fingers", "dom"], rows), ""]
    crow = contrast_rows(arms)
    if crow:
        L += ["## Arm contrasts: what differs, and what the numbers can say", "",
              "Arm means (from -> to). The configs of the first two rows were diffed (and are checked by the tests): each changes "
              "one group of settings, so those rows are controlled contrasts. The filter row changes the output after the network "
              "and leaves the translation as it is. The last row changes many settings at once.", ""]
        L += [_table(["contrast", "what differs", "acc_err", "jit_pred", "RA", "phi_transl", "acc_err_only_transl", "share_transl"], crow), ""]
    if hist:
        L += ["## Historical arm: evalx against the old evaluator's recursive RA (step 3000)", ""]
        L += [_table(["run", "RA evalx (mm)", "RA old evaluator (mm)", "difference (mm)", "within 0.05 mm"],
                     [[n, _f(runs[n]["overall"].get("ra_mpjpe_mm"), 6), f"{runs[n]['ra_old_evaluator_mm']:.6f}",
                       f"{runs[n]['ra_diff_to_old_evaluator_mm']:+.2e}" if "ra_diff_to_old_evaluator_mm" in runs[n] else "n/a",
                       ("yes" if abs(runs[n]["ra_diff_to_old_evaluator_mm"]) < 0.05 else "NO")
                       if "ra_diff_to_old_evaluator_mm" in runs[n] else "non-finite"] for n in hist]), ""]
    L += ["## Caveats", "",
          "- Two sequences (`zgz_global` 1386 frames in 1 segment, `zgz_local` 1204 frames in 41 segments) are the development "
          "and the test set at once (AGENTS.md); jitter is measured on one subject. Seed spread of `acc_err` over the runs of an arm "
          "is in the table (sd).",
          f"- `jit_pred` contains the hand's own motion (`jit_gt` {_f(ref.get('jit_gt_mm'))}); compare arms by `acc_err`, which does not.",
          f"- Static-step statistics are meaningless here: the ground truth moves less than {meta['static_mm']:g} mm on only "
          f"{ref.get('n_static')} of {ref.get('n_steps')} steps; they stay in the JSON only."]
    sa = standalone_line(arms)
    if sa:
        L += ["- " + sa]
    L += ["- Re-evaluating the same checkpoint moves the numbers: " + noise_floor_line(res),
          f"- A translation runaway (the predicted translation moving by more than {JUMP_MM:.0f} mm in at least {100 * RUNAWAY_FRAC:.0f}% of the "
          "steps) makes `jit_pred` the runaway's speed; such runs are marked with a dagger.", ""]
    L += ["## Reproduce", "", "```",
          "cd EventHands1_dt   # tools/dt/jitter_offline.py, tests/test_dt_jitter.py",
          'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python tools/dt/jitter_offline.py --runs "*" hand_data51/track_render51_dr_so3fk \\',
          "    hand_data51/track_render51_dr_so3fk_rep2 --out-prefix outputs/dt/reports/jitter_offline --variants --jobs 4 --threads 2 \\",
          "    --doc outputs/dt/reports/jitter_offline_doc.md",
          "CUDA_VISIBLE_DEVICES=5 python tools/tracking/evalx.py eval --run-dir outputs/hand_data51/track_render51_dr_so3fk \\",
          "    --config <EventHands>/configs/eventhands_track_render51_dr_so3fk.yaml --manifest <data>/splits_semkine.json \\",
          "    --ckpt step=3000 --controls --tf --perturb --suffix dt     # and the _rep2 run",
          "python -m pytest tests/test_dt_jitter.py -q", "```"]
    return "\n".join(L) + "\n"


def render_md(res: dict, headline, detail) -> str:
    runs, arms, meta = res["runs"], res["arms"], res["meta"]
    names = sorted(runs)

    def label(n):
        tag = runs[n]["npz_tag"]
        o = runs[n]["overall"]
        return n + ("" if tag == "last_tf_pert" else f" ({tag})") + (" †" if runaway(o) else "") + (" ‡" if o.get("nonfinite_sequences") else "")

    L = ["# Offline jitter attribution of the recorded closed-loop runs", ""]
    L += [f"Generated by `tools/dt/jitter_offline.py` on {meta['created']} from the per-step arrays "
          f"(`model|<seq>|pred / gt / run`) of {meta['n_runs']} runs; `evalx.py` sha1 {meta['evalx_sha1']}. "
          f"CPU only, no new closed loop. Sequences: {', '.join(meta['sequences'])} (frame-weighted). Each run is read from "
          f"`evalx_val_core_last_tf_pert.npz` (the last checkpoint, closed loop); where a name is shown in parentheses the run's "
          f"only / preferred recorded evaluation is that one instead.", ""]
    L += ["Summary (generated from the tables below):", ""] + summary_lines(res) + [""]
    L += ["Columns. `RA` recursive root-aligned MPJPE (mm, mean of the recorded per-step values). `jit_*`: first-difference "
          "joint jitter, mm per 50 ms step (mean over 21 absolute joints, within segments; the quantity `model/eval_track.py` "
          "calls jitter), `gt` = the ground truth's own motion. `acc_*`: second difference, mm per step^2. `acc_err`: the "
          "acceleration of the error pred - gt (zero for perfect tracking and for any constant offset). `phi_X` / share: the "
          "Shapley attribution of `acc_err` to the blocks translation (3), root axis-angle (3) and fingers (45): "
          "v(S) = acc_err of the ground truth with the blocks of S taken from the prediction; shares add up to 100 % and "
          "are not linear (negative or >100 % means two blocks' errors partly cancel). `jit_only_X`: first-difference jitter "
          "of the ground truth with only block X taken from the prediction (raw, not additive); `acc_err_only_X`: the same for "
          "the error acceleration (the stand-alone effect of block X, unlike phi_X independent of the other blocks' errors). "
          "`dom` = block with the largest phi; `a~b` when the two largest shares are within 10 points.", ""]
    L += ["## 1. Per run (overall = frame-weighted over the sequences)", ""]
    rows = []
    for n in names:
        o = runs[n]["overall"]
        rows.append([label(n)] + _head_cells(o) + [_f(o.get(f"shap_{b}_mm")) for b in BLOCKS] + _share_cells(o)
                    + [dominance(o)])
    L += [_table(["run", "RA", "jit_gt", "jit_pred", "jit_pred/jit_gt", "acc_gt", "acc_pred", "acc_err", "phi_transl",
                  "phi_root", "phi_fingers", "share_transl", "share_root", "share_fingers", "dom"], rows), ""]
    L += ["## 2. Per arm (plain mean over the seeds / replicates; shares from the mean phi)", ""]
    rows = []
    for a, d in arms.items():
        m = d["mean"]
        sd = d["sd_over_runs"]
        rows.append([a + (" †" if d["transl_runaway_runs"] else "") + (" ‡" if d.get("nonfinite_runs") else ""), d["n_runs"]]
                    + _head_cells(m) + _share_cells(m)
                    + [dominance(m) if m else "non-finite",
                       f"{_f(sd.get('acc_err_mm'))} / {_f(sd.get('jit_pred_mm'))} / {_f(sd.get('ra_mpjpe_mm'))}" if sd else "n/a"])
    L += [_table(["arm", "n", "RA", "jit_gt", "jit_pred", "jit_pred/jit_gt", "acc_gt", "acc_pred", "acc_err",
                  "share_transl", "share_root", "share_fingers", "dom", "sd over runs: acc_err / jit_pred / RA"], rows), ""]
    L += ["## 3. Counterfactuals (stand-alone block effects), static steps, root angular acceleration, translation jumps", ""]
    rows = []
    for n in names:
        o = runs[n]["overall"]
        rows.append([label(n), _f(o.get("jit_gt_mm")), _f(o.get("jit_only_transl_mm")), _f(o.get("jit_only_root_mm")),
                     _f(o.get("jit_only_fingers_mm")), _f(o.get("jit_pred_mm")), _f(o.get("acc_err_only_transl_mm")),
                     _f(o.get("acc_err_only_root_mm")), _f(o.get("acc_err_only_fingers_mm")), _f(o.get("rot_acc_gt_deg")),
                     _f(o.get("rot_acc_pred_deg")), _f(o.get("jit_pred_static_mm")), _f(o.get("acc_err_static_mm")),
                     _n(o.get("n_steps")), _n(o.get("n_static")), _n(o.get("n_transl_jump")), _f(o.get("transl_step_max_mm"), 1)])
    L += [_table(["run", "jit_gt", "jit_only_transl", "jit_only_root", "jit_only_fingers", "jit_pred (all blocks)",
                  "acc_err_only_transl", "acc_err_only_root", "acc_err_only_fingers",
                  "rot_acc_gt (deg/step^2)", "rot_acc_pred (deg/step^2)", "jit_pred on static GT steps",
                  "acc_err on static GT steps", "n_steps", "n_static", f"steps with transl jump > {JUMP_MM:.0f} mm",
                  "largest predicted transl step (mm)"], rows), ""]
    L += ["`n_static` is the number of steps on which the ground truth moves less than 0.5 mm: 1 step in `zgz_global` and none "
          "in `zgz_local`, so the two static-step columns are single-step values (or n/a) and carry no information; the hand "
          "never rests in these recordings.", ""]
    if headline:
        L += ["## 4. Headline runs", ""]
        rows = []
        for n in names:
            if any(fnmatch.fnmatchcase(n, p) for p in headline):
                o = runs[n]["overall"]
                rows.append([label(n)] + _head_cells(o) + _share_cells(o) + [dominance(o)])
        L += [_table(["run", "RA", "jit_gt", "jit_pred", "jit_pred/jit_gt", "acc_gt", "acc_pred", "acc_err",
                      "share_transl", "share_root", "share_fingers", "dom"], rows), ""]
    L += ["## 5. Per sequence (detail runs)", ""]
    rows = []
    for n in names:
        if any(fnmatch.fnmatchcase(n, p) for p in detail):
            for s, o in runs[n]["sequences"].items():
                rows.append([label(n), s, o["n_frames"], o["n_segments"]] + _head_cells(o) + _share_cells(o)
                            + [dominance(o)])
    L += [_table(["run", "sequence", "frames", "segments", "RA", "jit_gt", "jit_pred", "jit_pred/jit_gt", "acc_gt",
                  "acc_pred", "acc_err", "share_transl", "share_root", "share_fingers", "dom"], rows), ""]
    hist = [n for n in names if "ra_old_evaluator_mm" in runs[n]]
    if hist:
        L += ["## 6. Historical arm: does evalx reproduce the old evaluator's recursive RA?", ""]
        rows = [[n, runs[n]["npz_tag"], _f(runs[n]["overall"].get("ra_mpjpe_mm"), 6), f"{runs[n]['ra_old_evaluator_mm']:.6f}",
                 f"{runs[n]['ra_diff_to_old_evaluator_mm']:+.6f}" if "ra_diff_to_old_evaluator_mm" in runs[n] else "n/a",
                 ("yes" if abs(runs[n]["ra_diff_to_old_evaluator_mm"]) < 0.05 else "NO")
                 if "ra_diff_to_old_evaluator_mm" in runs[n] else "non-finite"] for n in hist]
        L += [_table(["run", "npz", "RA evalx (mm)", "RA old evaluator (mm)", "difference", "within 0.05 mm"], rows), ""]
    var_rows = []
    for n in names:
        for t, v in sorted(runs[n].get("variants", {}).items()):
            o = v["overall"]
            var_rows.append([n, t] + _head_cells(o) + _share_cells(o) + [dominance(o)])
    if var_rows:
        L += ["## 7. Other recorded evaluations of the same runs (`--variants`)", "",
              "`last_tf_pert_gpu`: a second evaluation of the same last checkpoint (a GPU re-evaluation, "
              "docs/S38_ROOT_TRACKING_VERDICT.md); the closed loop is chaotic, so primary vs `_gpu` shows how much a re-run moves "
              "every number above: " + noise_floor_line(res) + " `selected`: the checkpoint chosen by the selection step. "
              "`*_gain1`: the same checkpoint with the filter gains set to 1 (`configs/s38/*_gain1.yaml`: no filtering).", ""]
        L += [_table(["run", "evaluation", "RA", "jit_gt", "jit_pred", "jit_pred/jit_gt", "acc_gt", "acc_pred", "acc_err",
                      "share_transl", "share_root", "share_fingers", "dom"], var_rows), ""]
    if meta["skipped"]:
        L += ["## 8. Run directories without a usable npz (skipped)", "", ", ".join(sorted(meta["skipped"])), ""]
    L += ["† translation runaway, see the summary."]
    if _nonfinite_runs(runs):
        L += ["", "‡ non-finite predictions (NaN / inf in `pred`, `gt` or the recorded errors): the affected sequence is not "
              "analysed, the run has no overall values and no attribution and is left out of the counts and arm means; "
              "an arm marked ‡ averages its finite runs only."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------ main
def run(args) -> dict:
    cfg = load_config(args.config)
    root = Path(cfg["DATA"]["ROOT"])
    manifest = Path(cfg["DATA"].get("SPLITS_MANIFEST") or root / "splits_semkine.json")
    runs_root = Path(args.runs_root)
    dirs = resolve_runs(args.runs, runs_root)
    jobs, skipped, plan = [], {}, {}
    for d in dirs:
        primary, others = pick_npz(d, args.npz)
        if primary is None:
            skipped[d.name] = "no npz"
            continue
        plan[d.name] = {"dir": str(d), "npz_tag": variant_tag(primary)}
        jobs.append({"name": d.name, "variant": None, "npz": str(primary)})
        if args.variants:
            jobs += [{"name": d.name, "variant": variant_tag(o), "npz": str(o)} for o in others]
    for j in jobs:
        j.update(mano_npz=cfg["MANO"]["NPZ"], root=str(root), manifest=str(manifest), threads=args.threads,
                 static_mm=args.static_mm)
    print(f"{len(plan)} runs with an npz ({len(jobs)} files), {len(skipped)} skipped; jobs={args.jobs} "
          f"threads={args.threads}", flush=True)
    if not plan:
        raise SystemExit(f"no run directory with an evaluation npz ({', '.join(args.npz)}) matched {args.runs}")
    t0 = time.time()
    if args.jobs > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.jobs, mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(analyse_job, jobs))
    else:
        results = [analyse_job(j) for j in jobs]
    runs = {}
    for r in results:
        if r["variant"] is None:
            runs[r["name"]] = {**plan[r["name"]], "arm": arm_of(r["name"]), **r["entry"]}
            runs[r["name"]]["transl_runaway"] = runaway(r["entry"]["overall"])
            runs[r["name"]]["nonfinite"] = bool(r["entry"]["overall"].get("nonfinite_sequences"))
    for r in results:
        if r["variant"] is not None:
            runs[r["name"]].setdefault("variants", {})[r["variant"]] = r["entry"]
    # the same evaluation recorded under two names (byte-identical per-step predictions)
    groups = {}
    for name in sorted(runs):
        groups.setdefault(runs[name]["pred_sha1"], []).append(name)
    identical = sorted(g for g in groups.values() if len(g) > 1)
    for g in identical:
        for name in g:
            runs[name]["same_arrays_as"] = [m for m in g if m != name]
    for name in sorted(runs):
        o = runs[name]["overall"]
        if runs[name]["nonfinite"]:
            where = "; ".join(f"{s}: {', '.join(runs[name]['sequences'][s]['nonfinite'])}" for s in o["nonfinite_sequences"])
            print(f"  {name:34s} NON-FINITE ({where}) -- no jitter values, run flagged", flush=True)
        else:
            print(f"  {name:34s} RA {_f(o.get('ra_mpjpe_mm'))}  jit gt/pred {_f(o.get('jit_gt_mm'))}/{_f(o.get('jit_pred_mm'))}  "
                  f"acc gt/pred/err {_f(o.get('acc_gt_mm'))}/{_f(o.get('acc_pred_mm'))}/{_f(o.get('acc_err_mm'))}  "
                  f"share T/R/F {'/'.join(_pct(o.get(f'shap_share_{b}')) for b in BLOCKS)}"
                  f"{'  [translation runaway]' if runs[name]['transl_runaway'] else ''}", flush=True)
        if args.verbose:
            for s, d in list(runs[name]["sequences"].items()) + [("overall", o)]:
                print(f"    [{s}] " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                                   for k, v in sorted(d.items())}), flush=True)
    bad_runs = _nonfinite_runs(runs)
    bad_vars = _nonfinite_variants(runs)
    if bad_runs:
        print(f"WARNING: non-finite predictions (NaN / inf) in {len(bad_runs)} run(s): {', '.join(bad_runs)} -- their sequences "
              "are not analysed and the runs have no overall values (marked with a double dagger in the report)",
              file=sys.stderr, flush=True)
    if bad_vars:
        print(f"WARNING: non-finite predictions (NaN / inf) in {len(bad_vars)} other recorded evaluation(s) (--variants): "
              f"{', '.join(bad_vars)} -- listed without values", file=sys.stderr, flush=True)
    seqs = sorted({s for r in runs.values() for s in r["sequences"]})
    meta = {"tool": "tools/dt/jitter_offline.py", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evalx": str(Path(EX.__file__).resolve()),
            "evalx_sha1": hashlib.sha1(Path(EX.__file__).read_bytes()).hexdigest()[:12],
            "args": {k: v for k, v in vars(args).items()},
            "config": str(Path(args.config)), "mano_npz": cfg["MANO"]["NPZ"], "manifest": str(manifest),
            "static_mm": args.static_mm, "npz_patterns": list(args.npz), "variants": bool(args.variants),
            "sequences": seqs, "n_runs": len(runs), "skipped": skipped, "blocks": list(BLOCKS),
            "nonfinite_runs": bad_runs, "nonfinite_variants": bad_vars, "identical_evaluations": identical,
            "units": {"jit_*": "mm per 50 ms step", "acc_*": "mm per step^2", "rot_acc_*": "deg per step^2",
                      "ra_mpjpe_mm / transl_err_mm / transl_step_max_mm": "mm", "root_rot_deg": "deg"},
            "overall": "mean over sequences weighted by n_frames (sequences lacking a key are left out of that key); "
                       "transl_step_max_mm is a maximum, n_* are sums; *_pooled are ratios of the overall means; a run with "
                       "a non-finite sequence has no overall values (nonfinite_sequences)",
            "shapley": "phi_X = 1/3 [v(X)-v()] + 1/6 [v(XY)-v(Y)] + 1/6 [v(XZ)-v(Z)] + 1/3 [v(all)-v(YZ)], "
                       "v(X)=acc_err_only_X, v(YZ)=acc_err_allbut_X, v(all)=acc_err; sum asserted to "
                       f"{SHAPLEY_TOL:g}; shap_share_X = phi_X / acc_err",
            "transl_runaway": f"n_transl_jump (predicted translation step > {JUMP_MM:g} mm) >= {RUNAWAY_FRAC:g} * n_steps",
            "seconds": round(time.time() - t0, 1)}
    return {"meta": meta, "runs": runs, "arms": arm_means(runs)}


def write_outputs(res: dict, prefix: Path, headline, detail, doc=None):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    jp, mp_ = Path(f"{prefix}.json"), Path(f"{prefix}.md")
    jp.write_text(json.dumps(_clean(res), indent=1, allow_nan=False))
    mp_.write_text(render_md(res, headline, detail))
    if doc:
        Path(doc).write_text(render_doc(res, headline))
    return jp, mp_


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run names or globs under --runs-root (also paths: absolute, or under outputs/ or the cwd)")
    ap.add_argument("--out-prefix", required=True, help="writes <prefix>.json and <prefix>.md")
    ap.add_argument("--runs-root", default=str(REPO / "outputs" / "semkine"))
    ap.add_argument("--config", default=str(REPO / DEFAULT_CONFIG), help="only for DATA.ROOT / manifest / MANO.NPZ")
    ap.add_argument("--npz", nargs="+", default=list(DEFAULT_NPZ),
                    help="file name patterns tried in order in each run directory (first match wins)")
    ap.add_argument("--variants", action="store_true", help="also analyse the other evalx_val_core_*.npz of each run")
    ap.add_argument("--headline", nargs="*", default=list(DEFAULT_DETAIL), help="run globs for the headline table")
    ap.add_argument("--detail", nargs="*", default=list(DEFAULT_DETAIL), help="run globs with per-sequence rows in the md")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (each with --threads torch threads)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--static-mm", type=float, default=EX.STATIC_MM)
    ap.add_argument("--doc", default=None, help="also write a compact markdown for docs/ to this path")
    ap.add_argument("--verbose", action="store_true", help="print every key of every sequence")
    args = ap.parse_args(argv)
    res = run(args)
    jp, mp_ = write_outputs(res, Path(args.out_prefix), args.headline, args.detail, args.doc)
    print("wrote", jp, "and", mp_, *(["and", args.doc] if args.doc else []), flush=True)
    return res


if __name__ == "__main__":
    main()
