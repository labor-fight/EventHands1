#!/usr/bin/env python3
"""DT2 package B: tracker-filter sweep on zgz with a two-fold interleaved-block cross-fit.

`semkine.anchored.AdaptiveFilter` (event-count-dependent root / finger / translation gains, optional alpha-beta
velocity) is evaluated closed loop on the two zgz sequences (the fixed protocol of `tools/dt/filter_eval.py`:
`evalx.run_sequence`, 50 ms steps, rng 0) for a grid of specs on the three dt_dz_l3 seeds. zgz is both the
development and the test set (AGENTS.md); the user allowed gain selection on it provided the selection is cross-fitted
and reported with in-fold and out-of-fold numbers and a statement of the optimism that remains.

SELECTION RULE (fixed before any sweep result was looked at)

  Folds    Every 50 ms evaluation step belongs to the 10 s block floor(end_ms / 10000) of its own sequence; even blocks
           are fold A, odd blocks fold B (both sequences, every segment). The two folds hold about half of the steps each.
  Fold RA  RA_F(spec) = mean over the fold's steps (both zgz sequences pooled, per-step root-aligned MPJPE, mm) of the
           closed-loop evaluation of the spec, averaged over the seeds the spec was run on. `acc_F(spec)` is the same for
           the root-aligned acceleration error (`evalx.jitter_decomp` `acc_err_ra`, a second difference of the error, mm per
           step^2; a second difference is assigned to the fold of its middle step). The bare tracker `raw` (no filter) is
           evaluated on the same seeds and gives `acc_F(raw)`.
  Admit    a spec is admissible on fold F iff acc_F(spec) <= acc_F(raw): the filter must not add jitter to the tracker.
  Select   on fold F: among the admissible specs the one with the smallest RA_F (ties: the smaller tag); it is REPORTED on the
           other fold F': the out-of-fold RA_F'(spec). In-fold RA_F is reported next to it (optimistic by construction).
           Selection on A is reported on B and selection on B on A; nothing from fold F' enters the choice on F.
  Recommend  the spec selected on both folds if it is the same, otherwise the one of the two selected specs with the
           smaller out-of-fold RA. The cross-validated estimate of the whole procedure is the step-weighted mean of the
           two out-of-fold RAs; it is reported against the same folds' RA of the pre-registered primary (0.5, 1, 0.5)
           (host FilteredTracker) and of `raw`.

  Stages   (seed 3407 first, each stage's candidates are fixed by the previous stage's fold-wise selections only)
           1a  constant root gain {0.35, 0.5, 0.65, 0.8, 1.0} x translation 0.5 x fingers 1.0 (5 specs) and event-count root
               gain nodes (300: lo, 3000: mid, 30000: hi), lo in {0.3, 0.5}, mid in {0.5, 0.7}, hi in {0.8, 1.0} (8 specs)
           1b  alpha-beta (beta_root = beta_trans = beta in {0.1, 0.2, 0.3}) on the constant 0.5 spec and on the best event-count
               spec of each fold (best = the fold's selection among the 8, 1 or 2 specs)
           1c  finger gain 0.8 (1.0 is already in the grid) on the best spec of each fold among all of 1a and 1b
           2   the top 5 admissible specs of fold A and the top 5 of fold B (ranked on their own fold only) plus the constant
               0.5 spec, rerun on seeds 3408 and 3409; every table's `select` uses only specs with all seeds present.

Optimism that remains (stated in every report): the two folds are interleaved blocks of the same two sequences, one
subject, one hand and the same checkpoints (the tracker is not refitted and not independent between folds); the grid
itself was designed after the open-loop gain grids and the per-sequence attributions seen on these very sequences
(global prefers a root gain ~0.75, local ~0.25); a seed average of three networks is not a new subject. Out-of-fold
numbers are therefore still optimistic as an estimate for a new subject.

    python tools/dt/filter_sweep_2fold.py run --stage 1a --runs dt_dz_l3_s3407 --gpus 4,7,5 --cpus 4,76:5,77:71,143
    python tools/dt/filter_sweep_2fold.py run --stage 1b --runs dt_dz_l3_s3407 ...      (reads the stage-1a results)
    python tools/dt/filter_sweep_2fold.py run --stage 2  --runs dt_dz_l3_s3408 dt_dz_l3_s3409 ...
    python tools/dt/filter_sweep_2fold.py report --runs dt_dz_l3_s3407 dt_dz_l3_s3408 dt_dz_l3_s3409 --out outputs/dt2/filter_sweep/report
    python tools/dt/filter_sweep_2fold.py eval --run-dir outputs/semkine/dt_dz_l3_s3407 --spec-file specs.json   (one worker)
    python tools/dt/filter_sweep_2fold.py stage-jobs --stage 1a --runs dt_dz_l3_s3407 --gpus 4,5,6,7 --window-ms 200 \\
        --clip-run-start --out-dir outputs/dt3/sweep > jobs.json                          (sched2 job JSON, nothing launched)

DT3 window options (`eval`, `run`, `stage-jobs`; all off by default): `--window-ms W` (evidence window of the closed loop, default
50 = the protocol), `--clip-run-start` (windows clipped to the segment's start), `--count-mode norm` (the filter's event count is
the window's count rescaled to a 50 ms rate), `--window-set 150,200,300 --min-events N`. They are the options of
`tools/dt/filter_eval.py` / `tools/tracking/evalx.py`. When any of them is on the evaluations are written with `no_wtag`: the
tags stay the spec hashes (`raw`, the primary's gain tag), so `stage_specs` / `load_table` / `report` work unchanged on a
separate `--out-dir` that holds the runs of one window setting (give each setting its own directory; the json's `window` entry
records the setting).

Per run and spec: `<out>/<run>/<tag>.json` (+ `.npz` of the per-step arrays; the json holds the summary of `evalx.evaluate`
and `post.folds`, the fold sums this script's report is made of).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "dt"), str(REPO / "tools" / "tracking")]
import filter_eval as FE                                         # noqa: E402
from semkine.anchored import AdaptiveFilter                       # noqa: E402

EX = FE.EX
FOLDS = ("A", "B")
PRIMARY_TAG = "r0.5_f1.0_t0.5"                                    # the pre-registered primary, host FilteredTracker
DEFAULT_OUT = REPO / "outputs" / "dt2" / "filter_sweep"
PY = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"       # the interpreter of `stage-jobs`' sched2 jobs
SEEDS = ("3407", "3408", "3409")
NODE_N = (300, 3000, 30000)
TRANS_GAIN = 0.5
TOP_K = 5
SUM_KEYS = ("n", "ra_sum", "n2", "acc_err_sum", "acc_pred_sum", "acc_gt_sum", "n1", "spd_pred_sum", "spd_gt_sum", "k1_n", "k1_sum")


# ------------------------------------------------------------------------------------------ the grid
def const_specs():
    return [{"a_root": g, "a_rest": 1.0, "a_trans": TRANS_GAIN} for g in (0.35, 0.5, 0.65, 0.8, 1.0)]


def event_spec(lo, mid, hi, rest=1.0, **kw):
    return {"a_root": [[n, g] for n, g in zip(NODE_N, (lo, mid, hi))], "a_rest": rest, "a_trans": TRANS_GAIN, **kw}


def event_specs():
    return [event_spec(lo, mid, hi) for lo, mid, hi in itertools.product((0.3, 0.5), (0.5, 0.7), (0.8, 1.0))]


def with_beta(spec, beta):
    """`spec` with the alpha-beta velocity on (both betas = `beta`; explicit keys, so a canonical spec is overridden too)."""
    return {**spec, "beta_root": beta, "beta_trans": beta}


def stage1a_specs():
    return const_specs() + event_specs()


def _cano(spec):
    return AdaptiveFilter.canonical_spec(spec)


def _tag(spec):
    return FE.spec_tag(spec)


# ------------------------------------------------------------------------------------ fold statistics
def fold_post(arrays, seqs, mano, device, root, cfg):
    """`post.folds` of one evaluation: per fold (A even 10 s blocks, B odd) and sequence class (global / local) the sums
    the report averages: step count and RA sum; second-difference count and the sums of the root-aligned acceleration
    of the error / prediction / ground truth (mm per step^2); first-difference count and the root angular speed sums
    (deg per step); the 10 deg perturbation trials' one-step divergence (retention k1) sum and count."""
    acc = {f: {c: dict.fromkeys(SUM_KEYS, 0.0) for c in ("global", "local")} for f in FOLDS}
    blocks = {f: set() for f in FOLDS}
    for si, (s, d) in enumerate(seqs):
        cls = "local" if "_local" in s else "global"
        pred, gt = arrays[f"model|{s}|pred"], arrays[f"model|{s}|gt"]
        end, run = arrays[f"model|{s}|end"], arrays[f"model|{s}|run"]
        ra = arrays[f"model|{s}|mpjpe_ra_mm"].astype(np.float64)
        betas = np.load(root / d / f"{s}_aux.npz", allow_pickle=True)["betas"]
        bt = torch.tensor(betas, dtype=torch.float32, device=device).view(1, -1)
        pj = EX.mano_fk(mano, pred.astype(np.float32), bt, device)[0]
        gj = EX.mano_fk(mano, gt.astype(np.float32), bt, device)[0]
        pr, gr = pj - pj[:, :1], gj - gj[:, :1]

        def second(j):
            return np.linalg.norm(j[2:] - 2 * j[1:-1] + j[:-2], axis=-1).mean(-1) * 1000

        fold = (end // EX.BLOCK_MS) % 2                                    # 0: A, 1: B
        for b in np.unique(end // EX.BLOCK_MS):
            blocks[FOLDS[int(b) % 2]].add((si, int(b)))
        same1 = run[1:] == run[:-1]
        same2 = same1[1:] & same1[:-1]
        e2, p2, g2 = second(pr - gr), second(pr), second(gr)
        ps = EX.rot_err_deg(pred[1:].astype(np.float32), pred[:-1].astype(np.float32))
        gs = EX.rot_err_deg(gt[1:].astype(np.float32), gt[:-1].astype(np.float32))
        k1 = i0 = None
        if f"perturb|{s}|10|div" in arrays:
            i0 = np.asarray(FE.trial_plan(run, arrays[f"model|{s}|elapsed"]), dtype=int)
            k1 = arrays[f"perturb|{s}|10|div"][:, 0].astype(np.float64) / 10.0
            assert len(i0) == len(k1), "perturbation trials do not match the plan"
        for fi, f in enumerate(FOLDS):
            a = acc[f][cls]
            m = fold == fi
            a["n"] += float(m.sum())
            a["ra_sum"] += float(ra[m].sum())
            m2 = same2 & (fold[1:-1] == fi)
            a["n2"] += float(m2.sum())
            a["acc_err_sum"] += float(e2[m2].sum())
            a["acc_pred_sum"] += float(p2[m2].sum())
            a["acc_gt_sum"] += float(g2[m2].sum())
            m1 = same1 & (fold[1:] == fi)
            a["n1"] += float(m1.sum())
            a["spd_pred_sum"] += float(ps[m1].sum())
            a["spd_gt_sum"] += float(gs[m1].sum())
            if k1 is not None:
                mk = fold[i0] == fi
                a["k1_n"] += float(mk.sum())
                a["k1_sum"] += float(k1[mk].sum())
    return {"folds": acc, "n_blocks": {f: len(blocks[f]) for f in FOLDS},
            "rule": "fold A = even floor(end_ms/10000), B = odd; second differences by their middle step"}


def pool(sums, classes=("global", "local")):
    out = dict.fromkeys(SUM_KEYS, 0.0)
    for c in classes:
        for k in SUM_KEYS:
            out[k] += sums[c][k]
    return out


def fold_metrics(sums):
    """Metrics of pooled fold sums: RA, acceleration error, acceleration ratio, root speed ratio, k1 (None when empty)."""
    div = lambda a, b: float(a / b) if b > 0 else None                     # noqa: E731
    return {"n": int(sums["n"]), "ra": div(sums["ra_sum"], sums["n"]), "acc_err": div(sums["acc_err_sum"], sums["n2"]),
            "acc_ratio": div(sums["acc_pred_sum"], sums["acc_gt_sum"]),
            "speed_ratio": div(sums["spd_pred_sum"], sums["spd_gt_sum"]), "k1": div(sums["k1_sum"], sums["k1_n"])}


# ----------------------------------------------------------------------------------------- evaluation
def window_args(a):
    """The window options of a sub-command's args (`--window-ms`, `--clip-run-start`, `--count-mode`, `--window-set`,
    `--min-events`; absent fields are the defaults) and `non_default`: any of them is on."""
    w = {"window_ms": int(getattr(a, "window_ms", EX.STEP)), "clip_run_start": bool(getattr(a, "clip_run_start", False)),
         "count_mode": getattr(a, "count_mode", "last50") or "last50",
         "window_set": EX.parse_window_set(getattr(a, "window_set", ()) or ()), "min_events": int(getattr(a, "min_events", 0))}
    EX.check_window_opts(argparse.Namespace(window_mode="fixed", **w))
    w["non_default"] = bool(w["window_ms"] != EX.STEP or w["clip_run_start"] or w["count_mode"] != "last50" or w["window_set"])
    out_dir = getattr(a, "out_dir", None)
    if w["non_default"] and out_dir and Path(out_dir).resolve() == DEFAULT_OUT.resolve():
        # the tags stay the spec hashes: in the recorded 50 ms sweep's directory every spec would count as done (and `raw` /
        # the primary would be skipped or overwritten)
        raise ValueError(f"a non-default window needs its own --out-dir, not the recorded sweep's {DEFAULT_OUT}")
    return w


def window_argv(a):
    """The `eval` command-line options that carry the non-default window options of `a` (empty by default: a default
    sweep's commands are exactly the historical ones)."""
    w = window_args(a)
    argv = []
    if w["window_ms"] != EX.STEP:
        argv += ["--window-ms", str(w["window_ms"])]
    if w["clip_run_start"]:
        argv += ["--clip-run-start"]
    if w["count_mode"] != "last50":
        argv += ["--count-mode", w["count_mode"]]
    if w["window_set"]:
        argv += ["--window-set", ",".join(str(x) for x in w["window_set"])]
    if w["min_events"]:
        argv += ["--min-events", str(w["min_events"])]
    return argv


def cmd_eval(a):
    specs = FE.load_filter_specs(a.spec_file) if a.spec_file else []
    run = Path(a.run_dir)
    out_dir = Path(a.out_dir) / run.name
    todo = [sp for sp in specs if not (out_dir / f"{_tag(sp)}.json").exists()]
    want_raw = a.raw and not (out_dir / "raw.json").exists()
    want_prim = a.primary and not (out_dir / f"{PRIMARY_TAG}.json").exists()
    print(f"{run.name}: {len(todo)} of {len(specs)} specs to run" + (" + raw" if want_raw else "")
          + (" + primary(host)" if want_prim else ""), flush=True)
    if not (todo or want_raw or want_prim):
        return
    wopt = window_args(a)
    ns = argparse.Namespace(run_dir=str(run), ckpt=a.ckpt, gains=("0.5,1,0.5" if want_prim else ""),
                            filter_spec=json.dumps(todo) if todo else "", raw=want_raw, controls=False, tf=False,
                            perturb=not a.no_perturb, out_dir=str(out_dir), split="val_core", manifest=None, config=None,
                            suffix="", device=a.device, threads=a.threads, no_npz=False,
                            window_ms=wopt["window_ms"], clip_run_start=wopt["clip_run_start"], count_mode=wopt["count_mode"],
                            window_set=wopt["window_set"], min_events=wopt["min_events"],
                            no_wtag=wopt["non_default"])             # a non-default window keeps the per-spec tags = the spec hashes
    FE.cmd_eval(ns, post=fold_post)


# ------------------------------------------------------------------------------------------ tables
def _spec_of(j):
    f = j.get("filter")
    if f is None:
        return "raw", None
    if f["class"].endswith("AdaptiveFilter"):
        return "spec", f["spec"]
    return "host", {"a_root": f["a_root"], "a_rest": f["a_rest"], "a_trans": f["a_trans"]}


def load_table(in_dir: Path, runs):
    """`{tag: {"kind", "spec", "runs": {run: stats}}}` of every evaluation with fold sums under `in_dir/<run>/`."""
    table = {}
    for run in runs:
        for jp in sorted((Path(in_dir) / run).glob("*.json")):
            j = json.loads(jp.read_text())
            if "post" not in j or "model" not in j:
                continue
            kind, spec = _spec_of(j)
            tag = jp.stem
            summ = FE.summarize(j)
            jit = j["model"].get("jitter", {})
            pert = (j.get("perturb") or {}).get("10")
            folds = {f: fold_metrics(pool(j["post"]["folds"][f])) for f in FOLDS}
            allf = fold_metrics({k: sum(pool(j["post"]["folds"][f])[k] for f in FOLDS) for k in SUM_KEYS})
            ra_check = abs(allf["ra"] - summ["ra_mm"]["overall"])
            stats = {"ra": summ["ra_mm"], "root_speed_ratio": summ["root_speed_ratio"], "acc_err": jit.get("acc_err_ra_mm"),
                     "acc_ratio": jit.get("acc_ratio_ra"), "k1": None if pert is None else pert["retention_k1_k2_k5_k10_k20"][0],
                     "root_rot_deg": summ["root_rot_deg"], "folds": folds, "fold_all": allf, "ra_check": ra_check,
                     "n_blocks": j["post"]["n_blocks"], "wall_s": j.get("wall_s")}
            table.setdefault(tag, {"kind": kind, "spec": spec, "runs": {}})["runs"][run] = stats
    return table


def seed_mean(row, runs, getter):
    vals = [getter(row["runs"][r]) for r in runs if r in row["runs"]]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if len(vals) == len(runs) else None


def _fold_ra(row, runs, f):
    return seed_mean(row, runs, lambda s: s["folds"][f]["ra"])


def _fold_acc(row, runs, f):
    return seed_mean(row, runs, lambda s: s["folds"][f]["acc_err"])


def admissible(table, tag, runs, f):
    raw = table.get("raw")
    row = table[tag]
    if raw is None or any(r not in row["runs"] or r not in raw["runs"] for r in runs):
        return False
    a, b = _fold_acc(row, runs, f), _fold_acc(raw, runs, f)
    return a is not None and b is not None and a <= b


def ranked(table, runs, f, kinds=("spec",), among=None):
    """Admissible specs (with every run of `runs` present) on fold `f`, best RA_f first (ties by tag)."""
    cand = [t for t, row in table.items() if row["kind"] in kinds and (among is None or t in among)
            and all(r in row["runs"] for r in runs) and admissible(table, t, runs, f)]
    return sorted(cand, key=lambda t: (_fold_ra(table[t], runs, f), t))


def select_two_fold(table, runs, among=None):
    """The selection rule of the module docstring. Returns the per-fold selections (in-fold / out-of-fold RA), the
    recommendation and the cross-validated estimate against the primary and the bare tracker."""
    sel = {}
    for f, g in (("A", "B"), ("B", "A")):
        rk = ranked(table, runs, f, among=among)
        if not rk:
            sel[f] = None
            continue
        t = rk[0]
        sel[f] = {"tag": t, "spec": table[t]["spec"], "selected_on": f, "reported_on": g,
                  "in_fold_ra": _fold_ra(table[t], runs, f), "out_of_fold_ra": _fold_ra(table[t], runs, g),
                  "in_fold_acc": _fold_acc(table[t], runs, f), "out_of_fold_acc": _fold_acc(table[t], runs, g),
                  "ranking": rk[:TOP_K]}
    out = {"selection": sel, "seeds": list(runs)}
    if sel["A"] is None or sel["B"] is None:
        out["recommended"] = None
        return out
    rec = sel["A"] if sel["A"]["tag"] == sel["B"]["tag"] else min((sel["A"], sel["B"]), key=lambda s: (s["out_of_fold_ra"], s["tag"]))
    out["recommended"] = {"tag": rec["tag"], "spec": rec["spec"], "rule": "selected on both folds" if sel["A"]["tag"] == sel["B"]["tag"]
                          else "smaller out-of-fold RA of the two selected specs"}
    n = {f: np.mean([table["raw"]["runs"][r]["folds"][f]["n"] for r in runs]) for f in FOLDS}
    cv = (n["B"] * sel["A"]["out_of_fold_ra"] + n["A"] * sel["B"]["out_of_fold_ra"]) / (n["A"] + n["B"])
    ref = {}
    for name in ("raw", PRIMARY_TAG):
        if name in table and all(r in table[name]["runs"] for r in runs):
            ra = {f: _fold_ra(table[name], runs, f) for f in FOLDS}
            # the reference on the folds the procedure was REPORTED on: A-selection on B, B-selection on A
            ref[name] = {"ra_A": ra["A"], "ra_B": ra["B"], "cv_matched": (n["B"] * ra["B"] + n["A"] * ra["A"]) / (n["A"] + n["B"])}
    out["cv"] = {"out_of_fold_ra": cv, "n_steps": {f: float(n[f]) for f in FOLDS}, "reference": ref,
                 "delta_vs": {k: cv - v["cv_matched"] for k, v in ref.items()}}
    return out


# ------------------------------------------------------------------------------------------- stages
def stage_specs(stage: str, in_dir: Path, seed0=("dt_dz_l3_s3407",)):
    """The specs of a stage; stages after 1a are defined by the fold-wise selections on the seed-3407 results so far."""
    if stage == "1a":
        return stage1a_specs()
    table = load_table(in_dir, list(seed0))
    ev_tags = {_tag(s) for s in event_specs()}
    if stage == "1b":
        base = [const_specs()[1]]                                              # constant 0.5
        for f in FOLDS:
            rk = ranked(table, list(seed0), f, among=ev_tags)
            if rk:
                base.append(table[rk[0]]["spec"])
        uniq = {}
        for b in base:
            uniq[_tag(b)] = b
        return [with_beta(_cano(b), beta) for b in uniq.values() for beta in (0.1, 0.2, 0.3)]
    if stage == "1c":
        best = {}
        for f in FOLDS:
            rk = ranked(table, list(seed0), f)
            if rk:
                best[rk[0]] = table[rk[0]]["spec"]
        return [{**_cano(sp), "a_rest": 0.8} for sp in best.values()]
    if stage == "2":
        chosen = {}
        for f in FOLDS:
            for t in ranked(table, list(seed0), f)[:TOP_K]:
                chosen[t] = table[t]["spec"]
        const05 = const_specs()[1]
        chosen.setdefault(_tag(const05), _cano(const05))
        return list(chosen.values())
    raise ValueError(f"unknown stage {stage!r}")


def cmd_stage_specs(a):
    specs = stage_specs(a.stage, Path(a.out_dir))
    print(json.dumps(specs, indent=1))


def run_parallel(jobs, gpus, cpus, log_dir: Path):
    """Run `jobs` (argv lists) on `len(gpus)` worker slots, each pinned to its GPU and logical-core set (`taskset`)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    pending, running, rc = list(enumerate(jobs)), {}, 0
    free = list(range(len(gpus)))
    while pending or running:
        while pending and free:
            slot = free.pop(0)
            i, argv = pending.pop(0)
            env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpus[slot]), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
            cmd = ["taskset", "-c", cpus[slot], "nice", "-n", "5", sys.executable, *argv]
            lf = open(log_dir / f"job{i}_{int(time.time())}.log", "w")
            running[slot] = (subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=str(REPO)), lf, i)
        time.sleep(2)
        for slot, (p, lf, i) in list(running.items()):
            if p.poll() is not None:
                lf.close()
                rc |= int(p.returncode != 0)
                print(f"job {i} finished rc={p.returncode} (gpu {gpus[slot]})", flush=True)
                del running[slot]
                free.append(slot)
    return rc


def stage_chunks(a, out: Path, n_chunks: int):
    """The worker processes of a stage: `(specs, [(run, chunk index, eval argv without the script path), ...])`. A run's specs
    are split round-robin into `n_chunks` chunks, one process (one model load) per non-empty chunk (chunk 0 also holds the
    `--refs` evaluations); each chunk's specs are written to `<out>/_specs/stage<stage>_<run>_c<k>.json`. The window options of
    `a` (none by default) are appended to every `eval` command."""
    wargv = window_argv(a)                                   # validates the window options before anything is written
    specs = stage_specs(a.stage, out)
    spec_dir = out / "_specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for run in a.runs:
        runp = f"outputs/semkine/{run}"
        chunks = [specs[i::n_chunks] for i in range(n_chunks)]
        for ci, ch in enumerate(chunks):
            if not ch and not (ci == 0 and a.refs):
                continue
            sf = spec_dir / f"stage{a.stage}_{run}_c{ci}.json"
            sf.write_text(json.dumps(ch))
            argv = ["eval", "--run-dir", runp, "--spec-file", str(sf), "--out-dir", str(out)]
            if ci == 0 and a.refs:
                argv += ["--raw", "--primary"]
            items.append((run, ci, argv + wargv))
    return specs, items


def cmd_run(a):
    out = Path(a.out_dir)
    gpus = [g.strip() for g in a.gpus.split(",")]
    cpus = [c.strip() for c in a.cpus.split(":")]
    assert len(gpus) == len(cpus) and all(int(g) in range(2, 8) for g in gpus), "GPUs 2-7 only, one cpu set per GPU"
    specs, items = stage_chunks(a, out, len(gpus))
    jobs = [[str(Path(__file__).resolve()), *argv] for _, _, argv in items]
    print(f"stage {a.stage}: {len(specs)} specs, {len(jobs)} jobs on gpus {gpus}", flush=True)
    sys.exit(run_parallel(jobs, gpus, cpus, out / "_logs"))


def stage_jobs(a):
    """sched2 job dicts (`tools/tracking/sched2.py submit`) of a stage's spec chunks, one per `eval` process: the same chunks and
    commands as `run`, but nothing is launched (the specs of each chunk are written to `<out-dir>/_specs` as `run` does). Ids
    `dt3_sw_<stage>_<run>_c<k>`; `--gpus` (only GPUs 4-7 are accepted) sets the number of chunks per run, sched2 places the jobs.
    The commands use the repo-relative script and run directory (`cwd` is this repo), the absolute spec file and `--out-dir`."""
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    if not gpus or any(not g.isdigit() or int(g) not in range(4, 8) for g in gpus) or len(set(gpus)) != len(gpus):
        raise SystemExit("--gpus: distinct GPUs from 4-7 only (e.g. 4,5,6,7)")
    _, items = stage_chunks(a, Path(a.out_dir).resolve(), len(gpus))
    return [{"id": f"dt3_sw_{a.stage}_{run}_c{ci}", "kind": "eval", "cwd": str(REPO), "prio": 75, "cores": 1, "mem_mib": 2500,
             "cmd": [PY, "tools/dt/filter_sweep_2fold.py", *argv]} for run, ci, argv in items]


def cmd_stage_jobs(a):
    print(json.dumps(stage_jobs(a), indent=1))


# ------------------------------------------------------------------------------------------- report
def _f(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def readable(tag, row):
    return tag.split("_", 1)[1] if row["kind"] == "spec" else ("bare tracker" if tag == "raw" else f"FilteredTracker {tag}")


def table_md(table, runs, order, title):
    L = [f"### {title}", "",
         "| spec (tag) | RA all | RA global | RA local | RA fold A | RA fold B | acc_err all | acc_err A / B | acc ratio | root speed ratio g / l | k1 10 deg | adm A | adm B |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in order:
        row = table[t]
        g = lambda fn: seed_mean(row, runs, fn)                                  # noqa: E731
        adm = lambda f: "-" if t == "raw" else ("yes" if admissible(table, t, runs, f) else "no")      # noqa: E731
        L.append(f"| {readable(t, row)} (`{t}`) | {_f(g(lambda s: s['ra']['overall']))} | {_f(g(lambda s: s['ra']['global']))} | "
                 f"{_f(g(lambda s: s['ra']['local']))} | {_f(_fold_ra(row, runs, 'A'))} | {_f(_fold_ra(row, runs, 'B'))} | "
                 f"{_f(g(lambda s: s['acc_err']), 2)} | {_f(_fold_acc(row, runs, 'A'), 2)} / {_f(_fold_acc(row, runs, 'B'), 2)} | "
                 f"{_f(g(lambda s: s['acc_ratio']), 2)} | "
                 f"{_f(g(lambda s: s['root_speed_ratio']['global']), 2)} / {_f(g(lambda s: s['root_speed_ratio']['local']), 2)} | "
                 f"{_f(g(lambda s: s['k1']), 2)} | {adm('A')} | {adm('B')} |")
    return L + [""]


def bucket_table(in_dir: Path, runs, tags):
    """`{tag: {bucket: {"n": mean steps per seed, "ra": mean RA over seeds}}}` from the evaluation jsons' `by_events`."""
    out = {}
    for tag in tags:
        acc = {}
        for run in runs:
            j = json.loads((Path(in_dir) / run / f"{tag}.json").read_text())
            for b, v in FE.pooled_by_events(j).items():
                acc.setdefault(b, []).append((v["n"], v["mpjpe_ra_mm"]))
        out[tag] = {b: {"n": float(np.mean([x[0] for x in v])), "ra": float(np.mean([x[1] for x in v]))} for b, v in acc.items()
                    if len(v) == len(runs)}
    return out


def cmd_report(a):
    in_dir = Path(a.in_dir)
    runs = [Path(r).name for r in a.runs]
    table = load_table(in_dir, runs)
    full = [t for t, row in table.items() if all(r in row["runs"] for r in runs)]
    stage1 = [runs[0]]
    res = {"runs": runs, "n_specs_total": len(table), "n_specs_all_seeds": len(full)}
    L = ["# DT2 package B: AdaptiveFilter two-fold sweep on zgz", "",
         f"Runs: {', '.join(runs)}. Folds: even / odd 10 s blocks (A / B) of each zgz sequence. Numbers are mm (RA = root-aligned MPJPE), "
         "means over the seeds listed; `adm` = acc_err not above the bare tracker's on that fold (the selection constraint). `acc_err all` / `acc ratio` "
         "are evalx's whole-sequence values (root-aligned acceleration of the error in mm per step^2, and prediction / ground-truth acceleration), frame-weighted over the "
         "two sequences; the fold acc_err values pool the steps of both sequences, so the two folds do not average to `acc_err all` exactly. "
         "The columns `RA fold A` / `RA fold B` are the out-of-fold RA of a selection made on B / on A (and the in-fold RA of one made on A / B); "
         "the selections themselves are listed under each table.", ""]
    # stage-1 view (first seed only): every spec that was run on it
    t1 = load_table(in_dir, stage1)
    sel1 = select_two_fold(t1, stage1) if "raw" in t1 else None
    order1 = sorted(t1, key=lambda t: (t1[t]["kind"] != "raw", t1[t]["kind"] != "host", t1[t]["runs"][stage1[0]]["ra"]["overall"]))
    L += table_md(t1, stage1, order1, f"Stage 1 grid, seed {stage1[0].split('_s')[-1]} only ({len(t1)} evaluations), sorted by overall RA")
    res["stage1"] = {"seed_run": stage1[0], "selection": sel1}
    if sel1:
        L += ["Stage-1 selection (this seed only; each fold's choice uses that fold only):", ""]
        for f in FOLDS:
            s = sel1["selection"][f]
            if s:
                L.append(f"- fold {f}: `{s['tag']}`  in-fold RA {_f(s['in_fold_ra'])}, out-of-fold RA {_f(s['out_of_fold_ra'])}; "
                         f"top {TOP_K} on this fold: {', '.join('`%s`' % x for x in s['ranking'])}")
        L.append("")
    if len(runs) > 1 and "raw" in table:
        order = sorted(full, key=lambda t: (table[t]["kind"] != "raw", table[t]["kind"] != "host", table[t]["runs"][runs[0]]["ra"]["overall"]))
        L += table_md(table, runs, order, f"Specs run on all {len(runs)} seeds (seed means), sorted by seed-1 overall RA")
        among = [t for t in full if table[t]["kind"] == "spec"]
        fin = select_two_fold(table, runs, among)
        res["final"] = fin
        L += ["## Two-fold selection over the specs run on all seeds", ""]
        for f in FOLDS:
            s = fin["selection"][f]
            if s:
                raw_in, raw_out = _fold_ra(table["raw"], runs, f), _fold_ra(table["raw"], runs, s["reported_on"])
                L.append(f"- selected on fold {f}: `{s['tag']}` spec {json.dumps(s['spec'])}: in-fold RA {_f(s['in_fold_ra'])} "
                         f"({s['in_fold_ra'] - raw_in:+.3f} vs the bare tracker on that fold), out-of-fold RA ({s['reported_on']}) {_f(s['out_of_fold_ra'])} "
                         f"({s['out_of_fold_ra'] - raw_out:+.3f} vs bare); acc_err in-fold {_f(s['in_fold_acc'], 2)}, out-of-fold {_f(s['out_of_fold_acc'], 2)}")
        if fin.get("recommended"):
            L += ["", f"- recommended: `{fin['recommended']['tag']}` ({fin['recommended']['rule']}); spec {json.dumps(fin['recommended']['spec'])}"]
            cv = fin["cv"]
            L.append(f"- cross-validated estimate of the procedure (step-weighted mean of the two out-of-fold RAs): {_f(cv['out_of_fold_ra'])} mm; "
                     + "; ".join(f"{k}: same-folds RA {_f(v['cv_matched'])} (procedure {cv['delta_vs'][k]:+.3f})" for k, v in cv["reference"].items()))
            L += ["", "Per seed, recommended spec vs the bare tracker and the primary (overall RA):", "",
                  "| seed run | bare | primary (host) | recommended | global (bare / primary / rec.) | local (bare / primary / rec.) |", "|---|---|---|---|---|---|"]
            for r in runs:
                rw, pr, rc = (table[x]["runs"][r]["ra"] for x in ("raw", PRIMARY_TAG, fin["recommended"]["tag"]))
                L.append(f"| {r} | {_f(rw['overall'], 4)} | {_f(pr['overall'], 4)} | {_f(rc['overall'], 4)} | "
                         f"{_f(rw['global'], 2)} / {_f(pr['global'], 2)} / {_f(rc['global'], 2)} | {_f(rw['local'], 2)} / {_f(pr['local'], 2)} / {_f(rc['local'], 2)} |")
            res["recommended_per_seed"] = {r: table[fin["recommended"]["tag"]]["runs"][r]["ra"] for r in runs}
            tags = ("raw", PRIMARY_TAG, fin["recommended"]["tag"])
            bk = bucket_table(in_dir, runs, tags)
            res["by_events"] = bk
            L += ["", "RA by raw events per 50 ms packet (pooled over both sequences, seed means; a sequence contributes a bucket with >= 20 steps):", "",
                  "| events / 50 ms | steps per seed | bare | primary (host) | recommended | recommended - bare |", "|---|---|---|---|---|---|"]
            for b in sorted(bk["raw"], key=lambda k: int(k[1:].split(",")[0])):
                v = [bk[t].get(b) for t in tags]
                if all(x is not None for x in v):
                    L.append(f"| {b} | {v[0]['n']:.0f} | {_f(v[0]['ra'])} | {_f(v[1]['ra'])} | {_f(v[2]['ra'])} | {v[2]['ra'] - v[0]['ra']:+.3f} |")
    L += ["", "## Reading rules and the optimism that remains", "",
          "- In-fold numbers are optimistic by construction (the spec was picked for being best on exactly those steps). Out-of-fold numbers "
          "are the honest ones for this protocol but remain optimistic as an estimate for a new subject: the folds are interleaved blocks of the "
          "same two sequences (one subject, one hand, the same trained checkpoints, adjacent in time), the grid was designed after the open-loop gain "
          "grids and per-sequence attributions computed on these sequences, and a 3-seed mean is three networks, not three subjects.",
          "- One filter spec is applied to both sequences; the open-loop preference differs by sequence (global ~0.75, local ~0.25), so the "
          "per-sequence columns (RA global / local) show what a single setting trades.",
          "- A closed-loop evaluation is chaotic at the 1e-6 level (a re-run of the same spec on another code path can move RA by a few hundredths of a "
          "mm); differences of that size between neighbouring specs are not ranking evidence.", ""]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res["table"] = {t: {"kind": r["kind"], "spec": r["spec"], "runs": r["runs"]} for t, r in table.items()}
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("wrote", out.with_suffix(".json"), out.with_suffix(".md"))


def add_window_args(p):
    """The DT3 window options of `eval` / `run` / `stage-jobs` (all off by default; see the module docstring)."""
    p.add_argument("--window-ms", type=int, default=EX.STEP, help="evidence window of the closed loop in ms (default 50 = the protocol)")
    p.add_argument("--clip-run-start", action="store_true", help="clip every evidence window to its segment's start (training's rule)")
    p.add_argument("--count-mode", choices=EX.COUNT_MODES, default="last50",
                   help="event count handed to the filter: last50 = the 50 ms packet's, norm = the window's count at a 50 ms rate")
    p.add_argument("--window-set", type=EX.parse_window_set, default=(),
                   help="comma list of windows in ms, e.g. 150,200,300 (with --min-events): the smallest holding enough events")
    p.add_argument("--min-events", type=int, default=0, help="--window-set: the raw event count that decides the window")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="one worker: evaluate specs (and optionally raw / primary) on one run")
    e.add_argument("--run-dir", required=True)
    e.add_argument("--spec-file", default="", help="JSON list of specs (or a JSON string)")
    e.add_argument("--out-dir", default=str(DEFAULT_OUT))
    e.add_argument("--raw", action="store_true", help="also the bare tracker (tag raw)")
    e.add_argument("--primary", action="store_true", help="also the pre-registered primary (0.5, 1, 0.5) as the host FilteredTracker")
    e.add_argument("--ckpt", default="last")
    e.add_argument("--device", default=None)
    e.add_argument("--threads", type=int, default=0)
    e.add_argument("--no-perturb", action="store_true")
    add_window_args(e)
    s = sub.add_parser("stage-specs", help="print a stage's specs as JSON")
    s.add_argument("--stage", required=True, choices=("1a", "1b", "1c", "2"))
    s.add_argument("--out-dir", default=str(DEFAULT_OUT))
    r = sub.add_parser("run", help="evaluate a stage's specs on the given runs, one worker per GPU / cpu set")
    r.add_argument("--stage", required=True, choices=("1a", "1b", "1c", "2"))
    r.add_argument("--runs", nargs="+", required=True)
    r.add_argument("--out-dir", default=str(DEFAULT_OUT))
    r.add_argument("--gpus", default="4,7,5", help="comma-separated physical GPU ids (2-7 only)")
    r.add_argument("--cpus", default="4,76:5,77:71,143", help="colon-separated logical-core sets, one per GPU")
    r.add_argument("--refs", action="store_true", help="also the bare tracker and the host primary on each run")
    add_window_args(r)
    sj = sub.add_parser("stage-jobs", help="print a stage's spec chunks as sched2 job JSON (nothing is launched)")
    sj.add_argument("--stage", required=True, choices=("1a", "1b", "1c", "2"))
    sj.add_argument("--runs", nargs="+", required=True)
    sj.add_argument("--out-dir", required=True, help="where the evaluations will be written; the chunks' spec files are written to "
                    "<out-dir>/_specs (required: the recorded 50 ms sweep's directory must not be the default)")
    sj.add_argument("--gpus", default="4,5,6,7", help="comma-separated physical GPU ids (4-7 only); their number is the number of chunks per run")
    sj.add_argument("--refs", action="store_true", help="also the bare tracker and the host primary on each run")
    add_window_args(sj)
    rp = sub.add_parser("report", help="fold tables, the two-fold selection and the markdown report")
    rp.add_argument("--runs", nargs="+", required=True)
    rp.add_argument("--in-dir", default=str(DEFAULT_OUT))
    rp.add_argument("--out", default=str(DEFAULT_OUT / "report"))
    a = ap.parse_args(argv)
    if a.cmd in ("eval", "run", "stage-jobs"):
        try:
            window_args(a)
        except ValueError as err:
            ap.error(str(err))
    {"eval": cmd_eval, "stage-specs": cmd_stage_specs, "run": cmd_run, "stage-jobs": cmd_stage_jobs, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
