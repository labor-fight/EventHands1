#!/usr/bin/env python3
"""S1 statistics: bucketed metrics, paired bootstrap, and the gate arithmetic.

The S0 audit registered the noise floor this project actually has: 0.4 mm between
same-config replicates, about 1.1 mm across retrainings, and a 2.43 mm observed replicate range.
A large part of the project's history consists of differences smaller than that being read as
effects. The functions here exist so a verdict has to be expressed as an interval.

Three comparison regimes, deliberately kept separate because they have different resolutions:

  `same_checkpoint`   one set of weights, inference changed (delta-trust, routing thresholds,
                      filter gains). Inference is bitwise deterministic on this machine, so the
                      only noise is the evaluation sample itself and a paired bootstrap over
                      sequences is the whole story.
  `retrained`         weights differ. Replicate scatter dominates; requires >= 2 replicates and
                      the comparison is between pooled means, never between best-of-N minima.
  `bucket`            a subset of steps. Support is small, so the CI is wide and the
                      `underpowered` flag from `buckets.py` bars a claim outright.

The bootstrap resamples *sequences*, not steps. Steps inside a sequence are strongly dependent
(a tracker that drifts stays drifted for many steps), so resampling steps would produce
intervals several times too narrow.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_N_BOOT = 10_000
DEFAULT_ALPHA = 0.05

#: Registered noise floor from the S0 audit.
NOISE_FLOOR_MM = {
    "same_config_replicate": 0.4,
    "cross_training": 1.1,
    "replicate_range": 2.43,
}


@dataclass
class Interval:
    mean: float
    lo: float
    hi: float
    n: int

    @property
    def excludes_zero(self) -> bool:
        return (self.lo > 0.0) or (self.hi < 0.0)

    @property
    def half_width(self) -> float:
        return 0.5 * (self.hi - self.lo)

    def __str__(self) -> str:
        return f"{self.mean:+.3f} [{self.lo:+.3f}, {self.hi:+.3f}] (n={self.n})"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["excludes_zero"] = self.excludes_zero
        d["half_width"] = self.half_width
        return d


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w = weights.sum()
    return float((values * weights).sum() / w) if w > 0 else float("nan")


def sequence_bootstrap(per_seq_values: Sequence[float], per_seq_weights: Sequence[float],
                       n_boot: int = DEFAULT_N_BOOT, alpha: float = DEFAULT_ALPHA,
                       seed: int = 0) -> Interval:
    """CI on a frame-weighted mean, resampling whole sequences with replacement."""
    v = np.asarray(per_seq_values, dtype=np.float64)
    w = np.asarray(per_seq_weights, dtype=np.float64)
    ok = np.isfinite(v) & (w > 0)
    v, w = v[ok], w[ok]
    if len(v) == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    point = _weighted_mean(v, w)
    if len(v) == 1:
        # One sequence carries no between-sequence information; say so instead of
        # reporting a zero-width interval.
        return Interval(point, float("-inf"), float("inf"), int(w.sum()))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    draws = (v[idx] * w[idx]).sum(1) / w[idx].sum(1)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return Interval(point, float(lo), float(hi), int(w.sum()))


def paired_sequence_bootstrap(a_values: Dict[str, float], b_values: Dict[str, float],
                              weights: Dict[str, float], n_boot: int = DEFAULT_N_BOOT,
                              alpha: float = DEFAULT_ALPHA, seed: int = 0) -> Interval:
    """CI on `mean(a) - mean(b)` over the sequences both arms were evaluated on.

    Paired because both arms see identical inputs: the between-sequence variance, which is by far
    the largest term, cancels. This is what makes a 0.5 mm same-checkpoint difference decidable
    when the unpaired cross-training resolution is 1.1 mm.
    """
    keys = sorted(set(a_values) & set(b_values) & set(weights))
    if not keys:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    d = np.array([a_values[k] - b_values[k] for k in keys], dtype=np.float64)
    w = np.array([weights[k] for k in keys], dtype=np.float64)
    ok = np.isfinite(d) & (w > 0)
    d, w = d[ok], w[ok]
    if len(d) == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    point = _weighted_mean(d, w)
    if len(d) == 1:
        return Interval(point, float("-inf"), float("inf"), int(w.sum()))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    draws = (d[idx] * w[idx]).sum(1) / w[idx].sum(1)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return Interval(point, float(lo), float(hi), int(w.sum()))


def pooled_replicate_stats(values: Sequence[float]) -> Dict[str, float]:
    """Mean, sem and range over replicates. Reported *instead of* the minimum.

    `track_render51`'s headline 19.26 mm is the minimum of four replicates whose distribution is
    centred at 20.6-20.7 mm. Comparing a new method's mean against another method's minimum is
    the single easiest way to manufacture an improvement, so the minimum is reported here only as
    a labelled diagnostic.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return {"n": 0}
    return {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "sem": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"),
        "std": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
        "min_diagnostic_only": float(v.min()),
        "max": float(v.max()),
        "range": float(v.max() - v.min()),
    }


def bucket_metrics(per_step: Dict[str, np.ndarray], masks: Dict[str, np.ndarray],
                   min_support: int = 200) -> Dict[str, dict]:
    """Mean of each per-step metric inside each bucket, with support and a power flag."""
    out: Dict[str, dict] = {}
    for name, m in masks.items():
        n = int(m.sum())
        entry: Dict[str, object] = {"n": n, "underpowered": n < min_support}
        for k, v in per_step.items():
            v = np.asarray(v)
            entry[k] = float(v[m].mean()) if n and len(v) == len(m) else None
        out[name] = entry
    return out


@dataclass
class GateResult:
    name: str
    regime: str
    interval: Optional[Interval]
    threshold: float
    direction: str          # "lower_is_better" | "higher_is_better" | "no_worse_than"
    verdict: str            # PASS | FAIL | UNDERPOWERED
    note: str = ""

    def as_dict(self) -> dict:
        d = {
            "name": self.name, "regime": self.regime, "threshold": self.threshold,
            "direction": self.direction, "verdict": self.verdict, "note": self.note,
        }
        d["interval"] = self.interval.as_dict() if self.interval else None
        return d


def decide_gate(name: str, interval: Interval, threshold: float, direction: str,
                regime: str = "same_checkpoint", min_support: int = 200) -> GateResult:
    """Turn an interval into a verdict, refusing to decide when the interval is uninformative.

    `interval` is always "candidate minus reference", so negative is an improvement for a
    lower-is-better metric.
    """
    note = ""
    if not np.isfinite(interval.mean):
        return GateResult(name, regime, interval, threshold, direction, "UNDERPOWERED",
                          "no finite estimate")
    if interval.n < min_support:
        note = f"support {interval.n} < {min_support}"
        return GateResult(name, regime, interval, threshold, direction, "UNDERPOWERED", note)
    if not np.isfinite(interval.lo) or not np.isfinite(interval.hi):
        return GateResult(name, regime, interval, threshold, direction, "UNDERPOWERED",
                          "single sequence: no between-sequence variance available")
    if regime == "retrained" and interval.half_width < NOISE_FLOOR_MM["same_config_replicate"]:
        note = (f"CI half-width {interval.half_width:.3f} mm is below the {NOISE_FLOOR_MM['same_config_replicate']} mm "
                "replicate floor; the interval understates retraining noise")

    if direction == "lower_is_better":
        ok = interval.hi < threshold
    elif direction == "higher_is_better":
        ok = interval.lo > threshold
    elif direction == "no_worse_than":
        # Non-inferiority: the whole interval must stay on the acceptable side.
        ok = interval.hi < threshold
    else:
        raise ValueError(direction)
    return GateResult(name, regime, interval, threshold, direction,
                      "PASS" if ok else "FAIL", note)


def format_gates(gates: Sequence[GateResult]) -> str:
    lines = [f"{'gate':38s} {'regime':16s} {'interval':34s} {'thr':>8s}  verdict"]
    lines.append("-" * 108)
    for g in gates:
        iv = str(g.interval) if g.interval else "n/a"
        lines.append(f"{g.name:38s} {g.regime:16s} {iv:34s} {g.threshold:8.3f}  {g.verdict}"
                     + (f"   [{g.note}]" if g.note else ""))
    return "\n".join(lines)


def per_sequence_table(results: Sequence[dict], key: str = "mpjpe_ra_mm"
                       ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """`(values, weights)` keyed by sequence, the input shape the bootstrap helpers want."""
    vals = {r["seq"]: float(r[key]) for r in results if r.get(key) is not None}
    wts = {r["seq"]: float(r["n_frames"]) for r in results if r.get(key) is not None}
    return vals, wts


def write_per_sequence_csv(path, entries: Sequence[dict], buckets: Sequence[str] = ()) -> None:
    """Long-format CSV: one row per (arm, sequence), plus one column per bucket metric."""
    import csv

    cols = ["arm", "checkpoint", "step_ms", "seq", "category", "n_frames", "n_runs",
            "mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm",
            "jitter_all_mm_per_step", "gt_move_mm_per_step"]
    cols += [f"bucket_{b}_mpjpe_ra_mm" for b in buckets]
    cols += [f"bucket_{b}_n" for b in buckets]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for e in entries:
            row = {k: e.get(k) for k in cols}
            for b in buckets:
                bm = (e.get("buckets") or {}).get(b) or {}
                row[f"bucket_{b}_mpjpe_ra_mm"] = bm.get("mpjpe_ra_mm")
                row[f"bucket_{b}_n"] = bm.get("n")
            w.writerow(row)
