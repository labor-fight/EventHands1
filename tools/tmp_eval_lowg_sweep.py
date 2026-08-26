"""Score the S24 retention sweep against the pre-registered gate.

The gate has two independent halves and both are checked here, because the sweep can succeed at the
mechanism and still fail at the outcome -- which is the single most likely way this ends, given that
every prior recipe that improved single-step error made tracking worse:

1. **Mechanism.** `G_rand` must fall monotonically in lambda. If it does not, the penalty is not
   controlling the quantity it is named after and nothing downstream is interpretable.
2. **Outcome.** Recursive RA-MPJPE must beat the `s1_track_domrand` baseline (16.64 mm, s3407) for at
   least one lambda. A monotone `G_rand` with flat or worse RA is a *negative* result about the
   causal claim, not a tuning problem, and is reported as such.

Reuses the frozen recursive protocol from `tools/tmp_probe_echo_gain.py` rather than reimplementing
it, so the numbers are directly comparable to the frontier table this sweep was designed from.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

#: the arm this sweep must beat, from the two-seed frontier table (docs/PLAN_SELECTION_VERDICT).
BASELINE = {"label": "track_domrand", "bias_mm": 14.94, "gain_rand": 0.236,
            "gain_self": 0.272, "recur_mm": 16.64}


def _run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=REPO, text=True, **kw)


def _selected(run_dir: Path, split: str, step_ms: int, force: bool):
    """Pick the checkpoint on the fixed save grid, by recursive RA on the eval split."""
    sel = run_dir / f"selection_{split}_step{step_ms}.json"
    if force or not sel.exists():
        r = _run([PY, "tools/select_checkpoint.py", "--run-dir", run_dir,
                  "--split", split, "--step-ms", step_ms], capture_output=True)
        (run_dir / "select.log").write_text((r.stdout or "") + (r.stderr or ""))
        if r.returncode != 0:
            print((r.stderr or "")[-2500:])
            return None, None
    if not sel.exists():
        return None, None
    chosen = json.loads(sel.read_text())["selected"]
    return chosen["ckpt"], chosen["mpjpe_ra_mm"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["0p03", "0p11", "0p32", "1p05"])
    ap.add_argument("--lams", nargs="+", type=float, default=[0.0316, 0.1053, 0.3159, 1.0530])
    ap.add_argument("--seed", default="s3407")
    ap.add_argument("--split", default="val_core")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--reselect", action="store_true")
    ap.add_argument("--out", default="outputs/semkine/lowg_sweep_s3407.json")
    a = ap.parse_args()

    rows, arms = [], []
    for tag, lam in zip(a.tags, a.lams):
        run_dir = REPO / f"outputs/semkine/s24_lowg_{tag}_{a.seed}"
        cfg = REPO / f"configs/semkine/s24_lowg_{tag}.yaml"
        if not run_dir.exists():
            print(f"skip {tag}: {run_dir} missing")
            continue
        ckpt, ra = _selected(run_dir, a.split, a.step_ms, a.reselect)
        if ckpt is None:
            print(f"skip {tag}: no checkpoint selected")
            continue
        rows.append({"tag": tag, "lam": lam, "ckpt": str(ckpt), "recur_mm": ra})
        arms += ["--arm", f"lowg_{tag}={ckpt}:{cfg}"]

    gains = {}
    if arms:
        out = REPO / "outputs/semkine/lowg_sweep_gain.json"
        r = _run([PY, "tools/tmp_probe_echo_gain.py", *arms, "--split", a.split,
                  "--step-ms", a.step_ms, "--out", out], capture_output=True)
        print((r.stdout or "")[-3000:])
        if out.exists():
            for label, row in json.loads(out.read_text())["arms"].items():
                p = row["pooled"]
                gains[label] = {"gain_rand": p["gain_rand_pooled"],
                                "gain_self": p["gain_self_pooled"]}

    for r in rows:
        r.update(gains.get(f"lowg_{r['tag']}", {}))

    print(f"\n{'arm':<14}{'lambda':>9}{'G_rand':>9}{'G_self':>9}{'recur_mm':>10}")
    print(f"{BASELINE['label']:<14}{0.0:>9.4g}{BASELINE['gain_rand']:>9.3f}"
          f"{BASELINE['gain_self']:>9.3f}{BASELINE['recur_mm']:>10.2f}   <- baseline")
    for r in sorted(rows, key=lambda x: x["lam"]):
        g, gs = r.get("gain_rand"), r.get("gain_self")
        ra = r.get("recur_mm")
        print(f"{'lowg_' + r['tag']:<14}{r['lam']:>9.4g}"
              f"{(f'{g:.3f}' if g is not None else '--'):>9}"
              f"{(f'{gs:.3f}' if gs is not None else '--'):>9}"
              f"{(f'{ra:.2f}' if ra is not None else '--'):>10}")

    ordered = [r for r in sorted(rows, key=lambda x: x["lam"]) if r.get("gain_rand") is not None]
    gseq = [r["gain_rand"] for r in ordered]
    mono = all(b <= x + 1e-9 for x, b in zip(gseq, gseq[1:])) and bool(gseq)
    beats = [r for r in rows if (r.get("recur_mm") or 1e9) < BASELINE["recur_mm"]]
    print(f"\ngate 1 mechanism  G_rand monotone in lambda : {'PASS' if mono else 'FAIL'}  {gseq}")
    print(f"gate 2 outcome    recursive RA < {BASELINE['recur_mm']:.2f} mm  : "
          f"{'PASS' if beats else 'FAIL'}"
          + (f"  best={min(beats, key=lambda r: r['recur_mm'])['tag']}" if beats else ""))

    # A pure absolute predictor has G = 0 by construction and measured 19.32 mm, worse than this
    # baseline's 16.64 at G ~= 0.22, so the optimum is interior and the curve should be U-shaped.
    # Which arm wins therefore says something specific, and the largest lambda overshooting into
    # absolute-prediction collapse is a predicted outcome rather than a failed run.
    have_ra = [r for r in sorted(rows, key=lambda x: x["lam"]) if r.get("recur_mm") is not None]
    interior = None
    if len(have_ra) >= 3:
        best = min(have_ra, key=lambda r: r["recur_mm"])
        interior = best is not have_ra[0] and best is not have_ra[-1]
        print(f"shape             best lambda is interior     : "
              f"{'yes' if interior else 'no'}  (best={best['tag']} at "
              f"{best['recur_mm']:.2f} mm, G={best.get('gain_rand', float('nan')):.3f})")
    if mono and not beats:
        print("\nInterpretation: the penalty controls retention but lowering retention did not\n"
              "improve tracking -- that falsifies the causal reading of the frontier correlation.")

    res = {"baseline": BASELINE, "rows": rows, "gate_monotone": mono,
           "gate_beats_baseline": bool(beats), "best_is_interior": interior}
    Path(REPO / a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(REPO / a.out).write_text(json.dumps(res, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
