#!/usr/bin/env python3
r"""S9: does analytic observability routing beat the cheap alternatives?

RDOR decides which degrees of freedom a packet of events is allowed to update, from the measurement
geometry alone (`semkine/router.py`). This script runs it inside the recursive evaluator against
every control a reviewer would reach for first:

* `baseline` -- nothing skipped. A low bar, kept only as the reference point.
* `delta_trust` -- constant shrinkage of the update. The pre-registered control: it is the
  zero-parameter special case of "weak information means a small step", and S7 found it within
  0.2 mm of the ground-truth oracle at a 50 ms step. RDOR must beat *this*, not the baseline.
* `event_count` -- update everything when the packet is large, nothing when it is small. Tests
  whether the routing needs geometry at all, or whether packet size already says it.
* `no_schur` -- the same router with the root left in. Isolates the one step the method claims is
  essential: without it a finger is credited with evidence the unresolved wrist could explain.
* `random` -- skips at RDOR's own measured rate, blind to which groups. Separates the value of the
  *choice* from the value of skipping per se.

All arms share one checkpoint and one event stream, so the comparison is paired per sequence and
the 0.4 mm training noise floor does not apply; a paired sequence bootstrap is the test.

Verdict rule, registered in advance: PASS requires RDOR to beat delta-trust with a paired
confidence interval excluding zero. Beating the baseline alone is not a result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from config import load_config                          # noqa: E402
from mano_layer import ManoLayer                         # noqa: E402
from model import MNISTModel                            # noqa: E402

from semkine import eval_track as ET                    # noqa: E402
from semkine import metrics as MT                       # noqa: E402
from semkine import oracle as OR                        # noqa: E402
from semkine import router as RD                        # noqa: E402
from semkine.dataset import sequences_for_split         # noqa: E402

KEY = "mpjpe_ra_mm"


def evaluate(model, mano, cfg, root, seqs, step_ms, device, seed, policy=None,
             delta_trust=1.0):
    rng = np.random.default_rng(seed)
    rows = []
    if policy is not None and hasattr(policy, "reset"):
        policy.reset()
    for seq, legacy_dir in seqs:
        r = ET.track_sequence(model, mano, cfg, root, legacy_dir, seq, step_ms, device,
                              rng, 1.0, delta_trust, None, active_policy=policy)
        if r:
            rows.append(r)
    n = sum(r["n_frames"] for r in rows)
    out = {
        "overall": {k: float(sum(r[k] * r["n_frames"] for r in rows) / max(n, 1))
                    for k in ("mpjpe_ra_mm", "mpjpe_abs_mm", "mpvpe_ra_mm")},
        "n_frames": n,
        "per_seq": {r["seq"]: r[KEY] for r in rows},
        "weights": {r["seq"]: r["n_frames"] for r in rows},
        "jitter": float(np.mean([r["jitter_all_mm_per_step"] for r in rows
                                 if r["jitter_all_mm_per_step"] is not None])),
    }
    if policy is not None and hasattr(policy, "stats"):
        out["policy"] = policy.stats()
    elif policy is not None and hasattr(policy, "rate"):
        out["policy"] = {"update_rate": policy.rate()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="defaults to the config recorded in the run directory")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", default="4,8,12")
    ap.add_argument("--gain-on", type=float, default=1.0)
    ap.add_argument("--gain-off", type=float, default=0.25)
    ap.add_argument("--min-dwell", type=int, default=2)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="outputs/semkine/s9_rdor")
    a = ap.parse_args()

    cfg_path = a.config
    if cfg_path is None:
        meta = Path(a.ckpt).parent / "training_metadata.json"
        cfg_path = json.loads(meta.read_text())["config_path"]
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel.load_from_checkpoint(a.ckpt, cfg=cfg, map_location=device).to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    seqs = sequences_for_split(root, a.split, None)
    if a.only:
        pats = a.only.split(",")
        seqs = [s for s in seqs if any(p in s[0] for p in pats)]
    assert seqs, "no sequences selected"
    print(f"{len(seqs)} sequences in split={a.split}, config={cfg_path}")

    def rdor(**kw):
        return RD.RDORPolicy(mano=mano, gain_on=a.gain_on, gain_off=a.gain_off,
                             min_dwell=a.min_dwell, **kw)

    arms = {}
    base = evaluate(model, mano, cfg, root, seqs, a.step_ms, device, a.seed)
    arms["baseline"] = base
    print(f"baseline              RA={base['overall'][KEY]:8.4f}  jitter={base['jitter']:.3f}")

    for dt in (0.5, 0.7):
        r = evaluate(model, mano, cfg, root, seqs, a.step_ms, device, a.seed, delta_trust=dt)
        arms[f"delta_trust_{dt:g}"] = r
        print(f"delta_trust {dt:<4g}      RA={r['overall'][KEY]:8.4f}  jitter={r['jitter']:.3f}")

    for budget in [int(x) for x in a.budgets.split(",")]:
        p = rdor(budget=budget)
        r = evaluate(model, mano, cfg, root, seqs, a.step_ms, device, a.seed, policy=p)
        arms[f"rdor_b{budget}"] = r
        print(f"rdor budget={budget:<4d}     RA={r['overall'][KEY]:8.4f}  "
              f"rate={p.rate():.3f}  jitter={r['jitter']:.3f}")

    best = min((k for k in arms if k.startswith("rdor_")),
               key=lambda k: arms[k]["overall"][KEY])
    b_best = int(best.split("_b")[1])

    # Ablations and same-rate control, all at the winning budget so the comparison isolates one
    # thing at a time rather than confounding the ablation with a different update rate.
    p = rdor(budget=b_best, eliminate_root=False)
    arms["no_schur"] = evaluate(model, mano, cfg, root, seqs, a.step_ms, device, a.seed, policy=p)
    print(f"no_schur              RA={arms['no_schur']['overall'][KEY]:8.4f}  rate={p.rate():.3f}")

    thr = arms[best]["policy"]["update_rate"]
    q = OR.ActivePolicy(mode="random", thresh=thr, seed=a.seed + 1)
    arms["random_same_rate"] = evaluate(model, mano, cfg, root, seqs, a.step_ms, device,
                                        a.seed, policy=q)
    print(f"random @rate={thr:.3f}    RA={arms['random_same_rate']['overall'][KEY]:8.4f}")

    # The cheap heuristic: gate the whole update on the packet's size. Swept rather than tuned to
    # one point, and the arm nearest RDOR's own update rate is the one carried into the comparison,
    # so the two differ in what they route on and not in how much they skip.
    for et in (200, 1000, 4000):
        p = OR.ActivePolicy(mode="events", event_thresh=et, seed=a.seed)
        arms[f"event_count_{et}"] = evaluate(model, mano, cfg, root, seqs, a.step_ms, device,
                                             a.seed, policy=p)
        arms[f"event_count_{et}"]["policy"] = {"update_rate": p.rate()}
        print(f"event_count >={et:<5d}   RA={arms[f'event_count_{et}']['overall'][KEY]:8.4f}  "
              f"rate={p.rate():.3f}")
    ev_arm = min((k for k in arms if k.startswith("event_count_")),
                 key=lambda k: abs(arms[k]["policy"]["update_rate"] - thr))

    cmp = {}
    ctrl = min((k for k in arms if k.startswith(("delta_trust_", "random_", "event_count_"))),
               key=lambda k: arms[k]["overall"][KEY])
    for other in ("baseline", ctrl, "no_schur", "random_same_rate", ev_arm):
        if other == best or other not in arms or f"{best}_vs_{other}" in cmp:
            continue
        ci = MT.paired_sequence_bootstrap(arms[best]["per_seq"], arms[other]["per_seq"],
                                          arms[best]["weights"], seed=a.seed)
        cmp[f"{best}_vs_{other}"] = ci.as_dict()
        print(f"{best} - {other:<18s}: {ci.mean:+.4f} mm  [{ci.lo:+.4f}, {ci.hi:+.4f}]  "
              f"excludes_zero={ci.excludes_zero}")

    c_ctrl = cmp[f"{best}_vs_{ctrl}"]
    beats = lambda c: c["mean"] < 0 and c["excludes_zero"]                 # noqa: E731
    if beats(c_ctrl):
        verdict = "PASS"
    elif c_ctrl["half_width"] > abs(c_ctrl["mean"]):
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FAIL"
    print(f"\nS9 verdict: {verdict}   (best={best}, strongest control={ctrl}, "
          f"n_sequences={len(arms[best]['per_seq'])})")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p_out = out / f"s9_rdor_{a.split}_step{a.step_ms}.json"
    p_out.write_text(json.dumps({"checkpoint": a.ckpt, "config": str(cfg_path),
                                 "split": a.split, "step_ms": a.step_ms,
                                 "verdict": verdict, "best": best, "control": ctrl,
                                 "comparisons": cmp, "arms": arms}, indent=2))
    print("wrote", p_out)


if __name__ == "__main__":
    main()
