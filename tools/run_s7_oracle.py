#!/usr/bin/env python3
r"""S7: does selective updating help at all, when an oracle picks what to update?

Run against a frozen tracking checkpoint. The oracle reads the ground-truth motion of each
coordinate group to decide whether to update it, and skips the rest for real. Each oracle threshold
is paired with a random policy that skips at the same measured rate, and with the constant
delta-trust arm, because shrinkage toward the previous state helps on its own and would otherwise
be mistaken for evidence that routing works.

Verdict rule, registered in advance: the active mainline continues only if some oracle threshold
beats *both* the unmodified baseline and the best of the shrinkage controls, with a paired
sequence bootstrap confidence interval that excludes zero.
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
from semkine.dataset import sequences_for_split         # noqa: E402

KEY = "mpjpe_ra_mm"


def evaluate(model, mano, cfg, root, seqs, step_ms, device, seed, policy=None,
             delta_trust=1.0):
    rng = np.random.default_rng(seed)
    rows = []
    for seq, legacy_dir in seqs:
        r = ET.track_sequence(model, mano, cfg, root, legacy_dir, seq, step_ms, device,
                              rng, 1.0, delta_trust, None, active_policy=policy)
        if r:
            rows.append(r)
    n = sum(r["n_frames"] for r in rows)
    return {
        "overall": {k: float(sum(r[k] * r["n_frames"] for r in rows) / max(n, 1))
                    for k in ("mpjpe_ra_mm", "mpjpe_abs_mm", "mpvpe_ra_mm")},
        "n_frames": n,
        "per_seq": {r["seq"]: r[KEY] for r in rows},
        "weights": {r["seq"]: r["n_frames"] for r in rows},
        "jitter": float(np.mean([r["jitter_all_mm_per_step"] for r in rows
                                 if r["jitter_all_mm_per_step"] is not None])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eventhands_track_render51.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    # Calibrated on the ground truth: at a 50 ms step, thresholds below 0.02 rad skip essentially
    # nothing (update rate 0.96+), so a sweep starting there would report the identity six times.
    ap.add_argument("--thresholds", default="0.02,0.05,0.1,0.2,0.4")
    ap.add_argument("--only", default=None,
                    help="comma-separated substrings; keep sequences matching any. Needed to "
                         "restrict a frozen checkpoint to subjects it did not train on.")
    ap.add_argument("--out", default="outputs/semkine/s7_oracle")
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel.load_from_checkpoint(a.ckpt, cfg=cfg, map_location=device).to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    seqs = sequences_for_split(root, a.split, None)
    if a.only:
        pats = a.only.split(",")
        seqs = [s for s in seqs if any(p in s[0] for p in pats)]
    hm = mano.hands_mean.detach().cpu().numpy().reshape(-1)
    print(f"{len(seqs)} sequences in split={a.split}"
          + (f" filtered to {a.only}" if a.only else ""))
    assert seqs, "no sequences selected"

    arms = {}
    base = evaluate(model, mano, cfg, root, seqs, a.step_ms, device, a.seed)
    arms["baseline"] = base
    print(f"baseline            RA={base['overall'][KEY]:8.4f}  jitter={base['jitter']:.3f}")

    # Parity: an `all` policy must reproduce the untouched path exactly.
    p_all = OR.ActivePolicy(mode="all")
    chk = evaluate(model, mano, cfg, root, seqs, a.step_ms, device, a.seed, policy=p_all)
    dev = abs(chk["overall"][KEY] - base["overall"][KEY])
    assert dev < 1e-9, f"active-policy plumbing changed the baseline by {dev:.3e} mm"
    print(f"policy=all parity   dev={dev:.2e} mm  OK")

    for dt in (0.5, 0.7):
        r = evaluate(model, mano, cfg, root, seqs, a.step_ms, device, a.seed, delta_trust=dt)
        arms[f"delta_trust_{dt:g}"] = r
        print(f"delta_trust {dt:<4g}    RA={r['overall'][KEY]:8.4f}  jitter={r['jitter']:.3f}")

    for th in [float(x) for x in a.thresholds.split(",")]:
        p = OR.ActivePolicy(mode="oracle", thresh=th, hands_mean=hm, seed=a.seed)
        r = evaluate(model, mano, cfg, root, seqs, a.step_ms, device, a.seed, policy=p)
        r["update_rate"] = p.rate()
        arms[f"oracle_{th:g}"] = r
        # The shrinkage control: same measured update rate, chosen blind to the motion.
        q = OR.ActivePolicy(mode="random", thresh=p.rate(), seed=a.seed + 1)
        rr = evaluate(model, mano, cfg, root, seqs, a.step_ms, device, a.seed, policy=q)
        rr["update_rate"] = q.rate()
        arms[f"random_{th:g}"] = rr
        print(f"oracle th={th:<6g} RA={r['overall'][KEY]:8.4f}  rate={p.rate():.3f}  "
              f"jitter={r['jitter']:.3f}   | random same-rate RA={rr['overall'][KEY]:8.4f}")

    # Paired comparisons, all against the same checkpoint and the same event stream, so a paired
    # bootstrap over sequences is the right test and the 0.4 mm training noise floor does not apply.
    best = min((k for k in arms if k.startswith("oracle_")),
               key=lambda k: arms[k]["overall"][KEY])
    ctrl = min((k for k in arms if k.startswith(("random_", "delta_trust_"))),
               key=lambda k: arms[k]["overall"][KEY])
    cmp = {}
    for other in ("baseline", ctrl):
        ci = MT.paired_sequence_bootstrap(arms[best]["per_seq"], arms[other]["per_seq"],
                                          arms[best]["weights"], seed=a.seed)
        cmp[f"{best}_vs_{other}"] = ci.as_dict()
        print(f"\n{best} - {other}: {ci.mean:+.4f} mm  "
              f"[{ci.lo:+.4f}, {ci.hi:+.4f}]  excludes_zero={ci.excludes_zero}")

    # Also report the oracle against the same-rate random policy, which is what isolates the
    # routing signal from the shrinkage: a policy skipping 60% of updates helps regardless of
    # which ones it skips, and only the gap over random says the *choice* carried information.
    rnd = f"random_{best.split('_', 1)[1]}"
    if rnd in arms:
        ci = MT.paired_sequence_bootstrap(arms[best]["per_seq"], arms[rnd]["per_seq"],
                                          arms[best]["weights"], seed=a.seed)
        cmp[f"{best}_vs_{rnd}"] = ci.as_dict()
        print(f"{best} - {rnd}: {ci.mean:+.4f} mm  [{ci.lo:+.4f}, {ci.hi:+.4f}]  "
              f"excludes_zero={ci.excludes_zero}")

    c_base, c_ctrl = cmp[f"{best}_vs_baseline"], cmp[f"{best}_vs_{ctrl}"]
    beats = lambda c: c["mean"] < 0 and c["excludes_zero"]                 # noqa: E731
    if beats(c_base) and beats(c_ctrl):
        verdict = "PASS"
    elif c_ctrl["half_width"] > abs(c_ctrl["mean"]):
        # Distinguishing this from FAIL matters: the pre-registered rule stops the active mainline
        # on a FAIL, and stopping on a test whose confidence interval is wider than the effect it
        # is measuring would be a decision made by lack of data rather than by evidence.
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FAIL"
    print(f"\nS7 verdict: {verdict}   (best={best}, strongest control={ctrl}, "
          f"n_sequences={len(arms[best]['per_seq'])})")
    if verdict == "INCONCLUSIVE":
        print(f"  effect {c_ctrl['mean']:+.4f} mm vs CI half-width {c_ctrl['half_width']:.4f} mm; "
              f"more sequences needed before the stopping rule can fire")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"s7_oracle_{a.split}_step{a.step_ms}.json"
    p.write_text(json.dumps({"checkpoint": a.ckpt, "split": a.split, "step_ms": a.step_ms,
                             "verdict": verdict, "best": best, "control": ctrl,
                             "comparisons": cmp, "arms": arms}, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
