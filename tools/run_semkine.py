#!/usr/bin/env python3
r"""The SemKine mainline evaluation: one checkpoint, one event stream, every arm.

Arms, in the order they build on each other, so each line of the output attributes its difference to
exactly one added mechanism:

    baseline          the frozen tracker, nothing added
    delta_trust       constant shrinkage of every update -- the cheap control S7 showed is strong
    filter            S12 alone: Lie-manifold filtering with information-weighted measurements
    filter+router     S9 added: which degrees of freedom the packet is allowed to move
    filter+router+anchor  S13 added: triggered fusion with a frozen absolute predictor

Every arm shares the checkpoint and the event stream, so comparisons are paired per sequence and the
0.4 mm replicate floor does not apply; the reported intervals are paired sequence bootstraps.

Run at several step sizes. S7 established that at 50 ms almost every joint moves enough to be worth
updating, which caps what routing can add at about 0.2 mm -- inside the noise floor. The premise of
the active branch is about *below-resolution* motion, so the short windows are where it either works
or does not.
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
from semkine import mainline as ML                      # noqa: E402
from semkine import metrics as MT                       # noqa: E402
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
    out = {
        "overall": {k: float(sum(r[k] * r["n_frames"] for r in rows) / max(n, 1))
                    for k in ("mpjpe_ra_mm", "mpjpe_abs_mm", "mpvpe_ra_mm")},
        "n_frames": n,
        "per_seq": {r["seq"]: r[KEY] for r in rows},
        "weights": {r["seq"]: r["n_frames"] for r in rows},
        "drift": {r["seq"]: r["drift"] for r in rows},
        "jitter": float(np.mean([r["jitter_all_mm_per_step"] for r in rows
                                 if r["jitter_all_mm_per_step"] is not None])),
    }
    if policy is not None and hasattr(policy, "stats"):
        out["policy"] = policy.stats()
    return out


def make_anchor(cfg_path, ckpt, device, step_ms):
    """A frozen absolute predictor as the anchor, per the plan's correction 7.

    It reads the same events the tracker does, over the same window, and predicts the pose without
    any prior. Being dense and expensive is acceptable because the trigger runs it rarely; being
    already trained is the point.
    """
    if not ckpt:
        return None
    acfg = load_config(cfg_path)
    amodel = MNISTModel.load_from_checkpoint(ckpt, cfg=acfg,
                                            map_location=device).to(device).eval()
    assert not acfg["MODEL"].get("PREDICT_DELTA", False), \
        "the anchor must be absolute; a delta model would need a prior and could not anchor"

    @torch.no_grad()
    def anchor(lnes):
        if lnes is None:
            return None
        # An absolute model ignores `prevpos`, but the signature requires it, so a zero state is
        # passed. Asserted above that the checkpoint really is absolute, since handing a tracking
        # checkpoint a zero prior would silently produce a prediction relative to nothing.
        zero = torch.zeros(lnes.shape[0], 51, device=lnes.device, dtype=lnes.dtype)
        return amodel(lnes, zero)

    return anchor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--anchor-ckpt", default=None)
    ap.add_argument("--anchor-config", default="configs/eventhands_abs_full51.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--step-ms", type=int, nargs="+", default=[50])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--info-divisor", type=float, default=2.0e4)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="outputs/semkine/mainline")
    a = ap.parse_args()

    cfg_path = a.config
    if cfg_path is None:
        cfg_path = json.loads((Path(a.ckpt).parent
                               / "training_metadata.json").read_text())["config_path"]
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel.load_from_checkpoint(a.ckpt, cfg=cfg, map_location=device).to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    seqs = sequences_for_split(root, a.split, None)
    if a.only:
        seqs = [s for s in seqs if any(p in s[0] for p in a.only.split(","))]
    assert seqs, "no sequences selected"
    anchor = make_anchor(a.anchor_config, a.anchor_ckpt, device, a.step_ms[0])
    print(f"{len(seqs)} sequences, config={cfg_path}, anchor={'yes' if anchor else 'no'}")

    report = {}
    for step_ms in a.step_ms:
        fk = dict(info_divisor=a.info_divisor)
        rk = dict(budget=a.budget)
        arms = {}
        arms["baseline"] = evaluate(model, mano, cfg, root, seqs, step_ms, device, a.seed)
        arms["delta_trust_0.5"] = evaluate(model, mano, cfg, root, seqs, step_ms, device,
                                          a.seed, delta_trust=0.5)
        specs = [("filter", dict(use_filter=True, use_router=False)),
                 ("filter+router", dict(use_filter=True, use_router=True))]
        if anchor is not None:
            specs.append(("filter+router+anchor",
                          dict(use_filter=True, use_router=True, anchor_model=anchor)))
        for name, kw in specs:
            pol = ML.SemKinePolicy(mano=mano, step_ms=step_ms, router_kwargs=rk,
                                   filter_kwargs=fk, **kw)
            arms[name] = evaluate(model, mano, cfg, root, seqs, step_ms, device, a.seed,
                                  policy=pol)
        for name, r in arms.items():
            extra = ""
            if "policy" in r:
                p = r["policy"]
                extra = (f"  rate={p.get('update_rate', 1):.3f}"
                         f"  anchored={p.get('n_anchored', 0)}")
            print(f"  [{step_ms:>4} ms] {name:<22s} RA={r['overall'][KEY]:8.4f}"
                  f"  jitter={r['jitter']:.3f}{extra}")

        cmp = {}
        for name in arms:
            if name in ("baseline", "delta_trust_0.5"):
                continue
            for other in ("baseline", "delta_trust_0.5"):
                ci = MT.paired_sequence_bootstrap(arms[name]["per_seq"], arms[other]["per_seq"],
                                                  arms[name]["weights"], seed=a.seed)
                cmp[f"{name}_vs_{other}"] = ci.as_dict()
                print(f"           {name} - {other}: {ci.mean:+.4f} mm "
                      f"[{ci.lo:+.4f}, {ci.hi:+.4f}] excludes_zero={ci.excludes_zero}")
        report[str(step_ms)] = {"arms": arms, "comparisons": cmp}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"mainline_{a.split}_{Path(a.ckpt).stem}.json"
    p.write_text(json.dumps({"checkpoint": a.ckpt, "config": str(cfg_path),
                             "anchor_ckpt": a.anchor_ckpt, "split": a.split,
                             "by_step_ms": report}, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
