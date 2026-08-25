#!/usr/bin/env python3
r"""Paired arm comparison across checkpoints and step sizes, for the S19-S22 gates.

`run_s9_rdor.py` compares policies on one checkpoint; this compares *checkpoints*, which is what
G1 (parity at 50 ms), G2 (the variable-rate claim), and G3 (the KSSF mechanism ablation) all need.
The two differ in one methodological respect that decides which threshold applies:

* Same checkpoint, different inference-time switch -- G3's `--ablate-kssf`, and the `--step-ms`
  sweep of one arm. The event stream and the weights are identical, so the paired sequence
  bootstrap is the whole test and thresholds of a few tenths of a millimetre are decidable.
* Different checkpoints -- G1 and G2 compare separately trained arms. The pairing still removes
  the between-sequence variance, which is the largest term, but it cannot remove training noise:
  `ARCHITECTURE_AUDIT.md` §5 measures that at 0.4 mm for a same-config replicate and 1.1 mm for a
  cross-training single-variable claim. So a cross-checkpoint verdict is reported per seed *and*
  as the worse of the seeds, and `decide_gate` is told the regime is `retrained` so it annotates
  an interval narrower than the replicate floor instead of letting it look decisive.

An arm is `label=path/to.ckpt`. Its config comes from the `training_metadata.json` beside it, so
an arm always evaluates under the protocol it was trained with.
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
from semkine.dataset import sequences_for_split         # noqa: E402

KEY = "mpjpe_ra_mm"


def resolve_config(ckpt: Path, override: str | None) -> str:
    if override:
        return override
    meta = ckpt.parent / "training_metadata.json"
    if meta.exists():
        return json.loads(meta.read_text())["config_path"]
    cand = sorted(ckpt.parent.glob("*.yaml"))
    assert cand, f"no config for {ckpt}: pass --config"
    return str(cand[0])


def evaluate(model, mano, cfg, root, seqs, step_ms, device, seed, delta_trust=1.0):
    rng = np.random.default_rng(seed)
    rows = [ET.track_sequence(model, mano, cfg, root, d, s, step_ms, device, rng,
                              1.0, delta_trust, None) for s, d in seqs]
    rows = [r for r in rows if r]
    n = sum(r["n_frames"] for r in rows)
    return {
        "overall": {k: float(sum(r[k] * r["n_frames"] for r in rows) / max(n, 1))
                    for k in ("mpjpe_ra_mm", "mpjpe_abs_mm", "mpvpe_ra_mm")},
        "n_frames": n,
        "per_seq": {r["seq"]: r[KEY] for r in rows},
        "weights": {r["seq"]: float(r["n_frames"]) for r in rows},
        "jitter": float(np.mean([r["jitter_all_mm_per_step"] for r in rows
                                 if r["jitter_all_mm_per_step"] is not None])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="LABEL=CKPT")
    ap.add_argument("--config", default=None, help="applies to every arm; normally left unset")
    ap.add_argument("--split", default="val_core")
    ap.add_argument("--step-ms", default="50", help="comma-separated list")
    ap.add_argument("--seed", type=int, default=0)
    # Repeatable, because a worse-of-two-seeds verdict needs each candidate seed compared against
    # each control seed: pairing only against one control would let the verdict inherit whichever
    # control replicate happened to land badly, which is the training noise S0 measured at 0.4 mm.
    ap.add_argument("--reference", action="append", default=None,
                    help="label every other arm is compared against; repeatable")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="mm; candidate minus reference must stay below this")
    ap.add_argument("--direction", default="no_worse_than",
                    choices=("no_worse_than", "lower_is_better"))
    ap.add_argument("--regime", default="retrained", choices=("retrained", "same_checkpoint"))
    ap.add_argument("--ablate-kssf", default="",
                    help="comma-separated labels evaluated with the KSSF channels silenced")
    ap.add_argument("--gate-name", default="gate")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = [int(s) for s in str(a.step_ms).split(",")]
    ablate = {s for s in a.ablate_kssf.split(",") if s}
    arms = {}
    for spec in a.arm:
        label, _, ck = spec.partition("=")
        assert ck, f"--arm needs LABEL=CKPT, got {spec!r}"
        ckpt = Path(ck)
        cfg_path = resolve_config(ckpt, a.config)
        cfg = load_config(cfg_path)
        model = MNISTModel.load_from_checkpoint(str(ckpt), cfg=cfg,
                                                map_location=device).to(device).eval()
        if label in ablate:
            assert getattr(model, "encoder_kssf", False), f"{label} has no KSSF to ablate"
            model.ablate_kssf = True
        mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
        root = Path(cfg["DATA"]["ROOT"])
        seqs = sequences_for_split(root, a.split, None)
        for step in steps:
            r = evaluate(model, mano, cfg, root, seqs, step, device, a.seed)
            arms[(label, step)] = r
            print(f"{label:<28s} step={step:>3d}ms  RA={r['overall'][KEY]:8.4f}  "
                  f"abs={r['overall']['mpjpe_abs_mm']:8.3f}  jitter={r['jitter']:.3f}",
                  flush=True)
        del model
        torch.cuda.empty_cache()

    gates, cmp = [], {}
    refs = a.reference or []
    for ref_label in refs:
        for (label, step), r in arms.items():
            if label in refs:
                continue
            ref = arms.get((ref_label, step))
            if ref is None:
                continue
            ci = MT.paired_sequence_bootstrap(r["per_seq"], ref["per_seq"], r["weights"],
                                              seed=a.seed)
            name = f"{a.gate_name}:{label}-{ref_label}@{step}ms"
            g = MT.decide_gate(name, ci, a.threshold, a.direction, regime=a.regime)
            gates.append(g)
            cmp[name] = ci.as_dict()
    if gates:
        print("\n" + MT.format_gates(gates))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": a.split, "steps": steps, "seed": a.seed, "reference": refs,
        "threshold": a.threshold, "direction": a.direction, "regime": a.regime,
        "ablate_kssf": sorted(ablate),
        "arms": {f"{k[0]}@{k[1]}ms": v for k, v in arms.items()},
        "comparisons": cmp, "gates": [g.as_dict() for g in gates],
    }, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
