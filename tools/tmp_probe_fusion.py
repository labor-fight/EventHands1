#!/usr/bin/env python3
"""Break the loop instead of improving it.

The echo-gain probe measures what the recursive protocol actually does to an error: retention per
step is 0.75 for the KEG arm and 0.61 for the dense arm, on the model's own error *and* on random
directions, so a per-window offset is multiplied by roughly 1/(1-G) = 4.0 and 2.6. The per-window
offset itself is 12.1 mm and 11.0 mm under perfect conditioning, against 2.9 mm of true inter-frame
motion, and it is 91% autocorrelated -- so it is a standing error the loop compounds, not noise the
loop averages away.

That makes retention the lever, and an absolute predictor has retention zero by construction: it
never reads the previous state, so nothing it outputs can carry a past error forward. This measures
the zero-training blend

    x_k = (1 - alpha) f_trk(x_{k-1}) + alpha f_abs(events_k)

across alpha, under the same frozen recursive protocol every other number in this project is quoted
against. alpha = 0 is the production tracker, alpha = 1 is the pure anchor with the loop cut. If the
minimum sits strictly inside, the deficit is amplification and not per-window accuracy.

The blend is linear in the 51-vector, which is the same thing the recorded zero-training fusion did;
`semkine/anchor.py` is right that this is wrong on rotations, and that is a reason to prefer its
SO(3) fusion if this measurement says the direction is worth pursuing at all.

Temporary: delete after the verdict lands.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from config import load_config                                    # noqa: E402
from mano_layer import ManoLayer                                  # noqa: E402
from model import MNISTModel                                      # noqa: E402
from semkine import eval_track as ET                              # noqa: E402
from semkine.dataset import sequences_for_split, splits_manifest   # noqa: E402


def _load(ckpt, cfg_path, device):
    cfg = load_config(cfg_path)
    m = MNISTModel.load_from_checkpoint(ckpt, cfg=cfg, map_location=device).to(device).eval()
    return m, cfg


@torch.no_grad()
def rollout(trk, abs_m, mano, cfg, root, legacy_dir, seq, step_ms, device, alphas, max_steps,
            duty=None):
    """Sweep the blend weight, or match a duty cycle against it.

    Two mechanisms predict the same headline number and are usually conflated. Continuous dilution
    says every step must carry some anchor, because what is being fought is a standing bias that the
    loop retains at rate G. Occasional rescue says the anchor only has to fire when the tracker has
    wandered, which is what a hysteresis trigger implements and what E-3DPSM's appendix tests as
    periodic state resets.

    `duty=k` replaces the tracker outright every k-th step and leaves the other steps alone, so its
    *average* anchor weight is 1/k. Compared against the constant blend at alpha = 1/k, the two
    spend exactly the same amount of anchor and differ only in how it is spread. Whichever wins
    names the mechanism.
    """
    events, offsets, aux, pos51 = ET.load_sequence(root, legacy_dir, seq)
    tsub_path = root / legacy_dir / f"{seq}_tsub.npy"
    tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
    use_raw = bool(getattr(trk, "encoder_name", ""))
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    for m in (trk, abs_m):
        if hasattr(m, "set_hand_context"):
            m.set_hand_context(betas, K)
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
    zero = torch.zeros(1, 51, device=device)

    def fk(params):
        o = []
        for i in range(0, len(params), 2048):
            c = torch.from_numpy(np.ascontiguousarray(params[i:i + 2048], np.float32)).to(device)
            dec = ET.decode_to_mano_inputs(c, "mano_full_axis_angle",
                                           mano.hands_components, mano.hands_mean)
            _, j = mano(betas.expand(len(c), -1), dec["global_orient"],
                        dec["local_full_aa"], dec["transl"])
            o.append(j.cpu().numpy())
        return np.concatenate(o)

    out = {}
    for alpha in alphas:
        k = int(round(1.0 / alpha)) if (duty and alpha > 0) else 0
        rng = np.random.default_rng(0)          # identical initialisation for every alpha
        preds, gts = [], []
        n_anchored = 0
        n = 0
        for a, b in runs:
            ends = np.arange(a + step_ms - 1, b, step_ms, dtype=np.int64)
            if not len(ends):
                continue
            prev = pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)
            for end in ends:
                if max_steps and n >= max_steps:
                    break
                lnes = torch.from_numpy(
                    ET.build_lnes(events, offsets, int(end), step_ms)).unsqueeze(0).to(device)
                # Under a duty cycle the step is all tracker or all anchor, never a mixture.
                a_eff = (1.0 if (k and n % k == 0) else 0.0) if duty else alpha
                if a_eff < 1.0:
                    t = torch.from_numpy(
                        np.ascontiguousarray(prev, np.float32)).view(1, -1).to(device)
                    if use_raw:
                        ev5 = ET._window_events(events, offsets, tsub, int(end), step_ms)
                        p_trk = trk.forward_packet(
                            ET.make_eval_packet(ev5, t, betas, K, step_ms, device)
                        ).cpu().numpy()[0]
                    else:
                        p_trk = trk(lnes, t).cpu().numpy()[0]
                else:
                    p_trk = prev
                if a_eff > 0.0:
                    p_abs = abs_m(lnes, zero).cpu().numpy()[0]
                    n_anchored += 1
                else:
                    p_abs = p_trk
                p = (1.0 - a_eff) * p_trk + a_eff * p_abs
                preds.append(p)
                gts.append(pos51[int(end)])
                prev = p
                n += 1
            if max_steps and n >= max_steps:
                break
        pj, gj = fk(np.stack(preds)), fk(np.stack(gts))
        P, G = ET.root_align(pj), ET.root_align(gj)
        out[alpha] = {
            "n": int(len(P)),
            "ra_mm": float(np.linalg.norm(P - G, axis=-1).mean() * 1000),
            "abs_mm": float(np.linalg.norm(pj - gj, axis=-1).mean() * 1000),
            # Jitter is the other half of the anchor's recorded win, so it has to be reported too.
            "jitter_mm": float(np.linalg.norm(P[1:] - P[:-1], axis=-1).mean() * 1000),
            "anchor_rate": float(n_anchored / max(n, 1)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trk-ckpt", required=True)
    ap.add_argument("--trk-config", required=True)
    ap.add_argument("--abs-ckpt", required=True)
    ap.add_argument("--abs-config", default="configs/semkine/s1_abs_domrand.yaml")
    ap.add_argument("--route", default="soft")
    ap.add_argument("--split", default="val_core")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--duty", action="store_true",
                    help="spend the anchor intermittently at full weight instead of continuously, "
                         "with average weight matched to each alpha")
    ap.add_argument("--label", default="arm")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.environ["EVENTHANDS_KEG_ROUTE"] = a.route
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trk, cfg = _load(a.trk_ckpt, a.trk_config, device)
    abs_m, _ = _load(a.abs_ckpt, a.abs_config, device)
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    alphas = [float(x) for x in a.alphas.split(",")]

    per = []
    for seq, d in sequences_for_split(root, a.split, splits_manifest(cfg)):
        row = rollout(trk, abs_m, mano, cfg, root, d, seq, a.step_ms, device, alphas, a.max_steps,
                      duty=a.duty)
        row["seq"] = seq
        per.append(row)
        print(f"  {seq}: " + "  ".join(f"a{al:g}={row[al]['ra_mm']:.2f}" for al in alphas),
              flush=True)

    pooled = {}
    for al in alphas:
        w = np.array([p[al]["n"] for p in per], np.float64)
        pooled[str(al)] = {k: float(np.sum([p[al][k] * wi for p, wi in zip(per, w)]) / w.sum())
                           for k in ("ra_mm", "abs_mm", "jitter_mm", "anchor_rate")}
        pooled[str(al)]["n"] = int(w.sum())
    print(json.dumps(pooled, indent=2))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"label": a.label, "trk": a.trk_ckpt, "abs": a.abs_ckpt, "split": a.split,
         "duty": bool(a.duty),
         "pooled": pooled,
         "per_sequence": [{("seq" if k == "seq" else str(k)): v for k, v in p.items()}
                          for p in per]}, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
