#!/usr/bin/env python3
"""What sets the 14.45 mm single-step bias floor, and can any architecture move it?

§11.2 closed the evidence-quantity lever: G bottoms out at 0.21 and the window optimum is already
occupied, so `recur = bias/(1-G)` leaves only the bias. §11.4 asked the question this probe answers
-- the floor is 6 to 31 times the motion each step has to track, so it is not a tracking error, and
before anyone rewrites the encoder we should know which of four things it is.

The four are separable, and two of them point in opposite directions:

* **capacity / overfitting** -- bias on `train` versus `val_core`. Equal means the floor is intrinsic
  to the problem as posed, not a generalisation gap the network could close with more parameters.
* **annotation** -- ground truth is sampled every millisecond, so a real hand's root-aligned joint
  trajectory is locally quadratic to well under a millimetre. Whatever residual a local quadratic
  fit leaves is annotation noise, and it lower-bounds any achievable error.
* **observability** -- bias against the number of events *available* in the window. If the floor is
  set by event scarcity then the information was never captured, and no encoder recovers it.
* **representation** -- bias against the fraction of available events LNES *discards* under
  last-writer-wins. If the floor is set by discarding, the information is present and a
  non-saturating encoder can reach it.

The last two are the decision. They are measured on the same windows, so their partial correlations
separate cleanly, and they cannot both be the answer.

Teacher-forced only: one forward per window, no recursion, so this is the bias term of the
amplification law and nothing else.

Temporary: `/tmp`-class diagnostic per the experiment contract; delete after the verdict lands.
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

from config import load_config                                    # noqa: E402
from mano_layer import ManoLayer                                  # noqa: E402
from model import MNISTModel                                      # noqa: E402
from semkine import eval_track as ET                              # noqa: E402
from semkine.dataset import sequences_for_split, splits_manifest   # noqa: E402


class Fk:
    def __init__(self, mano, betas, device):
        self.mano, self.betas, self.device = mano, betas, device

    @torch.no_grad()
    def __call__(self, params: np.ndarray) -> np.ndarray:
        p = torch.from_numpy(np.ascontiguousarray(params, np.float32)).to(self.device)
        dec = ET.decode_to_mano_inputs(p, "mano_full_axis_angle",
                                       self.mano.hands_components, self.mano.hands_mean)
        _, j = self.mano(self.betas.expand(len(p), -1), dec["global_orient"],
                         dec["local_full_aa"], dec["transl"])
        return j.cpu().numpy()


def _ra(j: np.ndarray) -> np.ndarray:
    return ET.root_align(j)


def _event_stats(events, offsets, end: int, window: int) -> tuple:
    """Available events, slots LNES keeps, and the fraction it drops to keep one per slot."""
    start = end - window + 1
    a0, a1 = int(offsets[start]), int(offsets[end + 1])
    n = a1 - a0
    if n <= 0:
        return 0, 0, 0.0
    ev = np.asarray(events[a0:a1])
    slot = ((ev[:, 1].astype(np.int64) * ET.W + ev[:, 0].astype(np.int64)) * 2
            + np.clip(ev[:, 2].astype(np.int64), 0, 1))
    occ = int(len(np.unique(slot)))
    return int(n), occ, float(1.0 - occ / n)


def _gt_residual(ra_win: np.ndarray) -> float:
    """Residual of a local quadratic fit to the root-aligned GT, in mm.

    Ground truth is sampled per millisecond. Over a few tens of milliseconds a hand joint follows a
    quadratic to far below the measurement scale, so the residual is dominated by whatever noise the
    annotation carries and is an upper bound on how well any model could have matched it.
    """
    k = len(ra_win)
    t = np.arange(k, dtype=np.float64) - (k - 1) / 2.0
    A = np.stack([np.ones_like(t), t, t * t], 1)
    flat = ra_win.reshape(k, -1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, flat, rcond=None)
    resid = flat - A @ coef
    mid = resid[k // 2].reshape(-1, 3)
    return float(np.linalg.norm(mid, axis=-1).mean() * 1000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", default="val_core,train")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--window-ms", type=int, default=0, help="0 = same as --step-ms")
    ap.add_argument("--samples", type=int, default=120, help="instants per sequence")
    ap.add_argument("--gt-halfwidth", type=int, default=10, help="+/- ms for the GT smoothness fit")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel.load_from_checkpoint(a.ckpt, cfg=cfg, map_location=device).to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    win = a.window_ms or a.step_ms
    hw = a.gt_halfwidth

    out = {"ckpt": a.ckpt, "step_ms": a.step_ms, "window_ms": win,
           "gt_halfwidth_ms": hw, "splits": {}}
    for split in a.splits.split(","):
        rows = []
        for seq, d in sequences_for_split(root, split, splits_manifest(cfg)):
            events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
            betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
            K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
            if hasattr(model, "set_hand_context"):
                model.set_hand_context(betas, K)
            fk = Fk(mano, betas, device)
            runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
            lo_pad = max(a.step_ms, win) + hw
            cand = [np.linspace(int(lo) + lo_pad, int(hi) - 1 - hw, a.samples, dtype=np.int64)
                    for lo, hi in runs if int(hi) - 1 - hw - (int(lo) + lo_pad) > 1]
            if not cand:
                continue
            ends = np.unique(np.concatenate(cand))
            srows = []
            for end in ends:
                end = int(end)
                g = pos51[end - a.step_ms].copy()          # perfect conditioning: no recursion
                tgt = pos51[end].copy()
                lnes = torch.from_numpy(
                    ET.build_lnes(events, offsets, end, win)).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model(lnes, torch.from_numpy(g).view(1, -1).to(device)).cpu().numpy()[0]
                j = fk(np.stack([pred, tgt, g]))
                bias = float(np.linalg.norm(_ra(j[0]) - _ra(j[1]), axis=-1).mean() * 1000)
                move = float(np.linalg.norm(_ra(j[2]) - _ra(j[1]), axis=-1).mean() * 1000)
                n_ev, occ, ovr = _event_stats(events, offsets, end, win)
                gtr = _gt_residual(fk(np.stack([pos51[t] for t in
                                                range(end - hw, end + hw + 1)])))
                srows.append({"bias": bias, "move": move, "n_events": n_ev,
                              "occupied": occ, "overwrite": ovr, "gt_resid": gtr})
            rows += srows
            b = np.array([r["bias"] for r in srows])
            print(f"  [{split}] {seq}: bias={b.mean():.2f}mm  "
                  f"gt_resid={np.mean([r['gt_resid'] for r in srows]):.3f}mm  "
                  f"n_ev={np.mean([r['n_events'] for r in srows]):.0f}  "
                  f"ovr={np.mean([r['overwrite'] for r in srows])*100:.0f}%", flush=True)

        A = {k: np.array([r[k] for r in rows], np.float64) for k in rows[0]}
        pooled = {k: float(v.mean()) for k, v in A.items()}
        pooled["n"] = len(rows)
        out["splits"][split] = {"pooled": pooled,
                                "rows": [{k: float(v) for k, v in r.items()} for r in rows]}
        print(f"[{split}] pooled bias={pooled['bias']:.3f}mm  gt_resid={pooled['gt_resid']:.3f}mm  "
              f"n_events={pooled['n_events']:.0f}  overwrite={pooled['overwrite']*100:.1f}%  "
              f"n={len(rows)}\n", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
