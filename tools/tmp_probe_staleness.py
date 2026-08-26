#!/usr/bin/env python3
"""Plan B's precondition: how much does the packet-start KSSF cost?

`forward_packet` rasterises the kinematic semantic field once, at the previous state, and every
event in the window is lifted through it -- contour normal, signed distance, skinning weights, part
code. At 50 ms the hand moves 0.49 rad of local pose per window (`need_step_pose`, halo probe), so
the last event in a packet is read against geometry that is a whole step out of date. If that is
what the loop keeps feeding back, the fix is temporal geometry (plan B) and nothing else.

The measurement is a single-variable swap. Nothing changes except which pose the field is built at:

    prev_start   one field at the conditioning state          (production)
    gt_start     one field at the ground-truth pose at t0     (isolates conditioning drift)
    gt_<R>ms     a field every R ms at the ground-truth pose  (oracle: drift *and* staleness gone)
    cv_<R>ms     a field every R ms at a constant-velocity
                 extrapolation of the last two states         (causal, actually implementable)

Under teacher forcing `prev_start` and `gt_start` are the same run, which is the built-in check that
the swap machinery is inert. The events, the network weights, the aggregator and the head are
identical in every condition: the field is batched over the R-ms knots and each event queries the
knot it falls in, through the *same* `kssf_event_channels` the production path calls, so the K = 1
case is bit-identical to `forward_packet` rather than a re-implementation of it.

Temporary: delete after the verdict lands.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
from semkine import keg as KEG                                    # noqa: E402
from semkine.dataset import sequences_for_split                   # noqa: E402
from semkine.events import EV_BATCH, EV_T                         # noqa: E402


_ORIG = KEG.kssf_event_channels


class KnotFields:
    """Redirects the per-event lift to a field batched over intra-window knots.

    Installed by monkey-patching `semkine.keg.kssf_event_channels`, which `forward_packet` imports
    at call time. When `poses` is None the original function runs untouched, so an arm can be probed
    in production mode through exactly the same code path.
    """

    def __init__(self, kssf, betas, camera_K, pose_repr):
        self.kssf, self.betas, self.K, self.repr = kssf, betas, camera_K, pose_repr
        self.poses = None
        self.knot_us = None
        self.n_knots_seen = []

    def arm(self, poses51: np.ndarray, knot_us: int) -> None:
        self.poses = poses51
        self.knot_us = int(knot_us)

    def disarm(self) -> None:
        self.poses = None

    def __call__(self, fields, events: torch.Tensor, halo: bool = False):
        if self.poses is None or events.shape[0] == 0:
            return _ORIG(fields, events, halo=halo)
        n_k = len(self.poses)
        p = torch.from_numpy(np.ascontiguousarray(self.poses, np.float32)).to(events.device)
        f = self.kssf(p, self.betas.expand(n_k, -1), self.K.expand(n_k, 3, 3), self.repr)
        ev = events.clone()
        # Each event reads the knot covering its own timestamp; the last knot covers the tail.
        k = torch.div(events[:, EV_T] * 1e6, self.knot_us, rounding_mode="floor")
        ev[:, EV_BATCH] = k.clamp_(0, n_k - 1)
        self.n_knots_seen.append(n_k)
        return _ORIG(f, ev, halo=halo)


def _cv_poses(prev: np.ndarray, prev2: np.ndarray, n_k: int, knot_ms: int,
              step_ms: int) -> np.ndarray:
    """Constant-velocity extrapolation of the last two states: causal, no ground truth."""
    vel = (prev - prev2) / float(step_ms)
    return np.stack([prev + vel * (k * knot_ms) for k in range(n_k)])


@torch.no_grad()
def probe_sequence(model, mano, cfg, root, legacy_dir, seq, step_ms, device, conds,
                   stride, max_steps, teacher_forced):
    events, offsets, aux, pos51 = ET.load_sequence(root, legacy_dir, seq)
    tsub_path = root / legacy_dir / f"{seq}_tsub.npy"
    tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    Kmat = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    if hasattr(model, "set_hand_context"):
        model.set_hand_context(betas, Kmat)
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)

    knots = KnotFields(model.kssf_on(device), betas, Kmat, model.pose_repr)
    KEG.kssf_event_channels = knots

    rng = np.random.default_rng(0)
    acc = {c: {"pred": [], "gt": []} for c in conds}
    span = []
    n_steps = 0
    try:
        for a, b in runs:
            ends = np.arange(a + step_ms - 1, b, step_ms, dtype=np.int64)
            if not len(ends):
                continue
            prev = pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)
            prev_prev = prev.copy()
            prev_end = int(a)
            for end in ends:
                if max_steps and n_steps >= max_steps:
                    break
                ev5 = ET._window_events(events, offsets, tsub, int(end), step_ms)
                w_start = int(end) - step_ms + 1
                cond_state = pos51[prev_end].copy() if teacher_forced else prev
                prev_t = torch.from_numpy(
                    np.ascontiguousarray(cond_state, np.float32)).view(1, -1).to(device)
                pkt = ET.make_eval_packet(ev5, prev_t, betas, Kmat, step_ms, device)
                production = None

                if (n_steps % stride) == 0:
                    for name, (src, knot_ms) in conds.items():
                        if src == "prev" and knot_ms == step_ms:
                            knots.disarm()
                        else:
                            n_k = max(1, step_ms // knot_ms)
                            if src == "gt":
                                idx = np.clip(w_start + np.arange(n_k) * knot_ms,
                                              0, len(pos51) - 1)
                                poses = pos51[idx].astype(np.float32)
                            else:
                                p2 = (pos51[max(prev_end - step_ms, 0)] if teacher_forced
                                      else prev_prev)
                                poses = _cv_poses(cond_state, p2, n_k, knot_ms, step_ms)
                            knots.arm(poses, knot_ms * 1000)
                        p = model.forward_packet(pkt).cpu().numpy()[0]
                        knots.disarm()
                        acc[name]["pred"].append(p)
                        acc[name]["gt"].append(pos51[int(end)])
                        if src == "prev":
                            production = p
                    lo = np.clip(w_start, 0, len(pos51) - 1)
                    hi = np.clip(int(end), 0, len(pos51) - 1)
                    span.append((pos51[lo], pos51[hi]))

                if production is None:
                    production = model.forward_packet(pkt).cpu().numpy()[0]
                prev_prev, prev = prev, production
                prev_end = int(end)
                n_steps += 1
            if max_steps and n_steps >= max_steps:
                break
    finally:
        KEG.kssf_event_channels = _ORIG

    rows = {c: _metrics(acc[c]["pred"], acc[c]["gt"], mano, betas, device) for c in conds}
    # How far the hand actually travels between the first and the last knot. Without this the
    # null result is unreadable: "refreshing changed nothing" only means something once we know
    # the geometry being refreshed had really moved.
    if span:
        rows["_knot_span"] = _metrics([s[0] for s in span], [s[1] for s in span],
                                      mano, betas, device)
    return rows, n_steps


def _metrics(pred, gt, mano, betas, device) -> dict:
    def fk(p):
        o = []
        for i in range(0, len(p), 2048):
            c = torch.from_numpy(np.ascontiguousarray(p[i:i + 2048], np.float32)).to(device)
            dec = ET.decode_to_mano_inputs(c, "mano_full_axis_angle",
                                           mano.hands_components, mano.hands_mean)
            _, j = mano(betas.expand(len(c), -1), dec["global_orient"],
                        dec["local_full_aa"], dec["transl"])
            o.append(j.cpu().numpy())
        return np.concatenate(o)

    if not pred:
        return {"n": 0, "ra_mm": float("nan"), "abs_mm": float("nan")}
    pj, gj = fk(np.stack(pred)), fk(np.stack(gt))
    return {
        "n": int(len(pred)),
        "ra_mm": float(np.linalg.norm(ET.root_align(pj) - ET.root_align(gj), axis=-1).mean() * 1000),
        "abs_mm": float(np.linalg.norm(pj - gj, axis=-1).mean() * 1000),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--route", default="soft")
    ap.add_argument("--split", default="val_core")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--knots-ms", default="10,5,2")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--regime", default="both", choices=["tf", "recursive", "both"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.environ["EVENTHANDS_KEG_ROUTE"] = a.route
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(a.config)
    model = MNISTModel.load_from_checkpoint(a.ckpt, cfg=cfg, map_location=device).to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])

    conds = {"prev_start": ("prev", a.step_ms), "gt_start": ("gt", a.step_ms)}
    for r in [int(x) for x in a.knots_ms.split(",") if x]:
        conds[f"gt_{r}ms"] = ("gt", r)
        conds[f"cv_{r}ms"] = ("cv", r)

    out = {"ckpt": a.ckpt, "step_ms": a.step_ms, "split": a.split, "regimes": {}}
    regimes = ["tf", "recursive"] if a.regime == "both" else [a.regime]
    for reg in regimes:
        per = []
        for seq, d in sequences_for_split(root, a.split, None):
            srow, n = probe_sequence(model, mano, cfg, root, d, seq, a.step_ms, device,
                                     conds, a.stride, a.max_steps, reg == "tf")
            per.append(srow)
            print(f"  [{reg}] {seq}: {n} steps  span={srow.get('_knot_span', {}).get('ra_mm', 0):.2f}mm  "
                  + "  ".join(f"{c}={srow[c]['ra_mm']:.2f}" for c in conds), flush=True)
        # Frame-weighted pooling, per sequence so FK uses each sequence's own betas.
        rows = {}
        for c in list(conds) + ["_knot_span"]:
            w = np.array([s[c]["n"] for s in per], np.float64)
            rows[c] = {"n": int(w.sum()),
                       "ra_mm": float(np.sum([s[c]["ra_mm"] * wi for s, wi in zip(per, w)]) / w.sum()),
                       "abs_mm": float(np.sum([s[c]["abs_mm"] * wi for s, wi in zip(per, w)]) / w.sum())}
        base = rows["prev_start"]["ra_mm"]
        for c, r in rows.items():
            r["delta_vs_prev_start_mm"] = r["ra_mm"] - base
        out["regimes"][reg] = rows
        for c, r in rows.items():
            print(f"[{reg}] {c:<12} RA={r['ra_mm']:7.3f} abs={r['abs_mm']:8.3f} "
                  f"d={r['delta_vs_prev_start_mm']:+7.3f}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
