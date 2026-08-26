#!/usr/bin/env python3
"""How much of the tracking error is a standing offset rather than tracking failure?

Two measurements from the echo-gain probe reframe the problem:

* the hand moves **2.9 mm** RA between consecutive 50 ms samples, while the single-step error under
  perfect conditioning is **12.1 mm** -- the motion to be tracked is four times below the
  estimator's own noise floor;
* the per-step residual is **91% autocorrelated** step to step, and 60% of its magnitude is the
  same vector all sequence long.

A standing offset is not amplified like noise: it is multiplied by the loop's geometric series
`1 / (1 - G)`, which the same probe measures at 0.75 for KEG and 0.61 for the dense arm. So a
constant few millimetres at the input becomes tens of millimetres at the output. This probe asks how
much is left if the standing part is removed, three ways, none of which is a method -- they are
upper bounds that say where the error lives:

    raw          the frozen recursive metric
    -mean        minus the per-sequence mean per-joint offset in the root-aligned frame
    -rigid       minus a per-sequence rigid alignment (Procrustes on the root-aligned joints)
    -scale       minus a per-sequence rigid + uniform scale (hand size / shape mismatch)

Run in both regimes: if the offset is already there under teacher forcing, it is the estimator's,
not the loop's, and no closed-loop training fixes it.

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
from semkine.dataset import sequences_for_split                   # noqa: E402


def _mpjpe(a, b):
    return float(np.linalg.norm(a - b, axis=-1).mean() * 1000)


def _procrustes(P, G, scale=False):
    """Best single rigid (optionally similarity) map from the whole prediction set onto the truth.

    Fitted once per sequence over every frame jointly, so it can only absorb a *constant* pose
    offset -- a per-frame fit would absorb the tracking error itself and prove nothing.
    """
    X = P.reshape(-1, 3)
    Y = G.reshape(-1, 3)
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    s = 1.0
    if scale:
        s = float((S * np.array([1, 1, d])).sum() / max((Xc ** 2).sum(), 1e-12))
    return ((Xc @ R) * s + ym).reshape(P.shape)


@torch.no_grad()
def rollout(model, mano, cfg, root, legacy_dir, seq, step_ms, device, teacher_forced, max_steps):
    events, offsets, aux, pos51 = ET.load_sequence(root, legacy_dir, seq)
    tsub_path = root / legacy_dir / f"{seq}_tsub.npy"
    tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
    use_raw = bool(getattr(model, "encoder_name", ""))
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    if hasattr(model, "set_hand_context"):
        model.set_hand_context(betas, K)
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)

    rng = np.random.default_rng(0)
    preds, gts = [], []
    n = 0
    for a, b in runs:
        ends = np.arange(a + step_ms - 1, b, step_ms, dtype=np.int64)
        if not len(ends):
            continue
        prev = pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)
        prev_end = int(a)
        for end in ends:
            if max_steps and n >= max_steps:
                break
            cond = pos51[prev_end].copy() if teacher_forced else prev
            t = torch.from_numpy(np.ascontiguousarray(cond, np.float32)).view(1, -1).to(device)
            if use_raw:
                ev5 = ET._window_events(events, offsets, tsub, int(end), step_ms)
                p = model.forward_packet(
                    ET.make_eval_packet(ev5, t, betas, K, step_ms, device)).cpu().numpy()[0]
            else:
                x = torch.from_numpy(
                    ET.build_lnes(events, offsets, int(end), step_ms)).unsqueeze(0).to(device)
                p = model(x, t).cpu().numpy()[0]
            preds.append(p)
            gts.append(pos51[int(end)])
            prev, prev_end = p, int(end)
            n += 1
        if max_steps and n >= max_steps:
            break

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

    return fk(np.stack(preds)), fk(np.stack(gts))


def analyse(pj, gj) -> dict:
    P, G = ET.root_align(pj), ET.root_align(gj)
    E = P - G
    bias = E.mean(0, keepdims=True)
    out = {
        "n": int(len(P)),
        "ra_raw_mm": _mpjpe(P, G),
        "ra_minus_mean_mm": _mpjpe(P - bias, G),
        "ra_minus_rigid_mm": _mpjpe(_procrustes(P, G), G),
        "ra_minus_scale_mm": _mpjpe(_procrustes(P, G, scale=True), G),
        "abs_raw_mm": _mpjpe(pj, gj),
        # Share of the squared residual that a single constant vector explains.
        "bias_energy_frac": float((bias ** 2).sum() * len(P) / max((E ** 2).sum(), 1e-12)),
        "bias_norm_mm": float(np.linalg.norm(bias, axis=-1).mean() * 1000),
        "gt_motion_mm": float(np.linalg.norm(G[1:] - G[:-1], axis=-1).mean() * 1000),
        "pred_motion_mm": float(np.linalg.norm(P[1:] - P[:-1], axis=-1).mean() * 1000),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="LABEL=CKPT:CONFIG")
    ap.add_argument("--route", default="soft")
    ap.add_argument("--split", default="val_core")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.environ["EVENTHANDS_KEG_ROUTE"] = a.route
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {"step_ms": a.step_ms, "split": a.split, "arms": {}}
    for spec in a.arm:
        label, rest = spec.split("=", 1)
        ckpt, cfg_path = rest.rsplit(":", 1)
        cfg = load_config(cfg_path)
        model = MNISTModel.load_from_checkpoint(ckpt, cfg=cfg, map_location=device).to(device).eval()
        mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
        root = Path(cfg["DATA"]["ROOT"])
        row = {"ckpt": ckpt}
        for reg in ("teacher_forced", "recursive"):
            per = []
            for seq, d in sequences_for_split(root, a.split, None):
                pj, gj = rollout(model, mano, cfg, root, d, seq, a.step_ms, device,
                                 reg == "teacher_forced", a.max_steps)
                s = analyse(pj, gj)
                s["seq"] = seq
                per.append(s)
                print(f"  [{reg}] {seq}: raw={s['ra_raw_mm']:.2f} -mean={s['ra_minus_mean_mm']:.2f} "
                      f"-rigid={s['ra_minus_rigid_mm']:.2f} -scale={s['ra_minus_scale_mm']:.2f} "
                      f"biasE={s['bias_energy_frac']:.2f} gtmove={s['gt_motion_mm']:.2f}", flush=True)
            w = np.array([p["n"] for p in per], np.float64)
            pooled = {k: float(np.sum([p[k] * wi for p, wi in zip(per, w)]) / w.sum())
                      for k in per[0] if isinstance(per[0][k], float)}
            row[reg] = {"per_sequence": per, "pooled": pooled}
            print(f"[{reg}] {label}: " + "  ".join(
                f"{k}={pooled[k]:.3f}" for k in
                ("ra_raw_mm", "ra_minus_mean_mm", "ra_minus_rigid_mm",
                 "ra_minus_scale_mm", "bias_energy_frac")), flush=True)
        result["arms"][label] = row

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
