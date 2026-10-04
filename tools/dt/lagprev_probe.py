#!/usr/bin/env python3
"""DT2 zero-training probe: evidence window W > 50 ms with the conditioning state taken at the WINDOW START (the training
semantics: prev = state at end - W, predict the state at end), instead of 50 ms ago. The evaluated steps are unchanged
(a+49, a+99, ...); step k of a segment conditions on its own prediction k - W/50 steps earlier (the protocol's noisy
initial state while the window is still clipped to the segment start, exactly as training clips windows to the run start).
Bare model, mode `model` only. Writes outputs/dt2/lagprev/<run>_w<W>.json (+ npz).

    python tools/dt/lagprev_probe.py --run-dir outputs/semkine/dt_dz_l3_s3407 --window-ms 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "tracking")]
import evalx as EX                                                    # noqa: E402
from config import load_config                                        # noqa: E402
from mano_layer import ManoLayer                                      # noqa: E402
from model import MNISTModel                                          # noqa: E402
from semkine import eval_track as ET                                  # noqa: E402
from semkine import events as EV                                      # noqa: E402
from semkine.dataset import sequences_for_split                       # noqa: E402

STEP = EX.STEP


@torch.no_grad()
def run_sequence_lag(model, cfg, root, d, seq, device, rng, mode="model", wmode="fixed", win=STEP, min_events=0, max_win=300):
    assert mode == "model" and wmode == "fixed"
    events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    camera_K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    model.set_hand_context(betas, camera_K)
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
    ev_ch = EV.event_channels(cfg)
    preds, gts, ends_all, runs_all, elapsed, counts, prevs = [], [], [], [], [], [], []
    for run_id, (a, b) in enumerate(runs):
        ends = np.arange(a + STEP - 1, b, STEP, dtype=np.int64)
        if not len(ends):
            continue
        init = pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)           # same rng draws as evalx
        init_t = torch.from_numpy(init).view(1, -1).to(device)
        hist = {}                                                              # end -> prediction (this segment)
        for end in ends:
            end = int(end)
            w = int(min(win, end - a + 1))                                     # clipped to the segment, like training
            t_prev = end - w                                                   # the state the window starts from
            prev_t = hist.get(t_prev, init_t)
            x = torch.from_numpy(ET.build_lnes(events, offsets, end, w, ev_ch)).unsqueeze(0).to(device)
            pred = model(x, prev_t)
            hist[end] = pred
            prevs.append(prev_t.cpu().numpy()[0]); preds.append(pred.cpu().numpy()[0]); gts.append(pos51[end])
            ends_all.append(end); runs_all.append(run_id); elapsed.append(end - a)
            counts.append(int(offsets[end + 1] - offsets[end - STEP + 1]))
    return {"pred": np.stack(preds).astype(np.float32), "gt": np.stack(gts).astype(np.float32), "end": np.asarray(ends_all),
            "run": np.asarray(runs_all), "elapsed": np.asarray(elapsed), "count": np.asarray(counts), "betas": aux["betas"],
            "prev": np.stack(prevs).astype(np.float32), "fstate": []}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--window-ms", type=int, default=200)
    ap.add_argument("--out-dir", default=str(REPO / "outputs/dt2/lagprev"))
    cli = ap.parse_args()
    run = Path(cli.run_dir)
    cfg = load_config(run / "config_resolved.yaml")
    device = torch.device("cuda")
    ckpt, step, _ = EX.find_ckpt(run, "last")
    model = MNISTModel.load_from_checkpoint(str(ckpt), cfg=cfg, map_location=device).to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    seqs = sequences_for_split(root, "val_core", Path(cfg["DATA"]["SPLITS_MANIFEST"]))
    EX.run_sequence = run_sequence_lag
    a = argparse.Namespace(controls=False, tf=False, perturb=False, window_mode="fixed", window_ms=cli.window_ms,
                           min_events=0, max_window_ms=300)
    summary, arrays = EX.evaluate(model, cfg, mano, root, seqs, device, a, label=f"{run.name} lagprev w{cli.window_ms}")
    out_dir = Path(cli.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{run.name}_w{cli.window_ms}"
    np.savez_compressed(out_dir / f"{tag}.npz", **arrays)
    (out_dir / f"{tag}.json").write_text(json.dumps({"run": run.name, "ckpt": str(ckpt), "step": step, "window_ms": cli.window_ms,
                                                     "prev": "window start (own prediction W ms earlier)", **summary}, indent=1))
    m = summary["model"]
    print(f"{tag}: RA {m['overall']['mpjpe_ra_mm']:.3f} (g {m['zgz_global']['mpjpe_ra_mm'][0]:.2f} l {m['zgz_local']['mpjpe_ra_mm'][0]:.2f}) "
          f"rot {m['overall']['root_rot_deg']:.2f} acc_err_ra {m['jitter']['acc_err_ra_mm']:.2f} "
          f"root speed g/l {m['zgz_global']['motion']['root_speed_ratio']:.2f}/{m['zgz_local']['motion']['root_speed_ratio']:.2f}")


if __name__ == "__main__":
    main()
