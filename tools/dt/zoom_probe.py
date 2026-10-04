#!/usr/bin/env python3
"""DT2 zero-training probe: evaluate a dense tracker with the zgz events ZOOMED by `s` about the principal point, exactly the
depth-mode scale augmentation of training (semkine.domrand: events `q -> s (q - c) + c`, labels: root camera depth / s, K
unchanged). The tracker runs in the zoomed world (its state carries the zoomed depth); every prediction is mapped back
(depth * s) before scoring against the untouched ground truth. Tests whether zgz_local's small, far hand (718 mm, past the
training depths) is limited by resolution / depth-OOD. Bare model, 50 ms window, mode `model` only.

    python tools/dt/zoom_probe.py --run-dir outputs/semkine/dt_dz_l3_s3407 --scale 1.25
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
from semkine import domrand as DR                                     # noqa: E402
from semkine import eval_track as ET                                  # noqa: E402
from semkine import events as EV                                      # noqa: E402
from semkine.dataset import mano_root_joint, sequences_for_split      # noqa: E402

STEP = EX.STEP
SCALE = 1.0
RENDER_SCALE = 0.375
MANO_NPZ = None


def zoom_sample():
    return DR.DomRandSample(0.0, float(SCALE), (0.0, 0.0), 1.0, 0, "depth")


def zoom_state(x51: np.ndarray, K: np.ndarray, j0: np.ndarray) -> np.ndarray:
    out, _ = DR.transform_labels(np.stack([x51, x51]).astype(np.float32), zoom_sample(), K, j0, RENDER_SCALE)
    return out[0]


def unzoom_state(x51: np.ndarray, j0: np.ndarray) -> np.ndarray:
    y = np.asarray(x51, np.float64).copy()
    y[2] = (float(j0[2]) + y[2]) * SCALE - float(j0[2])
    return y.astype(np.float32)


@torch.no_grad()
def run_sequence_zoom(model, cfg, root, d, seq, device, rng, mode="model", wmode="fixed", win=STEP, min_events=0, max_win=300):
    assert mode == "model" and wmode == "fixed"
    events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
    K = np.asarray(aux["camera_K"], np.float64).reshape(3, 3)
    j0 = mano_root_joint(np.asarray(aux["betas"], np.float32), MANO_NPZ)
    chk = zoom_state(pos51[0], K, j0)
    assert np.allclose(unzoom_state(chk, j0), pos51[0], atol=1e-5), "inverse of the depth-mode zoom"
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
        init = pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)
        prev_z = zoom_state(init, K, j0)
        prev_t = torch.from_numpy(prev_z).view(1, -1).to(device)
        for end in ends:
            end = int(end)
            w = int(min(win, end + 1))
            start = end - w + 1
            a0, a1 = int(offsets[start]), int(offsets[end + 1])
            if a1 > a0:
                ev = np.asarray(events[a0:a1])
                xs, ys = ev[:, 0].astype(np.int64), ev[:, 1].astype(np.int64)
                keep = DR.transform_events(xs, ys, zoom_sample(), K, RENDER_SCALE, ET.W, ET.H)
                ms_rel = np.repeat(np.arange(w, dtype=np.float32), np.diff(offsets[start:end + 2]).astype(np.int64))
                lnes = EV.splat_event_image(xs[keep], ys[keep], np.clip(ev[:, 2], 0, 1)[keep], ms_rel[keep], w, ET.H, ET.W, ev_ch)
            else:
                lnes = np.zeros((ET.H, ET.W, 2 * len(ev_ch)), np.float32)
            x = torch.from_numpy(lnes).unsqueeze(0).to(device)
            pred = model(x, prev_t)
            prevs.append(unzoom_state(prev_t.cpu().numpy()[0], j0))
            preds.append(unzoom_state(pred.cpu().numpy()[0], j0))
            prev_t = pred
            gts.append(pos51[end]); ends_all.append(end); runs_all.append(run_id); elapsed.append(end - a)
            counts.append(int(offsets[end + 1] - offsets[end - STEP + 1]))
    return {"pred": np.stack(preds).astype(np.float32), "gt": np.stack(gts).astype(np.float32), "end": np.asarray(ends_all),
            "run": np.asarray(runs_all), "elapsed": np.asarray(elapsed), "count": np.asarray(counts), "betas": aux["betas"],
            "prev": np.stack(prevs).astype(np.float32), "fstate": []}


def main() -> None:
    global SCALE, MANO_NPZ
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--scale", type=float, default=1.25)
    ap.add_argument("--out-dir", default=str(REPO / "outputs/dt2/zoom"))
    cli = ap.parse_args()
    SCALE = cli.scale
    run = Path(cli.run_dir)
    cfg = load_config(run / "config_resolved.yaml")
    MANO_NPZ = cfg["MANO"]["NPZ"]
    device = torch.device("cuda")
    ckpt, step, _ = EX.find_ckpt(run, "last")
    model = MNISTModel.load_from_checkpoint(str(ckpt), cfg=cfg, map_location=device).to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    seqs = sequences_for_split(root, "val_core", Path(cfg["DATA"]["SPLITS_MANIFEST"]))
    EX.run_sequence = run_sequence_zoom
    a = argparse.Namespace(controls=False, tf=False, perturb=False, window_mode="fixed", window_ms=STEP, min_events=0, max_window_ms=300)
    summary, arrays = EX.evaluate(model, cfg, mano, root, seqs, device, a, label=f"{run.name} zoom {SCALE:g}")
    out_dir = Path(cli.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{run.name}_z{SCALE:g}"
    np.savez_compressed(out_dir / f"{tag}.npz", **arrays)
    (out_dir / f"{tag}.json").write_text(json.dumps({"run": run.name, "ckpt": str(ckpt), "step": step, "zoom": SCALE, **summary}, indent=1))
    m = summary["model"]
    print(f"{tag}: RA {m['overall']['mpjpe_ra_mm']:.3f} (g {m['zgz_global']['mpjpe_ra_mm'][0]:.2f} l {m['zgz_local']['mpjpe_ra_mm'][0]:.2f}) "
          f"rot {m['overall']['root_rot_deg']:.2f} abs {m['overall']['mpjpe_abs_mm']:.1f} acc_err_ra {m['jitter']['acc_err_ra_mm']:.2f}")


if __name__ == "__main__":
    main()
