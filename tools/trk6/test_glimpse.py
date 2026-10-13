#!/usr/bin/env python3
"""trk6 / S42 debug gate for MODEL.GLIMPSE (CPU, real zgz + train frames, GT previous state):

1. geometry: the crop render equals the full-frame render resampled with the same grid (nearest) on >= 97 % of pixels;
2. coverage: share of the window's events inside the hand's full-frame box (GT mesh bbox + 8 px) that land in the crop;
3. support: share of crop pixels lit by the render / by events; event-render overlap (events near the splat);
4. a PNG panel per frame (LNES crop, render crop, overlay) under outputs/trk6/debug/.
    python tools/trk6/test_glimpse.py [--size 112] [--kappa 1.5] [--span 0.2]
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "model")); sys.path.insert(0, str(REPO / "tools"))
from config import load_config                                        # noqa: E402
from model import MNISTModel                                          # noqa: E402
from semkine import eval_track as ET                                  # noqa: E402
from semkine import events as EV                                      # noqa: E402
from semkine.dataset import sequences_for_split                       # noqa: E402

DATA = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--kappa", type=float, default=1.5)
    ap.add_argument("--span", type=float, default=0.20)
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    torch.manual_seed(0)
    cfg = load_config(str(REPO / "configs/dt/dt_dz_l3.yaml"))
    cfg_g = copy.deepcopy(cfg)
    cfg_g["MODEL"].update({"GLIMPSE": a.size, "GLIMPSE_KAPPA": a.kappa, "GLIMPSE_SPAN_M": a.span})
    m = MNISTModel(cfg_g).eval()
    ev_ch = EV.event_channels(cfg)
    out_dir = REPO / "outputs/trk6/debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = sequences_for_split(DATA, "val_core") + [s for s in sequences_for_split(DATA, "train") if s[0] in ("ycy_global", "lfz_local")]
    stats = []
    rng = np.random.default_rng(0)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    for seq, d in seqs:
        events, offsets, aux, pos51 = ET.load_sequence(DATA, d, seq)
        runs = np.asarray(aux["valid_runs_ms"]).reshape(-1, 2)
        betas = torch.tensor(aux["betas"], dtype=torch.float32).view(1, -1)
        K = torch.tensor(aux["camera_K"], dtype=torch.float32).view(1, 3, 3)
        cand = np.concatenate([np.arange(r0 + 199, r1, 50) for r0, r1 in runs if r1 - r0 > 250])
        for end in rng.choice(cand, size=min(a.n // len(seqs) + 1, len(cand)), replace=False):
            prev = torch.from_numpy(np.asarray(pos51[end - 50], np.float32)).view(1, -1)
            lnes = torch.from_numpy(ET.build_lnes(events, offsets, int(end), 200, ev_ch)).unsqueeze(0)
            with torch.no_grad():
                rend_c, grid = m._glimpse_inputs(prev, betas, K)
                full = m._render_prev(prev, betas, K)                  # (1, 180, 240, 2), full-frame splat
                full_s = F.max_pool2d(F.grid_sample(full.permute(0, 3, 1, 2), grid, mode="nearest", align_corners=False), 2)
                ev_c = m.glimpse_lnes(lnes, grid)
                verts, _ = m._fk(prev, betas)
                fx, fy, cx, cy = m._intrinsics(K)
                u = (fx[:, None] * verts[..., 0] / verts[..., 2] + cx[:, None])[0]
                v = (fy[:, None] * verts[..., 1] / verts[..., 2] + cy[:, None])[0]
            sil_c = rend_c[0, ..., 0] > 0
            sil_s = full_s[0, 0] > 0
            # crop splat vs resampled full splat: both are point sets; compare dilated supports
            dil = lambda t: F.max_pool2d(t.float()[None, None], 3, 1, 1)[0, 0] > 0           # noqa: E731
            agree = float(((dil(sil_c) == dil(sil_s))).float().mean())
            # event coverage of the hand box
            evf = lnes[0].abs().sum(-1) > 0                           # (180, 240)
            x0, x1 = int(max(u.min() - 8, 0)), int(min(u.max() + 8, 239))
            y0, y1 = int(max(v.min() - 8, 0)), int(min(v.max() + 8, 179))
            n_box = int(evf[y0:y1 + 1, x0:x1 + 1].sum())
            evc = ev_c[0].abs().sum(0) > 0
            # events in the crop counted on the full grid: map crop pixels back (unique full pixels)
            geo = m.glimpse_geometry(verts, K)
            uc, vc, hs = float(geo["uc"][0]), float(geo["vc"][0]), float(geo["side"][0]) / 2
            yy, xx = torch.meshgrid(torch.arange(180.0), torch.arange(240.0), indexing="ij")
            cov_mask = (xx >= uc - hs) & (xx <= uc + hs) & (yy >= vc - hs) & (yy <= vc + hs)
            box = torch.zeros(180, 240, dtype=torch.bool)
            box[y0:y1 + 1, x0:x1 + 1] = True
            cover = float((evf & box & cov_mask).sum()) / max(n_box, 1)
            near = float((evc & dil(dil(sil_c))).sum()) / max(int(evc.sum()), 1)
            side = float(m.glimpse_geometry(verts, K)["side"][0])
            stats.append({"seq": seq, "end": int(end), "agree": agree, "cover": cover, "lit_render": float(sil_c.float().mean()),
                          "lit_ev": float(evc.float().mean()), "ev_near_render": near, "side_px": side,
                          "hand_px": float(max(u.max() - u.min(), v.max() - v.min()))})
            if plt is not None and len(stats) <= 8:
                fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
                ax[0].imshow(lnes[0].abs().sum(-1) > 0, cmap="gray"); ax[0].add_patch(plt.Rectangle(
                    (uc - hs, vc - hs), 2 * hs, 2 * hs, fill=False, ec="r"))
                ax[0].set_title(f"{seq} @{end} full frame")
                ax[1].imshow(ev_c[0].abs().sum(0) > 0, cmap="gray"); ax[1].set_title("LNES crop")
                ax[2].imshow(rend_c[0, ..., 1], cmap="viridis"); ax[2].set_title("render crop (inv)")
                ov = np.zeros((a.size, a.size, 3)); ov[..., 0] = (ev_c[0].abs().sum(0) > 0).numpy(); ov[..., 1] = sil_c.numpy()
                ax[3].imshow(ov); ax[3].set_title("events(R) / render(G)")
                for x in ax: x.axis("off")
                fig.tight_layout(); fig.savefig(out_dir / f"glimpse_{len(stats):02d}_{seq}_{end}.png", dpi=80); plt.close(fig)
    keys = ("agree", "cover", "lit_render", "lit_ev", "ev_near_render", "side_px", "hand_px")
    for k in keys:
        v = np.array([s[k] for s in stats])
        print(f"{k:15s} mean {v.mean():.3f} min {v.min():.3f} max {v.max():.3f}")
    worst = sorted(stats, key=lambda s: s["cover"])[:3]
    print("lowest coverage:", [(s["seq"], s["end"], round(s["cover"], 3)) for s in worst])
    ok = np.mean([s["agree"] for s in stats]) >= 0.97 and np.min([s["cover"] for s in stats]) >= 0.90
    print("GATE", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
