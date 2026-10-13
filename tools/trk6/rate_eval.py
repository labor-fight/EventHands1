#!/usr/bin/env python3
"""trk6: the S39 stack run at a faster output rate (e.g. 10 ms steps = 100 Hz), scored at the official 50 ms timestamps.

The loop is `tools/tracking/evalx.py run_sequence` (model mode, W200 clipped to the segment start, protocol initial noise
from one rng seeded 0 across the split), with the step as a parameter. Only segments that have at least one official
50 ms step are tracked (the same segments and the same rng draws as the official evaluation), and the error is scored at
the official ends a+49, a+99, ... (the 50 ms grid is a subset of the 10 ms grid), so the numbers compare with the main
table directly; step = 50 reproduces the official evaluation bit for bit. The filter is fed the event count of the last
50 ms (clipped to the segment start and rescaled to a 50 ms rate), as the ramp1000 gate is calibrated in that unit.
Filter variants:
  same     the S39 spec unchanged, applied at every step (its gains were calibrated for 50 ms steps)
  per-step the spec's gains converted to the step, a' = 1 - (1 - a)^(step / 50): the same time constant for a held target
Reported: RA-MPJPE / RA-MPVPE (50 ms grid), acc_err_ra on the 50 ms grid (jitter, comparable to S39), velocity
cross-correlation lag in ms (on the step grid), displacement gain over 250 ms.
    python tools/trk6/rate_eval.py --run-dir outputs/semkine/dt_dz_l3_s3407 --step 10 --out F.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "model")); sys.path.insert(0, str(REPO / "tools"))
from config import load_config                                        # noqa: E402
from mano_layer import ManoLayer                                      # noqa: E402
from model import MNISTModel                                          # noqa: E402
from semkine import eval_track as ET                                  # noqa: E402
from semkine import events as EV                                      # noqa: E402
from semkine.anchored import AdaptiveFilter                           # noqa: E402
from semkine.dataset import sequences_for_split                       # noqa: E402
from semkine.eval_track import mano_fk                                # noqa: E402
from tracking.evalx import find_ckpt                                  # noqa: E402

SPEC = REPO / "outputs/dt3/gate_specs/ramp1000.json"
GRID = 50


def convert(spec, step):
    """Per-step gains with the 50 ms schedule's time constant: a' = 1 - (1 - a)^(step / 50)."""
    f = lambda g: 1.0 - (1.0 - float(g)) ** (step / GRID)                     # noqa: E731
    out = dict(spec)
    for k in ("a_root", "a_rest", "a_trans"):
        v = spec[k]
        out[k] = f(v) if isinstance(v, (int, float)) else [[n, f(g)] for n, g in v]
    return out


@torch.no_grad()
def track(net, cfg, root, seqs, dev, step, window):
    ev_ch = EV.event_channels(cfg)
    rng = np.random.default_rng(0)
    res = {}
    for seq, d in seqs:
        events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
        net.set_hand_context(torch.tensor(aux["betas"], dtype=torch.float32, device=dev).view(1, -1),
                             torch.tensor(aux["camera_K"], dtype=torch.float32, device=dev).view(1, 3, 3))
        P, G, E, R, ON = [], [], [], [], []
        for run_id, (a, b) in enumerate(np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)):
            if not len(np.arange(a + GRID - 1, b, GRID)):
                continue                                   # the official evaluation skips it (no rng draw either)
            prev = torch.from_numpy(pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)).view(1, -1).to(dev)
            net.reset_state(prev)
            for end in np.arange(a + step - 1, b, step, dtype=np.int64):
                end = int(end)
                lo = max(int(a), end - GRID + 1)
                n = float(offsets[end + 1] - offsets[lo]) * GRID / (end - lo + 1)
                w = min(window, end + 1 - int(a))
                x = torch.from_numpy(ET.build_lnes(events, offsets, end, w, ev_ch)).unsqueeze(0).to(dev)
                prev = net(x, prev, n_events=n if step != GRID else int(offsets[end + 1] - offsets[lo]))
                P.append(prev.cpu().numpy()[0]); G.append(np.asarray(pos51[end], np.float32))
                E.append(end); R.append(run_id); ON.append((end - int(a) + 1) % GRID == 0)
        res[seq] = {"pred": np.stack(P).astype(np.float32), "gt": np.stack(G), "end": np.asarray(E), "run": np.asarray(R),
                    "on": np.asarray(ON), "betas": np.asarray(aux["betas"], np.float32)}
    return res


def metrics(mano, r, step):
    dev = torch.device("cpu")
    b = torch.from_numpy(r["betas"]).view(1, -1)
    pj, pv = mano_fk(mano, r["pred"], b, dev)
    gj, gv = mano_fk(mano, r["gt"], b, dev)
    ra = lambda x: x - x[..., :1, :]                                          # noqa: E731
    on, run = r["on"], r["run"]
    J = np.linalg.norm(ra(pj) - ra(gj), axis=-1).mean(-1) * 1000
    V = np.linalg.norm(ra(pv) - ra(gv), axis=-1).mean(-1) * 1000
    # jitter on the 50 ms grid: acceleration of the root-aligned joint error (evalx.jitter_decomp acc_err_ra_mm)
    e = (ra(pj) - ra(gj))[on]
    ro = run[on]
    s2 = (ro[2:] == ro[1:-1]) & (ro[1:-1] == ro[:-2])
    acc = np.linalg.norm(e[2:] - 2 * e[1:-1] + e[:-2], axis=-1).mean(-1)[s2].mean() * 1000
    # lag (velocity cross-correlation, step grid) and 250 ms displacement gain
    P, Gr = ra(pj), ra(gj)
    same = run[1:] == run[:-1]
    vp = (P[1:] - P[:-1]).reshape(len(P) - 1, -1); vg = (Gr[1:] - Gr[:-1]).reshape(len(Gr) - 1, -1)
    vp = vp - vp[same].mean(0); vg = vg - vg[same].mean(0)
    kmax = max(2, 300 // step)
    ks = np.arange(-kmax // 2, kmax + 1)
    xc = []
    for k in ks:
        if k >= 0:
            m = same[k:] & same[:len(same) - k] if k else same
            aa, bb = vp[k:], vg[:len(vg) - k]
        else:
            m = same[:len(same) + k] & same[-k:]
            aa, bb = vp[:len(vp) + k], vg[-k:]
        xc.append((aa[m] * bb[m]).sum() / np.sqrt((aa[m] ** 2).sum() * (bb[m] ** 2).sum()))
    xc = np.asarray(xc)
    i = int(xc.argmax())
    lag = float(ks[i])
    if 0 < i < len(ks) - 1:
        y0, y1, y2 = xc[i - 1], xc[i], xc[i + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            lag = float(ks[i] + 0.5 * (y0 - y2) / den)
    k5 = 250 // step
    mk = np.r_[np.zeros(k5, bool), run[k5:] == run[:-k5]]
    dP = (P[k5:] - P[:-k5])[mk[k5:]].reshape(-1); dG = (Gr[k5:] - Gr[:-k5])[mk[k5:]].reshape(-1)
    return {"n50": int(on.sum()), "mpjpe_ra": float(J[on].mean()), "mpvpe_ra": float(V[on].mean()),
            "acc_err_ra_50": float(acc), "lag_ms": lag * step, "gain_250ms": float(dP @ dG / (dG @ dG))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="step=6000")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--window-ms", type=int, default=200)
    ap.add_argument("--variants", default="same,per-step")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert GRID % a.step == 0
    dev = torch.device("cuda")
    run = Path(a.run_dir)
    cfg = load_config(next(iter(sorted(run.glob("*.yaml")))))
    ck, _, _ = find_ckpt(run, a.ckpt)
    model = MNISTModel.load_from_checkpoint(str(ck), cfg=cfg, map_location=dev).to(dev).eval()
    mano = ManoLayer(str(REPO / "assets/mano_right.npz"), add_mean=False).eval()
    root = Path(cfg["DATA"]["ROOT"])
    seqs = sequences_for_split(root, "val_core")
    base = AdaptiveFilter.canonical_spec(json.loads(SPEC.read_text()))
    out = {"run": run.name, "step": a.step, "variants": {}}
    for v in a.variants.split(","):
        spec = base if v == "same" else convert(base, a.step)
        net = AdaptiveFilter.from_spec(model, AdaptiveFilter.canonical_spec(spec)).to(dev).eval()
        res = track(net, cfg, root, seqs, dev, a.step, a.window_ms)
        out["variants"][v] = {seq: metrics(mano, r, a.step) for seq, r in res.items()}
        m = out["variants"][v]
        print(f"{run.name} step {a.step} ms filter {v:8s} | local J {m['zgz_local']['mpjpe_ra']:.3f} V {m['zgz_local']['mpvpe_ra']:.3f} "
              f"acc50 {m['zgz_local']['acc_err_ra_50']:.3f} lag {m['zgz_local']['lag_ms']:+.1f} ms gain250 {m['zgz_local']['gain_250ms']:.3f} | "
              f"global J {m['zgz_global']['mpjpe_ra']:.3f} V {m['zgz_global']['mpvpe_ra']:.3f} acc50 {m['zgz_global']['acc_err_ra_50']:.3f} "
              f"lag {m['zgz_global']['lag_ms']:+.1f} ms", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
