#!/usr/bin/env python3
"""trk6: how the S39 stack's accuracy depends on its initial state (deployment question, not a training arm).

The full S39 stack (AdaptiveFilter ramp1000, W200 clipped to the segment start, last-50 ms count) on zgz, every segment
started from a chosen initial state instead of the protocol's GT(a) + N(0, 5 mm / 0.05 rad / 0.05 rad):
  scale k   GT(a) + k x the protocol noise (k = 0, 1, 2, 4; k = 1 reproduces the official numbers bit for bit)
  depth+30  the protocol noise plus +30 mm on the initial depth (a typical monocular RGB initialiser bias)
  beta0     the protocol noise, but the tracker renders with the MANO mean shape (betas = 0) instead of the sequence's
            GT shape, and its output mesh is built with betas = 0 too (as a live system without a shape calibration)
Reported per condition: the initial state's own error (RA-MPJPE of GT(a)+noise, absolute translation, root angle) and
the tracking RA-MPJPE / RA-MPVPE per sequence, plus local RA by segment age.
    python tools/trk6/init_sens.py --run-dir outputs/semkine/dt_dz_l3_s3407 --ckpt step=6000 --out F.json
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
from semkine.anchored import AdaptiveFilter                           # noqa: E402
from semkine.dataset import sequences_for_split                       # noqa: E402
from semkine.eval_track import mano_fk                                # noqa: E402
from tracking import evalx                                            # noqa: E402
from tracking.evalx import find_ckpt, rot_err_deg                     # noqa: E402

SPEC = REPO / "outputs/dt3/gate_specs/ramp1000.json"
ORIG_NOISE = ET.sample_init_noise
ORIG_LOAD = ET.load_sequence
AGES = (0.0, 0.5, 1.0, 2.0, 4.0, 1e9)


def ra(x):
    return x - x[..., :1, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="step=6000")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    run = Path(a.run_dir)
    cfg = load_config(next(iter(sorted(run.glob("*.yaml")))))
    ck, step, _ = find_ckpt(run, a.ckpt)
    model = MNISTModel.load_from_checkpoint(str(ck), cfg=cfg, map_location=dev).to(dev).eval()
    net = AdaptiveFilter.from_spec(model, AdaptiveFilter.canonical_spec(json.loads(SPEC.read_text()))).to(dev).eval()
    mano = ManoLayer(str(REPO / "assets/mano_right.npz"), add_mean=False).to(dev).eval()
    root = Path(cfg["DATA"]["ROOT"])
    seqs = sequences_for_split(root, "val_core")
    conds = {"scale0": (0.0, 0.0, False), "scale1": (1.0, 0.0, False), "scale2": (2.0, 0.0, False),
             "scale4": (4.0, 0.0, False), "depth+30": (1.0, 0.030, False), "beta0": (1.0, 0.0, True)}
    out = {"run": run.name, "step": step, "conds": {}}
    for name, (k, dz, beta0) in conds.items():
        def noise(cfg_, rng_, scale, _k=k, _dz=dz):
            v = ORIG_NOISE(cfg_, rng_, scale * _k) if _k > 0 else (ORIG_NOISE(cfg_, rng_, scale) * 0.0)
            v = v.copy()
            v[2] += _dz
            return v

        def load(root_, d_, seq_, _b=beta0):
            ev, off, aux, pos = ORIG_LOAD(root_, d_, seq_)
            if not _b:
                return ev, off, aux, pos
            aux = {key: aux[key] for key in aux.files}
            aux["gt_betas"] = aux["betas"].copy()
            aux["betas"] = np.zeros_like(aux["betas"])
            return ev, off, aux, pos

        ET.sample_init_noise, ET.load_sequence = noise, load
        rng = np.random.default_rng(0)                    # one stream across the split, as evalx.evaluate
        res = {}
        with torch.no_grad():
            for seq, d in seqs:
                r = evalx.run_sequence(net, cfg, root, d, seq, dev, rng, mode="model", wmode="fixed", win=200,
                                       clip_run_start=True, count_mode="last50")
                aux = ORIG_LOAD(root, d, seq)[2]
                b_gt = torch.tensor(aux["betas"], dtype=torch.float32, device=dev).view(1, -1)
                b_pred = torch.zeros_like(b_gt) if beta0 else b_gt
                pj, pv = mano_fk(mano, r["pred"], b_pred, dev)
                gj, gv = mano_fk(mano, r["gt"], b_gt, dev)
                J = np.linalg.norm(ra(pj) - ra(gj), axis=-1).mean(-1) * 1000
                V = np.linalg.norm(ra(pv) - ra(gv), axis=-1).mean(-1) * 1000
                first = np.r_[True, r["run"][1:] != r["run"][:-1]]
                init = r["prev"][first]
                ij, _ = mano_fk(mano, init, b_pred, dev)
                pos51 = ORIG_LOAD(root, d, seq)[3]
                g0 = np.stack([np.asarray(pos51[int(e) - int(el)], np.float32)          # GT at the segment start a
                               for e, el in zip(r["end"][first], r["elapsed"][first])])
                gj0, _ = mano_fk(mano, g0, b_gt, dev)
                el = r["elapsed"] / 1000.0
                res[seq] = {"mpjpe_ra": float(J.mean()), "mpvpe_ra": float(V.mean()),
                            "init_mpjpe_ra": float((np.linalg.norm(ra(ij) - ra(gj0), axis=-1).mean(-1) * 1000).mean()),
                            "init_transl_mm": float((np.linalg.norm(init[:, :3] - g0[:, :3], axis=-1) * 1000).mean()),
                            "init_root_deg": float(rot_err_deg(init, g0).mean()),
                            "by_age": {f"[{lo},{hi})": float(J[(el >= lo) & (el < hi)].mean())
                                       for lo, hi in zip(AGES[:-1], AGES[1:]) if ((el >= lo) & (el < hi)).any()}}
        ET.sample_init_noise, ET.load_sequence = ORIG_NOISE, ORIG_LOAD
        out["conds"][name] = res
        g, l = res["zgz_global"], res["zgz_local"]
        print(f"{run.name} {name:9s} init RA {l['init_mpjpe_ra']:5.2f} mm / T {l['init_transl_mm']:5.1f} mm / "
              f"root {l['init_root_deg']:5.2f} deg | local J {l['mpjpe_ra']:.3f} V {l['mpvpe_ra']:.3f} | "
              f"global J {g['mpjpe_ra']:.3f} V {g['mpvpe_ra']:.3f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
