#!/usr/bin/env python3
"""trk6 diagnostics on stored evalx/filter_eval trajectories (no new inference).

For every mode stored in the npz (model / raw / hold / noevents / tf ...) and sequence:
  - RA-MPJPE / RA-MPVPE by segment age bucket (s since the segment start),
  - oracle substitutions (diagnostic only, never a result): root rotation <- GT, fingers <- GT,
  - segment length histogram.
    python tools/trk6/diag_traj.py NPZ [NPZ ...] [--out out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))
from mano_layer import ManoLayer                                      # noqa: E402
from semkine.eval_track import mano_fk                                # noqa: E402

AGE_EDGES_S = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9)


DATA = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51")


def betas_of(seq):
    from semkine.eval_track import load_sequence
    from semkine.dataset import sequences_for_split
    d = dict(sequences_for_split(DATA, "val_core"))[seq]
    return np.asarray(load_sequence(DATA, d, seq)[2]["betas"], np.float32)


def ra_err(mano, p, g, betas, dev):
    pj, pv = mano_fk(mano, p, betas, dev)
    gj, gv = mano_fk(mano, g, betas, dev)
    ra = lambda x: x - x[..., :1, :]                                            # noqa: E731
    return (np.linalg.norm(ra(pj) - ra(gj), axis=-1).mean(-1) * 1000,
            np.linalg.norm(ra(pv) - ra(gv), axis=-1).mean(-1) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    dev = torch.device("cpu")
    mano = ManoLayer(str(REPO / "assets/mano_right.npz"), add_mean=False).to(dev).eval()
    out = {}
    for f in a.npz:
        z = np.load(f, allow_pickle=True)
        keys = sorted({k.rsplit("|", 1)[0] for k in z.files})
        res = {}
        for mk in keys:
            mode, seq = mk.split("|")
            if f"{mk}|pred" not in z.files:
                continue
            betas = torch.from_numpy(betas_of(seq)).view(1, -1)
            p, g = z[f"{mk}|pred"].astype(np.float32), z[f"{mk}|gt"].astype(np.float32)
            el = z[f"{mk}|elapsed"] / 1000.0
            run = z[f"{mk}|run"]
            j, v = ra_err(mano, p, g, betas, dev)
            r = {"n": int(len(p)), "mpjpe_ra": float(j.mean()), "mpvpe_ra": float(v.mean())}
            for name, sl in (("rootGT", slice(3, 6)), ("fingGT", slice(6, 51))):
                q = p.copy()
                q[:, sl] = g[:, sl]
                jj, vv = ra_err(mano, q, g, betas, dev)
                r[f"{name}_mpjpe_ra"] = float(jj.mean())
                r[f"{name}_mpvpe_ra"] = float(vv.mean())
            ages = {}
            for lo, hi in zip(AGE_EDGES_S[:-1], AGE_EDGES_S[1:]):
                s = (el >= lo) & (el < hi)
                if s.any():
                    ages[f"[{lo},{hi})"] = {"n": int(s.sum()), "mpjpe_ra": float(j[s].mean()), "mpvpe_ra": float(v[s].mean())}
            r["by_age"] = ages
            lens = np.bincount(run)
            lens = lens[lens > 0] * 0.05
            r["seg"] = {"n": int(len(lens)), "len_s_mean": float(lens.mean()), "len_s_median": float(np.median(lens)),
                        "len_s_min": float(lens.min()), "len_s_max": float(lens.max())}
            res[mk] = r
        out[f] = res
        for mk, r in res.items():
            print(f"{Path(f).parent.name} {mk:22s} J={r['mpjpe_ra']:.2f} V={r['mpvpe_ra']:.2f} "
                  f"rootGT J={r['rootGT_mpjpe_ra']:.2f} fingGT J={r['fingGT_mpjpe_ra']:.2f} seg={r['seg']}")
            print("     age:", "  ".join(f"{k}:{v['mpjpe_ra']:.2f}({v['n']})" for k, v in r["by_age"].items()))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
