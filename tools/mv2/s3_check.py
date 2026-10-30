#!/usr/bin/env python3
"""MV2 S3 stop rule for a GPU arm (docs/EXPERIMENT_RECORDS.md#mv2-prereg-20261026, M1 GPU arm).

On SELECT (every 8th endpoint of the four base recordings, W200, prev = GT(end - 50 ms)), for the arm's checkpoint and
S39 at the same step and seed: E3 RA (mm) and one-step carry of a root (8 deg) / finger (N(0, 0.10)) prev perturbation,
carry_probe's definitions (`tools/trk6/carry_probe.py`). Stop (do not continue to 6000) when E3 RA > S39 + 0.3 mm or a
carry > S39's + 0.05, or the arm's outputs are not finite. A pass only lets the run continue; it never releases a claim.

    CUDA_VISIBLE_DEVICES=<2..7> python tools/mv2/s3_check.py --arm-ckpt C --arm-config Y --step 2000 --seed 3407 --out D
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mv2 import common as C  # noqa: E402
from mv2.pa_probe import RA, aa_compose, gather  # noqa: E402

E3_TOL_MM, CARRY_TOL = 0.3, 0.05


@torch.no_grad()
def readings(model, samples, device):
    import copy
    ra = RA(copy.deepcopy(model).cpu())
    model.to(device)
    rows = {"e3": [], "carry_root_num": 0.0, "carry_root_den": 0.0, "carry_fing_num": 0.0, "carry_fing_den": 0.0}
    finite = True
    for ss in samples:
        idx = np.arange(len(ss))
        X = torch.from_numpy(ss.lnes(idx)).to(device)
        model.set_hand_context(torch.tensor(ss.betas).view(1, -1).to(device), torch.tensor(ss.K).view(1, 3, 3).to(device))
        rng = np.random.default_rng([20261026, int.from_bytes(ss.seq.encode()[:8].ljust(8, b"\0"), "little") % 2**31])
        P0 = ss.prev
        ax = rng.standard_normal((len(idx), 3))
        ax /= np.linalg.norm(ax, axis=1, keepdims=True)
        Pr = P0.copy()
        for i in range(len(idx)):
            Pr[i, 3:6] = aa_compose(ax[i] * np.deg2rad(8.0), P0[i, 3:6])
        Pf = P0.copy()
        Pf[:, 6:] += (rng.standard_normal((len(idx), 45)) * 0.10).astype(np.float32)

        def run(P):
            out = []
            for i0 in range(0, len(P), 64):
                out.append(model(X[i0:i0 + 64], torch.from_numpy(P[i0:i0 + 64]).to(device)).float().cpu().numpy())
            return np.concatenate(out)
        o0, o_r, o_f = run(P0), run(Pr), run(Pf)
        finite &= bool(np.isfinite(o0).all() and np.isfinite(o_r).all() and np.isfinite(o_f).all())
        B = np.repeat(ss.betas[None], len(idx), 0)
        j0, jg, jp0 = ra.joints(o0, B), ra.joints(ss.gt, B), ra.joints(P0, B)
        rows["e3"].append(np.linalg.norm(j0 - jg, axis=-1).mean(-1) * 1000)
        for name, P, O in (("root", Pr, o_r), ("fing", Pf, o_f)):
            din = (ra.joints(P, B) - jp0).reshape(len(idx), -1)
            dout = (ra.joints(O, B) - j0).reshape(len(idx), -1)
            rows[f"carry_{name}_num"] += float((din * dout).sum())
            rows[f"carry_{name}_den"] += float((din * din).sum())
    return {"e3_ra_mm": float(np.concatenate(rows["e3"]).mean()),
            "carry_root": rows["carry_root_num"] / rows["carry_root_den"],
            "carry_fingers": rows["carry_fing_num"] / rows["carry_fing_den"], "finite": finite}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-ckpt", required=True)
    ap.add_argument("--arm-config", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if not a.smoke:
        C.verify_protocol()
    dev = C.device_from_env()
    from config import load_config
    from mv2.slr_entry import patch
    SLR = patch()
    arm = SLR.load_from_checkpoint(a.arm_ckpt, cfg=load_config(a.arm_config), map_location=dev).eval()
    ref, _ = C.load_s39(a.seed, dev, step=a.step, cls=SLR)
    samples = gather(C.base_seqs(C.SELECT), 8, max_n=8 if a.smoke else None)
    r_arm, r_ref = readings(arm, samples, dev), readings(ref, samples, dev)
    stop = (not r_arm["finite"]) or r_arm["e3_ra_mm"] > r_ref["e3_ra_mm"] + E3_TOL_MM or \
        r_arm["carry_root"] > r_ref["carry_root"] + CARRY_TOL or r_arm["carry_fingers"] > r_ref["carry_fingers"] + CARRY_TOL
    res = {"arm": r_arm, "s39_same_step": r_ref, "step": a.step, "seed": a.seed, "stop": bool(stop),
           "rule": f"stop if E3 > S39 + {E3_TOL_MM} mm or carry > S39 + {CARRY_TOL} or non-finite"}
    if a.smoke:
        print("S3 smoke ok:", sorted(r_arm), "finite", r_arm["finite"])
        return
    C.save_json(Path(a.out) / "s3_check.json", res)
    print(("STOP" if stop else "CONTINUE"), flush=True)


if __name__ == "__main__":
    main()
