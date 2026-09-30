#!/usr/bin/env python3
"""S37 root innovation, stage 1: the pre-registered mechanism gates (docs/S37_ROOT_INNOVATION_PREREG.md §3).

On each seed's selected checkpoint, zgz protocol (val_core, 50 ms), nothing trains:

  G1  teacher-forced rotation error with a clean prev (GT at the packet start), mean over all zgz
      packets, degrees                                                     -> <= 3.0 per seed
  G2  correction gain of a 10 deg prev rotation about each camera axis (composed about the wrist,
      events fixed): 1 - (share of the perturbation the output keeps), mean over x/y/z and every
      4th non-empty packet; full model and with the innovation features zeroed
                                                                           -> full >= 0.5 and
                                                                              ablated <= 0.2 x full, per seed
  G3  closed-loop RA with the innovation zeroed minus normal (the make_s36_row rng protocol)
                                                                           -> two-seed mean >= 1.5 mm
The accuracy gate G4 is the main row (`tools/make_s36_row.py --run s37_rootinnov`).

    CUDA_VISIBLE_DEVICES=4 python tools/probe_s37_rootinnov.py \\
        --runs outputs/semkine/s37_rootinnov_s3407 outputs/semkine/s37_rootinnov_s3408 \\
        --out outputs/semkine/probe_s37_rootinnov.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))
sys.path.insert(0, str(REPO / "tools"))

from probe_s37_route import load_run                              # noqa: E402
from semkine import eval_track as ET                              # noqa: E402
from semkine.dataset import sequences_for_split, splits_manifest  # noqa: E402

STEP = 50
G1_MAX_DEG = 3.0
G2_MIN_GAIN = 0.5
G2_MAX_ABLATED_SHARE = 0.2
G3_MIN_MM = 1.5


def aa_to_R(aa):
    th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = aa / th
    K = torch.zeros(*aa.shape[:-1], 3, 3, device=aa.device, dtype=aa.dtype)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    s, c = th.sin()[..., None], th.cos()[..., None]
    return eye + s * K + (1 - c) * (K @ K)


def R_to_aa(R):
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    th = ((tr - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7).acos()
    w = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    return w / (2 * th.sin()[..., None]).clamp_min(1e-9) * th[..., None]


def rot_deg(a, b):
    tr = (aa_to_R(a).transpose(-1, -2) @ aa_to_R(b)).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(((tr - 1) / 2).clamp(-1, 1).acos())


class Seq:
    def __init__(self, root, d, seq, device):
        self.name = seq
        self.events, self.offsets, aux, self.pos51 = ET.load_sequence(root, d, seq)
        p = root / d / f"{seq}_tsub.npy"
        self.tsub = np.load(p, mmap_mode="r") if p.exists() else None
        self.betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
        self.K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
        self.rows = []
        for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
            p0 = int(a)
            for e in np.arange(a + STEP - 1, b, STEP, dtype=np.int64):
                self.rows.append((p0, int(e)))
                p0 = int(e)

    def ev(self, end):
        return ET._window_events(self.events, self.offsets, self.tsub, int(end), STEP)


def batch_of(s, rows, prev, device):
    evs, ptr = [], [0]
    for j, (_, e) in enumerate(rows):
        ev = s.ev(e).copy()
        ev[:, 0] = j
        evs.append(ev)
        ptr.append(ptr[-1] + len(ev))
    B = len(rows)
    b = ET.make_eval_packet(np.concatenate(evs), prev, s.betas.expand(B, -1), s.K.expand(B, -1, -1), STEP, device)
    b.ptr = torch.tensor(ptr, dtype=torch.int64, device=device)
    for f in ("sequence_id", "t_start_us", "t_end_us", "delta_t_s", "is_sequence_start", "is_sequence_end"):
        setattr(b, f, getattr(b, f).expand(B).contiguous())
    return b


@torch.no_grad()
def g1_tf_rotation(model, seqs, device, bs=64):
    per, all_err = {}, []
    for s in seqs:
        model.set_hand_context(s.betas, s.K)
        errs = []
        for i0 in range(0, len(s.rows), bs):
            rows = s.rows[i0:i0 + bs]
            prev = torch.from_numpy(np.stack([s.pos51[p] for p, _ in rows]).astype(np.float32)).to(device)
            gt = torch.from_numpy(np.stack([s.pos51[e] for _, e in rows]).astype(np.float32)).to(device)
            out = model.forward_packet(batch_of(s, rows, prev, device)).float()
            errs.append(rot_deg(out[:, 3:6], gt[:, 3:6]))
        e = torch.cat(errs)
        per[s.name] = float(e.mean())
        all_err.append(e)
    return {"per_seq": per, "all": float(torch.cat(all_err).mean())}


@torch.no_grad()
def g2_gain(model, seqs, device, deg=10.0, stride=4):
    def run(ablate):
        model.ablate_innovation = ablate
        acc = {ax: [] for ax in range(3)}
        for s in seqs:
            model.set_hand_context(s.betas, s.K)
            for p, e in s.rows[::stride]:
                ev5 = s.ev(e)
                if len(ev5) == 0:
                    continue
                prev = torch.from_numpy(s.pos51[p].astype(np.float32).copy()).view(1, -1).to(device)
                base = model.forward_packet(ET.make_eval_packet(ev5, prev, s.betas, s.K, STEP, device))
                for ax in range(3):
                    r = torch.zeros(1, 3, device=device)
                    r[0, ax] = math.radians(deg)
                    pert = prev.clone()
                    pert[:, 3:6] = R_to_aa(aa_to_R(r) @ aa_to_R(prev[:, 3:6]))
                    o = model.forward_packet(ET.make_eval_packet(ev5, pert, s.betas, s.K, STEP, device))
                    w = R_to_aa(aa_to_R(o[:, 3:6].float()) @ aa_to_R(base[:, 3:6].float()).transpose(-1, -2))
                    acc[ax].append(1.0 - float(w[0, ax]) / math.radians(deg))
        model.ablate_innovation = False
        per_ax = {f"ax{ax}": float(np.mean(v)) for ax, v in acc.items()}
        return {"mean": float(np.mean(list(per_ax.values()))), **per_ax, "n": len(acc[0])}
    return {"full": run(False), "innovation_zeroed": run(True)}


@torch.no_grad()
def g3_closed(model, cfg, root, seq_dirs, device):
    def run(ablate):
        model.ablate_innovation = ablate
        rng = np.random.default_rng(0)
        res = [ET.track_sequence(model, model.mano, cfg, root, d, s, STEP, device, rng, 1.0, 1.0, None)
               for s, d in seq_dirs]
        model.ablate_innovation = False
        n = sum(r["n_frames"] for r in res)
        return {"ra": sum(r["mpjpe_ra_mm"] * r["n_frames"] for r in res) / n,
                "per_seq": {r["seq"]: r["mpjpe_ra_mm"] for r in res}}
    return {"normal": run(False), "innovation_zeroed": run(True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda")
    out = {"gates": {"G1_max_deg": G1_MAX_DEG, "G2_min_gain": G2_MIN_GAIN,
                     "G2_max_ablated_share": G2_MAX_ABLATED_SHARE, "G3_min_mm": G3_MIN_MM}, "runs": {}}
    for spec in a.runs:
        t0 = time.time()
        model, cfg, sel = load_run(spec, device)
        assert getattr(model, "root_innovation", False), f"{spec} is not a root-innovation run"
        root = Path(cfg["DATA"]["ROOT"])
        seq_dirs = sequences_for_split(root, "val_core", splits_manifest(cfg))
        seqs = [Seq(root, d, s, device) for s, d in seq_dirs]
        r = {"ckpt": sel["ckpt"], "step": sel.get("step")}
        r["G1_tf_rotation_deg"] = g1_tf_rotation(model, seqs, device)
        r["G2_gain_10deg"] = g2_gain(model, seqs, device)
        r["G3_closed_loop"] = g3_closed(model, cfg, root, seq_dirs, device)
        g2 = r["G2_gain_10deg"]
        r["pass"] = {
            "G1": r["G1_tf_rotation_deg"]["all"] <= G1_MAX_DEG,
            "G2": (g2["full"]["mean"] >= G2_MIN_GAIN
                   and g2["innovation_zeroed"]["mean"] <= G2_MAX_ABLATED_SHARE * g2["full"]["mean"]),
        }
        r["seconds"] = round(time.time() - t0, 1)
        out["runs"][Path(spec).name] = r
        print(Path(spec).name, json.dumps(r, indent=1), flush=True)
        del model
        torch.cuda.empty_cache()
    runs = list(out["runs"].values())
    d3 = [x["G3_closed_loop"]["innovation_zeroed"]["ra"] - x["G3_closed_loop"]["normal"]["ra"] for x in runs]
    out["G3_mean_worsening_mm"] = float(np.mean(d3))
    out["pass"] = {"G1_all_seeds": all(x["pass"]["G1"] for x in runs),
                   "G2_all_seeds": all(x["pass"]["G2"] for x in runs),
                   "G3_two_seed_mean": out["G3_mean_worsening_mm"] >= G3_MIN_MM}
    print(json.dumps(out["pass"]), f"G3 mean {out['G3_mean_worsening_mm']:.2f} mm", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
