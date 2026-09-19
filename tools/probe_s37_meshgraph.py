#!/usr/bin/env python3
"""Split an arm's zgz error into global rotation and articulation, teacher forced and closed loop.

RA-MPJPE is root-aligned but not rotation-aligned, so a wrong root rotation moves every joint. Per
step the 21 root-aligned joints are Kabsch-aligned to the ground truth: the residual after the
optimal rotation is the articulation error (`ra_rotaligned`), the angle of that rotation is the
global rotation error (`rot_deg`). Run on the selected checkpoint (teacher forced + closed loop, the
selection protocol's own loop and rng) and, with `--grid`, closed loop on every checkpoint of the
run -- the correlation of grid RA with the two parts says which one a run's checkpoint-to-checkpoint
instability lives in (S37 mesh graph: 0.96 with rotation, -0.01 with articulation).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
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
from semkine.dataset import sequences_for_split, splits_manifest  # noqa: E402


class _Rec:
    def __init__(self):
        self.pred = []

    def __call__(self, tag, prev_t, pred):
        if tag == "step":
            self.pred.append(pred.detach().clone())
        return pred


def _fk_joints_ra(mano, params, betas):
    dec = ET.decode_to_mano_inputs(params, "mano_full_axis_angle", mano.hands_components, mano.hands_mean)
    _, j = mano(betas.expand(len(params), -1), dec["global_orient"], dec["local_full_aa"], dec["transl"])
    return j - j[:, :1]


def _kabsch(P, Q):
    """Per-sample rotation minimising |R P - Q|; P, Q `(B, N, 3)` already root-aligned."""
    Pc, Qc = P - P.mean(1, keepdim=True), Q - Q.mean(1, keepdim=True)
    U, _, Vt = torch.linalg.svd(Pc.transpose(1, 2) @ Qc)
    d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    return Vt.transpose(1, 2) @ D @ U.transpose(1, 2)


def decompose(pj, gj) -> dict:
    R = _kabsch(pj, gj)
    ra = (pj - gj).norm(dim=-1).mean(-1) * 1000
    ra_al = ((R @ pj.transpose(1, 2)).transpose(1, 2) - gj).norm(dim=-1).mean(-1) * 1000
    tr = R.diagonal(dim1=1, dim2=2).sum(-1)
    ang = torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))
    return {"ra": float(ra.mean()), "ra_rotaligned": float(ra_al.mean()),
            "rot_p50_deg": float(ang.median()), "rot_p90_deg": float(ang.quantile(0.9)),
            "frac_rot_gt_20deg": float((ang > 20).float().mean()), "n_steps": int(len(ra))}


def _windows(aux, step_ms):
    ends, prev_ends = [], []
    for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
        e = np.arange(a + step_ms - 1, b, step_ms, dtype=np.int64)
        if len(e) == 0:
            continue                          # a run shorter than one step has no window at all
        ends += e.tolist()
        prev_ends += np.concatenate([[a], e[:-1]]).tolist()
    return ends, prev_ends


@torch.no_grad()
def probe_run(run_dir: Path, seq_name: str, step_ms: int, device, grid: bool) -> dict:
    sel = json.loads((run_dir / f"selection_val_core_step{step_ms}.json").read_text())
    cfg_path = REPO / "configs/semkine" / (re.sub(r"_s\d+$", "_s3407", run_dir.name) + ".yaml")
    cfg = load_config(str(cfg_path))
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    mani = splits_manifest(cfg)
    if mani and not Path(mani).is_absolute() and not Path(mani).exists():
        mani = root / mani
    seq, legacy_dir = next((s, d) for s, d in sequences_for_split(root, "val_core", mani) if s == seq_name)
    events, offsets, aux, pos51 = ET.load_sequence(root, legacy_dir, seq)
    tsub_path = root / legacy_dir / f"{seq}_tsub.npy"
    tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    camera_K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    ends, prev_ends = _windows(aux, step_ms)
    gj = _fk_joints_ra(mano, torch.from_numpy(np.stack([pos51[e] for e in ends])).to(device), betas)

    def closed_loop(model):
        rec = _Rec()
        ET.track_sequence(model, mano, cfg, root, legacy_dir, seq, step_ms, device,
                          np.random.default_rng(0), 1.0, 1.0, None, state_hook=rec)
        return decompose(_fk_joints_ra(mano, torch.cat(rec.pred), betas), gj)

    model = MNISTModel.load_from_checkpoint(str(REPO / sel["selected"]["ckpt"]), cfg=cfg,
                                            map_location=device).to(device).eval()
    model.set_hand_context(betas, camera_K)
    tf = []
    for end, pe in zip(ends, prev_ends):
        prev = torch.from_numpy(pos51[pe].copy()).view(1, -1).to(device)
        ev5 = ET._window_events(events, offsets, tsub, int(end), step_ms)
        tf.append(model.forward_packet(ET.make_eval_packet(ev5, prev, betas, camera_K, step_ms, device)))
    out = {"seq": seq_name, "selected_step": sel["selected"]["step"],
           "teacher_forced": decompose(_fk_joints_ra(mano, torch.cat(tf), betas), gj),
           "closed_loop": closed_loop(model)}
    print(f"{run_dir.name} step {out['selected_step']} [{seq_name}]")
    for k in ("teacher_forced", "closed_loop"):
        v = out[k]
        print(f"  {k:14s} RA {v['ra']:6.2f}  rot-aligned {v['ra_rotaligned']:6.2f}  "
              f"rot p50/p90 {v['rot_p50_deg']:5.1f}/{v['rot_p90_deg']:5.1f} deg  frac>20deg {v['frac_rot_gt_20deg']:.2f}")
    if grid:
        rows = []
        ckpts = sorted(glob.glob(str(run_dir / f"{run_dir.name}-step=*.ckpt")),
                       key=lambda p: int(re.search(r"step=(\d+)", p).group(1)))
        for ck in ckpts:
            m = MNISTModel.load_from_checkpoint(ck, cfg=cfg, map_location=device).to(device).eval()
            r = closed_loop(m)
            r["step"] = int(re.search(r"step=(\d+)", ck).group(1))
            rows.append(r)
            print(f"    step {r['step']:5d}: RA {r['ra']:6.2f}  rot-aligned {r['ra_rotaligned']:6.2f}  "
                  f"rot p50/p90 {r['rot_p50_deg']:5.1f}/{r['rot_p90_deg']:5.1f}")
            del m
        ra = np.array([r["ra"] for r in rows])
        al = np.array([r["ra_rotaligned"] for r in rows])
        rot = np.array([r["rot_p50_deg"] for r in rows])
        out["grid"] = rows
        out["grid_summary"] = {
            "ra_mean": float(ra.mean()), "ra_sd": float(ra.std()),
            "rotaligned_mean": float(al.mean()), "rotaligned_sd": float(al.std()),
            "rot_p50_mean": float(rot.mean()), "rot_p50_sd": float(rot.std()),
            "corr_ra_rot": float(np.corrcoef(ra, rot)[0, 1]),
            "corr_ra_rotaligned": float(np.corrcoef(ra, al)[0, 1]),
        }
        g = out["grid_summary"]
        print(f"  grid: RA {g['ra_mean']:.2f}+-{g['ra_sd']:.2f} | rot-aligned {g['rotaligned_mean']:.2f}+-{g['rotaligned_sd']:.2f} "
              f"| rot p50 {g['rot_p50_mean']:.1f}+-{g['rot_p50_sd']:.1f} deg | corr(RA, rot) {g['corr_ra_rot']:.2f} "
              f"corr(RA, rot-aligned) {g['corr_ra_rotaligned']:.2f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs, e.g. outputs/semkine/s37_meshgraph_s3407")
    ap.add_argument("--seq", default="zgz_local")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--grid", action="store_true", help="also decompose the closed loop of every grid checkpoint")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {Path(r).name: probe_run(Path(r), a.seq, a.step_ms, device, a.grid) for r in a.runs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
