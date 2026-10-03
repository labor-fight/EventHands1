#!/usr/bin/env python3
"""S37 FK graph: the six pre-registered mechanism probes on the selected checkpoints.

Nothing here trains; everything is measured on the held-out subject (val_core = zgz) with the
trained checkpoints, S36 on the same packets as the control. Sections follow
docs/S37_EXPERIMENT_RECORDS.md [FKGRAPH] section 5:

  H1  assignment quality (recall = share of events inside the 16 px band, purity = share of in-band
      events whose node's joint agrees with the assignment under the ground-truth prev) with the
      previous state taken as GT / curriculum small noise / curriculum large noise / the model's own
      closed-loop prev; plus the observation ablation and an event-swap test.
  H2  closed-loop RA with the model's own prev, and with *oracle assignment* (events assigned by the
      GT prev while everything else still sees the model's own prev).
  H3  absolute error decomposed into wrist translation / global rotation / fingers, and the root
      head's own response to a prev perturbation with the events held fixed.
  H4  teacher-forced single-step RA on training sequences against the held-out ones.
  H5  update ratio (applied / needed) and direction cosine of the pose update.
  H6  observation sufficiency: the same checkpoint with only the flow channels, only count+offset,
      or all channels (`obs_feature_mask`), teacher-forced and closed-loop.

    CUDA_VISIBLE_DEVICES=6 python tools/probe_s37_fkgraph.py \\
        --runs outputs/semkine/s36_eventgnn_s3407 outputs/semkine/s36_eventgnn_s3408 \\
               outputs/semkine/s37_fkgraph_s3407 outputs/semkine/s37_fkgraph_s3408 \\
        --out outputs/semkine/probe_s37_fkgraph.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from config import load_config                                      # noqa: E402
from model import MNISTModel                                        # noqa: E402
from semkine import eval_track as ET                                # noqa: E402
from semkine.dataset import sequences_for_split, splits_manifest    # noqa: E402
from semkine.fk_graph import N_JOINTS, OBS_DIM, assign_and_observe  # noqa: E402

SMALL = (0.005, 0.05, 0.05)
LARGE = (0.05, 0.3, 0.3)
TRAIN_SEQS = ("lyq_local", "lr_global", "ch_local", "ylf_global")
#: H6 channel subsets over [share, du, dv, spread, t, polarity, b_u, b_v]
MASKS = {"flow_only": [0, 0, 0, 0, 0, 0, 1, 1],
         "count_offset_only": [1, 1, 1, 0, 0, 0, 0, 0],
         "no_flow": [1, 1, 1, 1, 1, 1, 0, 0]}


def load_run(spec: str, device):
    run_dir, _, ckpt = spec.partition(":")
    run = REPO / run_dir
    if ckpt:
        sel = {"ckpt": ckpt, "step": None}
    else:
        sels = sorted(run.glob("selection_val_core_step50*.json"))
        if not sels:
            raise SystemExit(f"{run_dir}: no selection JSON; run tools/select_checkpoint.py first")
        sel = json.loads(sels[-1].read_text())["selected"]
    cfg = load_config(json.loads((run / "training_metadata.json").read_text())["config_path"])
    model = MNISTModel.load_from_checkpoint(sel["ckpt"], cfg=cfg, map_location=device)
    return model.to(device).eval(), cfg, sel


def arm_of(run_name: str) -> str:
    return re.sub(r"_s\d{4}$", "", run_name)


def noise51(rng, scales):
    n = np.zeros(51, np.float32)
    n[0:3] = rng.standard_normal(3) * scales[0]
    n[3:6] = rng.standard_normal(3) * scales[1]
    n[6:51] = rng.standard_normal(45) * scales[2]
    return n


def rot_angle_deg(aa_a, aa_b):
    def to_mat(aa):
        th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        k = aa / th
        K = torch.zeros(aa.shape[0], 3, 3, device=aa.device)
        K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
        K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
        K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
        th = th.unsqueeze(-1)
        return torch.eye(3, device=aa.device) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)
    R = to_mat(aa_a).transpose(1, 2) @ to_mat(aa_b)
    tr = (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]).clamp(-1.0, 3.0)
    return torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1.0, 1.0)))


def runs_of(aux, pos51, step_ms):
    out = []
    for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
        ends = np.arange(a + step_ms - 1, b, step_ms, dtype=np.int64)
        if len(ends):
            out.append({"a": int(a), "ends": ends, "init": pos51[a], "gts": [pos51[e] for e in ends]})
    return out


class Recorder:
    """State hook for `ET.track_sequence`: records prev/pred per step; in oracle mode, tells the
    model to assign events with the ground-truth prev of that step."""

    def __init__(self, model, gt_by_run, oracle=False):
        self.model, self.gt_by_run, self.oracle = model, gt_by_run, oracle
        self.run, self.step = -1, 0
        self.prevs, self.preds = [], []

    def __call__(self, tag, prev_t, pred):
        if tag == "run_start":
            self.run += 1
            self.step = 0
            if self.oracle:
                self.model.route_prev_override = torch.from_numpy(self.gt_by_run[self.run]["init"].copy()).view(1, -1)
            return pred
        if tag == "step" and pred is not None:
            self.prevs.append(prev_t.detach().cpu().numpy()[0].copy())
            self.preds.append(pred.detach().cpu().numpy()[0].copy())
            if self.oracle:
                gts = self.gt_by_run[self.run]["gts"]
                self.model.route_prev_override = torch.from_numpy(gts[min(self.step, len(gts) - 1)].copy()).view(1, -1)
            self.step += 1
        return pred


def is_fk(model):
    return bool(getattr(model, "fk_graph", False))


@torch.no_grad()
def closed_loop(model, cfg, root, seqs, step_ms, device, oracle=False, ablate=False, mask=None):
    if is_fk(model):
        model.ablate_evidence = bool(ablate)
        model.obs_feature_mask = None if mask is None else torch.tensor(mask, dtype=torch.float32, device=device)
    ra = ab = n = 0.0
    wrist, rot, prevs_all = [], [], []
    for seq, d in seqs:
        events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
        runs = runs_of(aux, pos51, step_ms)
        rec = Recorder(model, runs, oracle=oracle and is_fk(model))
        r = ET.track_sequence(model, model.mano, cfg, root, d, seq, step_ms, device,
                              np.random.default_rng(0), 1.0, 1.0, None, state_hook=rec)
        if is_fk(model):
            model.route_prev_override = None
        if r is None:
            continue
        ra += r["mpjpe_ra_mm"] * r["n_frames"]
        ab += r["mpjpe_abs_mm"] * r["n_frames"]
        n += r["n_frames"]
        gts = np.concatenate([np.stack(x["gts"]) for x in runs])
        preds = np.stack(rec.preds)
        p, g = torch.from_numpy(preds).to(device), torch.from_numpy(gts).to(device)
        wrist.append(((p[:, :3] - g[:, :3]).norm(dim=-1) * 1000).cpu().numpy())
        rot.append(rot_angle_deg(p[:, 3:6], g[:, 3:6]).cpu().numpy())
        prevs_all.append(np.stack(rec.prevs))
    if is_fk(model):
        model.ablate_evidence = False
        model.obs_feature_mask = None
    wrist, rot = np.concatenate(wrist), np.concatenate(rot)
    return {"ra_mm": ra / max(n, 1), "abs_mm": ab / max(n, 1), "n_frames": int(n),
            "wrist_mm_p50": float(np.median(wrist)), "wrist_mm_p90": float(np.percentile(wrist, 90)),
            "rot_deg_p50": float(np.median(rot)), "rot_deg_p90": float(np.percentile(rot, 90))}, \
        np.concatenate(prevs_all)


@torch.no_grad()
def teacher_forced(model, cfg, root, seqs, step_ms, device, max_packets=None, closed_prevs=None,
                   swap_gap_ms=1000, seed=0, masks=None):
    fk = is_fk(model)
    rng = np.random.default_rng(seed)
    gap = max(1, swap_gap_ms // step_ms)
    tf_ra, tf_abs, up_num, up_den, cos, swap_disp, true_step, abl_ra = [], [], [], [], [], [], [], []
    mask_ra = {k: [] for k in (masks or {})}
    assign_stats = {c: {"in_band": 0, "agree": 0, "both": 0, "n": 0} for c in ("gt", "small", "large", "closed")}
    node_joint = None
    if fk:
        spec = model.fk_spec
        node_joint = torch.cat([torch.arange(N_JOINTS), spec.vert_joint, torch.tensor([-1])]).to(device)
    k_closed = n_done = 0

    def joints_of(params, betas):
        return model._fk(params.float(), betas)[1]

    def ra_of(pred, gt, betas):
        jp, jg = joints_of(pred, betas), joints_of(gt, betas)
        return float(((jp - jp[:, :1]) - (jg - jg[:, :1])).norm(dim=-1).mean()) * 1000

    def abs_of(pred, gt, betas):
        return float((joints_of(pred, betas) - joints_of(gt, betas)).norm(dim=-1).mean()) * 1000

    def assignment(prev, pk, betas, K):
        bf, kf = model._resolve_betas_K(prev, betas, K)
        uv = model._fk_node_uv(prev, bf, kf)
        _, assign = assign_and_observe(pk.events, pk.ptr, uv, pk.delta_t_s, model.fk_band_px)
        return node_joint[assign]

    for seq, d in seqs:
        events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
        tsub_p = root / d / f"{seq}_tsub.npy"
        tsub = np.load(tsub_p, mmap_mode="r") if tsub_p.exists() else None
        betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
        K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
        model.set_hand_context(betas, K)
        for run in runs_of(aux, pos51, step_ms):
            ends, prev_idx, cache = run["ends"], run["a"], {}

            def window(end):
                if end not in cache:
                    cache[end] = ET._window_events(events, offsets, tsub, int(end), step_ms)
                return cache[end]

            for i, end in enumerate(ends):
                if max_packets is not None and n_done >= max_packets:
                    break
                prev = torch.from_numpy(pos51[prev_idx].copy()).view(1, -1).to(device)
                gt = torch.from_numpy(pos51[int(end)].copy()).view(1, -1).to(device)
                ev_t = window(int(end))
                pk = ET.make_eval_packet(ev_t, prev, betas, K, step_ms, device)
                f0 = model.forward_packet(pk)
                tf_ra.append(ra_of(f0, gt, betas))
                tf_abs.append(abs_of(f0, gt, betas))
                dp, dn = (f0 - prev)[0, 6:], (gt - prev)[0, 6:]
                up_num.append(float(dp.norm()))
                up_den.append(float(dn.norm()))
                cos.append(float(torch.dot(dp, dn) / (dp.norm() * dn.norm() + 1e-9)))
                true_step.append(float((joints_of(gt, betas) - joints_of(prev, betas)).norm(dim=-1).mean()) * 1000)
                j = i + gap if i + gap < len(ends) else i - gap
                if 0 <= j < len(ends):
                    f2 = model.forward_packet(ET.make_eval_packet(window(int(ends[j])), prev, betas, K, step_ms, device))
                    swap_disp.append(float((joints_of(f2, betas) - joints_of(f0, betas)).norm(dim=-1).mean()) * 1000)
                if fk:
                    model.ablate_evidence = True
                    abl_ra.append(ra_of(model.forward_packet(pk), gt, betas))
                    model.ablate_evidence = False
                    for name, m in (masks or {}).items():
                        model.obs_feature_mask = torch.tensor(m, dtype=torch.float32, device=device)
                        mask_ra[name].append(ra_of(model.forward_packet(pk), gt, betas))
                    model.obs_feature_mask = None
                    if ev_t.shape[0]:
                        j_gt = assignment(prev, pk, betas, K)
                        conds = {"gt": prev,
                                 "small": prev + torch.from_numpy(noise51(rng, SMALL)).to(device).view(1, -1),
                                 "large": prev + torch.from_numpy(noise51(rng, LARGE)).to(device).view(1, -1)}
                        if closed_prevs is not None and k_closed < len(closed_prevs):
                            conds["closed"] = torch.from_numpy(closed_prevs[k_closed].copy()).view(1, -1).to(device)
                        for c, pv in conds.items():
                            j_c = assignment(pv, pk, betas, K)
                            inb = j_c >= 0
                            both = inb & (j_gt >= 0)
                            s = assign_stats[c]
                            s["in_band"] += int(inb.sum()); s["agree"] += int((both & (j_c == j_gt)).sum())
                            s["both"] += int(both.sum()); s["n"] += int(j_c.numel())
                k_closed += 1
                n_done += 1
                prev_idx = int(end)

    out = {"n_packets": n_done, "tf_ra_mm": float(np.mean(tf_ra)), "tf_abs_mm": float(np.mean(tf_abs)),
           "update_ratio_pose": float(np.sum(up_num) / max(np.sum(up_den), 1e-9)),
           "alignment_pose": float(np.mean(cos)), "true_step_mm": float(np.mean(true_step)),
           "event_swap_displacement_mm": float(np.mean(swap_disp)) if swap_disp else None,
           "event_dependence": float(np.mean(swap_disp) / max(np.mean(true_step), 1e-9)) if swap_disp else None}
    if fk:
        out["tf_ra_observations_zeroed_mm"] = float(np.mean(abl_ra))
        out["tf_ra_by_mask_mm"] = {k: float(np.mean(v)) for k, v in mask_ra.items()}
        out["assignment"] = {c: {"recall_in_band": s["in_band"] / max(s["n"], 1),
                                 "purity_vs_gt": s["agree"] / max(s["both"], 1), "n_events": s["n"]}
                             for c, s in assign_stats.items() if s["n"] > 0}
    return out


@torch.no_grad()
def root_gain(model, cfg, root, seqs, step_ms, device, max_packets=400):
    trans = torch.zeros(51, device=device); trans[0] = 0.036
    perts = {"trans_36mm": trans}
    for ax, name in enumerate(("rot_x_10deg", "rot_y_10deg", "rot_z_10deg")):
        r = torch.zeros(51, device=device); r[3 + ax] = math.radians(10.0)
        perts[name] = r
    acc = {k: [] for k in perts}
    n = 0

    def head_delta(pk):
        delta = model.forward_packet(pk) - pk.prev_state
        if model.prevpos_embed:
            delta = delta - model.prev_mlp(pk.prev_state)
        return delta[0]

    for seq, d in seqs:
        events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
        tsub_p = root / d / f"{seq}_tsub.npy"
        tsub = np.load(tsub_p, mmap_mode="r") if tsub_p.exists() else None
        betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
        K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
        model.set_hand_context(betas, K)
        for run in runs_of(aux, pos51, step_ms):
            for end in run["ends"][::4]:
                if n >= max_packets:
                    break
                prev = torch.from_numpy(pos51[max(int(end) - step_ms, run["a"])].copy()).view(1, -1).to(device)
                ev5 = ET._window_events(events, offsets, tsub, int(end), step_ms)
                if ev5.shape[0] == 0:
                    continue
                base = head_delta(ET.make_eval_packet(ev5, prev, betas, K, step_ms, device))
                for k, dvec in perts.items():
                    resp = head_delta(ET.make_eval_packet(ev5, prev + dvec.view(1, -1), betas, K, step_ms, device))
                    diff, dd = (resp - base)[:6], dvec[:6]
                    acc[k].append(float(-(diff * dd).sum() / (dd * dd).sum()))
                n += 1
    return {k: float(np.mean(v)) for k, v in acc.items()} | {"n_packets": n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs, or RUN_DIR:CKPT")
    ap.add_argument("--split", default="val_core")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--train-seqs", nargs="*", default=list(TRAIN_SEQS))
    ap.add_argument("--train-max-packets", type=int, default=1500)
    ap.add_argument("--control", default="s36_eventgnn")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = torch.device("cuda")
    rows = {}
    for spec in a.runs:
        name = Path(spec.partition(":")[0]).name
        t0 = time.time()
        model, cfg, sel = load_run(spec, device)
        root = Path(cfg["DATA"]["ROOT"])
        seqs = sequences_for_split(root, a.split, splits_manifest(cfg))
        if a.smoke:
            seqs = seqs[:1]
        fk = is_fk(model)
        r = {"ckpt": sel["ckpt"], "step": sel.get("step"), "fk_graph": fk}
        r["closed"], prevs = closed_loop(model, cfg, root, seqs, a.step_ms, device)
        if fk:
            r["closed_observations_zeroed"], _ = closed_loop(model, cfg, root, seqs, a.step_ms, device, ablate=True)
            r["closed_oracle_assignment"], _ = closed_loop(model, cfg, root, seqs, a.step_ms, device, oracle=True)
            r["closed_by_mask"] = {k: closed_loop(model, cfg, root, seqs, a.step_ms, device, mask=m)[0]["ra_mm"]
                                   for k, m in MASKS.items()}
        r["tf"] = teacher_forced(model, cfg, root, seqs, a.step_ms, device, closed_prevs=prevs,
                                 max_packets=200 if a.smoke else None, masks=MASKS if fk else None)
        r["root_gain"] = root_gain(model, cfg, root, seqs, a.step_ms, device, max_packets=40 if a.smoke else 400)
        picked = [] if a.smoke else [(s, d) for s, d in sequences_for_split(root, "train", splits_manifest(cfg))
                                      if s in set(a.train_seqs)]
        r["train_tf"] = {}
        for s, d in picked:
            tr = teacher_forced(model, cfg, root, [(s, d)], a.step_ms, device, max_packets=a.train_max_packets)
            r["train_tf"][s] = {"tf_ra_mm": tr["tf_ra_mm"], "n_packets": tr["n_packets"]}
        if picked:
            r["train_tf_mean_mm"] = float(np.mean([v["tf_ra_mm"] for v in r["train_tf"].values()]))
            r["tf_gap_zgz_minus_train_mm"] = r["tf"]["tf_ra_mm"] - r["train_tf_mean_mm"]
        r["seconds"] = round(time.time() - t0, 1)
        rows[name] = r
        c, t = r["closed"], r["tf"]
        print(f"{name}: closed RA {c['ra_mm']:.2f} abs {c['abs_mm']:.1f} | TF RA {t['tf_ra_mm']:.2f} | amp x{c['ra_mm'] / t['tf_ra_mm']:.2f} "
              f"| wrist p50 {c['wrist_mm_p50']:.1f} rot p50 {c['rot_deg_p50']:.1f} | event_dep {t['event_dependence']:.2f} "
              f"| gain t {r['root_gain']['trans_36mm']:.2f} rx {r['root_gain']['rot_x_10deg']:.2f} ry {r['root_gain']['rot_y_10deg']:.2f} "
              f"rz {r['root_gain']['rot_z_10deg']:.2f} | tf gap {r.get('tf_gap_zgz_minus_train_mm', float('nan')):.2f}", flush=True)
        if fk:
            print(f"    obs zeroed: closed {r['closed_observations_zeroed']['ra_mm']:.2f} (TF {t['tf_ra_observations_zeroed_mm']:.2f}); "
                  f"oracle assignment: closed {r['closed_oracle_assignment']['ra_mm']:.2f}; masks closed "
                  + ", ".join(f"{k} {v:.2f}" for k, v in r["closed_by_mask"].items())
                  + " | TF " + ", ".join(f"{k} {v:.2f}" for k, v in t["tf_ra_by_mask_mm"].items()), flush=True)
            for cnd, v in t["assignment"].items():
                print(f"    assignment[{cnd:>6}]: recall {v['recall_in_band']:.3f} purity {v['purity_vs_gt']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    arms = {}
    for name, r in rows.items():
        arms.setdefault(arm_of(name), []).append(r)
    m = lambda vals: float(np.mean(vals))  # noqa: E731
    summary = {}
    for k, v in arms.items():
        s = {"n_seeds": len(v), "closed_ra": m([x["closed"]["ra_mm"] for x in v]),
             "closed_abs": m([x["closed"]["abs_mm"] for x in v]), "tf_ra": m([x["tf"]["tf_ra_mm"] for x in v]),
             "amplification": m([x["closed"]["ra_mm"] / x["tf"]["tf_ra_mm"] for x in v]),
             "wrist_mm_p50": m([x["closed"]["wrist_mm_p50"] for x in v]), "rot_deg_p50": m([x["closed"]["rot_deg_p50"] for x in v]),
             "event_dependence": m([x["tf"]["event_dependence"] for x in v]),
             "update_ratio_pose": m([x["tf"]["update_ratio_pose"] for x in v]), "alignment_pose": m([x["tf"]["alignment_pose"] for x in v]),
             "root_gain": {kk: m([x["root_gain"][kk] for x in v]) for kk in v[0]["root_gain"] if kk != "n_packets"},
             "tf_gap_zgz_minus_train_mm": m([x["tf_gap_zgz_minus_train_mm"] for x in v if "tf_gap_zgz_minus_train_mm" in x] or [float("nan")])}
        if v[0]["fk_graph"]:
            s["closed_ra_observations_zeroed"] = m([x["closed_observations_zeroed"]["ra_mm"] for x in v])
            s["closed_ra_oracle_assignment"] = m([x["closed_oracle_assignment"]["ra_mm"] for x in v])
            s["closed_ra_by_mask"] = {kk: m([x["closed_by_mask"][kk] for x in v]) for kk in MASKS}
            s["tf_ra_by_mask"] = {kk: m([x["tf"]["tf_ra_by_mask_mm"][kk] for x in v]) for kk in MASKS}
            s["assignment"] = {c: {kk: m([x["tf"]["assignment"][c][kk] for x in v if c in x["tf"]["assignment"]])
                                   for kk in ("recall_in_band", "purity_vs_gt")} for c in v[0]["tf"]["assignment"]}
        summary[k] = s
    out = {"protocol": {"split": a.split, "step_ms": a.step_ms, "train_seqs": a.train_seqs, "control": a.control,
                        "small_noise": SMALL, "large_noise": LARGE, "masks": MASKS, "obs_channels":
                        ["share", "du", "dv", "spread", "t", "polarity", "b_u", "b_v"]},
           "runs": rows, "arms": summary, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
